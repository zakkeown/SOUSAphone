"""End-to-end audio → rudiment prediction pipeline."""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from sousa.models.feature_inference import FeatureInferenceModel
from sousa.models.onset_transformer import OnsetTransformerModel
from sousa.utils.rudiments import get_inverse_mapping


class OnsetDetector:
    """Detect onsets and estimate tempo from audio using librosa."""

    def detect_all(
        self, audio: np.ndarray, sr: int = 22050
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Detect onsets and estimate tempo in a single pass.

        Computes the onset envelope once and reuses it for both onset
        detection and beat tracking, avoiding a redundant STFT.

        Returns:
            times: onset times in seconds
            strengths: onset strengths (0-1 normalized)
            tempo_bpm: estimated tempo in BPM
        """
        import librosa

        # Compute onset envelope once (the expensive mel-spectrogram step)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

        # Onset detection — peak picking on precomputed envelope
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Strengths at onset frames
        strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])
        if len(strengths) > 0 and strengths.max() > 0:
            strengths = strengths / strengths.max()

        # Beat tracking — reuse onset envelope (skips second STFT)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])

        return onset_times, strengths, float(tempo)

    def detect(
        self, audio: np.ndarray, sr: int = 22050
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect onset times and strengths."""
        times, strengths, _ = self.detect_all(audio, sr=sr)
        return times, strengths

    def estimate_tempo(self, audio: np.ndarray, sr: int = 22050) -> float:
        """Estimate tempo in BPM."""
        _, _, tempo = self.detect_all(audio, sr=sr)
        return tempo


class RudimentPipeline:
    """Full audio -> rudiment prediction pipeline.

    Chains: onset detection -> feature inference -> OnsetTransformer classification.
    """

    def __init__(
        self,
        feature_model_path: Optional[str] = None,
        classifier_model_path: Optional[str] = None,
        max_seq_len: int = 256,
    ):
        self.max_seq_len = max_seq_len
        self.id2label = get_inverse_mapping()
        self.detector = OnsetDetector()

        # Load models if paths provided
        self.feature_model: Optional[FeatureInferenceModel] = None
        self.classifier: Optional[OnsetTransformerModel] = None

        if feature_model_path is not None:
            config = self._load_config(
                feature_model_path, "feature_inference_config.json"
            )
            if config is not None:
                self.feature_model = FeatureInferenceModel.from_config(config)
            else:
                self.feature_model = FeatureInferenceModel()
            state = torch.load(feature_model_path, map_location="cpu", weights_only=True)
            self.feature_model.load_state_dict(state)
            self.feature_model.eval()

        if classifier_model_path is not None:
            config = self._load_config(
                classifier_model_path, "onset_transformer_config.json"
            )
            if config is not None:
                self.classifier = OnsetTransformerModel.from_config(config)
            else:
                self.classifier = OnsetTransformerModel()
            state = torch.load(classifier_model_path, map_location="cpu", weights_only=True)
            self.classifier.load_state_dict(state)
            self.classifier.eval()

    @staticmethod
    def _load_config(
        model_path: str, config_filename: str
    ) -> Optional[dict]:
        """Load a JSON config from the same directory as the weights file.

        Returns None if the config file doesn't exist.
        """
        config_path = Path(model_path).parent / config_filename
        if config_path.is_file():
            with open(config_path) as f:
                return json.load(f)
        return None

    def prepare_raw_onsets(
        self,
        onset_times: np.ndarray,
        onset_strengths: np.ndarray,
        tempo_bpm: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert onset detection output to model input tensors.

        Returns:
            raw_onsets: (1, max_seq_len, 3)
            attention_mask: (1, max_seq_len)
        """
        n = min(len(onset_times), self.max_seq_len)

        ioi_ms = np.zeros(n, dtype=np.float32)
        if n > 1:
            ioi_ms[1:] = np.diff(onset_times[:n]) * 1000.0  # sec -> ms

        strengths = onset_strengths[:n].astype(np.float32)
        tempo = np.full(n, tempo_bpm, dtype=np.float32)

        features = np.stack([ioi_ms, strengths, tempo], axis=1)  # (n, 3)

        # Pad to max_seq_len
        padded = np.zeros((self.max_seq_len, 3), dtype=np.float32)
        padded[:n] = features

        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:n] = 1.0

        return (
            torch.from_numpy(padded).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )

    @torch.no_grad()
    def predict(
        self, audio: np.ndarray, sr: int = 22050
    ) -> dict:
        """Run full pipeline: audio -> rudiment prediction.

        Returns dict with: predicted_rudiment, confidence, top5, onset_times,
        onset_strengths, tempo_bpm, predicted_features.
        """
        assert self.feature_model is not None and self.classifier is not None

        # Determine device from model parameters
        device = next(self.feature_model.parameters()).device

        # Stage 1: Onset detection + tempo (single STFT pass)
        onset_times, onset_strengths, tempo_bpm = self.detector.detect_all(
            audio, sr=sr
        )

        if len(onset_times) == 0:
            return {"error": "No onsets detected in audio"}

        # Stage 2: Prepare input
        raw_onsets, mask = self.prepare_raw_onsets(
            onset_times, onset_strengths, tempo_bpm
        )
        raw_onsets = raw_onsets.to(device)
        mask = mask.to(device)

        # Stage 3: Feature inference
        features = self.feature_model(raw_onsets, attention_mask=mask)
        # Apply sigmoid to binary features for downstream use
        binary_idx = FeatureInferenceModel.BINARY_INDICES
        features_processed = features.clone()
        features_processed[:, :, binary_idx] = torch.sigmoid(
            features[:, :, binary_idx]
        )

        # Stage 4: Classification
        logits = self.classifier(features_processed, attention_mask=mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # Top-5 predictions
        top5_probs, top5_indices = probs.topk(5)
        top5 = [
            {"rudiment": self.id2label[idx.item()], "confidence": prob.item()}
            for prob, idx in zip(top5_probs, top5_indices)
        ]

        return {
            "predicted_rudiment": top5[0]["rudiment"],
            "confidence": top5[0]["confidence"],
            "top5": top5,
            "onset_times": onset_times,
            "onset_strengths": onset_strengths,
            "tempo_bpm": tempo_bpm,
            "predicted_features": features_processed.squeeze(0).cpu().numpy(),
            "attention_mask": mask.squeeze(0).cpu().numpy(),
        }
