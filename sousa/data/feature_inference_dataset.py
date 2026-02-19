"""PyTorch Dataset for feature inference model training.

Provides (raw_onsets, target_features) pairs where raw_onsets simulate
what librosa onset detection would produce and target_features are the
full 12-dim vectors from OnsetDataset._encode_strokes().
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sousa.data.dataset import load_split_metadata
from sousa.data.onset_dataset import OnsetDataset


class FeatureInferenceDataset(Dataset):
    """Dataset that provides (raw_onsets, target_features) pairs.

    Raw onsets simulate librosa onset detection output: (ioi_ms, onset_strength,
    tempo_bpm) per stroke. Target features are the full 12-dim vectors computed
    by OnsetDataset._encode_strokes().

    Args:
        dataset_path: Path to SOUSA dataset directory.
        split: 'train', 'val', or 'test'.
        max_seq_len: Pad/truncate sequences to this length.
        augment: Whether to apply noise augmentation to raw_onsets.
        timing_jitter_std_ms: Std dev of Gaussian timing jitter in ms.
        strength_noise_std: Std dev of Gaussian multiplicative noise on velocity.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str,
        max_seq_len: int = 256,
        augment: bool = False,
        timing_jitter_std_ms: float = 10.0,
        strength_noise_std: float = 0.15,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.timing_jitter_std_ms = timing_jitter_std_ms
        self.strength_noise_std = strength_noise_std

        # Load metadata filtered by split
        self.metadata = load_split_metadata(self.dataset_path, split)

        if len(self.metadata) == 0:
            raise ValueError(f"No samples found for split '{split}'")

        # Load strokes and group by sample_id
        strokes_path = self.dataset_path / "labels" / "strokes.parquet"
        if not strokes_path.exists():
            raise FileNotFoundError(f"Strokes file not found: {strokes_path}")

        valid_ids = set(self.metadata["sample_id"].values)
        strokes = pd.read_parquet(strokes_path)
        strokes = strokes[strokes["sample_id"].isin(valid_ids)]

        # Build lookup: sample_id -> DataFrame of strokes (sorted by time)
        self.strokes_by_sample: Dict[str, pd.DataFrame] = {
            sid: group.sort_values("actual_time_ms").reset_index(drop=True)
            for sid, group in strokes.groupby("sample_id")
        }

        # Build tempo lookup from metadata
        self.tempo_by_sample: Dict[str, float] = dict(
            zip(self.metadata["sample_id"], self.metadata["tempo_bpm"])
        )

        # Reuse OnsetDataset's _encode_strokes() method for target features.
        # Create a lightweight instance without running __init__.
        self._onset_dataset = OnsetDataset.__new__(OnsetDataset)
        self._onset_dataset.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]
        tempo_bpm = float(self.tempo_by_sample[sample_id])

        strokes_df = self.strokes_by_sample.get(sample_id)
        if strokes_df is None or len(strokes_df) == 0:
            return {
                "raw_onsets": torch.zeros(self.max_seq_len, 3, dtype=torch.float32),
                "target_features": torch.zeros(self.max_seq_len, 12, dtype=torch.float32),
                "attention_mask": torch.zeros(self.max_seq_len, dtype=torch.float32),
            }

        # Compute target features from clean data (always deterministic)
        target_features = self._onset_dataset._encode_strokes(strokes_df, tempo_bpm)

        # Build raw onsets
        times = strokes_df["actual_time_ms"].values.copy().astype(np.float64)
        velocities = strokes_df["actual_velocity"].values.copy().astype(np.float64)

        if self.augment:
            # Timing jitter: Gaussian noise on onset times.
            # Don't re-sort — preserves stroke alignment with target_features
            # so raw_onsets[i] always corresponds to target_features[i].
            times = times + np.random.normal(0, self.timing_jitter_std_ms, size=len(times))
            times = np.clip(times, 0, None)

            # Strength noise: Gaussian multiplicative noise on velocity
            noise_factor = 1.0 + np.random.normal(0, self.strength_noise_std, size=len(velocities))
            velocities = np.clip(velocities * noise_factor, 0, 127)

        # Compute raw onset features
        n = len(times)
        ioi_ms = np.zeros(n, dtype=np.float32)
        if n > 1:
            ioi_ms[1:] = np.diff(times).astype(np.float32)

        onset_strength = (velocities / 127.0).astype(np.float32)
        tempo_col = np.full(n, tempo_bpm, dtype=np.float32)

        raw_onsets = torch.from_numpy(
            np.stack([ioi_ms, onset_strength, tempo_col], axis=1)
        )  # (n, 3)

        seq_len = len(raw_onsets)

        # Pad or truncate
        if seq_len >= self.max_seq_len:
            raw_onsets = raw_onsets[: self.max_seq_len]
            target_features = target_features[: self.max_seq_len]
            mask = torch.ones(self.max_seq_len, dtype=torch.float32)
        else:
            pad_len = self.max_seq_len - seq_len
            raw_onsets = torch.cat([
                raw_onsets,
                torch.zeros(pad_len, 3, dtype=torch.float32),
            ])
            target_features = torch.cat([
                target_features,
                torch.zeros(pad_len, 12, dtype=torch.float32),
            ])
            mask = torch.cat([
                torch.ones(seq_len, dtype=torch.float32),
                torch.zeros(pad_len, dtype=torch.float32),
            ])

        return {
            "raw_onsets": raw_onsets,
            "target_features": target_features,
            "attention_mask": mask,
        }
