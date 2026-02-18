"""PyTorch Dataset for onset-based rudiment classification."""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sousa.data.dataset import normalize_rudiment_slug, load_split_metadata
from sousa.utils.rudiments import get_rudiment_mapping


class OnsetDataset(Dataset):
    """Dataset that builds per-stroke feature vectors from strokes.parquet.

    Each sample is a sequence of 12-dimensional stroke feature vectors,
    padded to max_seq_len. Features encode tempo-invariant rhythm, velocity,
    stroke type, sticking, buzz info, and metric position.

    Args:
        dataset_path: Path to SOUSA dataset (contains metadata.csv, labels/strokes.parquet)
        split: 'train', 'val', or 'test'
        max_seq_len: Pad/truncate sequences to this length
        max_samples: Maximum total samples (None = all). Uses 80/10/10 ratio.
        soundfonts: Filter by soundfont names
        augmentation_presets: Filter by augmentation presets
        tempo_range: (min_bpm, max_bpm) filter
    """

    def __init__(
        self,
        dataset_path: str,
        split: str,
        max_seq_len: int = 128,
        max_samples: Optional[int] = None,
        soundfonts: Optional[list[str]] = None,
        augmentation_presets: Optional[list[str]] = None,
        tempo_range: Optional[tuple[int, int]] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.max_seq_len = max_seq_len
        self.rudiment_to_id = get_rudiment_mapping()

        # Load metadata filtered by split (supports both legacy CSV and new parquet format)
        meta = load_split_metadata(self.dataset_path, split)

        if soundfonts is not None:
            meta = meta[meta["soundfont"].isin(soundfonts)].reset_index(drop=True)
        if augmentation_presets is not None:
            meta = meta[meta["augmentation_preset"].isin(augmentation_presets)].reset_index(drop=True)
        if tempo_range is not None:
            min_t, max_t = tempo_range
            meta = meta[(meta["tempo_bpm"] >= min_t) & (meta["tempo_bpm"] <= max_t)].reset_index(drop=True)

        if len(meta) == 0:
            raise ValueError(f"No samples after filtering (split={split})")

        # Limit samples if requested (same 80/10/10 logic as SOUSADataset)
        if max_samples is not None:
            split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
            max_for_split = int(max_samples * split_ratios.get(split, 0.1))
            if len(meta) > max_for_split:
                meta = meta.sample(n=max_for_split, random_state=42).reset_index(drop=True)

        self.metadata = meta

        # Load strokes and group by sample_id
        strokes_path = self.dataset_path / "labels" / "strokes.parquet"
        if not strokes_path.exists():
            raise FileNotFoundError(f"Strokes file not found: {strokes_path}")

        # Only load columns we need, and only rows for our split's sample_ids
        valid_ids = set(self.metadata["sample_id"].values)
        strokes = pd.read_parquet(strokes_path)
        strokes = strokes[strokes["sample_id"].isin(valid_ids)]

        # Build lookup: sample_id → DataFrame of strokes (sorted by time)
        self.strokes_by_sample = {
            sid: group.sort_values("actual_time_ms").reset_index(drop=True)
            for sid, group in strokes.groupby("sample_id")
        }

        # Build tempo lookup from metadata
        self.tempo_by_sample = dict(zip(meta["sample_id"], meta["tempo_bpm"]))

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]
        rudiment_slug = normalize_rudiment_slug(row["rudiment_slug"])
        label = self.rudiment_to_id[rudiment_slug]
        tempo_bpm = self.tempo_by_sample[sample_id]

        # Get strokes for this sample
        strokes_df = self.strokes_by_sample.get(sample_id)
        if strokes_df is None or len(strokes_df) == 0:
            # Return zeros if no strokes found (shouldn't happen)
            return {
                "onset_features": torch.zeros(self.max_seq_len, 12, dtype=torch.float32),
                "attention_mask": torch.zeros(self.max_seq_len, dtype=torch.float32),
                "label": label,
                "sample_id": sample_id,
            }

        features = self._encode_strokes(strokes_df, tempo_bpm)
        seq_len = len(features)

        # Pad or truncate to max_seq_len
        if seq_len >= self.max_seq_len:
            features = features[:self.max_seq_len]
            mask = torch.ones(self.max_seq_len, dtype=torch.float32)
        else:
            pad_len = self.max_seq_len - seq_len
            features = torch.cat([
                features,
                torch.zeros(pad_len, 12, dtype=torch.float32),
            ])
            mask = torch.cat([
                torch.ones(seq_len, dtype=torch.float32),
                torch.zeros(pad_len, dtype=torch.float32),
            ])

        return {
            "onset_features": features,
            "attention_mask": mask,
            "label": label,
            "sample_id": sample_id,
        }

    def _encode_strokes(self, df: pd.DataFrame, tempo_bpm: float) -> torch.Tensor:
        """Encode a DataFrame of strokes into a (num_strokes, 12) feature tensor."""
        beat_ms = 60_000.0 / tempo_bpm  # ms per beat

        times = df["actual_time_ms"].values
        n = len(df)

        # IOI: inter-onset interval normalized by beat duration
        ioi = np.zeros(n, dtype=np.float32)
        if n > 1:
            ioi[1:] = np.diff(times) / beat_ms

        # Velocity normalized to [0, 1]
        norm_velocity = df["actual_velocity"].values.astype(np.float32) / 127.0

        # Binary features
        is_grace = df["is_grace_note"].values.astype(np.float32)
        is_accent = df["is_accent"].values.astype(np.float32)

        # Stroke type features
        stroke_types = df["stroke_type"].values
        is_tap = (stroke_types == "tap").astype(np.float32)
        is_diddle = (stroke_types == "diddle").astype(np.float32)
        is_buzz = (stroke_types == "buzz").astype(np.float32)

        # Hand: R=1, L=0
        hand_r = (df["hand"].values == "R").astype(np.float32)

        # Diddle position normalized (0, 0.5, or 1 for position 0, 1, 2)
        diddle_pos_raw = df["diddle_position"].values
        diddle_pos = np.where(np.isnan(diddle_pos_raw), 0.0, diddle_pos_raw / 2.0).astype(np.float32)

        # Flam spacing normalized by beat duration
        flam_raw = df["flam_spacing_ms"].values
        norm_flam = np.where(np.isnan(flam_raw), 0.0, flam_raw / beat_ms).astype(np.float32)

        # Position in beat: cumulative IOI mod beat, normalized
        cum_ioi_ms = np.cumsum(np.concatenate([[0], np.diff(times)]))
        position_in_beat = ((cum_ioi_ms % beat_ms) / beat_ms).astype(np.float32)

        # Buzz count normalized (max is 8 sub-strokes)
        buzz_count_raw = df["buzz_count"].values if "buzz_count" in df.columns else np.zeros(n)
        norm_buzz_count = np.where(
            np.isnan(buzz_count_raw), 0.0, buzz_count_raw / 8.0
        ).astype(np.float32)

        features = np.stack([
            ioi,               # 0: norm_ioi
            norm_velocity,     # 1: norm_velocity
            is_grace,          # 2: is_grace
            is_accent,         # 3: is_accent
            is_tap,            # 4: is_tap
            is_diddle,         # 5: is_diddle
            hand_r,            # 6: hand_R
            diddle_pos,        # 7: diddle_pos
            norm_flam,         # 8: norm_flam_spacing
            position_in_beat,  # 9: position_in_beat
            is_buzz,           # 10: is_buzz
            norm_buzz_count,   # 11: norm_buzz_count
        ], axis=1)  # (n, 12)

        return torch.from_numpy(features)
