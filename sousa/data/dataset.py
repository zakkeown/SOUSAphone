"""PyTorch Dataset for SOUSA drum rudiment classification."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import pandas as pd
from torch.utils.data import Dataset

from sousa.utils.audio import load_audio
from sousa.utils.rudiments import get_rudiment_mapping


def normalize_rudiment_slug(slug: str) -> str:
    """
    Normalize rudiment slug from dataset to canonical form.

    Handles special cases where dataset naming differs from canonical PAS naming.

    Args:
        slug: Rudiment slug from dataset (with underscores)

    Returns:
        Canonical rudiment slug (with hyphens)
    """
    # Convert underscores to hyphens
    canonical = slug.replace("_", "-")

    # Handle special cases
    if canonical == "paradiddle-diddle":
        canonical = "single-paradiddle-diddle"

    return canonical


def load_split_metadata(dataset_path: Path, split: str) -> pd.DataFrame:
    """Load sample metadata filtered by split.

    Supports both legacy (metadata.csv with split column) and new
    (samples.parquet + splits.json with profile-based splits) formats.

    Returns:
        DataFrame of samples for the requested split.
    """
    legacy_path = dataset_path / "metadata.csv"
    if legacy_path.exists():
        df = pd.read_csv(legacy_path)
        return df[df["split"] == split].reset_index(drop=True)

    # New format: samples.parquet + splits.json
    samples_path = dataset_path / "labels" / "samples.parquet"
    splits_path = dataset_path / "splits.json"

    if not samples_path.exists():
        raise FileNotFoundError(f"Neither metadata.csv nor labels/samples.parquet found in {dataset_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found in {dataset_path}")

    df = pd.read_parquet(samples_path)
    with open(splits_path) as f:
        splits = json.load(f)

    profile_ids = set(splits[f"{split}_profile_ids"])
    return df[df["profile_id"].isin(profile_ids)].reset_index(drop=True)


class SOUSADataset(Dataset):
    """PyTorch Dataset for SOUSA drum rudiment audio classification.

    Loads metadata from CSV, filters by split (train/val/test), and loads
    audio waveforms from FLAC files with resampling and padding/cropping.

    Supports curriculum learning through filtering by soundfont, augmentation,
    and tempo range.

    Supports tempo normalization: time-stretches audio so all samples sound
    like they were performed at a reference tempo (e.g., 120 BPM). This makes
    spectrogram patterns consistent across tempos.

    Args:
        dataset_path: Path to dataset directory containing metadata.csv
        split: Dataset split to load ('train', 'val', or 'test')
        sample_rate: Target sample rate for audio (default: 16000 Hz)
        max_duration: Maximum audio duration in seconds (default: 5.0)
        transform: Optional transform to apply to audio (default: None)
        max_samples: Maximum samples to use (None = all)
        soundfonts: List of soundfont names to include (None = all)
        augmentation_presets: List of augmentation presets to include (None = all)
        tempo_range: (min, max) tempo range to include (None = all)
        reference_tempo: If set, time-stretch all audio to this tempo (BPM).
            Requires tempo_bpm column in metadata. (default: None = no normalization)
    """

    def __init__(
        self,
        dataset_path: str,
        split: str,
        sample_rate: int = 16000,
        max_duration: float = 5.0,
        transform: Optional[Callable] = None,
        max_samples: int = None,
        soundfonts: Optional[list[str]] = None,
        augmentation_presets: Optional[list[str]] = None,
        tempo_range: Optional[tuple[int, int]] = None,
        reference_tempo: Optional[float] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_audio_samples = int(max_duration * sample_rate)
        self.transform = transform
        self.reference_tempo = reference_tempo

        # Load rudiment mapping
        self.rudiment_to_id = get_rudiment_mapping()

        # Load metadata filtered by split
        self.metadata = load_split_metadata(self.dataset_path, split)

        if len(self.metadata) == 0:
            raise ValueError(f"No samples found for split '{split}'")

        # Apply curriculum learning filters
        if soundfonts is not None:
            self.metadata = self.metadata[
                self.metadata["soundfont"].isin(soundfonts)
            ].reset_index(drop=True)

        if augmentation_presets is not None:
            self.metadata = self.metadata[
                self.metadata["augmentation_preset"].isin(augmentation_presets)
            ].reset_index(drop=True)

        if tempo_range is not None:
            min_tempo, max_tempo = tempo_range
            self.metadata = self.metadata[
                (self.metadata["tempo_bpm"] >= min_tempo) &
                (self.metadata["tempo_bpm"] <= max_tempo)
            ].reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(
                f"No samples found after filtering (split={split}, "
                f"soundfonts={soundfonts}, augmentation_presets={augmentation_presets}, "
                f"tempo_range={tempo_range})"
            )

        # Limit samples if max_samples specified
        # Use 80/10/10 split ratio to determine per-split limits
        if max_samples is not None:
            split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
            max_for_split = int(max_samples * split_ratios.get(split, 0.1))
            if len(self.metadata) > max_for_split:
                # Randomly sample to get diverse subset (with fixed seed for reproducibility)
                self.metadata = self.metadata.sample(n=max_for_split, random_state=42).reset_index(drop=True)

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
                - sample_id: Unique identifier for the sample
                - rudiment_slug: Rudiment slug (e.g., 'flam')
                - label: Integer class label
                - audio: Audio waveform tensor (1D, shape: [num_samples])
        """
        row = self.metadata.iloc[idx]

        sample_id = row["sample_id"]
        rudiment_slug = row["rudiment_slug"]

        # Normalize rudiment slug to canonical form
        rudiment_slug_canonical = normalize_rudiment_slug(rudiment_slug)

        # Get label from mapping
        if rudiment_slug_canonical not in self.rudiment_to_id:
            raise KeyError(
                f"Rudiment '{rudiment_slug}' (canonical: '{rudiment_slug_canonical}') "
                f"not found in rudiment mapping. Available rudiments: {sorted(self.rudiment_to_id.keys())}"
            )

        label = self.rudiment_to_id[rudiment_slug_canonical]

        # Load audio
        audio_path = self.dataset_path / row["audio_path"]
        audio = load_audio(
            audio_path=audio_path,
            sample_rate=self.sample_rate,
            max_samples=self.max_audio_samples,
        )

        # Tempo normalization: time-stretch audio to reference tempo
        # This makes spectrogram patterns consistent across different tempos.
        # A 60 BPM sample has wide-spaced hits -> compress to 120 BPM (fewer samples)
        # A 180 BPM sample has tight-spaced hits -> stretch to 120 BPM (more samples)
        if self.reference_tempo is not None and "tempo_bpm" in row.index:
            tempo_bpm = float(row["tempo_bpm"])
            if tempo_bpm > 0 and abs(tempo_bpm - self.reference_tempo) > 1.0:
                stretch_factor = tempo_bpm / self.reference_tempo
                new_length = max(1, int(audio.shape[0] * stretch_factor))
                # Resample waveform in time (for percussion, pitch shift is acceptable)
                audio = torch.nn.functional.interpolate(
                    audio.unsqueeze(0).unsqueeze(0),
                    size=new_length,
                    mode="linear",
                    align_corners=False,
                ).squeeze()
                # Re-pad/crop to target length after stretching
                if audio.shape[0] < self.max_audio_samples:
                    padding = self.max_audio_samples - audio.shape[0]
                    audio = torch.nn.functional.pad(audio, (0, padding))
                elif audio.shape[0] > self.max_audio_samples:
                    audio = audio[: self.max_audio_samples]

        # Apply optional transform
        if self.transform is not None:
            audio = self.transform(audio)

        return {
            "sample_id": sample_id,
            "rudiment_slug": rudiment_slug,
            "label": label,
            "audio": audio,
        }
