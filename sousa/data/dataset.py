"""PyTorch Dataset for SOUSA drum rudiment classification."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

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


class SOUSADataset(Dataset):
    """PyTorch Dataset for SOUSA drum rudiment audio classification.

    Loads metadata from CSV, filters by split (train/val/test), and loads
    audio waveforms from FLAC files with resampling and padding/cropping.

    Args:
        dataset_path: Path to dataset directory containing metadata.csv
        split: Dataset split to load ('train', 'val', or 'test')
        sample_rate: Target sample rate for audio (default: 16000 Hz)
        max_duration: Maximum audio duration in seconds (default: 5.0)
        transform: Optional transform to apply to audio (default: None)
    """

    def __init__(
        self,
        dataset_path: str,
        split: str,
        sample_rate: int = 16000,
        max_duration: float = 5.0,
        transform: Optional[Callable] = None,
        use_tiny: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.transform = transform

        # Load rudiment mapping
        self.rudiment_to_id = get_rudiment_mapping()

        # Load metadata based on use_tiny parameter
        if use_tiny:
            metadata_file = "metadata_tiny.csv"
        else:
            metadata_file = "metadata.csv"

        metadata_path = self.dataset_path / metadata_file

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Filter by split
        self.metadata = df[df["split"] == split].reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(f"No samples found for split '{split}'")

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
            max_samples=self.max_samples,
        )

        # Apply optional transform
        if self.transform is not None:
            audio = self.transform(audio)

        return {
            "sample_id": sample_id,
            "rudiment_slug": rudiment_slug,
            "label": label,
            "audio": audio,
        }
