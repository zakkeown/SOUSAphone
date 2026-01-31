"""PyTorch Dataset for SOUSA drum rudiment classification."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
from torch.utils.data import Dataset

from sousa.utils.audio import load_audio
from sousa.utils.rudiments import get_rudiment_mapping


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
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.transform = transform

        # Load rudiment mapping
        self.rudiment_to_id = get_rudiment_mapping()

        # Load and filter metadata
        metadata_path = self.dataset_path / "metadata.csv"
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
        label = self.rudiment_to_id[rudiment_slug]

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
