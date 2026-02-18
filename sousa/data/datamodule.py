"""Lightning DataModule for SOUSA dataset."""

from typing import Any, Callable, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sousa.data.dataset import SOUSADataset
from sousa.data.transforms import MelSpectrogramTransform, ComposeTransforms
from sousa.data.augmentations import SpecAugment, TimeStretch


class SOUSADataModule(pl.LightningDataModule):
    """
    Lightning DataModule for SOUSA rudiment classification.

    Handles train/val/test split creation and dataloader management.
    Supports curriculum learning through data filtering.
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 16,
        num_workers: int = 4,
        sample_rate: int = 16000,
        max_duration: float = 5.0,
        use_spectrogram: bool = True,
        use_specaugment: bool = False,
        specaugment_params: Optional[dict[str, Any]] = None,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 160,
        max_length: int = 1024,
        max_samples: Optional[int] = None,
        normalize_spec: bool = True,
        norm_mean: float = -4.2677393,
        norm_std: float = 4.5689974,
        # Curriculum learning filters
        soundfonts: Optional[list[str]] = None,
        augmentation_presets: Optional[list[str]] = None,
        tempo_range: Optional[tuple[int, int]] = None,
        # Tempo normalization
        reference_tempo: Optional[float] = None,
        # Waveform augmentation
        use_time_stretch: bool = False,
        time_stretch_min: float = 0.8,
        time_stretch_max: float = 1.2,
    ):
        """
        Initialize DataModule.

        Args:
            dataset_path: Path to SOUSA dataset
            batch_size: Batch size for training
            num_workers: Number of dataloader workers
            sample_rate: Audio sample rate
            max_duration: Max audio duration (seconds)
            use_spectrogram: Whether to convert audio to mel-spectrogram
            use_specaugment: Whether to use SpecAugment on training data
            specaugment_params: Parameters for SpecAugment
            n_mels: Number of mel filterbanks (model-specific)
            n_fft: FFT window size (1024 avoids zero filterbanks)
            hop_length: Hop length for STFT
            max_length: Target time frames for spectrograms
            max_samples: Maximum total samples to use (None = use all)
            normalize_spec: Whether to normalize spectrograms
            norm_mean: Mean for normalization (AST default: -4.2677)
            norm_std: Std for normalization (AST default: 4.5690)
            soundfonts: List of soundfonts to include (curriculum learning)
            augmentation_presets: List of augmentation presets to include
            tempo_range: (min, max) tempo range to include
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.use_spectrogram = use_spectrogram
        self.use_specaugment = use_specaugment
        self.max_samples = max_samples
        self.pin_memory = torch.cuda.is_available()

        # Curriculum learning filters
        self.soundfonts = soundfonts
        self.augmentation_presets = augmentation_presets
        self.tempo_range = tempo_range
        self.reference_tempo = reference_tempo

        # Create base transform if needed
        self.base_transform = None
        if self.use_spectrogram:
            self.base_transform = MelSpectrogramTransform(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                target_length=max_length,
                normalize=normalize_spec,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )

        # Create waveform-level augmentations (applied before spectrogram)
        self.time_stretch = None
        if use_time_stretch:
            self.time_stretch = TimeStretch(
                min_rate=time_stretch_min,
                max_rate=time_stretch_max,
            )

        # Create SpecAugment transform if requested
        self.specaugment: Optional[SpecAugment] = None
        if use_specaugment and specaugment_params:
            self.specaugment = SpecAugment(**specaugment_params)

        # Build train transform chain: [time_stretch] -> [spectrogram] -> [specaugment]
        train_transforms: list[Callable[..., Any]] = []
        if self.time_stretch:
            train_transforms.append(self.time_stretch)
        if self.base_transform:
            train_transforms.append(self.base_transform)
        if self.specaugment:
            train_transforms.append(self.specaugment)

        self.train_transform: Optional[ComposeTransforms] = None
        if train_transforms:
            self.train_transform = ComposeTransforms(train_transforms)

        # Validation/test uses only base transform (no augmentation)
        self.val_transform = self.base_transform

    def setup(self, stage: str) -> None:
        """Create datasets for each split."""
        if stage == "fit":
            self.train_dataset = SOUSADataset(
                dataset_path=self.dataset_path,
                split="train",
                sample_rate=self.sample_rate,
                max_duration=self.max_duration,
                transform=self.train_transform,
                max_samples=self.max_samples,
                soundfonts=self.soundfonts,
                augmentation_presets=self.augmentation_presets,
                tempo_range=self.tempo_range,
                reference_tempo=self.reference_tempo,
            )
            self.val_dataset = SOUSADataset(
                dataset_path=self.dataset_path,
                split="val",
                sample_rate=self.sample_rate,
                max_duration=self.max_duration,
                transform=self.val_transform,
                max_samples=self.max_samples,
                soundfonts=self.soundfonts,
                augmentation_presets=self.augmentation_presets,
                tempo_range=self.tempo_range,
                reference_tempo=self.reference_tempo,
            )

        if stage == "test":
            self.test_dataset = SOUSADataset(
                dataset_path=self.dataset_path,
                split="test",
                sample_rate=self.sample_rate,
                max_duration=self.max_duration,
                transform=self.val_transform,
                max_samples=self.max_samples,
                soundfonts=self.soundfonts,
                augmentation_presets=self.augmentation_presets,
                tempo_range=self.tempo_range,
                reference_tempo=self.reference_tempo,
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
