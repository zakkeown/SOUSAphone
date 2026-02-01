"""Lightning DataModule for SOUSA dataset."""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sousa.data.dataset import SOUSADataset
from sousa.data.transforms import MelSpectrogramTransform, ComposeTransforms
from sousa.data.augmentations import SpecAugment


class SOUSADataModule(pl.LightningDataModule):
    """
    Lightning DataModule for SOUSA rudiment classification.

    Handles train/val/test split creation and dataloader management.
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
        specaugment_params: dict = None,
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
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.use_spectrogram = use_spectrogram
        self.use_specaugment = use_specaugment
        self.pin_memory = torch.cuda.is_available()

        # Create base transform if needed
        self.base_transform = None
        if self.use_spectrogram:
            self.base_transform = MelSpectrogramTransform(
                sample_rate=sample_rate,
                n_mels=128,
                n_fft=400,
                hop_length=160,
                target_length=1024,  # AST expects 1024 time frames
            )

        # Create SpecAugment transform if requested
        if use_specaugment and specaugment_params:
            self.specaugment = SpecAugment(**specaugment_params)
        else:
            self.specaugment = None

        # Create train transform (base + augmentation)
        if self.base_transform and self.specaugment:
            self.train_transform = ComposeTransforms([self.base_transform, self.specaugment])
        else:
            self.train_transform = self.base_transform

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
            )
            self.val_dataset = SOUSADataset(
                dataset_path=self.dataset_path,
                split="val",
                sample_rate=self.sample_rate,
                max_duration=self.max_duration,
                transform=self.val_transform,
            )

        if stage == "test":
            self.test_dataset = SOUSADataset(
                dataset_path=self.dataset_path,
                split="test",
                sample_rate=self.sample_rate,
                max_duration=self.max_duration,
                transform=self.val_transform,
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
