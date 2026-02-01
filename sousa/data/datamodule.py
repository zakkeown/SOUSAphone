"""Lightning DataModule for SOUSA dataset."""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sousa.data.dataset import SOUSADataset


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
    ):
        """
        Initialize DataModule.

        Args:
            dataset_path: Path to SOUSA dataset
            batch_size: Batch size for training
            num_workers: Number of dataloader workers
            sample_rate: Audio sample rate
            max_duration: Max audio duration (seconds)
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage: str) -> None:
        """Create datasets for each split."""
        if stage == "fit":
            self.train_dataset = SOUSADataset(
                dataset_path=self.dataset_path,
                split="train",
                sample_rate=self.sample_rate,
                max_duration=self.max_duration,
            )
            self.val_dataset = SOUSADataset(
                dataset_path=self.dataset_path,
                split="val",
                sample_rate=self.sample_rate,
                max_duration=self.max_duration,
            )

        if stage == "test":
            self.test_dataset = SOUSADataset(
                dataset_path=self.dataset_path,
                split="test",
                sample_rate=self.sample_rate,
                max_duration=self.max_duration,
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
