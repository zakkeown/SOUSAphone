"""Lightning DataModule for Feature Inference Model training."""

from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sousa.data.feature_inference_dataset import FeatureInferenceDataset


class FeatureInferenceDataModule(pl.LightningDataModule):
    """DataModule wrapping FeatureInferenceDataset for training.

    Provides train/val/test dataloaders with appropriate augmentation
    settings (augmentation on for train, off for val/test).
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        max_seq_len: int = 256,
        timing_jitter_std_ms: float = 10.0,
        strength_noise_std: float = 0.15,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.timing_jitter_std_ms = timing_jitter_std_ms
        self.strength_noise_std = strength_noise_std
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_dataset = FeatureInferenceDataset(
                self.dataset_path,
                split="train",
                max_seq_len=self.max_seq_len,
                augment=True,
                timing_jitter_std_ms=self.timing_jitter_std_ms,
                strength_noise_std=self.strength_noise_std,
            )
            self.val_dataset = FeatureInferenceDataset(
                self.dataset_path,
                split="val",
                max_seq_len=self.max_seq_len,
                augment=False,
            )
        if stage == "test":
            self.test_dataset = FeatureInferenceDataset(
                self.dataset_path,
                split="test",
                max_seq_len=self.max_seq_len,
                augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
