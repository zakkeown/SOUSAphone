"""Lightning DataModule for onset-based rudiment classification."""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sousa.data.onset_dataset import OnsetDataset


class OnsetDataModule(pl.LightningDataModule):
    """DataModule wrapping OnsetDataset for stroke-level classification.

    No audio transforms needed — onset data is deterministic.
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        max_seq_len: int = 128,
        max_samples: int = None,
        soundfonts: list[str] = None,
        augmentation_presets: list[str] = None,
        tempo_range: tuple[int, int] = None,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        self.soundfonts = soundfonts
        self.augmentation_presets = augmentation_presets
        self.tempo_range = tempo_range
        self.pin_memory = torch.cuda.is_available()

    def _dataset_kwargs(self):
        return dict(
            dataset_path=self.dataset_path,
            max_seq_len=self.max_seq_len,
            max_samples=self.max_samples,
            soundfonts=self.soundfonts,
            augmentation_presets=self.augmentation_presets,
            tempo_range=self.tempo_range,
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = OnsetDataset(split="train", **self._dataset_kwargs())
            self.val_dataset = OnsetDataset(split="val", **self._dataset_kwargs())
        if stage == "test":
            self.test_dataset = OnsetDataset(split="test", **self._dataset_kwargs())

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
