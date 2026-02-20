"""Tests for FeatureInferenceDataModule."""

import pytest

from sousa.data.feature_inference_datamodule import FeatureInferenceDataModule


# mock_fi_dataset fixture is defined in tests/data/conftest.py


def test_datamodule_setup(mock_fi_dataset):
    dm = FeatureInferenceDataModule(
        dataset_path=str(mock_fi_dataset), batch_size=2, num_workers=0
    )
    dm.setup("fit")
    assert hasattr(dm, "train_dataset")
    assert hasattr(dm, "val_dataset")


def test_train_dataloader(mock_fi_dataset):
    dm = FeatureInferenceDataModule(
        dataset_path=str(mock_fi_dataset), batch_size=2, num_workers=0
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert "raw_onsets" in batch
    assert "target_features" in batch
    assert "attention_mask" in batch


def test_val_dataloader(mock_fi_dataset):
    dm = FeatureInferenceDataModule(
        dataset_path=str(mock_fi_dataset), batch_size=2, num_workers=0
    )
    dm.setup("fit")
    batch = next(iter(dm.val_dataloader()))
    assert "raw_onsets" in batch
    assert "target_features" in batch
    assert "attention_mask" in batch


def test_batch_shapes(mock_fi_dataset):
    dm = FeatureInferenceDataModule(
        dataset_path=str(mock_fi_dataset), batch_size=2, num_workers=0,
        max_seq_len=64,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["raw_onsets"].shape[0] == 2
    assert batch["raw_onsets"].shape[2] == 3
    assert batch["target_features"].shape[2] == 12
    assert batch["attention_mask"].shape[0] == 2


def test_datamodule_augmentation_params(mock_fi_dataset):
    """DataModule should pass augmentation params to train dataset."""
    dm = FeatureInferenceDataModule(
        dataset_path=str(mock_fi_dataset), batch_size=2, num_workers=0,
        timing_jitter_std_ms=20.0, strength_noise_std=0.3,
    )
    dm.setup("fit")
    assert dm.train_dataset.augment is True
    assert dm.train_dataset.timing_jitter_std_ms == 20.0
    assert dm.train_dataset.strength_noise_std == 0.3


def test_val_dataset_no_augmentation(mock_fi_dataset):
    """Validation dataset should not use augmentation."""
    dm = FeatureInferenceDataModule(
        dataset_path=str(mock_fi_dataset), batch_size=2, num_workers=0
    )
    dm.setup("fit")
    assert dm.val_dataset.augment is False
