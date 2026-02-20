"""Tests for FeatureInferenceDataset."""

import pytest
import torch

from sousa.data.feature_inference_dataset import FeatureInferenceDataset


# mock_fi_dataset fixture is defined in tests/data/conftest.py


def test_dataset_loads(mock_fi_dataset):
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train")
    assert len(ds) == 2


def test_getitem_returns_correct_keys(mock_fi_dataset):
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train")
    sample = ds[0]
    assert "raw_onsets" in sample
    assert "target_features" in sample
    assert "attention_mask" in sample


def test_raw_onsets_shape(mock_fi_dataset):
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train", max_seq_len=32)
    sample = ds[0]
    assert sample["raw_onsets"].shape == (32, 3)
    assert sample["raw_onsets"].dtype == torch.float32


def test_target_features_shape(mock_fi_dataset):
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train", max_seq_len=32)
    sample = ds[0]
    assert sample["target_features"].shape == (32, 12)
    assert sample["target_features"].dtype == torch.float32


def test_attention_mask_shape(mock_fi_dataset):
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train", max_seq_len=32)
    sample = ds[0]
    assert sample["attention_mask"].shape == (32,)
    assert sample["attention_mask"][:8].sum() == 8
    assert sample["attention_mask"][8:].sum() == 0


def test_noise_augmentation_changes_input(mock_fi_dataset):
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train", augment=True)
    s1 = ds[0]["raw_onsets"]
    s2 = ds[0]["raw_onsets"]
    assert not torch.allclose(s1, s2)


def test_no_augmentation_is_deterministic(mock_fi_dataset):
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train", augment=False)
    s1 = ds[0]["raw_onsets"]
    s2 = ds[0]["raw_onsets"]
    assert torch.allclose(s1, s2)
