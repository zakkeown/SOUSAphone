"""Tests for FeatureInferenceDataset."""

import pandas as pd
import pytest
import torch

from sousa.data.feature_inference_dataset import FeatureInferenceDataset


@pytest.fixture
def mock_fi_dataset(tmp_path):
    """Create minimal mock dataset for feature inference training."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "labels").mkdir()

    # Create mock metadata CSV
    metadata = dataset_dir / "metadata.csv"
    metadata.write_text(
        "sample_id,rudiment_slug,split,audio_path,duration,tempo_bpm,soundfont,augmentation_preset\n"
        "s001,flam,train,audio/flam/s001.flac,2.5,120,piano,clean\n"
        "s002,single-stroke-roll,train,audio/single-stroke-roll/s002.flac,3.0,100,piano,clean\n"
        "s003,flam,val,audio/flam/s003.flac,2.2,120,piano,clean\n"
    )

    # Create mock strokes parquet
    strokes_data = []
    # Sample s001: 8 strokes of a flam
    for i in range(8):
        strokes_data.append({
            "sample_id": "s001",
            "actual_time_ms": i * 250.0,
            "actual_velocity": 80 + (i % 2) * 30,
            "is_grace_note": i % 2 == 0,
            "is_accent": i % 2 == 1,
            "stroke_type": "flam" if i % 2 == 0 else "tap",
            "hand": "R" if i % 2 == 0 else "L",
            "diddle_position": float("nan"),
            "flam_spacing_ms": 30.0 if i % 2 == 0 else float("nan"),
            "buzz_count": float("nan"),
        })
    # Sample s002: 16 strokes
    for i in range(16):
        strokes_data.append({
            "sample_id": "s002",
            "actual_time_ms": i * 125.0,
            "actual_velocity": 90,
            "is_grace_note": False,
            "is_accent": i % 4 == 0,
            "stroke_type": "tap",
            "hand": "R" if i % 2 == 0 else "L",
            "diddle_position": float("nan"),
            "flam_spacing_ms": float("nan"),
            "buzz_count": float("nan"),
        })
    # Sample s003 (val)
    for i in range(8):
        strokes_data.append({
            "sample_id": "s003",
            "actual_time_ms": i * 250.0,
            "actual_velocity": 85,
            "is_grace_note": False,
            "is_accent": False,
            "stroke_type": "tap",
            "hand": "R" if i % 2 == 0 else "L",
            "diddle_position": float("nan"),
            "flam_spacing_ms": float("nan"),
            "buzz_count": float("nan"),
        })

    strokes_df = pd.DataFrame(strokes_data)
    strokes_df.to_parquet(dataset_dir / "labels" / "strokes.parquet")

    return dataset_dir


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
