import pytest
from pathlib import Path

from sousa.data.dataset import SOUSADataset


@pytest.fixture
def mock_dataset_path(tmp_path):
    """Create minimal mock dataset structure"""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Create mock metadata CSV
    metadata = dataset_dir / "metadata.csv"
    metadata.write_text(
        "sample_id,rudiment_slug,split,audio_path,duration\n"
        "sample_001,flam,train,audio/sample_001.flac,2.5\n"
        "sample_002,paradiddle,train,audio/sample_002.flac,3.0\n"
        "sample_003,flam,val,audio/sample_003.flac,2.2\n"
    )

    return dataset_dir

def test_dataset_loads_train_split(mock_dataset_path):
    """Dataset should load only train split samples"""
    dataset = SOUSADataset(
        dataset_path=str(mock_dataset_path),
        split="train"
    )
    assert len(dataset) == 2  # Only 2 train samples

def test_dataset_loads_val_split(mock_dataset_path):
    """Dataset should load only val split samples"""
    dataset = SOUSADataset(
        dataset_path=str(mock_dataset_path),
        split="val"
    )
    assert len(dataset) == 1  # Only 1 val sample

def test_dataset_has_rudiment_mapping(mock_dataset_path):
    """Dataset should have rudiment_slug to class_id mapping"""
    dataset = SOUSADataset(
        dataset_path=str(mock_dataset_path),
        split="train"
    )
    assert hasattr(dataset, 'rudiment_to_id')
    assert isinstance(dataset.rudiment_to_id, dict)

def test_dataset_getitem_returns_correct_structure(mock_dataset_path):
    """Dataset __getitem__ should return dict with sample_id, rudiment_slug, label"""
    dataset = SOUSADataset(dataset_path=str(mock_dataset_path), split="train")
    sample = dataset[0]

    assert "sample_id" in sample
    assert "rudiment_slug" in sample
    assert "label" in sample
    assert isinstance(sample["label"], int)

def test_dataset_missing_metadata_raises_error(tmp_path):
    """Dataset should raise FileNotFoundError when metadata.csv is missing"""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    # No metadata.csv created

    with pytest.raises(FileNotFoundError):
        SOUSADataset(dataset_path=str(dataset_dir), split="train")

def test_dataset_empty_split_raises_error(mock_dataset_path):
    """Dataset should raise ValueError when split has no samples"""
    with pytest.raises(ValueError, match="No samples found for split 'test'"):
        SOUSADataset(dataset_path=str(mock_dataset_path), split="test")
