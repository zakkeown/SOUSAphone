import pytest
import numpy as np
import soundfile as sf

from sousa.data.datamodule import SOUSADataModule


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
        "sample_002,single-paradiddle,train,audio/sample_002.flac,3.0\n"
        "sample_003,flam,val,audio/sample_003.flac,2.2\n"
        "sample_004,drag,test,audio/sample_004.flac,2.0\n"
    )

    # Create audio directory and files for dataloader tests
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir()

    sample_rate = 16000
    duration = 2.0
    samples = int(sample_rate * duration)

    for sample_id in ["sample_001", "sample_002", "sample_003", "sample_004"]:
        audio_path = audio_dir / f"{sample_id}.flac"
        # Create random audio
        audio_data = np.random.randn(samples).astype(np.float32) * 0.1
        sf.write(audio_path, audio_data, sample_rate)

    return dataset_dir


def test_datamodule_initializes(mock_dataset_path):
    """DataModule should initialize with config"""
    dm = SOUSADataModule(
        dataset_path=str(mock_dataset_path),
        batch_size=4,
        num_workers=0,
    )
    assert dm is not None


def test_datamodule_setup(mock_dataset_path):
    """Setup should create train/val/test datasets"""
    dm = SOUSADataModule(
        dataset_path=str(mock_dataset_path),
        batch_size=4,
    )
    dm.setup("fit")

    assert hasattr(dm, 'train_dataset')
    assert hasattr(dm, 'val_dataset')
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0


def test_datamodule_dataloaders(mock_dataset_path):
    """DataModule should provide dataloaders"""
    dm = SOUSADataModule(
        dataset_path=str(mock_dataset_path),
        batch_size=2,
    )
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    assert train_loader is not None
    assert val_loader is not None

    # Check batch
    batch = next(iter(train_loader))
    assert 'audio' in batch
    assert 'label' in batch


def test_datamodule_setup_test_stage(mock_dataset_path):
    """Setup should create test dataset for test stage"""
    dm = SOUSADataModule(
        dataset_path=str(mock_dataset_path),
        batch_size=4,
    )
    dm.setup("test")

    assert hasattr(dm, 'test_dataset')
    assert len(dm.test_dataset) > 0


def test_datamodule_test_dataloader(mock_dataset_path):
    """DataModule should provide test dataloader"""
    dm = SOUSADataModule(
        dataset_path=str(mock_dataset_path),
        batch_size=2,
    )
    dm.setup("test")

    test_loader = dm.test_dataloader()

    assert test_loader is not None

    # Check batch
    batch = next(iter(test_loader))
    assert 'audio' in batch
    assert 'label' in batch
