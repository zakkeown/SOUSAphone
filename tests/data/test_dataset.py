import pytest
import numpy as np
import soundfile as sf
import torch

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

def test_dataset_getitem_returns_correct_structure(mock_dataset_with_audio):
    """Dataset __getitem__ should return dict with sample_id, rudiment_slug, label, audio"""
    dataset = SOUSADataset(dataset_path=str(mock_dataset_with_audio), split="train")
    sample = dataset[0]

    assert "sample_id" in sample
    assert "rudiment_slug" in sample
    assert "label" in sample
    assert "audio" in sample
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

@pytest.fixture
def mock_dataset_with_audio(tmp_path):
    """Create mock dataset with actual audio files"""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir()

    # Create mock audio files
    sample_rate = 16000
    duration = 2.0
    samples = int(sample_rate * duration)

    for sample_id in ["sample_001", "sample_002"]:
        audio_path = audio_dir / f"{sample_id}.flac"
        # Create random audio
        audio_data = np.random.randn(samples).astype(np.float32) * 0.1
        sf.write(audio_path, audio_data, sample_rate)

    # Create metadata
    metadata = dataset_dir / "metadata.csv"
    metadata.write_text(
        "sample_id,rudiment_slug,split,audio_path,duration\n"
        f"sample_001,flam,train,audio/sample_001.flac,{duration}\n"
        f"sample_002,paradiddle,train,audio/sample_002.flac,{duration}\n"
    )

    return dataset_dir

def test_dataset_loads_audio(mock_dataset_with_audio):
    """Dataset should load audio waveform"""
    dataset = SOUSADataset(
        dataset_path=str(mock_dataset_with_audio),
        split="train"
    )
    sample = dataset[0]

    assert 'audio' in sample
    assert isinstance(sample['audio'], torch.Tensor)
    assert sample['audio'].dtype == torch.float32

def test_dataset_resamples_audio(mock_dataset_with_audio):
    """Dataset should resample to target sample rate"""
    dataset = SOUSADataset(
        dataset_path=str(mock_dataset_with_audio),
        split="train",
        sample_rate=16000,
        max_duration=2.0  # Match the audio duration
    )
    sample = dataset[0]

    # Audio should be resampled to 16kHz
    assert sample['audio'].shape[0] == 16000 * 2  # 2 seconds

def test_dataset_pads_short_audio(mock_dataset_with_audio):
    """Dataset should pad audio shorter than max_duration"""
    dataset = SOUSADataset(
        dataset_path=str(mock_dataset_with_audio),
        split="train",
        max_duration=5.0  # Longer than 2.0s audio
    )
    sample = dataset[0]

    expected_samples = int(16000 * 5.0)
    assert sample['audio'].shape[0] == expected_samples

def test_dataset_converts_stereo_to_mono(tmp_path):
    """Dataset should convert stereo audio to mono"""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir()

    # Create stereo audio file
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    stereo_audio = np.random.randn(samples, 2).astype(np.float32) * 0.1  # 2 channels
    audio_path = audio_dir / "stereo.flac"
    sf.write(audio_path, stereo_audio, sample_rate)

    # Create metadata
    metadata = dataset_dir / "metadata.csv"
    metadata.write_text(
        "sample_id,rudiment_slug,split,audio_path,duration\n"
        f"stereo,flam,train,audio/stereo.flac,{duration}\n"
    )

    dataset = SOUSADataset(dataset_path=str(dataset_dir), split="train", max_duration=1.0)
    sample = dataset[0]

    # Should be mono (1D)
    assert sample['audio'].ndim == 1
    assert sample['audio'].shape[0] == 16000

def test_dataset_crops_long_audio(tmp_path):
    """Dataset should crop audio longer than max_duration"""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir()

    # Create 10-second audio
    sample_rate = 16000
    duration = 10.0
    samples = int(sample_rate * duration)
    long_audio = np.random.randn(samples).astype(np.float32) * 0.1
    audio_path = audio_dir / "long.flac"
    sf.write(audio_path, long_audio, sample_rate)

    metadata = dataset_dir / "metadata.csv"
    metadata.write_text(
        "sample_id,rudiment_slug,split,audio_path,duration\n"
        f"long,flam,train,audio/long.flac,{duration}\n"
    )

    # Max duration 5 seconds
    dataset = SOUSADataset(dataset_path=str(dataset_dir), split="train", max_duration=5.0)
    sample = dataset[0]

    # Should be cropped to 5 seconds
    expected_samples = int(16000 * 5.0)
    assert sample['audio'].shape[0] == expected_samples

def test_load_audio_missing_file_raises_error(tmp_path):
    """load_audio should raise FileNotFoundError for missing file"""
    from sousa.utils.audio import load_audio
    missing_file = tmp_path / "missing.flac"

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        load_audio(missing_file)

def test_load_audio_invalid_sample_rate(tmp_path):
    """load_audio should raise ValueError for invalid sample rate"""
    from sousa.utils.audio import load_audio

    # Create a valid audio file
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    audio_path = dataset_dir / "test.flac"
    audio_data = np.random.randn(16000).astype(np.float32) * 0.1
    sf.write(audio_path, audio_data, 16000)

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        load_audio(audio_path, sample_rate=0)

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        load_audio(audio_path, sample_rate=-1)

def test_load_audio_invalid_max_samples(tmp_path):
    """load_audio should raise ValueError for invalid max_samples"""
    from sousa.utils.audio import load_audio

    # Create a valid audio file
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    audio_path = dataset_dir / "test.flac"
    audio_data = np.random.randn(16000).astype(np.float32) * 0.1
    sf.write(audio_path, audio_data, 16000)

    with pytest.raises(ValueError, match="max_samples must be positive"):
        load_audio(audio_path, max_samples=0)

    with pytest.raises(ValueError, match="max_samples must be positive"):
        load_audio(audio_path, max_samples=-1)

def test_load_audio_resampling(tmp_path):
    """load_audio should correctly resample audio"""
    from sousa.utils.audio import load_audio

    # Create audio at 8kHz
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    audio_path = dataset_dir / "test.flac"
    audio_data = np.random.randn(8000).astype(np.float32) * 0.1  # 1 second at 8kHz
    sf.write(audio_path, audio_data, 8000)

    # Load and resample to 16kHz
    audio = load_audio(audio_path, sample_rate=16000, max_samples=16000)

    # Should be resampled to 16kHz (1 second = 16000 samples)
    assert audio.shape[0] == 16000
