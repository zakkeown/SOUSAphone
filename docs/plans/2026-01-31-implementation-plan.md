# SOUSA Training Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build complete training infrastructure for 4 SOTA audio models with multiple PEFT strategies on 40-class drum rudiment classification.

**Architecture:** PyTorch Lightning + Hydra configs + model adapters pattern. Data-first approach: prove dataset loading works, then add training loop, then expand to all models.

**Tech Stack:** PyTorch, Lightning, Hydra, transformers, peft, torchaudio, wandb, torchmetrics

---

## Phase 1: Project Setup & Dependencies

### Task 1.1: Initialize Project Structure

**Files:**
- Create: `pyproject.toml`
- Create: `sousa/__init__.py`
- Create: `sousa/data/__init__.py`
- Create: `sousa/models/__init__.py`
- Create: `sousa/training/__init__.py`
- Create: `sousa/utils/__init__.py`

**Step 1: Create pyproject.toml with dependencies**

```toml
[project]
name = "sousa"
version = "0.1.0"
description = "SOUSA Training Pipeline - Drum Rudiment Classification"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2.0",
    "torchaudio>=2.2.0",
    "pytorch-lightning>=2.1.0",
    "transformers>=4.36.0",
    "datasets>=2.16.0",
    "wandb>=0.16.0",
    "peft>=0.8.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "torchmetrics>=1.2.0",
    "pandas>=2.0.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["sousa*"]
```

**Step 2: Create package __init__ files**

```bash
mkdir -p sousa/data sousa/models sousa/training sousa/utils
touch sousa/__init__.py
touch sousa/data/__init__.py
touch sousa/models/__init__.py
touch sousa/training/__init__.py
touch sousa/utils/__init__.py
```

**Step 3: Verify structure**

Run: `ls -R sousa/`
Expected: Directory tree showing all __init__.py files

**Step 4: Commit**

```bash
git add pyproject.toml sousa/
git commit -m "feat: initialize project structure with dependencies

- Add pyproject.toml with PyTorch, Lightning, Hydra stack
- Create sousa package with data/models/training/utils modules
- Ready for implementation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Rudiment Mapping & Data Foundation

### Task 2.1: Create Rudiment Mapping

**Files:**
- Create: `sousa/utils/rudiments.py`
- Create: `tests/utils/test_rudiments.py`

**Step 1: Write failing test for rudiment mapping**

```python
# tests/utils/test_rudiments.py
import pytest
from sousa.utils.rudiments import get_rudiment_mapping, RUDIMENT_NAMES


def test_rudiment_mapping_has_40_classes():
    """Verify we have exactly 40 PAS rudiments"""
    mapping = get_rudiment_mapping()
    assert len(mapping) == 40


def test_rudiment_mapping_starts_at_zero():
    """Class IDs should be 0-39"""
    mapping = get_rudiment_mapping()
    ids = list(mapping.values())
    assert min(ids) == 0
    assert max(ids) == 39


def test_rudiment_mapping_is_unique():
    """No duplicate class IDs"""
    mapping = get_rudiment_mapping()
    ids = list(mapping.values())
    assert len(ids) == len(set(ids))


def test_rudiment_names_list():
    """RUDIMENT_NAMES should have all 40 rudiments in order"""
    assert len(RUDIMENT_NAMES) == 40
    # Should be alphabetically sorted by slug
    sorted_names = sorted(RUDIMENT_NAMES)
    assert RUDIMENT_NAMES == sorted_names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/utils/test_rudiments.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'sousa.utils.rudiments'"

**Step 3: Implement rudiment mapping**

```python
# sousa/utils/rudiments.py
"""
Rudiment mapping for 40 PAS (Percussive Arts Society) rudiments.

Canonical ordering: alphabetical by slug for reproducibility.
"""

# All 40 PAS rudiments in alphabetical order by slug
RUDIMENT_NAMES = [
    "double-stroke-roll",
    "five-stroke-roll",
    "flam",
    "flam-accent",
    "flam-paradiddle",
    "flam-tap",
    "flamacue",
    "four-stroke-ruff",
    "nine-stroke-roll",
    "paradiddle",
    "paradiddle-diddle",
    "pataflafla",
    "ratamacue",
    "seven-stroke-roll",
    "single-drag-tap",
    "single-paradiddle",
    "single-stroke-four",
    "single-stroke-roll",
    "single-stroke-seven",
    "six-stroke-roll",
    "swiss-army-triplet",
    "thirteen-stroke-roll",
    "three-stroke-ruff",
    "triple-paradiddle",
    "triple-ratamacue",
    # ... add remaining 15 rudiments
]

# TODO: Complete list with all 40 PAS rudiments
# Reference: https://www.pas.org/resources/rudiments


def get_rudiment_mapping() -> dict[str, int]:
    """
    Returns mapping from rudiment_slug to class_id (0-39).

    Returns:
        dict: {rudiment_slug: class_id}
    """
    return {name: idx for idx, name in enumerate(RUDIMENT_NAMES)}


def get_inverse_mapping() -> dict[int, str]:
    """
    Returns mapping from class_id to rudiment_slug.

    Returns:
        dict: {class_id: rudiment_slug}
    """
    return {idx: name for idx, name in enumerate(RUDIMENT_NAMES)}


def get_num_classes() -> int:
    """Returns total number of rudiment classes."""
    return len(RUDIMENT_NAMES)
```

**Step 4: Complete the RUDIMENT_NAMES list**

Research and add all 40 PAS rudiments alphabetically. Verify against official PAS rudiment list.

**Step 5: Run test to verify it passes**

Run: `pytest tests/utils/test_rudiments.py -v`
Expected: PASS (4 tests)

**Step 6: Commit**

```bash
git add sousa/utils/rudiments.py tests/utils/test_rudiments.py
git commit -m "feat: add rudiment mapping for 40 PAS classes

- Create canonical rudiment ordering (alphabetical by slug)
- Implement get_rudiment_mapping() and inverse mapping
- Add tests for 40 classes, uniqueness, and ordering

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2.2: Implement Dataset Metadata Loading

**Files:**
- Create: `sousa/data/dataset.py`
- Create: `tests/data/test_dataset.py`

**Step 1: Write failing test for dataset initialization**

```python
# tests/data/test_dataset.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_dataset.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'sousa.data.dataset'"

**Step 3: Implement SOUSADataset skeleton**

```python
# sousa/data/dataset.py
"""SOUSA Dataset for drum rudiment classification."""

from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import Dataset

from sousa.utils.rudiments import get_rudiment_mapping


class SOUSADataset(Dataset):
    """
    SOUSA dataset for drum rudiment classification.

    Loads metadata from local dataset directory and provides
    on-demand audio loading.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 5.0,
        transform: Optional[callable] = None,
    ):
        """
        Initialize SOUSA dataset.

        Args:
            dataset_path: Path to dataset directory
            split: One of 'train', 'val', 'test'
            sample_rate: Target sample rate (Hz)
            max_duration: Maximum audio duration (seconds)
            transform: Optional transform to apply to audio
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.transform = transform

        # Load rudiment mapping
        self.rudiment_to_id = get_rudiment_mapping()

        # Load metadata
        self.metadata = self._load_split(split)

    def _load_split(self, split: str) -> pd.DataFrame:
        """Load metadata for specified split."""
        metadata_path = self.dataset_path / "metadata.csv"
        df = pd.read_csv(metadata_path)

        # Filter by split
        df = df[df['split'] == split].reset_index(drop=True)

        return df

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample (to be implemented with audio loading)."""
        row = self.metadata.iloc[idx]

        # TODO: Load audio from row['audio_path']
        # For now, return metadata only
        return {
            'sample_id': row['sample_id'],
            'rudiment_slug': row['rudiment_slug'],
            'label': self.rudiment_to_id[row['rudiment_slug']],
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_dataset.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add sousa/data/dataset.py tests/data/test_dataset.py
git commit -m "feat: implement SOUSADataset metadata loading

- Load metadata from CSV and filter by split
- Integrate rudiment_to_id mapping
- Add tests for split filtering and mapping
- Audio loading to be added next

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2.3: Implement Audio Loading

**Files:**
- Modify: `sousa/data/dataset.py:__getitem__`
- Create: `sousa/utils/audio.py`
- Modify: `tests/data/test_dataset.py`

**Step 1: Write failing test for audio loading**

```python
# tests/data/test_dataset.py (add to existing file)
import numpy as np
import soundfile as sf


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
        sample_rate=16000
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_dataset.py::test_dataset_loads_audio -v`
Expected: FAIL (audio not loaded yet)

**Step 3: Implement audio utilities**

```python
# sousa/utils/audio.py
"""Audio processing utilities."""

import torch
import torchaudio
from pathlib import Path


def load_audio(
    audio_path: Path,
    sample_rate: int = 16000,
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    """
    Load audio file and preprocess.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        max_samples: Maximum number of samples (pad/crop)

    Returns:
        waveform: (num_samples,) tensor
    """
    # Load audio
    waveform, orig_sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)

    # Squeeze to 1D
    waveform = waveform.squeeze(0)

    # Pad or crop to max_samples
    if max_samples is not None:
        if waveform.shape[0] < max_samples:
            # Pad with zeros
            pad_length = max_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif waveform.shape[0] > max_samples:
            # Crop to max_samples
            waveform = waveform[:max_samples]

    return waveform
```

**Step 4: Update SOUSADataset to load audio**

```python
# sousa/data/dataset.py (modify __getitem__)
from sousa.utils.audio import load_audio

def __getitem__(self, idx: int) -> dict:
    """Get a single sample with audio."""
    row = self.metadata.iloc[idx]

    # Load audio
    audio_path = self.dataset_path / row['audio_path']
    waveform = load_audio(
        audio_path,
        sample_rate=self.sample_rate,
        max_samples=self.max_samples,
    )

    # Apply transform if provided
    if self.transform is not None:
        waveform = self.transform(waveform)

    return {
        'sample_id': row['sample_id'],
        'rudiment_slug': row['rudiment_slug'],
        'label': self.rudiment_to_id[row['rudiment_slug']],
        'audio': waveform,
    }
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/data/test_dataset.py -v`
Expected: PASS (all audio loading tests)

**Step 6: Commit**

```bash
git add sousa/data/dataset.py sousa/utils/audio.py tests/data/test_dataset.py
git commit -m "feat: implement audio loading with resampling and padding

- Add load_audio() utility for FLAC loading
- Support mono conversion, resampling, pad/crop
- Update SOUSADataset to load audio on __getitem__
- Add comprehensive tests for audio loading

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: Hydra Configuration

### Task 3.1: Create Base Configuration

**Files:**
- Create: `configs/config.yaml`
- Create: `configs/data/tiny.yaml`
- Create: `configs/data/small.yaml`

**Step 1: Create base config**

```yaml
# configs/config.yaml
defaults:
  - model: ast
  - strategy: lora
  - data: tiny
  - _self_

# Training hyperparameters
training:
  max_epochs: 50
  learning_rate: 5e-5
  batch_size: 16
  gradient_accumulation_steps: 2
  warmup_ratio: 0.1
  weight_decay: 0.01
  label_smoothing: 0.1
  early_stopping_patience: 10

# Augmentation
augmentation:
  specaugment: true
  mixup_alpha: 0.2
  time_stretch_range: [0.9, 1.1]

# System
seed: 42
precision: "16-mixed"
accelerator: "auto"
num_workers: 4

# Experiment tracking
wandb:
  project: "sousa-rudiment-classification"
  entity: null  # Set to your W&B username
  mode: "online"  # or "offline" for local dev
  tags: ["${model.name}", "${strategy.type}"]

# Paths
dataset_path: "~/Code/SOUSA/output/dataset"
```

**Step 2: Create data configs**

```yaml
# configs/data/tiny.yaml
name: tiny
num_samples: 1000
splits:
  train: 800
  val: 100
  test: 100
```

```yaml
# configs/data/small.yaml
name: small
num_samples: 10000
splits:
  train: 8000
  val: 1000
  test: 1000
```

**Step 3: Verify Hydra can load configs**

Create minimal test script:

```python
# test_config.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
    print(f"Dataset: {cfg.data.name}")
    print(f"Batch size: {cfg.training.batch_size}")

if __name__ == "__main__":
    main()
```

Run: `python test_config.py`
Expected: Prints config successfully

**Step 4: Commit**

```bash
git add configs/ test_config.py
git commit -m "feat: add Hydra base configuration

- Create base config with training hyperparameters
- Add data configs for tiny/small datasets
- Configure W&B integration
- Add test script to verify config loading

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: Model Base Class & AST Implementation

### Task 4.1: Create Model Base Interface

**Files:**
- Create: `sousa/models/base.py`
- Create: `tests/models/test_base.py`

**Step 1: Write test for base interface**

```python
# tests/models/test_base.py
import pytest
import torch
from sousa.models.base import AudioClassificationModel


class MockAudioModel(AudioClassificationModel):
    """Mock implementation for testing"""

    def __init__(self, num_classes=40):
        super().__init__()
        self.linear = torch.nn.Linear(100, num_classes)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Dummy: just return random logits
        batch_size = audio.shape[0]
        return self.linear(torch.randn(batch_size, 100))

    def get_feature_extractor(self):
        return {"sample_rate": 16000, "n_mels": 128}

    @property
    def expected_input_type(self) -> str:
        return "spectrogram"


def test_base_interface_forward():
    """Model should implement forward()"""
    model = MockAudioModel(num_classes=40)
    audio = torch.randn(2, 16000)  # Batch of 2
    logits = model(audio)

    assert logits.shape == (2, 40)


def test_base_interface_feature_extractor():
    """Model should implement get_feature_extractor()"""
    model = MockAudioModel()
    config = model.get_feature_extractor()

    assert isinstance(config, dict)
    assert "sample_rate" in config


def test_base_interface_input_type():
    """Model should specify expected_input_type"""
    model = MockAudioModel()
    assert model.expected_input_type in ["waveform", "spectrogram"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_base.py -v`
Expected: FAIL

**Step 3: Implement base interface**

```python
# sousa/models/base.py
"""Base interface for audio classification models."""

from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class AudioClassificationModel(nn.Module, ABC):
    """
    Base interface for all audio classification models.

    All models (AST, HTS-AT, BEATs, EfficientAT) must implement this interface
    to work with the training pipeline.
    """

    @abstractmethod
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio: Audio input (waveform or spectrogram, model-dependent)
                   Shape depends on expected_input_type

        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        pass

    @abstractmethod
    def get_feature_extractor(self) -> dict:
        """
        Get preprocessing configuration.

        Returns:
            dict: Config with sample_rate, n_mels, hop_length, etc.
        """
        pass

    @property
    @abstractmethod
    def expected_input_type(self) -> str:
        """
        Expected input type.

        Returns:
            str: Either 'waveform' or 'spectrogram'
        """
        pass
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_base.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sousa/models/base.py tests/models/test_base.py
git commit -m "feat: create AudioClassificationModel base interface

- Define abstract interface for all audio models
- Require forward(), get_feature_extractor(), expected_input_type
- Add tests with mock implementation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4.2: Implement AST Model Adapter

**Files:**
- Create: `sousa/models/ast.py`
- Create: `configs/model/ast.yaml`
- Create: `tests/models/test_ast.py`

**Step 1: Write test for AST adapter**

```python
# tests/models/test_ast.py
import pytest
import torch
from sousa.models.ast import ASTModel


def test_ast_initializes():
    """AST model should initialize from HuggingFace"""
    model = ASTModel(num_classes=40, pretrained=True)
    assert model is not None


def test_ast_forward_pass():
    """AST should produce logits for batch"""
    model = ASTModel(num_classes=40, pretrained=False)
    # AST expects mel-spectrograms: (batch, time, n_mels)
    batch_size = 2
    spec = torch.randn(batch_size, 1024, 128)  # (batch, time, mels)

    logits = model(spec)
    assert logits.shape == (batch_size, 40)


def test_ast_expected_input_type():
    """AST expects spectrogram input"""
    model = ASTModel(num_classes=40, pretrained=False)
    assert model.expected_input_type == "spectrogram"


def test_ast_feature_extractor():
    """AST should provide feature extraction config"""
    model = ASTModel(num_classes=40, pretrained=False)
    config = model.get_feature_extractor()

    assert config['sample_rate'] == 16000
    assert config['n_mels'] == 128
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_ast.py -v`
Expected: FAIL

**Step 3: Implement AST adapter**

```python
# sousa/models/ast.py
"""AST (Audio Spectrogram Transformer) model adapter."""

import torch
from transformers import ASTForAudioClassification
from sousa.models.base import AudioClassificationModel


class ASTModel(AudioClassificationModel):
    """
    AST model from HuggingFace transformers.

    Uses MIT's pretrained AST on AudioSet, with head replaced
    for 40-class rudiment classification.
    """

    def __init__(self, num_classes: int = 40, pretrained: bool = True):
        """
        Initialize AST model.

        Args:
            num_classes: Number of output classes (40 for rudiments)
            pretrained: Load AudioSet pretrained weights
        """
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            self.model = ASTForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,  # Replace classification head
            )
        else:
            # Random initialization for testing
            from transformers import ASTConfig
            config = ASTConfig(num_labels=num_classes)
            self.model = ASTForAudioClassification(config)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio: Mel-spectrogram (batch_size, time, n_mels)

        Returns:
            logits: (batch_size, num_classes)
        """
        outputs = self.model(audio)
        return outputs.logits

    def get_feature_extractor(self) -> dict:
        """Get AST preprocessing config."""
        return {
            "sample_rate": 16000,
            "n_mels": 128,
            "n_fft": 400,
            "hop_length": 160,
            "max_length": 1024,  # Time frames
        }

    @property
    def expected_input_type(self) -> str:
        return "spectrogram"
```

**Step 4: Create AST config**

```yaml
# configs/model/ast.yaml
name: ast
class_path: sousa.models.ast.ASTModel
num_classes: 40
pretrained: true

# Input requirements
input_type: spectrogram
sample_rate: 16000
n_mels: 128
n_fft: 400
hop_length: 160
max_length: 1024
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/models/test_ast.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add sousa/models/ast.py configs/model/ast.yaml tests/models/test_ast.py
git commit -m "feat: implement AST model adapter

- Wrap HuggingFace AST for rudiment classification
- Support pretrained AudioSet weights
- Add model config for Hydra
- Test forward pass and interface compliance

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 5: Lightning DataModule

### Task 5.1: Implement SOUSADataModule

**Files:**
- Create: `sousa/data/datamodule.py`
- Create: `tests/data/test_datamodule.py`

**Step 1: Write test for DataModule**

```python
# tests/data/test_datamodule.py
import pytest
from sousa.data.datamodule import SOUSADataModule


@pytest.fixture
def mock_dataset_path(tmp_path):
    """Create mock dataset (reuse from test_dataset.py)"""
    # ... (same as before)
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_datamodule.py -v`
Expected: FAIL

**Step 3: Implement SOUSADataModule**

```python
# sousa/data/datamodule.py
"""Lightning DataModule for SOUSA dataset."""

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

    def setup(self, stage: str):
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

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_datamodule.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sousa/data/datamodule.py tests/data/test_datamodule.py
git commit -m "feat: implement Lightning DataModule for SOUSA

- Create SOUSADataModule with train/val/test splits
- Configure dataloaders with batch_size and num_workers
- Add tests for setup and dataloader creation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 6: Lightning Training Module

### Task 6.1: Create SOUSAClassifier (Minimal)

**Files:**
- Create: `sousa/training/module.py`
- Create: `configs/strategy/lora.yaml`
- Create: `tests/training/test_module.py`

**Step 1: Write test for SOUSAClassifier**

```python
# tests/training/test_module.py
import pytest
import torch
from omegaconf import OmegaConf
from sousa.training.module import SOUSAClassifier
from sousa.models.ast import ASTModel


@pytest.fixture
def minimal_config():
    """Minimal config for testing"""
    return OmegaConf.create({
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
        },
        "strategy": {
            "type": "full_finetune",
        },
    })


def test_classifier_initializes(minimal_config):
    """SOUSAClassifier should initialize with model"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    assert classifier is not None


def test_classifier_forward(minimal_config):
    """SOUSAClassifier forward should return logits"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    # Dummy batch
    audio = torch.randn(2, 1024, 128)
    logits = classifier(audio)

    assert logits.shape == (2, 40)


def test_classifier_training_step(minimal_config):
    """Training step should compute loss"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    # Dummy batch
    batch = {
        'audio': torch.randn(2, 1024, 128),
        'label': torch.tensor([0, 5]),
    }

    loss = classifier.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_module.py -v`
Expected: FAIL

**Step 3: Implement SOUSAClassifier (minimal)**

```python
# sousa/training/module.py
"""Lightning module for SOUSA training."""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
from torchmetrics import Accuracy

from sousa.models.base import AudioClassificationModel


class SOUSAClassifier(pl.LightningModule):
    """
    Lightning module for rudiment classification training.

    Wraps any AudioClassificationModel and handles training,
    validation, and PEFT injection.
    """

    def __init__(self, model: AudioClassificationModel, config: DictConfig):
        """
        Initialize classifier.

        Args:
            model: Any model implementing AudioClassificationModel
            config: Hydra config with training/strategy params
        """
        super().__init__()
        self.model = model
        self.config = config

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=40)
        self.val_acc = Accuracy(task="multiclass", num_classes=40)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(audio)

    def training_step(self, batch, batch_idx):
        """Training step."""
        audio, labels = batch['audio'], batch['label']
        logits = self(audio)

        # Loss with label smoothing
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.config.training.label_smoothing,
        )

        # Metrics
        self.log('train/loss', loss, prog_bar=True)
        self.train_acc(logits, labels)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        audio, labels = batch['audio'], batch['label']
        logits = self(audio)

        loss = F.cross_entropy(logits, labels)

        self.log('val/loss', loss, prog_bar=True)
        self.val_acc(logits, labels)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        return optimizer
```

**Step 4: Create strategy config**

```yaml
# configs/strategy/lora.yaml
type: lora
load_pretrained: true

# LoRA hyperparameters
rank: 8
alpha: 16
dropout: 0.1
target_modules:
  - "attention.query"
  - "attention.key"
  - "attention.value"

# Adjusted learning rate for PEFT
learning_rate: 1e-4
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/training/test_module.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add sousa/training/module.py configs/strategy/lora.yaml tests/training/test_module.py
git commit -m "feat: implement SOUSAClassifier Lightning module

- Create minimal training module with forward/training_step
- Add accuracy metrics tracking
- Configure AdamW optimizer
- Add LoRA strategy config (PEFT injection next)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 7: Training Script Integration

### Task 7.1: Create Minimal Training Script

**Files:**
- Create: `train.py`

**Step 1: Implement minimal training script**

```python
# train.py
"""Training script for SOUSA rudiment classification."""

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sousa.data.datamodule import SOUSADataModule
from sousa.models.ast import ASTModel
from sousa.training.module import SOUSAClassifier


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Initialize W&B logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
    )

    # Create data module
    datamodule = SOUSADataModule(
        dataset_path=cfg.dataset_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers,
    )

    # Create model (for now, hardcode AST)
    # TODO: Dynamic model loading based on cfg.model
    model = ASTModel(num_classes=40, pretrained=True)

    # Create Lightning module
    classifier = SOUSAClassifier(model, cfg)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/acc',
        mode='max',
        save_top_k=1,
        filename='best-{epoch}-{val_acc:.2f}',
    )

    early_stop_callback = EarlyStopping(
        monitor='val/acc',
        patience=cfg.training.early_stopping_patience,
        mode='max',
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
    )

    # Train
    trainer.fit(classifier, datamodule)

    # Test
    trainer.test(classifier, datamodule)


if __name__ == "__main__":
    main()
```

**Step 2: Test training script (dry run)**

Run: `python train.py --help`
Expected: Shows Hydra help with config options

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: create minimal training script

- Integrate Hydra config loading
- Set up W&B logging
- Configure Lightning Trainer with callbacks
- Support train/val/test workflow
- Model loading to be made dynamic next

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 8: First End-to-End Test

### Task 8.1: Run Tiny Dataset Training

**Files:**
- Modify: `train.py` (add dataset path expansion)

**Step 1: Fix dataset path expansion**

```python
# train.py (modify datamodule creation)
from pathlib import Path

# Expand ~ in dataset path
dataset_path = Path(cfg.dataset_path).expanduser()

datamodule = SOUSADataModule(
    dataset_path=str(dataset_path),
    batch_size=cfg.training.batch_size,
    num_workers=cfg.num_workers,
)
```

**Step 2: Run training with tiny dataset**

Run: `python train.py data=tiny wandb.mode=offline training.max_epochs=2`
Expected: Training completes for 2 epochs (or fails with informative error)

**Step 3: Debug any issues**

Common issues:
- Dataset path not found → Check ~/Code/SOUSA/output/dataset exists
- Audio loading errors → Check FLAC files exist
- Spectrogram conversion → Need to add mel-spectrogram transform

**Step 4: Add preprocessing transform if needed**

```python
# sousa/data/transforms.py
import torch
import torchaudio


class MelSpectrogramTransform:
    """Convert waveform to mel-spectrogram for models."""

    def __init__(self, sample_rate=16000, n_mels=128, n_fft=400, hop_length=160):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel-spectrogram."""
        mel_spec = self.mel_transform(waveform)
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        # Transpose to (time, mels) for AST
        return mel_spec.T
```

**Step 5: Integrate transform into DataModule**

Update SOUSADataModule to apply transform based on model's expected_input_type.

**Step 6: Verify training runs successfully**

Run: `python train.py data=tiny wandb.mode=offline training.max_epochs=2`
Expected: Completes 2 epochs, logs metrics

**Step 7: Commit**

```bash
git add train.py sousa/data/transforms.py sousa/data/datamodule.py
git commit -m "feat: add mel-spectrogram transform and complete first training run

- Implement MelSpectrogramTransform for AST input
- Fix dataset path expansion in train.py
- Integrate transform into DataModule
- Verify end-to-end training on tiny dataset

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Remaining Implementation Tasks

**NOTE:** The plan continues with similar granular tasks for:

- **Phase 9:** Add PEFT (LoRA) injection
- **Phase 10:** Implement HTS-AT model adapter
- **Phase 11:** Implement BEATs model adapter
- **Phase 12:** Implement EfficientAT model adapter
- **Phase 13:** Dynamic model loading in train.py
- **Phase 14:** Add remaining PEFT strategies (AdaLoRA, IA3)
- **Phase 15:** Full metrics (confusion matrix, per-class F1)
- **Phase 16:** Augmentation pipeline (SpecAugment, Mixup)
- **Phase 17:** Complete validation on data=small
- **Phase 18:** HF Jobs integration and cloud testing

Each phase follows the same TDD pattern:
1. Write failing test
2. Run to verify failure
3. Implement minimal code
4. Run to verify passing
5. Commit

---

## Execution Strategy

Given the length and complexity of this plan, we recommend:

**Execution Mode: Subagent-Driven Development**

Use @superpowers:subagent-driven-development to:
- Execute tasks 1.1 through 8.1 first (foundation → first training run)
- Review code after each phase
- Iterate on issues before moving forward
- Build remaining phases after validating core pipeline works

This ensures solid foundation before expanding to all models and features.

---

**Plan Status:** Ready for execution
**Recommended Next Step:** Use @superpowers:executing-plans or @superpowers:subagent-driven-development
