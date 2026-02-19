# HF Space Demo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a two-model pipeline (Feature Inference Model + OnsetTransformer) with a Gradio demo on HF Spaces that classifies drum rudiments from uploaded audio.

**Architecture:** Audio → librosa onset detection + beat tracking → Feature Inference Transformer (predicts 12-dim features from raw onsets) → existing OnsetTransformer (classifies rudiment from 12-dim features). Deployed as a Gradio app on HF Spaces (free CPU tier).

**Tech Stack:** PyTorch, PyTorch Lightning, Hydra, librosa, Gradio, matplotlib, HuggingFace Hub

**Design doc:** `docs/plans/2026-02-19-hf-space-demo-design.md`

---

## Task 1: Feature Inference Model Architecture

The Feature Inference Model is a small Transformer that predicts the 12-dimensional per-stroke features the OnsetTransformer expects, given raw onset detection output (onset_time_ms, onset_strength, tempo_bpm).

**Files:**
- Create: `sousa/models/feature_inference.py`
- Test: `tests/models/test_feature_inference.py`

**Step 1: Write the failing tests**

```python
# tests/models/test_feature_inference.py
"""Tests for FeatureInferenceModel."""

import pytest
import torch

from sousa.models.feature_inference import FeatureInferenceModel


def test_model_initializes():
    """FeatureInferenceModel should initialize with default params."""
    model = FeatureInferenceModel()
    assert model is not None


def test_forward_shape():
    """Forward pass should return (batch, seq_len, 12) features."""
    model = FeatureInferenceModel()
    # Input: (batch=2, seq_len=32, input_dim=3)
    raw_onsets = torch.randn(2, 32, 3)
    mask = torch.ones(2, 32)
    output = model(raw_onsets, attention_mask=mask)
    assert output.shape == (2, 32, 12)


def test_forward_with_padding():
    """Model should handle padded sequences correctly."""
    model = FeatureInferenceModel()
    raw_onsets = torch.randn(2, 64, 3)
    # Second sequence is shorter (only 20 real tokens)
    mask = torch.ones(2, 64)
    mask[1, 20:] = 0
    output = model(raw_onsets, attention_mask=mask)
    assert output.shape == (2, 64, 12)


def test_forward_no_mask():
    """Model should work without attention mask."""
    model = FeatureInferenceModel()
    raw_onsets = torch.randn(2, 32, 3)
    output = model(raw_onsets)
    assert output.shape == (2, 32, 12)


def test_output_dim_configurable():
    """Output dimension should be configurable."""
    model = FeatureInferenceModel(output_dim=6)
    raw_onsets = torch.randn(2, 32, 3)
    output = model(raw_onsets)
    assert output.shape == (2, 32, 6)


def test_parameter_count():
    """Model should be small (under 200K params)."""
    model = FeatureInferenceModel()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 200_000


def test_gradients_flow():
    """Gradients should flow through the model."""
    model = FeatureInferenceModel()
    raw_onsets = torch.randn(2, 32, 3, requires_grad=True)
    output = model(raw_onsets)
    loss = output.sum()
    loss.backward()
    assert raw_onsets.grad is not None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/test_feature_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sousa.models.feature_inference'`

**Step 3: Write the implementation**

```python
# sousa/models/feature_inference.py
"""Feature Inference Model: predicts 12-dim onset features from raw onset detection output."""

from typing import Optional

import torch
import torch.nn as nn


class FeatureInferenceModel(nn.Module):
    """Transformer that predicts OnsetTransformer's 12-dim feature vectors
    from raw onset detection output (onset_time_ms, onset_strength, tempo_bpm).

    This bridges the gap between audio onset detection (which only provides
    timing and strength) and the OnsetTransformer's rich feature space
    (which includes stroke types, sticking, grace notes, etc.).

    Args:
        input_dim: Raw onset features per stroke (default 3: ioi_ms, strength, tempo_bpm)
        output_dim: Target features per stroke (default 12: OnsetTransformer feature space)
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Number of Transformer encoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for positional encoding
    """

    # Indices of binary features in the 12-dim output
    BINARY_INDICES = [2, 3, 4, 5, 6, 10]  # is_grace, is_accent, is_tap, is_diddle, hand_R, is_buzz
    # Indices of continuous features in the 12-dim output
    CONTINUOUS_INDICES = [0, 1, 7, 8, 9, 11]  # norm_ioi, norm_velocity, diddle_pos, norm_flam_spacing, position_in_beat, norm_buzz_count

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, raw_onsets: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            raw_onsets: (batch, seq_len, input_dim) — raw onset features per stroke
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            (batch, seq_len, output_dim) — predicted features per stroke
        """
        batch_size, seq_len, _ = raw_onsets.shape

        positions = torch.arange(seq_len, device=raw_onsets.device)
        x = self.input_proj(raw_onsets) + self.pos_embedding(positions)

        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.output_proj(x)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/test_feature_inference.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add sousa/models/feature_inference.py tests/models/test_feature_inference.py
git commit -m "feat: add FeatureInferenceModel architecture"
```

---

## Task 2: Feature Inference Dataset

Dataset that loads SOUSA strokes.parquet, computes raw onset inputs (simulating onset detection with noise augmentation), and targets the full 12-dim feature vectors.

**Files:**
- Create: `sousa/data/feature_inference_dataset.py`
- Test: `tests/data/test_feature_inference_dataset.py`
- Reference: `sousa/data/onset_dataset.py` — reuse `_encode_strokes()` logic for computing target features

**Step 1: Write the failing tests**

```python
# tests/data/test_feature_inference_dataset.py
"""Tests for FeatureInferenceDataset."""

import pytest
import numpy as np
import pandas as pd
import torch

from sousa.data.feature_inference_dataset import FeatureInferenceDataset


@pytest.fixture
def mock_fi_dataset(tmp_path):
    """Create minimal mock dataset for feature inference training."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "labels").mkdir()

    # Create mock metadata
    metadata = dataset_dir / "metadata.csv"
    metadata.write_text(
        "sample_id,rudiment_slug,split,audio_path,duration,tempo_bpm,soundfont,augmentation_preset\n"
        "s001,flam,train,audio/flam/s001.flac,2.5,120,piano,clean\n"
        "s002,single-stroke-roll,train,audio/single-stroke-roll/s002.flac,3.0,100,piano,clean\n"
        "s003,flam,val,audio/flam/s003.flac,2.2,120,piano,clean\n"
    )

    # Create mock strokes parquet with required columns
    strokes_data = []
    # Sample s001: 8 strokes of a flam (grace+primary pairs)
    for i in range(8):
        strokes_data.append({
            "sample_id": "s001",
            "actual_time_ms": i * 250.0,
            "actual_velocity": 80 + (i % 2) * 30,  # alternating
            "is_grace_note": i % 2 == 0,
            "is_accent": i % 2 == 1,
            "stroke_type": "flam" if i % 2 == 0 else "tap",
            "hand": "R" if i % 2 == 0 else "L",
            "diddle_position": float("nan"),
            "flam_spacing_ms": 30.0 if i % 2 == 0 else float("nan"),
            "buzz_count": float("nan"),
        })
    # Sample s002: 16 strokes of single-stroke-roll
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
    # Sample s003 (val): 8 strokes
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

    # Create splits.json
    import json
    splits = {
        "train": ["s001", "s002"],
        "val": ["s003"],
        "test": [],
    }
    (dataset_dir / "splits.json").write_text(json.dumps(splits))

    return dataset_dir


def test_dataset_loads(mock_fi_dataset):
    """Dataset should load train split."""
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train")
    assert len(ds) == 2


def test_getitem_returns_correct_keys(mock_fi_dataset):
    """Each sample should have raw_onsets, target_features, and attention_mask."""
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train")
    sample = ds[0]
    assert "raw_onsets" in sample
    assert "target_features" in sample
    assert "attention_mask" in sample


def test_raw_onsets_shape(mock_fi_dataset):
    """raw_onsets should be (max_seq_len, 3)."""
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train", max_seq_len=32)
    sample = ds[0]
    assert sample["raw_onsets"].shape == (32, 3)
    assert sample["raw_onsets"].dtype == torch.float32


def test_target_features_shape(mock_fi_dataset):
    """target_features should be (max_seq_len, 12)."""
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train", max_seq_len=32)
    sample = ds[0]
    assert sample["target_features"].shape == (32, 12)
    assert sample["target_features"].dtype == torch.float32


def test_attention_mask_shape(mock_fi_dataset):
    """attention_mask should be (max_seq_len,), 1 for real, 0 for padding."""
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train", max_seq_len=32)
    sample = ds[0]
    assert sample["attention_mask"].shape == (32,)
    # s001 has 8 strokes, so first 8 should be 1, rest 0
    assert sample["attention_mask"][:8].sum() == 8
    assert sample["attention_mask"][8:].sum() == 0


def test_noise_augmentation_changes_input(mock_fi_dataset):
    """With augment=True, raw_onsets should differ across calls (stochastic)."""
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train", augment=True)
    s1 = ds[0]["raw_onsets"]
    s2 = ds[0]["raw_onsets"]
    # Stochastic — very unlikely to be identical
    assert not torch.allclose(s1, s2)


def test_no_augmentation_is_deterministic(mock_fi_dataset):
    """With augment=False, raw_onsets should be deterministic."""
    ds = FeatureInferenceDataset(str(mock_fi_dataset), split="train", augment=False)
    s1 = ds[0]["raw_onsets"]
    s2 = ds[0]["raw_onsets"]
    assert torch.allclose(s1, s2)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/data/test_feature_inference_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

The dataset needs to:
1. Load strokes.parquet and metadata (reuse `load_split_metadata` from `sousa.data.dataset`)
2. For each sample: compute raw onset input (ioi_ms, strength, tempo_bpm) from ground truth
3. Compute the 12-dim target features (reuse `OnsetDataset._encode_strokes()` logic)
4. Optionally add noise augmentation to the raw input (timing jitter, strength noise)

```python
# sousa/data/feature_inference_dataset.py
"""Dataset for training the Feature Inference Model."""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sousa.data.onset_dataset import OnsetDataset
from sousa.data.dataset import load_split_metadata


class FeatureInferenceDataset(Dataset):
    """Dataset that provides (raw_onsets, target_features) pairs for training
    the Feature Inference Model.

    Raw onsets simulate what librosa onset detection would produce:
    (ioi_ms, onset_strength, tempo_bpm) per stroke.

    Target features are the full 12-dim vectors computed by OnsetDataset._encode_strokes().

    Args:
        dataset_path: Path to SOUSA dataset
        split: 'train', 'val', or 'test'
        max_seq_len: Pad/truncate to this length
        augment: Add noise to raw onsets (timing jitter, strength noise)
        timing_jitter_ms: Std dev of Gaussian timing jitter in ms
        strength_noise: Std dev of Gaussian strength noise (fraction of velocity)
    """

    def __init__(
        self,
        dataset_path: str,
        split: str,
        max_seq_len: int = 256,
        augment: bool = False,
        timing_jitter_ms: float = 10.0,
        strength_noise: float = 0.15,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.timing_jitter_ms = timing_jitter_ms
        self.strength_noise = strength_noise

        # Load metadata
        meta = load_split_metadata(self.dataset_path, split)
        self.metadata = meta

        # Load strokes
        strokes_path = self.dataset_path / "labels" / "strokes.parquet"
        if not strokes_path.exists():
            raise FileNotFoundError(f"Strokes file not found: {strokes_path}")

        valid_ids = set(meta["sample_id"].values)
        strokes = pd.read_parquet(strokes_path)
        strokes = strokes[strokes["sample_id"].isin(valid_ids)]

        self.strokes_by_sample = {
            sid: group.sort_values("actual_time_ms").reset_index(drop=True)
            for sid, group in strokes.groupby("sample_id")
        }

        self.tempo_by_sample = dict(zip(meta["sample_id"], meta["tempo_bpm"]))

        # Reuse OnsetDataset's feature encoding logic
        self._onset_dataset = OnsetDataset.__new__(OnsetDataset)
        self._onset_dataset.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        sample_id = row["sample_id"]
        tempo_bpm = self.tempo_by_sample[sample_id]

        strokes_df = self.strokes_by_sample.get(sample_id)
        if strokes_df is None or len(strokes_df) == 0:
            return self._empty_sample()

        # Compute target features using OnsetDataset's encoding
        target_features = self._onset_dataset._encode_strokes(strokes_df, tempo_bpm)

        # Compute raw onset input (simulating onset detection)
        raw_onsets = self._compute_raw_onsets(strokes_df, tempo_bpm)

        seq_len = len(target_features)

        # Pad or truncate
        if seq_len >= self.max_seq_len:
            target_features = target_features[: self.max_seq_len]
            raw_onsets = raw_onsets[: self.max_seq_len]
            mask = torch.ones(self.max_seq_len, dtype=torch.float32)
        else:
            pad_len = self.max_seq_len - seq_len
            target_features = torch.cat(
                [target_features, torch.zeros(pad_len, 12, dtype=torch.float32)]
            )
            raw_onsets = torch.cat(
                [raw_onsets, torch.zeros(pad_len, 3, dtype=torch.float32)]
            )
            mask = torch.cat(
                [
                    torch.ones(seq_len, dtype=torch.float32),
                    torch.zeros(pad_len, dtype=torch.float32),
                ]
            )

        return {
            "raw_onsets": raw_onsets,
            "target_features": target_features,
            "attention_mask": mask,
        }

    def _compute_raw_onsets(
        self, df: pd.DataFrame, tempo_bpm: float
    ) -> torch.Tensor:
        """Compute raw onset features simulating onset detection output.

        Returns (num_strokes, 3) tensor: [ioi_ms, onset_strength, tempo_bpm].
        """
        times = df["actual_time_ms"].values.copy()
        velocities = df["actual_velocity"].values.astype(np.float32).copy()
        n = len(df)

        # Add noise augmentation during training
        if self.augment:
            times = times + np.random.normal(0, self.timing_jitter_ms, n)
            times = np.maximum(times, 0)  # no negative times
            times.sort()  # maintain order after jitter

            vel_noise = np.random.normal(0, self.strength_noise, n)
            velocities = np.clip(velocities * (1 + vel_noise), 0, 127)

        # IOI in ms (0 for first stroke)
        ioi_ms = np.zeros(n, dtype=np.float32)
        if n > 1:
            ioi_ms[1:] = np.diff(times)

        # Onset strength normalized to [0, 1]
        onset_strength = velocities / 127.0

        # Tempo repeated for every stroke
        tempo = np.full(n, tempo_bpm, dtype=np.float32)

        features = np.stack([ioi_ms, onset_strength, tempo], axis=1)
        return torch.from_numpy(features)

    def _empty_sample(self) -> Dict[str, Any]:
        return {
            "raw_onsets": torch.zeros(self.max_seq_len, 3, dtype=torch.float32),
            "target_features": torch.zeros(self.max_seq_len, 12, dtype=torch.float32),
            "attention_mask": torch.zeros(self.max_seq_len, dtype=torch.float32),
        }
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_feature_inference_dataset.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add sousa/data/feature_inference_dataset.py tests/data/test_feature_inference_dataset.py
git commit -m "feat: add FeatureInferenceDataset with noise augmentation"
```

---

## Task 3: Feature Inference Training Module

A Lightning module with mixed BCE+MSE loss for training the Feature Inference Model.

**Files:**
- Create: `sousa/training/feature_module.py`
- Test: `tests/training/test_feature_module.py`

**Step 1: Write the failing tests**

```python
# tests/training/test_feature_module.py
"""Tests for FeatureInferenceModule."""

import pytest
import torch
from omegaconf import OmegaConf

from sousa.models.feature_inference import FeatureInferenceModel
from sousa.training.feature_module import FeatureInferenceModule


@pytest.fixture
def fi_config():
    return OmegaConf.create({
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
        },
    })


def test_module_initializes(fi_config):
    model = FeatureInferenceModel()
    module = FeatureInferenceModule(model, fi_config)
    assert module is not None


def test_training_step_returns_loss(fi_config):
    model = FeatureInferenceModel()
    module = FeatureInferenceModule(model, fi_config)

    batch = {
        "raw_onsets": torch.randn(4, 32, 3),
        "target_features": torch.randn(4, 32, 12),
        "attention_mask": torch.ones(4, 32),
    }
    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_validation_step_returns_loss(fi_config):
    model = FeatureInferenceModel()
    module = FeatureInferenceModule(model, fi_config)

    batch = {
        "raw_onsets": torch.randn(4, 32, 3),
        "target_features": torch.randn(4, 32, 12),
        "attention_mask": torch.ones(4, 32),
    }
    loss = module.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_loss_respects_mask(fi_config):
    """Loss should only consider non-padded positions."""
    model = FeatureInferenceModel()
    module = FeatureInferenceModule(model, fi_config)

    raw = torch.randn(2, 32, 3)
    target = torch.randn(2, 32, 12)

    # Full mask vs partial mask should give different losses
    mask_full = torch.ones(2, 32)
    mask_partial = torch.ones(2, 32)
    mask_partial[:, 16:] = 0

    loss_full = module._compute_loss(raw, target, mask_full)
    loss_partial = module._compute_loss(raw, target, mask_partial)

    assert not torch.allclose(loss_full, loss_partial)


def test_configure_optimizers(fi_config):
    model = FeatureInferenceModel()
    module = FeatureInferenceModule(model, fi_config)
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/training/test_feature_module.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# sousa/training/feature_module.py
"""Lightning module for Feature Inference Model training."""

from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Optimizer

from sousa.models.feature_inference import FeatureInferenceModel


class FeatureInferenceModule(pl.LightningModule):
    """Lightning module for training the Feature Inference Model.

    Uses mixed loss: BCE for binary features, MSE for continuous features.
    Only computes loss over non-padded positions (respects attention_mask).
    """

    def __init__(self, model: FeatureInferenceModel, config: DictConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

    def forward(
        self, raw_onsets: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.model(raw_onsets, attention_mask=attention_mask)

    def _compute_loss(
        self,
        raw_onsets: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixed BCE + MSE loss over non-padded positions."""
        pred = self.model(raw_onsets, attention_mask=mask)

        # Expand mask to feature dimension
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq, 1)

        # Binary features: BCE with logits (model outputs raw logits)
        binary_idx = FeatureInferenceModel.BINARY_INDICES
        binary_pred = pred[:, :, binary_idx]
        binary_target = target[:, :, binary_idx]
        bce_loss = F.binary_cross_entropy_with_logits(
            binary_pred, binary_target, reduction="none"
        )
        bce_loss = (bce_loss * mask_expanded).sum() / mask.sum().clamp(min=1)

        # Continuous features: MSE
        cont_idx = FeatureInferenceModel.CONTINUOUS_INDICES
        cont_pred = pred[:, :, cont_idx]
        cont_target = target[:, :, cont_idx]
        mse_loss = F.mse_loss(cont_pred, cont_target, reduction="none")
        mse_loss = (mse_loss * mask_expanded).sum() / mask.sum().clamp(min=1)

        return bce_loss + mse_loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._compute_loss(
            batch["raw_onsets"], batch["target_features"], batch["attention_mask"]
        )
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._compute_loss(
            batch["raw_onsets"], batch["target_features"], batch["attention_mask"]
        )
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/training/test_feature_module.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add sousa/training/feature_module.py tests/training/test_feature_module.py
git commit -m "feat: add FeatureInferenceModule with mixed BCE+MSE loss"
```

---

## Task 4: Training Script + Hydra Config

A dedicated training script for the Feature Inference Model, plus Hydra config.

**Files:**
- Create: `train_feature_model.py`
- Create: `configs/model/feature_inference.yaml`
- Create: `sousa/data/feature_inference_datamodule.py`
- Test: `tests/data/test_feature_inference_datamodule.py`

**Step 1: Write the DataModule tests**

```python
# tests/data/test_feature_inference_datamodule.py
"""Tests for FeatureInferenceDataModule."""

import pytest
import torch

from sousa.data.feature_inference_datamodule import FeatureInferenceDataModule


def test_datamodule_setup(mock_fi_dataset):
    """DataModule should set up train/val datasets."""
    dm = FeatureInferenceDataModule(
        dataset_path=str(mock_fi_dataset), batch_size=2, num_workers=0
    )
    dm.setup("fit")
    assert hasattr(dm, "train_dataset")
    assert hasattr(dm, "val_dataset")


def test_train_dataloader(mock_fi_dataset):
    """Train dataloader should yield batches."""
    dm = FeatureInferenceDataModule(
        dataset_path=str(mock_fi_dataset), batch_size=2, num_workers=0
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert "raw_onsets" in batch
    assert "target_features" in batch
    assert "attention_mask" in batch
```

Note: reuse the `mock_fi_dataset` fixture from Task 2 by adding a shared `conftest.py`.

**Step 2: Write DataModule, config, and training script**

```python
# sousa/data/feature_inference_datamodule.py
"""Lightning DataModule for Feature Inference Model training."""

from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sousa.data.feature_inference_dataset import FeatureInferenceDataset


class FeatureInferenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        max_seq_len: int = 256,
        timing_jitter_ms: float = 10.0,
        strength_noise: float = 0.15,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.timing_jitter_ms = timing_jitter_ms
        self.strength_noise = strength_noise
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = FeatureInferenceDataset(
                self.dataset_path, split="train", max_seq_len=self.max_seq_len,
                augment=True, timing_jitter_ms=self.timing_jitter_ms,
                strength_noise=self.strength_noise,
            )
            self.val_dataset = FeatureInferenceDataset(
                self.dataset_path, split="val", max_seq_len=self.max_seq_len,
                augment=False,
            )
        if stage == "test":
            self.test_dataset = FeatureInferenceDataset(
                self.dataset_path, split="test", max_seq_len=self.max_seq_len,
                augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        )
```

```yaml
# configs/model/feature_inference.yaml
name: feature_inference
class_path: sousa.models.feature_inference.FeatureInferenceModel
input_type: onset
input_dim: 3
output_dim: 12
d_model: 64
nhead: 4
num_layers: 3
dim_feedforward: 128
dropout: 0.1
max_seq_len: 256
```

```python
# train_feature_model.py
"""Training script for Feature Inference Model."""

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

from sousa.models.feature_inference import FeatureInferenceModel
from sousa.data.feature_inference_datamodule import FeatureInferenceDataModule
from sousa.training.feature_module import FeatureInferenceModule


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        tags=["feature-inference"],
    )

    dataset_path = Path(cfg.dataset_path).expanduser()

    model = FeatureInferenceModel(
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
    )

    datamodule = FeatureInferenceDataModule(
        dataset_path=str(dataset_path),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers,
        max_seq_len=cfg.model.max_seq_len,
    )

    module = FeatureInferenceModule(model, cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_loss:.4f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=cfg.training.early_stopping_patience,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
```

**Step 3: Run tests**

Run: `python -m pytest tests/data/test_feature_inference_datamodule.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add train_feature_model.py configs/model/feature_inference.yaml \
    sousa/data/feature_inference_datamodule.py tests/data/test_feature_inference_datamodule.py
git commit -m "feat: add feature inference training script, config, and datamodule"
```

---

## Task 5: Train the Feature Inference Model

Run training on the SOUSA dataset. This task is manual/interactive.

**Step 1: Run training**

```bash
python train_feature_model.py \
    model=feature_inference \
    strategy=full_finetune \
    dataset_path=~/Code/SOUSA/output/dataset \
    training.learning_rate=1e-3 \
    training.batch_size=4 \
    training.max_epochs=50 \
    wandb.mode=online
```

**Step 2: Evaluate results**

Check W&B dashboard for val/loss convergence. The model should converge since:
- The mapping from (ioi, strength, tempo) to features is deterministic (modulo noise)
- The model has access to full sequence context (Transformer attention)
- Training data is abundant (100K samples)

**Step 3: Export model weights**

```bash
# After training, extract best checkpoint weights
python -c "
import torch
ckpt = torch.load('path/to/best-checkpoint.ckpt', map_location='cpu')
torch.save(ckpt['state_dict'], 'hf_upload_feature/pytorch_model.bin')
"
```

**Step 4: Upload to HF**

Create `zkeown/sousaphone-feature-model` on HF Hub and push weights + config.

**Step 5: Commit any changes**

```bash
git add -A && git commit -m "chore: add feature inference model artifacts"
```

---

## Task 6: Inference Pipeline

End-to-end audio → rudiment prediction pipeline.

**Files:**
- Create: `sousa/inference/__init__.py`
- Create: `sousa/inference/pipeline.py`
- Test: `tests/inference/test_pipeline.py`

**Step 1: Write the failing tests**

```python
# tests/inference/test_pipeline.py
"""Tests for inference pipeline."""

import pytest
import numpy as np
import torch

from sousa.inference.pipeline import OnsetDetector, RudimentPipeline


class TestOnsetDetector:
    def test_detect_returns_times_and_strengths(self):
        """Should return onset times and strengths from audio."""
        detector = OnsetDetector()
        # 2 seconds of audio at 22050 Hz with some impulses
        audio = np.zeros(44100, dtype=np.float32)
        audio[11025] = 1.0  # impulse at 0.5s
        audio[22050] = 1.0  # impulse at 1.0s
        audio[33075] = 1.0  # impulse at 1.5s

        times, strengths = detector.detect(audio, sr=22050)
        assert isinstance(times, np.ndarray)
        assert isinstance(strengths, np.ndarray)
        assert len(times) == len(strengths)
        assert len(times) > 0

    def test_estimate_tempo(self):
        """Should estimate tempo from audio."""
        detector = OnsetDetector()
        # Generate click track at 120 BPM (0.5s intervals)
        sr = 22050
        audio = np.zeros(sr * 4, dtype=np.float32)
        for i in range(8):
            idx = int(i * 0.5 * sr)
            if idx < len(audio):
                audio[idx : idx + 100] = 1.0

        tempo = detector.estimate_tempo(audio, sr=sr)
        assert isinstance(tempo, float)
        assert tempo > 0


class TestRudimentPipeline:
    def test_pipeline_initializes(self):
        """Pipeline should initialize with model paths."""
        # Use dummy paths — just test initialization logic
        pipeline = RudimentPipeline(
            feature_model_path=None, classifier_model_path=None
        )
        assert pipeline is not None

    def test_prepare_raw_onsets(self):
        """Should convert onset times/strengths to model input tensor."""
        pipeline = RudimentPipeline(
            feature_model_path=None, classifier_model_path=None
        )
        times = np.array([0.0, 0.25, 0.5, 0.75])
        strengths = np.array([0.8, 0.6, 0.9, 0.5])
        tempo = 120.0

        raw_onsets, mask = pipeline.prepare_raw_onsets(times, strengths, tempo)
        assert raw_onsets.shape[0] == 1  # batch dim
        assert raw_onsets.shape[2] == 3  # (ioi_ms, strength, tempo)
        assert mask.shape[0] == 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/inference/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# sousa/inference/__init__.py
# (empty)
```

```python
# sousa/inference/pipeline.py
"""End-to-end audio → rudiment prediction pipeline."""

from typing import Optional, Tuple

import numpy as np
import torch

from sousa.models.feature_inference import FeatureInferenceModel
from sousa.models.onset_transformer import OnsetTransformerModel
from sousa.utils.rudiments import get_inverse_mapping


class OnsetDetector:
    """Detect onsets and estimate tempo from audio using librosa."""

    def detect(
        self, audio: np.ndarray, sr: int = 22050
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect onset times and strengths.

        Returns:
            times: onset times in seconds
            strengths: onset strengths (0-1 normalized)
        """
        import librosa

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, onset_envelope=onset_env, backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Get strengths at onset frames
        strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])
        if len(strengths) > 0:
            strengths = strengths / strengths.max()  # normalize to [0, 1]

        return onset_times, strengths

    def estimate_tempo(self, audio: np.ndarray, sr: int = 22050) -> float:
        """Estimate tempo in BPM."""
        import librosa

        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        return float(tempo)


class RudimentPipeline:
    """Full audio → rudiment prediction pipeline.

    Chains: onset detection → feature inference → OnsetTransformer classification.
    """

    def __init__(
        self,
        feature_model_path: Optional[str] = None,
        classifier_model_path: Optional[str] = None,
        max_seq_len: int = 256,
    ):
        self.max_seq_len = max_seq_len
        self.id2label = get_inverse_mapping()
        self.detector = OnsetDetector()

        # Load models if paths provided
        self.feature_model: Optional[FeatureInferenceModel] = None
        self.classifier: Optional[OnsetTransformerModel] = None

        if feature_model_path is not None:
            self.feature_model = FeatureInferenceModel()
            state = torch.load(feature_model_path, map_location="cpu")
            self.feature_model.load_state_dict(state)
            self.feature_model.eval()

        if classifier_model_path is not None:
            self.classifier = OnsetTransformerModel(
                num_classes=40, feature_dim=12, d_model=64, nhead=4,
                num_layers=3, dim_feedforward=128, dropout=0.0, max_seq_len=256,
            )
            state = torch.load(classifier_model_path, map_location="cpu")
            self.classifier.load_state_dict(state)
            self.classifier.eval()

    def prepare_raw_onsets(
        self,
        onset_times: np.ndarray,
        onset_strengths: np.ndarray,
        tempo_bpm: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert onset detection output to model input tensors.

        Returns:
            raw_onsets: (1, max_seq_len, 3)
            attention_mask: (1, max_seq_len)
        """
        n = min(len(onset_times), self.max_seq_len)

        ioi_ms = np.zeros(n, dtype=np.float32)
        if n > 1:
            ioi_ms[1:] = np.diff(onset_times[:n]) * 1000.0  # sec → ms

        strengths = onset_strengths[:n].astype(np.float32)
        tempo = np.full(n, tempo_bpm, dtype=np.float32)

        features = np.stack([ioi_ms, strengths, tempo], axis=1)  # (n, 3)

        # Pad to max_seq_len
        padded = np.zeros((self.max_seq_len, 3), dtype=np.float32)
        padded[:n] = features

        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:n] = 1.0

        return (
            torch.from_numpy(padded).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )

    @torch.no_grad()
    def predict(
        self, audio: np.ndarray, sr: int = 22050
    ) -> dict:
        """Run full pipeline: audio → rudiment prediction.

        Returns dict with: predicted_rudiment, confidence, top5, onset_times,
        onset_strengths, tempo_bpm, predicted_features.
        """
        assert self.feature_model is not None and self.classifier is not None

        # Stage 1: Onset detection
        onset_times, onset_strengths = self.detector.detect(audio, sr=sr)
        tempo_bpm = self.detector.estimate_tempo(audio, sr=sr)

        if len(onset_times) == 0:
            return {"error": "No onsets detected in audio"}

        # Stage 2: Prepare input
        raw_onsets, mask = self.prepare_raw_onsets(
            onset_times, onset_strengths, tempo_bpm
        )

        # Stage 3: Feature inference
        features = self.feature_model(raw_onsets, attention_mask=mask)
        # Apply sigmoid to binary features for downstream use
        binary_idx = FeatureInferenceModel.BINARY_INDICES
        features_processed = features.clone()
        features_processed[:, :, binary_idx] = torch.sigmoid(
            features[:, :, binary_idx]
        )

        # Stage 4: Classification
        logits = self.classifier(features_processed, attention_mask=mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # Top-5 predictions
        top5_probs, top5_indices = probs.topk(5)
        top5 = [
            {"rudiment": self.id2label[idx.item()], "confidence": prob.item()}
            for prob, idx in zip(top5_probs, top5_indices)
        ]

        return {
            "predicted_rudiment": top5[0]["rudiment"],
            "confidence": top5[0]["confidence"],
            "top5": top5,
            "onset_times": onset_times,
            "onset_strengths": onset_strengths,
            "tempo_bpm": tempo_bpm,
            "predicted_features": features_processed.squeeze(0).numpy(),
            "attention_mask": mask.squeeze(0).numpy(),
        }
```

**Step 4: Run tests**

Run: `python -m pytest tests/inference/test_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sousa/inference/__init__.py sousa/inference/pipeline.py \
    tests/inference/__init__.py tests/inference/test_pipeline.py
git commit -m "feat: add end-to-end inference pipeline"
```

---

## Task 7: Visualizations

matplotlib-based visualization functions for the Gradio demo.

**Files:**
- Create: `space/visualizations.py`

**Step 1: Write visualization functions**

```python
# space/visualizations.py
"""Visualization functions for SOUSAphone Gradio demo."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Stroke type colors
STROKE_COLORS = {
    "tap": "#4A90D9",
    "accent": "#E74C3C",
    "grace/flam": "#F39C12",
    "diddle": "#2ECC71",
    "buzz": "#9B59B6",
}

# Rudiment notation patterns (canonical sticking for each rudiment)
# Populated for the most common rudiments; others show "see PAS chart"
RUDIMENT_STICKING = {
    "single-stroke-roll": "R L R L R L R L",
    "single-stroke-four": "R L R L",
    "single-stroke-seven": "R L R L R L R",
    "double-stroke-open-roll": "R R L L R R L L",
    "five-stroke-roll": "R R L L R",
    "seven-stroke-roll": "R R L L R R L",
    "nine-stroke-roll": "R R L L R R L L R",
    "single-paradiddle": "R L R R L R L L",
    "double-paradiddle": "R L R L R R L R L R L L",
    "triple-paradiddle": "R L R L R L R R L R L R L R L L",
    "single-paradiddle-diddle": "R L R R L L",
    "flam": "lR rL",
    "flam-accent": "lR R L rL L R",
    "flam-tap": "lR R rL L",
    "flamacue": "lR L R L rL",
    "flam-paradiddle": "lR L R R rL R L L",
    "drag": "llR rrL",
    "single-drag-tap": "llR L rrL R",
    "double-drag-tap": "llR llR L rrL rrL R",
    "lesson-25": "llR L R L llR L R L",
    "single-ratamacue": "llR L R L",
    "double-ratamacue": "llR llR L R L",
    "triple-ratamacue": "llR llR llR L R L",
    "multiple-bounce-roll": "z z z z z z z z",
}


def plot_onset_timeline(
    audio: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    predicted_features: np.ndarray,
    attention_mask: np.ndarray,
) -> plt.Figure:
    """Plot waveform with color-coded onset markers."""
    fig, ax = plt.subplots(figsize=(12, 3))

    # Plot waveform
    t = np.arange(len(audio)) / sr
    ax.plot(t, audio, color="#cccccc", linewidth=0.5, alpha=0.7)

    # Plot onset markers colored by predicted stroke type
    n_real = int(attention_mask.sum())
    for i in range(min(n_real, len(onset_times))):
        feat = predicted_features[i]
        # Determine stroke type from binary features
        color = STROKE_COLORS["tap"]  # default
        if feat[2] > 0.5:  # is_grace
            color = STROKE_COLORS["grace/flam"]
        elif feat[5] > 0.5:  # is_diddle
            color = STROKE_COLORS["diddle"]
        elif feat[10] > 0.5:  # is_buzz
            color = STROKE_COLORS["buzz"]
        elif feat[3] > 0.5:  # is_accent
            color = STROKE_COLORS["accent"]

        ax.axvline(onset_times[i], color=color, alpha=0.8, linewidth=1.5)

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in STROKE_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Detected Onsets")
    fig.tight_layout()
    return fig


def plot_feature_heatmap(
    predicted_features: np.ndarray,
    attention_mask: np.ndarray,
) -> plt.Figure:
    """Plot 12×N heatmap of predicted features."""
    n_real = int(attention_mask.sum())
    features = predicted_features[:n_real].T  # (12, n_real)

    feature_names = [
        "IOI", "velocity", "grace", "accent", "tap",
        "diddle", "hand_R", "diddle_pos", "flam_sp",
        "beat_pos", "buzz", "buzz_ct",
    ]

    fig, ax = plt.subplots(figsize=(max(8, n_real * 0.3), 4))
    im = ax.imshow(features, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(12))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("Stroke #")
    ax.set_title("Predicted Features")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def format_rudiment_notation(rudiment_name: str) -> str:
    """Get notation string for a rudiment."""
    sticking = RUDIMENT_STICKING.get(rudiment_name)
    display_name = rudiment_name.replace("-", " ").title()
    if sticking:
        return f"**{display_name}**\n\nSticking: `{sticking}`"
    return f"**{display_name}**\n\nSee PAS International Drum Rudiments chart for sticking."
```

**Step 2: Commit**

```bash
git add space/visualizations.py
git commit -m "feat: add visualization functions for Gradio demo"
```

---

## Task 8: Gradio App

The main Gradio application for the HF Space.

**Files:**
- Create: `space/app.py`
- Create: `space/requirements.txt`
- Create: `space/README.md`

**Step 1: Write the Gradio app**

```python
# space/app.py
"""SOUSAphone Gradio demo — Drum Rudiment Classifier."""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download

# Add parent to path so we can import sousa modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from sousa.inference.pipeline import RudimentPipeline
from space.visualizations import (
    plot_onset_timeline,
    plot_feature_heatmap,
    format_rudiment_notation,
)


def load_pipeline() -> RudimentPipeline:
    """Download models from HF Hub and initialize pipeline."""
    feature_model_path = hf_hub_download(
        repo_id="zkeown/sousaphone-feature-model", filename="pytorch_model.bin"
    )
    classifier_model_path = hf_hub_download(
        repo_id="zkeown/sousaphone", filename="pytorch_model.bin"
    )
    return RudimentPipeline(
        feature_model_path=feature_model_path,
        classifier_model_path=classifier_model_path,
    )


pipeline = load_pipeline()


def classify(audio_input):
    """Main classification function called by Gradio."""
    if audio_input is None:
        return None, None, None, None

    sr, audio = audio_input

    # Convert to float32 mono
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Normalize
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()

    result = pipeline.predict(audio, sr=sr)

    if "error" in result:
        return result["error"], None, None, None

    # Confidence chart (dict for gr.Label)
    confidences = {r["rudiment"]: r["confidence"] for r in result["top5"]}

    # Rudiment notation
    notation = format_rudiment_notation(result["predicted_rudiment"])

    # Onset timeline plot
    timeline_fig = plot_onset_timeline(
        audio, sr,
        result["onset_times"],
        result["predicted_features"],
        result["attention_mask"],
    )

    # Feature heatmap
    heatmap_fig = plot_feature_heatmap(
        result["predicted_features"],
        result["attention_mask"],
    )

    return confidences, notation, timeline_fig, heatmap_fig


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(label="Upload or record a drum rudiment"),
    outputs=[
        gr.Label(num_top_classes=5, label="Prediction"),
        gr.Markdown(label="Rudiment Notation"),
        gr.Plot(label="Onset Timeline"),
        gr.Plot(label="Feature Heatmap"),
    ],
    title="SOUSAphone — Drum Rudiment Classifier",
    description=(
        "Upload a recording of a drum rudiment and SOUSAphone will identify which "
        "of the 40 PAS International Drum Rudiments it is. The model detects onsets, "
        "infers stroke-level features, and classifies using a lightweight Transformer."
    ),
    examples=[],  # Add example audio files later
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
```

```
# space/requirements.txt
torch>=2.0
librosa>=0.10.0
gradio>=5.0
numpy>=1.24
soundfile>=0.12
matplotlib>=3.7
huggingface_hub>=0.20
```

```yaml
# space/README.md
---
title: SOUSAphone
emoji: "\U0001F941"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.12.0"
python_version: "3.11"
app_file: app.py
pinned: false
models:
  - zkeown/sousaphone
  - zkeown/sousaphone-feature-model
datasets:
  - zkeown/sousa
tags:
  - audio-classification
  - drum-rudiments
  - music
  - percussion
short_description: "Classify all 40 PAS drum rudiments from audio"
---

# SOUSAphone Demo

Upload a drum rudiment performance and SOUSAphone identifies which of the 40 PAS International Drum Rudiments it is.

**Pipeline:** Audio → Onset Detection → Feature Inference → OnsetTransformer → Classification

See [zkeown/sousaphone](https://huggingface.co/zkeown/sousaphone) for model details.
```

**Step 2: Commit**

```bash
git add space/app.py space/requirements.txt space/README.md
git commit -m "feat: add Gradio app and Space configuration"
```

---

## Task 9: Deploy to HF Spaces

Push the Space to HF and verify it works.

**Step 1: Create the Space repo on HF**

```bash
# Create the Space
huggingface-cli repo create sousaphone-demo --type space --space-sdk gradio
```

**Step 2: Push space code**

```bash
cd /Users/zakkeown/Code/SOUSAphone

# Clone the space repo
git clone https://huggingface.co/spaces/zkeown/sousaphone-demo /tmp/sousaphone-demo

# Copy space files + sousa modules needed for inference
cp space/app.py space/requirements.txt space/README.md /tmp/sousaphone-demo/
cp -r sousa/ /tmp/sousaphone-demo/sousa/
cp -r space/visualizations.py /tmp/sousaphone-demo/space/visualizations.py

# Push
cd /tmp/sousaphone-demo
git add -A && git commit -m "Initial Space deployment" && git push
```

**Step 3: Verify**

Visit `https://huggingface.co/spaces/zkeown/sousaphone-demo` and test with audio.

**Step 4: Final commit back in SOUSAphone**

```bash
cd /Users/zakkeown/Code/SOUSAphone
git add -A && git commit -m "chore: finalize HF Space demo"
```

---

## Summary

| Task | Component | New Files |
|------|-----------|-----------|
| 1 | Feature Inference Model | `sousa/models/feature_inference.py`, `tests/models/test_feature_inference.py` |
| 2 | Feature Inference Dataset | `sousa/data/feature_inference_dataset.py`, `tests/data/test_feature_inference_dataset.py` |
| 3 | Training Module | `sousa/training/feature_module.py`, `tests/training/test_feature_module.py` |
| 4 | Training Script + Config | `train_feature_model.py`, `configs/model/feature_inference.yaml`, `sousa/data/feature_inference_datamodule.py` |
| 5 | Train Model | (manual — run training, evaluate, upload weights) |
| 6 | Inference Pipeline | `sousa/inference/pipeline.py`, `tests/inference/test_pipeline.py` |
| 7 | Visualizations | `space/visualizations.py` |
| 8 | Gradio App | `space/app.py`, `space/requirements.txt`, `space/README.md` |
| 9 | Deploy | (manual — push to HF Spaces) |

Tasks 1-4 and 6-8 can be implemented with TDD. Task 5 is interactive (training). Task 9 is deployment.
