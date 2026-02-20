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
    model = FeatureInferenceModel()
    module = FeatureInferenceModule(model, fi_config)
    raw = torch.randn(2, 32, 3)
    target = torch.randn(2, 32, 12)
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


def test_forward_returns_predictions(fi_config):
    """Forward pass should return predictions with correct shape."""
    model = FeatureInferenceModel()
    module = FeatureInferenceModule(model, fi_config)
    raw_onsets = torch.randn(2, 16, 3)
    mask = torch.ones(2, 16)
    pred = module(raw_onsets, attention_mask=mask)
    assert pred.shape == (2, 16, 12)


def test_loss_is_non_negative(fi_config):
    """BCE + MSE loss should always be non-negative."""
    model = FeatureInferenceModel()
    module = FeatureInferenceModule(model, fi_config)
    raw = torch.randn(4, 32, 3)
    target = torch.rand(4, 32, 12)  # Use rand for valid BCE targets in [0,1]
    mask = torch.ones(4, 32)
    loss = module._compute_loss(raw, target, mask)
    assert loss.item() >= 0


def test_optimizer_lr_matches_config(fi_config):
    """Optimizer LR should match config."""
    model = FeatureInferenceModel()
    module = FeatureInferenceModule(model, fi_config)
    optimizer = module.configure_optimizers()
    assert optimizer.defaults["lr"] == 1e-3
    assert optimizer.defaults["weight_decay"] == 0.01
