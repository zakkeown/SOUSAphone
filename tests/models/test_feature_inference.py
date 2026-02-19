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
    raw_onsets = torch.randn(2, 32, 3)
    mask = torch.ones(2, 32)
    output = model(raw_onsets, attention_mask=mask)
    assert output.shape == (2, 32, 12)


def test_forward_with_padding():
    """Model should handle padded sequences correctly."""
    model = FeatureInferenceModel()
    raw_onsets = torch.randn(2, 64, 3)
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


def test_exceeds_max_seq_len():
    """Model should raise ValueError when sequence exceeds max_seq_len."""
    model = FeatureInferenceModel(max_seq_len=32)
    raw_onsets = torch.randn(1, 64, 3)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(raw_onsets)
