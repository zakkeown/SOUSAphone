"""Tests for FeatureInferenceModel."""

import json
import tempfile
from pathlib import Path

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


def test_get_config_keys():
    """get_config should return all architecture keys."""
    model = FeatureInferenceModel()
    config = model.get_config()
    expected_keys = {
        "input_dim", "output_dim", "d_model", "nhead",
        "num_layers", "dim_feedforward", "dropout", "max_seq_len",
    }
    assert set(config.keys()) == expected_keys


def test_from_config_round_trip():
    """from_config(get_config()) should produce an identical architecture."""
    original = FeatureInferenceModel(input_dim=5, output_dim=8, d_model=32,
                                      nhead=2, num_layers=2, dim_feedforward=64,
                                      dropout=0.2, max_seq_len=64)
    config = original.get_config()
    rebuilt = FeatureInferenceModel.from_config(config)
    assert rebuilt.get_config() == config

    x = torch.randn(1, 16, 5)
    assert original(x).shape == rebuilt(x).shape


def test_save_load_weights_round_trip():
    """Save and load state_dict through from_config should produce same output."""
    model = FeatureInferenceModel()
    model.eval()

    x = torch.randn(1, 32, 3)
    original_out = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / "model.bin"
        torch.save(model.state_dict(), weights_path)

        loaded = FeatureInferenceModel.from_config(model.get_config())
        loaded.load_state_dict(torch.load(weights_path, weights_only=True))
        loaded.eval()

        loaded_out = loaded(x)
        assert torch.allclose(original_out, loaded_out)


def test_config_json_serialization():
    """Config should be JSON-serializable and round-trip through JSON."""
    model = FeatureInferenceModel()
    config = model.get_config()
    json_str = json.dumps(config)
    restored = json.loads(json_str)
    assert restored == config

    rebuilt = FeatureInferenceModel.from_config(restored)
    assert rebuilt.get_config() == config
