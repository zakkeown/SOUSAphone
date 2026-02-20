"""Tests for OnsetTransformerModel config and round-trip."""

import json
import tempfile
from pathlib import Path

import torch

from sousa.models.onset_transformer import OnsetTransformerModel


def test_default_constructor_matches_production():
    """Default constructor should produce a model compatible with shipped weights."""
    model = OnsetTransformerModel()
    assert model.num_classes == 40
    config = model.get_config()
    assert config["feature_dim"] == 12
    assert config["max_seq_len"] == 256


def test_get_config_keys():
    """get_config should return all architecture keys."""
    model = OnsetTransformerModel()
    config = model.get_config()
    expected_keys = {
        "num_classes", "feature_dim", "d_model", "nhead",
        "num_layers", "dim_feedforward", "dropout", "max_seq_len",
    }
    assert set(config.keys()) == expected_keys


def test_from_config_round_trip():
    """from_config(get_config()) should produce an identical architecture."""
    original = OnsetTransformerModel(num_classes=20, feature_dim=8, d_model=32,
                                     nhead=2, num_layers=2, dim_feedforward=64,
                                     dropout=0.2, max_seq_len=64)
    config = original.get_config()
    rebuilt = OnsetTransformerModel.from_config(config)

    assert rebuilt.get_config() == config

    # Same output shape
    x = torch.randn(1, 16, 8)
    assert original(x).shape == rebuilt(x).shape


def test_save_load_weights_round_trip():
    """Save and load state_dict through from_config should produce same output."""
    model = OnsetTransformerModel()
    model.eval()

    x = torch.randn(1, 32, 12)
    original_out = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / "model.bin"
        torch.save(model.state_dict(), weights_path)

        loaded = OnsetTransformerModel.from_config(model.get_config())
        loaded.load_state_dict(torch.load(weights_path, weights_only=True))
        loaded.eval()

        loaded_out = loaded(x)
        assert torch.allclose(original_out, loaded_out)


def test_config_json_serialization():
    """Config should be JSON-serializable and round-trip through JSON."""
    model = OnsetTransformerModel()
    config = model.get_config()
    json_str = json.dumps(config)
    restored = json.loads(json_str)
    assert restored == config

    rebuilt = OnsetTransformerModel.from_config(restored)
    assert rebuilt.get_config() == config


def test_forward_shape():
    """Forward pass should return (batch, num_classes) logits."""
    model = OnsetTransformerModel()
    x = torch.randn(2, 32, 12)
    mask = torch.ones(2, 32)
    out = model(x, attention_mask=mask)
    assert out.shape == (2, 40)
