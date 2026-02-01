"""Tests for AudioClassificationModel base interface."""

import pytest
import torch
import torch.nn as nn

from sousa.models.base import AudioClassificationModel


class MockAudioModel(AudioClassificationModel):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.randn(audio.shape[0], 40)

    def get_feature_extractor(self) -> dict:
        return {"sample_rate": 16000, "max_duration": 5.0}

    @property
    def expected_input_type(self) -> str:
        return "waveform"


def test_cannot_instantiate_base_class():
    """Test that the abstract base class cannot be instantiated."""
    with pytest.raises(TypeError):
        AudioClassificationModel()


def test_mock_model_implements_interface():
    model = MockAudioModel()
    audio = torch.randn(2, 16000)
    logits = model(audio)
    assert logits.shape == (2, 40)

    feature_config = model.get_feature_extractor()
    assert "sample_rate" in feature_config

    assert model.expected_input_type in ["waveform", "spectrogram"]


def test_is_pytorch_module():
    model = MockAudioModel()
    assert isinstance(model, torch.nn.Module)
