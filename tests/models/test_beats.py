# tests/models/test_beats.py
import pytest
import torch
from sousa.models.beats import BEATsModel


def test_beats_initializes():
    """BEATs model should initialize from HuggingFace"""
    model = BEATsModel(num_classes=40, pretrained=True)
    assert model is not None


def test_beats_forward_pass():
    """BEATs should produce logits for batch"""
    model = BEATsModel(num_classes=40, pretrained=False)
    # BEATs expects waveforms: (batch, num_samples)
    batch_size = 2
    sample_rate = 16000
    duration = 5.0  # seconds
    num_samples = int(sample_rate * duration)
    waveform = torch.randn(batch_size, num_samples)  # (batch, samples)

    logits = model(waveform)
    assert logits.shape == (batch_size, 40)


def test_beats_expected_input_type():
    """BEATs expects waveform input"""
    model = BEATsModel(num_classes=40, pretrained=False)
    assert model.expected_input_type == "waveform"


def test_beats_feature_extractor():
    """BEATs should provide feature extraction config"""
    model = BEATsModel(num_classes=40, pretrained=False)
    config = model.get_feature_extractor()

    assert config['sample_rate'] == 16000
    assert config['max_duration'] == 5.0
