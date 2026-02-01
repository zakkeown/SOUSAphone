# tests/models/test_efficientat.py
import pytest
import torch
from sousa.models.efficientat import EfficientATModel


def test_efficientat_initializes():
    """EfficientAT model should initialize"""
    model = EfficientATModel(num_classes=40, pretrained=True)
    assert model is not None


def test_efficientat_forward_pass():
    """EfficientAT should produce logits for batch"""
    model = EfficientATModel(num_classes=40, pretrained=False)
    # EfficientAT expects mel-spectrograms: (batch, time, n_mels)
    batch_size = 2
    spec = torch.randn(batch_size, 1024, 128)  # (batch, time, mels)

    logits = model(spec)
    assert logits.shape == (batch_size, 40)


def test_efficientat_expected_input_type():
    """EfficientAT expects spectrogram input"""
    model = EfficientATModel(num_classes=40, pretrained=False)
    assert model.expected_input_type == "spectrogram"


def test_efficientat_feature_extractor():
    """EfficientAT should provide feature extraction config"""
    model = EfficientATModel(num_classes=40, pretrained=False)
    config = model.get_feature_extractor()

    assert config['sample_rate'] == 16000
    assert config['n_mels'] == 128
