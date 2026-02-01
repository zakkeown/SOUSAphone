# tests/models/test_htsat.py
import pytest
import torch
from sousa.models.htsat import HTSATModel


def test_htsat_initializes():
    """HTS-AT model should initialize from HuggingFace"""
    model = HTSATModel(num_classes=40, pretrained=True)
    assert model is not None


def test_htsat_forward_pass():
    """HTS-AT should produce logits for batch"""
    model = HTSATModel(num_classes=40, pretrained=False)
    # HTS-AT expects mel-spectrograms: (batch, time, n_mels)
    # Note: CLAP/HTS-AT uses spec_size=256 for time dimension
    batch_size = 2
    spec = torch.randn(batch_size, 256, 128)  # (batch, time, mels)

    logits = model(spec)
    assert logits.shape == (batch_size, 40)


def test_htsat_expected_input_type():
    """HTS-AT expects spectrogram input"""
    model = HTSATModel(num_classes=40, pretrained=False)
    assert model.expected_input_type == "spectrogram"


def test_htsat_feature_extractor():
    """HTS-AT should provide feature extraction config"""
    model = HTSATModel(num_classes=40, pretrained=False)
    config = model.get_feature_extractor()

    assert config['sample_rate'] == 16000
    assert config['n_mels'] == 128
