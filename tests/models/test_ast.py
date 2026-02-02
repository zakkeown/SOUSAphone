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
