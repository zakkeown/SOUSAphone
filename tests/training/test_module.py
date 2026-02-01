# tests/training/test_module.py
import pytest
import torch
from omegaconf import OmegaConf
from sousa.training.module import SOUSAClassifier
from sousa.models.ast import ASTModel
from peft import PeftModel


@pytest.fixture
def minimal_config():
    """Minimal config for testing"""
    return OmegaConf.create({
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
        },
        "strategy": {
            "type": "full_finetune",
        },
    })


def test_classifier_initializes(minimal_config):
    """SOUSAClassifier should initialize with model"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    assert classifier is not None


def test_classifier_forward(minimal_config):
    """SOUSAClassifier forward should return logits"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    # Dummy batch
    audio = torch.randn(2, 1024, 128)
    logits = classifier(audio)

    assert logits.shape == (2, 40)


def test_classifier_training_step(minimal_config):
    """Training step should compute loss"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    # Dummy batch
    batch = {
        'audio': torch.randn(2, 1024, 128),
        'label': torch.tensor([0, 5]),
    }

    loss = classifier.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar


def test_classifier_validation_step(minimal_config):
    """Validation step should compute loss"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    batch = {
        'audio': torch.randn(2, 1024, 128),
        'label': torch.tensor([0, 5]),
    }

    loss = classifier.validation_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_classifier_configure_optimizers(minimal_config):
    """Should configure AdamW optimizer"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    optimizer = classifier.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == 1e-4
    assert optimizer.defaults['weight_decay'] == 0.01


@pytest.fixture
def lora_config():
    """Config with LoRA strategy"""
    return OmegaConf.create({
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
        },
        "strategy": {
            "type": "lora",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": ["attention.query", "attention.key", "attention.value"],
        },
    })


def test_lora_is_applied(lora_config):
    """LoRA should be applied when strategy.type == 'lora'"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, lora_config)

    # Model should be wrapped in PeftModel
    assert isinstance(classifier.model, PeftModel)


def test_lora_reduces_trainable_params(lora_config):
    """LoRA should significantly reduce trainable parameters"""
    # Create two classifiers: one with full finetune, one with LoRA
    full_finetune_config = OmegaConf.create({
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
        },
        "strategy": {
            "type": "full_finetune",
        },
    })

    model_full = ASTModel(num_classes=40, pretrained=False)
    classifier_full = SOUSAClassifier(model_full, full_finetune_config)

    model_lora = ASTModel(num_classes=40, pretrained=False)
    classifier_lora = SOUSAClassifier(model_lora, lora_config)

    # Count trainable params
    full_params = sum(p.numel() for p in classifier_full.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for p in classifier_lora.parameters() if p.requires_grad)

    # LoRA should have significantly fewer trainable params
    assert lora_params < full_params
    # Should be less than 10% of original
    assert lora_params < 0.1 * full_params


def test_non_lora_config_still_works(minimal_config):
    """Non-LoRA configs should still work (backward compatibility)"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    # Should NOT be a PeftModel
    assert not isinstance(classifier.model, PeftModel)

    # Should still be able to forward pass
    audio = torch.randn(2, 1024, 128)
    logits = classifier(audio)
    assert logits.shape == (2, 40)


@pytest.fixture
def adalora_config():
    """Config with AdaLoRA strategy"""
    return OmegaConf.create({
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
        },
        "strategy": {
            "type": "adalora",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": ["attention.query", "attention.key", "attention.value"],
            "init_r": 12,
            "target_r": 8,
            "tinit": 0,
            "tfinal": 1000,
            "deltaT": 10,
            "total_step": 10000,
        },
    })


def test_adalora_is_applied(adalora_config):
    """AdaLoRA should be applied when strategy.type == 'adalora'"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, adalora_config)

    # Model should be wrapped in PeftModel
    assert isinstance(classifier.model, PeftModel)


def test_adalora_reduces_trainable_params(adalora_config):
    """AdaLoRA should significantly reduce trainable parameters"""
    # Create two classifiers: one with full finetune, one with AdaLoRA
    full_finetune_config = OmegaConf.create({
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
        },
        "strategy": {
            "type": "full_finetune",
        },
    })

    model_full = ASTModel(num_classes=40, pretrained=False)
    classifier_full = SOUSAClassifier(model_full, full_finetune_config)

    model_adalora = ASTModel(num_classes=40, pretrained=False)
    classifier_adalora = SOUSAClassifier(model_adalora, adalora_config)

    # Count trainable params
    full_params = sum(p.numel() for p in classifier_full.parameters() if p.requires_grad)
    adalora_params = sum(p.numel() for p in classifier_adalora.parameters() if p.requires_grad)

    # AdaLoRA should have significantly fewer trainable params
    assert adalora_params < full_params
    # Should be less than 10% of original
    assert adalora_params < 0.1 * full_params


@pytest.fixture
def ia3_config():
    """Config with IA3 strategy"""
    return OmegaConf.create({
        "training": {
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
        },
        "strategy": {
            "type": "ia3",
            "target_modules": ["attention.query", "attention.key", "attention.value", "feed_forward.dense"],
        },
    })


def test_ia3_is_applied(ia3_config):
    """IA3 should be applied when strategy.type == 'ia3'"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, ia3_config)

    # Model should be wrapped in PeftModel
    assert isinstance(classifier.model, PeftModel)


def test_ia3_reduces_trainable_params(ia3_config):
    """IA3 should significantly reduce trainable parameters"""
    # Create two classifiers: one with full finetune, one with IA3
    full_finetune_config = OmegaConf.create({
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
        },
        "strategy": {
            "type": "full_finetune",
        },
    })

    model_full = ASTModel(num_classes=40, pretrained=False)
    classifier_full = SOUSAClassifier(model_full, full_finetune_config)

    model_ia3 = ASTModel(num_classes=40, pretrained=False)
    classifier_ia3 = SOUSAClassifier(model_ia3, ia3_config)

    # Count trainable params
    full_params = sum(p.numel() for p in classifier_full.parameters() if p.requires_grad)
    ia3_params = sum(p.numel() for p in classifier_ia3.parameters() if p.requires_grad)

    # IA3 should have significantly fewer trainable params
    assert ia3_params < full_params
    # IA3 is even more efficient than LoRA, should be less than 5% of original
    assert ia3_params < 0.05 * full_params
