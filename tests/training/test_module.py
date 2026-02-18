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
        "model": {
            "peft_target_modules": ["query", "key", "value"],
            "peft_modules_to_save": ["model.classifier"],
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


def test_f1_metrics_initialized(minimal_config):
    """F1 metrics should be initialized"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    # Check validation metrics exist
    assert hasattr(classifier, 'val_f1')
    assert hasattr(classifier, 'val_f1_per_class')

    # Check test metrics exist
    assert hasattr(classifier, 'test_f1')
    assert hasattr(classifier, 'test_f1_per_class')


def test_confusion_matrix_initialized(minimal_config):
    """Confusion matrix metrics should be initialized"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    # Check validation metrics exist
    assert hasattr(classifier, 'val_confusion')

    # Check test metrics exist
    assert hasattr(classifier, 'test_confusion')


def test_metrics_update_on_validation_step(minimal_config):
    """Advanced metrics should update during validation step"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    # Create batch
    batch = {
        'audio': torch.randn(4, 1024, 128),
        'label': torch.tensor([0, 5, 10, 15]),
    }

    # Run validation step
    classifier.validation_step(batch, batch_idx=0)

    # Metrics should have been updated (not computed yet, but internal state changed)
    # We can't directly check internal state, but we can compute to verify they work
    f1 = classifier.val_f1.compute()
    f1_per_class = classifier.val_f1_per_class.compute()
    cm = classifier.val_confusion.compute()

    # Check shapes and types
    assert isinstance(f1, torch.Tensor)
    assert f1.ndim == 0  # Scalar
    assert isinstance(f1_per_class, torch.Tensor)
    assert f1_per_class.shape == (40,)  # Per-class F1
    assert isinstance(cm, torch.Tensor)
    assert cm.shape == (40, 40)  # Confusion matrix


def test_metrics_update_on_test_step(minimal_config):
    """Advanced metrics should update during test step"""
    model = ASTModel(num_classes=40, pretrained=False)
    classifier = SOUSAClassifier(model, minimal_config)

    # Create batch
    batch = {
        'audio': torch.randn(4, 1024, 128),
        'label': torch.tensor([0, 5, 10, 15]),
    }

    # Run test step
    classifier.test_step(batch, batch_idx=0)

    # Compute metrics to verify they work
    f1 = classifier.test_f1.compute()
    f1_per_class = classifier.test_f1_per_class.compute()
    cm = classifier.test_confusion.compute()

    # Check shapes and types
    assert isinstance(f1, torch.Tensor)
    assert f1.ndim == 0  # Scalar
    assert isinstance(f1_per_class, torch.Tensor)
    assert f1_per_class.shape == (40,)  # Per-class F1
    assert isinstance(cm, torch.Tensor)
    assert cm.shape == (40, 40)  # Confusion matrix
