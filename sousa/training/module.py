"""Lightning module for SOUSA training."""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.optim import Optimizer
from torchmetrics import Accuracy
from peft import LoraConfig, get_peft_model

from sousa.models.base import AudioClassificationModel


class SOUSAClassifier(pl.LightningModule):
    """
    Lightning module for rudiment classification training.

    Wraps any AudioClassificationModel and handles training,
    validation, and PEFT injection.
    """

    def __init__(self, model: AudioClassificationModel, config: DictConfig):
        """
        Initialize classifier.

        Args:
            model: Any model implementing AudioClassificationModel
            config: Hydra config with training/strategy params
        """
        super().__init__()
        self.model = model
        self.config = config

        # Apply PEFT if configured
        if hasattr(config, 'strategy') and config.strategy.type == "lora":
            # Log original trainable params
            original_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            # Create LoRA config
            peft_config = LoraConfig(
                r=config.strategy.rank,
                lora_alpha=config.strategy.alpha,
                lora_dropout=config.strategy.dropout,
                target_modules=list(config.strategy.target_modules),
                bias="none",
                task_type="FEATURE_EXTRACTION",  # Using FEATURE_EXTRACTION for audio models
            )

            # Apply LoRA
            self.model = get_peft_model(self.model, peft_config)

            # Log reduced trainable params
            peft_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"LoRA applied: {original_params:,} -> {peft_params:,} trainable params")
            print(f"Trainable params reduced to {100 * peft_params / original_params:.2f}%")

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])

        # Metrics
        num_classes = model.num_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(audio)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        audio, labels = batch['audio'], batch['label']
        logits = self(audio)

        # Loss with label smoothing
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.config.training.label_smoothing,
        )

        # Metrics
        self.log('train/loss', loss, prog_bar=True)
        self.train_acc(logits, labels)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        audio, labels = batch['audio'], batch['label']
        logits = self(audio)

        loss = F.cross_entropy(logits, labels)

        self.log('val/loss', loss, prog_bar=True)
        self.val_acc(logits, labels)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        audio, labels = batch['audio'], batch['label']
        logits = self(audio)

        loss = F.cross_entropy(logits, labels)

        self.log('test/loss', loss, prog_bar=True)
        self.test_acc(logits, labels)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        return optimizer
