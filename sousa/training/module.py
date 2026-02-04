"""Lightning module for SOUSA training."""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.optim import Optimizer
from torchmetrics import Accuracy, ConfusionMatrix, F1Score
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import seaborn as sns

from sousa.models.base import AudioClassificationModel
from sousa.data.augmentations import Mixup


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

            # Get model-specific target modules
            target_modules = list(config.model.peft_target_modules)

            # Get modules_to_save (classifier head must remain trainable)
            modules_to_save = None
            if hasattr(config.model, 'peft_modules_to_save'):
                modules_to_save = list(config.model.peft_modules_to_save)

            peft_config = LoraConfig(
                r=config.strategy.rank,
                lora_alpha=config.strategy.alpha,
                lora_dropout=config.strategy.dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
                bias="none",
            )

            # Apply PEFT
            self.model = get_peft_model(self.model, peft_config)

            # Log reduced trainable params
            peft_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"{config.strategy.type.upper()} applied: {original_params:,} -> {peft_params:,} trainable params")
            print(f"Trainable params reduced to {100 * peft_params / original_params:.2f}%")
            if modules_to_save:
                print(f"Modules kept fully trainable: {modules_to_save}")

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])

        # Metrics
        num_classes = model.num_classes

        # Initialize Mixup if configured (after num_classes is set)
        if hasattr(config, 'augmentation') and config.augmentation.mixup:
            self.mixup = Mixup(alpha=config.augmentation.mixup_alpha, num_classes=num_classes)
        else:
            self.mixup = None
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Advanced validation metrics
        self.val_f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",  # Macro-averaged F1
        )
        self.val_f1_per_class = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="none",  # Per-class F1
        )
        self.val_confusion = ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
        )

        # Advanced test metrics
        self.test_f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )
        self.test_f1_per_class = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="none",
        )
        self.test_confusion = ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model.

        With task_type=None, PEFT uses the base PeftModel wrapper which
        accepts *args and passes them through to our model's forward().
        """
        return self.model(audio)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Apply Mixup
        if self.mixup and self.training:
            batch = self.mixup(batch)

        audio = batch['audio']
        labels = batch['label']
        logits = self(audio)

        # Handle soft labels from Mixup
        if labels.dim() > 1:  # Soft labels
            loss = -(labels * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            # For accuracy, use original labels if available
            if 'original_label' in batch:
                self.train_acc(logits, batch['original_label'])
        else:  # Hard labels
            loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=self.config.training.label_smoothing,
            )
            self.train_acc(logits, labels)

        # Metrics
        self.log('train/loss', loss, prog_bar=True)
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

        # Update advanced metrics
        self.val_f1(logits, labels)
        self.val_f1_per_class(logits, labels)
        self.val_confusion(logits, labels)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        audio, labels = batch['audio'], batch['label']
        logits = self(audio)

        loss = F.cross_entropy(logits, labels)

        self.log('test/loss', loss, prog_bar=True)
        self.test_acc(logits, labels)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)

        # Update advanced metrics
        self.test_f1(logits, labels)
        self.test_f1_per_class(logits, labels)
        self.test_confusion(logits, labels)

        return loss

    def on_validation_epoch_end(self):
        """Log advanced metrics at end of validation epoch."""
        # Log macro F1
        f1 = self.val_f1.compute()
        self.log('val/f1_macro', f1, sync_dist=True)

        # Log per-class F1 (get top-5 worst and best)
        f1_per_class = self.val_f1_per_class.compute()

        # Get rudiment names
        from sousa.utils.rudiments import RUDIMENT_NAMES

        # Log worst 5 classes
        worst_indices = f1_per_class.argsort()[:5]
        for idx in worst_indices:
            self.log(f'val/f1_worst/{RUDIMENT_NAMES[idx]}', f1_per_class[idx])

        # Log best 5 classes
        best_indices = f1_per_class.argsort()[-5:]
        for idx in best_indices:
            self.log(f'val/f1_best/{RUDIMENT_NAMES[idx]}', f1_per_class[idx])

        # Create confusion matrix figure
        cm = self.val_confusion.compute()
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(
            cm.cpu().numpy(),
            annot=False,
            fmt='d',
            cmap='Blues',
            xticklabels=RUDIMENT_NAMES,
            yticklabels=RUDIMENT_NAMES,
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Log to W&B
        import wandb
        self.logger.experiment.log({
            "val/confusion_matrix": wandb.Image(fig),
            "epoch": self.current_epoch
        })
        plt.close(fig)

        # Reset metrics
        self.val_f1.reset()
        self.val_f1_per_class.reset()
        self.val_confusion.reset()

    def on_test_epoch_end(self):
        """Log advanced metrics at end of test epoch."""
        # Log macro F1
        f1 = self.test_f1.compute()
        self.log('test/f1_macro', f1)

        # Log per-class F1 (get top-5 worst and best)
        f1_per_class = self.test_f1_per_class.compute()

        # Get rudiment names
        from sousa.utils.rudiments import RUDIMENT_NAMES

        # Log worst 5 classes
        worst_indices = f1_per_class.argsort()[:5]
        for idx in worst_indices:
            self.log(f'test/f1_worst/{RUDIMENT_NAMES[idx]}', f1_per_class[idx])

        # Log best 5 classes
        best_indices = f1_per_class.argsort()[-5:]
        for idx in best_indices:
            self.log(f'test/f1_best/{RUDIMENT_NAMES[idx]}', f1_per_class[idx])

        # Create confusion matrix figure
        cm = self.test_confusion.compute()
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(
            cm.cpu().numpy(),
            annot=False,
            fmt='d',
            cmap='Blues',
            xticklabels=RUDIMENT_NAMES,
            yticklabels=RUDIMENT_NAMES,
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Log to W&B
        import wandb
        self.logger.experiment.log({
            "test/confusion_matrix": wandb.Image(fig),
            "epoch": self.current_epoch
        })
        plt.close(fig)

        # Reset metrics
        self.test_f1.reset()
        self.test_f1_per_class.reset()
        self.test_confusion.reset()

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizer and scheduler."""
        # Use strategy-specific learning rate when PEFT is active
        lr = self.config.training.learning_rate
        if hasattr(self.config, 'strategy') and hasattr(self.config.strategy, 'learning_rate'):
            lr = self.config.strategy.learning_rate
            print(f"Using strategy learning rate: {lr}")

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=self.config.training.weight_decay,
        )
        return optimizer
