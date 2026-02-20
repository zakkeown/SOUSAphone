"""Lightning module for Feature Inference Model training."""

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Optimizer

from sousa.models.feature_inference import FeatureInferenceModel


class FeatureInferenceModule(pl.LightningModule):
    """Lightning module for training the Feature Inference Model.

    Uses mixed loss: BCE for binary features, MSE for continuous features.
    Only computes loss over non-padded positions (respects attention_mask).
    """

    def __init__(self, model: FeatureInferenceModel, config: DictConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=["model"])

    def forward(
        self,
        raw_onsets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        result: torch.Tensor = self.model(raw_onsets, attention_mask=attention_mask)
        return result

    def _compute_loss(
        self,
        raw_onsets: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mixed BCE + MSE loss, respecting padding mask.

        Args:
            raw_onsets: (batch, seq_len, input_dim) raw onset features.
            target: (batch, seq_len, output_dim) ground-truth feature vectors.
            mask: (batch, seq_len) attention mask (1 = real, 0 = padding).

        Returns:
            Scalar loss tensor (bce_loss + mse_loss).
        """
        pred = self.model(raw_onsets, attention_mask=mask)

        # Expand mask from (batch, seq_len) to (batch, seq_len, 1) for broadcasting
        mask_expanded = mask.unsqueeze(-1)

        # Binary features: BCE with logits
        binary_idx = FeatureInferenceModel.BINARY_INDICES
        bce_loss = F.binary_cross_entropy_with_logits(
            pred[:, :, binary_idx], target[:, :, binary_idx], reduction="none"
        )
        bce_loss = (bce_loss * mask_expanded).sum() / mask.sum().clamp(min=1)

        # Continuous features: MSE
        cont_idx = FeatureInferenceModel.CONTINUOUS_INDICES
        mse_loss = F.mse_loss(
            pred[:, :, cont_idx], target[:, :, cont_idx], reduction="none"
        )
        mse_loss = (mse_loss * mask_expanded).sum() / mask.sum().clamp(min=1)

        total: torch.Tensor = bce_loss + mse_loss
        return total

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        loss = self._compute_loss(
            batch["raw_onsets"], batch["target_features"], batch["attention_mask"]
        )
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        loss = self._compute_loss(
            batch["raw_onsets"], batch["target_features"], batch["attention_mask"]
        )
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        """Configure AdamW optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        return optimizer
