"""Onset Transformer model for stroke-level rudiment classification."""

from typing import Optional

import torch
import torch.nn as nn
from sousa.models.base import AudioClassificationModel


class OnsetTransformerModel(AudioClassificationModel):
    """
    Small Transformer encoder that classifies rudiments from per-stroke
    onset features (IOI, velocity, grace notes, diddles, sticking, etc.).

    ~111K parameters. Trains from scratch — no pretraining needed.
    """

    def __init__(
        self,
        num_classes: int = 40,
        pretrained: bool = False,
        feature_dim: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Project per-stroke features to model dimension
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, onset_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            onset_features: (batch, seq_len, feature_dim)
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len, _ = onset_features.shape

        # Project features and add positional encoding
        positions = torch.arange(seq_len, device=onset_features.device)
        x = self.input_proj(onset_features) + self.pos_embedding(positions)

        # Transformer expects src_key_padding_mask where True = ignore
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)  # True where padded

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Mean pooling over non-padded positions
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)  # (batch, seq, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        x = self.layer_norm(x)
        logits: torch.Tensor = self.classifier(x)
        return logits

    def get_feature_extractor(self) -> dict:
        return {}

    @property
    def expected_input_type(self) -> str:
        return "onset"
