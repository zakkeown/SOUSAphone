"""Feature Inference Model: predicts 12-dim onset features from raw onset detection output."""

from typing import Optional

import torch
import torch.nn as nn


class FeatureInferenceModel(nn.Module):
    """Transformer that predicts OnsetTransformer's 12-dim feature vectors
    from raw onset detection output (onset_time_ms, onset_strength, tempo_bpm).

    This bridges the gap between audio onset detection (which only provides
    timing and strength) and the OnsetTransformer's rich feature space
    (which includes stroke types, sticking, grace notes, etc.).

    Args:
        input_dim: Raw onset features per stroke (default 3: ioi_ms, strength, tempo_bpm)
        output_dim: Target features per stroke (default 12: OnsetTransformer feature space)
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Number of Transformer encoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length for positional encoding
    """

    # Indices of binary features in the 12-dim output
    BINARY_INDICES = [2, 3, 4, 5, 6, 10]  # is_grace, is_accent, is_tap, is_diddle, hand_R, is_buzz
    # Indices of continuous features in the 12-dim output
    CONTINUOUS_INDICES = [0, 1, 7, 8, 9, 11]  # norm_ioi, norm_velocity, diddle_pos, norm_flam_spacing, position_in_beat, norm_buzz_count

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, raw_onsets: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            raw_onsets: (batch, seq_len, input_dim) -- raw onset features per stroke
            attention_mask: (batch, seq_len) -- 1 for real tokens, 0 for padding

        Returns:
            (batch, seq_len, output_dim) -- predicted features per stroke
        """
        batch_size, seq_len, _ = raw_onsets.shape

        # Project features and add positional encoding
        positions = torch.arange(seq_len, device=raw_onsets.device)
        x = self.input_proj(raw_onsets) + self.pos_embedding(positions)

        # Transformer expects src_key_padding_mask where True = ignore
        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.output_proj(x)
