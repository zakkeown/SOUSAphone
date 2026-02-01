# sousa/models/beats.py
"""BEATs (Bidirectional Encoder representation from Audio Transformers) model adapter."""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from sousa.models.base import AudioClassificationModel


class BEATsModel(AudioClassificationModel):
    """
    BEATs model from HuggingFace transformers.

    Uses Microsoft's WavLM (similar to BEATs architecture) pretrained on
    audio data, with head replaced for 40-class rudiment classification.

    Note: BEATs uses raw waveform input, not spectrograms.
    """

    def __init__(self, num_classes: int = 40, pretrained: bool = True):
        """
        Initialize BEATs model.

        Args:
            num_classes: Number of output classes (40 for rudiments)
            pretrained: Load WavLM pretrained weights
        """
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            # Use WavLM-base-plus (similar to BEATs architecture)
            self.encoder = Wav2Vec2Model.from_pretrained("microsoft/wavlm-base-plus")
            hidden_size = self.encoder.config.hidden_size
        else:
            # Random initialization for testing
            config = Wav2Vec2Config()
            self.encoder = Wav2Vec2Model(config)
            hidden_size = config.hidden_size

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio: Raw waveform (batch_size, num_samples)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Encode the waveform
        outputs = self.encoder(audio)

        # Mean pooling over time dimension
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        pooled_output = outputs.last_hidden_state.mean(dim=1)

        # Apply classification head
        logits = self.classifier(pooled_output)

        return logits

    def get_feature_extractor(self) -> dict:
        """Get BEATs preprocessing config."""
        return {
            "sample_rate": 16000,
            "max_duration": 5.0,  # seconds
        }

    @property
    def expected_input_type(self) -> str:
        return "waveform"
