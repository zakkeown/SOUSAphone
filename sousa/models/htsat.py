# sousa/models/htsat.py
"""HTS-AT (Hierarchical Token-Semantic Audio Transformer) model adapter."""

import torch
import torch.nn as nn
from transformers import ClapModel, ClapConfig
from sousa.models.base import AudioClassificationModel


class HTSATModel(AudioClassificationModel):
    """
    HTS-AT model from HuggingFace transformers via CLAP.

    Uses LAION's pretrained CLAP model with HTS-AT audio encoder,
    with head replaced for 40-class rudiment classification.
    """

    def __init__(self, num_classes: int = 40, pretrained: bool = True):
        """
        Initialize HTS-AT model.

        Args:
            num_classes: Number of output classes (40 for rudiments)
            pretrained: Load CLAP pretrained weights
        """
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            # Load CLAP model with HTS-AT audio encoder
            # Note: Pretrained model uses 64 mel bins, we'll need to adapt
            clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            self.audio_encoder = clap_model.audio_model.audio_encoder
            # Get the hidden size from the config
            self.hidden_size = clap_model.config.audio_config.hidden_size
            self.num_mel_bins = clap_model.config.audio_config.num_mel_bins
        else:
            # Random initialization for testing (64 mel bins matches pretrained)
            from transformers import ClapAudioConfig
            audio_config = ClapAudioConfig(num_mel_bins=64)
            config = ClapConfig(audio_config=audio_config)
            clap_model = ClapModel(config)
            self.audio_encoder = clap_model.audio_model.audio_encoder
            self.hidden_size = config.audio_config.hidden_size
            self.num_mel_bins = config.audio_config.num_mel_bins

        # Add classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio: Mel-spectrogram (batch_size, time, n_mels)

        Returns:
            logits: (batch_size, num_classes)
        """
        # HTS-AT expects (batch, 1, time, n_mels) for Conv2d
        if audio.dim() == 3:
            audio = audio.unsqueeze(1)  # Add channel dimension

        # Get audio encoder outputs
        encoder_outputs = self.audio_encoder(audio)

        # Use the pooled output from the encoder
        # pooler_output has shape (batch_size, hidden_size)
        pooled_output = encoder_outputs.pooler_output

        # Apply classification head
        logits = self.classifier(pooled_output)

        return logits

    def get_feature_extractor(self) -> dict:
        """Get HTS-AT preprocessing config."""
        return {
            "sample_rate": 16000,
            "n_mels": self.num_mel_bins,
            "n_fft": 400,
            "hop_length": 160,
            "max_length": 256,  # Time frames (CLAP spec_size)
        }

    @property
    def expected_input_type(self) -> str:
        return "spectrogram"
