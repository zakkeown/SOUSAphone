# sousa/models/ast.py
"""AST (Audio Spectrogram Transformer) model adapter."""

import torch
from transformers import ASTForAudioClassification
from sousa.models.base import AudioClassificationModel


class ASTModel(AudioClassificationModel):
    """
    AST model from HuggingFace transformers.

    Uses MIT's pretrained AST on AudioSet, with head replaced
    for 40-class rudiment classification.
    """

    def __init__(self, num_classes: int = 40, pretrained: bool = True):
        """
        Initialize AST model.

        Args:
            num_classes: Number of output classes (40 for rudiments)
            pretrained: Load AudioSet pretrained weights
        """
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            self.model = ASTForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,  # Replace classification head
            )
        else:
            # Random initialization for testing
            from transformers import ASTConfig
            config = ASTConfig(num_labels=num_classes)
            self.model = ASTForAudioClassification(config)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio: Mel-spectrogram (batch_size, time, n_mels)

        Returns:
            logits: (batch_size, num_classes)
        """
        outputs = self.model(audio)
        logits: torch.Tensor = outputs.logits
        return logits

    def get_feature_extractor(self) -> dict:
        """Get AST preprocessing config."""
        return {
            "sample_rate": 16000,
            "n_mels": 128,
            "n_fft": 400,
            "hop_length": 160,
            "max_length": 1024,  # Time frames
        }

    @property
    def expected_input_type(self) -> str:
        return "spectrogram"
