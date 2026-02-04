"""Base interface for audio classification models.

This module defines the abstract base class that all audio classification models
must implement to ensure compatibility with the SOUSA training pipeline.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AudioClassificationModel(nn.Module, ABC):
    """Abstract base class for all audio classification models.

    Model adapters must inherit from this class and implement the required
    abstract methods.
    """

    @abstractmethod
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            audio: Input audio tensor. Shape depends on expected_input_type:
                - "waveform": (batch_size, num_samples)
                - "spectrogram": (batch_size, time_steps, n_mels)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        pass

    @abstractmethod
    def get_feature_extractor(self) -> dict:
        """Get the feature extraction configuration for this model.

        Returns:
            Dictionary containing preprocessing parameters like:
                - sample_rate: Expected audio sample rate
                - max_duration: Maximum audio duration in seconds
                - n_mels: Number of mel bins (if using spectrograms)
                - etc.
        """
        pass

    @property
    @abstractmethod
    def expected_input_type(self) -> str:
        """Get the expected input type for this model.

        Returns:
            Either "waveform" or "spectrogram"
        """
        pass
