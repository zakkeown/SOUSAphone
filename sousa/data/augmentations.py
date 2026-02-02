"""Audio augmentation techniques."""
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """SpecAugment augmentation for spectrograms.

    Implements frequency and time masking.
    """

    def __init__(
        self,
        freq_mask_param: int = 30,
        time_mask_param: int = 40,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment.

        Args:
            spec: Spectrogram (batch, time, freq) or (time, freq)

        Returns:
            Augmented spectrogram
        """
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size, time_steps, freq_bins = spec.shape
        spec = spec.clone()

        # Apply frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, freq_bins - f)
            spec[:, :, f0:f0+f] = 0

        # Apply time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, time_steps))
            t0 = random.randint(0, time_steps - t)
            spec[:, t0:t0+t, :] = 0

        if squeeze:
            spec = spec.squeeze(0)

        return spec


class Mixup:
    """Mixup augmentation for audio/spectrograms.

    Mixes two samples and their labels.
    """

    def __init__(self, alpha: float = 0.2, num_classes: int = 40):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply Mixup to a batch.

        Args:
            batch: Dict with 'audio' and 'label' keys

        Returns:
            Mixed batch dict
        """
        audio = batch['audio']
        labels = batch['label']

        batch_size = audio.size(0)

        # Sample mixing coefficients
        if self.alpha > 0:
            lam = torch.from_numpy(
                np.random.beta(self.alpha, self.alpha, batch_size)
            ).float().to(audio.device)
        else:
            lam = torch.ones(batch_size).to(audio.device)

        # Shuffle batch
        indices = torch.randperm(batch_size).to(audio.device)

        # Mix audio
        lam = lam.view(-1, *([1] * (audio.dim() - 1)))
        mixed_audio = lam * audio + (1 - lam) * audio[indices]

        # Mix labels (one-hot encoding)
        labels_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        lam_labels = lam.squeeze()
        mixed_labels = lam_labels.unsqueeze(1) * labels_one_hot + \
                       (1 - lam_labels).unsqueeze(1) * labels_one_hot[indices]

        return {
            'audio': mixed_audio,
            'label': mixed_labels,  # Soft labels
            'original_label': labels,  # Keep for reference
        }
