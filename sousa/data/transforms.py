"""Audio transforms for preprocessing."""

from typing import Callable, List

import torch
import torchaudio


# AST's official normalization statistics from ASTFeatureExtractor
# These match the pretrained model's expected input distribution
AST_NORM_MEAN = -4.2677393
AST_NORM_STD = 4.5689974


class MelSpectrogramTransform:
    """Convert waveform to mel-spectrogram for models."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,  # Changed from 400 to avoid zero filterbanks
        hop_length: int = 160,
        target_length: int = 1024,
        normalize: bool = True,
        norm_mean: float = AST_NORM_MEAN,
        norm_std: float = AST_NORM_STD,
    ):
        """
        Initialize mel-spectrogram transform.

        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel filterbanks
            n_fft: FFT window size (use 1024+ to avoid zero filterbanks with 128 mels)
            hop_length: Number of samples between successive frames
            target_length: Target number of time frames (for AST: 1024)
            normalize: Whether to normalize the spectrogram
            norm_mean: Mean for normalization (AST default: -4.2677)
            norm_std: Std for normalization (AST default: 4.5690)
        """
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.target_length = target_length
        self.normalize = normalize
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel-spectrogram.

        Args:
            waveform: Input waveform tensor (1D)

        Returns:
            Mel-spectrogram in log scale, normalized, transposed to (time, mels) for AST
            Shape: (target_length, n_mels)
        """
        # Ensure waveform is 2D (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale (matching AST's preprocessing)
        mel_spec = torch.log(mel_spec + 1e-9)

        # Normalize to match AST's expected input distribution
        if self.normalize:
            mel_spec = (mel_spec - self.norm_mean) / self.norm_std

        # Remove channel dimension and transpose to (time, mels) for AST
        mel_spec = mel_spec.squeeze(0).T  # Shape: (time, n_mels)

        # Pad or crop to target length
        current_length = mel_spec.size(0)
        if current_length < self.target_length:
            # Pad with zeros (after normalization, 0 represents mean)
            padding = self.target_length - current_length
            mel_spec = torch.nn.functional.pad(mel_spec, (0, 0, 0, padding))
        elif current_length > self.target_length:
            # Crop to target length (take from beginning)
            mel_spec = mel_spec[: self.target_length, :]

        return mel_spec


class ComposeTransforms:
    """Compose multiple transforms together."""

    def __init__(self, transforms: List[Callable]):
        """
        Initialize composed transforms.

        Args:
            transforms: List of transforms to apply sequentially
        """
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all transforms sequentially.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        for transform in self.transforms:
            if transform is not None:
                x = transform(x)
        return x
