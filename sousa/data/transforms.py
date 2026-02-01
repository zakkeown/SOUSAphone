"""Audio transforms for preprocessing."""

import torch
import torchaudio


class MelSpectrogramTransform:
    """Convert waveform to mel-spectrogram for models."""

    def __init__(
        self,
        sample_rate=16000,
        n_mels=128,
        n_fft=400,
        hop_length=160,
        target_length=1024,
    ):
        """
        Initialize mel-spectrogram transform.

        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            target_length: Target number of time frames (for AST: 1024)
        """
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.target_length = target_length

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel-spectrogram.

        Args:
            waveform: Input waveform tensor (1D)

        Returns:
            Mel-spectrogram in log scale, transposed to (time, mels) for AST
            Shape: (target_length, n_mels)
        """
        # Ensure waveform is 2D (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)

        # Remove channel dimension and transpose to (time, mels) for AST
        mel_spec = mel_spec.squeeze(0).T  # Shape: (time, n_mels)

        # Pad or crop to target length
        current_length = mel_spec.size(0)
        if current_length < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_length
            mel_spec = torch.nn.functional.pad(mel_spec, (0, 0, 0, padding))
        elif current_length > self.target_length:
            # Crop to target length (take from beginning)
            mel_spec = mel_spec[: self.target_length, :]

        return mel_spec
