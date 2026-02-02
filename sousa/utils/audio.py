"""Audio loading utilities for SOUSA"""

from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torchaudio


def load_audio(
    audio_path: Path,
    sample_rate: int = 16000,
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    """Load audio file with resampling and optional padding/cropping.

    Args:
        audio_path: Path to audio file (FLAC format)
        sample_rate: Target sample rate for resampling (default: 16kHz)
        max_samples: Maximum number of samples. If provided:
            - Audio longer than max_samples will be cropped
            - Audio shorter than max_samples will be zero-padded

    Returns:
        Audio waveform as 1D torch.Tensor with shape (num_samples,)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If sample_rate or max_samples are invalid

    Example:
        >>> audio = load_audio("sample.flac", sample_rate=16000, max_samples=16000*5)
        >>> audio.shape
        torch.Size([80000])  # 5 seconds at 16kHz
    """
    # Validate inputs
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")

    # Load audio file using soundfile
    waveform, orig_sample_rate = sf.read(audio_path, dtype='float32')

    # Convert to torch tensor
    waveform = torch.from_numpy(waveform)

    # Convert to mono if stereo (average channels)
    # soundfile returns shape (samples,) for mono, (samples, channels) for stereo
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    # Resample if necessary
    if orig_sample_rate != sample_rate:
        # Add channel dimension for resampler
        waveform = waveform.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate,
            new_freq=sample_rate
        )
        waveform = resampler(waveform)
        # Remove channel dimension
        waveform = waveform.squeeze(0)

    # Pad or crop to max_samples if specified
    if max_samples is not None:
        current_samples = waveform.shape[0]

        if current_samples < max_samples:
            # Pad with zeros
            padding = max_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_samples > max_samples:
            # Crop to max_samples
            waveform = waveform[:max_samples]

    return waveform
