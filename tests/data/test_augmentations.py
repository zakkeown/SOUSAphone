"""Tests for data augmentation techniques."""

import pytest
import torch
import numpy as np

from sousa.data.augmentations import SpecAugment, Mixup


class TestSpecAugment:
    """Tests for SpecAugment."""

    def test_specaugment_init(self):
        """Test SpecAugment initialization."""
        aug = SpecAugment(
            freq_mask_param=30,
            time_mask_param=40,
            n_freq_masks=2,
            n_time_masks=2,
        )
        assert aug.freq_mask_param == 30
        assert aug.time_mask_param == 40
        assert aug.n_freq_masks == 2
        assert aug.n_time_masks == 2

    def test_specaugment_2d_input(self):
        """Test SpecAugment with 2D input (time, freq)."""
        aug = SpecAugment(
            freq_mask_param=10,
            time_mask_param=10,
            n_freq_masks=1,
            n_time_masks=1,
        )

        # Create dummy spectrogram (time, freq)
        spec = torch.randn(100, 128)
        augmented = aug(spec)

        # Should preserve shape
        assert augmented.shape == spec.shape

        # Should have some zeros (masked regions)
        assert (augmented == 0).any()

    def test_specaugment_3d_input(self):
        """Test SpecAugment with 3D input (batch, time, freq)."""
        aug = SpecAugment(
            freq_mask_param=10,
            time_mask_param=10,
            n_freq_masks=1,
            n_time_masks=1,
        )

        # Create dummy spectrogram batch
        spec = torch.randn(8, 100, 128)
        augmented = aug(spec)

        # Should preserve shape
        assert augmented.shape == spec.shape

        # Should have some zeros (masked regions)
        assert (augmented == 0).any()

    def test_specaugment_masking(self):
        """Test that SpecAugment actually masks frequency and time."""
        aug = SpecAugment(
            freq_mask_param=20,
            time_mask_param=20,
            n_freq_masks=1,
            n_time_masks=1,
        )

        # Create spectrogram with all ones
        spec = torch.ones(100, 128)
        augmented = aug(spec)

        # Count zeros (should have masked regions)
        n_zeros = (augmented == 0).sum().item()
        assert n_zeros > 0, "SpecAugment should create masked regions"

    def test_specaugment_no_modification_with_zero_masks(self):
        """Test that SpecAugment with 0 masks doesn't modify input."""
        aug = SpecAugment(
            freq_mask_param=10,
            time_mask_param=10,
            n_freq_masks=0,
            n_time_masks=0,
        )

        spec = torch.randn(100, 128)
        augmented = aug(spec)

        # Should be identical (except for cloning)
        assert torch.allclose(augmented, spec)

    def test_specaugment_preserves_dtype(self):
        """Test that SpecAugment preserves tensor dtype."""
        aug = SpecAugment()

        spec_float32 = torch.randn(100, 128, dtype=torch.float32)
        augmented_float32 = aug(spec_float32)
        assert augmented_float32.dtype == torch.float32

        spec_float64 = torch.randn(100, 128, dtype=torch.float64)
        augmented_float64 = aug(spec_float64)
        assert augmented_float64.dtype == torch.float64

    def test_specaugment_edge_cases(self):
        """Test SpecAugment with edge cases."""
        aug = SpecAugment(
            freq_mask_param=128,  # Entire freq dimension
            time_mask_param=100,  # Entire time dimension
            n_freq_masks=1,
            n_time_masks=1,
        )

        spec = torch.ones(100, 128)
        augmented = aug(spec)

        # Should still have valid shape
        assert augmented.shape == spec.shape


class TestMixup:
    """Tests for Mixup."""

    def test_mixup_init(self):
        """Test Mixup initialization."""
        mixup = Mixup(alpha=0.2)
        assert mixup.alpha == 0.2

    def test_mixup_creates_soft_labels(self):
        """Test that Mixup creates soft labels."""
        mixup = Mixup(alpha=0.2)

        # Create dummy batch
        num_classes = 40
        batch = {
            'audio': torch.randn(8, 1024, 128),
            'label': torch.randint(0, num_classes, (8,)),
        }

        mixed_batch = mixup(batch)

        # Should have soft labels (2D)
        assert mixed_batch['label'].dim() == 2
        assert mixed_batch['label'].shape[0] == 8
        assert mixed_batch['label'].shape[1] == num_classes

        # Soft labels should sum to 1
        assert torch.allclose(
            mixed_batch['label'].sum(dim=1),
            torch.ones(8),
            atol=1e-6
        )

    def test_mixup_preserves_original_labels(self):
        """Test that Mixup preserves original labels."""
        mixup = Mixup(alpha=0.2)

        batch = {
            'audio': torch.randn(8, 1024, 128),
            'label': torch.randint(0, 40, (8,)),
        }

        original_labels = batch['label'].clone()
        mixed_batch = mixup(batch)

        # Should preserve original labels
        assert 'original_label' in mixed_batch
        assert torch.equal(mixed_batch['original_label'], original_labels)

    def test_mixup_mixes_audio(self):
        """Test that Mixup actually mixes audio."""
        mixup = Mixup(alpha=0.2)

        # Create batch with distinct patterns
        audio1 = torch.ones(4, 1024, 128)
        audio2 = torch.zeros(4, 1024, 128)
        audio = torch.cat([audio1, audio2], dim=0)

        batch = {
            'audio': audio,
            'label': torch.randint(0, 40, (8,)),
        }

        mixed_batch = mixup(batch)

        # Mixed audio should be different from original
        # (unless lambda happens to be exactly 0 or 1, which is unlikely)
        assert not torch.equal(mixed_batch['audio'], batch['audio'])

    def test_mixup_alpha_zero(self):
        """Test Mixup with alpha=0 (no mixing)."""
        mixup = Mixup(alpha=0.0)

        batch = {
            'audio': torch.randn(8, 1024, 128),
            'label': torch.randint(0, 40, (8,)),
        }

        original_audio = batch['audio'].clone()
        mixed_batch = mixup(batch)

        # With alpha=0, audio should be unchanged
        assert torch.allclose(mixed_batch['audio'], original_audio)

    def test_mixup_shape_preservation(self):
        """Test that Mixup preserves tensor shapes."""
        mixup = Mixup(alpha=0.2)

        # Test with different audio shapes
        for shape in [(4, 1024, 128), (8, 512, 64), (16, 2048, 256)]:
            num_classes = 40
            batch = {
                'audio': torch.randn(*shape),
                'label': torch.randint(0, num_classes, (shape[0],)),
            }

            mixed_batch = mixup(batch)

            # Audio shape should be preserved
            assert mixed_batch['audio'].shape == batch['audio'].shape

            # Soft labels should have correct shape
            assert mixed_batch['label'].shape == (shape[0], num_classes)

    def test_mixup_device_handling(self):
        """Test that Mixup works with different devices."""
        mixup = Mixup(alpha=0.2)

        # CPU batch
        batch_cpu = {
            'audio': torch.randn(4, 1024, 128),
            'label': torch.randint(0, 40, (4,)),
        }

        mixed_cpu = mixup(batch_cpu)
        assert mixed_cpu['audio'].device == batch_cpu['audio'].device
        assert mixed_cpu['label'].device == batch_cpu['audio'].device

        # GPU batch (if available)
        if torch.cuda.is_available():
            batch_gpu = {
                'audio': torch.randn(4, 1024, 128).cuda(),
                'label': torch.randint(0, 40, (4,)).cuda(),
            }

            mixed_gpu = mixup(batch_gpu)
            assert mixed_gpu['audio'].device == batch_gpu['audio'].device
            assert mixed_gpu['label'].device == batch_gpu['audio'].device

    def test_mixup_reproducibility(self):
        """Test that Mixup is reproducible with same random seed."""
        # Create batch first
        batch_audio = torch.randn(8, 1024, 128)
        batch_label = torch.randint(0, 40, (8,))

        # Set random seed
        torch.manual_seed(42)
        np.random.seed(42)

        mixup1 = Mixup(alpha=0.2)
        batch1 = {
            'audio': batch_audio.clone(),
            'label': batch_label.clone(),
        }
        mixed1 = mixup1(batch1)

        # Reset seed and repeat
        torch.manual_seed(42)
        np.random.seed(42)

        mixup2 = Mixup(alpha=0.2)
        batch2 = {
            'audio': batch_audio.clone(),
            'label': batch_label.clone(),
        }
        mixed2 = mixup2(batch2)

        # Results should be identical
        assert torch.allclose(mixed1['audio'], mixed2['audio'])
        assert torch.allclose(mixed1['label'], mixed2['label'])

    def test_mixup_lambda_range(self):
        """Test that Mixup lambda values are in valid range."""
        mixup = Mixup(alpha=0.2)

        batch = {
            'audio': torch.randn(100, 1024, 128),
            'label': torch.randint(0, 40, (100,)),
        }

        # Run multiple times to check lambda distribution
        for _ in range(10):
            mixed_batch = mixup(batch)

            # All soft label values should be between 0 and 1
            assert (mixed_batch['label'] >= 0).all()
            assert (mixed_batch['label'] <= 1).all()
