"""Tests for dynamic model loading in train.py."""

import importlib
import pytest
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path


class TestDynamicModelLoading:
    """Test that all models load correctly via dynamic loading."""

    @pytest.fixture
    def config_path(self):
        """Get absolute path to configs directory."""
        return str(Path(__file__).parent.parent.parent / "configs")

    @pytest.mark.parametrize("model_name", ["ast", "htsat", "beats", "efficientat"])
    def test_model_class_resolution(self, config_path, model_name):
        """Test that model class_path resolves correctly for each model."""
        with initialize_config_dir(config_dir=config_path, version_base=None):
            cfg = compose(config_name="config", overrides=[f"model={model_name}"])

            # Test class path resolution
            assert hasattr(cfg.model, 'class_path')
            module_path, class_name = cfg.model.class_path.rsplit('.', 1)

            # Test module import
            module = importlib.import_module(module_path)
            assert module is not None

            # Test class retrieval
            model_class = getattr(module, class_name)
            assert model_class is not None

            # Test model instantiation
            model = model_class(
                num_classes=cfg.model.num_classes,
                pretrained=False,  # Don't download weights for tests
            )
            assert model is not None
            assert hasattr(model, 'forward')

    @pytest.mark.parametrize("model_name,expected_type", [
        ("ast", "spectrogram"),
        ("htsat", "spectrogram"),
        ("beats", "waveform"),
        ("efficientat", "spectrogram"),
    ])
    def test_input_type_detection(self, config_path, model_name, expected_type):
        """Test that input_type is correctly specified for each model."""
        with initialize_config_dir(config_dir=config_path, version_base=None):
            cfg = compose(config_name="config", overrides=[f"model={model_name}"])

            assert hasattr(cfg.model, 'input_type')
            assert cfg.model.input_type == expected_type

    def test_all_models_have_num_classes(self, config_path):
        """Test that all model configs have num_classes parameter."""
        models = ["ast", "htsat", "beats", "efficientat"]

        for model_name in models:
            with initialize_config_dir(config_dir=config_path, version_base=None):
                cfg = compose(config_name="config", overrides=[f"model={model_name}"])

                assert hasattr(cfg.model, 'num_classes')
                assert cfg.model.num_classes == 40

    def test_all_models_have_pretrained_flag(self, config_path):
        """Test that all model configs have pretrained flag."""
        models = ["ast", "htsat", "beats", "efficientat"]

        for model_name in models:
            with initialize_config_dir(config_dir=config_path, version_base=None):
                cfg = compose(config_name="config", overrides=[f"model={model_name}"])

                assert hasattr(cfg.model, 'pretrained')
                assert isinstance(cfg.model.pretrained, bool)

    @pytest.mark.parametrize("model_name,input_type", [
        ("ast", "spectrogram"),
        ("htsat", "spectrogram"),
        ("beats", "waveform"),
        ("efficientat", "spectrogram"),
    ])
    def test_model_forward_pass(self, config_path, model_name, input_type):
        """Test that each model can perform a forward pass with correct input."""
        with initialize_config_dir(config_dir=config_path, version_base=None):
            cfg = compose(config_name="config", overrides=[f"model={model_name}"])

            # Dynamically load model
            module_path, class_name = cfg.model.class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            # Instantiate model without pretrained weights
            model = model_class(
                num_classes=cfg.model.num_classes,
                pretrained=False,
            )
            model.eval()

            # Create appropriate input based on model config
            batch_size = 2
            if input_type == "spectrogram":
                # Use n_mels and max_length from config
                n_mels = cfg.model.get('n_mels', 128)
                max_length = cfg.model.get('max_length', 1024)
                # (batch, time, mels) — matches model convention
                audio = torch.randn(batch_size, max_length, n_mels)
            else:  # waveform
                # (batch, samples)
                audio = torch.randn(batch_size, 16000 * 5)  # 5 seconds

            # Forward pass
            with torch.no_grad():
                logits = model(audio)

            # Verify output shape
            assert logits.shape == (batch_size, cfg.model.num_classes)
