"""Tests for inference pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np
import torch

from sousa.inference.pipeline import OnsetDetector, RudimentPipeline
from sousa.models.feature_inference import FeatureInferenceModel
from sousa.models.onset_transformer import OnsetTransformerModel


class TestOnsetDetector:
    def test_detect_returns_times_and_strengths(self):
        """Should return onset times and strengths from audio."""
        detector = OnsetDetector()
        # 2 seconds of audio at 22050 Hz with some impulses
        audio = np.zeros(44100, dtype=np.float32)
        audio[11025] = 1.0  # impulse at 0.5s
        audio[22050] = 1.0  # impulse at 1.0s
        audio[33075] = 1.0  # impulse at 1.5s

        times, strengths = detector.detect(audio, sr=22050)
        assert isinstance(times, np.ndarray)
        assert isinstance(strengths, np.ndarray)
        assert len(times) == len(strengths)
        assert len(times) > 0

    def test_estimate_tempo(self):
        """Should estimate tempo from audio."""
        detector = OnsetDetector()
        # Generate click track at 120 BPM (0.5s intervals)
        sr = 22050
        audio = np.zeros(sr * 4, dtype=np.float32)
        for i in range(8):
            idx = int(i * 0.5 * sr)
            if idx < len(audio):
                audio[idx : idx + 100] = 1.0

        tempo = detector.estimate_tempo(audio, sr=sr)
        assert isinstance(tempo, float)
        assert tempo > 0


class TestRudimentPipeline:
    def test_pipeline_initializes(self):
        """Pipeline should initialize with model paths."""
        pipeline = RudimentPipeline(
            feature_model_path=None, classifier_model_path=None
        )
        assert pipeline is not None

    def test_prepare_raw_onsets(self):
        """Should convert onset times/strengths to model input tensor."""
        pipeline = RudimentPipeline(
            feature_model_path=None, classifier_model_path=None
        )
        times = np.array([0.0, 0.25, 0.5, 0.75])
        strengths = np.array([0.8, 0.6, 0.9, 0.5])
        tempo = 120.0

        raw_onsets, mask = pipeline.prepare_raw_onsets(times, strengths, tempo)
        assert raw_onsets.shape[0] == 1  # batch dim
        assert raw_onsets.shape[2] == 3  # (ioi_ms, strength, tempo)
        assert mask.shape[0] == 1

    def test_load_config_found(self):
        """_load_config should return dict when config file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"input_dim": 3, "output_dim": 12}
            config_path = Path(tmpdir) / "feature_inference_config.json"
            config_path.write_text(json.dumps(config))
            model_path = Path(tmpdir) / "model.bin"
            model_path.touch()

            result = RudimentPipeline._load_config(
                str(model_path), "feature_inference_config.json"
            )
            assert result == config

    def test_load_config_missing(self):
        """_load_config should return None when config file is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.bin"
            model_path.touch()

            result = RudimentPipeline._load_config(
                str(model_path), "feature_inference_config.json"
            )
            assert result is None

    def test_pipeline_loads_from_config(self):
        """Pipeline should load models using config files when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save feature inference model + config
            feat_model = FeatureInferenceModel()
            feat_weights = tmpdir / "feature_inference_model.bin"
            torch.save(feat_model.state_dict(), feat_weights)
            feat_config = tmpdir / "feature_inference_config.json"
            feat_config.write_text(json.dumps(feat_model.get_config()))

            # Save classifier model + config
            cls_model = OnsetTransformerModel()
            cls_weights = tmpdir / "pytorch_model.bin"
            torch.save(cls_model.state_dict(), cls_weights)
            cls_config = tmpdir / "onset_transformer_config.json"
            cls_config.write_text(json.dumps(cls_model.get_config()))

            # Load via pipeline
            pipeline = RudimentPipeline(
                feature_model_path=str(feat_weights),
                classifier_model_path=str(cls_weights),
            )
            assert pipeline.feature_model is not None
            assert pipeline.classifier is not None

    def test_pipeline_loads_without_config(self):
        """Pipeline should fall back to default constructors without config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            feat_model = FeatureInferenceModel()
            feat_weights = tmpdir / "feature_inference_model.bin"
            torch.save(feat_model.state_dict(), feat_weights)

            cls_model = OnsetTransformerModel()
            cls_weights = tmpdir / "pytorch_model.bin"
            torch.save(cls_model.state_dict(), cls_weights)

            pipeline = RudimentPipeline(
                feature_model_path=str(feat_weights),
                classifier_model_path=str(cls_weights),
            )
            assert pipeline.feature_model is not None
            assert pipeline.classifier is not None
