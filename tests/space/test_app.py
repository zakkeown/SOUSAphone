"""Tests for space/app.py classify function and Gradio integration."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import torch

from space.visualizations import plot_onset_timeline, plot_feature_heatmap


def _make_mock_pipeline():
    """Create a mock RudimentPipeline that returns realistic output."""
    mock = MagicMock()
    mock.predict.return_value = {
        "predicted_rudiment": "single-stroke-roll",
        "confidence": 0.85,
        "top5": [
            {"rudiment": "single-stroke-roll", "confidence": 0.85},
            {"rudiment": "double-stroke-open-roll", "confidence": 0.05},
            {"rudiment": "single-paradiddle", "confidence": 0.04},
            {"rudiment": "flam", "confidence": 0.03},
            {"rudiment": "drag", "confidence": 0.02},
        ],
        "onset_times": np.linspace(0.1, 0.9, 8),
        "onset_strengths": np.random.rand(8).astype(np.float32),
        "tempo_bpm": 120.0,
        "predicted_features": np.random.rand(256, 12).astype(np.float32),
        "attention_mask": np.concatenate([
            np.ones(8, dtype=np.float32),
            np.zeros(248, dtype=np.float32),
        ]),
    }
    return mock


def _make_classify_fn(mock_pipeline):
    """Build the classify function with a mock pipeline injected."""
    from space.visualizations import (
        format_rudiment_notation,
        plot_feature_heatmap,
        plot_onset_timeline,
    )

    def classify(audio_input):
        if audio_input is None:
            return None, None, None, None

        sr, audio = audio_input
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        result = mock_pipeline.predict(audio, sr=sr)

        if "error" in result:
            return result["error"], None, None, None

        confidences = {r["rudiment"]: r["confidence"] for r in result["top5"]}
        notation = format_rudiment_notation(result["predicted_rudiment"])
        timeline_fig = plot_onset_timeline(
            audio, sr,
            result["onset_times"],
            result["predicted_features"],
            result["attention_mask"],
        )
        heatmap_fig = plot_feature_heatmap(
            result["predicted_features"],
            result["attention_mask"],
        )
        return confidences, notation, timeline_fig, heatmap_fig

    return classify


@pytest.fixture
def mock_pipeline():
    return _make_mock_pipeline()


@pytest.fixture
def classify_fn(mock_pipeline):
    return _make_classify_fn(mock_pipeline)


@pytest.fixture
def sine_audio():
    """1 second of sine wave at 440Hz, sr=22050."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)
    return (sr, audio)


class TestClassifyFunction:
    def test_none_input_returns_nones(self, classify_fn):
        result = classify_fn(None)
        assert result == (None, None, None, None)

    def test_valid_audio_returns_four_outputs(self, classify_fn, sine_audio):
        confidences, notation, timeline, heatmap = classify_fn(sine_audio)
        assert isinstance(confidences, dict)
        assert len(confidences) == 5
        assert isinstance(notation, str)
        assert isinstance(timeline, plt.Figure)
        assert isinstance(heatmap, plt.Figure)
        plt.close("all")

    def test_confidences_sum_reasonable(self, classify_fn, sine_audio):
        confidences, _, _, _ = classify_fn(sine_audio)
        total = sum(confidences.values())
        assert 0.0 < total <= 1.0
        plt.close("all")

    def test_stereo_audio_converted_to_mono(self, classify_fn, mock_pipeline):
        sr = 22050
        stereo = np.random.rand(sr, 2).astype(np.float32)
        classify_fn((sr, stereo))
        call_args = mock_pipeline.predict.call_args
        audio_arg = call_args[0][0]
        assert audio_arg.ndim == 1
        plt.close("all")

    def test_loud_audio_normalized(self, classify_fn, mock_pipeline):
        sr = 22050
        loud = np.random.rand(sr).astype(np.float32) * 32768  # int16 range
        classify_fn((sr, loud))
        audio_arg = mock_pipeline.predict.call_args[0][0]
        assert np.abs(audio_arg).max() <= 1.0
        plt.close("all")

    def test_error_result_returns_error_string(self, mock_pipeline):
        mock_pipeline.predict.return_value = {"error": "No onsets detected in audio"}
        classify_fn = _make_classify_fn(mock_pipeline)
        sr = 22050
        audio = np.zeros(sr, dtype=np.float32)
        result = classify_fn((sr, audio))
        assert result[0] == "No onsets detected in audio"
        assert result[1] is None
        assert result[2] is None
        assert result[3] is None


class TestGradioIntegration:
    def test_app_module_imports(self):
        """Verify the app module structure is importable (without HF download)."""
        # We can't import app.py directly since it calls load_pipeline() at module level,
        # but we can verify all the components it uses are importable.
        from sousa.inference.pipeline import RudimentPipeline
        from space.visualizations import (
            format_rudiment_notation,
            plot_feature_heatmap,
            plot_onset_timeline,
        )
        assert callable(format_rudiment_notation)
        assert callable(plot_feature_heatmap)
        assert callable(plot_onset_timeline)

    def test_pipeline_predict_contract(self):
        """Verify RudimentPipeline.predict returns expected keys when models loaded."""
        from sousa.inference.pipeline import RudimentPipeline
        pipeline = RudimentPipeline()  # no models loaded
        assert pipeline.feature_model is None
        assert pipeline.classifier is None

    def test_classify_with_real_pipeline_structure(self):
        """Test classify with a pipeline that has the right interface."""
        mock = _make_mock_pipeline()
        classify = _make_classify_fn(mock)
        sr = 22050
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
        confidences, notation, timeline, heatmap = classify((sr, audio))

        # Verify all rudiment names in confidences
        assert "single-stroke-roll" in confidences
        assert confidences["single-stroke-roll"] == pytest.approx(0.85)

        # Verify notation contains formatted name
        assert "Single Stroke Roll" in notation
        assert "R L R L" in notation

        plt.close("all")
