"""Tests for space/visualizations.py."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from space.visualizations import (
    RUDIMENT_STICKING,
    STROKE_COLORS,
    format_rudiment_notation,
    plot_feature_heatmap,
    plot_onset_timeline,
)


@pytest.fixture
def dummy_audio():
    """1 second of sine wave at 440Hz."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)
    return audio, sr


@pytest.fixture
def dummy_predictions():
    """Fake pipeline output with 5 onsets."""
    n = 5
    max_seq_len = 256
    onset_times = np.linspace(0.1, 0.9, n)
    predicted_features = np.zeros((max_seq_len, 12), dtype=np.float32)
    # Set some binary features above threshold for different strokes
    predicted_features[0, 3] = 0.9  # accent
    predicted_features[1, 2] = 0.8  # grace
    predicted_features[2, 5] = 0.7  # diddle
    predicted_features[3, 10] = 0.9  # buzz
    predicted_features[4, 4] = 0.6  # tap (default)
    attention_mask = np.zeros(max_seq_len, dtype=np.float32)
    attention_mask[:n] = 1.0
    return onset_times, predicted_features, attention_mask


class TestPlotOnsetTimeline:
    def test_returns_figure(self, dummy_audio, dummy_predictions):
        audio, sr = dummy_audio
        onset_times, features, mask = dummy_predictions
        fig = plot_onset_timeline(audio, sr, onset_times, features, mask)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_handles_empty_onsets(self, dummy_audio):
        audio, sr = dummy_audio
        onset_times = np.array([])
        features = np.zeros((256, 12), dtype=np.float32)
        mask = np.zeros(256, dtype=np.float32)
        fig = plot_onset_timeline(audio, sr, onset_times, features, mask)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotFeatureHeatmap:
    def test_returns_figure(self, dummy_predictions):
        _, features, mask = dummy_predictions
        fig = plot_feature_heatmap(features, mask)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correct_dimensions(self, dummy_predictions):
        _, features, mask = dummy_predictions
        fig = plot_feature_heatmap(features, mask)
        ax = fig.axes[0]
        images = ax.get_images()
        assert len(images) == 1
        # Heatmap should be 12 rows x 5 cols (n_real onsets)
        assert images[0].get_array().shape == (12, 5)
        plt.close(fig)


class TestFormatRudimentNotation:
    def test_known_rudiment(self):
        result = format_rudiment_notation("single-stroke-roll")
        assert "Single Stroke Roll" in result
        assert "R L R L R L R L" in result

    def test_unknown_rudiment(self):
        result = format_rudiment_notation("some-unknown-rudiment")
        assert "Some Unknown Rudiment" in result
        assert "PAS International" in result

    def test_all_stickings_are_strings(self):
        for name, sticking in RUDIMENT_STICKING.items():
            assert isinstance(sticking, str)
            assert len(sticking) > 0


class TestStrokeColors:
    def test_all_colors_are_hex(self):
        for name, color in STROKE_COLORS.items():
            assert color.startswith("#")
            assert len(color) == 7
