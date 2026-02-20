"""Tests for inference pipeline."""

import pytest
import numpy as np
import torch

from sousa.inference.pipeline import OnsetDetector, RudimentPipeline


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
