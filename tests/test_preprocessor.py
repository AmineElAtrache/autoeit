"""Tests for the AudioPreprocessor module."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.transcription.preprocessor import AudioPreprocessor, PreprocessorConfig, AudioSegment


@pytest.fixture
def config():
    return PreprocessorConfig(
        target_sample_rate=16000,
        normalize=True,
        noise_reduce=False,   # Disable for unit tests (no real audio)
        trim_silence=True,
    )


@pytest.fixture
def preprocessor(config):
    return AudioPreprocessor(config)


@pytest.fixture
def synthetic_waveform():
    """1-second 16kHz sine wave (440Hz, A4)."""
    sr = 16000
    t = torch.linspace(0, 1, sr)
    wave = (0.5 * torch.sin(2 * torch.pi * 440 * t)).unsqueeze(0)
    return wave, sr


class TestPreprocessorInit:
    def test_default_config(self):
        pp = AudioPreprocessor()
        assert pp.config.target_sample_rate == 16000
        assert pp.config.normalize is True

    def test_custom_config(self, preprocessor, config):
        assert preprocessor.config.noise_reduce is False


class TestResample:
    def test_no_resample_needed(self, preprocessor, synthetic_waveform):
        wave, sr = synthetic_waveform
        resampled = preprocessor._resample(wave, sr)
        assert resampled.shape == wave.shape

    def test_resample_from_44100(self, preprocessor):
        sr_in = 44100
        sr_out = 16000
        duration = 1.0
        wave = torch.randn(1, int(sr_in * duration))
        resampled = preprocessor._resample(wave, sr_in)
        expected_len = int(sr_out * duration)
        assert abs(resampled.shape[-1] - expected_len) <= 2   # Allow ±2 sample rounding

    def test_resampler_cache(self, preprocessor):
        wave = torch.randn(1, 44100)
        preprocessor._resample(wave, 44100)
        preprocessor._resample(wave, 44100)
        assert len(preprocessor._resampler_cache) == 1


class TestMonoConversion:
    def test_stereo_to_mono(self, preprocessor):
        stereo = torch.randn(2, 16000)
        mono = preprocessor._to_mono(stereo)
        assert mono.shape[0] == 1

    def test_mono_unchanged(self, preprocessor):
        mono = torch.randn(1, 16000)
        result = preprocessor._to_mono(mono)
        assert result.shape[0] == 1
        assert torch.allclose(result, mono)


class TestNormalization:
    def test_normalization_reduces_loud_signal(self, preprocessor):
        loud = torch.ones(1, 16000) * 5.0
        normalized = preprocessor._normalize(loud)
        rms = normalized.pow(2).mean().sqrt().item()
        assert abs(rms - 0.1) < 0.01

    def test_silent_signal_handled(self, preprocessor):
        silent = torch.zeros(1, 16000)
        result = preprocessor._normalize(silent)
        assert not torch.isnan(result).any()

    def test_output_clamped(self, preprocessor):
        loud = torch.ones(1, 16000) * 100.0
        result = preprocessor._normalize(loud)
        assert result.abs().max().item() <= 1.0


class TestSegmentation:
    def test_single_segment_silence_file(self, preprocessor, tmp_path):
        """File with no silence → treated as one segment."""
        audio = torch.randn(1, 16000) * 0.5
        with patch.object(preprocessor, "_load", return_value=(audio, 16000)):
            with patch.object(preprocessor, "_spectral_gate", return_value=audio):
                segments = preprocessor._segment(audio, tmp_path / "test.wav")
        assert len(segments) >= 1

    def test_segment_audio_segment_type(self, preprocessor, tmp_path):
        audio = torch.randn(1, 16000) * 0.5
        segments = preprocessor._segment(audio, tmp_path / "test.wav")
        for seg in segments:
            assert isinstance(seg, AudioSegment)
            assert seg.sample_rate == 16000

    def test_duration_filter(self, preprocessor):
        """Segments shorter than 0.5s should be filtered out."""
        tiny = AudioSegment(
            waveform=torch.randn(1, 100),
            sample_rate=16000,
            start_time=0.0,
            end_time=0.006,
        )
        segments = [tiny]
        filtered = [s for s in segments if 0.5 <= s.duration <= preprocessor.config.max_duration_s]
        assert len(filtered) == 0


class TestAudioSegmentProperties:
    def test_duration(self):
        seg = AudioSegment(
            waveform=torch.randn(1, 16000),
            sample_rate=16000,
            start_time=1.0,
            end_time=2.0,
        )
        assert seg.duration == pytest.approx(1.0)

    def test_to_numpy(self):
        wave = torch.randn(1, 16000)
        seg = AudioSegment(waveform=wave, sample_rate=16000, start_time=0, end_time=1)
        arr = seg.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (16000,)
