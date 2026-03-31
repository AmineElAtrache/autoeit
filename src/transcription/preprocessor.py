"""
Audio Preprocessor for AutoEIT
Handles noise reduction, silence trimming, normalization, and segmentation
of raw learner audio before ASR transcription.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorConfig:
    """Configuration for the audio preprocessor."""

    target_sample_rate: int = 16000  # Whisper expects 16kHz
    target_channels: int = 1  # Mono
    min_silence_ms: int = 300  # ms of silence to split on
    silence_threshold_db: float = -40.0  # dB threshold for silence
    max_duration_s: float = 30.0  # Max segment length (Whisper limit)
    normalize: bool = True  # RMS normalization
    noise_reduce: bool = True  # Spectral noise gating
    noise_gate_db: float = -35.0  # Noise gate threshold
    trim_silence: bool = True  # Trim leading/trailing silence
    speed_perturbation: bool = False  # Data augmentation only
    speed_factors: Tuple[float, ...] = (0.9, 1.0, 1.1)


@dataclass
class AudioSegment:
    """A processed audio segment ready for transcription."""

    waveform: torch.Tensor  # Shape: (1, N)
    sample_rate: int
    start_time: float  # Seconds from original file start
    end_time: float
    file_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_numpy(self) -> np.ndarray:
        return self.waveform.squeeze().numpy()


class AudioPreprocessor:
    """
    Preprocesses raw audio files from EIT recordings.

    Handles the challenges of L2 learner speech:
    - Variable recording quality and background noise
    - Disfluencies and non-speech sounds (laughter, hesitation)
    - Inconsistent loudness across participants/sessions

    Example:
        >>> preprocessor = AudioPreprocessor(PreprocessorConfig())
        >>> segments = preprocessor.process("participant_01.wav")
        >>> for seg in segments:
        ...     print(f"Segment {seg.start_time:.1f}s – {seg.end_time:.1f}s")
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()
        self._resampler_cache: dict = {}
        logger.info("AudioPreprocessor initialized with config: %s", self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio_path: str | Path) -> list[AudioSegment]:
        """
        Full preprocessing pipeline for a single audio file.

        Args:
            audio_path: Path to .wav, .mp3, .m4a, or .flac file.

        Returns:
            List of AudioSegment objects, one per EIT item response.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Processing: %s", audio_path.name)

        # Step 1: Load and standardize
        waveform, sr = self._load(audio_path)

        # Step 2: Resample to 16kHz if needed
        waveform = self._resample(waveform, sr)

        # Step 3: Convert to mono
        waveform = self._to_mono(waveform)

        # Step 4: Noise reduction (spectral gating)
        if self.config.noise_reduce:
            waveform = self._spectral_gate(waveform)

        # Step 5: RMS normalization
        if self.config.normalize:
            waveform = self._normalize(waveform)

        # Step 6: Segment on silence boundaries
        segments = self._segment(waveform, audio_path)

        # Step 7: Trim silence from individual segments
        if self.config.trim_silence:
            segments = [self._trim_segment(s) for s in segments]

        # Step 8: Drop segments that are too long (>30s) or too short (<0.5s)
        segments = [
            s for s in segments if 0.5 <= s.duration <= self.config.max_duration_s
        ]

        logger.info("Produced %d segments from %s", len(segments), audio_path.name)
        return segments

    def process_batch(
        self, audio_paths: list[str | Path]
    ) -> dict[str, list[AudioSegment]]:
        """Process multiple audio files."""
        results = {}
        for path in audio_paths:
            try:
                results[str(path)] = self.process(path)
            except Exception as e:
                logger.error("Failed to process %s: %s", path, e)
                results[str(path)] = []
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> Tuple[torch.Tensor, int]:
        """Load audio file using torchaudio."""
        try:
            waveform, sr = torchaudio.load(str(path))
            return waveform, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {path}") from e

    def _resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample to target sample rate."""
        if orig_sr == self.config.target_sample_rate:
            return waveform
        key = (orig_sr, self.config.target_sample_rate)
        if key not in self._resampler_cache:
            self._resampler_cache[key] = T.Resample(
                orig_sr, self.config.target_sample_rate
            )
        return self._resampler_cache[key](waveform)

    def _to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert stereo or multi-channel to mono by averaging."""
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _normalize(
        self, waveform: torch.Tensor, target_rms: float = 0.1
    ) -> torch.Tensor:
        """RMS normalization."""
        rms = waveform.pow(2).mean().sqrt()
        if rms > 1e-8:
            waveform = waveform * (target_rms / rms)
        return waveform.clamp(-1.0, 1.0)

    def _spectral_gate(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Simple spectral noise gate using STFT magnitude masking.
        Estimates noise from the first 0.5s of audio (assumed to be
        pre-stimulus silence in EIT recordings).
        """
        sr = self.config.target_sample_rate
        n_fft = 512
        hop_length = 128
        noise_samples = int(0.5 * sr)

        stft = torch.stft(
            waveform.squeeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
        )
        magnitude = stft.abs()

        # Estimate noise floor from first 0.5s
        noise_frames = max(1, noise_samples // hop_length)
        noise_profile = magnitude[:, :noise_frames].mean(dim=1, keepdim=True)
        gate = self.config.noise_gate_db
        mask = magnitude > (noise_profile * 10 ** (gate / 20))
        denoised_stft = stft * mask

        denoised = torch.istft(
            denoised_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            length=waveform.shape[-1],
        )
        return denoised.unsqueeze(0)

    def _segment(self, waveform: torch.Tensor, path: Path) -> list[AudioSegment]:
        """
        Split audio into segments based on silence detection.
        Uses energy-based VAD to find silence boundaries.
        """
        sr = self.config.target_sample_rate
        frame_size = int(0.02 * sr)  # 20ms frames
        threshold = 10 ** (self.config.silence_threshold_db / 20)
        min_silence_frames = int(self.config.min_silence_ms / 20)

        audio = waveform.squeeze(0).numpy()
        n_frames = len(audio) // frame_size

        # Compute per-frame RMS energy
        energy = np.array(
            [
                np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2))
                for i in range(n_frames)
            ]
        )

        # Detect speech/silence frames
        is_speech = energy > threshold

        # Find segment boundaries
        segments = []
        in_speech = False
        start_frame = 0
        silence_count = 0

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    start_frame = i
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        end_frame = i - silence_count
                        seg_waveform = torch.tensor(
                            audio[start_frame * frame_size : end_frame * frame_size]
                        ).unsqueeze(0)
                        segments.append(
                            AudioSegment(
                                waveform=seg_waveform,
                                sample_rate=sr,
                                start_time=start_frame * frame_size / sr,
                                end_time=end_frame * frame_size / sr,
                                file_path=path,
                            )
                        )
                        in_speech = False
                        silence_count = 0

        # Handle last segment
        if in_speech:
            seg_waveform = torch.tensor(audio[start_frame * frame_size :]).unsqueeze(0)
            segments.append(
                AudioSegment(
                    waveform=seg_waveform,
                    sample_rate=sr,
                    start_time=start_frame * frame_size / sr,
                    end_time=len(audio) / sr,
                    file_path=path,
                )
            )

        # Fall back to treating whole file as one segment
        if not segments:
            segments = [
                AudioSegment(
                    waveform=waveform,
                    sample_rate=sr,
                    start_time=0.0,
                    end_time=len(audio) / sr,
                    file_path=path,
                )
            ]

        return segments

    def _trim_segment(self, segment: AudioSegment) -> AudioSegment:
        """Trim leading and trailing silence from a segment."""
        audio = segment.waveform.squeeze(0).numpy()
        threshold = 10 ** (self.config.silence_threshold_db / 20)
        frame_size = int(0.02 * segment.sample_rate)

        # Find first and last speech frame
        n_frames = len(audio) // frame_size
        speech_frames = [
            i
            for i in range(n_frames)
            if np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2))
            > threshold
        ]

        if not speech_frames:
            return segment

        start = speech_frames[0] * frame_size
        end = (speech_frames[-1] + 1) * frame_size
        trimmed = torch.tensor(audio[start:end]).unsqueeze(0)

        return AudioSegment(
            waveform=trimmed,
            sample_rate=segment.sample_rate,
            start_time=segment.start_time + start / segment.sample_rate,
            end_time=segment.start_time + end / segment.sample_rate,
            file_path=segment.file_path,
            metadata=segment.metadata,
        )
