"""Integration tests for transcription evaluation."""

import pytest
import json
from pathlib import Path
import tempfile
import pandas as pd

from src.transcription.evaluator import (
    TranscriptionEvaluator,
    TranscriptionEvalResult,
)


@pytest.fixture
def evaluator():
    return TranscriptionEvaluator(target_wer=0.10)


@pytest.fixture
def sample_data():
    """Sample transcription pairs for testing."""
    return [
        {
            "audio": "test_001.wav",
            "reference": "yo fui al mercado ayer",
            "hypothesis": "yo fui al mercado ayer",  # Perfect match
        },
        {
            "audio": "test_002.wav",
            "reference": "mi hermana estudia en la universidad",
            "hypothesis": "mi hermana estudia la universidad",  # 1 deletion
        },
        {
            "audio": "test_003.wav",
            "reference": "nosotros viajamos a españa",
            "hypothesis": "nosotros viajamos españa",  # 1 deletion
        },
        {
            "audio": "test_004.wav",
            "reference": "el gato está durmiendo",
            "hypothesis": "el gato esta durmiendo",  # 1 accent error
        },
    ]


class TestTranscriptionEvaluator:
    def test_perfect_match(self, evaluator):
        result = evaluator._evaluate_sample(
            "test.wav",
            "yo fui al mercado",
            "yo fui al mercado",
        )
        assert result.exact_match is True
        assert result.wer == 0.0

    def test_single_deletion(self, evaluator):
        result = evaluator._evaluate_sample(
            "test.wav",
            "yo fui al mercado ayer",
            "yo fui mercado ayer",
        )
        assert result.exact_match is False
        assert result.wer > 0.0

    def test_case_insensitive(self, evaluator):
        result = evaluator._evaluate_sample(
            "test.wav",
            "Yo Fui Al Mercado",
            "yo fui al mercado",
        )
        assert result.exact_match is True

    def test_aggregation(self, evaluator, sample_data):
        samples = [
            evaluator._evaluate_sample(
                s["audio"],
                s["reference"],
                s["hypothesis"],
            )
            for s in sample_data
        ]
        
        eval_result = evaluator._aggregate(samples)
        assert eval_result.total_samples == len(sample_data)
        assert eval_result.avg_wer >= 0.0
        assert 0.0 <= eval_result.exact_match_rate <= 1.0

    def test_evaluate_csv(self, evaluator):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            data = {
                "audio": ["a.wav", "b.wav"],
                "hypothesis": [
                    "yo fui al mercado",
                    "mi hermana estudia",
                ],
                "reference": [
                    "yo fui al mercado",
                    "mi hermana estudia en la universidad",
                ],
            }
            pd.DataFrame(data).to_csv(csv_path, index=False)
            
            eval_result = evaluator.evaluate_csv(
                csv_path,
                hyp_col="hypothesis",
                ref_col="reference",
                audio_col="audio",
            )
            
            assert eval_result.total_samples == 2
            assert eval_result.exact_match_rate == 0.5

    def test_save_report(self, evaluator, sample_data):
        samples = [
            evaluator._evaluate_sample(
                s["audio"],
                s["reference"],
                s["hypothesis"],
            )
            for s in sample_data
        ]
        
        eval_result = evaluator._aggregate(samples)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            evaluator.save_report(eval_result, output_path)
            
            assert output_path.exists()
            with open(output_path) as f:
                report = json.load(f)
            
            assert "avg_wer" in report
            assert "total_samples" in report


class TestErrorAnalysis:
    def test_error_counting(self, evaluator):
        result = evaluator._analyze_errors(
            "yo fui al mercado ayer",
            "yo fui mercado ayer",  # missing "al"
        )
        assert result["ref_tokens"] == 4
        assert result["hyp_tokens"] == 3
        assert result["deletions"] > 0

    def test_insertion_detection(self, evaluator):
        result = evaluator._analyze_errors(
            "yo fui al mercado",
            "yo fui al mercado ayer extra",  # added "ayer extra"
        )
        assert result["insertions"] > 0


class TestMeetsTarget:
    def test_meets_target_with_low_wer(self, evaluator):
        # 90% agreement corresponds to ~10% WER
        samples = [
            evaluator._evaluate_sample(
                f"test_{i}.wav",
                "yo fui al mercado ayer",
                "yo fui al mercado ayer",  # Perfect match
            )
            for i in range(10)
        ]
        
        eval_result = evaluator._aggregate(samples)
        assert eval_result.meets_target is True
        assert eval_result.avg_wer <= evaluator.target_wer

    def test_fails_target_with_high_wer(self, evaluator):
        samples = [
            evaluator._evaluate_sample(
                f"test_{i}.wav",
                "yo fui al mercado ayer con mi madre",
                "yo fui",  # Only first 2 words
            )
            for i in range(10)
        ]
        
        eval_result = evaluator._aggregate(samples)
        assert eval_result.meets_target is False
        assert eval_result.avg_wer > evaluator.target_wer
