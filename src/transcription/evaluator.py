"""
Transcription evaluation framework.
Compares ASR output against human transcriptions and validates 90% agreement goal.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
from jiwer import compute_wer, compute_cer

from src.utils.metrics import (
    word_error_rate,
    percent_agreement,
    cohens_kappa,
)

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionEvalResult:
    """Results for a single transcription."""

    audio_file: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    exact_match: bool
    errors: Dict[str, Any]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TranscriptionEvaluation:
    """Overall evaluation metrics."""

    total_samples: int
    avg_wer: float
    avg_cer: float
    exact_match_rate: float
    samples: List[TranscriptionEvalResult]
    meets_target: bool
    target_wer: float = 0.10
    target_agreement: float = 0.90

    def to_dict(self) -> Dict:
        return {
            "total_samples": self.total_samples,
            "avg_wer": self.avg_wer,
            "avg_cer": self.avg_cer,
            "exact_match_rate": self.exact_match_rate,
            "meets_target": self.meets_target,
            "target_wer": self.target_wer,
            "target_agreement": self.target_agreement,
            "samples": [s.to_dict() for s in self.samples],
        }


class TranscriptionEvaluator:
    """Evaluate transcription quality against human references."""

    def __init__(self, target_wer: float = 0.10):
        """
        Args:
            target_wer: Target Word Error Rate (goal: ≤10% for 90% agreement)
        """
        self.target_wer = target_wer

    def evaluate_csv(
        self,
        csv_path: str | Path,
        hyp_col: str = "hypothesis",
        ref_col: str = "reference",
        audio_col: Optional[str] = "audio",
    ) -> TranscriptionEvaluation:
        """
        Evaluate transcriptions from a CSV file.

        Args:
            csv_path: Path to CSV with transcriptions
            hyp_col: Column name for ASR output
            ref_col: Column name for reference text
            audio_col: Column name for audio filename (optional)

        Returns:
            TranscriptionEvaluation with detailed metrics
        """
        df = pd.read_csv(csv_path)

        if hyp_col not in df.columns or ref_col not in df.columns:
            raise ValueError(f"CSV must contain '{hyp_col}' and '{ref_col}' columns")

        samples = []
        for idx, row in df.iterrows():
            audio_file = (
                row.get(audio_col, f"sample_{idx:03d}")
                if audio_col
                else f"sample_{idx:03d}"
            )
            result = self._evaluate_sample(
                audio_file,
                row[ref_col],
                row[hyp_col],
            )
            samples.append(result)

        return self._aggregate(samples)

    def _evaluate_sample(
        self,
        audio_file: str,
        reference: str,
        hypothesis: str,
    ) -> TranscriptionEvalResult:
        """Evaluate a single transcription pair."""
        ref_norm = reference.lower().strip()
        hyp_norm = hypothesis.lower().strip()

        # Compute metrics
        wer = compute_wer(ref_norm, hyp_norm)
        cer = compute_cer(ref_norm, hyp_norm)
        exact_match = ref_norm == hyp_norm

        # Detailed error analysis
        errors = self._analyze_errors(ref_norm, hyp_norm)

        return TranscriptionEvalResult(
            audio_file=audio_file,
            reference=reference,
            hypothesis=hypothesis,
            wer=wer,
            cer=cer,
            exact_match=exact_match,
            errors=errors,
        )

    def _analyze_errors(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """Analyze error types."""
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()

        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)

        # Token-level analysis
        matched = len(set(ref_tokens) & set(hyp_tokens))
        deletions = ref_len - matched  # Words in ref but not in hyp
        insertions = hyp_len - matched  # Words in hyp but not in ref

        return {
            "ref_tokens": ref_len,
            "hyp_tokens": hyp_len,
            "matched_tokens": matched,
            "deletions": max(0, deletions),
            "insertions": max(0, insertions),
        }

    def _aggregate(
        self, samples: List[TranscriptionEvalResult]
    ) -> TranscriptionEvaluation:
        """Aggregate results across samples."""
        if not samples:
            return TranscriptionEvaluation(
                total_samples=0,
                avg_wer=0.0,
                avg_cer=0.0,
                exact_match_rate=0.0,
                samples=[],
                meets_target=False,
            )

        avg_wer = sum(s.wer for s in samples) / len(samples)
        avg_cer = sum(s.cer for s in samples) / len(samples)
        exact_match_rate = sum(s.exact_match for s in samples) / len(samples)

        meets_target = avg_wer <= self.target_wer

        return TranscriptionEvaluation(
            total_samples=len(samples),
            avg_wer=avg_wer,
            avg_cer=avg_cer,
            exact_match_rate=exact_match_rate,
            samples=samples,
            meets_target=meets_target,
            target_wer=self.target_wer,
        )

    def save_report(
        self,
        evaluation: TranscriptionEvaluation,
        output_path: str | Path,
    ) -> None:
        """Save evaluation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(evaluation.to_dict(), f, indent=2)

        logger.info(f"Report saved to {output_path}")

    def print_summary(self, evaluation: TranscriptionEvaluation) -> None:
        """Print summary statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("TRANSCRIPTION EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Samples: {evaluation.total_samples}")
        logger.info(
            f"Avg WER: {evaluation.avg_wer:.2%} (target: ≤{self.target_wer:.0%})"
        )
        logger.info(f"Avg CER: {evaluation.avg_cer:.2%}")
        logger.info(f"Exact Match Rate: {evaluation.exact_match_rate:.2%}")
        logger.info(f"Meets Target: {'✓ YES' if evaluation.meets_target else '✗ NO'}")
        logger.info("=" * 60)

        if not evaluation.meets_target:
            logger.warning(
                f"⚠️ WER {evaluation.avg_wer:.2%} exceeds target {self.target_wer:.0%}"
            )


def compare_transcriptions(
    baseline_csv: str | Path,
    test_csv: str | Path,
    output_json: str | Path,
) -> None:
    """
    Compare two sets of transcriptions for inter-rater reliability.

    Args:
        baseline_csv: CSV with reference/baseline transcriptions
        test_csv: CSV with test transcriptions
        output_json: Output file for comparison report
    """
    baseline_df = pd.read_csv(baseline_csv)
    test_df = pd.read_csv(test_csv)

    if len(baseline_df) != len(test_df):
        raise ValueError("CSVs must have same number of rows")

    evaluator = TranscriptionEvaluator()

    comparison_results = []
    for idx, (b_row, t_row) in enumerate(
        zip(baseline_df.itertuples(), test_df.itertuples())
    ):
        baseline_text = b_row.transcription or b_row.hypothesis
        test_text = t_row.transcription or t_row.hypothesis

        result = evaluator._evaluate_sample(
            f"sample_{idx:03d}",
            baseline_text,
            test_text,
        )
        comparison_results.append(result)

    aggregated = evaluator._aggregate(comparison_results)

    evaluator.save_report(aggregated, output_json)
    evaluator.print_summary(aggregated)
