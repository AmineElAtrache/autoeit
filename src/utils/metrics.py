"""
Evaluation Metrics for AutoEIT
Implements WER, agreement rates, Cohen's kappa, and EIT-specific metrics
for evaluating both transcription and scoring quality.
"""

import logging
from collections import Counter
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def word_error_rate(hypotheses: list[str], references: list[str]) -> float:
    """
    Compute Word Error Rate (WER) using dynamic programming edit distance.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference words.

    Args:
        hypotheses: List of ASR output strings.
        references: List of reference transcript strings.

    Returns:
        WER as a float in [0, ∞) (>1 if more errors than words).
    """
    assert len(hypotheses) == len(references), "Length mismatch"
    total_errors = 0
    total_words = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.lower().split()
        ref_tokens = ref.lower().split()
        total_errors += _edit_distance(hyp_tokens, ref_tokens)
        total_words += len(ref_tokens)

    return total_errors / max(total_words, 1)


def _edit_distance(hyp: list[str], ref: list[str]) -> int:
    """Levenshtein edit distance between two token sequences."""
    n, m = len(hyp), len(ref)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[m]


def percent_agreement(scores_a: list[int], scores_b: list[int]) -> float:
    """
    Compute percent exact agreement between two sets of integer scores.

    Args:
        scores_a: Scores from rater A (or system).
        scores_b: Scores from rater B (or human gold standard).

    Returns:
        Agreement rate in [0, 1].
    """
    assert len(scores_a) == len(scores_b)
    matches = sum(a == b for a, b in zip(scores_a, scores_b))
    return matches / len(scores_a)


def cohens_kappa(scores_a: list[int], scores_b: list[int], max_score: int = 6) -> float:
    """
    Compute Cohen's weighted kappa (quadratic weights) for ordinal ratings.

    Quadratic weights penalize disagreements proportionally to the squared
    distance between categories — standard for Likert/rubric scales.

    Args:
        scores_a: First rater's scores.
        scores_b: Second rater's scores.
        max_score: Maximum possible score.

    Returns:
        Weighted kappa in [-1, 1].
    """
    n = len(scores_a)
    k = max_score + 1
    conf_matrix = np.zeros((k, k), dtype=float)
    for a, b in zip(scores_a, scores_b):
        conf_matrix[a][b] += 1

    weights = np.array(
        [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    )

    row_marginal = conf_matrix.sum(axis=1) / n
    col_marginal = conf_matrix.sum(axis=0) / n
    expected = np.outer(row_marginal, col_marginal)

    po = 1 - (weights * conf_matrix / n).sum()
    pe = 1 - (weights * expected).sum()
    return po / pe if pe != 0 else 1.0


def score_difference_stats(
    auto_scores: list[int],
    human_scores: list[int],
    scale: int = 120,
) -> dict:
    """
    Compute summary statistics for total score differences across a full EIT protocol.

    The EIT consists of 20 sentences, each scored 0–6, giving a total of 0–120.
    The target is <10 point mean absolute difference (MAD) on the 120-point scale.

    Args:
        auto_scores: Per-sentence automated scores.
        human_scores: Per-sentence human scores.
        scale: Max possible total score (default 120 for 20 × 6).

    Returns:
        Dictionary with mean_abs_diff, rmse, max_diff, within_10_pct.
    """
    assert len(auto_scores) == len(human_scores)
    n_items = len(auto_scores)
    items_per_protocol = 20  # Standard EIT protocol
    n_protocols = n_items // items_per_protocol

    protocol_diffs = []
    for i in range(n_protocols):
        start = i * items_per_protocol
        end = start + items_per_protocol
        auto_total = sum(auto_scores[start:end])
        human_total = sum(human_scores[start:end])
        protocol_diffs.append(abs(auto_total - human_total))

    diffs = (
        np.array(protocol_diffs)
        if protocol_diffs
        else np.array([abs(sum(auto_scores) - sum(human_scores))])
    )

    return {
        "mean_abs_diff": float(diffs.mean()),
        "rmse": float(np.sqrt((diffs**2).mean())),
        "max_diff": float(diffs.max()),
        "within_10_pts": float((diffs <= 10).mean()),
        "n_protocols": len(diffs),
    }


def confusion_matrix(
    predicted: list[int], actual: list[int], labels: Optional[list[int]] = None
) -> np.ndarray:
    """Compute confusion matrix for scoring evaluation."""
    from typing import Optional

    if labels is None:
        labels = sorted(set(predicted) | set(actual))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    for p, a in zip(predicted, actual):
        if p in label_to_idx and a in label_to_idx:
            matrix[label_to_idx[a]][label_to_idx[p]] += 1
    return matrix


def print_evaluation_report(
    hypotheses: list[str],
    references: list[str],
    auto_scores: list[int],
    human_scores: list[int],
) -> None:
    """Print a formatted evaluation report to stdout."""
    wer = word_error_rate(hypotheses, references)
    agreement = percent_agreement(auto_scores, human_scores)
    kappa = cohens_kappa(auto_scores, human_scores)
    diff_stats = score_difference_stats(auto_scores, human_scores)

    print("=" * 60)
    print("AutoEIT Evaluation Report")
    print("=" * 60)
    print(f"  Transcription WER:         {wer:.1%}")
    print(f"  Scoring Agreement:         {agreement:.1%}")
    print(f"  Cohen's Weighted Kappa:    {kappa:.3f}")
    print(f"  Mean Score Difference:     {diff_stats['mean_abs_diff']:.1f} pts")
    print(f"  Within 10-pt threshold:    {diff_stats['within_10_pts']:.1%}")
    print("=" * 60)
