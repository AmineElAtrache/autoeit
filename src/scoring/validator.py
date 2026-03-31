"""
Scoring validation and ML ensemble for AutoEIT.

Provides inter-rater reliability assessment and ML-based fallback
for edge cases where rule-based scoring has low confidence.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoringConsistency:
    """Measure of consistency between automated and human scores."""
    mean_abs_diff: float
    max_diff: int
    exact_agreement: float
    within_1_point: float
    kappa: float  # Cohen's weighted kappa


class ScoringValidator:
    """
    Validates scoring consistency against human rater baseline.
    
    Computes inter-rater agreement metrics and identifies
    systematic biases or problem cases.
    """
    
    def __init__(self, max_score: int = 6):
        """
        Args:
            max_score: Maximum possible score (0-6 for EIT)
        """
        self.max_score = max_score
    
    def validate_against_baseline(
        self,
        auto_scores: List[int],
        human_scores: List[int],
    ) -> Dict:
        """
        Compare automated scores against human baseline.
        
        Args:
            auto_scores: Automated system scores
            human_scores: Expert human rater scores (or mean of multiple raters)
        
        Returns:
            Dict with consistency metrics and error analysis
        """
        assert len(auto_scores) == len(human_scores), "Length mismatch"
        
        diffs = np.array([abs(a - h) for a, h in zip(auto_scores, human_scores)])
        
        metrics = {
            "n_samples": len(auto_scores),
            "mean_abs_diff": float(np.mean(diffs)),
            "std_diff": float(np.std(diffs)),
            "max_diff": int(np.max(diffs)),
            "min_diff": int(np.min(diffs)),
            "exact_agreement": float(np.mean(diffs == 0)),
            "within_1_point": float(np.mean(diffs <= 1)),
            "within_2_points": float(np.mean(diffs <= 2)),
            "kappa": self._compute_kappa(auto_scores, human_scores),
        }
        
        # Identify problem scores
        problem_idx = np.where(diffs > 2)[0]
        metrics["problem_cases"] = int(len(problem_idx))
        metrics["problem_rate"] = float(len(problem_idx) / len(auto_scores))
        
        return metrics
    
    def _compute_kappa(self, scores_a: List[int], scores_b: List[int]) -> float:
        """Compute Cohen's weighted kappa for ordinal scores."""
        n = len(scores_a)
        k = self.max_score + 1
        
        # Confusion matrix
        conf_matrix = np.zeros((k, k), dtype=float)
        for a, b in zip(scores_a, scores_b):
            conf_matrix[int(a)][int(b)] += 1
        
        # Quadratic weights for ordinal scale
        weights = np.array([
            [(i - j) ** 2 / (k - 1) ** 2 for j in range(k)]
            for i in range(k)
        ])
        
        row_marginal = conf_matrix.sum(axis=1) / n
        col_marginal = conf_matrix.sum(axis=0) / n
        expected = np.outer(row_marginal, col_marginal)
        
        po = 1 - (weights * conf_matrix / n).sum()
        pe = 1 - (weights * expected).sum()
        
        return float(po / pe) if pe != 0 else 1.0
    
    def protocol_level_agreement(
        self,
        auto_scores: List[int],
        human_scores: List[int],
        items_per_protocol: int = 20,
    ) -> Dict:
        """
        Compute agreement at protocol level (total score across multiple items).
        
        EIT typically has 20 items × 6 points = 120 point total.
        Target: Within 10 points (8.3%) of human rater.
        
        Args:
            auto_scores: Per-item automated scores
            human_scores: Per-item human scores
            items_per_protocol: Items per EIT protocol (default: 20)
        
        Returns:
            Protocol-level agreement metrics
        """
        n_protocols = len(auto_scores) // items_per_protocol
        if n_protocols == 0:
            return {"error": "Not enough scores for protocol-level analysis"}
        
        protocol_diffs = []
        for i in range(n_protocols):
            start = i * items_per_protocol
            end = start + items_per_protocol
            auto_total = sum(auto_scores[start:end])
            human_total = sum(human_scores[start:end])
            protocol_diffs.append(abs(auto_total - human_total))
        
        diffs = np.array(protocol_diffs)
        
        return {
            "n_protocols": len(diffs),
            "max_total_points": items_per_protocol * self.max_score,  # 120
            "mean_protocol_diff": float(np.mean(diffs)),
            "std_protocol_diff": float(np.std(diffs)),
            "max_protocol_diff": int(np.max(diffs)),
            "within_10_pts": float(np.mean(diffs <= 10)),
            "within_5_pct": float(np.mean(diffs <= items_per_protocol * self.max_score * 0.05)),
        }


class ScoringEnsemble:
    """
    ML-based fallback for low-confidence rule-based scores.
    
    When the rule-based rubric engine produces a score with low confidence
    (near decision boundaries), this ensemble can provide additional signal
    by combining multiple scoring heuristics.
    """
    
    def __init__(self):
        """Initialize ensemble with multiple scoring heuristics."""
        self.heuristics = [
            self._heuristic_jaccard,
            self._heuristic_subsequence,
            self._heuristic_semantic,
        ]
    
    def predict(
        self,
        hypothesis: str,
        reference: str,
        rule_score: int,
        rule_confidence: float,
        max_score: int = 6,
    ) -> Tuple[int, float]:
        """
        Combine rule-based and heuristic scores.
        
        Args:
            hypothesis: Learner's response
            reference: Target sentence
            rule_score: Score from rule-based engine
            rule_confidence: Confidence (0-1) of rule-based score
            max_score: Maximum possible score
        
        Returns:
            (final_score, confidence)
        """
        if rule_confidence >= 0.85:
            # High confidence in rule-based score
            return rule_score, rule_confidence
        
        # Low confidence: consult ensemble
        scores = [self._scale_score(h(hypothesis, reference), max_score) for h in self.heuristics]
        scores.append(rule_score)  # Include rule score
        
        # Weighted average (weight rule score less if low confidence)
        weights = [0.2, 0.2, 0.2, 0.4 * rule_confidence]  # Rule gets less weight
        final_score = np.average(scores, weights=weights)
        
        # Return rounded score with ensemble confidence
        return int(round(final_score)), rule_confidence * 0.5 + 0.5
    
    def _heuristic_jaccard(self, hypothesis: str, reference: str) -> float:
        """Jaccard similarity (0-1 mapped to 0-6)."""
        hyp_tokens = set(hypothesis.lower().split())
        ref_tokens = set(reference.lower().split())
        if not ref_tokens:
            return 0.0
        intersection = len(hyp_tokens & ref_tokens)
        union = len(hyp_tokens | ref_tokens)
        return intersection / max(union, 1)
    
    def _heuristic_subsequence(self, hypothesis: str, reference: str) -> float:
        """Longest common subsequence ratio."""
        hyp = hypothesis.lower().split()
        ref = reference.lower().split()
        lcs_len = self._lcs_length(hyp, ref)
        max_len = max(len(hyp), len(ref))
        return lcs_len / max(max_len, 1)
    
    def _heuristic_semantic(self, hypothesis: str, reference: str) -> float:
        """
        Simple semantic similarity (content word overlap).
        """
        function_words = {
            "el", "la", "los", "las", "un", "una", "de", "a", "en",
            "con", "por", "para", "que", "y", "o", "pero", "se",
        }
        
        hyp_content = [w for w in hypothesis.lower().split() if w not in function_words]
        ref_content = [w for w in reference.lower().split() if w not in function_words]
        
        if not ref_content:
            return 1.0
        
        overlap = len(set(hyp_content) & set(ref_content))
        return overlap / len(ref_content)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def _scale_score(self, value: float, max_score: int) -> float:
        """Scale [0, 1] value to [0, max_score]."""
        return max(0, min(max_score, value * max_score))
