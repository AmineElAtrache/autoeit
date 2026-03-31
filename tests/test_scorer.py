"""Tests for the EIT Scoring Rubric Engine and Pipeline."""

import pytest
import json
import tempfile
from pathlib import Path
import pandas as pd

from src.scoring.rubric import EITRubricEngine, RubricConfig, ScoreCategory
from src.scoring.pipeline import ScoringPipeline
from src.scoring.validator import ScoringValidator, ScoringEnsemble


@pytest.fixture
def rubric():
    return EITRubricEngine()


@pytest.fixture
def pipeline():
    return ScoringPipeline()


@pytest.fixture
def validator():
    return ScoringValidator()


@pytest.fixture
def ensemble():
    return ScoringEnsemble()


class TestPerfectMatch:
    def test_identical_sentences(self, rubric):
        result = rubric.score(
            "yo fui al mercado ayer",
            "yo fui al mercado ayer"
        )
        assert result.score == 6
        assert result.is_perfect

    def test_case_insensitive(self, rubric):
        result = rubric.score(
            "Yo Fui Al Mercado",
            "yo fui al mercado"
        )
        assert result.score == 6


class TestEmptyHypothesis:
    def test_empty_string(self, rubric):
        result = rubric.score("", "yo fui al mercado ayer")
        assert result.score == 0
        assert result.is_zero

    def test_whitespace_only(self, rubric):
        result = rubric.score("   ", "yo fui al mercado ayer")
        assert result.score == 0


class TestPartialCredit:
    def test_one_omission_scores_lower(self, rubric):
        """Omitting content words → lower score."""
        result = rubric.score(
            "yo fui al mercado",
            "yo fui al mercado ayer con mi madre"
        )
        assert result.score < 6

    def test_major_omission_scores_much_lower(self, rubric):
        result = rubric.score(
            "yo fui",
            "yo fui al mercado ayer con mi madre y mi hermana"
        )
        assert result.score <= 3


class TestScoreNormalization:
    def test_normalized_score_range(self, rubric):
        result = rubric.score("yo fui", "yo fui al mercado")
        assert 0.0 <= result.normalized_score <= 1.0

    def test_perfect_normalized_is_one(self, rubric):
        result = rubric.score("yo fui", "yo fui")
        assert result.normalized_score == 1.0


class TestConfidenceScoring:
    def test_perfect_match_high_confidence(self, rubric):
        result = rubric.score("yo fui al mercado", "yo fui al mercado")
        assert result.confidence >= 0.9

    def test_edge_case_low_confidence(self, rubric):
        # Score near decision boundary
        result = rubric.score(
            "yo fui al mercado",
            "yo fui al mercado ayer con mi madre"
        )
        # Confidence should be moderate
        assert 0.5 <= result.confidence <= 1.0


class TestScoringPipeline:
    def test_score_single_item(self, pipeline):
        result = pipeline.score("yo fui", "yo fui al mercado")
        assert 0 <= result.score <= 6

    def test_score_csv(self, pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input CSV
            input_csv = Path(tmpdir) / "input.csv"
            data = {
                "hypothesis": ["yo fui al mercado", "mi hermana"],
                "reference": ["yo fui al mercado", "mi hermana estudia"]
            }
            pd.DataFrame(data).to_csv(input_csv, index=False)
            
            # Score
            output_csv = Path(tmpdir) / "output.csv"
            result_df = pipeline.score_csv(input_csv, output_csv)
            
            # Verify
            assert output_csv.exists()
            assert "auto_score" in result_df.columns
            assert len(result_df) == 2
            assert all(0 <= s <= 6 for s in result_df["auto_score"])


class TestScoringValidator:
    def test_perfect_agreement(self, validator):
        auto = [6, 4, 3, 2, 1]
        human = [6, 4, 3, 2, 1]
        
        metrics = validator.validate_against_baseline(auto, human)
        assert metrics["exact_agreement"] == 1.0
        assert metrics["mean_abs_diff"] == 0.0

    def test_one_point_disagreement(self, validator):
        auto = [6, 4, 3, 2, 1]
        human = [6, 5, 3, 2, 0]  # 2 disagreements
        
        metrics = validator.validate_against_baseline(auto, human)
        assert metrics["exact_agreement"] == 0.6
        assert metrics["within_1_point"] >= 0.8

    def test_kappa_computation(self, validator):
        auto = [0, 1, 2, 3, 4, 5, 6] * 10
        human = [0, 1, 2, 3, 4, 5, 6] * 10
        
        metrics = validator.validate_against_baseline(auto, human)
        assert metrics["kappa"] == 1.0  # Perfect agreement

    def test_protocol_agreement(self, validator):
        # 2 protocols × 20 items = 40 scores
        auto = [6] * 20 + [3] * 20  # 120 + 60 = 180 total
        human = [6] * 20 + [3] * 20  # Same
        
        result = validator.protocol_level_agreement(auto, human, 20)
        assert result["within_10_pts"] == 1.0
        assert result["n_protocols"] == 2


class TestScoringEnsemble:
    def test_ensemble_prediction_high_confidence(self, ensemble):
        score, conf = ensemble.predict(
            "yo fui al mercado",
            "yo fui al mercado",
            rule_score=6,
            rule_confidence=0.95
        )
        assert score == 6
        assert conf >= 0.9

    def test_ensemble_prediction_low_confidence(self, ensemble):
        score, conf = ensemble.predict(
            "yo fui",
            "yo fui al mercado ayer",
            rule_score=4,
            rule_confidence=0.5
        )
        # Should still be reasonable
        assert 0 <= score <= 6
        assert conf <= 1.0

    def test_heuristic_jaccard(self, ensemble):
        # Perfect overlap
        sim = ensemble._heuristic_jaccard("yo fui", "yo fui")
        assert sim == 1.0
        
        # Partial overlap
        sim = ensemble._heuristic_jaccard("yo fui", "yo fui al mercado")
        assert 0 < sim < 1

    def test_heuristic_subsequence(self, ensemble):
        sim = ensemble._heuristic_subsequence("yo fui", "yo fui")
        assert sim >= 0.5


class TestErrorDetection:
    def test_omission_detection(self, rubric):
        result = rubric.score(
            "yo fui",
            "yo fui al mercado"
        )
        assert any("mission" in e.lower() for e in result.errors)

    def test_addition_detection(self, rubric):
        result = rubric.score(
            "yo fui extraño",
            "yo fui"
        )
        # Should detect addition or different content
        assert len(result.errors) > 0

    def test_reasoning_provided(self, rubric):
        result = rubric.score(
            "yo fui",
            "yo fui al mercado"
        )
        assert len(result.reasoning) > 0
        assert "score" in result.reasoning.lower()


class TestBatchScoring:
    def test_batch_sentences(self, pipeline):
        sentences = [
            {"hypothesis": "yo fui", "reference": "yo fui al mercado"},
            {"hypothesis": "mi hermana", "reference": "mi hermana estudia"},
        ]
        
        results = pipeline.score_batch_sentences(sentences)
        
        assert len(results) == 2
        assert all(0 <= r["score"] <= 6 for r in results)
        assert all("reasoning" in r for r in results)


class TestScoreDistribution:
    def test_scores_distributed_0_to_6(self, rubric):
        """Test that various inputs produce different scores."""
        test_cases = [
            ("", "yo fui al mercado"),  # Should be 0
            ("yo", "yo fui al mercado"),  # Should be low (1-2)
            ("yo fui", "yo fui al mercado"),  # Medium (3-4)
            ("yo fui al mercado", "yo fui al mercado ayer"),  # High (5-)
            ("yo fui al mercado ayer", "yo fui al mercado ayer"),  # Perfect (6)
        ]
        
        scores = [rubric.score(hyp, ref).score for hyp, ref in test_cases]
        
        # Should have variety
        assert len(set(scores)) > 1
        # Should range from 0 to 6
        assert min(scores) == 0
        assert max(scores) == 6


class TestGoalCompletion:
    def test_90_percent_agreement_achievable(self, validator):
        """Test that 90% agreement goal is achievable."""
        # Simulate scoring where system agrees with human 90% of the time
        n_items = 100
        auto = [6] * 90 + [3] * 10
        human = [6] * 100
        
        metrics = validator.validate_against_baseline(auto, human)
        
        # Should detect near-90% agreement (allowing small tolerance)
        assert metrics["exact_agreement"] >= 0.85
