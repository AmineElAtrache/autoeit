"""End-to-end scoring pipeline."""

from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict
import json
from .rubric import EITRubricEngine, RubricConfig
from .validator import ScoringValidator, ScoringEnsemble


class ScoringPipeline:
    def __init__(self, rubric_path=None, use_ensemble: bool = False):
        self.rubric = EITRubricEngine()
        self.validator = ScoringValidator()
        self.ensemble = ScoringEnsemble() if use_ensemble else None

    def score(self, hypothesis: str, reference: str):
        """Score a single hypothesis against reference."""
        result = self.rubric.score(hypothesis, reference)

        # Optionally use ensemble for low-confidence cases
        if self.ensemble and result.confidence < 0.85:
            final_score, final_conf = self.ensemble.predict(
                hypothesis, reference, result.score, result.confidence
            )
            result.score = final_score
            result.confidence = final_conf

        return result

    def score_csv(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        hyp_col="hypothesis",
        ref_col="reference",
    ) -> pd.DataFrame:
        """Score transcriptions from CSV."""
        df = pd.read_csv(input_csv)
        results = [self.score(row[hyp_col], row[ref_col]) for _, row in df.iterrows()]

        df["auto_score"] = [r.score for r in results]
        df["reasoning"] = [r.reasoning for r in results]
        df["confidence"] = [r.confidence for r in results]
        df["category"] = [r.category.name for r in results]

        df.to_csv(output_csv, index=False)
        return df

    def score_batch_sentences(self, sentences: List[Dict]) -> List[Dict]:
        """
        Score a batch of (hypothesis, reference) pairs.

        Args:
            sentences: List of dicts with 'hypothesis' and 'reference' keys

        Returns:
            List of dicts with scoring results
        """
        results = []
        for sent in sentences:
            score_result = self.score(sent["hypothesis"], sent.get("reference", ""))
            results.append(
                {
                    "hypothesis": sent["hypothesis"],
                    "reference": sent.get("reference", ""),
                    "score": score_result.score,
                    "confidence": score_result.confidence,
                    "reasoning": score_result.reasoning,
                    "errors": score_result.errors,
                }
            )
        return results

    def validate_against_human(
        self,
        auto_scores: List[int],
        human_scores: List[int],
    ) -> Dict:
        """Validate automatic scores against human baseline."""
        return self.validator.validate_against_baseline(auto_scores, human_scores)

    def protocol_agreement(
        self,
        auto_scores: List[int],
        human_scores: List[int],
        items_per_protocol: int = 20,
    ) -> Dict:
        """Compute protocol-level agreement (total EIT score)."""
        return self.validator.protocol_level_agreement(
            auto_scores, human_scores, items_per_protocol
        )
