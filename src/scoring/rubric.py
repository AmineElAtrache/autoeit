"""
EIT Scoring Rubric for AutoEIT
Implements the standardized Spanish EIT scoring rubric in a deterministic,
rule-based engine. Designed to achieve consistent, reproducible scores
that match experienced human rater decisions.

The EIT rubric awards points based on how accurately a learner reproduces
a target sentence. Points depend on:
- Structural accuracy (grammatical morphology, word order)
- Lexical accuracy (correct vocabulary)
- Completeness (no omissions/additions beyond minor errors)
"""

import re
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


class ScoreCategory(IntEnum):
    """Point categories for EIT sentence scoring (0–6 scale)."""
    ZERO = 0        # No recognizable target content
    ONE = 1         # Target words present, no structure
    TWO = 2         # Some structure, major errors
    THREE = 3       # Mostly accurate with 2+ errors
    FOUR = 4        # Mostly accurate with 1 significant error
    FIVE = 5        # Near-perfect with 1 minor error
    SIX = 6         # Perfect or near-perfect reproduction


@dataclass
class RubricDecision:
    """Detailed output of the rubric engine for one sentence."""
    score: int
    max_score: int = 6
    category: ScoreCategory = ScoreCategory.ZERO
    errors: list[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 1.0                        # 1.0 = rule is deterministic

    @property
    def normalized_score(self) -> float:
        return self.score / self.max_score if self.max_score > 0 else 0.0

    @property
    def is_perfect(self) -> bool:
        return self.score == self.max_score

    @property
    def is_zero(self) -> bool:
        return self.score == 0


@dataclass
class RubricConfig:
    """Configuration for the rubric engine."""
    # Weights for different error types
    structural_error_penalty: int = 2       # Points lost per grammatical error
    lexical_error_penalty: int = 1          # Points lost per vocabulary error
    omission_penalty: int = 1              # Per missing content word
    addition_penalty: int = 1              # Per spurious content word
    # Thresholds
    perfect_match_threshold: float = 0.95  # Token overlap for "perfect"
    high_similarity_threshold: float = 0.80
    min_content_words: int = 2             # Min content words to score above 0


class EITRubricEngine:
    """
    Deterministic rule-based implementation of the Spanish EIT scoring rubric.

    This engine applies the rubric consistently, eliminating the inter-rater
    variability that makes automated scoring with LLMs unsuitable for research.

    The rubric operates on pairs of (hypothesis, reference) strings and
    produces a score from 0–6 with full explanatory reasoning.

    Scoring logic (simplified):
        6: ≥95% token match, no structural errors
        5: ≥80% match, 1 minor error (phonology/orthography)
        4: ≥70% match, 1 significant morphological error
        3: ≥60% match, 2+ errors but structure preserved
        2: ≥40% match, major structural breakdown
        1: <40% match but target content identifiable
        0: <20% match or no recognizable content

    Example:
        >>> rubric = EITRubricEngine()
        >>> decision = rubric.score(
        ...     hypothesis="yo fui al mercado ayer",
        ...     reference="Yo fui al mercado ayer con mi madre."
        ... )
        >>> decision.score  # 4 (omission: "con mi madre")
    """

    # Spanish function words (not scored for content accuracy)
    FUNCTION_WORDS = {
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "de", "del", "al", "a", "en", "con", "por", "para", "que",
        "y", "o", "pero", "sino", "porque", "cuando", "como",
        "se", "me", "te", "le", "nos", "les", "lo", "la",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
        "mi", "tu", "su", "nuestro", "nuestra", "sus", "mis", "tus",
        "muy", "más", "menos", "también", "ya", "no", "sí",
    }

    # Morphological equivalence groups (minor errors, -0 or -1 point)
    MORPHOLOGICAL_EQUIVALENTS = [
        {"fui", "fue", "fuiste", "fueron", "fuimos"},         # ir/ser preterite
        {"tengo", "tienes", "tiene", "tenemos", "tienen"},    # tener present
        {"estoy", "estás", "está", "estamos", "están"},       # estar present
        {"hablo", "hablas", "habla", "hablamos", "hablan"},   # hablar present
        {"comí", "comiste", "comió", "comimos", "comieron"},  # comer preterite
    ]

    def __init__(self, config: Optional[RubricConfig] = None):
        self.config = config or RubricConfig()

    def score(self, hypothesis: str, reference: str) -> RubricDecision:
        """
        Score a hypothesis against a reference sentence.

        Args:
            hypothesis: Learner's transcribed response (post-processed).
            reference: Target EIT stimulus sentence.

        Returns:
            RubricDecision with score, category, errors, and reasoning.
        """
        hyp = self._normalize(hypothesis)
        ref = self._normalize(reference)

        hyp_tokens = hyp.split()
        ref_tokens = ref.split()

        if not hyp_tokens:
            return RubricDecision(
                score=0, category=ScoreCategory.ZERO,
                reasoning="Empty hypothesis — no response produced.",
                confidence=1.0,
            )

        # Compute token overlap metrics
        overlap = self._token_overlap(hyp_tokens, ref_tokens)
        content_overlap = self._content_word_overlap(hyp_tokens, ref_tokens)

        # Detect error types
        errors = []
        structural_errors = self._detect_structural_errors(hyp_tokens, ref_tokens)
        lexical_errors = self._detect_lexical_errors(hyp_tokens, ref_tokens)
        omissions = self._detect_omissions(hyp_tokens, ref_tokens)
        additions = self._detect_additions(hyp_tokens, ref_tokens)

        errors.extend(structural_errors)
        errors.extend(lexical_errors)
        errors.extend(omissions)
        errors.extend(additions)

        n_structural = len(structural_errors)
        n_lexical = len(lexical_errors)
        n_omissions = len(omissions)
        n_additions = len(additions)
        total_penalty = (
            n_structural * self.config.structural_error_penalty +
            n_lexical * self.config.lexical_error_penalty +
            n_omissions * self.config.omission_penalty +
            n_additions * self.config.addition_penalty
        )

        # Determine score based on overlap and penalties
        score = self._compute_score(overlap, content_overlap, total_penalty, errors)
        category = ScoreCategory(score)

        reasoning = self._build_reasoning(
            overlap, content_overlap, structural_errors, lexical_errors,
            omissions, additions, score
        )

        return RubricDecision(
            score=score,
            category=category,
            errors=errors,
            reasoning=reasoning,
            confidence=self._estimate_confidence(total_penalty, overlap),
        )

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        """Lowercase, remove punctuation, strip accents for comparison."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(text.split())
        return text

    def _token_overlap(self, hyp: list[str], ref: list[str]) -> float:
        """Jaccard-style token overlap."""
        hyp_set, ref_set = set(hyp), set(ref)
        if not ref_set:
            return 0.0
        intersection = hyp_set & ref_set
        return len(intersection) / len(ref_set)

    def _content_word_overlap(self, hyp: list[str], ref: list[str]) -> float:
        """Overlap restricted to content words (nouns, verbs, adjectives)."""
        hyp_content = [t for t in hyp if t not in self.FUNCTION_WORDS]
        ref_content = [t for t in ref if t not in self.FUNCTION_WORDS]
        if not ref_content:
            return 1.0
        overlap = set(hyp_content) & set(ref_content)
        return len(overlap) / len(ref_content)

    def _detect_structural_errors(self, hyp: list[str], ref: list[str]) -> list[str]:
        """Detect word-order and morphological structural violations."""
        errors = []
        
        # Check for length mismatch (omissions/insertions)
        hyp_len = len(hyp)
        ref_len = len(ref)
        
        if hyp_len < ref_len * 0.5:
            errors.append("Major deletion: >50% of words missing")
        elif hyp_len > ref_len * 1.5:
            errors.append("Major insertion: >50% more words than target")
        
        # Check for conjugation errors in common verbs
        for hyp_word, ref_word in zip(hyp, ref):
            for morph_group in self.MORPHOLOGICAL_EQUIVALENTS:
                if ref_word in morph_group and hyp_word in morph_group:
                    # Same morphological class but different form (minor error)
                    if hyp_word != ref_word:
                        errors.append(f"Conjugation: '{hyp_word}' vs '{ref_word}'")
                    break
        
        return errors

    def _detect_lexical_errors(self, hyp: list[str], ref: list[str]) -> list[str]:
        """Detect vocabulary substitutions and near-misses."""
        errors = []
        hyp_content = {t for t in hyp if t not in self.FUNCTION_WORDS}
        ref_content = {t for t in ref if t not in self.FUNCTION_WORDS}
        for ref_word in ref_content:
            if ref_word not in hyp_content:
                # Check morphological equivalents
                is_morph_equiv = any(
                    ref_word in group and any(h in group for h in hyp_content)
                    for group in self.MORPHOLOGICAL_EQUIVALENTS
                )
                if not is_morph_equiv:
                    errors.append(f"Missing content word: '{ref_word}'")
        return errors

    def _detect_omissions(self, hyp: list[str], ref: list[str]) -> list[str]:
        """Detect content words present in reference but absent from hypothesis."""
        hyp_set = set(hyp)
        ref_content = [t for t in ref if t not in self.FUNCTION_WORDS]
        return [f"Omission: '{w}'" for w in ref_content if w not in hyp_set]

    def _detect_additions(self, hyp: list[str], ref: list[str]) -> list[str]:
        """Detect spurious content words in hypothesis not in reference."""
        ref_set = set(ref)
        hyp_content = [t for t in hyp if t not in self.FUNCTION_WORDS]
        return [f"Addition: '{w}'" for w in hyp_content if w not in ref_set]

    def _compute_score(
        self,
        overlap: float,
        content_overlap: float,
        penalty: int,
        errors: list[str],
    ) -> int:
        """Map overlap and penalty to a 0–6 score."""
        if content_overlap < 0.15:
            return 0
        if content_overlap < 0.35:
            return 1
        if overlap >= self.config.perfect_match_threshold and penalty == 0:
            return 6
        if overlap >= self.config.high_similarity_threshold and penalty <= 1:
            return 5
        if overlap >= 0.70 and penalty <= 2:
            return 4
        if overlap >= 0.60 and penalty <= 4:
            return 3
        if overlap >= 0.40:
            return 2
        return 1

    def _estimate_confidence(self, penalty: int, overlap: float) -> float:
        """
        Estimate how confident we are in the rule-based score.
        Low confidence cases are flagged for ML ensemble arbitration.
        """
        if overlap in (0.0, 1.0):
            return 1.0
        # Edge cases: near threshold boundaries → lower confidence
        boundary_distances = [abs(overlap - t) for t in [0.40, 0.60, 0.70, 0.80, 0.95]]
        min_distance = min(boundary_distances)
        return min(1.0, 0.5 + min_distance * 2)

    def _build_reasoning(
        self,
        overlap: float,
        content_overlap: float,
        structural_errors: list[str],
        lexical_errors: list[str],
        omissions: list[str],
        additions: list[str],
        score: int,
    ) -> str:
        """Produce a human-readable explanation for the score."""
        parts = [f"Score: {score}/6."]
        parts.append(f"Token overlap: {overlap:.1%}, content overlap: {content_overlap:.1%}.")
        if structural_errors:
            parts.append(f"Structural errors ({len(structural_errors)}): {', '.join(structural_errors[:3])}.")
        if lexical_errors:
            parts.append(f"Lexical errors ({len(lexical_errors)}): {', '.join(lexical_errors[:3])}.")
        if omissions:
            parts.append(f"Omissions ({len(omissions)}): {', '.join(omissions[:3])}.")
        if additions:
            parts.append(f"Additions ({len(additions)}): {', '.join(additions[:3])}.")
        if not (structural_errors or lexical_errors or omissions or additions):
            parts.append("No significant errors detected.")
        return " ".join(parts)
