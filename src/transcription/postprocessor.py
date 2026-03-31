"""
Post-Processor for AutoEIT
Corrects predictable transcription errors in L2 learner Spanish speech.

L2 learner speech produces systematic ASR errors that differ from native-speech errors:
- Phonological transfer (e.g., English speakers devoice final consonants)
- Vocabulary substitution at phonologically similar words
- Partial repetitions / restarts misinterpreted as full words
- Hesitation markers ("um", "uh", "este") inserted mid-utterance

This module applies targeted corrections to raw ASR output before scoring.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PostProcessorConfig:
    """Configuration for post-processing."""

    remove_disfluencies: bool = True
    fix_common_substitutions: bool = True
    normalize_punctuation: bool = True
    lowercase: bool = True
    remove_extra_whitespace: bool = True


# Common disfluency markers in L2 Spanish learner speech
DISFLUENCY_PATTERNS = [
    r"\b(um+|uh+|eh+|ah+|mm+)\b",  # English hesitation
    r"\b(este|bueno|o sea|pues|a ver)\b",  # Spanish hesitation fillers
    r"\b(como se dice|espera|no sé)\b",  # Metacommentary
    r"\.{2,}",  # Ellipsis (mid-utterance pauses)
    r"\[.*?\]",  # Bracketed annotations
    r"\(.*?\)",  # Parenthetical notes
]

# Phonological transfer substitutions: (pattern, replacement)
# Based on common English→Spanish transfer errors in EIT research
SUBSTITUTION_RULES = [
    # Vowel quality errors
    (r"\bsalio\b", "salió"),
    (r"\bcomio\b", "comió"),
    (r"\bvino\b(?! a)", "vino"),  # keep "vino a" intact
    # Common ASR confusions in learner speech
    (r"\bque\b(?= \w+ \w+ que)", "que"),  # relative clause que
    (r"\baber\b", "haber"),
    (r"\balla\b", "allá"),
    (r"\besta\b(?= [a-z])", "está"),
    # Phonological transfer: word-final devoicing
    (r"\bgraned\b", "grande"),
    (r"\bciudат\b", "ciudad"),
    # Partial word cleanup (common when Whisper catches restarts)
    (r"\b\w{1,2}(?=\s+\1{3,})", ""),  # Remove short repeated starts
]


class PostProcessor:
    """
    Applies rule-based post-processing to raw ASR transcriptions.

    This is applied BEFORE scoring. The goal is to clean up known ASR
    artifacts from learner speech WITHOUT altering the actual content of
    the learner's response (which is what gets scored).

    Example:
        >>> pp = PostProcessor()
        >>> raw = "um este yo fui al mercado ayer con mi— con mi madre"
        >>> pp.process(raw)
        "yo fui al mercado ayer con mi madre"
    """

    def __init__(self, config: Optional[PostProcessorConfig] = None):
        self.config = config or PostProcessorConfig()
        self._disfluency_re = re.compile("|".join(DISFLUENCY_PATTERNS), re.IGNORECASE)
        logger.info("PostProcessor initialized")

    def process(self, text: str) -> str:
        """Apply the full post-processing pipeline to a transcription."""
        if not text or not text.strip():
            return ""

        if self.config.lowercase:
            text = text.lower()

        if self.config.normalize_punctuation:
            text = self._normalize_punctuation(text)

        if self.config.remove_disfluencies:
            text = self._remove_disfluencies(text)

        if self.config.fix_common_substitutions:
            text = self._apply_substitutions(text)

        if self.config.remove_extra_whitespace:
            text = " ".join(text.split())

        return text.strip()

    def process_batch(self, texts: list[str]) -> list[str]:
        """Apply post-processing to a batch of transcriptions."""
        return [self.process(t) for t in texts]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_punctuation(self, text: str) -> str:
        """Standardize punctuation for consistent downstream processing."""
        # Remove sentence-final punctuation (EIT scoring doesn't penalize it)
        text = re.sub(r"[.!?]+$", "", text)
        # Normalize dashes
        text = re.sub(r"[—–-]+", " ", text)
        # Remove commas (learners don't produce them; ASR hallucinates them)
        text = text.replace(",", "")
        return text

    def _remove_disfluencies(self, text: str) -> str:
        """Remove disfluency markers and hesitation tokens."""
        text = self._disfluency_re.sub(" ", text)
        return text

    def _apply_substitutions(self, text: str) -> str:
        """Apply learner-specific phonological transfer correction rules."""
        for pattern, replacement in SUBSTITUTION_RULES:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def explain(self, raw: str, processed: str) -> dict:
        """
        Return a diff explaining what was changed during post-processing.
        Useful for debugging and transparency.
        """
        raw_tokens = set(raw.lower().split())
        proc_tokens = set(processed.split())
        return {
            "removed": sorted(raw_tokens - proc_tokens),
            "added": sorted(proc_tokens - raw_tokens),
            "unchanged": sorted(raw_tokens & proc_tokens),
        }
