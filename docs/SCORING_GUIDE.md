# AutoEIT Scoring Guide - Technical Reference

Complete reference for the EIT automated scoring engine.

## Table of Contents

1. [Architecture](#architecture)
2. [Scoring Scale & Rubric](#scoring-scale--rubric)
3. [Rubric Engine](#rubric-engine)
4. [Validator Framework](#validator-framework)
5. [Ensemble System](#ensemble-system)
6. [Pipeline](#pipeline)
7. [CLI Reference](#cli-reference)
8. [API Reference](#api-reference)
9. [Error Detection](#error-detection)
10. [Performance Tuning](#performance-tuning)
11. [Best Practices](#best-practices)

## Architecture

### Overview

```
Input (Hypothesis + Reference)
      ↓
[EITRubricEngine] (Rule-based scoring)
      ↓
      ├─ Confidence >= 0.85? → Output Score ✓
      │
      └─ Confidence < 0.85? → [ScoringEnsemble]
                                ├─ Jaccard Heuristic
                                ├─ LCS Heuristic
                                └─ Semantic Heuristic
                                      ↓
                                [Final Score]
      ↓
[ScoringValidator] (Validation metrics)
      ├─ Item-level agreement (90% target)
      └─ Protocol-level agreement (<10 pts variance)
```

### Components

| Component | Purpose | File |
|-----------|---------|------|
| **EITRubricEngine** | Rule-based scoring with NLP features | `src/scoring/rubric.py` |
| **ScoringValidator** | Item + protocol level agreement measurement | `src/scoring/validator.py` |
| **ScoringEnsemble** | ML ensemble for low-confidence cases | `src/scoring/validator.py` |
| **ScoringPipeline** | Orchestrates all components | `src/scoring/pipeline.py` |
| **CLI** | Command-line interface for batch processing | `scripts/run_scoring.py` |

## Scoring Scale & Rubric

### 0-6 Point Scale

```python
class ScoreCategory(Enum):
    ZERO = 0    # No response / unintelligible
    ONE = 1     # Vocabulary errors dominate
    TWO = 2     # Vocabulary blocks comprehension
    THREE = 3   # Structure + vocabulary issues
    FOUR = 4    # Minor errors, mostly correct
    FIVE = 5    # Near-perfect, small omissions
    SIX = 6     # Perfect or near-perfect
```

### Scoring Decision Process

```
1. Normalize both strings (lowercase, punctuation removal)
2. Tokenize and compute overlap metrics
3. Check for specific error patterns:
   - Structural errors (missing/extra function words)
   - Lexical errors (content word mismatches)
   - Verb conjugation errors
   - Omissions (words in reference but not hypothesis)
   - Additions (words in hypothesis but not reference)
4. Assign raw score based on overlap & errors
5. Compute confidence based on decision clarity
6. If confidence < 0.85: Consult ensemble
7. Return: score, category, confidence, reasoning, errors
```

### Rules Summary

| Condition | Score | Notes |
|-----------|-------|-------|
| Empty hypothesis | 0 | No response |
| Vocabulary-only errors | 1-2 | Content words missing |
| Vocabulary + structure errors | 3 | Comprehensible but flawed |
| Minor errors | 4 | Mostly correct grammatically |
| Small omissions | 5 | Near-perfect production |
| Perfect or near-perfect | 6 | Matches reference closely |

## Rubric Engine

### Class: EITRubricEngine

```python
from src.scoring.rubric import EITRubricEngine, RubricDecision

rubric = EITRubricEngine()
result = rubric.score(hypothesis, reference)
```

### RubricDecision (Output)

```python
@dataclass
class RubricDecision:
    score: int                  # 0-6 points
    reasoning: str             # Human-readable explanation
    normalized_score: float    # 0.0 to 1.0 scale
    confidence: float          # 0.0 to 1.0 confidence level
    errors: List[str]          # List of detected errors
    category: ScoreCategory    # Named category
    is_perfect: bool          # True if score == 6
    is_zero: bool             # True if score == 0
```

### Key Methods

#### `score(hypothesis: str, reference: str) → RubricDecision`

Main scoring method.

```python
result = rubric.score("yo fui", "yo fui al mercado")
print(f"Score: {result.score}/6")
print(f"Errors: {result.errors}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Reasoning: {result.reasoning}")
```

**Scoring Process:**
1. Normalize both strings (case, punctuation)
2. Tokenize into words
3. Compute Jaccard similarity, recall, precision
4. Detect structural/lexical errors
5. Map metrics → score using weighted rules
6. Estimate confidence
7. Assign category name

#### `_normalize_text(text: str) → str`

Normalizes input text for comparison:
- Convert to lowercase
- Remove punctuation
- Strip extra whitespace
- Remove accents (optional)

```python
normalized = rubric._normalize_text("¡Yo Fui Al Mercado!")
# "yo fui al mercado"
```

#### `_compute_token_metrics(hyp_tokens, ref_tokens) → dict`

Computes overlap statistics:

```python
metrics = rubric._compute_token_metrics(
    ["yo", "fui"],
    ["yo", "fui", "al", "mercado"]
)
# {
#   "overlap": 2,
#   "jaccard": 0.5,
#   "recall": 1.0,
#   "precision": 0.5
# }
```

**Metrics:**
- `overlap`: Exact token matches
- `jaccard`: (intersection) / (union) 
- `recall`: overlap / len(reference)
- `precision`: overlap / len(hypothesis)

#### `_detect_error_patterns() → List[str]`

Detects specific error types:

1. **Structural Errors:**
   - Missing/extra function words (articles, prepositions)
   - Conjugation issues
   - Deletion/insertion > 50%

2. **Lexical Errors:**
   - Content word mismatches
   - Spelling variations

3. **Omissions:**
   - Words in reference but not hypothesis

4. **Additions:**
   - Extra words not in reference

```python
result = rubric.score("yo fui", "yo fui al mercado ayer")
# errors: ["Omission: ayer (adverb)", "Omission: al"]
```

### Configuration

```python
from src.scoring.rubric import RubricConfig

config = RubricConfig(
    use_verb_groups=True,        # Spanish verb conjugation checking
    min_content_word_overlap=0.5, # Minimum required overlap
    structural_error_weight=0.3,  # Structural error penalty
    confidence_threshold=0.85     # Ensemble trigger
)

rubric = EITRubricEngine(config)
```

## Validator Framework

### Class: ScoringValidator

Measures scoring consistency against human baseline.

```python
from src.scoring.validator import ScoringValidator

validator = ScoringValidator()
```

#### Item-Level Agreement

Compares individual item scores:

```python
auto_scores = [6, 4, 3, 2, 1]
human_scores = [6, 4, 3, 2, 1]

metrics = validator.validate_against_baseline(auto_scores, human_scores)

print(metrics)
# {
#   "exact_agreement": 1.0,
#   "mean_abs_diff": 0.0,
#   "within_1_point": 1.0,
#   "within_2_points": 1.0,
#   "kappa": 1.0,
#   "problem_cases": []
# }
```

**Metrics:**
- `exact_agreement`: Fraction of exact matches (90% target)
- `mean_abs_diff`: Mean absolute difference (lower is better)
- `within_N_point`: Fraction within N points
- `kappa`: Cohen's weighted kappa for ordinal scale (0.0 to 1.0)
- `problem_cases`: Cases with >2 point difference

#### Protocol-Level Agreement

Compares total EIT scores (20 items × 6 pts = 120 total):

```python
# 2 protocols × 20 items = 40 scores
auto_scores = [6, 4, 3, ...] * 2  # 40 items
human_scores = [6, 4, 3, ...] * 2

result = validator.protocol_level_agreement(
    auto_scores, 
    human_scores, 
    items_per_protocol=20
)

print(result)
# {
#   "n_protocols": 2,
#   "protocol_totals_auto": [120, 115],
#   "protocol_totals_human": [120, 118],
#   "protocol_diffs": [0, 3],
#   "mean_protocol_diff": 1.5,
#   "within_10_pts": 1.0,  # All protocols within 10-pt margin
#   "within_5_pct": 0.5     # 50% within 5% of total
# }
```

**Protocol Metrics:**
- `n_protocols`: Number of protocols scored
- `protocol_diffs`: Per-protocol score difference
- `mean_protocol_diff`: Average difference across protocols
- `within_10_pts`: Fraction of protocols within 10-point margin
- `within_5_pct`: Fraction within 5% of reference total

### Cohen's Kappa

Computes weighted kappa for ordinal scale (0-6):

```python
auto = [0, 1, 2, 3, 4, 5, 6] * 10
human = [0, 1, 2, 3, 4, 5, 6] * 10

kappa = validator._compute_kappa(auto, human)
# 1.0 (perfect agreement)
```

**Interpretation:**
- `< 0.20`: Poor agreement
- `0.20-0.40`: Fair agreement
- `0.40-0.60`: Moderate agreement
- `0.60-0.80`: Substantial agreement ← Target for robust scoring
- `0.80-1.00`: Almost perfect agreement ← Excellent!

## Ensemble System

### Class: ScoringEnsemble

Provides ML-based fallback when confidence is low (<0.85).

```python
from src.scoring.validator import ScoringEnsemble

ensemble = ScoringEnsemble()
```

### Prediction Method

```python
final_score, final_confidence = ensemble.predict(
    hypothesis="yo fui",
    reference="yo fui al mercado",
    rule_score=4,
    rule_confidence=0.7
)
# (4, 0.75)  # Ensemble combines rule + heuristics
```

### Three Heuristics

#### 1. Jaccard Similarity

**Token-level set similarity:**

```
Jaccard = |intersection| / |union|
        = 2 / 4 = 0.5
```

```python
sim = ensemble._heuristic_jaccard("yo fui", "yo fui al mercado")
# 0.5
```

Captures: Overall word overlap

#### 2. Longest Common Subsequence (LCS)

**Preserves word order, penalizes reordering:**

```
LCS("yo fui", "yo fui al mercado") = "yo fui"
LCS_ratio = len(LCS) / max(len(hyp), len(ref))
           = 2 / 4 = 0.5
```

```python
sim = ensemble._heuristic_subsequence("yo fui", "yo fui al mercado")
# 0.5
```

Captures: Structural preservation, word order

#### 3. Semantic Overlap

**Content word overlap (verbs, nouns, adjectives):**

```python
sim = ensemble._heuristic_semantic("yo fui", "yo fui al mercado")
# 0.66 (2 content words match / 3 in reference)
```

Captures: Meaningful word coverage

### Combination Strategy

When confidence < 0.85:

```python
final_score = weighted_average([
    jaccard_score,
    lcs_score,
    semantic_score,
    rule_score
], weights=[0.2, 0.2, 0.2, 0.4 * rule_confidence])

# Rule score has higher weight but is discounted by confidence
```

**Effect:** 
- High-confidence rule scores (0.85+) dominate
- Low-confidence rule scores (0.3-0.5) weighted equally with heuristics
- Confidence adjusted downward slightly after ensemble

## Pipeline

### Class: ScoringPipeline

Orchestrates rubric → ensemble → validation.

```python
from src.scoring.pipeline import ScoringPipeline

pipeline = ScoringPipeline(use_ensemble=True)
```

### Single Item Scoring

```python
result = pipeline.score(
    hypothesis="yo fui",
    reference="yo fui al mercado"
)

print(result.score)        # 4-5
print(result.confidence)   # 0.75-0.85
print(result.reasoning)    # Explanation
```

### Batch Scoring (CSV)

```python
import pandas as pd

# Prepare CSV
df = pd.DataFrame({
    'hypothesis': ['yo fui', 'mi hermana'],
    'reference': ['yo fui al mercado', 'mi hermana estudia']
})
df.to_csv('input.csv', index=False)

# Score
output_df = pipeline.score_csv('input.csv', 'output.csv')

print(output_df)
#    hypothesis           reference  auto_score category  ...
# 0  yo fui       yo fui al mercado           4     FOUR  ...
```

### Batch Scoring (JSON)

```python
sentences = [
    {"hypothesis": "yo fui", "reference": "yo fui al mercado"},
    {"hypothesis": "mi hermana", "reference": "mi hermana estudia"}
]

results = pipeline.score_batch_sentences(sentences)

for r in results:
    print(f"{r['score']}/6: {r['reasoning']}")
```

### Validation Integration

```python
# Score against baseline
auto_scores = df['auto_score'].tolist()
human_scores = df['human_score'].tolist()

# Item-level validation
item_metrics = pipeline.validate_against_human(auto_scores, human_scores)
print(f"Exact agreement: {item_metrics['exact_agreement']:.1%}")

# Protocol-level validation (20 items per protocol)
protocol_metrics = pipeline.protocol_agreement(
    auto_scores, 
    human_scores, 
    items_per_protocol=20
)
print(f"Within 10 pts: {protocol_metrics['within_10_pts']:.1%}")
```

## CLI Reference

### Command 1: score

**Score transcriptions (CSV)**

```bash
python scripts/run_scoring.py score \
  --input-csv transcriptions.csv \
  --output-csv scored.csv \
  [--use-ensemble]
```

**Options:**
- `--input-csv`: Input CSV with hypothesis, reference columns
- `--output-csv`: Output CSV with scores
- `--use-ensemble`: Enable ensemble for low-confidence cases (default: disabled)

**Input CSV:**
```csv
hypothesis,reference
"yo fui","yo fui al mercado"
```

**Output CSV:**
```csv
hypothesis,reference,auto_score,category,reasoning,confidence
"yo fui","yo fui al mercado",4,"FOUR","Missing adverb",0.82
```

### Command 2: validate

**Validate against human baseline**

```bash
python scripts/run_scoring.py validate \
  --scored-csv student_vs_human.csv \
  --output-json validation.json \
  [--items-per-protocol 20]
```

**Options:**
- `--scored-csv`: CSV with auto_score and human_score columns
- `--output-json`: Output validation report
- `--items-per-protocol`: Items per protocol (default: 20)

**Input CSV:**
```csv
hypothesis,reference,auto_score,human_score
"yo fui","yo fui al mercado",4,4
```

**Output JSON:**
```json
{
  "item_level": {
    "exact_agreement": 0.90,
    "mean_abs_diff": 0.42,
    "kappa": 0.88,
    "within_1_point": 0.98
  },
  "protocol_level": {
    "n_protocols": 5,
    "within_10_pts": 1.0
  },
  "summary": {
    "meets_goal": true
  }
}
```

### Command 3: compare-raters

**Inter-rater agreement analysis**

```bash
python scripts/run_scoring.py compare-raters \
  --rater1-scores rater1.csv \
  --rater2-scores rater2.csv \
  --output-json agreement.json
```

**Output:**
```json
{
  "exact_agreement": 0.85,
  "mean_absolute_diff": 0.6,
  "kappa": 0.82,
  "agreement_breakdown": {
    "exact": 85,
    "within_1pt": 98,
    "within_2pts": 100
  }
}
```

### Command 4: score-batch

**Batch JSON processing**

```bash
python scripts/run_scoring.py score-batch \
  --input-json batch.json \
  --output-json results.json
```

**Input JSON:**
```json
[
  {"hypothesis": "yo fui", "reference": "yo fui al mercado"}
]
```

**Output JSON:**
```json
{
  "scores": [...],
  "distribution": {
    "0": 0,
    "1": 2,
    "2": 5,
    "3": 15,
    "4": 28,
    "5": 35,
    "6": 15
  }
}
```

## API Reference

### EITRubricEngine

```python
from src.scoring.rubric import EITRubricEngine, RubricConfig, ScoreCategory

# Create with defaults
rubric = EITRubricEngine()

# Create with custom config
config = RubricConfig(
    use_verb_groups=True,
    min_content_word_overlap=0.5,
    structural_error_weight=0.3,
    confidence_threshold=0.85
)
rubric = EITRubricEngine(config)

# Score
result = rubric.score(hypothesis="yo fui", reference="yo fui al mercado")

# Access result fields
print(f"Score: {result.score}")           # 0-6
print(f"Category: {result.category}")     # ScoreCategory enum
print(f"Confidence: {result.confidence}") # 0.0-1.0
print(f"Reasoning: {result.reasoning}")   # str
print(f"Errors: {result.errors}")         # List[str]
```

### ScoringValidator

```python
from src.scoring.validator import ScoringValidator

validator = ScoringValidator()

# Item-level validation
item_metrics = validator.validate_against_baseline(
    auto_scores=[6, 4, 3],
    human_scores=[6, 4, 3]
)
# Returns: dict with exact_agreement, mean_abs_diff, kappa, etc.

# Protocol-level validation
protocol_metrics = validator.protocol_level_agreement(
    auto_scores=[6, 4, 3, 2, 1] * 4,      # 20 items
    human_scores=[6, 4, 3, 2, 1] * 4,
    items_per_protocol=20
)
# Returns: dict with protocol totals, within_10_pts, etc.
```

### ScoringEnsemble

```python
from src.scoring.validator import ScoringEnsemble

ensemble = ScoringEnsemble()

# Get prediction
score, confidence = ensemble.predict(
    hypothesis="yo fui",
    reference="yo fui al mercado",
    rule_score=4,
    rule_confidence=0.7
)
```

### ScoringPipeline

```python
from src.scoring.pipeline import ScoringPipeline

# Create with optional ensemble
pipeline = ScoringPipeline(use_ensemble=True)

# Single item
result = pipeline.score("yo fui", "yo fui al mercado")

# CSV batch
output_df = pipeline.score_csv('input.csv', 'output.csv')

# JSON batch
results = pipeline.score_batch_sentences(json_sentences)

# Validation
item_metrics = pipeline.validate_against_human(auto_scores, human_scores)
protocol_metrics = pipeline.protocol_agreement(auto_scores, human_scores, 20)
```

## Error Detection

### Error Types

1. **Structural Errors** - Function word issues (articles, prepositions, conjunctions)
   ```
   "fui mercado" → Missing "al" (preposition)
   ```

2. **Lexical Errors** - Content word issues (verbs, nouns, adjectives)
   ```
   "yo fue" → Wrong verb conjugation (fue instead of fui)
   ```

3. **Omissions** - Words in reference but not hypothesis
   ```
   "yo fui" vs "yo fui al mercado" → Missing "al mercado"
   ```

4. **Additions** - Words in hypothesis not in reference
   ```
   "yo fui mucho" vs "yo fui" → Extra "mucho"
   ```

### Accessing Error Info

```python
result = rubric.score("yo fui", "yo fui al mercado")

for error in result.errors:
    print(error)
    # "Omission: al (preposition)"
    # "Omission: mercado (noun)"
    # "Omission: ayer (adverb)"
```

## Performance Tuning

### Throughput Optimization

**CPU Mode (baseline):**
```bash
# ~100 sentences/second
python scripts/run_scoring.py score \
  --input-csv large_batch.csv \
  --output-csv results.csv
```

**GPU Mode (if CUDA available):**
```bash
export CUDA_VISIBLE_DEVICES=0
# ~500 sentences/second
```

### Memory Usage

- **Per instance:** ~2 MB (minimal)
- **Batch processing:** Scales linearly
- **Typical batch (1000 items):** ~2-5 MB

### Latency

- **Warmup:** ~500ms (first request)
- **Per item:** <10ms (CPU) or <5ms (GPU)
- **Batch of 1000:** ~5-10 seconds

### Optimization Strategies

1. **Batch Processing:** Use `score_csv()` or `score_batch_sentences()` instead of looping `.score()`
2. **GPU:** Enable GPU if available (5x speedup)
3. **Ensemble Reduction:** Disable ensemble for high-confidence scenarios
4. **Caching:** For repeated hypothesis-reference pairs, cache results

## Best Practices

### Data Preparation

1. **Clean input:** Remove extra whitespace, normalize encoding
   ```python
   def clean_text(s):
       import unicodedata
       # Normalize unicode (é → e)
       s = unicodedata.normalize('NFD', s)
       s = s.encode('ascii', 'ignore').decode('utf-8')
       return s.strip()
   ```

2. **Handle edge cases:** Empty strings, very long sequences
   ```python
   if not hypothesis or not reference:
       return 0  # No response
   ```

3. **Consistent format:** Both hypothesis and reference should be complete sentences

### Validation Workflow

1. **Prepare baseline:** Get 20-30 items scored by 2+ human raters
2. **Score with system:** Use `score` command on baseline items
3. **Compare:** Use `validate` command against human scores
4. **Analyze:** Check exact_agreement ≥ 90%, kappa ≥ 0.80
5. **Iterate:** If below target, review error cases

### Interpreting Metrics

```python
metrics = validator.validate_against_baseline(auto, human)

# Good indicators:
# - exact_agreement >= 0.90 ✓
# - mean_abs_diff <= 0.5 ✓
# - kappa >= 0.80 ✓
# - within_1_point >= 0.95 ✓

# Problem indicators:
# - exact_agreement < 0.85: Systematic bias or rubric misalignment
# - mean_abs_diff > 1.0: Wide variance from human scores
# - kappa < 0.60: Low ordinal agreement
# - Clusters in problem_cases: Specific error types missed
```

### Troubleshooting Low Accuracy

**Symptom:** Low exact_agreement (<85%)

**Diagnosis:**
- Check mean_abs_diff: If high (>1.0), bias issue
- Review problem_cases: Are they systematic?
- Compare kappa: If low, ordinal scale issue

**Solutions:**
1. Review scoring rules for that score level
2. Check human rater agreement (inter-rater reliability)
3. Examine example cases where system disagrees most
4. Consider ensemble with custom weights

**Example:**
```python
# Identify where system disagrees most
disagreements = []
for auto, human in zip(auto_scores, human_scores):
    if auto != human:
        disagreements.append((auto, human, abs(auto-human)))

# Sort by magnitude
disagreements.sort(key=lambda x: x[2], reverse=True)

# Review top 10
for auto, human, diff in disagreements[:10]:
    print(f"System: {auto}/6, Human: {human}/6 (diff: {diff})")
```

### Continuous Improvement

1. **Monthly validation:** Re-validate against new human baseline
2. **Error tracking:** Log all cases with >1 point difference
3. **Rubric updates:** Refine scoring rules based on error patterns
4. **Ensemble tuning:** Adjust heuristic weights
5. **Version control:** Track all changes to rubric/ensemble

