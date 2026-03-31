"""
CLI for AutoEIT scoring pipeline.
Supports batch scoring, validation, and inter-rater agreement analysis.
"""

import click
import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src.scoring.pipeline import ScoringPipeline
from src.scoring.validator import ScoringValidator
from src.utils.metrics import (
    cohens_kappa,
    percent_agreement,
    score_difference_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """AutoEIT Scoring CLI"""
    pass


@cli.command()
@click.option(
    "--input-csv",
    type=click.Path(exists=True),
    required=True,
    help="CSV with 'hypothesis' and 'reference' columns",
)
@click.option(
    "--output-csv",
    type=click.Path(),
    required=True,
    help="Output CSV with scores",
)
@click.option(
    "--use-ensemble",
    is_flag=True,
    default=False,
    help="Use ML ensemble for low-confidence scores",
)
def score(input_csv: str, output_csv: str, use_ensemble: bool):
    """
    Score transcriptions using the EIT rubric engine.
    
    Input CSV format:
        hypothesis,reference
        "yo fui","yo fui al mercado ayer"
        "mi hermana","mi hermana estudia en la universidad"
    
    Example:
        python scripts/run_scoring.py score \\
            --input-csv transcriptions.csv \\
            --output-csv scored_results.csv
    """
    logger.info(f"Scoring transcriptions from {input_csv}")
    
    pipeline = ScoringPipeline(use_ensemble=use_ensemble)
    df = pipeline.score_csv(input_csv, output_csv)
    
    logger.info(f"Results saved to {output_csv}")
    logger.info(f"\nSample results:")
    print(df[["hypothesis", "reference", "auto_score", "confidence"]].head(10).to_string())


@cli.command()
@click.option(
    "--scored-csv",
    type=click.Path(exists=True),
    required=True,
    help="CSV with 'auto_score' and 'human_score' columns",
)
@click.option(
    "--output-json",
    type=click.Path(),
    required=True,
    help="Output JSON with validation metrics",
)
@click.option(
    "--items-per-protocol",
    type=int,
    default=20,
    help="Items per EIT protocol (default: 20)",
)
def validate(scored_csv: str, output_json: str, items_per_protocol: int):
    """
    Validate scoring consistency against human baseline.
    
    Input CSV format (must have actual human scores):
        auto_score,human_score,hypothesis,reference
        6,6,"yo fui al mercado","yo fui al mercado"
        4,4,"mi hermana","mi hermana estudia en la universidad"
    
    Example:
        python scripts/run_scoring.py validate \\
            --scored-csv scored_with_human.csv \\
            --output-json validation.json
    """
    logger.info(f"Validating scores from {scored_csv}")
    
    df = pd.read_csv(scored_csv)
    
    if "auto_score" not in df.columns or "human_score" not in df.columns:
        raise ValueError("CSV must have 'auto_score' and 'human_score' columns")
    
    auto_scores = df["auto_score"].tolist()
    human_scores = df["human_score"].tolist()
    
    # Compute item-level metrics
    validator = ScoringValidator()
    item_metrics = validator.validate_against_baseline(auto_scores, human_scores)
    
    # Compute protocol-level metrics
    protocol_metrics = validator.protocol_level_agreement(
        auto_scores, human_scores, items_per_protocol
    )
    
    # Combine results
    results = {
        "item_level": item_metrics,
        "protocol_level": protocol_metrics,
        "summary": {
            "meets_goal": item_metrics["exact_agreement"] >= 0.90,
            "protocol_accuracy": protocol_metrics.get("within_10_pts", 0),
        }
    }
    
    # Save
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n=== Validation Results ===")
    logger.info(f"Exact Agreement: {item_metrics['exact_agreement']:.1%} (goal: ≥90%)")
    logger.info(f"Mean Absolute Difference: {item_metrics['mean_abs_diff']:.2f}")
    logger.info(f"Cohen's Kappa: {item_metrics['kappa']:.3f}")
    
    if "protocol_level" in results:
        logger.info(f"\nProtocol-Level (20 items × 6 = 120 pts):")
        logger.info(f"  Within 10 pts: {protocol_metrics['within_10_pts']:.1%}")
        logger.info(f"  Mean difference: {protocol_metrics['mean_protocol_diff']:.1f} pts")
    
    logger.info(f"Results saved to {output_json}")


@cli.command()
@click.option(
    "--input-csv",
    type=click.Path(exists=True),
    required=True,
    help="CSV with transcriptions and references",
)
@click.option(
    "--output-json",
    type=click.Path(),
    required=True,
    help="Output JSON with comparison",
)
@click.option(
    "--scorer-a",
    type=str,
    default="auto",
    help="Name of first rater/system",
)
@click.option(
    "--scorer-b",
    type=str,
    default="human",
    help="Name of second rater/system",
)
def compare_raters(input_csv: str, output_json: str, scorer_a: str, scorer_b: str):
    """
    Compare scoring consistency between two raters/systems.
    
    Input CSV format:
        hypothesis,reference,scorer_a_score,scorer_b_score
    
    Example:
        python scripts/run_scoring.py compare-raters \\
            --input-csv scoring_comparison.csv \\
            --output-json comparison.json \\
            --scorer-a "auto" \\
            --scorer-b "human"
    """
    logger.info(f"Comparing {scorer_a} vs {scorer_b}")
    
    df = pd.read_csv(input_csv)
    
    # Validate columns
    col_a = f"{scorer_a}_score"
    col_b = f"{scorer_b}_score"
    
    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError(f"CSV must have '{col_a}' and '{col_b}' columns")
    
    scores_a = df[col_a].tolist()
    scores_b = df[col_b].tolist()
    
    # Compute agreement
    validator = ScoringValidator()
    metrics = validator.validate_against_baseline(scores_a, scores_b)
    
    results = {
        "comparison": {
            "rater_a": scorer_a,
            "rater_b": scorer_b,
        },
        "metrics": metrics,
    }
    
    # Save
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info(f"\n{scorer_a} vs {scorer_b}:")
    logger.info(f"  Exact agreement: {metrics['exact_agreement']:.1%}")
    logger.info(f"  Mean absolute difference: {metrics['mean_abs_diff']:.2f}")
    logger.info(f"  Cohen's Kappa: {metrics['kappa']:.3f}")
    logger.info(f"  Results saved to {output_json}")


@cli.command()
@click.option(
    "--sentences-json",
    type=click.Path(exists=True),
    required=True,
    help="JSON with list of {hypothesis, reference} objects",
)
@click.option(
    "--output-json",
    type=click.Path(),
    required=True,
    help="Output JSON with scores",
)
def score_batch(sentences_json: str, output_json: str):
    """
    Score a batch of sentences from JSON.
    
    JSON format:
        [
            {"hypothesis": "yo fui", "reference": "yo fui al mercado"},
            {"hypothesis": "mi hermana", "reference": "mi hermana estudia"}
        ]
    
    Example:
        python scripts/run_scoring.py score-batch \\
            --sentences-json sentences.json \\
            --output-json scored.json
    """
    logger.info(f"Scoring batch from {sentences_json}")
    
    with open(sentences_json, "r") as f:
        sentences = json.load(f)
    
    pipeline = ScoringPipeline()
    results = pipeline.score_batch_sentences(sentences)
    
    # Save
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    logger.info(f"Scored {len(results)} sentences")
    logger.info(f"Results saved to {output_json}")
    
    # Summary
    scores = [r["score"] for r in results]
    logger.info(f"\nScore distribution:")
    for i in range(7):
        count = scores.count(i)
        pct = 100 * count / len(scores)
        logger.info(f"  Score {i}: {count:3d} ({pct:5.1f}%)")


if __name__ == "__main__":
    cli()
