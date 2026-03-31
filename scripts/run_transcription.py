"""
CLI for AutoEIT transcription pipeline.
Supports batch transcription, evaluation, and model fine-tuning.
"""

import click
import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src.transcription.pipeline import TranscriptionPipeline
from src.transcription.trainer import WhisperEITTrainer, TrainerConfig, EITAudioDataset
from src.utils.metrics import word_error_rate, percent_agreement

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """AutoEIT Transcription CLI"""
    pass


@cli.command()
@click.option(
    "--audio-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing audio files (.wav, .mp3, .flac)",
)
@click.option(
    "--output-csv",
    type=click.Path(),
    required=True,
    help="Output CSV with transcriptions",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    default=None,
    help="Path to fine-tuned model checkpoint (LoRA adapter)",
)
def transcribe(audio_dir: str, output_csv: str, checkpoint: Optional[str]):
    """
    Batch transcribe audio files.
    
    Example:
        python scripts/run_transcription.py transcribe \\
            --audio-dir /path/to/audio \\
            --output-csv results.csv \\
            --checkpoint checkpoints/whisper-eit-v2
    """
    logger.info(f"Transcribing audio from {audio_dir}")
    
    pipeline = TranscriptionPipeline()
    if checkpoint:
        logger.info(f"Loading fine-tuned model from {checkpoint}")
        pipeline = TranscriptionPipeline.from_pretrained(checkpoint)
    
    df = pipeline.transcribe_batch(audio_dir, output_csv)
    logger.info(f"Results saved to {output_csv}")
    logger.info(f"\nSummary:\n{df.head()}")


@cli.command()
@click.option(
    "--results-csv",
    type=click.Path(exists=True),
    required=True,
    help="CSV with 'transcription' and 'reference' columns",
)
@click.option(
    "--output-json",
    type=click.Path(),
    required=True,
    help="Output JSON with evaluation metrics",
)
def evaluate(results_csv: str, output_json: str):
    """
    Evaluate transcription quality against human references.
    
    Computes WER and percent agreement.
    
    Example:
        python scripts/run_transcription.py evaluate \\
            --results-csv transcriptions.csv \\
            --output-json eval_metrics.json
    """
    logger.info(f"Evaluating transcriptions from {results_csv}")
    
    df = pd.read_csv(results_csv)
    
    if "transcription" not in df.columns or "reference" not in df.columns:
        raise ValueError("CSV must have 'transcription' and 'reference' columns")
    
    hypotheses = df["transcription"].tolist()
    references = df["reference"].tolist()
    
    # Compute metrics
    wer = word_error_rate(hypotheses, references)
    agreement = percent_agreement(hypotheses, references)
    
    metrics = {
        "wer": float(wer),
        "agreement": float(agreement),
        "samples": len(hypotheses),
        "goal_wer": 0.10,
        "meets_target": wer <= 0.10,
    }
    
    # Save results
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\n=== Evaluation Results ===")
    logger.info(f"WER: {wer:.2%} (target: ≤10%)")
    logger.info(f"Exact Agreement: {agreement:.2%}")
    logger.info(f"Samples: {len(hypotheses)}")
    logger.info(f"Results saved to {output_json}")


@cli.command()
@click.option(
    "--train-audio-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory with training audio files",
)
@click.option(
    "--train-manifest",
    type=click.Path(exists=True),
    required=True,
    help="JSON manifest with training samples (audio, reference)",
)
@click.option(
    "--eval-manifest",
    type=click.Path(exists=True),
    default=None,
    help="JSON manifest with validation samples (optional)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="checkpoints/whisper-eit-LoRA",
    help="Output directory for fine-tuned model",
)
@click.option(
    "--epochs",
    type=int,
    default=3,
    help="Number of training epochs",
)
@click.option(
    "--batch-size",
    type=int,
    default=4,
    help="Batch size per device",
)
def train(
    train_audio_dir: str,
    train_manifest: str,
    eval_manifest: Optional[str],
    output_dir: str,
    epochs: int,
    batch_size: int,
):
    """
    Fine-tune Whisper-large-v3 on learner speech using LoRA.
    
    Example:
        python scripts/run_transcription.py train \\
            --train-audio-dir /path/to/train/audio \\
            --train-manifest data/train_manifest.json \\
            --eval-manifest data/eval_manifest.json \\
            --output-dir checkpoints/whisper-eit-v2 \\
            --epochs 3 \\
            --batch-size 4
    """
    logger.info("=== Starting Whisper Fine-tuning ===")
    
    config = TrainerConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
    )
    
    trainer = WhisperEITTrainer(config)
    trainer.load_model_and_processor()
    
    # Load datasets
    train_dataset = EITAudioDataset(
        train_audio_dir,
        train_manifest,
        trainer.processor,
    )
    
    eval_dataset = None
    if eval_manifest:
        eval_dataset = EITAudioDataset(
            train_audio_dir,
            eval_manifest,
            trainer.processor,
        )
    
    # Train
    metrics = trainer.train(train_dataset, eval_dataset)
    
    # Save model
    trainer.save_model(output_dir)
    
    logger.info(f"Training complete. Model saved to {output_dir}")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")


@cli.command()
@click.option(
    "--input-dir",
    type=click.Path(),
    required=True,
    help="Directory with audio files",
)
@click.option(
    "--output-manifest",
    type=click.Path(),
    required=True,
    help="Output JSON manifest",
)
def create_manifest(input_dir: str, output_manifest: str):
    """
    Create a data manifest for training/evaluation.
    
    Scans input directory for audio files and creates a JSON manifest.
    You must manually add 'reference' fields for training.
    
    Manifest format:
        [
            {"audio": "file1.wav", "reference": "target sentence"},
            {"audio": "file2.wav", "reference": "another sentence"}
        ]
    
    Example:
        python scripts/run_transcription.py create-manifest \\
            --input-dir data/audio \\
            --output-manifest data/manifest.json
    """
    input_dir = Path(input_dir)
    audio_files = sorted(
        list(input_dir.glob("*.wav")) +
        list(input_dir.glob("*.mp3")) +
        list(input_dir.glob("*.flac"))
    )
    
    manifest = [
        {
            "audio": f.name,
            "reference": "[ADD REFERENCE TEXT HERE]"
        }
        for f in audio_files
    ]
    
    with open(output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created manifest with {len(manifest)} samples")
    logger.info(f"Manifest saved to {output_manifest}")
    logger.info("⚠️  Please edit the manifest and add reference transcriptions")


if __name__ == "__main__":
    cli()
