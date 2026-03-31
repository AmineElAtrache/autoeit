"""
Fine-tuning trainer for Whisper-large-v3 on learner speech.
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Any
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for Whisper fine-tuning."""
    model_id: str = "openai/whisper-large-v3"
    language: str = "es"
    task: str = "transcribe"
    
    # LoRA config
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    
    # Training hyperparameters
    output_dir: str = "checkpoints/whisper-eit-LoRA"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Hardware
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Validation & checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    
    # Logging
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])


class EITAudioDataset(Dataset):
    """Dataset for EIT learner audio with reference transcriptions."""
    
    def __init__(
        self,
        audio_dir: Path,
        manifest_file: Path,
        processor: WhisperProcessor,
        max_duration_s: float = 30.0,
    ):
        """
        Args:
            audio_dir: Directory containing audio files.
            manifest_file: JSON manifest with entries:
                [{"audio": "file.wav", "reference": "target text"}, ...]
            processor: WhisperProcessor for feature extraction.
            max_duration_s: Maximum audio duration in seconds.
        """
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.max_duration_samples = int(max_duration_s * 16000)
        
        with open(manifest_file, "r") as f:
            self.samples = json.load(f)
        
        logger.info(f"Loaded {len(self.samples)} samples from {manifest_file}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        audio_path = self.audio_dir / sample["audio"]
        reference = sample["reference"]
        
        try:
            # Load audio at 16kHz
            waveform, sr = torchaudio.load(str(audio_path))
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            waveform = waveform.squeeze(0).numpy()
            
            # Truncate to max duration
            if len(waveform) > self.max_duration_samples:
                waveform = waveform[:self.max_duration_samples]
            
            # Process audio to mel-features
            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
            )
            
            # Tokenize reference
            labels = self.processor.tokenizer(reference).input_ids
            
            return {
                "input_features": inputs.input_features.squeeze(0),
                "labels": torch.tensor(labels),
            }
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return silence
            return {
                "input_features": torch.zeros((80, 3000)),
                "labels": torch.tensor([self.processor.tokenizer.eos_token_id]),
            }


class WhisperEITTrainer:
    """Trainer for fine-tuning Whisper on EIT learner data."""
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.model = None
        self.processor = None
        self.trainer = None
    
    def load_model_and_processor(self) -> None:
        """Load base Whisper model and processor."""
        logger.info(f"Loading model: {self.config.model_id}")
        
        self.processor = WhisperProcessor.from_pretrained(self.config.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
        )
        
        # Freeze base model
        self.model.encoder.requires_grad_(False)
        self.model.decoder.model.requires_grad_(False)
        
        # Apply LoRA
        if self.config.use_lora:
            logger.info("Applying LoRA fine-tuning configuration")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
        
        self.model.to(self.config.device)
        logger.info(f"Model loaded on {self.config.device}")
    
    def train(
        self,
        train_dataset: EITAudioDataset,
        eval_dataset: Optional[EITAudioDataset] = None,
    ) -> Dict[str, float]:
        """
        Fine-tune the model on the provided dataset.
        
        Args:
            train_dataset: Training dataset.
            eval_dataset: Validation dataset (optional).
        
        Returns:
            Training metrics dictionary.
        """
        if self.model is None:
            self.load_model_and_processor()
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            evaluation_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            logging_dir="./logs",
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics if eval_dataset else None,
        )
        
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        return train_result.metrics
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute WER for validation."""
        from jiwer import compute_wer
        
        predictions, labels = eval_pred
        
        # Decode predictions and references
        pred_ids = np.argmax(predictions, axis=-1)
        label_ids = labels
        
        pred_str = self.processor.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = self.processor.batch_decode(
            label_ids, skip_special_tokens=True
        )
        
        wer = compute_wer(label_str, pred_str)
        
        return {"wer": wer}
    
    def save_model(self, output_dir: str | Path) -> None:
        """Save the fine-tuned model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(output_dir))
            logger.info(f"Model saved to {output_dir}")
        else:
            logger.error("Model does not support save_pretrained")
    
    def merge_and_unload(self) -> None:
        """Merge LoRA weights into base model and unload adapter."""
        if hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
            logger.info("LoRA weights merged")
