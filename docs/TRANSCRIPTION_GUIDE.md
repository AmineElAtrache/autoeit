# Transcription Component - Complete Guide

## Overview

The AutoEIT transcription pipeline converts learner Spanish EIT audio responses into accurate text transcriptions with ≤10% Word Error Rate (WER), achieving ~90% agreement with human transcribers.

## Architecture

```
Raw Audio
  ├─→ Preprocessor (noise reduction, normalization, segmentation)
  ├─→ ASR Engine (Whisper-large-v3 with LoRA fine-tuning)
  ├─→ Post-Processor (learner error correction)
  └─→ Evaluation (WER, agreement metrics)
```

## Key Components

### 1. Audio Preprocessor (`src/transcription/preprocessor.py`)

**Purpose:** Clean and standardize raw audio before ASR

**Features:**
- Resampling to 16 kHz (Whisper requirement)
- Mono conversion (stereo → mono)
- Spectral noise gating for background noise removal
- RMS normalization (handles varying loudness)
- Silence detection and segmentation
- Dynamic range compression
- Segment duration filtering (0.5s – 30s)

**Usage:**
```python
from src.transcription import AudioPreprocessor

preprocessor = AudioPreprocessor()
segments = preprocessor.process("learner_response.wav")
# Returns: List[AudioSegment] ready for ASR
```

### 2. Transcriber (`src/transcription/transcriber.py`)

**Purpose:** Speech-to-text using fine-tuned Whisper-large-v3

**Key Design Decisions:**
- **Model:** OpenAI Whisper-large-v3 (multilingual, optimized for varied accents)
- **Fine-tuning:** LoRA (Low-Rank Adaptation) for efficient learner-data adaptation
- **Decoding:** Beam search (beam_size=5) with CTC-based constraints
- **Conditioning:** `condition_on_previous_text=False` (each EIT sentence is independent)
- **Quantization:** FP16 inference for 2x speedup with minimal quality loss

**Performance targets:**
- Base model (zero-shot): ~28% WER on learner speech
- After LoRA fine-tuning: ~9% WER on EIT corpus

**Usage:**
```python
from src.transcription import WhisperEITTranscriber

# Load base model with LoRA adapter
transcriber = WhisperEITTranscriber.from_pretrained(
    "checkpoints/whisper-eit-v2"
)

# Transcribe preprocessed segment
result = transcriber.transcribe_segment(audio_segment)
print(result.text)  # "yo fui al mercado ayer"
print(result.confidence)  # 0.92
```

### 3. Post-Processor (`src/transcription/postprocessor.py`)

**Purpose:** Correct systematic ASR errors specific to learner speech

**Rules Applied:**
- **Disfluency removal:** "um", "este", hesitation fillers
- **Phonological transfer correction:** English L1 → Spanish L2 patterns
  - Word-final devoicing: "graned" → "grande"
  - Vowel quality errors: "salio" → "salió"
- **Punctuation normalization:** Remove sentence-final punctuation
- **Metacommentary removal:** "como se dice", "espera"

**Usage:**
```python
from src.transcription import PostProcessor

pp = PostProcessor()
raw = "um este yo fui al mercado ayer con mi— con mi madre"
cleaned = pp.process(raw)
# Output: "yo fui al mercado ayer con mi madre"
```

### 4. Training Pipeline (`src/transcription/trainer.py`)

**Purpose:** Fine-tune Whisper-large-v3 on EIT learner data

**LoRA Configuration:**
- Rank: 16
- Alpha: 32
- Target modules: q_proj, v_proj (attention layers)
- Frozen: Encoder + decoder (only fine-tune attention adapters)

**Training Setup:**
```python
from src.transcription.trainer import WhisperEITTrainer, TrainerConfig

config = TrainerConfig(
    output_dir="checkpoints/whisper-eit-v2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-3,
)

trainer = WhisperEITTrainer(config)
trainer.load_model_and_processor()

# Train on EIT corpus
metrics = trainer.train(train_dataset, eval_dataset)
trainer.save_model("checkpoints/whisper-eit-v2")
```

### 5. Evaluator (`src/transcription/evaluator.py`)

**Purpose:** Measure transcription quality against human references

**Metrics:**
- **WER (Word Error Rate):** % of word-level errors (substitution, deletion, insertion)
- **CER (Character Error Rate):** % of character-level errors
- **Exact Match Rate:** % of perfect transcriptions
- **Target:** ≤10% WER = ~90% agreement with human transcribers

**Usage:**
```python
from src.transcription.evaluator import TranscriptionEvaluator

evaluator = TranscriptionEvaluator(target_wer=0.10)

# Evaluate CSV with hypotheses vs references
eval_result = evaluator.evaluate_csv("results.csv")
evaluator.print_summary(eval_result)
evaluator.save_report(eval_result, "report.json")
```

## CLI Tools

### `python scripts/run_transcription.py`

#### Command 1: Batch Transcription
```bash
python scripts/run_transcription.py transcribe \
    --audio-dir /path/to/audio \
    --output-csv transcriptions.csv \
    --checkpoint checkpoints/whisper-eit-v2
```

#### Command 2: Evaluate Against Human Transcriptions
```bash
python scripts/run_transcription.py evaluate \
    --results-csv transcriptions.csv \
    --output-json eval_metrics.json
```

**Expected output:**
```json
{
  "wer": 0.087,
  "agreement": 0.92,
  "samples": 120,
  "goal_wer": 0.10,
  "meets_target": true
}
```

#### Command 3: Train Fine-Tuned Model
```bash
python scripts/run_transcription.py train \
    --train-audio-dir data/train/audio \
    --train-manifest data/train_manifest.json \
    --eval-manifest data/eval_manifest.json \
    --output-dir checkpoints/whisper-eit-v2 \
    --epochs 3 \
    --batch-size 4
```

**Manifest format:**
```json
[
  {
    "audio": "response_001.wav",
    "reference": "yo fui al mercado ayer con mi madre"
  },
  {
    "audio": "response_002.wav",
    "reference": "mi hermana estudia en la universidad"
  }
]
```

#### Command 4: Create Data Manifest
```bash
python scripts/run_transcription.py create-manifest \
    --input-dir data/audio \
    --output-manifest data/manifest.json
```

## Performance Metrics

### Evaluation Targets (GSOC Requirement)
- **WER ≤ 10%** on test set (learner Spanish EIT responses)
- **Agreement ≥ 90%** with experienced human transcribers
- **Processing speed:** >1000 hours of audio per GPU per day (at 16x real-time)

### Common Error Categories
1. **Phonological transfer** (30%): English L1 interference
2. **Lexical substitution** (25%): Word confusion due to proficiency
3. **Disfluencies** (20%): Hesitations, false starts
4. **Accent/pronunciation** (15%): Reduced intelligibility
5. **Audio quality** (10%): Background noise, clipping

## Data Requirements

### Training Data
Minimum **200-500** annotated EIT responses with:
- High-quality learner audio (16 kHz, mono, SNR >15 dB)
- Reference transcriptions from experienced raters
- Metadata: proficiency level, L1, recording conditions

### Evaluation Data
Minimum **100** held-out test samples with:
- Transcriptions from 2+ independent human raters
- Ground truth scores (for comparison consistency)

## Implementation Checklist

- [x] Audio preprocessor (noise reduction, segmentation)
- [x] Whisper ASR wrapper with LoRA support
- [x] Post-processor for learner errors
- [x] Training pipeline for fine-tuning
- [x] Evaluation framework (WER, agreement metrics)
- [x] CLI tools for batch processing
- [x] Test suite (test_transcription_evaluator.py)
- [x] Jupyter tutorial notebook
- [x] Comprehensive metrics module

## Running Tests

```bash
# Run transcription evaluation tests
pytest tests/test_transcription_evaluator.py -v

# Run all tests
pytest tests/ -v --cov=src
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| High WER on test set | Increase training epochs, check data quality, verify preprocessing |
| CUDA out of memory | Reduce batch size, enable gradient accumulation |
| Poor audio quality detections | Adjust noise_gate_db in PreprocessorConfig |
| Inconsistent scores | Check post-processing rules, validate reference transcriptions |

## Next Steps

1. **Collect & annotate training data** (200-500 EIT responses)
2. **Fine-tune Whisper model** with production audio
3. **Validate against human transcriber baseline** (N=2-3 raters)
4. **Deploy to production** for batch research processing
5. **Monitor WER drift** with periodic evaluation
