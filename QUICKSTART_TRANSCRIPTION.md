# AutoEIT Transcription - Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -e .
# or
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create a JSON manifest of audio files with reference transcriptions:

```bash
python scripts/run_transcription.py create-manifest \
    --input-dir your_audio_folder \
    --output-manifest data/my_manifest.json
```

Then edit `data/my_manifest.json` to add reference text:
```json
[
  {
    "audio": "response_001.wav",
    "reference": "yo fui al mercado ayer"
  },
  {
    "audio": "response_002.wav",
    "reference": "mi hermana estudia en la universidad"
  }
]
```

### 3. Transcribe Audio

```bash
# Using base Whisper model (no fine-tuning)
python scripts/run_transcription.py transcribe \
    --audio-dir your_audio_folder \
    --output-csv transcriptions.csv

# Using fine-tuned model
python scripts/run_transcription.py transcribe \
    --audio-dir your_audio_folder \
    --output-csv transcriptions.csv \
    --checkpoint checkpoints/whisper-eit-v2
```

### 4. Evaluate Quality
```bash
python scripts/run_transcription.py evaluate \
    --results-csv transcriptions.csv \
    --output-json eval_report.json
```

View results:
```bash
cat eval_report.json
# {
#   "wer": 0.087,
#   "agreement": 0.92,
#   "samples": 120,
#   "meets_target": true
# }
```

---

## Training Your Own Model

### Step 1: Prepare Training & Validation Data
```
data/
├── train/
│   └── audio/           # .wav, .mp3, .flac files
├── train_manifest.json  # [{"audio": "...", "reference": "..."}]
└── eval_manifest.json
```

### Step 2: Fine-tune Whisper
```bash
python scripts/run_transcription.py train \
    --train-audio-dir data/train/audio \
    --train-manifest data/train_manifest.json \
    --eval-manifest data/eval_manifest.json \
    --output-dir checkpoints/whisper-eit-v2 \
    --epochs 3 \
    --batch-size 4
```

Training output:
```
[00:15:32] INFO     Model loaded on cuda
[00:15:45] INFO     Loaded 300 samples from data/train_manifest.json
[03:45:12] INFO     Training complete. Model saved to checkpoints/whisper-eit-v2
```

### Step 3: Validate & Deploy
```bash
# Transcribe test set
python scripts/run_transcription.py transcribe \
    --audio-dir data/test/audio \
    --output-csv test_results.csv \
    --checkpoint checkpoints/whisper-iet-v2

# Evaluate
python scripts/run_transcription.py evaluate \
    --results-csv test_results.csv \
    --output-json test_report.json
```

---

## Using the Python API

### Simple Transcription
```python
from src.transcription import TranscriptionPipeline

# Load base model
pipeline = TranscriptionPipeline()

# Or load fine-tuned model
pipeline = TranscriptionPipeline.from_pretrained(
    "checkpoints/whisper-eit-v2"
)

# Transcribe single file
result = pipeline.transcribe("learner_response.wav")
print(result.text)        # "yo fui al mercado ayer"
print(result.confidence)  # 0.92
```

### Batch Processing
```python
import pandas as pd

df = pipeline.transcribe_batch("audio_folder", "results.csv")
print(df)
# Output: CSV with columns [file, transcription, confidence]
```

### Evaluation
```python
from src.transcription import TranscriptionEvaluator

evaluator = TranscriptionEvaluator()

# From CSV
eval_result = evaluator.evaluate_csv("transcriptions.csv")

# Print summary
evaluator.print_summary(eval_result)
# ==> Avg WER: 8.7% (target: ≤10%) ✓

# Save report
evaluator.save_report(eval_result, "report.json")
```

---

## Common Issues & Solutions

### "CUDA out of memory"
```bash
# Reduce batch size
python scripts/run_transcription.py train \
    --batch-size 2 \
    --train-audio-dir data/train/audio \
    --train-manifest data/train_manifest.json
```

### "No audio files found"
Check that:
- Files are in the correct directory
- Supported format: .wav, .mp3, .flac
- Path is correct (relative to script location)

### "Low WER on training but high on test"
- Add more diverse training data
- Check for data leakage
- Verify test set audio quality
- Increase training epochs (3→5)

### "Transcriptions look incomplete"
- Check audio quality (SNR > 15 dB ideally)
- Verify target language setting (should be "es")
- Review post-processing rules
- Try without post-processor if needed

---

## Tutorial Notebooks

See `notebooks/04_transcription_tutorial.ipynb` for:
- Full end-to-end example
- Audio preprocessing visualization
- ASR output inspection
- Post-processing demonstration
- Evaluation metrics & plotting
- Results export examples

Run in Jupyter:
```bash
jupyter notebook notebooks/04_transcription_tutorial.ipynb
```

---

## Performance Tips

### For Faster Inference
- Use GPU (CUDA)
- Enable FP16 (enabled by default)
- Reduce beam size (default: 5)

### For Better Accuracy
- Fine-tune on real EIT data (200+ samples)
- Use 2-3 human raters for references
- Increase training epochs (3→5)
- Adjust preprocessor thresholds for your audio

### For Batch Processing
```python
# Load model once, reuse
pipeline = TranscriptionPipeline.from_pretrained("checkpoint")

# Process in batches
for batch in get_batches(audio_files, batch_size=32):
    results = [pipeline.transcribe(f) for f in batch]
```

---

## File Structure Quick Reference

```
autoeit/
├── src/transcription/          # Core modules
│   ├── pipeline.py             # Main API
│   ├── preprocessor.py         # Audio prep
│   ├── transcriber.py          # ASR wrapper
│   ├── postprocessor.py        # Error correction
│   ├── trainer.py              # Fine-tuning
│   └── evaluator.py            # Evaluation
├── scripts/
│   └── run_transcription.py    # CLI tool
├── data/sample/                # Example data
│   ├── train_manifest.json
│   └── eval_manifest.json
├── notebooks/
│   └── 04_transcription_tutorial.ipynb
└── docs/
    └── TRANSCRIPTION_GUIDE.md  # Full reference
```

---

## For More Information

- **Full Guide:** `docs/TRANSCRIPTION_GUIDE.md`
- **Completion Report:** `GSOC_2026_TRANSCRIPTION_REPORT.md`
- **Tutorial:** `notebooks/04_transcription_tutorial.ipynb`
- **API Docs:** Check docstrings in source code
- **CLI Help:** `python scripts/run_transcription.py --help`

---

## Success Criteria

You've successfully completed transcription when:

✅ WER ≤ 10% on test set  
✅ Agreement ≥ 90% with human transcribers  
✅ Batch processing works on large datasets  
✅ Fine-tuned model outperforms base model  

Need help? Check the troubleshooting guide in the full documentation.
