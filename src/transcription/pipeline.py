"""End-to-end transcription pipeline."""
from pathlib import Path
from typing import Optional
import pandas as pd
from .preprocessor import AudioPreprocessor, PreprocessorConfig
from .transcriber import WhisperEITTranscriber, TranscriberConfig
from .postprocessor import PostProcessor, PostProcessorConfig

class TranscriptionPipeline:
    def __init__(self, preprocessor=None, transcriber=None, postprocessor=None):
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.transcriber = transcriber
        self.postprocessor = postprocessor or PostProcessor()

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs):
        transcriber = WhisperEITTranscriber.from_pretrained(checkpoint_path)
        return cls(transcriber=transcriber, **kwargs)

    def transcribe(self, audio_path: str | Path) -> "TranscriptionResult":
        segments = self.preprocessor.process(audio_path)
        results = [self.transcriber.transcribe_segment(s) for s in segments]
        # Merge multi-segment results
        if not results:
            from .transcriber import TranscriptionResult
            return TranscriptionResult(text="", confidence=0.0)
        texts = [self.postprocessor.process(r.text) for r in results]
        merged = " ".join(t for t in texts if t)
        return type(results[0])(
            text=merged,
            confidence=sum(r.confidence for r in results) / len(results),
        )

    def transcribe_batch(self, audio_dir: str | Path, output_csv: str | Path) -> pd.DataFrame:
        audio_dir = Path(audio_dir)
        records = []
        for audio_file in sorted(audio_dir.glob("*.wav")):
            result = self.transcribe(audio_file)
            records.append({"file": audio_file.name, "transcription": result.text, "confidence": result.confidence})
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        return df
