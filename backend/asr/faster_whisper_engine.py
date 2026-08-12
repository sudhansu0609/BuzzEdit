import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from faster_whisper import WhisperModel
from ..runtime.gpu_broker import gpu_broker

logger = logging.getLogger("faster_whisper_engine")

class FasterWhisperEngine:
    _instance: Optional["FasterWhisperEngine"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.model_size = "large-v3"
        return cls._instance

    def load_model(self, device: str = "cuda", compute_type: str = "float16") -> WhisperModel:
        if self.model is None:
            logger.info(f"Loading faster-whisper model '{self.model_size}' on {device} ({compute_type})...")
            # Check local cache first or let faster-whisper load
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4
                )
            except Exception as e:
                logger.warning(f"Failed loading faster-whisper on CUDA ({e}). Falling back to CPU int8...")
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4
                )
            logger.info("faster-whisper model loaded successfully.")
        return self.model

    async def transcribe_audio_async(
        self,
        audio_path: str,
        language: Optional[str] = "en"
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio file and return word-level timestamps list.
        Each word dict: {"word": str, "start": float, "end": float, "probability": float}
        """
        await gpu_broker.acquire_lease("faster_whisper", required_vram_mb=3000.0)
        try:
            model = self.load_model()
            segments, info = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            extracted_words: List[Dict[str, Any]] = []
            for seg in segments:
                if seg.words:
                    for w in seg.words:
                        extracted_words.append({
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability
                        })

            return extracted_words
        finally:
            await gpu_broker.release_lease("faster_whisper")

whisper_engine = FasterWhisperEngine()
