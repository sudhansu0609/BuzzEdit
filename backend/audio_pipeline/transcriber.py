import whisper
import logging
from pathlib import Path
from typing import Optional
from config import WHISPER_MODEL, WHISPER_DEVICE

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    _instance = None
    _model = None

    def __new__(cls, model_name: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            model_name = model_name or WHISPER_MODEL
            logger.info(f"Loading Whisper model: {model_name}")
            cls._model = whisper.load_model(model_name)
        return cls._instance

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        word_timestamps: bool = True,
    ) -> dict:
        logger.info(f"Transcribing: {audio_path}")
        result = self._model.transcribe(
            audio_path,
            language=language,
            verbose=False,
            word_timestamps=word_timestamps,
        )
        return result

    def transcribe_with_progress(
        self,
        audio_path: str,
        language: str = "en",
        callback=None,
    ) -> dict:
        logger.info(f"Transcribing with progress: {audio_path}")
        result = self._model.transcribe(
            audio_path,
            language=language,
            verbose=True if callback else False,
            word_timestamps=True,
        )
        if callback:
            callback(1.0, "Transcription complete")
        return result
