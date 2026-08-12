import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from faster_whisper import WhisperModel
from runtime.gpu_broker import gpu_broker

logger = logging.getLogger("faster_whisper_engine")

def register_cuda_dll_directories():
    """Register CUDA DLL directories on Windows for CTranslate2 / faster-whisper."""
    if sys.platform != "win32":
        return
    
    # 1. Search in current .venv site-packages nvidia binaries
    venv_site_packages = Path(__file__).parent.parent.parent / ".venv" / "Lib" / "site-packages"
    if venv_site_packages.exists():
        for bin_dir in venv_site_packages.glob("nvidia/*/bin"):
            if bin_dir.exists():
                try:
                    os.add_dll_directory(str(bin_dir))
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                    logger.info(f"Registered CUDA DLL directory: {bin_dir}")
                except Exception as e:
                    logger.debug(f"Failed adding DLL dir {bin_dir}: {e}")

    # 2. Search local ComfyUI torch lib fallback paths
    fallback_paths = [
        Path("B:/ComfyUI_windows_portable_nvidia_cu118_or_cpu/ai-toolkit/venv/Lib/site-packages/torch/lib"),
        Path("B:/ComfyUI_windows_portable_nvidia_cu118_or_cpu/ComfyUI_windows_portable/python_embeded/Lib/site-packages/torch/lib")
    ]
    for p in fallback_paths:
        if p.exists():
            try:
                os.add_dll_directory(str(p))
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
                logger.info(f"Registered CUDA fallback DLL directory: {p}")
            except Exception as e:
                logger.debug(f"Failed adding DLL fallback dir {p}: {e}")

register_cuda_dll_directories()

class FasterWhisperEngine:
    _instance: Optional["FasterWhisperEngine"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.model_device = None
            cls._instance.model_size = "large-v3"
        return cls._instance

    def load_model(self, device: str = "cuda", compute_type: str = "float16") -> WhisperModel:
        if self.model is None or self.model_device != device:
            logger.info(f"Loading faster-whisper model '{self.model_size}' on {device} ({compute_type})...")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4
                )
                self.model_device = device
            except Exception as e:
                logger.warning(f"Failed loading faster-whisper on {device} ({e}). Falling back to CPU int8...")
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4
                )
                self.model_device = "cpu"
            logger.info(f"faster-whisper model loaded successfully on {self.model_device}.")
        return self.model

    def _do_transcribe(self, model: WhisperModel, audio_path: str, language: Optional[str]) -> List[Dict[str, Any]]:
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

    async def transcribe_audio_async(
        self,
        audio_path: str,
        language: Optional[str] = "en"
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio file with CUDA acceleration and automatic CPU fallback if CUDA DLLs fail.
        """
        await gpu_broker.acquire_lease("faster_whisper", required_vram_mb=3000.0)
        try:
            model = self.load_model(device="cuda", compute_type="float16")
            try:
                return self._do_transcribe(model, audio_path, language)
            except Exception as e:
                logger.warning(f"faster-whisper CUDA execution error ({e}). Retrying on CPU (int8)...")
                self.model = None
                model = self.load_model(device="cpu", compute_type="int8")
                return self._do_transcribe(model, audio_path, language)
        finally:
            await gpu_broker.release_lease("faster_whisper")

whisper_engine = FasterWhisperEngine()
