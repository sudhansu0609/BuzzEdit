import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from faster_whisper import WhisperModel
from runtime.gpu_broker import gpu_broker
from .transliterate import to_hinglish, is_transliterable

logger = logging.getLogger("faster_whisper_engine")

def register_cuda_dll_directories():
    """Register CUDA DLL directories on Windows for CTranslate2 / faster-whisper."""
    if sys.platform != "win32":
        return
    
    # 1. This venv's own nvidia wheels. These match what CTranslate2 was built
    #    against, so if they are here nothing else should be on the search path.
    registered = 0
    venv_site_packages = Path(__file__).parent.parent.parent / ".venv" / "Lib" / "site-packages"
    if venv_site_packages.exists():
        for bin_dir in venv_site_packages.glob("nvidia/*/bin"):
            if bin_dir.exists():
                try:
                    os.add_dll_directory(str(bin_dir))
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                    logger.info(f"Registered CUDA DLL directory: {bin_dir}")
                    registered += 1
                except Exception as e:
                    logger.debug(f"Failed adding DLL dir {bin_dir}: {e}")

    # 2. A borrowed CUDA toolchain, used only if this venv has none of its own.
    #
    # This used to point at a cu118 ComfyUI build unconditionally, and *ahead of*
    # nothing — it was appended whether or not step 1 had already supplied cu12
    # libraries from this venv. Two CUDA generations on one DLL search path is
    # what produces "the procedure entry point could not be located in the
    # dynamic link library": the loader finds, say, cudnn from one toolchain and
    # cublas from the other, and their exported symbols do not match.
    #
    # So: only fall back when step 1 found nothing, and take the path from the
    # environment rather than hard-coding one machine's install.
    if registered:
        return

    borrowed = os.environ.get("CUDA_DLL_DIR")
    if not borrowed:
        return
    path = Path(borrowed)
    if not path.exists():
        logger.warning("CUDA_DLL_DIR is set to %s, which does not exist", path)
        return
    try:
        os.add_dll_directory(str(path))
        os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
        logger.info(f"Registered borrowed CUDA DLL directory: {path}")
    except Exception as e:
        logger.debug(f"Failed adding borrowed DLL dir {path}: {e}")

register_cuda_dll_directories()

# Languages whose Whisper decoder handles code-switched English fine. The reverse
# is not true: an `en` pass over Hindi does not transcribe, it *paraphrases into
# English*, producing words with no relationship to the audio. So when detection
# is torn between English and one of these, the Indic reading is the safe one.
_INDIC = {"hi", "ur", "pa", "bn", "ta", "te", "ml", "kn", "gu", "mr", "ne", "sd", "si"}


def choose_language(pooled: dict) -> str:
    """Pick the language from pooled window probabilities.

    Pure so it can be tested without a model. The one non-obvious rule: an
    English win does not stand when an Indic language is strongly present,
    because misreading Hinglish as `en` costs the whole transcript (paraphrase),
    while misreading English as `hi` costs nothing (it still transcribes the
    English words, and the romanizer passes Latin text through untouched).
    """
    if not pooled:
        return "en"
    best = max(pooled, key=pooled.get)
    if best == "en":
        indic = {lang: p for lang, p in pooled.items() if lang in _INDIC}
        if indic:
            contender = max(indic, key=indic.get)
            if indic[contender] >= 0.25:
                return contender
    return best


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

    def _do_transcribe(
        self,
        model: WhisperModel,
        audio_path: str,
        language: Optional[str],
        task: str = "transcribe",
    ) -> tuple[List[Dict[str, Any]], str]:
        """Run one Whisper pass. `language=None` auto-detects the spoken language;
        `task="translate"` produces English regardless of source language.
        Returns (words, detected_language)."""
        # Measured on a 195s Hinglish recording from this repo (words / p90 word
        # duration / words over 1s, where a real spoken word is 0.2-0.5s):
        #   vad_filter + autodetect      401 / 0.72s / 27   <- what this used to do
        #   no vad_filter, autodetect    367 / 0.88s / 33
        #   no vad, forced lang, no cond 435 / 0.51s / 13   <- this
        #
        # `vad_filter` trims silence before decoding and then maps timestamps back,
        # and words next to a removed chunk absorb it. Since the auto-edit runs its
        # own VAD over the original audio anyway, filtering here only costs accuracy.
        # `condition_on_previous_text` lets one bad segment drag the rest off course,
        # which is where the multi-second "words" come from.
        if language is None:
            # Whisper's own auto-detect reads the first 30s only, and one
            # atypical opening (music, English greeting) mislabels the whole
            # file. Detect over several windows spread through the recording.
            language = self._detect_language_pooled(model, audio_path)
        segments, info = model.transcribe(
            audio_path,
            language=language,
            task=task,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        detected = getattr(info, "language", language) or "en"
        # Only romanize the transcribe pass. The translate pass already emits
        # English (Latin) text, which must not be run through the Indic cleanup.
        romanize = task == "transcribe" and is_transliterable(detected)
        extracted_words: List[Dict[str, Any]] = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    native = w.word
                    # word == the primary display token. For non-Latin languages
                    # this becomes the Hinglish (romanized) form so captions and
                    # the timeline read in Latin letters; for English it is a
                    # no-op passthrough.
                    hinglish = to_hinglish(native, detected) if romanize else native
                    extracted_words.append({
                        "word": hinglish,
                        "word_native": native,
                        "hinglish": hinglish,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability
                    })
        return extracted_words, detected

    def _detect_language_pooled(self, model: WhisperModel, audio_path: str) -> Optional[str]:
        """Detect the spoken language from windows across the whole file."""
        try:
            from faster_whisper.audio import decode_audio
            audio = decode_audio(audio_path, sampling_rate=16000)
        except Exception as e:
            logger.warning(f"Language detection could not decode audio ({e}); "
                           "falling back to Whisper's own detection.")
            return None
        window = 30 * 16000
        length = len(audio)
        if length <= window:
            offsets = [0]
        else:
            offsets = [int(length * 0.05), int(length * 0.45),
                       min(length - window, int(length * 0.8))]
        pooled: dict = {}
        windows = 0
        for offset in offsets:
            sample = audio[offset:offset + window]
            if len(sample) < 16000:
                continue
            try:
                _lang, _prob, all_probs = model.detect_language(audio=sample)
            except Exception as e:
                logger.warning(f"Language detection window failed ({e})")
                continue
            windows += 1
            for lang, prob in (all_probs or [])[:8]:
                pooled[lang] = pooled.get(lang, 0.0) + float(prob)
        if not windows:
            return None
        pooled = {lang: p / windows for lang, p in pooled.items()}
        decided = choose_language(pooled)
        top = sorted(pooled.items(), key=lambda kv: -kv[1])[:3]
        logger.info(f"Language detection over {windows} windows: "
                    f"{[(l, round(p, 3)) for l, p in top]} -> {decided}")
        return decided

    def _transcribe_with_fallback(
        self, audio_path: str, language: Optional[str], task: str = "transcribe"
    ) -> tuple[List[Dict[str, Any]], str]:
        """Load the model (CUDA, then CPU int8 on failure) and run one pass."""
        model = self.load_model(device="cuda", compute_type="float16")
        try:
            return self._do_transcribe(model, audio_path, language, task)
        except Exception as e:
            logger.warning(f"faster-whisper CUDA execution error ({e}). Retrying on CPU (int8)...")
            self.model = None
            model = self.load_model(device="cpu", compute_type="int8")
            return self._do_transcribe(model, audio_path, language, task)

    async def preload_async(self) -> None:
        """Load the model without transcribing, so a caller can show a distinct
        'loading model' phase before the decode. No-op once it is resident.

        Goes through the VRAM broker exactly like a transcription so it never
        loads 3GB behind ComfyUI's back — which is why the model is not simply
        warmed at server startup.
        """
        if self.model is not None:
            return
        await gpu_broker.acquire_lease("faster_whisper", required_vram_mb=3000.0)
        try:
            await asyncio.to_thread(self.load_model)
        finally:
            await gpu_broker.release_lease("faster_whisper")

    async def transcribe_words_async(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], str]:
        """Transcribe and report the language that was used.

        Callers should persist the returned language and pass it back on the next
        run. Auto-detection is not stable on code-switched speech: the same
        Hinglish recording detects as `hi` on one pass and `en` on another, and an
        `en` pass does not transcribe it — it paraphrases it into English. That
        produced a timeline whose "words" were English translation tokens with no
        relationship to the audio, which is why cuts landed in arbitrary places.
        """
        await gpu_broker.acquire_lease("faster_whisper", required_vram_mb=3000.0)
        try:
            # Loading the 3GB model and decoding are seconds-to-minutes of blocking
            # C/CTranslate2 work. Run them off the event loop or the whole server
            # freezes — the job's own /status polls included — so the UI sits at the
            # last phase looking dead for the entire transcription. to_thread keeps
            # the loop live so progress and cancellation still flow.
            return await asyncio.to_thread(
                self._transcribe_with_fallback, audio_path, language, "transcribe")
        finally:
            await gpu_broker.release_lease("faster_whisper")

    async def transcribe_audio_async(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio with CUDA acceleration and automatic CPU fallback.
        `language=None` (default) auto-detects the spoken language rather than
        forcing English. Words are returned in the native script with an added
        romanized `hinglish` field; `word` holds the romanized form.
        """
        words, _ = await self.transcribe_words_async(audio_path, language)
        return words

    async def transcribe_full_async(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full transcription for the transcript panel: native words (with Hinglish),
        the detected language, and — for non-English speech — an English
        translation via Whisper's translate task.

        Returns: {words, language, native_text, hinglish_text, english_text}
        """
        await gpu_broker.acquire_lease("faster_whisper", required_vram_mb=3000.0)
        try:
            # Off the event loop — see transcribe_words_async: the model load and
            # decode block, and running them inline freezes the server for the
            # whole pass.
            words, detected = await asyncio.to_thread(
                self._transcribe_with_fallback, audio_path, language, "transcribe")
            native_text = " ".join(w["word_native"] for w in words).strip()
            hinglish_text = " ".join(w["hinglish"] for w in words).strip()

            english_text = ""
            # If the speech is already English, the transcription is the English.
            if detected == "en":
                english_text = native_text
            else:
                try:
                    en_words, _ = await asyncio.to_thread(
                        self._transcribe_with_fallback, audio_path, language, "translate")
                    english_text = " ".join(w["word_native"] for w in en_words).strip()
                except Exception as e:
                    logger.warning(f"English translation pass failed ({e}); leaving english_text empty.")

            return {
                "words": words,
                "language": detected,
                "native_text": native_text,
                "hinglish_text": hinglish_text,
                "english_text": english_text,
            }
        finally:
            await gpu_broker.release_lease("faster_whisper")

whisper_engine = FasterWhisperEngine()
