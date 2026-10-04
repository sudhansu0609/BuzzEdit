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


def _forced_alignment_enabled() -> bool:
    """Whether to re-time words with forced alignment. On by default (when its
    model is installed); set the `forced_alignment` app setting to false to keep
    Whisper's own timings."""
    try:
        from store.app_settings import AppSettings
        value = AppSettings().get("forced_alignment")
        return True if value is None else bool(value)
    except Exception:
        return True


def _gap_recovery_enabled() -> bool:
    """Whether to decode again the speech the main pass left without words (see
    gap_recovery.py). On by default; the `gap_recovery` app setting turns it off."""
    try:
        from store.app_settings import AppSettings
        value = AppSettings().get("gap_recovery")
        return True if value is None else bool(value)
    except Exception:
        return True


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


# Whisper's `initial_prompt` biases decoding toward words it contains, which is
# exactly what the user's own script vocabulary (proper nouns, brand names,
# loanwords) is for. Bounded well under Whisper's own ~224-token prompt window
# so it never gets silently truncated mid-word.
_MAX_INITIAL_PROMPT_TOKENS = 200


def _build_initial_prompt(vocabulary: Optional[List[str]]) -> Optional[str]:
    """Join `vocabulary` into a whitespace-separated prompt, capped at
    `_MAX_INITIAL_PROMPT_TOKENS` words. `None` (Whisper's own default) when
    there is nothing to bias with."""
    if not vocabulary:
        return None
    words = [str(w).strip() for w in vocabulary if str(w or "").strip()]
    if not words:
        return None
    return " ".join(words[:_MAX_INITIAL_PROMPT_TOKENS])


# A segment is "broken" -- worth one more decode -- when it is a real
# hallucination loop: the same few words over and over. Whisper's own test
# (zlib compression ratio > 2.4) cannot be used: Devanagari is 3 bytes a
# character and compresses far better than Latin text, so 26 of 36 perfectly
# normal segments of a real Hinglish take "failed" it -- which is also why
# Whisper's built-in temperature fallback re-decoded nearly everything and a
# 13:50 take took 20 minutes. Low confidence alone is no reason either: it is
# normal for Hinglish and retrying it bought nothing.
RETRY_TEMPERATURES = (0.2, 0.4, 0.6)
LOOP_MIN_WORDS = 12
LOOP_MAX_DISTINCT_SHARE = 0.3


def _is_broken(segment) -> bool:
    words = (getattr(segment, "text", "") or "").split()
    if len(words) < LOOP_MIN_WORDS:
        return False
    return len(set(words)) / len(words) < LOOP_MAX_DISTINCT_SHARE


def _retry_broken_segments(model, audio_path: str, segments: list, language, task, prompt) -> list:
    """Re-decode just the broken segments at a little temperature and keep a
    retry only when it is no longer broken. Everything else is untouched."""
    broken = {i for i, seg in enumerate(segments) if _is_broken(seg)}
    if not broken:
        return segments
    logger.info("Whisper: retrying %d of %d segments that decoded badly", len(broken), len(segments))
    fixed: list = []
    for index, seg in enumerate(segments):
        if index not in broken:
            fixed.append(seg)
            continue
        try:
            retry = list(model.transcribe(
                audio_path, language=language, task=task, word_timestamps=True,
                vad_filter=False, condition_on_previous_text=False, initial_prompt=prompt,
                temperature=RETRY_TEMPERATURES, clip_timestamps=[float(seg.start), float(seg.end)])[0])
        except Exception as e:
            logger.debug("Segment retry at %.1fs failed: %s", seg.start, e)
            retry = []
        fixed.extend(retry if retry and not any(_is_broken(r) for r in retry) else [seg])
    return fixed


class FasterWhisperEngine:
    _instance: Optional["FasterWhisperEngine"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.model_device = None
            cls._instance.model_size = "large-v3"
            cls._instance.last_gap_report = None
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

    def release_gpu(self) -> float:
        """Drop the ASR models so the language model can have the GPU. Returns MB freed.

        The auto-edit runs the ASR and then the language model, in that order, in
        one process — and the ASR was never letting go. On this machine that is
        about 3GB of faster-whisper plus 1.2GB of the MMS aligner still resident
        while LM Studio tries to fit a model into what is left, and the result was
        not a slow edit but a silently *dead* one: llama-server answered
        `{"error":"terminated"}`, `plan_fluent_cuts` got nothing, and the cut that
        shipped was the structural fallback with `used_llm: False` buried in a log.

        Safe to call at any time — the next transcription reloads what it needs,
        which costs a few seconds against model passes that cost minutes.
        """
        freed = 0.0
        try:
            from runtime.gpu_broker import gpu_broker
            before = gpu_broker.get_free_vram_mb()
        except Exception:
            before = None
        if self.model is not None:
            self.model = None
            self.model_device = None
        try:
            from .forced_align import release_model
            release_model()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if before is not None:
            try:
                from runtime.gpu_broker import gpu_broker
                freed = max(0.0, gpu_broker.get_free_vram_mb() - before)
            except Exception:
                freed = 0.0
        logger.info("ASR released the GPU (%.0f MB freed).", freed)
        return freed

    def _do_transcribe(
        self,
        model: WhisperModel,
        audio_path: str,
        language: Optional[str],
        task: str = "transcribe",
        vocabulary: Optional[List[str]] = None,
        glossary: Optional[Dict[str, str]] = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        """Run one Whisper pass. `language=None` auto-detects the spoken language;
        `task="translate"` produces English regardless of source language.
        `vocabulary` (Latin words from the user's script) both biases decoding
        via `initial_prompt` and feeds the post-decode Hinglish romanizer;
        `glossary` (channel spelling overrides) only feeds the romanizer.
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
        # One deterministic greedy-beam pass (temperature 0), then a retry of
        # only the segments that came out broken. Whisper's default fallback
        # re-decodes any doubtful segment at rising temperatures; on Hinglish
        # that fired constantly -- measured on 180 s of a real recording: 220 s
        # and a different transcript every run, against 34 s and an identical
        # one at temperature 0 with the same word count. The full 13:50 take
        # went from 20 minutes to about 2.5.
        prompt = _build_initial_prompt(vocabulary)
        segments, info = model.transcribe(
            audio_path,
            language=language,
            task=task,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            temperature=0.0,
        )
        segments = _retry_broken_segments(model, audio_path, list(segments), language, task, prompt)
        detected = getattr(info, "language", language) or "en"
        # Only romanize the transcribe pass. The translate pass already emits
        # English (Latin) text, which must not be run through the Indic cleanup.
        romanize = task == "transcribe" and is_transliterable(detected)

        def make_word(w) -> Dict[str, Any]:
            native = w.word
            # word == the primary display token. For non-Latin languages this
            # becomes the Hinglish (romanized) form so captions and the timeline
            # read in Latin letters; for English it is a no-op passthrough.
            hinglish = (to_hinglish(native, detected, vocabulary=vocabulary, glossary=glossary)
                        if romanize else native)
            return {
                "word": hinglish,
                "word_native": native,
                "hinglish": hinglish,
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            }

        extracted_words: List[Dict[str, Any]] = [
            make_word(w) for seg in segments for w in (seg.words or [])]

        # Replace Whisper's unreliable word times with acoustically-aligned ones.
        # Whisper's cross-attention timestamps drift by many seconds on this
        # footage, which makes the cut keep the wrong source frames (the video
        # plays a fumble the transcript says was removed). Forced alignment is
        # deterministic and accurate; it is optional and degrades to the Whisper
        # timings when its model is not installed. Only the transcribe pass — the
        # translate pass is English text we never cut on.
        aligning = task == "transcribe" and _forced_alignment_enabled()

        def align(words_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            try:
                from .forced_align import align_words
                return align_words(audio_path, words_in, detected, device=self.model_device or "cuda")
            except Exception as e:
                logger.warning("Forced-alignment step errored (%s); using Whisper timings.", e)
                return words_in

        if aligning and extracted_words:
            extracted_words = align(extracted_words)
        if task == "transcribe":
            self.last_gap_report = None
        # Speech the main pass skipped (a repeated take, a restart) is decoded again on
        # its own, then everything is aligned together: the aligner fits the whole
        # transcript to the whole recording, so a hole bends the words around it.
        if task == "transcribe" and extracted_words and _gap_recovery_enabled():
            from .gap_recovery import recover_gaps
            try:
                import torch
                torch.cuda.empty_cache()        # the aligner's cache, before Whisper decodes again
            except Exception:
                pass
            extracted_words, self.last_gap_report = recover_gaps(
                model, audio_path, extracted_words, detected, make_word, initial_prompt=prompt)
            if aligning and self.last_gap_report.get("recovered_words"):
                # Only the recovered words, each against its own hole's audio. Aligning the
                # whole recording a second time doubled the pass and, on a 28-minute file,
                # spilled the GPU into shared memory until the machine ran out of RAM.
                try:
                    from .forced_align import align_in_windows
                    recovered = [w for w in extracted_words if w.get("recovered")]
                    align_in_windows(audio_path, recovered, self.last_gap_report.get("gap_list") or [],
                                     device=self.model_device or "cuda")
                except Exception as e:
                    logger.warning("Aligning recovered words errored (%s); keeping Whisper timings.", e)
                extracted_words.sort(key=lambda w: float(w.get("start") or 0.0))
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
        self, audio_path: str, language: Optional[str], task: str = "transcribe",
        vocabulary: Optional[List[str]] = None, glossary: Optional[Dict[str, str]] = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        """Load the model (CUDA, then CPU int8 on failure) and run one pass."""
        model = self.load_model(device="cuda", compute_type="float16")
        try:
            return self._do_transcribe(model, audio_path, language, task, vocabulary, glossary)
        except Exception as e:
            logger.warning(f"faster-whisper CUDA execution error ({e}). Retrying on CPU (int8)...")
            self.model = None
            model = self.load_model(device="cpu", compute_type="int8")
            return self._do_transcribe(model, audio_path, language, task, vocabulary, glossary)

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
        language: Optional[str] = None,
        vocabulary: Optional[List[str]] = None,
        glossary: Optional[Dict[str, str]] = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        """Transcribe and report the language that was used.

        Callers should persist the returned language and pass it back on the next
        run. Auto-detection is not stable on code-switched speech: the same
        Hinglish recording detects as `hi` on one pass and `en` on another, and an
        `en` pass does not transcribe it — it paraphrases it into English. That
        produced a timeline whose "words" were English translation tokens with no
        relationship to the audio, which is why cuts landed in arbitrary places.

        `vocabulary` (Latin words from the user's script) biases decoding via
        `initial_prompt` and helps the Hinglish romanizer restore English
        loanwords Whisper wrote out in Devanagari; `glossary` (channel spelling
        overrides, already merged) only affects the romanizer.
        """
        await gpu_broker.acquire_lease("faster_whisper", required_vram_mb=3000.0)
        try:
            # Loading the 3GB model and decoding are seconds-to-minutes of blocking
            # C/CTranslate2 work. Run them off the event loop or the whole server
            # freezes — the job's own /status polls included — so the UI sits at the
            # last phase looking dead for the entire transcription. to_thread keeps
            # the loop live so progress and cancellation still flow.
            return await asyncio.to_thread(
                self._transcribe_with_fallback, audio_path, language, "transcribe",
                vocabulary, glossary)
        finally:
            await gpu_broker.release_lease("faster_whisper")

    async def transcribe_audio_async(
        self,
        audio_path: str,
        language: Optional[str] = None,
        vocabulary: Optional[List[str]] = None,
        glossary: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio with CUDA acceleration and automatic CPU fallback.
        `language=None` (default) auto-detects the spoken language rather than
        forcing English. Words are returned in the native script with an added
        romanized `hinglish` field; `word` holds the romanized form.
        """
        words, _ = await self.transcribe_words_async(audio_path, language, vocabulary, glossary)
        return words

    async def transcribe_full_async(
        self,
        audio_path: str,
        language: Optional[str] = None,
        vocabulary: Optional[List[str]] = None,
        glossary: Optional[Dict[str, str]] = None,
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
                self._transcribe_with_fallback, audio_path, language, "transcribe",
                vocabulary, glossary)
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
                "gap_recovery": getattr(self, "last_gap_report", None),
            }
        finally:
            await gpu_broker.release_lease("faster_whisper")

whisper_engine = FasterWhisperEngine()
