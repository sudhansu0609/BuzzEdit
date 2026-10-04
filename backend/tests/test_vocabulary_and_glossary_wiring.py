"""TranscribeJob's new fields and the `initial_prompt` builder."""

from backend.asr.faster_whisper_engine import _build_initial_prompt, _MAX_INITIAL_PROMPT_TOKENS
from backend.models import TranscribeJob


def test_transcribe_job_accepts_vocabulary_and_glossary():
    job = TranscribeJob(project_id="p1", vocabulary=["Cornwall", "life"], glossary="life3baje")
    assert job.vocabulary == ["Cornwall", "life"]
    assert job.glossary == "life3baje"


def test_transcribe_job_defaults_are_none():
    job = TranscribeJob(project_id="p1")
    assert job.vocabulary is None
    assert job.glossary is None


def test_build_initial_prompt_none_when_empty():
    assert _build_initial_prompt(None) is None
    assert _build_initial_prompt([]) is None
    assert _build_initial_prompt(["", "  "]) is None


def test_build_initial_prompt_joins_words():
    assert _build_initial_prompt(["Cornwall", "Gilovich"]) == "Cornwall Gilovich"


def test_build_initial_prompt_capped_at_200_tokens():
    words = [f"w{i}" for i in range(500)]
    prompt = _build_initial_prompt(words)
    assert len(prompt.split()) == _MAX_INITIAL_PROMPT_TOKENS == 200
    assert prompt.split()[0] == "w0"
    assert prompt.split()[-1] == "w199"
