import pytest
from backend.asr.faster_whisper_engine import whisper_engine

@pytest.mark.asyncio
async def test_whisper_transcription_resilience():
    # Verify engine loads without raising unhandled DLL errors
    model = whisper_engine.load_model()
    assert model is not None
