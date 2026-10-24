"""R1 of REELS_DESK_PLAN.md: can a local text-to-speech engine narrate a Hinglish
horror line with intonation, and does seed-vc turn it into the creator's voice?

    engines/tts/.venv/Scripts/python.exe engines/tts/probe_tts.py <out_dir> <ref.wav>

Writes <out_dir>/<engine>_<variant>.wav and a summary.json with timings. Engines
are tried in turn; one that is not installed or fails is recorded, not fatal.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
REF = Path(sys.argv[2])
ROOT = Path(__file__).resolve().parents[2]
TEXT_HI = ("रात के ठीक दो बजकर सत्रह मिनट पर, आपके फ़्लैट का दरवाज़ा अपने आप unlock हो जाता है। "
           "आप उठकर देखते हैं... बाहर कोई नहीं है। "
           "लेकिन अंदर से, आपकी अपनी आवाज़ कहती है... \"दरवाज़ा मत खोलना।\"")
summary = {"text": TEXT_HI, "ref": str(REF), "runs": []}


def record(name, started, path=None, error=None, **extra):
    row = {"name": name, "seconds": round(time.time() - started, 1), "file": str(path) if path else None, "error": error}
    row.update(extra)
    summary["runs"].append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def chatterbox():
    import torch
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    t0 = time.time()
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
    record("chatterbox_load", t0)
    for variant, kw in (("default", {}), ("cloned", {"audio_prompt_path": str(REF)}),
                        ("cloned_dramatic", {"audio_prompt_path": str(REF), "exaggeration": 0.7, "cfg_weight": 0.3})):
        t1 = time.time()
        try:
            wav = model.generate(TEXT_HI, language_id="hi", **kw)
            path = OUT / f"chatterbox_{variant}.wav"
            torchaudio.save(str(path), wav, model.sr)
            record(f"chatterbox_{variant}", t1, path, duration_s=round(wav.shape[-1] / model.sr, 1))
        except Exception as exc:  # noqa: BLE001
            record(f"chatterbox_{variant}", t1, error=f"{type(exc).__name__}: {exc}"[:300])
    del model
    torch.cuda.empty_cache()


def kokoro():
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline
    t0 = time.time()
    pipe = KPipeline(lang_code="h")
    record("kokoro_load", t0)
    for voice in ("hf_alpha", "hm_omega"):
        t1 = time.time()
        try:
            chunks = [audio for _, _, audio in pipe(TEXT_HI, voice=voice, speed=0.92)]
            audio = np.concatenate([np.asarray(c) for c in chunks])
            path = OUT / f"kokoro_{voice}.wav"
            sf.write(str(path), audio, 24000)
            record(f"kokoro_{voice}", t1, path, duration_s=round(len(audio) / 24000, 1))
        except Exception as exc:  # noqa: BLE001
            record(f"kokoro_{voice}", t1, error=f"{type(exc).__name__}: {exc}"[:300])


def seedvc(source: Path, tag: str):
    py = ROOT / "engines" / "seedvc" / ".venv" / "Scripts" / "python.exe"
    cli = ROOT / "engines" / "seedvc" / "cli.py"
    if not py.exists():
        record(f"seedvc_{tag}", time.time(), error="seed-vc venv not found"); return
    t0 = time.time()
    out = OUT / f"{source.stem}_as_sudhansu.wav"
    proc = subprocess.run([str(py), str(cli), "convert", "--source", str(source), "--ref", str(REF), "--out", str(out)],
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode == 0 and out.exists():
        record(f"seedvc_{tag}", t0, out)
    else:
        record(f"seedvc_{tag}", t0, error=(proc.stderr or proc.stdout)[-300:])


for fn in (chatterbox, kokoro):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        record(fn.__name__, time.time(), error=f"{type(exc).__name__}: {exc}"[:300])
for src in sorted(OUT.glob("chatterbox_default.wav")) + sorted(OUT.glob("kokoro_hm_omega.wav")):
    seedvc(src, src.stem)
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("PROBE DONE")
