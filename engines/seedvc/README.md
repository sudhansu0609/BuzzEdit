# seed-vc voice engine (BuzzEdit)

BuzzEdit's zero-shot speech-to-speech voice conversion: a line the creator recorded is re-voiced
as a target voice from one 5-20 s reference clip, keeping his timing, intonation and pauses
(`length_adjust = 1.0`). It is SkinWalker's engine, brought into BuzzEdit
(`VOICE_CHANGER_PLAN.md`, brief S0).

| Path | What |
|---|---|
| `cli.py` | the command line BuzzEdit runs: `warm`, `convert`, `apply-edit`, `voices list` -- JSON progress lines on stderr, one JSON result on stdout, exit 0 ok / 1 error / 2 engine unavailable (protocol in its docstring) |
| `segments.py` | SkinWalker's region stitcher (`parse_manifest`, `build_spans`, `apply_edit`), the creator's own code, copied unchanged |
| `presets.json` | SkinWalker's five presets (Woman, Child, Demon, Monster, Ghost) and engine defaults as data: voice stem, engine overrides, FX intent |
| `seed-vc/` | the seed-vc checkout (GPLv3), built by `scripts/setup_seedvc.ps1`; its checkpoints download into `seed-vc/checkpoints` |
| `.venv/` | seed-vc's own Python 3.11 environment (CUDA 12.8 torch, numpy 1.26, transformers 4.46) |

## How BuzzEdit finds and runs it

`backend/audio/seedvc_env.py` resolves the Python and the seed-vc folder: the `seedvc_python` /
`seedvc_dir` settings when set (empty = auto-detect), else `engines/seedvc/.venv` +
`engines/seedvc/seed-vc`, else SkinWalker's existing `skinWalker/.venv` + `skinWalker/vendor/seed-vc`
(nothing is moved until S4, so SkinWalker keeps working), else unavailable. `run_cli()` runs
`cli.py` with that Python, streams the progress, and returns the result; for the commands that load
the models it refuses while a render runs, hands the GPU over first and offloads everything after.

## A fresh machine

    powershell -ExecutionPolicy Bypass -File scripts\setup_seedvc.ps1
    engines\seedvc\.venv\Scripts\python.exe engines\seedvc\cli.py warm

The first `warm` (or the first conversion) downloads a few GB of checkpoints from HuggingFace; after
that it works offline. Without the checkpoints and without a connection the CLI exits 2.

## Licence

seed-vc (https://github.com/Plachtaa/seed-vc) is **GPLv3**. It runs only as a separate process
(`cli.py` under its own venv) and is never imported into BuzzEdit's backend, which only exchanges
files and JSON with it. The audio it generates is not bound by the GPL; redistributing the engine
together with an app would be. Converting into the voice of a real person needs that person's
consent.
