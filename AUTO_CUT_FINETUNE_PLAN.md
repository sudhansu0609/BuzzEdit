# A cut model of our own: fine-tuning the local editor on the creator's cuts

Status: planned 2026-10-21; step 1 measured and step 2 built 2026-10-22; waiting for 12-15 keys. Owner: BuzzEdit `asr/`.
How-to for the creator: `engines/finetune/README.md`.

## Why

No model on the market is made for retake detection; the products that do it
(Gling, AutoEdit, HeyGen cleanup) wrap a transcript and a general model, in the
cloud. Our editor already does more than they do (utterance table with audio
notes, retake sweep, majority vote, verifier), but its judgement is a general
model's. The creator's own finished cuts are the one thing nobody else has:
turned into training pairs they teach a local model *his* taste for which
attempt stays, how much of a false start goes, and where he keeps a pause.

The model to tune is the one that runs by default: `google/gemma-4-12b`
(local-model rule, 2026-10-21). The card is a 16 GB RTX 5060 Ti.

## What the model learns

Exactly the editor's job, in the editor's own format, so nothing else changes:

- **input**: the system prompt (`asr/editor.SYSTEM`) + one *window* of the
  utterance table (`asr/utterances.table`, 40-60 lines, ~5-8k tokens), with
  the style and speaker notes;
- **output**: the decisions JSON the editor parses (`asr/editor.parse`):
  retake groups with the one to keep, false-start trims, drops.

Training on windows, not whole takes, keeps every sample inside what the card
can train (8k tokens) and multiplies the samples: a 40-minute take is ~700
utterances, which is 12-18 overlapping windows. At run time the editor reads
larger parts (`asr/editor.parts_for`, up to ~24k tokens); a model trained on
8k windows still reads them, the window is only the training grain.

## How much data

One **pair** = one window + its decisions. Real pairs come from the creator's
takes; a 30-45 minute take gives 12-18.

| Level | Takes with his final cut | Real pairs | Plus | What to expect |
|---|---|---|---|---|
| Today | 5 (ep1, ep2, Life ep1/ep2, Beyond ep7) | ~50 | -- | too few to train on alone; the base model + teacher pairs |
| **Minimum to train** | 12-15 | 150-250 | 600 synthetic, ~100 teacher | a first signal: should match the base and start to show his habits |
| **Good** | 30-40 | 400-600 | 600 synthetic, teacher dropped | the target: retake choice and trim length in his taste, Opus-level catch |
| Strong | 60+ | 900+ | 300 synthetic | diminishing returns; refresh twice a year |

So: **every produced episode must add its pair** (step 8). At the current
rate (3-4 long videos a week across the channels) the minimum is reached in
about a month, "good" in about three. Nothing is trained before the minimum;
until then the base model runs, as it does now.

Rules for the data:
- the truth is his final video (`<recording>_final.mp4` beside the recording,
  after his own re-cut or approval), never the AI's cut;
- one take is held out whole (ep2, the biggest, 257 removed spans) and is
  never trained on: it is the test;
- teacher pairs (Opus's reads on the same takes) carry weight 0.5, synthetic
  retakes 0.3, his own cuts 1.0; teacher pairs are dropped once real pairs
  pass 400.

## Where it lives

```
BuzzEdit/data/finetune/            <- all training data (not in git)
  keys/            answer keys, one per take: data/eval/<name>.json (built by tools/autocut_truth.py)
  pairs/           real pairs, one JSONL per take:  pairs/raat3baje_ep1.jsonl
  teacher/         Opus's reads turned into pairs:   teacher/raat3baje_ep1.jsonl
  synthetic/       generated retakes:                synthetic/<date>.jsonl
  splits/          train.txt / val.txt / test.txt (lists of JSONL files; test = ep2)
  runs/<date>/     one training run: config.json, train.log, adapter/, bench.json
  models/          merged + quantised GGUF ready for LM Studio
BuzzEdit/engines/finetune/         <- the code and its own venv (never the backend venv)
  README.md  train_lora.py  export_gguf.py  requirements.txt  .venv/
~/.lmstudio/models/buzzcaf/buzzcaf-editor-12b/   <- where LM Studio sees the result
```

The raw recordings stay where they are (`B:\youtubeProjects\<Channel>\epN\`);
a key points at its take by path. `data/finetune/` is ~1 GB of text per 100
takes; the GGUF is ~7.5 GB per exported model; a run's adapter ~200 MB.

One pair (a line of JSONL):

```json
{"take": "raat3baje_ep1", "window": 3, "source": "creator", "weight": 1.0,
 "system": "<asr.editor.SYSTEM>",
 "input": "<style notes>\n\nThe recording:\n\nU41 [612.3-615.0s, pause before 0.4s] ...\n\nYour decisions (JSON only):",
 "output": "{\"groups\": [{\"utterances\": [\"U41\", \"U43\"], \"keep\": \"U43\", \"why\": \"retake\"}], \"trims\": [...], \"drops\": [...]}"}
```

Making pairs (`tools/autocut_dataset.py`, to write):

1. transcribe the raw take as the product does (cached `data/eval/cache/<name>_words.json`),
   build utterances and the table;
2. map each removed span of the key onto utterances: an utterance mostly
   inside a removed span is "removed"; a removed span inside an utterance is a
   trim; consecutive removed utterances that resemble the next kept one are a
   retake group (same similarity measure the sweep uses);
3. cut the table into windows of 40-60 utterances overlapping by 12; for each
   window write the decisions restricted to the utterances it shows;
4. write `pairs/<take>.jsonl`; append the take to `splits/train.txt` (or
   `test.txt` for the held-out one).

## How to train

QLoRA on `google/gemma-4-12b`: the base in 4-bit, a LoRA adapter trained on
the answer tokens only. Settings for the 16 GB card (measure the first run and
adjust; these are the starting point, not a promise):

| Setting | Value |
|---|---|
| base | google/gemma-4-12b (HF weights, 4-bit nf4 at load) |
| LoRA | r=16, alpha=32, dropout 0.05, on q/k/v/o and the MLP projections |
| sequence | 8192 tokens (drop a window that exceeds it) |
| batch | 1, gradient accumulation 8, gradient checkpointing on |
| learning rate | 1e-4, cosine, warm-up 3 % |
| epochs | 2 (3 when real pairs < 200) |
| loss | on the output (decisions JSON) only; sample weight by `weight` |
| validation | 10 % of the training takes' windows, by take |
| expected time | 4-8 h per run at ~600 pairs (first run tells) |

Toolchain: Unsloth (fastest on one consumer card) or PEFT + bitsandbytes +
TRL, Python 3.11, CUDA PyTorch, in `engines/finetune/.venv`. Export: merge the
adapter into the base, `convert_hf_to_gguf.py`, `llama-quantize Q4_K_M`, copy
into LM Studio's models folder as `buzzcaf/buzzcaf-editor-12b`.

The GPU rule applies: a run ejects every other model first and only starts
when both queues are idle (the same gate the bench waiter uses); the Studio's
idle offload must not unload the trainer, so the run holds the keep-awake and
registers with Sentinel like a render does.

## The gate: it must beat what we have

`tools/editor_bench.py raat3baje_ep2 --backend lm_studio --model <id> --runs 3`
on the held-out take, for the base gemma-4-12b, the fine-tune and (for
reference) the last Opus numbers:

- **caught** (share of his removed speech the model also removed): the
  fine-tune must beat the base by a clear margin and reach the Opus level
  (~90 %);
- **wrongly removed seconds**: must not rise above the base;
- **repeats / false starts remaining**: 0, as the v3 rule demands.

A fine-tune that fails the gate is not deployed; the base stays. A run's
`bench.json` is the record. Deploying = `auto_cut_editor.local_model` set to
`buzzcaf/buzzcaf-editor-12b` (Settings API or the app setting); rolling back =
setting it to `google/gemma-4-12b`.

## Steps and status

| # | Step | Status |
|---|---|---|
| 1 | Bench the base models on ep2 (gemma-4-12b, qwen 27B; Sarvam-M if needed) -- the number to beat | done 2026-10-22: gemma-4-12b 25 % caught / 87 s wrong; qwen3.8-27b 47 % / 43 s; Opus ~75 % / ~10 s (single call) |
| 2 | `tools/autocut_dataset.py`: keys -> windows -> pairs; `data/finetune/` layout; hold out ep2 | done 2026-10-22 (tests/test_autocut_dataset.py); run on the 5 keys |
| 3 | Step 8 first: every produced episode adds its key and pairs automatically | not started |
| 4 | Synthetic retake generator (from the kept utterances of all takes) | not started |
| 5 | Teacher pairs from the Opus product runs (weight 0.5) | not started |
| 6 | `engines/finetune/`: venv, `train_lora.py`, `export_gguf.py` | not started |
| 7 | First run when real pairs >= 150; bench on ep2 | blocked on data |
| 8 | Deploy as `buzzcaf-editor-12b` if it passes the gate | blocked on 7 |

Update the Status column after every item, so a cut-off leaves a usable hand-off.
