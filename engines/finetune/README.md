# Training the cut editor on your own cuts

The cut editor (what removes retakes, false starts and fumbles from a take)
runs on a general model. Measured on Raat3Baje ep2 (2026-10-22): Claude Opus
catches ~75 % of what you cut, the local Gemma 12B 25 %, Qwen 27B 47 %. The
only thing that can teach a local model *your* taste is your own finished
cuts. This folder is the training code; the data lives in
`BuzzEdit/data/finetune/`. The plan and the numbers: `AUTO_CUT_FINETUNE_PLAN.md`.

## Part 1, yours: one answer key per video

Nothing trains until **12-15 takes** have your own final cut (today: 5).
From then on, **every** produced episode should add one. Per episode:

1. **Record** as usual. The raw take stays where it is
   (`B:\youtubeProjects\<Channel>\epN\...mp4`).
2. **Make the cut yours.** Either fix the AI's cut in BuzzEdit's Cuts view
   until you would publish it, or cut by hand in any editor and export.
   The truth is what you would publish. The AI's untouched cut is never
   data, and a cut you only skimmed is bad data.
3. **Make the key**, one command, from `B:\youtubeProjects\Buzzcaf_Media\BuzzEdit`:

   - you corrected the cut inside BuzzEdit:
     ```
     .venv\Scripts\python.exe backend\tools\autocut_mark.py <project_id> raat3baje_ep3
     ```
   - you exported a final from anywhere (Premiere, BuzzEdit export, ...):
     ```
     .venv\Scripts\python.exe backend\tools\autocut_truth.py "<raw take.mp4>" "<your final.mp4>" raat3baje_ep3
     ```
   Either writes `data/eval/raat3baje_ep3.json`: which spans of the raw
   recording your final keeps and which it removes, found by matching your
   final's audio back onto the raw (so EQ, denoise and re-ordering are fine).

   Naming: `<channel>_ep<N>`, lower case: `raat3baje_ep3`, `life3baje_ep4`,
   `beyond3baje_ep8`. A Short recorded on its own: `raat3baje_short_<slug>`.

That is the whole manual part: one command per video. When you have 12-15
keys, say so; everything below is run for you.

What makes a key worth more: a long take (30-45 min gives 12-18 training
pairs; a 5-minute Short gives 1-2), a take with real fumbles and retakes
(a clean read teaches nothing), and a final you actually watched through.

## Part 2, automatic: keys become training pairs

`backend/tools/autocut_dataset.py` (built 2026-10-22) turns keys into what
the model learns from. It transcribes the raw take exactly as production
does, builds the utterance table the editor reads, maps your removed spans
onto it (a removed utterance that resembles a later kept one = a retake
group; a removed stretch inside a kept line = a trim; the rest = drops), cuts
the table into windows of 50 utterances overlapping by 12, and writes one
pair per window in the editor's exact prompt and answer format:

```
.venv\Scripts\python.exe backend\tools\autocut_dataset.py raat3baje_ep3          # one key
.venv\Scripts\python.exe backend\tools\autocut_dataset.py --all                   # every key in data/eval
```

Output per take: `data/finetune/pairs/<name>.jsonl` (the pairs),
`pairs/<name>.stats.json` (how many groups / drops / trims, how much of your
removed time the utterances cover), the key copied to `data/finetune/keys/`,
and the name added to `data/finetune/splits/train.txt`.

`raat3baje_ep2` is the **test take**: the biggest key (257 removed spans),
held out whole, never trained on, so the gate below measures something real.
`--test <name>` changes it; do not.

Passages you dropped for pacing (`editorial` in a key, whole sections no
transcript could reveal) are left out of the pairs on purpose: the model is
not taught to drop good content.

## Where the data lives

```
data/finetune/
  keys/       answer keys, one per take (copies of data/eval/<name>.json)
  pairs/      your cuts as training pairs, one JSONL per take, + .stats.json
  teacher/    Opus's reads on the same takes (weight 0.5; step 5, not built)
  synthetic/  generated retakes (weight 0.3; step 4, not built)
  splits/     train.txt  test.txt
  runs/       one folder per training run (adapter, log, bench result)
  models/     exported GGUF files, ready for LM Studio
```

Keep this folder out of git and out of any cleanup. Size: about 1 GB of
text per 100 takes; 7.5 GB per exported model.

## Part 3, later: training (step 6, not built yet)

One-time setup (Python 3.11, the CUDA build of PyTorch):

```
cd B:\youtubeProjects\Buzzcaf_Media\BuzzEdit\engines\finetune
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # unsloth, peft, trl, bitsandbytes, torch (cu12x)
```

A run (QLoRA on the base model, LoRA r=16, 8k tokens, batch 1 x 8, lr 1e-4,
2 epochs; 4-8 h for ~600 pairs on the 5060 Ti):

```
.venv\Scripts\python.exe train_lora.py --base google/gemma-4-12b --data ..\..\data\finetune --out ..\..\data\finetune\runs\2026-11-20
```

It starts only when nothing is producing, ejects every loaded model first,
and holds the keep-awake like a render. Given tonight's bench, the 27B Qwen
(`qwen3.8-27b-gsq-rco`, 47 % untrained) is the second base to try if the 12B
fine-tune falls short.

Export and install into LM Studio:

```
.venv\Scripts\python.exe export_gguf.py --run ..\..\data\finetune\runs\2026-11-20 --quant Q4_K_M
```

LM Studio then lists `buzzcaf/buzzcaf-editor-12b`.

## The gate: it must beat what we have

```
.venv\Scripts\python.exe backend\tools\editor_bench.py raat3baje_ep2 --backend lm_studio --model buzzcaf/buzzcaf-editor-12b --runs 3
.venv\Scripts\python.exe backend\tools\editor_bench.py raat3baje_ep2 --backend lm_studio --model google/gemma-4-12b --runs 3
```

Compare **caught** (share of your removed speech the model also removed;
base 12B 25 %, Opus ~75 %) and **wrongly removed seconds** (base 87 s, Opus
~10 s; must not go up). The new model is used only if it beats the base
clearly and reaches the Opus level. Until then the cut decision stays on
Claude with the local model as fallback.

Switch the editor to it (or back):

```
curl -X PUT http://127.0.0.1:8100/api/settings/auto_cut_editor -H "Content-Type: application/json" -d "{\"local_model\": \"buzzcaf/buzzcaf-editor-12b\"}"
curl -X PUT http://127.0.0.1:8100/api/settings/auto_cut_editor -H "Content-Type: application/json" -d "{\"local_model\": \"google/gemma-4-12b\"}"
```
