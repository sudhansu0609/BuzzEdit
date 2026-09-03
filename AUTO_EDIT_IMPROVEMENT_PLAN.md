# Auto-Edit: "One Fluent Shot" — Architecture & Improvement Plan

**Status: implemented 2026-08-14; §2.0, §2.1a and §2.3a added 2026-08-21; §2.1b (the
constrained span contract and its guards) added 2026-08-25 after the first live
validation runs.** Everything in §3 is in the working tree and covered by tests
(`543 passed`). §6 records what the live runs measured.

**Start here if the edit looks weak.** Read `warnings` in the report first. A weak edit is
usually not a threshold that needs tuning: for a long time it was the language model
silently never running at all (§2.3a), and before that it was every comparison being made
against a romanization that respells the same word between takes (§2.0).

This document is self-contained. A model reading only this file should be able to work on
the auto-edit without any other context.

---

## 1. What the feature is, and the one rule that governs it

BuzzEdit records an unscripted talking-head monologue and cuts it down automatically. The
speaker is a Hindi/English code-switcher; the transcript is romanised Hindi ("Hinglish")
produced by faster-whisper. The auto-edit removes:

- **filler sounds** — "um", "uh", breaths, non-lexical noise;
- **stutters** — a word said twice back to back;
- **false starts / retakes** — the speaker begins a sentence, abandons it, begins again,
  sometimes five times, before the sentence finally lands;
- **word and phrase repeats** — a phrase the speaker ended up saying twice;
- **incomplete sentences** — a thought begun and never finished, left stranded because
  the retry was worded differently enough that structural matching never paired them.

The goal is not "remove as much as possible". The goal is that **what remains plays as one
continuous, fluent take** — as though the speaker said it right the first time.

### The asymmetry rule

> **Missing a fumble is a blemish. Deleting something the speaker said is a defect.**

Every threshold, guard and refusal in this pipeline exists because of that asymmetry. When
tuning anything, this is the tie-breaker. Concretely it means every layer that can remove
words is **delete-only and alignment-checked**: the model is asked for the cleaned text in
the speaker's own words, that text is diffed back onto the real token sequence, and only
`delete` opcodes cut. `replace` keeps the original (a reworded span is a guess);
`insert` is ignored (there is no video for words nobody said). A model that paraphrases,
translates or hallucinates therefore **cannot delete anything** — it can only fail to.

### Three settled decisions — do not revisit without reading this

1. **The planner is audio-driven. Never go back to text-only.** Whisper emits clean
   readable text and deletes "um", "uh", stutters and false starts *before* we see them.
   Matching filler vocabulary against its output found **4 words out of 351** on the real
   195-second project. Worse, Whisper word timestamps absorb the pauses around them — 23
   words claimed over a second each, one claimed 10.13s, and 76s (39% of the video) sat
   sealed *inside* "words" where a gap-between-words rule could never reach it. The audio
   is the ground truth. Always pass `audio_path`.

2. **Rule-based grammar parsing is the wrong tool; the model as a fluency judge is the
   right one.** There is no dependable parser for romanised Hinglish code-switching; the
   ASR text is noisy, so ungrammaticality is not evidence of a fumble; and grammar cannot
   answer *which attempt to keep*, which is the actual decision. Do **not** reach for
   languagetool, spacy, or any parser. "Grammar check" in this codebase means
   `backend/asr/verify.py` — the model reads each surviving sentence and says whether it
   is complete speech.

3. **Language is pinned, not re-detected.** Whisper's language auto-detection is unstable
   on code-switched speech, and a run that lands on `en` *paraphrases into English instead
   of transcribing* — the timeline once had 351 English translation words while the ASR had
   produced 485 Hinglish ones, so every cut point was unrelated to the audio. The settled
   language lives in `project.settings["language"]` and is reused.

### Load-bearing ASR settings (`backend/asr/faster_whisper_engine.py`)

- `vad_filter=False` — the planner does its own VAD; Whisper's remapping is where absorbed
  timestamps come from.
- `condition_on_previous_text=False`.

Measured: 435 words / 0.51s p90 / 13 over-long words, versus 401 / 0.72s / 27 with the old
settings.

---

## 2. The pipeline as it stands

One entry point does the assembly: **`backend/asr/auto_edit.py`**. Four routes used to
build the call themselves and had drifted apart (one never extracted the audio, one never
passed the energy envelope, one threw the report away). They all delegate now.

```
route ──► auto_edit.extract_project_audio(source_video)      -> WAV or None
      ──► auto_edit.plan_auto_edit(words, audio_path, ...)   -> AutoEditPlan(words, report)
      ──► auto_edit.apply_report_to_timeline(tl, report)     -> speech_regions + energy_envelope
      ──► auto_edit.rebuild_and_check(tl, src_id, report)    -> rebuild V1/A1 + render-truth check
      ──► auto_edit.public_report(report)                    -> response payload (no bulk arrays)
```

`plan_auto_edit` wraps `fumble_engine.refine_disfluencies`, which runs these stages in
order. Each annotates the same list of word dicts with `disfluency`, `candidate`, `reason`,
and finally `enabled = not disfluency`.

| # | Stage | Where | What it does |
|---|-------|-------|--------------|
| 1 | VAD | `vad.analyse_speech` | 20ms RMS-dB frames; speech threshold set **adaptively** at `noise_floor + (peak − floor) × 0.22` from the 10th/95th percentiles. A fixed dB threshold is meaningless across microphones — the old −35 dB read quiet recordings as pure silence. Publishes `speech` regions and `envelope()`. |
| 2 | Timing repair | `align.repair_word_timings` | Clamps any word covering <85% speech onto the speech it actually covers; a word over pure silence shrinks to a point rather than being dropped. This is what exposes the silence Whisper sealed inside timestamps. |
| 3 | Missing fillers | `align.find_unvoiced_speech` + `fumble_engine._filler_word` | Speech regions no word claims = the fillers Whisper deleted. Inserted as `"[uh]"` **words** (`reason="filler_sound"`), so they flow through the timeline rebuild *and* appear in the transcript panel where the user can overrule them. Bounded 0.18–1.2s: anything longer is speech the ASR missed, and cutting it destroys content. |
| 4 | Text rules | `disfluency.analyze_disfluencies` | Hard fillers (um/uh/hmm + Hindi `aa`, `oo`) → cut. Immediate stutter → cut the earlier copy. Soft fillers / low-confidence micro-tokens → `candidate`. Adjacency is judged on the words that will **survive**, not the raw list. |
| 5 | Retakes | `retakes.find_retakes` / `apply_retakes` | Speech-repair model: `reparandum → interregnum → repair`. Detects a "rough copy" — an approximate repetition — with fuzzy token matching. ≥0.72 confidence → cut as `retake`; below → `candidate` as `false_start`. Structural, so it is language-independent. |
| 6 | **Fluency (the model names the cuts)** | `fluency.plan_fluent_cuts` | The authoritative cut layer. Asks the model for the runs to delete, with reasons; falls back per window to the older rewrite-and-diff contract. See §2.1 and §2.1a. |
| 7 | Adjudication *(fallback only)* | `llm.client.adjudicate_disfluencies` | Runs **only** when fluency returned `None`. Otherwise a deterministic aggressiveness table decides the candidates. |
| 8 | Stutter sweep | `fumble_engine._sweep_stutters` | Cuts *create* stutters: "kamare men [hol] men gae" reads fine until the cut removes "hol" and leaves "men men". Run after **every** pass that removes words. |
| 9 | **Best-take selection** | `retakes.choose_best_takes` | Reconsiders "keep the last take". See §3.3. |
| 10 | **Verification loop** | `verify.verify_until_clean` | Reads the edit back sentence by sentence, repairs, drops abandoned fragments, repeats. See §3.2. |
| 11 | **The last read** | `fluency.final_read` | One narrow read of the finished edit end to end. See §3.4. |
| 12 | Timeline rebuild | `timeline.ops.rebuild_primary_tracks` | Turns enabled words into V1/A1 segments. See §2.2. |

### 2.0 Which spelling each pass reads — `backend/asr/tokens.py`

**Every word carries two spellings, and reading the wrong one is what let repeats
survive.** Whisper decodes Hindi as Devanagari; `transliterate.to_hinglish` then makes a
romanized copy for the captions, the timeline and the UI.

The romanization is **not stable**. Re-transcribing the reference project's own unchanged
cut audio and diffing it against the plan scored **0.55 coverage** — not because anything
was missing, but because the same sounds came back spelled differently: `gayaa` for `gae`,
`hai` for `hain`, `par` for `pe`, `doston` for `dosto`. Within one transcript it is no
better: the recording's `jaj` and `jaz` are one word said twice, and `dil`/`deel`/`reel`
are one word said three times.

Every pass that decides a cut by *comparing* tokens was comparing those. So a sentence the
speaker plainly said twice read as two different sentences and played twice in the edit.

- **`spoken(word)`** — the native script. What every pass that JUDGES or COMPARES reads:
  fluency, the grammar audit, the retake matcher, the stutter sweep, `_long_repeats`,
  `cut_verify`.
- **`romanized(word)`** — the Latin form. Captions, the timeline's `text`, the forced
  aligner (MMS aligns on romanization), and the romanized filler vocabularies in
  `disfluency.py`, which are written in Latin letters and must keep reading Latin.

Two traps, both guarded and both locked in by tests:

1. **`retakes.phonetic` and `retakes.skeleton` are Latin-only by construction.** They exist
   to repair romanization drift — digraph folds, vowel stripping. `phonetic` keeps only
   ASCII, so on two Devanagari tokens it compares `""` with `""` and scores **every pair of
   unrelated Hindi words 0.85**, above the run threshold. `tokens.is_latin` gates both.
2. **`\w` drops Devanagari combining vowel signs**, so `नहीं` normalised to `नह` — merging a
   negation with words the matcher must keep apart. Every normaliser now keeps Unicode
   marks.

**`WordItem.word_native` persists the spoken script through the timeline.** It was being
dropped at every dict→WordItem conversion, so the "Auto Edit" button — the path a user
takes when the first cut disappointed them — re-planned from romanized text against prompts
that state their input is Devanagari, and the Devanagari negations in
`verify.PROTECTED_WORDS` could never match.

### 2.1 The fluency pass — `backend/asr/fluency.py`

`plan_fluent_cuts(words, ask) -> Optional[Dict[int, bool]]` returns a verdict for **only
the indices the model actually ruled on**. An index absent from the map means the model's
window was discarded and the structural decision stands there. Returning "all keeps" for a
discarded window once resurrected 81 structural cuts — absence and "keep" are different.

- **Windowing** — `windows(total, size=200, overlap=40)` yields `(start, end, core_start,
  core_end)`. Padding either side is context; verdicts are written only for the core, so a
  restart straddling a boundary is judged by the window that owns it.
- **Align from the RIGHT** — `align_deletions` reverses both sequences before diffing. This
  is load-bearing. Every attempt at a sentence opens with the same words, so a left-anchored
  diff matches attempt 1's opening and splices it onto attempt 5's continuation: the text
  reads perfectly and the *audio* jumps 15 seconds mid-sentence. Reversing makes ties fall
  to the last occurrence, which is the take that ran to completion.
- **`snap_runs_to_the_final_take`** — the model's best *text* may not be speakable from one
  take. Surviving blocks (grouped by <2s gaps, not raw runs) move onto the following take
  when the words before it are a rough copy (ratio ≥0.6), extending back to that take's own
  opening word. What survives then comes from one continuous stretch of speech.
- **Trust limits** (`judge_window`): ≥0.35 of the window came back verbatim; ≤0.75 deleted;
  longest single removal ≤ `max(60, 40% of the window)`. A *fraction*, not a fixed cap — the
  real recording opens with five attempts at one sentence and the correct edit deletes 48
  consecutive words, which a fixed 45-word ceiling threw away.
- **Model size decides everything.** A 7.5B model echoes the input back (2 cuts / 240
  words); a 26B-A4B removed 101 and collapsed a five-attempt pile-up.
  `DEFAULT_MODEL = "gemma-4-26b-a4b-it-ultra-uncensored-heretic"`, overridable via
  `AppSettings["llm_model"]`.

### 2.1a The span contract — `backend/asr/spans.py`

Everything in the four bullets above exists to reconstruct a decision the model already
made and was never asked to state. The rewrite contract asks for the cleaned *text* and
recovers the removals by diffing; right-anchored alignment, run snapping and
`admissible_repair_cuts` are all repairs to that guess.

So the planner now asks for the removals directly. Tokens are numbered, and the answer is a
list of runs to delete, each with a reason and **the words at its ends quoted back**:

```
DELETE 12-27 | दोस्तों ... आप | retake
DELETE 44    | उम              | filler
NONE
```

- **The quoted ends are what make an index trustworthy.** The standard objection to
  index-based output is that models miscount — and here a miscount is *detectable*: the
  quoted words do not match the tokens the numbers point at. The span is then relocated by
  searching for the quoted pair at the same length (a miscounted offset is a slip; a
  floating length would turn it into an arbitrary cut), or refused outright when those words
  are nowhere in the window. A model describing a transcript it was not given deletes
  nothing. An index scheme without this check is a guess wearing a number.
- **Still delete-only.** The model can name a run to remove and nothing else, so a model
  that paraphrases, translates or hallucinates can only fail to cut.
- **Reasons come from the model.** `retake` / `filler` / `stutter` / `repeat` land on
  `WordItem.reason`, which is what the transcript panel shows the user when they judge
  whether to put a word back. A rewrite says what to remove but never why. Reasons are read
  off the spans **before** merging: a filler abutting a retake merges into one run, and
  reading the reason off the merged span would label the "um" a retake.
- **Trust limits carry over** (`judge_spans`): ≤0.75 of the window deleted, longest run ≤
  `max(60, 40%)`. `MIN_MATCH_RATIO` does not — there is no rewritten text to have echoed,
  and every survivor is an original token by construction.
- **Fallback per window.** Asking for spans does not guarantee getting them.
  `looks_like_spans` tells a span answer (including a bare `NONE`) from a model that ignored
  the format and rewrote the transcript; a rewrite is read by the old path rather than
  mistaken for "nothing to cut here". Both counts ride in `report.fluency_windows`, and
  `public_report` warns when every window fell back.
- **`NONE` does not overrule structure — deliberately.** It is a real statement, unlike an
  echo, so the window is trusted and cuts nothing. Letting it restore that window's
  structural retakes is the safe direction for the asymmetry rule but the wrong direction
  for the fault being fixed: one lazy `NONE` puts a whole pile-up of attempts back into the
  edit. `fluency_windows["none_answers"]` counts how often the model declines, so this is
  decided on a measurement rather than on the strength of the argument.

`AppSettings["span_planner"] = false` reverts to the rewrite contract, which is how the two
are compared on the same footage.

### 2.1b What the live runs taught the span contract (2026-08-25)

The first end-to-end runs with a real model (gemma-4-12b, the largest that fits the
16.3GB card) found that **asking for DELETE lines in prose does not work on a 12B**: it
ignored the format on every window and rewrote — paraphrased, 6-11% verbatim — so every
window was discarded and the edit shipped structural-only. The fix and its guards:

- **The span answer is now requested as JSON under LM Studio's `json_schema` response
  format** (`fluency.SPAN_SCHEMA` + `JSON_SPAN_SYSTEM_PROMPT`, via
  `client.ask_with_schema`). Constrained decoding is what makes a small model answer in
  span form at all; the same quoted-ends verification still applies, so the answer is no
  more *trusted*, just reliably parseable. The prose path remains as the per-window
  fallback, and `looks_like_spans` needed `re.MULTILINE` — without it a multi-line DELETE
  answer matched nothing and was misread as a rewrite.
- **Selective refusal.** Under the rewrite contract one enormous removal discarded the
  whole window, losing the good cuts with the bad. A span verified its own quoted ends,
  so an oversized merged run now refuses *that run alone* (`oversized_runs` in the
  window stats); the rest of the window stands. Refused regions land in
  `WindowPlan.no_opinion` — never written as keeps, so structure decides there.
- **End-stretch relocation** (`spans._stretch`): the dominant real miscount is one end
  verified in place and the other a token or two off; the bad end moves to its quoted
  word within ±3 tokens. Full relocation at fixed length remains for both-ends-wrong.
- **The orphan-retake guard** (`fluency._copy_survives`): a span whose reason is
  `retake` claims a surviving copy exists — and the model once named BOTH attempts at
  the t-shirt story as retakes, deleting the story from the video entirely, with spans
  that verified and sizes that were legal. A retake span (≥4 words) whose tail has no
  rough copy (ratio ≥0.55) in what the window's own plan lets survive is refused into
  `no_opinion`; the structural decision — which always keeps the last copy — stands.
- **Partial keeps never restore** (`fumble_engine`): under the span contract "keep" is
  merely "not named in any deletion". A model that cut *into* a structural retake run
  has agreed the region is a flounder and drawn a sloppy boundary; restoring the words
  it did not name spliced fragments like "उन्होंने सारे बच्चों को ए सा जिस पे का photo"
  into the edit. Restores now apply only in runs the model cut nothing of.
- **No second fluency pass after a span answer.** A span answer already stated its
  removals; asking again invites the model to keep improving text it approved — a live
  second pass peppered 43 word-level holes through the small-shapes sieve. It still
  runs under the rewrite contract, where the first pass judged debris-laden text.
- **Data tokens are not debris** (`fluency._data_token`, and the same rule in
  `verify.repair_deletions`): "%" reads as punctuation but the audio says "percent",
  and "50" is data — live passes deleted both. Digit-carrying and symbol-only tokens
  are lifted out of repair deletions and barred from the short-run repair shape.
- **Hindi reduplication is not a stutter** (`disfluency.is_reduplication`): the sweep
  cut "अपने -अपने" down to one word and "वो सब अपने-अपने काम पर हैं" lost its meaning.
  The ASR's own hyphen and a vocabulary of words Hindi routinely doubles protect the
  pair, in both the first stutter rule and every later sweep.
- **A repeated grammatical frame is not a retake** (`retakes.FUNCTION_WORDS`): Hindi
  lists repeat their frame — "…गया हो या फिर…" after every item — and a ≤4-word
  structural match made only of frame words tore the frame out of the middle of a
  list. Such matches are skipped; one content word is enough to count as a restart.
- **Best-take sees the continuation** (`retakes._continuation`): judged in isolation
  the longest take wins, and a live run swapped in an earlier attempt whose extra words
  were a dangling "कि आप किसी". The prompt now carries a `then:` line — what the video
  says immediately after the take — and the chosen take must flow into it. Cut runs are
  also assembled *through* interleaved filler cuts (`_cut_runs`), which used to break
  every pile-up at each "[uh]" so no group ever formed.
- **The final read never cuts the sign-off**: it deleted the speaker's "बाय बाय" as
  debris. A cut touching the last two words stands only when the word recurs nearby
  (the remains of a doubled phrase).

### 2.2 The timeline rebuild — `backend/timeline/ops.py`

`rebuild_primary_tracks(timeline, primary_source_id)`:

1. Intersects enabled words with `timeline.speech_regions` → pieces. **This is what removes
   silence inside a word.** Without it, 29.6s of dead air rode along inside kept segments.
2. Merges pieces across gaps ≤ `max_pause_seconds` (0.40) — **unless** the gap contains
   ≥`min_removal_seconds` (0.25) of deliberately removed speech. This is the fix for the
   worst bug the pipeline ever had: a removed filler *is* a short gap, so every short cut
   was silently glued back into the render while the transcript showed it struck out.
3. Drops interior slivers < `min_segment_seconds` (0.35) — a fragment between two removals
   is a flash of a different head position on screen.
4. Pads each segment by `pause_padding_seconds` (0.12), clamping the padding out of removed
   speech (the pad otherwise re-covered most of a just-cut filler).
5. `_snap_cut` moves each cut ±2 frames to the quietest sample of `energy_envelope`, and
   only for a ≥2 dB dip. ASR boundaries are ±50ms guesses; the real articulation boundary is
   the energy dip. The hunt is bounded by removed speech (±1 frame grace) or it re-covers a
   short fumble from both ends.
6. Carries `transform`/`color` across rebuilds via `anchor_word_id`.

The compiler declicks every A1 segment with 8ms `afade` edges.

### 2.3 The language model — `backend/llm/`

- `lm_studio_client.clean_transcript(system, user) -> Optional[str]` is the **only** call
  the fluency, verification, best-take and final-read passes use. Plain text,
  `temperature=0.0`, 300s timeout.
- `lm_launcher.ensure_ready(...) -> Optional[str]` returns **the model id to use**, not a
  bool: preferred → load preferred → load any other catalogue model (max 3 tries). Models
  that fail to load go in `_unusable` for the session, because a failed load costs ~60s.
- **Readiness comes from `/api/v0/models`**, which carries a real `state`
  ("loaded"/"not-loaded"). `/v1/models` lists every model *downloaded* and is **not** a load
  check — believing it is why the whole LLM layer silently never ran for weeks.
- This LM Studio build **rejects `response_format: {"type": "json_object"}`**. Requests ask
  for `json_schema` and fall back to plain text. Every non-200 is logged with its body.

### 2.3a A model that loads is not a model that can serve

The lesson in the bullet above needed one more step, and not taking it cost the same weeks
a second time. `/api/v0/models` reported `state: loaded`, and it was telling the truth —
but the model was **larger than the GPU**, so llama.cpp had loaded it partly offloaded and
it died on the first real prompt:

```
LM Studio 400 during the fluency pass: {"error":"terminated"}
Fluency: no answer for words 160-341
Audit: no answer from the model; edit not verified
fluency_windows: {'answered': 2, 'failed': 1, 'untrusted': 0}
```

Measured on the reference machine: `qwen/qwen3.8-27b` is **17.7GB of weights on a 16.3GB
card**. Every model-driven layer — the fluency pass, the verification loop, best-take, the
final read — was silently doing nothing, and the edit that shipped was the structural
fallback. `reasons: {retake: 105, filler_sound: 33, stutter: 2, not_fluent: 1}` on a
195-second recording: that lone `not_fluent` was the whole contribution of the model.

Two causes, two fixes.

1. **The ASR never let go of the GPU.** Every path runs the ASR and then the model, in one
   process, and ~3GB of faster-whisper plus ~1.2GB of the MMS aligner stayed resident
   through all of it. `auto_edit.release_asr_gpu()` — via `whisper_engine.release_gpu()` and
   `forced_align.release_model()` — is called from `plan_auto_edit` before the model passes.
   The next transcription reloads in seconds, against model passes that cost minutes.

2. **`lms load` reports success for a model that does not fit.** `lm_launcher.model_sizes()`
   reads `sizeBytes` from `lms ls --json` (`/api/v0/models` carries state, type, arch and
   context length, but no size). `_fits` requires `weights × 1.15 + 1GB` — the KV cache and
   compute buffers are allocated on top of the weights, so fitting the weights alone is not
   fitting. Checked in two places, because they answer different questions:
   - **step 2, against TOTAL VRAM** — *can this card ever run this?* A resident model bigger
     than the card is rejected rather than accepted for being resident;
   - **step 3, against FREE VRAM** — *can it load right now?*

   Then the largest model that *does* fit is chosen, because capability on this task scales
   hard with size.

3. **The failure is now reported.** `public_report` says *"the language model never ran, so
   this is a structural edit only"*. `used_llm: False` in a log is how this hid.

> **When an edit looks weak, check `warnings` in the report before touching a threshold.**
> A structural-only edit is not a tuning problem.

---

## 3. What was changed, and why

### 3.1 One entry point (`backend/asr/auto_edit.py`)

Four routes ran the planner and had drifted apart. All differences were invisible because
each still returned a plausible timeline.

- `POST /api/rendering/{id}/auto_edit` — the UI's "Auto Edit" button.
- `POST /api/transcription/transcribe` — first pass.
- `POST /api/timeline/{id}/generate` — **was** omitting `energy_envelope`, so cut-snapping
  was dead on this path, and **was** computing the report and throwing it away.
- `POST /api/agents/full_auto_edit` — **was** raising `NameError` on every call
  (`edit_agent.py` referenced `audio_path` without ever assigning it) and never extracted
  audio at all.

All four now call `plan_auto_edit` / `apply_report_to_timeline` / `record_cut_coverage`, and
all four strip the bulk payloads (`speech`, `energy`) from their HTTP responses via
`public_report` — `transcribe` was returning one integer per 20ms of the recording to the UI.

### 3.2 The verification loop (`verify.verify_until_clean`)

Verification used to be **audit → repair → re-audit, once**. Residual broken sentences
shipped silently. One pass is not enough for a mechanical reason: repair works sentence by
sentence, and removing debris from one sentence changes where the *next* one begins —
sentences that were run together, or split by a pause a cut has now closed, are only judged
correctly on the following read.

```python
verify_until_clean(words, ask, max_rounds=3, drop_fragments=True) -> (quality, cut_indices)
```

`words` is the **full** word list; anything already `disfluency` is invisible to it. Each
round re-reads what actually survives. It stops on the first of: nothing broken, a round
that changed nothing (progress guard — asking again gets the same answer), the model going
silent, or `max_rounds`. **It always ends on a fresh audit**, so `quality["verdict"]` scores
what ships, never an intermediate state.

`quality` carries `sentences`, `broken`, `issues` (what is still wrong), `repairs` (what got
mended along the way — the last audit cannot show this), `rounds` (per-round history), and
`verdict` ∈ `clean` | `still_broken:<n>` | `not_verified`.

**Repair guards** (`repair_deletions`), any of which refuses the *whole* repair — a repair is
one judgement about one sentence and half of it is not a safer version:

- it touches a `PROTECTED_WORDS` token. Deleting a negation does not tidy a sentence, it
  reverses it: the first live run "repaired" *"hamane to notice naheen kiyaa"* (we did NOT
  notice) into its opposite. Hindi and English negations and quantifiers are simply off
  limits.
- match ratio < 0.5, deletions > 50% of the sentence, or fewer than 3 words left.
- it trims the **tail of a pause-split sentence** — that sentence looks unfinished whether
  or not it is, because the thought carries on in the next one. The live run removed
  *"ho sakataa hai ki vah sab"*, whose continuation was in the very next sentence. Debris in
  the middle is still fair game.

**New: whole-fragment dropping** (`find_droppable_fragments`). Repair cannot help a sentence
that is not damaged but *unfinished* — there is nothing inside it to delete. The whole
fragment goes or nothing does. This is the most dangerous cut in the pipeline, so:

- only broken sentences the repair pass could **not** mend (mending beats deleting);
- ≤ `MAX_FRAGMENT_WORDS` (12) — past that a "fragment" is a thought with content in it;
- no `PROTECTED_WORDS` — such a fragment is never even shown to the model;
- there must be a **following** sentence (nothing supersedes the last line);
- the model is shown the pair `A` (the fragment) / `B` (what follows) and must answer `DROP`
  only when A is incomplete **and** B says the same thing properly. Anything the parser does
  not understand counts as `KEEP` — the refusal is the default;
- total dropped ≤ `max(12 words, 15% of the edit)`, and never leaves <3 words.

**Prompts hammer one point.** Roughly half of a real audit's complaints were *transliteration
noise*, not editing debris: the ASR wrote "jaz" for *judge*, "sabsakraaib" for *subscribe*,
"phinamaanaa" for *phenomenon*. The audio is perfect and only the spelling is wrong. Every
prompt in `verify.py`, `retakes.py` and `fluency.py` says so in as many words, and because
repair is delete-only and aligned, a model that ignores the instruction and "corrects" a
spelling changes nothing.

### 3.3 Best-take selection (`retakes.choose_best_takes`)

"Keep the last take" is right almost always — a speaker restarts because the previous try
went wrong. But on the reference recording the *final* reading of the opening is itself
garbled — *"kabhee aisaa ha aap vah hai ki"* — while an earlier attempt says the line
cleanly. Structure cannot see that.

Runs after all cut decisions have landed, so it works on the settled plan. For each
contiguous run of words cut as `retake`, it finds the surviving take that follows and
enumerates **challengers**: attempts starting where the take's own opening word recurs
(which is what the speaker did — they went back to the top of the sentence).

A challenger only qualifies as a genuine alternative reading if:

- it is ≥ `MIN_TAKE_WORDS` (4) — shorter is a run-up, not a reading;
- it is ≥ 80% of the surviving take's length — a truncated version would swap away the
  sentence's ending;
- it matches the take at ≥0.6 similarity — otherwise they are different sentences and
  swapping puts speech from somewhere else into the edit.

The ordinary pile-up (*"dosto kya ap · dosto kya · dosto kya a · dosto"*) fails every one of
these, so it never reaches the model at all and the last take stands. Only when a
like-for-like pair exists is the model asked, and the prompt says *"if the takes are equally
good, choose the LAST one"*. A silent model, an unparseable answer, or a choice outside the
offered takes all leave the last take in place.

### 3.4 Cross-spelling matching and the last read

**`retakes.skeleton`** closes a known limit: the ASR does not romanise the same sound the
same way twice, and on borrowed words it barely tries — one take says *"cornwall
ooniversity"*, the next says *"kaॉnvel yoonivarsitee"*. Those share almost no letters, so the
repeat survived into the finished edit. The skeleton is the token's consonant frame:
`phonetic()` (which now strips stray Devanagari, so it compares against something), then
`c→k`, then vowels **and `y`** removed. Used as a **last-resort** check inside
`token_similarity`, scoring 0.8 — enough to keep a run going through one mangled word, not
enough to open a match on its own. Guarded by a 5-character token minimum and a 3-consonant
frame minimum, because "kar"/"kir"/"kur" all reduce to "kr" and Hindi is full of them. All
existing negative retake tests still pass.

**`fluency.final_read`** is the gate the whole pipeline aims at. Everything upstream judges a
transcript still full of debris, in windows, against rules; nothing upstream ever reads the
finished edit end to end. This does, with a narrow prompt asking only for leftover repeated
phrases, wedged stumbles, and abandoned tails — and telling the model in as many words that
*giving the text back unchanged is the expected answer, not a failure*.

Its output is filtered by **`fluency.admissible_repair_cuts`**, which is the guard that
saved the greeting. A pass re-reading an already-clean edit has almost nothing to do, and a
model with nothing to do starts improving the writing: one such pass cut *"dosto aapake
saath kabhee aisaa"* off the front of the video. Only two shapes are accepted, both
unmistakably debris:

- a run of ≤4 words (a stumble inside an otherwise good take);
- a word that recurs within ±15 surviving words (the remains of a doubled phrase).

`_second_fluency_pass` in `fumble_engine.py` now uses the same function.

### 3.5 The render is the truth (`timeline.ops.audit_cut_coverage`)

The word list is the plan; the V1 items are what will be played. Those disagreed for a long
time and nothing noticed, because the transcript panel reads the *plan*. A cut word is only
really cut if no V1 segment covers the midpoint of the speech under it.

`audit_cut_coverage(timeline) -> {checked, still_audible, examples}` runs on every path via
`auto_edit.record_cut_coverage`, lands in `report.quality["cut_words_still_audible"]`
(target: 0), logs a warning when non-zero, and is shown in the UI. Words with no speech
under them are not counted — a disabled word floating in silence carries no audio to leak.

> **When verifying an edit, check V1 source coverage of disabled-word midpoints — never just
> the `enabled` flags.**

### 3.6 Surfacing all of it to the user

- **`WordItem` now carries `reason` and `candidate`** (`backend/timeline/schema.py`). Both
  were being dropped at every dict→WordItem conversion (`builder.py`, `rendering.py`), so
  the UI could show *that* a word was cut but never *why* — the one thing a user needs in
  order to judge whether to put it back. A word the fluency pass *restores* now has its
  reason cleared, so a kept word never carries a stale explanation.
- **`TranscriptEditor.tsx` has a word-level "Cuts" view** — the words as running text, cut
  ones struck through and dimmed with a reason tooltip, click to cut or restore, double-click
  to jump there. It is wired to the **existing** `toggleWordApi` → `POST
  /api/timeline/{id}/toggle_word`, which was implemented on both sides and called by nothing.
  The backend rebuilds V1/A1 around the new decision and returns the whole timeline. The
  segment-text view remains, behind a toggle.
- **`RenderCompleteModal.tsx`** shows the grammar audit that was computed and never
  displayed: a verdict badge, how many sentences were mended and with which words, take
  swaps, and a red warning if any cut word is still audible. Reason labels moved to
  `src/lib/editReasons.ts`, shared by both components.

---

## 4. Where everything lives

```
backend/asr/
  auto_edit.py      the shared entry point; audio extraction, plan, report plumbing
  vad.py            adaptive speech/silence map + energy envelope
  align.py          word-timing repair; unclaimed-speech (missing filler) detection
  disfluency.py     filler vocabulary, stutters, low-confidence candidates
  retakes.py        rough-copy retake detection; phonetic/skeleton folds; choose_best_takes
  tokens.py         which spelling each pass reads: spoken() vs romanized() (§2.0)
  spans.py          the span contract: parse, verify against quoted ends, merge (§2.1a)
  fluency.py        the model names the edit; span + rewrite contracts; final_read
  verify.py         sentence audit, repair, fragment dropping, verify_until_clean
  fumble_engine.py  runs the stages in order and merges their verdicts (§5)
  transliterate.py  Devanagari → ITRANS → Hinglish (used during transcription, not planning)
backend/timeline/
  schema.py         Timeline, WordItem (+reason, +candidate), tunables
  ops.py            rebuild_primary_tracks, audit_cut_coverage, toggle_word
  builder.py        build_timeline_from_transcript
backend/llm/
  client.py         clean_transcript (the only call the edit passes use), adjudicate_disfluencies
  lm_launcher.py    ensure_ready, catalogue, get_status
src/components/
  TranscriptEditor.tsx    word-level cut review + overrule
  RenderCompleteModal.tsx the edit report and quality verdict
src/lib/editReasons.ts    shared reason labels/tooltips
```

Tunables worth knowing, all with comments explaining the measurement behind them:

| Constant | File | Value |
|---|---|---|
| `MAX_LOOKAHEAD`, `MAX_GAP_SECONDS`, `MAX_INTERREGNUM_GAP_SECONDS` | retakes.py | 45, 2.5s, 6.0s |
| `cut_confidence` | retakes.py | 0.72 |
| `MIN_TAKE_WORDS`, `MIN_TAKE_SIMILARITY`, `MIN_TAKE_COVERAGE` | retakes.py | 4, 0.6, 0.8 |
| `WINDOW_WORDS`, `OVERLAP_WORDS` | fluency.py | 200, 40 |
| `MIN_MATCH_RATIO`, `MAX_DELETED_FRACTION`, `MAX_DELETED_RUN_FRACTION` | fluency.py | 0.35, 0.75, 0.40 |
| `MAX_STUMBLE_WORDS`, `REPEAT_WINDOW` | fluency.py | 4, 15 |
| `span_planner`, `forced_alignment`, `auto_verify_cut` | AppSettings | on, on, on |
| `SENTENCE_PAUSE_SECONDS`, `MIN_JUDGED_WORDS` | verify.py | 2.5s, 4 |
| `MAX_REPAIR_FRACTION`, `MIN_REPAIR_MATCH_RATIO` | verify.py | 0.5, 0.5 |
| `MAX_FRAGMENT_WORDS`, `MAX_FRAGMENT_FRACTION`, `MAX_VERIFY_ROUNDS` | verify.py | 12, 0.15, 3 |
| `max_pause_seconds`, `pause_padding_seconds`, `min_removal_seconds`, `min_segment_seconds` | timeline/schema.py | 0.40, 0.12, 0.25, 0.35 |

---

## 5. Merge precedence — who wins when layers disagree

This is the behaviour of the auto-edit, in `fumble_engine.refine_disfluencies`. Top to
bottom:

1. **Non-lexical noise is never the model's call.** A word whose reason is in
   `{"filler", "filler_sound", "stutter"}` and is already cut skips the fluency verdict
   entirely. "[uh]" is not a word the model should have to reason about.
2. **Fluency cuts win** → `reason="not_fluent"`, unless the word was already labelled
   `retake`/`false_start` (that label is more specific and is preserved).
3. **Fluency keeps override structure** — a word structure cut as `retake`/`false_start`
   that the model kept is restored, and its reason is cleared. The model is the one that can
   see meaning.
4. **An index absent from the fluency map leaves structure alone.** Its window was
   discarded; absence is not "keep".
5. `_second_fluency_pass` — additive cuts only, filtered by `admissible_repair_cuts`. A
   second opinion may tighten the edit, never reopen it.
6. **`_long_repeats` overrides the model** on long verbatim repeats: a run of ≥4 words
   recurring within 25 words → the **first** copy goes, `reason="retake"`. The model reads a
   doubled phrase as rhetoric and keeps both; in an unscripted monologue the viewer hears a
   fumble.
7. Adjudication / the deterministic aggressiveness table runs **only if fluency returned
   `None`**. So does the second `apply_retakes` pass — re-running structure over the model's
   answer would restore exactly the mid-sentence cuts fluency exists to prevent.
8. Stutter sweep (after every removing pass).
9. **Best-take selection** — may swap a whole take, then sweeps again.
10. **Verification loop** — additive cuts, `reason="not_grammatical"`, subject to its own
    refusals.
11. **Final read** — additive cuts, `reason="not_fluent"`, only admissible shapes.
12. `enabled = not disfluency`; reasons tallied; report stashed on `words[0]["_edit_report"]`
    and read back with `last_report(words)`.

---

## 6. Testing, and what is still open

### Tests (`backend/tests/`, 497 passing)

| File | Locks in |
|---|---|
| `test_auto_edit.py` | VAD on synthesised WAVs, timing repair, unvoiced-filler detection, planner end to end |
| `test_asr.py` | filler/stutter/soft-filler/low-confidence rules, aggressiveness fail-safe |
| `test_retakes.py` | the four-attempt pile-up; **~half are negative** — bare two-word repeats, common three-word phrases, set phrases, rhetorical repetition, repetition across a long pause must all survive |
| `test_fluency.py` | alignment (reworded spans kept, invented words cut nothing), trust limits, windowing, take continuity |
| `test_verify.py` | splitting, verdict/repair parsing, delete-only repair, negations never removed |
| `test_verify_loop.py` | **new** — loop convergence, the no-progress stop, fragment-drop guards |
| `test_fumble_merge.py` | **new** — the §5 precedence with a stand-in model |
| `test_best_take.py` | **new** — the garbled-final-take swap, and every case it must refuse |
| `test_cut_coverage.py` | render-truth check, `admissible_repair_cuts` shapes |
| `test_cut_verify.py` | the cut re-transcribed and diffed against the plan; leak mapping back to source frames |
| `test_spans.py` | **new** — the span contract: parsing, the quoted-ends check, relocation at fixed length, refusal of invented words, `NONE` as a trusted answer that does not overrule structure |
| `test_model_fit.py` | **new** — a model larger than the card does not fit; a structural-only edit says so in the report |
| `test_timeline.py` | the rebuild against Whisper's swallowed silence |

Run: `./.venv/Scripts/python.exe -m pytest backend/tests -q` — **only `.venv` works**;
neither system Python has `faster_whisper`, so any other interpreter dies importing
`backend/asr/__init__.py`.

### Measured 2026-08-25 — six live runs on the real recording

The acceptance runs finally happened, against `B:\youtubeProjects\life3Baje\IMG_E2043.MOV`
(195.6s — the old `data/projects/7322a87c` was deleted; this is its source video), with
gemma-4-12b live, using `backend/tools/e2e_auto_edit.py`. Run artifacts are under
`data/projects/e2e_ref*_e2e/`; `edit_marked.txt` in each is the fastest way to audit a
cut by reading. What the runs measured, in order:

1. **Run 1 (before the fixes):** `used_fluency: false` — the 12B ignored the DELETE-line
   format on all 3 windows and paraphrased (6-11% verbatim), so the shipped edit was
   structural-only with 4 spliced sentences. This is the failure mode the user reported
   as "the cuts are terrible".
2. **Run 2 (JSON spans):** all 3 windows answered and trusted — and exposed the partial-
   restore Swiss-cheese and the second-pass over-cutting (§2.1b), plus "%"-deletion and
   the "अपने-अपने" stutter false positive.
3. **Runs 3-4:** exposed the model deleting both copies of the t-shirt story (→ orphan
   guard), the "गया हो या फिर" frame false-retake (→ FUNCTION_WORDS), and an 8-word
   "repair" that deleted the researchers' names (→ MAX_REPAIR_RUN).
4. **Runs 5-6 (all guards):** all content beats present, each told once. Run 6:
   67.4s programme from 195.6s, 26 V1 segments, `cut_verify: clean`,
   `cut_words_still_audible: 2` (breath-level), all three fluency windows trusted as
   spans, 17 orphan refusals reining in model over-cuts, and the doubled tellings gone
   (the cross-script `pair()` matcher sees "थॉमल गिल्गोविच" ≈ "Thommel Gilgovich").
   Residual `still_broken` counts are ASR-garble-level complaints, not edit damage.

**Whisper large-v3 transcribes this footage differently on every run** (407-539 words,
different content coverage, different romanizations, script flips). Judge any pipeline
change across several runs, never one.

The presentation pass was also validated live the same night (`tools/e2e_presentation.py`):
18/18 assets (3 Wan videos — after the `animated: [true]` history-flag fix in
`comfyui_bridge/client.py`), 15 cutaways placed, **60.3% coverage, 18 of 25 jump cuts
hidden under B-roll**, the rest disguised by the all-push-in punch-outs, 95 captions,
render clean. Two more bugs found live: `Project.status` lacked "presented" (killed the
thumbnail stage every night), and `ensure_ready` measured free VRAM without first asking
a warm ComfyUI to let go (killed the LLM planner on every second pass of a night).

### Previously not done — needs a real recording and a loaded model

The new LLM passes (verification loop, best-take, final read) are tested against **stand-in
models**. What a real model does with those prompts has not been measured. Before trusting
them on someone's footage:

1. Run the auto-edit on the reference project `data/projects/7322a87c` (195s) with the 26B
   model loaded, and record: **seconds removed**, **silence retained**,
   `quality["verdict"]`, `quality["rounds"]`, `take_swaps`, `final_read_cuts`, and
   `quality["cut_words_still_audible"]`.
2. **Acceptance:** verdict `clean` (or `still_broken` with a count lower than the previous
   build's); `cut_words_still_audible == 0`; silence retained at the padding floor
   (≈17.7s was the previous measurement, down from 36.2s); every negative test in
   `test_retakes.py` still green.
3. **Watch specifically for over-cutting.** Best-take and fragment-dropping are the two new
   passes that can remove content rather than debris. If `take_swaps > 0`, listen to those
   moments. If the fragment drops look wrong, `drop_fragments=False` on
   `verify_until_clean` disables that one pass without touching anything else.
4. Reference baseline to beat: **40.4s removed (21%)** with fillers found = 23, from the
   audio-driven rewrite; **76.4s (39%)** once retakes landed.

### Known limits that remain

- **`progress.md:634`** — where the final take is garbled *and* no earlier attempt is a
  full-length reading of it, best-take cannot help; the guards deliberately refuse. Fixing
  that would need a take-to-take join, which was tried and removed because it sounded like
  two takes.
- The frontend "Cuts" view rebuilds the timeline on every toggle. Fine for one word at a
  time; a bulk restore would want `toggle_word_range`, which exists on the backend
  (`timeline/ops.py`) and has no UI.
- `verify.normalised_text` is dead code.
