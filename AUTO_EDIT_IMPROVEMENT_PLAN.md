# Auto-Edit: "One Fluent Shot" — Architecture & Improvement Plan

**Status: implemented 2026-08-14.** Everything in §3 is in the working tree and covered
by tests (`357 passed`). §6 lists what is *not* done — the parts that need a real
recording and a loaded language model to validate.

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
| 6 | **Fluency (the model writes the edit)** | `fluency.plan_fluent_cuts` | The authoritative cut layer. See §2.1. |
| 7 | Adjudication *(fallback only)* | `llm.client.adjudicate_disfluencies` | Runs **only** when fluency returned `None`. Otherwise a deterministic aggressiveness table decides the candidates. |
| 8 | Stutter sweep | `fumble_engine._sweep_stutters` | Cuts *create* stutters: "kamare men [hol] men gae" reads fine until the cut removes "hol" and leaves "men men". Run after **every** pass that removes words. |
| 9 | **Best-take selection** | `retakes.choose_best_takes` | Reconsiders "keep the last take". See §3.3. |
| 10 | **Verification loop** | `verify.verify_until_clean` | Reads the edit back sentence by sentence, repairs, drops abandoned fragments, repeats. See §3.2. |
| 11 | **The last read** | `fluency.final_read` | One narrow read of the finished edit end to end. See §3.4. |
| 12 | Timeline rebuild | `timeline.ops.rebuild_primary_tracks` | Turns enabled words into V1/A1 segments. See §2.2. |

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
  fluency.py        the model writes the edit; align_deletions; admissible_repair_cuts; final_read
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

### Tests (`backend/tests/`, 357 passing)

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
| `test_cut_coverage.py` | **new** — render-truth check, `admissible_repair_cuts` shapes |
| `test_timeline.py` | the rebuild against Whisper's swallowed silence |

Run: `./.venv/Scripts/python.exe -m pytest backend/tests -q` — **only `.venv` works**;
neither system Python has `faster_whisper`, so any other interpreter dies importing
`backend/asr/__init__.py`.

### Not done — needs a real recording and a loaded model

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
