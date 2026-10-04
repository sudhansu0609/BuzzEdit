# Auto-Cut v2 — an AI editor that cuts like you

**Status: plan, written 2026-10-04. Nothing in §4 is built yet** (genre pause pacing, §5 last
row, shipped the same day). This replaces the *decision* layer described in
`AUTO_EDIT_IMPROVEMENT_PLAN.md`; its *mechanics* — VAD, forced alignment, cut placement,
pacing, render checks — are kept.

Numbers marked **measured** come from this repo's data on 2026-10-04. Numbers marked
**estimate** are to be replaced by measurements in the phase that builds the part (§9).

---

## 0. What "perfect" means, and how it is checked

"Perfect" has to be a number, or every change gets judged by feel. The yardstick is **your
own hand cuts**. For every recording you have cut yourself, the matcher
(`backend/tools/autocut_truth.py`, §6) works out exactly which seconds you kept. The engine
is then scored against that answer key, on recordings whose decisions it has never been
shown (leave-one-out, §6).

| Measure | Old engine (measured, Life3Baje ep1, 13.8 min) | Target |
|---|---|---|
| Good speech wrongly removed | 145.6 s | ≤ 1 s per 10 min |
| Junk left in (speech you removed, it kept) | 103.8 s | ≤ 5 s per 10 min |
| Retakes caught | 0 of 6 | ≥ 95% |
| False starts caught | 1 of 7 | ≥ 95% |
| "Sentences nobody said" (spliced across takes) | in every run before 2026-08 | 0 |
| Number of cuts vs yours | 169 vs your 30 | within ±20% |
| Pause feel | median 0.24 s vs your 0.40 s (Raat3Baje) | median within ±0.05 s of yours |
| Calls left for you to make | — | ≤ 5 per 10 min, one click each |

**The honest limit.** Some cuts are taste: which of two complete takes sounds better, or
whether a tangent is worth keeping. 184 s of the 372 s you cut from Life3Baje ep1 were of
that kind. No system can promise those without you. So this plan sends only those to a
short review list, and every answer you give becomes an example the editor follows next
time (§4.7). The target is for that list to shrink with every video.

## 1. Why the old engine can't get there

None of these are tuning problems. Each one is built into the design:

1. **It decides word by word, through a keyhole.** The model sees 200-word windows
   (`fluency.WINDOW_WORDS`). The retake matcher only pairs repeats within 45 words and a
   2.5–6 s gap (`retakes.MAX_LOOKAHEAD`, `MAX_GAP_SECONDS`, `MAX_INTERREGNUM_GAP_SECONDS`).
   Your retakes are whole passages said again 10–60 s later. For example, Life3Baje ep1
   753–777 s is one passage said three times, and you kept the third. That is out of reach
   by construction.
2. **It is deaf, and its transcript has holes.** Whisper writes clean text and skips
   repeated passages. Several stretches you cut contain zero transcribed words (Life3Baje ep1
   223.7–229.8 s and 383.8–392.0 s, measured). Nothing that isn't in the text can be judged.
3. **The judge is too small for the job.** gemma-4-12b reading romanised Hinglish needed a
   stack of guards (orphan guard, run-size limits, "admissible repair" shapes) to stop it
   destroying content. The same guards block the large, correct cuts, and your edits are
   mostly large ones.
4. **Rules stand in for judgement.** It relies on filler word lists, repeated-phrase
   matching and keyword genre detection. On Raat3Baje ep1, keyword detection called a horror
   story "cooking" (measured).
5. **It was never held to what you actually do.** There was one stored eval run and one
   answer key, made from in-app corrections. There was no way to turn your exported cuts
   into tests, until the matcher.

## 2. The idea

**A strong model makes the judgement calls.** It reads the *whole* recording the way an
editor watches a rough take:

- It decides at the level of **takes and sentences**, not single words.
- It is told **what the audio shows** that the text can't: an abrupt stop, trailing off, a
  restart, speech Whisper never wrote down.
- It knows **your taste** from your own past cuts.
- It **checks its own work** before anything ships.

**Code keeps only the mechanics:** exactly where in a pause to cut, padding, pause length,
frame accuracy, and checking the render. Those are physics, not taste. This is the line
between "smart" and "a set of rules".

## 3. Old vs new

| | Old engine | New editor |
|---|---|---|
| Who decides | Rules (filler lists, repeated phrases) plus a 12B model on 200-word windows | Claude reading the whole recording, with your examples |
| What it decides about | Single words | Whole attempts and sentences; word trims only for stutters and fillers |
| How far apart a retake can be | 45 words or ~6 s | Anywhere in the recording |
| Speech Whisper skipped | Invisible | Recovered, or shown to the model as a unit of speech |
| What it learns from the audio | Silence and filler sounds only | Abrupt stop, trailing off, restart, pitch reset, speaking rate, all described to the model |
| Your taste | Fixed thresholds | Learned from your own cuts and corrections |
| Protection against removing good content | Guards that also block correct big cuts | 3 runs must agree, a content-loss check, and a review list |
| Checking the result | Grammar check per window | Reading the whole finished programme, then listening to the render |
| Proof that it works | One stored run on one answer key | Scored on every hand cut you've made, leave-one-out |
| Money per video | ₹0 | ₹0 extra on your Claude plan (≈ $1.44 at API prices, §8) |

## 4. The pipeline

### 4.1 Listen: a transcript with no holes

- **Kept as is:** Whisper large-v3 (language pinned, `vad_filter=False`), MMS forced
  alignment (`forced_align.py`), VAD (`vad.py`), and filler-sound detection from the audio
  (`fumble_engine.find_unvoiced_speech`).
- **New, gap recovery:**
  - Find every stretch the VAD calls speech but no aligned word covers for ≥ 0.4 s.
  - Re-transcribe that stretch on its own, with 0.3 s of context, then force-align it.
    Whisper's skipping comes from long-form decoding (repeated text suppressed, segments
    dropped by its compression and log-probability thresholds), and a short clip decoded on
    its own usually comes out.
  - If a stretch is still empty, it becomes an `[unclear speech, 1.4 s]` unit, so nothing
    spoken is invisible to the editor.
- **Measured in Phase 1:** the share of VAD speech seconds with no words, before and after,
  on every answer key.

### 4.2 Describe: utterances, with ears

- **Utterances:** speech between pauses of ≥ 0.3 s, numbered U1…Un. Each carries its
  start/end, the pause before and after, and its text in both spellings (Devanagari as heard,
  plus the romanisation).
- **Acoustic notes:** measured per utterance and written in plain words for the model:
  - how it ends: falls and settles, stops abruptly mid-word, or trails off and fades;
  - a pitch reset at its start compared with the previous utterance (how a restart sounds);
  - speaking rate and loudness compared with the speaker's own median;
  - breaths, laughs, coughs and filler bursts found in the audio.
- **Similarity hints:** for each utterance, the closest-matching other utterances anywhere
  in the recording, e.g. `≈ U41 (82%), U44 (76%)`. This uses the cross-script matcher in
  `retakes.pair()`, which already sees "थॉमल गिल्गोविच" ≈ "Thommel Gilgovich". It makes sure
  a retake 60 s away gets noticed in a 300-line table.
- These are **measurements, not decisions**. Nothing in this step removes anything.

### 4.3 Hint: the script, as a hint only

- When a project has a written script (Studio has one, e.g. Raat3Baje ep1's
  `script/narration.md`), each utterance is tagged with the script sentence it matches
  (`S14, 90%`) or marked `ad-lib`. This extends `script_align.py` from words to utterances.
- **It is a hint, never a filter.** In Raat3Baje ep1's final cut, only 46% of the script was
  spoken and 37% of what was said was ad-lib (measured, `production/script_deviation.json`).
- What the hint does: it helps group attempts at the same line, and it tells the editor
  what the speaker meant to say. It never decides that unscripted speech goes.

### 4.4 Decide: the editor reads the whole recording

**The call:** one call over the whole recording, to Claude Opus 5.5 through your existing
`claude-local-api` proxy (`127.0.0.1:8787/v1`, OpenAI-compatible).
- `llm/client.py`'s `RemoteChatClient` already speaks that format.
- BuzzcafStudio already routes video-production planning there.

**Input:**
- the editor's brief;
- your channel's style guide (§4.7);
- 3–6 worked examples from your *other* recordings (a slice of the utterance table plus
  what you kept);
- the script hints;
- the full utterance table.

A 28-minute recording comes to about 40k tokens (estimate, §8).

**Output:** JSON that is checked before use.

- `groups`: attempts at the same content, which one stays, why, and how sure it is.
  `{"utterances": ["U41","U44","U47"], "keep": "U47", "why": "U41 and U44 stop mid-sentence; U47 completes it", "confidence": 0.93}`
- `drops`: utterances the viewer shouldn't hear. Examples: recording-setup chatter, an
  abandoned thought with no retry, dead air. Each has a kind, a reason and a confidence.
- `trims`: a stutter or filler inside a kept utterance, given as quoted words. The existing
  quoted-ends check (`spans.py`) refuses any trim whose quoted words don't match the
  transcript.
- **Everything not named is kept.** The editor can only delete. It cannot add, reword or
  reorder.
- **No timestamps:** the model never gives times, only utterance IDs and quoted words.
  Times come from forced alignment, because model timestamps aren't frame-accurate.

**Why this is judgement and not rules:**
- the model reads meaning ("these three are tries at the same thought"), not repeated
  phrases;
- it tells a deliberate repetition from a fumble by context;
- it judges whether a Hinglish sentence is complete;
- it follows your taste from your examples.

### 4.5 Double-check before anything ships

- **Agreement:** the editor runs three times (Opus 5.5 at medium effort; see §11 for why not
  Sonnet).
  - A cut that every run makes is applied.
  - A cut that only some runs make goes to a **focused second opinion**. Opus is shown just
    that spot (the attempts, their acoustic notes, the sentences around them) and asked one
    question: "which of these should the viewer hear?"
  - If it's still split, the decided rules apply (§10): between takes, keep the later one;
    otherwise, keep the content. The call is logged in the "decided under doubt" report.
- **Content-loss check** (this is the fix for "146 s of good speech removed"):
  - Every removed utterance is set beside the kept take that is supposed to cover it.
  - The model checks whether anything said *only* in the removed one would be lost: a
    name, a number, a beat of the story.
  - If something would be lost, the utterance is restored, or goes on the review list if
    restoring would bring a fumble back.
  - Removing content that appears nowhere else needs all three runs to agree *and* must pass
    this check.
- **Final read:** the model reads the kept text exactly as the viewer will hear it, top to
  bottom. Anything that still reads as a repeat, a broken sentence or a missing link gets
  fixed, either by removing more of the kept text or by restoring something removed. This is
  the job `verify.py` and the final read do today, but over the whole programme instead of
  one window at a time.

### 4.6 Place, then verify by listening

- **Placement is unchanged.** Decisions become word flags (`enabled=False`), and
  `rebuild_primary_tracks` does the rest as it does today:
  - cuts in the pause between utterances, on the quietest instant;
  - genre pacing (horror 1.2 s / 0.45 s);
  - no sliver segments;
  - padding never re-covers removed speech.
- **Verify by listening:** `cut_verify.py` already re-transcribes the rendered cut and
  compares it with the plan. New behaviour:
  - a leak (a removed word still audible) or a loss (a kept word clipped) within 150 ms of a
    cut point gets that cut point nudged and checked again, automatically;
  - anything larger goes on the review list instead of being "fixed" blindly, which is why
    automatic trimming was switched off before.

### 4.7 Review, and learn your taste

- **Review list in the Cuts view:** only the calls the engine isn't sure about, each shown
  as A / B with play buttons and the editor's reason. Everything else is already applied.
- **Learning without training:**
  - Every review answer is stored per channel as an example
    (`data/style/<channel>/examples.jsonl`).
  - So is every recording you cut by hand, once it has been through the matcher.
  - Each new edit's prompt carries the most similar examples.
  - It also carries a short **channel style guide** that the model rewrites from the
    examples every few videos. Seen so far: you keep the last complete attempt; horror keeps
    its dramatic pauses.
  - The more you correct, the fewer calls come back to you.

## 5. Each problem, and what fixes it

| Problem (evidence) | Fix |
|---|---|
| Long retakes 10–60 s apart missed (0 of 6) | Whole recording in one view (§4.4); similarity hints across the recording (§4.2); script-line hints (§4.3) |
| False starts missed (1 of 7) | Acoustic notes (abrupt stop or trailing off, then a restart) (§4.2); decisions per utterance; the model judges whether a sentence is complete |
| Good speech wrongly removed (145.6 s) | Decisions per take instead of word-by-word holes; 3 runs must agree; content-loss check; review list (§4.5) |
| Speech Whisper never wrote down | Gap recovery; every speech stretch becomes a unit even with no text (§4.1); optional word-for-word transcription (§7) |
| Which take to keep | Your rule, learned from your cuts (in every example checked so far: the last complete attempt); the model judges completeness using the acoustic notes; ties go to a second opinion, then to review |
| Too many tiny cuts (169 vs your 30) | Decisions per utterance; word trims only for clear stutters and fillers |
| Script and recording differ | The script is a hint per utterance, never a filter (§4.3) |
| Whisper transcribes differently every run | Transcript made once per project and cached; agreement across editor runs |
| Pauses too tight for horror | **Done 2026-10-04:** genre pacing (`presentation/genre.py` `_PACING`) |
| Keyword rules (genre, fillers) | The model decides; keywords only as a last-resort fallback |
| Whole sections dropped for the story (Raat3Baje ep1: one 8-minute block, 56% of the speech you removed) | A content decision the transcript can't reveal: a channel rule, a target length, or opt-in section proposals (§10 item 6) |

## 6. Measuring: answer keys, and leave-one-out

- **Finish `autocut_truth.py`.** The fix is already measured: apply pre-emphasis before
  correlating.
  - 1 s windows: the true take scores 0.78–0.84; random positions ≤ 0.13.
  - 40 ms blocks: true median 0.81; random 99.9th percentile 0.39.
  - Validation: kept spans come out in order, kept time adds up to the export's length, and
    each segment's confidence margin is reported.
  - Delete or replace the untrustworthy `data/eval/raat3baje_ep1_rules.json` from the
    earlier run.
- **Answer keys** (candidates found on disk):

  | Recording | Raw → your cut | Notes |
  |---|---|---|
  | Raat3Baje ep1 | `TQKE0621.MOV` → `temp4.mp4` | horror, 28.5 min raw |
  | Life3Baje ep1 | existing `data/eval/life3baje_ep1.json` | vlog |
  | Beyond3Baje ep7 | `OXCC3995.MOV` → `ep7Final.mp4` | documentary, *pair to be confirmed* |
  | others | — | any you name |

  Aim for at least 4 recordings across at least 3 channels.
- **Per key:** transcribe the raw once (cached), then label every utterance kept, removed or
  trimmed from the matcher's output. That one labelling is both the test and the source of
  examples.
- **Leave-one-out:** when scoring recording X, examples come only from the *other*
  recordings. This is what proves "any of my recordings" rather than "the ones it was shown".
- **Scoring:** the §0 table, per recording and on average. A change that makes any
  recording worse doesn't go in. `autocut_eval.py` gets extended to compare kept seconds and
  per-utterance decisions.

## 7. What isn't feasible, and the workaround

| Wanted | Why not today | Workaround |
|---|---|---|
| A model that *hears* the whole 28-min Hinglish recording, locally | No local model on a 16 GB card does this. Qwen3-Omni has no Hindi speech input; Qwen3.5-Omni has it but is far too large; Gemma 4 hears clips of ≤ 30 s only | Give the text model ears in words: acoustic notes and similarity hints (§4.2). Optional: Gemma 4 E4B (installed) for ≤ 30 s A/B comparisons, *if* LM Studio passes audio to it (a Phase 1 spike) |
| Cloud ears: Gemini hears the audio, understands Hindi, transcribes word for word (`gemini-3.5-transcribe` keeps um/uh, repeats and false starts) | It sends your audio to Google. That's your call, and it needs a key | Off by default. Turn it on only if the leave-one-out scores show misses that need hearing (choosing between deliveries, or speech gap recovery can't fill). Cost in §8 |
| The local 12B model as the editor | Too small to read 28 minutes and judge; that's why the guards exist | Claude through your proxy. Fallback order: Claude → the proxy's own ChatGPT/Codex failover → queue the cut for later (nightly). A local-model edit is offered only as a draft, with every uncertain call sent to review |
| Claude's usage window can run out (Claude Code shares the same pool) | Plan limits are per 5-hour session and per week | Run cuts when the pool is free (Studio has a nightly queue). Optional paid API key as overflow (≈ $1.44 per 28-min recording) |
| Guaranteed-perfect taste with no input from you | Taste is yours | Review list (≤ 5 per 10 min) plus the learning loop (§4.7) |
| Frame-exact timestamps from a model | Language model timestamps aren't frame-accurate | The model never gives times, only utterance IDs and quoted words; times come from forced alignment |

## 8. Cost

**Assumptions:**
- API prices were fetched on 2026-10-04 from claude.com/pricing (Opus 5.5 $4 / $20 per
  million input/output tokens) and ai.google.dev/gemini-api/docs/pricing (Gemini 3.5 Flash
  $1.50 / $9; Gemini 3.5 Transcribe $0.003 per audio minute + $0.002 per output minute).
- Tokens are **measured** for the main pass (§11: Opus 5.5 at medium effort, Raat3Baje ep1
  28.5 min, 17.9k in / 6.2–7.4k out). The other steps are estimates scaled from it.
- Every call through the proxy carries about 4k tokens of Claude Code's own prompt, so small
  questions are batched into one call.
- ₹88 per $ and ₹8 per kWh are assumptions.

**Per recording, sized like Raat3Baje ep1 (28.5 min raw), all Opus 5.5 at medium effort:**

| Step | Input tokens | Output tokens | API-equivalent |
|---|---|---|---|
| Main edit, whole recording (with script + style examples) | 33k | 7k | $0.27 |
| Agreement runs ×2 | 66k | 14k | $0.54 |
| Second opinions (all disputed spots, one call) | 20k | 5k | $0.18 |
| Content-loss check | 30k | 5k | $0.22 |
| Final read | 15k | 4k | $0.14 |
| Listening check | 12k | 2k | $0.09 |
| **Total** | **≈ 176k** | **≈ 37k** | **≈ $1.44 (≈ ₹127)** |

- **Through your proxy, as decided:** **₹0 extra.** It draws on the Claude plan you
  already pay for, at roughly 210k tokens per long recording.
- **If paid through the API instead:** about $0.05 per raw minute. That's about $0.014
  (≈ ₹1.3) per cut decision, since a recording this long has on the order of 100 cuts.
- **If Phase 3 shows the agreement runs aren't needed** (Opus at medium was already
  consistent between runs, §11), the total drops to about $0.90.
- **Time:** transcription took 5.6 min for 28.5 min of audio (measured). The main pass takes
  50–65 s. The whole cut is about 10–12 minutes.
- **Optional Gemini ears:** $0.14 per recording for a word-for-word transcript, plus $0.25
  per pass if Gemini also listens and judges. Gemini's free tier covers both models, but
  check its data-use terms before sending audio on the free tier.
- **Electricity (estimate):** about 8 min of GPU work plus waiting time is about 0.06 kWh,
  or **≈ ₹0.5 per recording**.

**Per month.** Volume measured from Studio projects for 2026-09: 8 long-form (≈ 200 raw
min) and 24 Shorts (≈ 50 raw min if each is recorded and cut).

| Item | Per month |
|---|---|
| Claude through the proxy | **₹0 extra** (≈ 1.9M tokens from your plan's pool; schedule at night if you hit limits) |
| Same via paid API, if every cut went through it | ≈ $13 (≈ ₹1,110) |
| Gemini ears (optional) | ≈ $1–4 |
| Electricity | < ₹10 |
| **Total new money, as decided** | **₹0** |

Cost scales linearly: about $0.05 per raw minute at API prices, so three times the volume
is about $38 a month on the API and still ₹0 extra on the proxy.

## 9. Phases and status

Each phase is one implementation brief. **Update the Status column after every item**, so
an interrupted run leaves a usable hand-off.

| # | Phase | Delivers | Done when | Status |
|---|---|---|---|---|
| 0 | Answer keys + baseline | Finished `autocut_truth.py`; keys for ≥ 2 recordings; raw transcripts cached; old engine scored on all keys | Keys validated (spans in order, durations add up); baseline §0 table filled | **in progress.** Matcher fixed and validated: it reproduces the trusted Life3Baje alignment (29 of 29 segments, 98.7% of kept time, edit points within ~0.1 s). Raat3Baje key built (`data/eval/raat3baje_ep1.json`: 91 segments, 98% of the export matched; one messy 15 s patch at export 487–503 s). Raat3Baje raw transcribed (3235 words, 5.6 min). **Left:** old engine scored on Raat3Baje (needs LM Studio) |
| 1 | Listen and describe | Gap recovery; `asr/utterances.py` (utterances, acoustic notes, similarity hints, script tags); Gemma-audio spike | ≥ 95% of VAD speech seconds covered by words or an explicit unit, on all keys | **built, not yet measured.** `asr/gap_recovery.py` runs inside the Whisper engine (app setting `gap_recovery`, on by default): Silero VAD finds speech with no aligned word, faster-whisper decodes only those stretches (`clip_timestamps`), and recovered words are aligned within their own hole (`forced_align.align_in_windows`). `asr/utterances.py` builds the table: every speech region a unit, audio notes, cross-script similarity, script tags (Studio `narration.md` via `spoken_script`). Measured (Silero VAD): **Life3Baje 18% of speech without words → 2% after recovery** (+359 words from 103 holes, 4.7 min total, no slower than before). Raat3Baje baseline 13%; its after-run was stopped twice by Claude Code's low-memory guard: the aligner ran its model over the whole 28-min file in one pass and overflowed the GPU into system RAM. Fixed: emissions are computed in 30 s chunks (`forced_align._emission`, tested frame-exact), recovered words align only within their hole. First signal on Raat3Baje: lines with *ends on a level pitch* (35%) and *speech the transcript missed* (39%) were cut at about twice the 18% base rate. Gemma-audio spike not done |
| 2 | The editor | `asr/editor.py`: brief, JSON schema, checks (quoted words, delete-only), Claude client pinned to Opus 5.5 at medium effort + fallbacks; per-request effort in the proxy (§11) | Runs on all keys; first leave-one-out score | **built, not yet measured with the full table.** `asr/editor.py` (refuses any answer the proxy served from another model); the proxy (`claudeAPI`, backed up before editing) takes `reasoning_effort` per request through `applyFlagSettings`, verified: medium on the main proxy gave 6.9k output tokens / 57 s, the medium profile, while requests without it stay at the server's `low`. `tools/editor_bench.py` scores against the keys (`--full`, `--examples`, `--no-notes`, `--no-hints`) |
| 3 | Double-check | Agreement runs, second opinions, content-loss check, final read | Wrongly removed ≤ 1 s / 10 min on all keys | **built, not yet measured.** `asr/double_check.py`: unanimous cuts stand; disputed ones get one focused second opinion (ties: later take, else keep); content-loss check restores lines whose content is said nowhere else; final read of the programme |
| 4 | Style memory | Examples and a channel style guide from the keys; retrieval of similar examples; section-level preferences (§10 item 6) | Leave-one-out retakes and false starts ≥ 95% | **built, not yet measured.** `asr/style.py` turns a matched hand cut into worked excerpts (retakes first); the bench gives each recording only the *other* recordings' excerpts |
| 5 | Into the product | New planner behind a `planner` setting inside `plan_auto_edit` (reaches all 5 entry points); listening-check nudges; report | End-to-end on a real project; render clean | not started |
| 6 | Review and learn | "Decided under doubt" report in the Cuts view; corrections saved as examples | A correction changes the next edit | not started |
| 7 | Acceptance | All §0 targets on all keys plus one new, unseen recording; switch the default; old planner kept as offline fallback | Targets met | not started |
| 8 (optional) | Cloud ears | Gemini word-for-word transcript and audio second opinion, behind a setting | Only if Phase 7 shows misses that need hearing | not started |

## 10. Decisions

**Decided 2026-10-04:**

1. **The editor runs on Claude through the `claude-local-api` proxy.** The transcript text
   goes to Claude; audio stays on the machine.
2. **Ties go to the later take.** When the editor can't tell which of two attempts is
   better, it keeps the second (later) one. With three or more attempts, it keeps the last
   complete one. This is applied automatically and is no longer a review call.
3. **Fully automatic by default.** With take ties settled by rule 2, any other uncertain
   call (drop this aside or keep it?) resolves to *keep*, following the asymmetry rule. The
   review list (§4.7) becomes an optional "what was decided under doubt" report, not a gate.
4. **Opus 5.5 only, at medium effort** (measured in §11). No Sonnet: through the proxy both
   cost ₹0, and Opus is the better editor. Low effort is out; high effort is no better than
   medium on either recording.
5. **Answer keys:** Life3Baje ep1 and Raat3Baje ep1.

**Still open:**

6. **Whole sections dropped for the story.** In Raat3Baje ep1, 490 s of the 875 s of speech
   you removed was one 8-minute block (1189–1679 s): the after-story discussion about old
   government buildings, the psychology of rules and forbidden places. That is a content
   decision, not a fumble, and no editor can infer it from the transcript alone. Options:
   - a channel rule ("Raat3Baje: keep the story, drop the after-story discussion");
   - a target length per channel;
   - the editor proposes section drops, applied only for channels that opt in.
7. **Cloud audio (Gemini)** stays off unless the numbers show it's needed.

## 11. Model choice and effort — measured 2026-10-04

**The proxy, before and after.**
- The proxy doesn't run your global `claude`. It runs Claude Code through the Claude Agent
  SDK it bundles. That was SDK 0.3.283, which bundles Claude Code 2.1.283 and has no
  `claude-sonnet-5-5` in its model catalog. Asked for it, the proxy silently failed over to
  ChatGPT (`gpt-6-astra`).
- **Upgraded to SDK 0.3.289 (Claude Code 2.1.289)** and restarted. Now `claude-sonnet-5-5`
  → Sonnet 5.5 and `claude-opus-5-5` → Opus 5.5.
- Side effect: the `sonnet` alias that Studio uses now serves Sonnet 5.5 instead of Sonnet 5.
- The editor must pin the model and **reject any answer whose `model` isn't the one it asked
  for**, because a model the proxy can't serve doesn't fail; it gets answered by another
  vendor's model.

**Effort.**
- The proxy sets effort server-wide. Your proxy runs at `low`, for Studio's planning calls.
- To test medium and high without touching it, two temporary copies ran on ports 8788 and
  8789, then were stopped.
- For production, the editor needs medium while Studio keeps low. Phase 2 therefore adds
  per-request effort to the proxy (one warm pool per effort level), or runs a second proxy
  instance at medium.

**The benchmark.**
- Setup: the whole recording as one bare prompt (no acoustic notes, hints or style examples
  yet), scored against your cut. Each word is capped at 1 s, so the transcript's timing holes
  don't count against judgement.
- API price per pass uses Opus 5.5 $4 / $20 and Sonnet 5.5 $2 / $10 per million tokens.
- "Junk caught" for Raat3Baje is also shown *without* the 8-minute editorial block (§10
  item 6), which measures cleanup judgement on its own.

| Recording | Model | Effort | Runs | Good speech wrongly removed | Junk caught | Junk caught, editorial block aside | Cuts | Output tokens | $ per pass | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| Life3Baje (13.8 min) | Opus 5.5 | low | 2 | 43–55 s | 65–66% | — | 24–31 | 1.4–1.5k | 0.08–0.09 | 13–14 s |
| | Opus 5.5 | **medium** | 2 | **11–20 s** | 53–57% | — | 32–36 | 6.4–8.2k | 0.18–0.22 | 53–65 s |
| | Opus 5.5 | high | 2 | 17–18 s | 56–58% | — | 33–34 | 10.1–11.3k | 0.26–0.28 | 81–92 s |
| | Sonnet 5.5 | low | 2 | 23–53 s | 55–64% | — | 19–21 | 0.9k | 0.04 | 8–9 s |
| | Sonnet 5.5 | medium | 2 | 54–56 s | 65–66% | — | 22–23 | 1.2–1.3k | 0.04 | 12–13 s |
| | Sonnet 5.5 | high | 2 | 8–21 s | 48–49% | — | 26–27 | 5.9–7.4k | 0.09–0.10 | 39–50 s |
| Raat3Baje (28.5 min) | Opus 5.5 | low | 1 | 51 s | 20% | 48% | 24 | 1.7k | 0.10 | 13 s |
| | Opus 5.5 | **medium** | 2 | **12–14 s** | 17–18% | **40–42%** | 28–31 | 6.2–7.4k | 0.20–0.22 | 50–58 s |
| | Opus 5.5 | high | 2 | 14–15 s | 18% | 41–42% | 31–32 | 9.2–10.2k | 0.26–0.28 | 72–83 s |
| | Sonnet 5.5 | high | 1 | 8 s | 16% | 38% | 24 | 7.1k | 0.11 | 46 s |
| *Old engine, Life3Baje* | gemma-4-12b + rules | — | — | *89 s* | *68%* | — | *152* | — | ₹0 | — |

**What it shows:**
- **Low effort is careless.** Opus at low removes about 4× more good speech than at medium,
  on both recordings.
- **High is no better than medium.** It's within a few seconds on both recordings, while
  costing ~30% more and taking ~40% longer. **Medium is enough.**
- **Opus at medium beats Sonnet at its best (high):**
  - it catches more (Life3Baje 53–57% vs 48–49%; Raat3Baje cleanup 40–42% vs 38%);
  - it removes about as little good speech;
  - through the proxy both cost ₹0.
- **Sonnet 5.5 isn't Sonnet 5.** At low and medium it is as aggressive as Opus at low. The
  earlier cautious behaviour belonged to Sonnet 5.
- **Absolute catch rates are still low,** and the reasons are known. About 20% of Raat3Baje's
  speech is missing from the transcript (Phase 1). This bare prompt has no acoustic notes,
  similarity hints, script hints or style examples yet (Phases 1–4). And one editorial block
  is a content decision only you can state (§10 item 6).
- **What's already good:** the measured weakness of the old engine, removing good speech
  (89 s on Life3Baje), is down to 11–20 s with a bare prompt, in about a fifth of the cuts.

**Decision:** Opus 5.5 at medium effort for every judgement step (§10 item 4). Phase 2
repeats this table with the full prompt.

## 12. Measured results — 2026-10-04 (full table, after Phases 1–4)

Transcripts with gap recovery and chunked alignment. Speech with no words:
- Life3Baje: 18% → 2%.
- Raat3Baje: 11% → 1%.

Chunked alignment kept GPU use under 8 GB (it used to fill all 16 GB). It also placed words *better*: 82% of word time lands on detected speech, against 76% for the single pass.

Scoring: Opus 5.5 at medium effort. Words are capped at 1 s each, and Raat3Baje's 8-minute editorial block is excluded.

| Setup | Life3Baje: good speech wrongly removed | Life3Baje: junk caught | Raat3Baje: good speech wrongly removed | Raat3Baje: junk caught | API cost per recording |
|---|---|---|---|---|---|
| Old engine | 89 s | 68% (152 cuts) | — | — | ₹0 |
| Bare prompt (before Phase 1) | 11–20 s | 53–57% | 12–14 s | 40–42% | $0.18–0.22 |
| Full table, single pass | 8.2 s | 69% | 19.0 s | 70% | $0.27 / $0.49 |
| + style examples (leave-one-out), 3 runs | **1.3–6.9 s** | 66–73% (29–33 cuts; creator's cut: 30) | 16.9–17.5 s | 67–71% | $0.26–0.28 / $0.45–0.50 |
| 3 runs, cut only where all agree | **1.3 s** | 64% | 14.7 s | 66% | ×3 |
| Full double-check (§4.5) + examples | 8.8 s | 65% | 19.0 s | 71% | $1.10 / $1.91 |

**What the numbers say:**
- **The full double-check did not pay for itself.** It cost about 4× a single pass and was no better. Agreement across 3 single passes does help: it took Life3Baje to 1.3 s. Decision: ship 3 parallel passes and cut only where all agree, with no separate final-read call.
- **Raat3Baje's remaining "wrongly removed" seconds are systematic** (all 3 runs agree on them), and they are mostly not damage:
  - about 5 s is keeping one of two identical takes (e.g. the two "हेलो दोस्तों…" intros) where the creator kept the other;
  - about 7 s is real stutters the creator left in ("के साथ के साथ", "इसे लोग इसे लोग").
- **The storytelling delivery note** (genre.editing_notes_for) didn't measurably change Raat3Baje.
- **Still missed:** mostly the after-story content decisions (the fiction disclaimer, the Garud Puran tangent), i.e. the §10 item 6 question.
