# Auto-Cut v3 — no repeats, no false starts

**Status: plan written 2026-10-08 (session cut off by the usage limit before implementation).
Nothing in §3 is built yet.** Builds on `AUTO_CUT_EDITOR_PLAN.md` (v2, shipped 2026-10-04:
Whisper → gap recovery → utterance table → 3 Opus reads → majority vote → greeting restart rule →
stutter rule → timeline). The mechanics stay; this plan changes what decides and how it is checked.

The creator's rule (2026-10-08): **no repeat and no false start may survive in what plays; the
deciding model must be the smartest available; this is the product's USP.** Good speech must still
not be lost (the asymmetry rule), so every extra removal below is gated by *"a kept line says the
same thing"* — nothing said only once is ever removed by the new passes.

## 1. Evidence from the last real cut (Raat3Baje ep2, project 656774a3, 2026-10-08 22:28)

| Measured | Value |
|---|---|
| Utterances / removed takes / trims / stutters by rule / restart cut | 513 / 201 / 52 / 100 / 119 |
| Answer parts **refused** (3 reads) | **105** — every sampled refusal is a trim whose quote copied the `[speech, no transcript, X s]` placeholder from the table (`editor.parse` matches quotes against words only). The model *did* find the stumbles; the code threw them away |
| Exact back-to-back phrase repeats **surviving in the kept stream** (scan of kept words, n ≥ 2) | **14**, e.g. 607 s "बारिश के बाद मिट्टी बारिश के बाद मिट्टी" (its trim was refused), 677 s "इस जोन में इस जोन में", 2065 s "मुझे एहसास हुआ कि अगर मुझे एहसास हुआ कि अगर" |
| Single-word repeats surviving | 10, all reduplication-protected words (कभी, बार, एक, अलग, जोर) — mostly legitimate |
| Words kept by a minority read (1 of 3 removed) | 146 words (~11 lines) — nobody asked a second question about them |
| Script hints in production | **none**: `editor_planner.plan_with_editor_sync` calls `utt.build(view, regions, audio=audio)` without `script_text`, although the project carries `script` and the bench uses it |
| Tokens / time | 167k in, 62k out, 193 s |

Bench baseline (2026-10-04, `editor_bench.py --product`): Life3Baje 2.3–6.2 s wrongly removed, 59–65 %
caught; Raat3Baje 8.7–11.3 s, 59 %. Target in §5.

## 2. Design: propose wide, verify strict, read it back

1. **Trims must work** (`asr/editor.py::parse`). Strip `[speech, no transcript, …]` tokens from the
   quote before matching; match on NFC-normalised tokens with punctuation removed; allow one edit per
   token when the quote has ≥ 3 tokens; when the matched word range spans an untranscribed hole, the
   hole is removed too (as a `removed_spans` entry). Refusals must drop from ~35 per read to ~0.
2. **Minority removals get a focused second question instead of silence**
   (`asr/editor_planner.py`, reuse `double_check._context` + `SECOND_OPINION`). A removal proposed by
   1 of 3 reads is a *candidate*; one call shows each candidate with ±3 lines, its notes and its
   "similar to" twin, and asks: *is this content said in a kept line (then remove), or is it the only
   place it is said (then keep)?* Majority removals still apply directly. No review panel (creator's rule).
3. **Retake sweep** (new `asr/retake_sweep.py`): a narrow prompt whose only job is to list groups of
   utterances that are attempts at the same content, run over chunks of ~80 utterances with 25 overlap
   plus every `similar to` pair. Groups not already removed go into the candidate set of step 2. The
   keep-choice follows the decided rule (later attempt wins unless the verifier says only an earlier one
   is complete).
4. **Deterministic phrase-repeat rule** (`asr/disfluency.py`, after the editor like the stutter rule):
   an exact back-to-back n-gram repeat (n = 2..8, spoken script, within one utterance or across kept
   neighbours with < 1.5 s between) cuts the earlier copy when its first copy ends in a hole, a
   cut-off note, or is immediately followed by the restart; in storytelling genres a clean, paused
   repeat ("वहाँ नीचे… वहाँ नीचे") is left to the verifier of step 2 instead. Reduplication list stays.
5. **Hindi sentence-completeness note** (`asr/utterances.py`): a measured note *"ends on a connective
   / postposition — unfinished"* when the last spoken token is a Hindi postposition (में से को के की का
   पर तक ने), a conjunction (और कि लेकिन मगर या जो जब अगर क्योंकि ताकि), an oblique pronoun (मुझे
   उसे हमें उन्हें) or an English connective (and but so because the a to of in that with). Measurement
   only; the model decides. Gives the false-start judgement a linguistic signal next to the acoustic one.
6. **Speaker style and dialect in the brief** (`editor.SYSTEM`, plus a `speaker_style` field the
   Studio sends on transcribe/recut from `writing_style.overview()`: fixed lines, catchphrases, dialect
   notes). Rules for the model: regional/dialect Hindi forms, code-switching and habitual phrases are
   style, never errors; never "correct" grammar; the greeting is said once; deliberate repetition for
   effect is kept; a retake is removed only when a kept line carries its content.
7. **Script hints in production**: pass the project's script (`spoken_script`) to `utt.build`.
8. **Final read, chunked and targeted** (`double_check.FINAL_READ`, ~80-line chunks with overlap):
   list what still plays twice, what breaks off unresumed, and seams a cut left ungrammatical; may
   remove only lines whose content is covered elsewhere, may restore removed lines. Then the repeat scan
   of step 4 runs once more as a guard.
9. **The model is a setting and is measured**: `auto_cut_editor.model` / `.effort`
   (`routes/settings.py`), default decided by the bench between `claude-fable-5-1` and
   `claude-opus-5-5` at medium and high. The proxy lists `claude-fable-5-1`; the editor already rejects
   answers served by any other model. (The bench run started 2026-10-08 was cut off by the usage limit;
   no numbers yet.)
10. **Report what the creator cares about**: `editor.repeats_remaining` (count of exact n-gram repeats
    in the kept stream, must be 0), `false_starts_remaining` (kept lines ending unfinished whose next
    kept line does not continue them), `refused`, `verified_candidates`; the Studio's production
    summary and `buzzedit_report` show them.

## 3. Phases and status (update the Status column after every item)

| # | Phase | Files | Done when | Status |
|---|---|---|---|---|
| 1 | Trim matching + holes inside trims | `asr/editor.py`, `tests/test_editor.py` | refusals ≈ 0 on a re-run of 656774a3; tests pass | built 2026-10-09, tests pass |
| 2 | Script hints in production; completeness note | `asr/editor_planner.py`, `asr/utterances.py`, tests | table shows `script P…` and the new note on ep2 | built 2026-10-09: `UNFINISHED_ENDINGS` note in utterances.py; `settings["script_text"]` from the project's script at both route call sites (popped after the call) → `utt.build(..., script_text=spoken_script(...))`; tests pass |
| 3 | Minority candidates → focused verify; retake sweep | `asr/editor_planner.py`, `asr/retake_sweep.py`, `asr/double_check.py`, tests | candidates logged with verdicts; no review list | built 2026-10-09: `retake_sweep.sweep` (80/25 chunks, union-find groups) + `double_check.verify_candidates` in the planner; verdicts noted on the words, report `candidates` / `verified_removed` / `verified_kept` / `sweep_groups`; no review list; tests pass |
| 4 | Phrase-repeat rule + repeats_remaining / false_starts_remaining in the report | `asr/disfluency.py`, `asr/editor_planner.py`, `asr/auto_edit.py::public_report` | ep2 re-cut reports 0 repeats | rule built 2026-10-09 in asr/repeats.py; planner integration built 2026-10-09 (up to 3 passes, storytelling repeats to the verifier, guard after the final read; `repeats_remaining` / `false_starts_remaining` in the report and as `public_report` warnings); tests pass; ep2 re-cut pending |
| 5 | Chunked final read | `asr/double_check.py`, `asr/editor_planner.py` | measured on the keys (§5) | built 2026-10-09: `double_check.final_read_chunked` (80/20 chunks, removes only with a kept covered_by, restart-rule lines never restored, restore wins) + stutter/repeat guard; tests pass; measurement on the keys pending |
| 6 | Style/dialect brief + `speaker_style` from the Studio | `asr/editor.py`, Studio `integrations/buzzedit_client.py`, `routes/transcription.py` | field reaches the prompt | **done 2026-10-09**: SYSTEM dialect/style rules; Studio `speaker_style` → transcribe → `settings` → `request()` |
| 7 | Model/effort setting + bench (Fable vs Opus, medium vs high, leave-one-out) | `routes/settings.py`, `tools/editor_bench.py` | table in §5 filled; default chosen by numbers | model/effort setting built 2026-10-09; bench pending |
| 8 | More answer keys: Beyond3Baje ep7 (`OXCC3995.MOV` 568 s → `ep7Final.mp4` 517 s), Life3Baje `MKLV3853.MOV` 697 s → `ep1/eptemp2.mp4` 473 s (pair to confirm with the matcher's `unmatched_s`) | `tools/autocut_truth.py`, `tools/build_style.py` | keys validated; examples rebuilt | keys built 2026-10-09 by the matcher: `raat3baje_ep2` (ep2Temp2.mp4 → the creator's ep2Final.mp4: 260 kept segments, 1033 s of speech removed, 97 s of the export unmatched), `beyond3baje_ep7` (1.9 s unmatched), `life3baje_ep2` (60 s unmatched, noted in the key); ep2's transcript cache taken from project 656774a3; transcription of the other two (GPU) and their style excerpts pending — stopped once because a live production needed the GPU |
| 9 | Acceptance: Re-cut Raat3Baje ep2 from the queue; creator watches | — | creator sees no repeat, no false start | bench against the creator's ep2 final cut (`raat3baje_ep2` key) 2026-10-09: §7 first run, §9 after the fixes; the queue re-cut waits for the creator (a production was running on the machine during this work) |

## 4. Implementation notes for whoever picks this up

- Run tests with `.venv/Scripts/python.exe -m pytest backend/tests/test_editor.py
  backend/tests/test_editor_planner.py backend/tests/test_utterances.py backend/tests/test_double_check.py -q`
  (32 passed on 2026-10-08 before any change).
- Repeat scan used for §1 (reusable for step 10): group kept words, normalise with
  `re.sub(r'[^\wऀ-ॿ]', '', t.lower())` on `word_native`, find `toks[i:i+n] == toks[i+n:i+2n]`.
- Project words store `start_frame`/`end_frame` (fps from `timeline.fps_num/fps_den`), not seconds.
- The proxy (`claude-local-api`, 127.0.0.1:8787) is at server-wide effort `low`; per-request
  `reasoning_effort` works. Pool size 3: three parallel reads use the whole pool, so run the sweep and
  the verify calls after the reads, not alongside.
- Keep majority rule and never reintroduce a review gate (creator's rule 2026-10-08).
- Implementation is to be done by Opus subagents per phase, one scoped brief each (creator's request).

## 5. Acceptance measures (per key, leave-one-out; fill before switching defaults)

| Measure | Target | Life3Baje ep1 | Raat3Baje ep1 | new keys |
|---|---|---|---|---|
| Repeats remaining in the kept stream | 0 | | | |
| False starts remaining | 0 | | | |
| Retakes / false starts caught (keys with kinds) | ≥ 95 % | | | |
| Good speech wrongly removed | ≤ 1 s / 10 min | | | |
| Refused answer parts | ≈ 0 | | | |

## 6. Model bench — measured 2026-10-09 (single pass, leave-one-out examples, trim fix in)

| Key | Model, effort | Wrongly removed | Caught | Cuts | Refused | Time |
|---|---|---|---|---|---|---|
| Life3Baje ep1 | Fable 5.1, medium | 11.1 s | 72.7 % | 46 | 0 | 722 s |
| Life3Baje ep1 | **Opus 5.5, medium** | **9.8 s** | **74.5 %** | 30 | 0 | 78 s |
| Raat3Baje ep1 | Fable 5.1, medium | 21.0 s | 72.8 % | 73 | 0 | 198 s |
| Raat3Baje ep1 | **Opus 5.5, medium** | **16.9 s** | 71.3 % | 73 | 0 | 128 s |

Fable 5.1 is not a better editor than Opus 5.5 on this task and is 2–9× slower, so Opus 5.5 at
medium stays the default (`auto_cut_editor.model` / `.effort` can still name Fable). "Caught" rose
from 66–73 % to 72–75 % against 2026-10-04 because trims now apply (0 refused); the extra wrongly
removed seconds are mostly stutters the creator left in, which the trims now cut. The smartness
the creator asked for comes from the pipeline (verify, sweep, final read), measured in §5.

## 7. Product path v3 — first measurement 2026-10-09 (3 reads + sweep + verify + phrase rule + chunked final read, 1 run each)

| Key | Wrongly removed | by reason | Caught | Cuts | Repeats left | False starts left | Steps failed | Time |
|---|---|---|---|---|---|---|---|---|
| Life3Baje ep1 (13.8 min) | 23.0 s | retake 12.3, stutter 9.9, false_start 1.0 | 77.3 % | 60 | 0 | 0 | none | 213 s |
| Raat3Baje ep1 (28 min) | 35.5 s | stutter 22.3, retake 10.7, repeat 2.3 | 73.2 % | 128 | 0 | 0 | none | 321 s |
| Raat3Baje ep2 (37 min, the creator's final cut, unseen) | 176.8 s | retake 141.7, stutter 36.7 | 90.9 % | 183 | 0 | 0 | none | 422 s |

Reading: no repeat and no false start survives on any key (the creator's first rule), and catch is up
(ep2 90.9 %). Two over-removals to fix before acceptance: (1) **the restart rule** — ep2 has
greeting-like lines at 0 s, 100 s and 516 s; the rule cut everything before 516 s, but the creator kept
235 s from before it (his restart was at 100 s): ~140 s of the ep2 "retake" seconds. Fix: a restart
must be confirmed by the opening content repeating after the later greeting (brief dispatched
2026-10-09). (2) **trims** — 118 / 211 / 241 trims per key; "stutter" seconds the creator kept are
partly stutters he left in (allowed by his new rule) and partly to be inspected with
`scratchpad/analyse_wrong.py` (per-take listing of wrongly removed words with reason and note).

## 8. Fixes after the first v3 measurement (2026-10-09)

- **Restart rule** (`asr/restart.py`): a greeting is "mera/main naam" or "naam hai" within 12 words of
  "dekh rahe" (the recogniser wrote the first ep2 greeting as "मैं नाम"); a later greeting confirms a
  restart only when the opening lines after it repeat the opening after an earlier greeting (character
  trigram cosine ≥ 0.5 on spelling-folded keys; two lines, or every line of the shorter opening). The
  greeting line itself is fixed and counts for nothing. On ep2 this now cuts U1–U17 (0–71 s, the
  abandoned first attempt) at the restart U18 (99 s) and leaves the kept greeting at 515 s alone.
- **Trim guard** (`asr/editor.py::parse`): a trim of 4+ words must be one copy of a phrase said twice in
  the same line (≥ 60 % token overlap with another window; holes between the copies are fine) and
  always removes the EARLIER copy; otherwise it is refused. The final read's trims are told to quote the
  first copy of a doubled phrase and never to tidy a sentence. (The long content trims measured in §7
  were exactly these: "मेरा चैनल है खयाल तीन बजे", "आपको एक ब्रश उठाना हो जाता है", …)
- **Failover retry** (`EditorClient.ask`): when the proxy serves another vendor's model (a transient
  "Couldn't confirm model" on Claude's side made two chunks of a run fall back to gpt-6-astra), the call
  is retried once after 3 s before the step is skipped.

## 9. Re-measure after the fixes — run 2026-10-09 20:22-20:55 (product path, Opus 5.5 medium, 1 run each)

| Key | Wrongly removed | by reason | Caught | Missed | Cuts | Repeats left | False starts left | Refused | Time |
|---|---|---|---|---|---|---|---|---|---|
| Life3Baje ep1 | 23.0 → **23.8 s** | retake 12.7, stutter 8.5, final_read 1.7, false_start 1.0 | 77.3 → **61.5 %** | 26.2 → 43.2 s | 60 → 67 | 0 | 0 | 1 → 40 | 208 s |
| Raat3Baje ep1 | 35.5 → **22.6 s** | stutter 9.8, retake 9.5, repeat 2.3, final_read 1.0 | 73.2 → **62.7 %** | 27.1 → 37.6 s | 128 → 126 | 0 | 0 | 0 → 45 | 325 s |
| Raat3Baje ep2 | 176.8 → **63.7 s** (second run 40.4 s) | retake 21.5 (7.5), stutter 22.8 (24.8), repeat 19.3 (3.6), final_read 0.7 (4.0) | 90.9 → **81.6 %** (79.4) | 26.0 → 52.4 s | 183 → 194 | 0 | 0 | 2 → 56 (63) | 534 s |

The second ep2 run and a second Raat3Baje ep1 run (20:49 / 20:55) saved the refusal notes and every
word's outcome (`detail` in the result file; `editor_bench.py` and `editor_planner.py` now keep them),
so the numbers can be read, not guessed:

- **The restart rule works**: ep2 "retake" over-removal 141.7 → 21.5 / 7.5 s, and no repeat or false
  start survives on any key.
- **The trim guard is a coin flip.** Every refusal but a handful is "not a repeated phrase" (ep2 55,
  ep1 46). Located against the creator's cut: ep2 24 of the refused trims were cuts he made and 29
  were not; ep1 11 were and 7 were not (28 not locatable). So the guard removes the long content
  trims of §7 (stutter seconds down on both Raat3Baje keys) but also blocks about half of the real
  retake trims, which is where the 10-15 point drop in "caught" and the +10-26 s "missed" come from.
  Next: a refused long trim should go to the verifier (one focused call: "is this a slip or the
  creator's sentence?") instead of being dropped outright.
- **Greetings are the creator's call, not a rule.** ep2 has greetings at 1 s, 99.8 s, 515.9 s and a
  sign-off-like line at 2198 s; the editor kept the 99.8 s one and removed the 515.9 s one, the key
  differs, and the creator said on 2026-10-09 that keeping two greetings in ep2 was deliberate. So
  the greeting seconds in the table are his choice, not an error to fix: no greeting rule beyond the
  restart rule of §8, and no greeting measure in §5.
- "Stutter" wrong seconds (ep2 24 s in 74 two-to-four-word stretches: "raat ki gehra ne", "liye liye
  yeh") are mostly stutters the creator left in, as §7 said; not a regression.
- The phrase-repeat rule alone, run offline on the raw ep2 words, removes only 3.2 s the creator
  kept; the 19.3 s of the first after-run was the 515 s greeting and a few two-word pairs.

Then re-cut ep2 from the Studio queue (Re-cut) and watch it.
