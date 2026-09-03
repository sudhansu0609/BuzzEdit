# Overnight Report — 2026-08-25 (auto-edit + B-roll over cuts)

**Watch first:** `data/output/e2e_ref6_presented.mp4` — your 195.6s recording cut to a
67.4s fluent programme, with B-roll covering the cuts, punch-outs on the rest, captions,
and generated video B-roll. Thumbnail: `data/projects/e2e_ref6/thumbnail.jpg`
("SPOTLIGHT EFFECT KO SAMJHEIN"). Open project `e2e_ref6` in the app to tweak anything.

## Why the cuts were terrible, and what changed

The language-model layer — the one thing that can judge "is what remains a complete
Hindi sentence?" — **had never actually worked on this machine**. Asked in prose,
gemma-4-12b (the biggest model that fits your GPU) ignored the format on every window
and paraphrased, so every edit you saw was the dumb structural fallback: right about
pile-ups, blind to meaning, and it shipped spliced half-sentences.

Six full validation runs on your real recording later (`data/projects/e2e_ref*_e2e/`
holds every run's audit trail), the pipeline now:

1. **Asks for cut spans as schema-constrained JSON** — the same model now answers
   correctly on every window, and every answer is still verified word-by-word against
   the transcript before a single cut lands.
2. **Refuses model mistakes selectively instead of wholesale** — an oversized run or a
   "retake" whose surviving copy doesn't exist (the model once deleted BOTH tellings of
   your t-shirt story) is refused alone; the structural decision stands there.
3. **Never lets a sloppy model boundary splice fragments** — restores only apply to
   whole attempts the model kept, the second-guessing pass is gone, and repairs can
   never delete numbers, "%" (spoken "percent"), long runs (it once deleted the
   researchers' names), your sign-off, or Hindi reduplication ("अपने-अपने").
4. **Sees retakes across Whisper's script flips** — "थॉमल गिल्गोविच" and "Thommel
   Gilgovich" are now the same words, so doubled tellings finally collapse.
5. **Knows Hindi grammar frames** — "…गया हो या फिर…" repeating through a list is no
   longer "a retake" (it once tore the frame out of your जूता/बाल examples).

Run 6 result: every story beat present and told exactly once, `cut_verify: clean`
(frame-accurate render), 0 leaked fumbles. Remaining oddities in the transcript panel
are Whisper mis-spellings — the audio underneath is your real voice and plays fine.

## B-roll now hides the cuts

- Every cutaway window **slides to straddle the nearest jump cut** (≥0.35s cover each
  side), so the switch to B-roll and back happens inside continuous takes. Tonight's
  render: **18 of 25 jump cuts are invisible under B-roll** (60.3% coverage).
- The other 7 cuts get the classic disguise: every zoom now **pushes in**, so each cut
  lands back at wide — a deliberate punch-out instead of a bare head-jump. (The old
  alternating zooms made scale *continuous* across every cut, showing the jump plainly.)
- Wan 2.2 **video B-roll works now** — a one-line bug in reading ComfyUI's job history
  (`animated: [true]` flags read as files) had crashed every video generation and
  tripped the give-up breaker, which is why past nights produced so few pictures.

## Also fixed along the way

- `Project.status` didn't allow "presented" → the thumbnail stage crashed every night.
- A warm ComfyUI blocked the LLM from loading on any second pass → the launcher now
  asks it to release VRAM first.
- The thumbnail workflow pointed at an uninstalled SD1.5 checkpoint → now Juggernaut-XL.
- Multi-line DELETE answers were misread as rewrites (`re.MULTILINE`).

## State

- **546 backend tests pass** (33 new tonight, each locking in one of the fixes above).
- Docs updated: `AUTO_EDIT_IMPROVEMENT_PLAN.md` §2.1b + §6 (live measurements),
  `PRESENTATION_PASS_PLAN.md` addendum (cut-hiding design).
- Nothing is committed — the working tree holds tonight's changes plus your earlier
  uncommitted work. Review and commit when you're happy.
- Heads-up: Whisper transcribes this footage a little differently every run, so two
  auto-edits of the same file will differ slightly. Judge changes across runs.
