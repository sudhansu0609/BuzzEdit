# Automation Roadmap — "feed the video, get the edit"

**Status 2026-09-03 (evening): rows 1–10, 12 and 13 of §7 are IMPLEMENTED** in the
working tree — see `DRESSING_REPORT.md` for what was built, how it was verified on the
real recording, and how to use it. Row 11 (character consistency, 2.5D parallax) needs
ComfyUI workflows and is the one item still open. §4.4's picture-in-picture, split screen
and freeze-frame, §2.2's emphasis words and bilingual line, §1.1's per-act music swaps and
§5.5's style-profile hook were added later the same day (`presentation/composite.py`,
`captions.py`, `sound.music_sections`, `director._apply_style_profile`). Everything below this line is the original
plan and stays as the reference for intent.

Written 2026-09-03 against the working tree. Companion to
`AUTO_EDIT_IMPROVEMENT_PLAN.md` (the cut) and `PRESENTATION_PASS_PLAN.md` (B-roll,
zooms, captions, title, atmosphere). This document lists what the unattended pass
does NOT yet do, ranked by how much each item changes the finished video for the
two target formats: **infotainment explainers** and **horror story narration**.

Every item names the modules it touches so a future session can pick one up cold.

---

## 0. What already runs unattended (so nothing here re-does it)

| Stage | Where | State |
|---|---|---|
| Fumble/retake/pause cut, verified word-by-word | `asr/*`, `timeline/ops.py` | done |
| Genre detection (15 genres) | `presentation/genre.py` | done |
| Topics → beats → ComfyUI stills + Wan2.2 video | `presentation/shotplan.py`, `assets.py` | done |
| B-roll placed to hide jump cuts, Ken Burns | `presentation/placement.py` | done |
| Face punch-ins, push-in per segment | `presentation/facezoom.py` | done |
| Word-timed captions (5 presets, native script) | `timeline/authoring.py` | done |
| Opening title, topic pop-ups | `director.py`, `shotplan.topic_popups` | done |
| One genre atmosphere layer | `genre.atmosphere_for` | done |
| Thumbnail | `agents/thumbnail_agent.py` | done |
| Reference-video style profile (cut rate, motion, colour, caption zones) | `style/*` | built, **not wired into the pass** |

What the schema can already draw but nothing plans automatically: full colour
grade with wheels, chroma key, adjustment layers, compound clips, transitions
(the whole xfade catalogue), rain/snow/lightning/fog/wind/grain/light-leak
layers, aspect bars, text with fade/pop/slide-up, A2+ audio mix tracks.

**The gaps, in one line each:** no sound at all beyond the voice; text is limited
to drawtext's four animations; no maps, charts or data cards; grade and
transitions are not chosen by context; the story's structure (setup → reveal →
climax) does not shape the edit; and the user's script is never read.

---

## 1. Sound design (largest gap; horror does not work without it)

The programme currently ships with the raw voice track and nothing else. For
horror this is the single biggest missing piece; for infotainment it is what
separates a talking head from a produced video.

### 1.1 Music bed, genre-picked, ducked under speech
- **Library:** `data/music/<genre>/*.mp3|wav` plus an optional `manifest.json`
  with `mood`, `bpm`, `intensity` tags. Loop or trim to programme length, 1.5s
  fade in / 3s fade out.
- **Ducking:** the timeline already stores `speech_regions` and
  `energy_envelope`, so ducking can be a deterministic `volume` expression
  keyed to speech (−12 dB under speech, −4 dB in pauses, 150 ms ramps) rather
  than `sidechaincompress`, which is harder to tune and pumps.
- **Structure-aware swaps:** a different cue per act (see §5.2) with a 2s
  crossfade at the topic boundary; horror ramps intensity toward the climax.
- **Generation fallback:** ComfyUI 0.33 runs ACE-Step / Stable Audio Open, so
  a `music_generation.json` workflow in `workflows/` can produce a bed from a
  genre prompt when the library is empty. Same manifest mechanism as B-roll.
- **Touches:** new `presentation/sound.py`; `PresentationSettings.music`,
  `music_volume`; `compiler.py` already mixes A2+ items with `amix` (line ~336),
  so placement is an `add_media_item` on `A2` with `origin="presentation"`.

### 1.2 SFX on beats
- Whoosh on every B-roll in/out and every windowed punch-in; soft "pop" on text
  pop-ins; riser under the last 3s before a reveal; stinger/boom on horror
  scare beats; heartbeat loop under the climax.
- The planner tags beats: `Beat.sfx: Optional[str]` from a fixed vocabulary
  (`whoosh`, `riser`, `stinger`, `boom`, `heartbeat`, `pop`, `none`); the
  library is `data/sfx/<tag>/*.wav`, chosen by seeded random so re-runs are
  stable.
- Volume relative to the voice, not absolute: measure voice LUFS once
  (`ebur128`), place SFX at −8 dB relative.

### 1.3 Ambience tied to the visual atmosphere
- Horror already gets `fog` visually; pair each atmosphere type with a loop
  (`wind`, `rain`, `room_tone`, `crickets`) at −22 dB. One mapping table next
  to `genre._ATMOSPHERE`.

### 1.4 Voice mastering
- `afftdn` denoise, `deesser`, gentle `acompressor`, then `loudnorm` to −14 LUFS
  (YouTube's target) on the final mix. All ffmpeg-native; goes on the tail of
  the audio chain in `compiler.py`. Also expose a `voice_preset` (`clean`,
  `podcast`, `horror_intimate` = slight low-shelf boost + more compression).

---

## 2. Text and typography

drawtext gives four animations and one colour per element. Most "YouTube text"
effects need per-word colour, per-character reveals and transforms.

### 2.1 Switch the caption/text engine to ASS (libass)
This one change unlocks most of the list below. ffmpeg's `subtitles`/`ass`
filter renders Advanced SubStation Alpha with karaoke (`\k`), per-word colour,
`\t` animated transforms (scale, rotation, colour), `\move`, `\fad`, `\blur`,
`\be` glow, outlines and shadows — all shaped by HarfBuzz, so Devanagari keeps
working. Fonts come from a `fontsdir=`, which also sidesteps the fontfile
segfault trap. drawtext stays for single static titles; anything animated is
written as one `.ass` file per track and burned in with one filter.
- **Touches:** new `render/ass.py` (writer), `TextStyle.animation` gains
  `typewriter`, `karaoke`, `glitch`, `shake`, `scale_in`, `blur_in`; `TextStyle`
  gains `highlight_color`. `compiler.py` swaps the text chain when any item
  needs ASS.
- Verify: burn a 10s test with Nirmala UI Devanagari, compare with the current
  drawtext render pixel-for-pixel on timing (the ±0.08s measurement stands).

### 2.2 Caption upgrades (once ASS exists)
- **Active-word highlight** (karaoke colour) — the standard Shorts look.
- **Emphasis words:** the planner marks 1–2 words per sentence to render larger
  or in the accent colour: numbers, "%", names, negations. `verify.PROTECTED_WORDS`
  and the number rules already identify most of these without a model call.
- **Bilingual line:** `TranscriptSegment.text_english` exists; an optional
  second, smaller line for Hindi projects.

### 2.3 New text element kinds the planner can place
| Kind | When | Look |
|---|---|---|
| Location/date card | horror: "Rajasthan, 1987"; any place+year in the transcript | top-left, typewriter, serif, fades over 4s |
| Character card | horror: first mention of a named person | name + one-line role, lower third |
| Source citation | infotainment: "study", "research", "university", a year | small lower-third, `Source: …`, on for 3s |
| Chapter title | every topic boundary from `find_topics` | full-width slam or whip-in, optional dip-to-black in horror |
| Stat callout | a number with a unit or "%" | big animated counter 0→N via ASS `\t` or drawtext `%{expr}` |
| Definition card | "X ka matlab hai …" / "X means …" | boxed term + one-line gloss |
| Quote card | "unhone kaha", quotation marks in the script | italic serif, attribution line |
| End screen | last 15–20s | subscribe + "watch next" zones left clear (YouTube end-screen safe area) |

All of these are `Beat` kinds. `BeatKind` grows from four to ~twelve and
`write_beats`' schema gets a per-kind `text` field; placement rules live next to
`place_popups`.

### 2.4 Horror text treatments
Flicker (alpha noise), glitch (two offset copies in red/cyan for 3 frames),
scale-in from 130%, blur-in, and a "whisper" style (low opacity, letter-spaced,
tiny) for the words the narrator lowers their voice on — the energy envelope
already exposes those moments (`emphasis_z` strongly negative).

---

## 3. Maps, charts and data graphics

### 3.1 Map cutaways (asked for explicitly)
- **Entity extraction:** one model call per topic returns `places[]`, `dates[]`,
  `numbers[]`, `people[]` as JSON (schema-constrained, same `ask_with_schema`
  path Gemma answers correctly).
- **Geocoding offline:** bundle GeoNames `cities1000` (~30 MB) plus a country
  centroid table in `data/geo/`; no API key, no network. Fuzzy match on
  romanised names (the transliterator in `asr/transliterate.py` helps here).
- **Rendering offline:** Natural Earth 10m shapefiles + `geopandas`/`matplotlib`
  (or `py-staticmaps` with cached tiles) to a 1920×1080 PNG at two or three
  zoom levels (country → region → city) with a pulsing marker; composite as a
  cutaway with a Ken Burns push from the wide level to the tight one, or xfade
  the levels. Two style sheets: `clean` (infotainment: light land, blue water,
  bold label) and `noir` (horror: near-black land, grey borders, red marker,
  grain layer on top).
- **Route maps:** two or more places in one topic → animated line between them.
- **Touches:** new `presentation/maps.py`, `Beat.kind="map"`, asset kind
  `image` so `placement.place_broll` needs no change.

### 3.2 Charts from spoken numbers
When a topic contains two or more comparable numbers ("40% vs 12%", "2010: 5
crore, 2020: 20 crore") render a bar/line chart card with matplotlib, animate
the grow-in by rendering 12 frames to a short PNG sequence or video asset. Only
fire when the model can name the axis; otherwise fall back to a stat callout.

### 3.3 Timelines
Three or more years in a topic → a horizontal timeline strip with the years and
a marker that slides as each is spoken (word timings give the exact moments).

### 3.4 Fix and use the `graphic` beat kind
Graphics are off because an animated transform pads transparent PNGs with
black. Fix: composite graphics through `build_overlay_transform` with the alpha
kept and no `zoompan`; animate position only. Then icons, arrows, highlight
circles and "X vs Y" split-screen labels become possible.

### 3.5 Comparison / split screen
"A vs B" phrasing → two generated images side by side with a centre divider and
labels; both halves are `broll_image` beats with fixed `Transform` halves.

---

## 4. Effects, grade and transitions chosen by context

### 4.1 Genre master grade (trivial, high visibility)
`COLOR_PRESETS` already has `moody`, `punchy_talking_head`, `teal_orange`,
`cinematic`, `vintage`. Map genre → master grade the way `_ATMOSPHERE` maps
genre → layer: horror `moody` + vignette 0.5, infotainment/science
`punchy_talking_head`, true crime `cinematic` + desaturate, devotional `warm`.
Tag `preset="presentation:<genre>"` so re-runs replace and user grades survive.

### 4.2 Mood per topic (adjustment layers)
The planner tags each topic with `mood ∈ {calm, build, tense, reveal, climax,
aftermath, comedic, hopeful}`. Each mood is a small recipe applied as an
adjustment item over the topic's span:
- `tense`: −10% saturation, +0.15 vignette, slow 6% push-in, fog +0.1.
- `reveal`: 3-frame dip-to-white on the first word of the sentence, whoosh.
- `climax` (horror): lightning layer windowed to the span, thunder SFX at the
  sentence with "achanak / suddenly / phir", 2-frame black flash at the peak,
  heartbeat loop.
- `aftermath`: desaturate to 60%, slow pull-back, music fades.
- `hopeful` (motivational/infotainment): warm shift, sunlight layer.

### 4.3 Horror effect pack (render-level primitives to add)
| Effect | ffmpeg | Note |
|---|---|---|
| Camera shake | `crop` with `random()` x/y expressions on an adjustment window | 0.4–0.8s bursts on stingers |
| RGB split / glitch | `rgbashift` + `tblend` for 2–4 frames | on reveals and text glitches |
| VHS / found-footage | `noise` + scanlines via `geq` + slight `chromashift` + 4:3 bars | one look preset, "flashback" tag |
| Flash frame | 1–2 frame white or red `color` overlay | pairs with boom SFX |
| Slow motion on video B-roll | `setpts` 0.5× + `minterpolate` | Wan clips are short; doubling them helps coverage too |
| Black-and-white flashback | `ColorGrade.saturation=0` + grain on an adjustment layer | for past-tense stretches the model tags |
| Light flicker on stills | alpha oscillation via `lutrgb` expression | horror stills only |

### 4.4 Infotainment effect pack
- **Freeze-frame + callout:** freeze the speaker on a key claim (`select`/
  `loop` on the frame), draw the stat over it, 1.5s, then resume.
- **Picture-in-picture:** speaker in a corner while B-roll fills the frame, for
  beats where continuity of the speaker matters (the model marks "keep speaker").
  Needs V1 duplicated as a V3 item with a corner `Transform`.
- **Highlight circle / arrow:** generated graphic over a B-roll still at a
  planner-given normalised (x, y) — most useful on diagrams and maps.
- **Speed ramp** into whip-cut transitions at topic changes.

### 4.5 Transitions picked per join
`default_transition` is one global choice. Instead: hard cut for intra-topic
jump cuts (already hidden), `whip_left`/`zoom_in` at infotainment topic
changes, `dip_to_black` (0.7s) at horror chapter boundaries, `blur` into
flashbacks, `pixelize`/glitch into reveals. Implemented in a small
`presentation/transitions.py` that reads topics + moods and calls
`clip_ops.set_transition` on the first V1 item of each topic.

### 4.6 Better image "edits" on generated stills
- **2.5D parallax:** a ComfyUI depth workflow (Depth Anything) produces a depth
  map per still; a displacement pass renders a 3s parallax clip. Turns a still
  into "video" cheaply — far cheaper than Wan and much better than Ken Burns.
- **Character consistency for horror:** stories have a recurring protagonist.
  Generate one character sheet per named person (IP-Adapter/PuLID workflow in
  ComfyUI), feed it into every still that mentions them. Without this the same
  "Ramesh" is a different face in every cutaway.
- **Per-still atmosphere:** fog/grain windowed to the still rather than the
  whole programme, so the speaker's shots stay clean.

### 4.7 Auto-reframe to 9:16 (distribution, not the edit itself)
YuNet face detection already runs per segment. Track the face per frame at 4 fps,
smooth it, and export a vertical variant with a face-following crop, captions
moved to the safe centre band, and B-roll centre-cropped. Also cut the top 3
topics (by `priority`) as standalone Shorts with their own hook title.

---

## 5. Planning intelligence (what makes the above land in the right places)

### 5.1 Read the script when there is one
The user often has the script. Accept `script.txt|md` on the project; align it
to the transcript with the existing `asr/forced_align.py`. Wins:
- Captions use the script's spelling, killing Whisper's misspellings outright.
- Names, places, dates and numbers come from the script, not from the model
  guessing over noisy ASR.
- Topic boundaries follow the script's paragraphs.
- Stage directions in the script (`[map: Jaipur]`, `[sfx: thunder]`,
  `[broll: old haveli at night]`) become beats directly — an optional
  bracket vocabulary the user can learn once.

### 5.2 Story structure
One model call over the topics returns acts: `hook, setup, build, reveal,
climax, aftermath, cta` (horror) or `hook, context, point_1..n, takeaway, cta`
(infotainment). The act drives B-roll density (`target_coverage` per act,
sparse in setup, dense in build, mostly speaker at the reveal so the face
carries it), music intensity, transition choice and the effect recipes in §4.2.

### 5.3 Cold open
Pick the single most gripping sentence (model choice, constrained to a
complete sentence from the transcript) and play it first, dip-to-black, then
the title, then the programme from its real start. Items can be reordered on
the timeline today; the sentence is a copy of an existing V1/A1 span.

### 5.4 Entity extraction feeds everything
One schema-constrained call per topic → `people, places, dates, numbers,
terms, quotes, sources, comparisons`. Every text kind in §2.3 and every
graphic in §3 keys off this list, so it is worth doing once and storing in the
shot plan JSON next to the beats.

### 5.5 Wire the style profile into the pass
`style/apply.py` can already impose a reference video's cut rate, push-in
frequency, grade and caption position. Add `PresentationSettings.style_profile`
so "edit like this channel" is one setting; the pass calls `apply_profile`
after placement.

### 5.6 YouTube metadata pack
From topics: 3 title options, description with timestamps (chapters), tags,
and a `chapters.txt`. Written next to the report; zero render cost.

### 5.7 Presentation verification (the morning report's missing half)
`cut_verify` checks the cut; nothing checks the presentation. Sample the render
at 2 fps and assert: speaker on screen ≥ (1 − coverage) of the time, no text
box intersecting the face box (`style/text_regions.py` finds text; YuNet finds
faces), caption contrast against the background above a floor, no two text
elements overlapping, music never louder than voice. Failures go in
`PresentationReport.degraded`.

---

## 6. Render engine notes (before the list above grows the filtergraph)

- Every layer today is one more branch in a single ffmpeg filtergraph. Maps,
  charts, animated counters, parallax and ASS text should be **rendered to
  assets first** (PNG, PNG sequence or short ProRes/H.264 clip under the
  project's assets dir, exactly as `assets.py` treats generated B-roll) and
  composited as ordinary media items. The compiler then never learns a new
  filter for them, and each asset caches by prompt hash.
- Audio gains an `A2` music item, an `A3` SFX lane and an `A4` ambience lane,
  all `origin="presentation"`, so `clear_generated` removes them on re-run.
- GPU contention is already brokered (`runtime/gpu_broker.py`); music
  generation and depth maps are ComfyUI jobs and queue behind B-roll.
- 32 GB RAM: parallax and chart rendering are CPU; keep them after the LLM is
  ejected (`_eject_llm`).

---

## 7. Suggested order

| # | Item | Why first | Status |
|---|---|---|---|
| 1 | Music bed + ducking + loudnorm (§1.1, §1.4) | biggest audible jump, all ffmpeg | done — `presentation/sound.py`, `render/audio.py` |
| 2 | Genre master grade + per-join transitions (§4.1, §4.5) | trivial, visible in every frame | done — `presentation/look.py` |
| 3 | Script ingestion (§5.1) | fixes caption spelling and feeds every later item | done — `asr/script_align.py`, `presentation/script.py`, `PUT /api/projects/{id}/script` |
| 4 | Entity extraction + text kinds: location/date, source, chapter, stat (§5.4, §2.3) | infotainment and horror both need these | done — `presentation/entities.py`, `cards.py` |
| 5 | ASS text engine + karaoke captions + horror text treatments (§2.1, §2.2, §2.4) | unlocks every animated text effect | done — `render/ass.py`, `karaoke_pop` caption preset |
| 6 | SFX + ambience (§1.2, §1.3) | rides on the beat tagging from #4 | done — synthesised fallbacks in `sound.py` |
| 7 | Mood per topic + horror effect pack (§4.2, §4.3) | the "horror edit" proper | done — `presentation/mood.py`; flash/shake/glitch/vhs/flicker in `render/atmosphere.py` |
| 8 | Maps (§3.1) | asked for; offline geodata is the only real work | done — `presentation/maps.py`, Natural Earth in `data/geo/` |
| 9 | Story structure + cold open (§5.2, §5.3) | shapes density and music | done — `presentation/structure.py` |
| 10 | Charts, timelines, graphic fix, split screen (§3.2–3.5) | infotainment polish | done — `charts.py`, `placement.py`, split screen in `composite.py` |
| 11 | Character consistency + 2.5D parallax (§4.6) | horror story quality | **open** — needs ComfyUI workflows |
| 12 | Presentation verification (§5.7) | keeps overnight runs honest as layers pile up | done — `presentation/verify.py` |
| 13 | Auto-reframe Shorts + metadata pack (§4.7, §5.6) | distribution | done — `presentation/shorts.py`, `structure.write_metadata` |

Each row is independent enough to be one overnight session, and each should
end the way the last ones did: a real render of `IMG_E2043.MOV`, judged across
several Whisper runs, with tests locking in the behaviour.
