# Dressing Report — 2026-09-03 (the automation roadmap, implemented)

**Watch first:** `data/output/e2e_ref6_dress_presented.mp4` — your real recording, dressed by
the new pass with **no language model and no ComfyUI running**: karaoke Devanagari captions
with the spoken word lit up, the horror grade, a synthesised drone and wind under a mastered
voice, whooshes on every cutaway, a chapter title, an end screen, chapter transitions, and
the previous night's B-roll left in place. With LM Studio up, the same pass also reads
names, places, numbers and quotes into cards and maps, tags every topic's mood and act, and
opens on the most gripping line. Open project `e2e_ref6_dress` in the app to see every layer
as an editable clip.

Everything in `AUTOMATION_ROADMAP.md` §7 rows 1–10, 12 and 13 is built, tested (646 backend
tests, up from 555), and wired into the app's Agent panel. Row 11 is open (see the end).

---

## What the pass now does, in the order it runs

| Stage | What it adds | Where |
|---|---|---|
| **Script** | Aligns your script to the transcript: fixes caption spelling ("sabsakraaib" → "subscribe"), makes its paragraphs the topics, and turns `[map: Jaipur]`, `[sfx: thunder]`, `[broll: …]`, `[stat: 25%]`, `[chapter: …]`, `[quote: … \| who]`, `[mood: tense]`, `[title: …]` into beats. `# Heading` lines are chapters. | `asr/script_align.py`, `presentation/script.py`, `PUT /api/projects/{id}/script`, Agent panel textarea |
| **Structure** | Tags each topic with an act (hook, setup, build, reveal, climax, aftermath, cta). Acts weigh the B-roll budget: dense in the build, the speaker's face at the reveal, almost nothing on the call to action. Picks the hook sentence. | `presentation/structure.py` |
| **Entities** | One schema-constrained call per topic (patterns without a model): people, places, dates, numbers, terms, quotes, sources. Each anchored to the second it is spoken. | `presentation/entities.py` |
| **Moods** | Each topic tagged calm / build / tense / reveal / climax / aftermath / comedic / hopeful, with a key moment. | `presentation/mood.py` |
| **Maps and charts** | A map cutaway for the first place in a topic, drawn offline from Natural Earth (cities, countries, states; `clean` and `noir` looks, routes between places). Bar charts and timeline strips for spoken numbers and years. | `presentation/maps.py`, `charts.py`, `data/geo/` (`python tools/fetch_geo.py`) |
| **Cards** | Chapter titles, stat call-outs (a number counting up), location/date cards (typewriter), character lower-thirds, source citations, definition and quote cards, an end screen. Zoned so a corner card and a centre card can share a moment but two centre cards cannot. | `presentation/cards.py`, presets in `timeline/presets.py` |
| **Animated text** | libass engine: karaoke word highlight, typewriter (whole Devanagari clusters), scale-in, blur-in, glitch, shake, flicker. Plain clips stay on drawtext. Captions default to `karaoke_pop`. | `render/ass.py`, `TextStyle.animation`, `TextClip.words` |
| **Mood recipes** | Per-topic adjustment layers: grade shift, slow push or pull, fog / lightning / flicker windows. At the key moment in horror and true crime: flash frame, camera shake, glitch, thunder or stinger, a riser leading in, a heartbeat loop under tense and climax. Other genres get half-strength grades and no hits. | `presentation/mood.py`, new treatments `flash`, `shake`, `glitch`, `vhs`, `flicker` in `render/atmosphere.py` |
| **Look** | A master grade per genre (moody for horror, punchy for explainers, …) and a transition at each chapter boundary that lands on a cut (dip to black for horror, a whip for explainers). Hand-set grades and transitions are never touched. | `presentation/look.py`, `genre.grade_for`, `genre.topic_transition_for` |
| **Cold open** | The hook sentence plays first (copied onto V2/A5 in the programme offset, so transcript rebuilds keep it), a beat of black, then the video. Karaoke caption on the hook, the bed starts under it. | `structure.apply_cold_open`, `Timeline.cold_open_frames` |
| **Sound** | Music bed from `data/music/<genre>/` (or a synthesised drone for horror / true crime), ducked under the voice by a sidechain compressor; whoosh on every cutaway, pop on text, stingers from the moods; wind / room-tone ambience; voice denoise + de-ess + compression; the mix normalised to −14 LUFS. All synthesised fallbacks are made once with ffmpeg into `data/audio_synth/`. | `presentation/sound.py`, `render/audio.py`, compiler audio path |
| **Composites** | Picture-in-picture: the speaker in the top-right corner over cutaways longer than 4 s (explainer genres; `pip: "on"` forces it). Split screen: "A vs B" as two stills side by side with labels, from a `[split: A \| B]` direction or a comparison the entity pass found. Freeze-frame: the speaker freezes for 1.4 s under each stat call-out while the number counts up. | `presentation/composite.py` |
| **Caption extras** | Numbers and stressed words (from the loudness envelope) drawn larger in the accent colour, two per card at most. An English line under Indic captions, translated sentence by sentence while the model is up (or from the transcript's own English). | `presentation/captions.py`, `TextClip.second_line` |
| **Per-act music** | The bed changes cue at act boundaries with a 2 s crossfade: calm for the setup, tense for the build, the drone rising at the climax, soft for the aftermath. Library moods come from `manifest.json` tags; the synthesised beds have `drone`, `drone_high` and `drone_low` variants. | `sound.music_sections`, `ACT_MUSIC` |
| **Style profile** | `style_profile: <id>` applies a measured reference video (look, transitions; motion only when the face zooms are off) after the genre look. | `director._apply_style_profile`, `style/apply.py` |
| **Shorts** | The best three topics as 9:16 clips; a landscape source is reframed to follow the speaker's face. | `presentation/shorts.py` |
| **Verification** | Text off the face (using the face anchors), cards not stacked, the bed ducked and quiet, every cutaway file present; on the render: loudness against the target, picture length against sound. Failures land in `degraded` and in the panel as amber warnings. | `presentation/verify.py`, `presentation_checks.json` |
| **Listing** | `metadata.json` (title options, description with chapter timestamps, tags) and `chapters.txt`. | `structure.write_metadata` |

Every layer is tagged by origin (`music`, `sfx`, `card`, `mood`, `coldopen`, …) so a re-run
replaces its own work and leaves anything you placed by hand.

## Verified on the real recording

Two full runs of `tools/dress_smoke.py e2e_ref6 horror` (a copy of your project, model and
ComfyUI deliberately off, so every layer that does not need generation is exercised):

- 144 s for the main render on the CPU encoder; 304 s with two Shorts cut and rendered as
  well (`e2e_ref6_dress_short1.mp4`, 21.7 s, and `_short2.mp4`, 45.8 s, both 1080×1920).
- All seven checks passed on the second run: text clear of the face (84 text items against
  6 face boxes), nothing stacked, bed ducked, every asset present, the mix at −14.2 LUFS
  against the −14 target with a −1.3 dBFS peak, picture and sound the same length.
- 81 karaoke caption cards in Devanagari, 23 sound effects, drone bed + wind, `moody`
  grade, fog, one chapter transition, chapter title and end screen, 17 segment zooms,
  faces found on 23% of segments.
- Frames pulled at 1.5 s, 8 s, 20 s, 33 s, 45 s and 60 s all show the layers where they
  should be; the ducking was measured on a synthetic test (bed 4+ dB lower under speech).
- Sample maps (Jaipur noir, Delhi→Mumbai route, Rajasthan highlighted, India on a portrait
  canvas) and a chart are in the session scratchpad and look right.

## Three bugs found on the way, all fixed

1. **Face detection had never worked.** `data/models/face_detection_yunet_2023mar.onnx` was a
   131-byte Git LFS *pointer*, not the model. Replaced with the real weights (fetched from the
   LFS media endpoint); the loader now refuses pointer files. OpenCV 5 finds your face at 94%.
2. **A render could hang for ever at frame 0.** The runner only drained ffmpeg's stderr at the
   end; a chatty filter fills the pipe and ffmpeg blocks silently. It now drains concurrently,
   and libass is no longer pointed at `C:\Windows\Fonts` (the trigger).
3. **Vertical footage was letterboxed into 1920×1080.** The project's default resolution wins
   only when its orientation matches the footage.

## How to use it

- **Agent panel → Dressing:** eight switches (music + ambience, effects, cards, maps,
  moods + hits, story structure, genre grade + transitions, verify) and a voice preset. All on
  by default. The report table below them shows cards, sound, look, story, script and any
  failed checks.
- **Script:** paste it in the panel and press *Align script to transcript*. Or
  `PUT /api/projects/{id}/script {"text": …}`.
- **Your own music / effects:** drop files in `data/music/<genre>/`, `data/sfx/<tag>/`
  (`whoosh`, `pop`, `boom`, `riser`, `stinger`, `heartbeat`, `thunder`, `flash`, `click`),
  `data/ambience/<kind>/` (`wind`, `rain`, `room_tone`). They take precedence over the
  synthesised ones. An optional `manifest.json` tags music by genre and mood.
- **Map data:** `python backend/tools/fetch_geo.py` once per machine (48 MB, public domain).
- **Every setting** is a field on `PresentationSettings` (`presentation/models.py`) and can be
  sent in the `settings` body of `/api/presentation/{id}/start`.

## Still open

- **Row 11:** character consistency across horror stills (an IP-Adapter / PuLID workflow) and
  2.5D parallax from depth maps. Both need ComfyUI workflows in `workflows/` before there is
  anything to wire.
- ComfyUI music generation as a bed source (the library and the synthesised beds are the
  only sources today).
- The Shorts count and the full vertical export are settings fields only (`shorts_clips`,
  `shorts_full`); the Agent panel has selects for the speaker corner, the English line and
  the style profile, but not for those two.
- Nothing is committed. The tree holds tonight's work on top of your earlier uncommitted
  changes; `git status` lists 19 new modules and 11 new test files.
