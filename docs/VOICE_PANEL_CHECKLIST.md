# Voice panel — manual smoke checklist (VOICE_CHANGER_PLAN.md brief S2b)

There is no browser test harness, so the panel is checked by hand. It covers every SkinWalker
capability (plan §1) from inside BuzzEdit: record and manage reference voices, convert a whole
file, batch, selective conversion of timeline regions with FX, and WAV / MP3 output.

**Before you start**

- The backend must have the voice routes (`routes/audio_voice.py`, briefs S1 and S2a), the media
  upload route `POST /api/projects/{id}/media/upload` and the download route
  `GET /api/media/download`. A backend started before they landed answers with a bare 404; the
  panel then shows yellow "not available on this backend yet" notes (the microphone source says
  "upload route not available yet") instead of failing. Restart BuzzEdit once.
- seed-vc must resolve (`GET /api/settings/seedvc`; today it is SkinWalker's `.venv` +
  `vendor/seed-vc`). The first conversion loads the model on the GPU (about 30 s; the very first
  run on a machine downloads about 3.2 GB of checkpoints). No render may be running: every voice
  route answers 409 during a render.
- Build the frontend (`npx vite build`) or run `npm run dev`.
- Use a project whose A1 has at least one cut (delete a word, or run Auto Edit), with two short
  audio or video files in the media pool, and have a microphone connected.

| # | Do this | Expect |
|---|---|---|
| 1 | Open **Layout menu → Voice**. | A floating **Voice** window with the chips **Convert / Batch / Regions / Voices** and a one-line hint under them. No yellow notes. |
| 2 | Open the **Voices** tab. | The worker line says *idle, no model loaded*; the **Consent** rule is shown above "Add a voice"; the library lists what `data/voices` holds (empty on a fresh machine). |
| 3 | **Record a voice**: press **Record**, speak 8-10 s of dry speech, press **Stop**, listen back, press **Use this take**. Name it `woman`, kind *woman*, consent note "my own voice", **Save to the library**. | While recording, the level meter moves with your voice and the time counts up to 60 s; "too loud" appears only on clipping. After saving: "Saved the voice "woman" (x s)", the voice is listed with kind, seconds and the consent note, and the hint about missing preset voices no longer names `woman`. The Convert tab's **Woman** and **Ghost** preset buttons lose their warning dot. |
| 4 | Record again but stop after 2 s. Then record and let it run. | "Too short: 2.0 s. Record at least 3 s." (nothing kept). The long take stops by itself at 60.0 s. |
| 5 | **Upload a voice**: **Upload a file → Choose a file…**, pick a 5-20 s clip, name it `demon`, **Save**. Then try the name `woman` again, and a 70 s file. | `demon` is listed. The taken name shows "A voice named "woman" exists" with Save disabled; the 70 s file is refused with the backend's "at most 60 s" message. |
| 6 | **Play**, **Rename** (type, Enter) and **Delete** (confirm) a spare voice. | It plays; the new name is listed; after the confirm it is gone. Rename to an invalid name (e.g. `.x`) shows the name rule. |
| 7 | **Convert a clip**: select a clip on the timeline, open **Convert**, choose **Selected clip**, press the **Woman** preset, then **Convert**. | The source line names the clip's file. The preset fills Voice, Match my pitch contour, Pitch shift and Quality steps, and the FX summary lists its rack (Reverb). Convert shows a progress bar with the job's messages (the Voices tab's worker line turns to *running on cuda*). Then a **Converted** card: it plays, the file is as long as the source, and the **Media** panel shows the new item without a manual refresh. |
| 8 | In the **Media** panel right-click an audio or video tile → **Convert voice…**. Choose **Custom**, a library voice, Pitch shift +3, **Convert**. | The Voice panel opens on **Convert** with that item as the source (Media pool chip). The conversion runs as in step 7. Right-clicking an image tile shows no menu. |
| 9 | **Convert a fresh recording**: **Microphone → Record**, say a line (5 s), **Stop**, **Upload and use as the source**, then **Convert**. | The source line reads "Recording <time>, in the media pool" and the recording appears in the Media panel; the conversion runs as in step 7. On a backend without the upload route the note says "upload route not available yet" and the take stays in the recorder. |
| 10 | In a **Converted** card press **Download WAV**, then **Download MP3**. | A save dialog for `<item>_<YYYYMMDD-HHMMSS>.wav`, then `.mp3`; both files play in a desktop player. |
| 11 | **Batch two files**: **Batch** tab, tick two items, press **Demon**, **Run on 2 files**. | Rows go *queued → converting · n% → done* one after another, with the overall bar below; then "2 converted: the new items are in the media pool", and each row gets **Play / WAV / MP3**. The Media panel shows two new items. |
| 12 | **Add a region**: Shift+drag a range on A1 inside one clip, right-click the range band → **Convert voice…**. | The Voice panel opens on **Regions** with "Added m:ss.cc–m:ss.cc in <voice or preset>…" and a *pending* row whose start/end match the band. |
| 13 | **Add a region across a cut**: Shift+drag a range that spans the join of two A1 clips, then **Regions → Add from selection**. Then add a range overlapping the first region. | A second pending row, sorted by start. The overlapping one is trimmed to start where the first ends (the note says so), or "Nothing added" when it lies inside it. |
| 14 | Edit row 2: preset **Monster**, Match on, Pitch −7. Close and reopen the panel (or reopen the project). | Monster fills the voice (`demon`), steps 30 and the FX count; the edits are still there after reopening (they are saved half a second after each change). |
| 15 | **Apply pending (2)**. | A progress bar, then "2 regions converted into one new version (<n>_voice_conversion.wav)". Both rows are *applied* (green) and locked. Play the programme: both ranges speak in the new voices, everything outside them is untouched, the seams make no click, and the cut, the B-roll and the length are unchanged. In the **Audio** panel (reopen it if it was open) the Programme (A1) rack holds each region's FX windowed to it. |
| 16 | **Undo last apply**. | "Undid the last apply…"; both rows are *pending* again, the original audio plays, and the regions' FX are gone from the A1 rack. Undo is disabled once nothing is applied. |
| 17 | Add a region whose voice is not in the library (e.g. a preset whose voice is missing) beside a valid one, and Apply. | The valid one is applied; the other is *failed* (red) with the error under the table and kept the original audio. Editing the failed row turns it *pending*. |
| 18 | **Voices → Stop** while the worker runs. | "Stopped: the model is unloaded and the GPU is free."; the worker line returns to idle and the VRAM drops back (nvidia-smi). |
| 19 | Start a render, then try Convert or Apply. | A yellow "Not now: A render is running…" note; nothing crashes. |

**Known gaps**

- The Audio panel reads its racks when it opens. If it is open while regions are applied or
  undone, close and reopen it before editing the A1 rack there; otherwise its next save writes
  back the rack it loaded, without the voice-region FX.
- A converted file carries no FX (they are live rack effects): put it on an A2 (or lower) track
  and add them in the Audio panel's Clip rack, as the FX summary says. Regions get theirs on the
  A1 rack automatically.
- SkinWalker's "formant" FX has no rack effect yet; the FX summary lists it as not in the rack.
