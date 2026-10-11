# Audio panel — manual smoke checklist (AUDIO_SUITE_PLAN.md brief A4, parts a and b)

There is no browser test harness, so the panel is checked by hand. Run it on a project that has
a transcript (Detect noise looks between the words) and a few seconds of room tone somewhere.

**Before you start**

- The backend must have the audio routes: `routes/audio.py`, `routes/audio_process.py` and the
  live socket `routes/audio_live.py` are registered in `backend/main.py`. A backend started
  before they landed answers every `/api/audio` call with a bare 404; the panel then shows
  yellow "not available on this backend yet" notes instead of failing. Restart BuzzEdit once.
- Build the frontend (`npx vite build`) or run `npm run dev`.
- The timeline's audio range (steps 11-15) is in programme seconds, what the ruler shows. Until
  the backend brief that plays and maps programme time lands, the Programme / Master live
  preview, Analyze and the process effects read those numbers as recording time, so on a cut
  project they reach a different stretch of the recording than the band covers.

| # | Do this | Expect |
|---|---|---|
| 1 | Select an A1 (or V1) clip, open **Inspector → Audio (this clip)** and press **Open Audio panel** (or Layout menu → Audio). | A floating **Audio** window: scope chips Programme (A1) / Clip / Master, a transport, meters, and three columns (Effects tree, Effects Rack, parameters). The status chip turns from *Connecting…* to **Live**. No yellow notes. |
| 2 | In the tree open **Filter and EQ** and click **Parametric Equalizer**. | It appears as rack entry 1 with a green power button and its parameters on the right (High-Pass, Low Shelf, Band 1–5, High Shelf, Low-Pass, each under its own switch; Master Gain). About half a second later the rack is saved (re-open the panel: it is still there). |
| 3 | Press **Play** (or click inside the panel and press **Space**). | Sound plays from the start of the selected clip; the position readout advances about four times a second. Space again pauses. |
| 4 | While it plays, drag **Band 3 → Gain** from 0 to +12 dB and back; then type 2000 into Band 3's Frequency box. | The tone change is heard while the slider moves (no restart, no gap). Changing a switch (e.g. High-Pass on) restarts the stream at the playhead with only a tick. |
| 5 | Press **Loop** (circular arrows). | Playback jumps to the range start and repeats the range shown next to the time (the clip, or the selection once one exists). Press Loop again to stop looping. |
| 6 | Press **A/B** (the button shows *A*) while playing. | It turns to **B** and the whole rack is bypassed (the dry sound); press again for **A** (processed). The power button on one effect bypasses only that effect. |
| 7 | With the Parametric EQ selected in the rack, use its preset row: pick **Vocal Presence**, then **Save…**, name it `My EQ test`, Save. | Picking a preset changes the knobs and is heard; the saved preset shows under *Your presets*, its name tags the rack entry, and the ✕ next to the menu deletes it. |
| 8 | In the tree open **Noise Reduction / Restoration** and click **Noise Reduction (process)**. In its Noise print card press **Detect noise**, check the From/To fields, press **Capture Noise Print**. Then **Preview**, then **Apply**, then **Undo**. | Detect fills a ≥ 0.8 s range between words; Capture draws the print curve with a dashed floor line. Preview renders up to 20 s and plays it in the card (switch on *Output Noise Only* to hear just the noise). Apply shows a progress bar, then *Versions: 1* and the live preview reconnects on the cleaned audio. Undo returns to *Versions: 0* and the original sound. |
| 9 | Open the **Analyze** section and press **Analyze selection**. | Integrated LUFS, Loudness range, True peak, Peak, RMS and Noise floor for the range, plus the spectrum picture. |
| 10 | Play again and watch the meters for 30 s. | Peak and RMS bars move with the sound; Momentary and Short-term LUFS update; the ring fills one wedge every half second (green near −14 LUFS, red above, blue below). Repeat steps 3–4 on the **Master** chip and on an A2 clip (**Clip** chip) to check the other scopes. |

**Part b: the timeline range, Essential Sound, the Inspector summary**

| # | Do this | Expect |
|---|---|---|
| 11 | On the timeline hold **Shift** and drag along the ruler; then Shift+drag from an empty part of an audio lane (a gap between clips, past the last clip, or the thin strip above or below the clips). | A translucent green band across the audio lanes with start, length and end labels, plus a green strip on the ruler. In the Audio panel the row under the transport reads *Selection m:ss.ss–m:ss.ss · n.nn s* and the transport's range says *Selection*. The ends snap to 10 ms. A plain drag on the ruler still scrubs, a plain drag on empty lane space still box-selects clips, and dragging a clip or its edges still moves or trims it. |
| 12 | Move the playhead and press **I**, move it on and press **O** (or use the toolbar's In / Out). | The band runs from the In point to the Out point. I / O are still Mark In / Mark Out, so the mark band and *Cut range* follow them as before. Pressing I after the band's end (or O before its start) removes the band until the other key is pressed again. |
| 13 | Switch on **⇹ Range** in the timeline toolbar, plain-drag on the ruler and on an empty part of an audio lane, then switch it off. | With Range on, a plain drag selects the range: the ruler does not scrub and audio lanes do not box-select (video and text lanes still do). With Range off, drags scrub and box-select again. Hover the green ruler strip: its tooltip names Shift+drag, I / O and Range, and how to clear the range. |
| 14 | Press **Esc** with the timeline focused. Make a range again, then click without dragging while holding Shift (or with Range on). Make one more and press **Clear** in the Audio panel. | Each one removes the band, and the panel row reads *No selection…*. A plain click on the ruler (Range off, no Shift) only moves the playhead and leaves the range alone. |
| 15 | Right-click the green ruler strip, then right-click inside the band on an audio lane (on a clip and on empty space). | The menu starts with **Audio effects on selection…** and **Clear Audio Range**; on a lane they sit above that lane's or clip's usual entries. The first opens (or brings up) the Audio panel with the range in its transport, and Loop plays it. |
| 16 | In the Audio panel click the **Essential Sound** tab at the top of the editor column. | The Dialogue card: Repair (Reduce Noise, Reduce Rumble, DeHum, DeEss, Reduce Reverb), Clarity (Dynamics, EQ, Enhance Speech), Creative (Reverb, shown only when the catalogue has Studio Reverb) and Clip Volume. Clicking a rack entry, a process or adding an effect switches back to the **Effect** tab. |
| 17 | While playing, move **Reduce Noise → Amount**, then **Reduce Rumble → Frequency**, pick an **EQ** preset and set **Enhance Speech → Voice**. | The first move of each control adds one rack entry tagged *essential:noise*, *essential:rumble*, *essential:eq*, *essential:enhance*, and the change is heard. Moving the control again changes that same entry: the rack never gets a second copy. Unticking a control bypasses its entry, and ticking it turns it back on. Switching the Reduce Noise engine replaces its entry. Re-open the panel and the entries are still there. Editing a tagged entry in the rack keeps its tag. |
| 18 | Under **Reduce Reverb** press **Open…**, then **Preview**, **Apply** and **Undo** with a range selected. | The DeepFilterNet process card (Noise Reduction (AI)) with the range as its range. Apply writes a new version and the live preview reconnects; Undo restores the previous version. It is never heard live. |
| 19 | Select an A1 clip, open **Inspector → Audio (this clip)**, then select an A2 clip. | Next to *Open Audio panel* is a one-line summary of the rack the panel opens on. That is A1's rack for an A1 or V1 clip and the clip's own on A2+, for example *3 effects: Parametric Equalizer, DeEsser, Hard Limiter*. Bypassed effects are marked *(off)*, and an empty rack reads *No effects*. |

If a step fails, note the yellow message (it carries the backend's own error text) and the
step number in the A4 row of `AUDIO_SUITE_PLAN.md` §4.
