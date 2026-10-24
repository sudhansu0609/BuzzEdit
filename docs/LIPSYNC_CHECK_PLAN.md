# Lip-sync check for voice-converted spans

Design
- `backend/audio/lipsync_check.py` compares mouth motion with the audio loudness envelope, span by span.
- Mouth: the eye-contact landmark net (`eyecontact.tracker.Tracker`) over only the span's frames (+0.5 s context), via a seek-capable NVDEC decoder; openness = |p13-p14| / |p78-p308| per frame (NaN = no face).
- Audio: mono PCM (`audio.analysis.read_mono`), 300-3000 Hz band-pass, rectified, 20 Hz low-pass, one value per video frame, log-compressed, scaled 0..1 (revised after the real check).
- Score: best Pearson correlation of the smoothed, mean-removed series over +-12 frames -> {corr, lag_frames}.
- Flag: corr_converted < max(0.15, 0.6 * corr_original) ("weak_match") or |lag_conv - lag_orig| > 3 ("lag"; real footage has the sound trailing the mouth by a steady 3-4 frames, so lag is judged relative to the original); face in < 60 % of frames -> "no_face", not flagged; spans < 1 s skipped (corr NaN).
- Wiring: `apply_voice_map` runs it best-effort on the converted spans (`lipsync` per conversion entry + `lipsync_flags`); `POST /api/audio/{project}/voice/lipsync_check` re-checks as a job.

| # | Deliverable | Status |
|---|---|---|
| 1 | `audio/lipsync_check.py` (mouth_openness, audio_envelope, sync_score, check_spans) | done: revised metric approved (band-pass + log envelope, max_lag 12, relative lag, corr floor 0.15); + `tracker.track_window` (SimpleDecoder seek path) |
| 2 | Wiring: apply_voice_map `lipsync` + `lipsync_flags`; POST voice/lipsync_check route | done: `voice_conversion._attach_lipsync` (best effort; skipped without video / landmarks) puts `lipsync` {worst span + spans} on each entry, metadata + `lipsync_flags`; route `POST /voice/lipsync_check` -> process job, 503 via `eyecontact.available()`, 409 from the router's render guard |
| 3 | Tests `tests/test_lipsync_check.py` (max 3 small tests, per scope change) | done: 3 tests (sync_score x3 parametrised, check_spans rules with mocks, route 503) = 5 passed; test_voice_conversion + test_eye_contact 47 passed |
| 4 | Real check on 674aefab_source.mp4 (GPU) | done (final rule). Original audio vs video: corr 0.331 (1391 s) / 0.288 (2170 s), absolute lag +4 / +3 frames (sound trails mouth; relative lag 0, not flagged). Shift 200 ms: flagged 'lag' on both (rel. lag +5 / -14; 2170 s found a spurious peak, corr 0.394). Replaced speech: flagged 'weak_match' on both (corr -0.006 / -0.079). 1.7-1.9 s per 8 s span warm (+ ~18 s one-off net load). Decode path: PyNvVideoCodec SimpleDecoder seek_to_index. First pass (spec metric) gave 0.235 / 0.275 -> revised |
| 5 | This plan doc | done |

## Frontend Status

| # | Added | Request |
|---|---|---|
| 1 | Voice panel > Convert tab: read-only "Lip-sync" card (only when the applied voice map has `voice_conversions`): one row per speaker (speaker, voice, seconds, chip ok / check: weak match / check: lag N frames, correlations in the tooltip), "N spans flagged" line, "Re-check lip-sync" button with JobProgress and the per-span result list (start-end, corr, lag, reason). Data read from `GET /api/projects/{id}` `settings.voice_map_applied.voice_conversions` (the project route does not return `metadata`); flags counted from the entries' `lipsync.spans` | `POST /api/audio/{project}/voice/lipsync_check` `{}` -> `{job_id}`, polled on `GET /api/audio/{project}/process/{job_id}` |
