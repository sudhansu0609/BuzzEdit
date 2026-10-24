/**
 * Convert (VOICE_CHANGER_PLAN.md §2.7): one whole file into a voice -- the selected clip's file,
 * a media-pool item, or a fresh microphone recording (uploaded into the media pool first) --
 * with a library voice or a preset, Match my pitch contour, Pitch shift and Quality steps. The
 * `convert_file` job writes a new 48 kHz WAV and registers it in the media pool (the pool is told
 * to refresh); the result plays here and downloads as WAV or MP3 under a timestamped name.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useProjectStore } from '../../hooks/store';
import { MediaEntry, getProject, listMedia } from '../../hooks/api';
import { AudioApiError, EffectDef, audioFileUrl, errorText } from '../../hooks/audioApi';
import {
  ConvertFileResult, LipsyncCheckResult, LipsyncSpan, LipsyncSummary, PresetsResponse, UploadedMedia,
  VoiceChoice, VoiceConversion, VoiceEntry, choiceLabel, convertFile, defaultChoice, downloadError,
  downloadMedia, emptyChoice, fileName, fileStem, formatSeconds, mediaPoolChanged, notifyMediaPoolChanged,
  rememberChoice, samePath, startError, startLipsyncCheck, timeStamp, uploadMedia, uploadMissing,
} from '../../hooks/voiceApi';
import JobProgress, { JobWatch, jobRunning, useJob, watchJob } from './JobProgress';
import Recorder, { RecordedClip } from './Recorder';
import { VoiceSettings } from './PresetPicker';

type SourceMode = 'clip' | 'media' | 'recording';

interface SourcePick {
  mediaId: string | null;
  path: string | null;
  label: string;
}

interface ConvertDone {
  out: ConvertFileResult;
  url: string;
  source: string;
  who: string;
  stamp: string;
}

const FX_HINT = "A converted file carries no FX: they are live rack effects. Put the file on an A2 (or lower) track and add them in the Audio panel's Clip rack; timeline regions get them automatically.";

export default function ConvertTab({ projectId, voices, presets, defs, request }: {
  projectId: string;
  voices: VoiceEntry[];
  presets: PresetsResponse | null;
  defs: Record<string, EffectDef>;
  /** the Media pool's "Convert voice…": this item becomes the source */
  request: { mediaId: string; seq: number } | null;
}) {
  const timeline = useProjectStore((s) => s.project?.timeline);
  const selectedClipId = useProjectStore((s) => s.selectedClipId);

  const [media, setMedia] = useState<MediaEntry[]>([]);
  const [mediaNote, setMediaNote] = useState<string | null>(null);
  const [mode, setMode] = useState<SourceMode>(() => (useProjectStore.getState().selectedClipId ? 'clip' : 'media'));
  const [mediaId, setMediaId] = useState('');
  const [recorded, setRecorded] = useState<UploadedMedia | null>(null);
  const [uploading, setUploading] = useState(false);
  const [choice, setChoice] = useState<VoiceChoice | null>(null);
  const [watch, setWatch] = useState<JobWatch | null>(null);
  const job = useJob(watch);
  const [result, setResult] = useState<ConvertDone | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  // -- the media pool's audio and video ---------------------------------------------------------
  const loadMedia = useCallback(async () => {
    try {
      const res = await listMedia(projectId);
      if (!alive.current) return;
      setMedia((res.media || []).filter((m) => (m.kind === 'audio' || m.kind === 'video') && !m.missing));
      setMediaNote(null);
    } catch (err) {
      if (alive.current) setMediaNote(`Could not load the media pool: ${errorText(err)}`);
    }
  }, [projectId]);

  useEffect(() => { void loadMedia(); }, [loadMedia]);
  useEffect(() => mediaPoolChanged.subscribe((id) => {
    if (id === projectId) void loadMedia();
  }), [projectId, loadMedia]);

  useEffect(() => {
    if (!mediaId && media.length) setMediaId(media[0].id);
  }, [media, mediaId]);

  useEffect(() => {
    if (!request) return;
    setMode('media');
    setMediaId(request.mediaId);
    setResult(null);
    void loadMedia();
  }, [request, loadMedia]);

  // -- the default voice, once the library and the presets are known ------------------------------
  useEffect(() => {
    if (choice) return;
    const pick = defaultChoice(presets, voices);
    if (pick) setChoice(pick);
  }, [choice, presets, voices]);

  // -- the source -------------------------------------------------------------------------------
  const clipSource = useMemo((): SourcePick | null => {
    const tl: any = timeline;
    const item = (tl?.items || []).find((i: any) => i?.id === selectedClipId);
    if (!item || (item.kind && item.kind !== 'media') || !item.source_id) return null;
    const src = tl?.sources?.[item.source_id];
    if (!src?.path || src.kind === 'image' || src.has_audio === false) return null;
    const path = String(src.path);
    const entry = media.find((m) => samePath(m.path, path) || samePath(m.source_path, path));
    const track = String(item.track || '').toUpperCase();
    return {
      mediaId: entry?.id ?? null,
      path: entry ? null : path,
      label: `${entry?.name || fileName(path)}${track ? ` (the ${track} clip's whole file)` : ''}`,
    };
  }, [timeline, selectedClipId, media]);

  const source: SourcePick | null = useMemo(() => {
    if (mode === 'clip') return clipSource;
    if (mode === 'media') {
      const entry = media.find((m) => m.id === mediaId);
      return entry ? { mediaId: entry.id, path: null, label: entry.name } : null;
    }
    return recorded
      ? { mediaId: recorded.media_id, path: null, label: `${recorded.name} (recording, ${Number(recorded.seconds || 0).toFixed(1)} s)` }
      : null;
  }, [mode, clipSource, media, mediaId, recorded]);

  // -- a fresh recording: upload -> media item ----------------------------------------------------
  const uploadRecording = async (clip: RecordedClip): Promise<boolean> => {
    setUploading(true);
    setNote(null);
    try {
      const uploaded = await uploadMedia(projectId, clip.blob, { filename: clip.filename, name: `Recording ${timeStamp()}` });
      notifyMediaPoolChanged(projectId);
      if (alive.current) {
        setRecorded(uploaded);
        setResult(null);
      }
      return true;
    } catch (err) {
      if (alive.current) {
        if (uploadMissing(err)) {
          const detail = err instanceof AudioApiError && err.detail && err.detail !== 'Not Found' ? ` (${err.detail})` : '';
          setNote(`The recording cannot reach the media pool: upload route not available yet${detail}. Restart BuzzEdit to load it.`);
        } else {
          setNote(`Upload failed: ${errorText(err)}`);
        }
      }
      return false;
    } finally {
      if (alive.current) setUploading(false);
    }
  };

  // -- convert ----------------------------------------------------------------------------------
  const running = jobRunning(job);

  const convert = async () => {
    if (!source) return;
    const c = choice ?? emptyChoice(presets);
    const sourceLabel = source.label;
    const who = choiceLabel(c);
    setNote(null);
    setResult(null);
    try {
      const { job_id: jobId } = await convertFile(projectId, {
        media_id: source.mediaId,
        path: source.mediaId ? null : source.path,
        voice: c.voice || null,
        preset: c.preset,
        params: c.params,
      });
      rememberChoice(c);
      const w = watchJob(projectId, jobId);
      setWatch(w);
      void w.done.then((final) => {
        if (final.status !== 'completed' || !final.result) return;
        notifyMediaPoolChanged(projectId);
        const out = final.result as unknown as ConvertFileResult;
        if (alive.current && out?.path) {
          setResult({ out, url: audioFileUrl(out.path), source: sourceLabel, who, stamp: timeStamp() });
        }
      });
    } catch (err) {
      setNote(startError(err, 'Convert'));
    }
  };

  const download = async (format: 'wav' | 'mp3') => {
    if (!result) return;
    setDownloading(format);
    setNote(null);
    try {
      await downloadMedia(result.out.path, format, `${fileStem(result.out.path)}_${result.stamp}`);
    } catch (err) {
      if (alive.current) setNote(downloadError(err));
    } finally {
      if (alive.current) setDownloading(null);
    }
  };

  return (
    <div className="voice-tab">
      <div className="audio-card">
        <div className="audio-card-title">Source</div>
        <div className="fx-scope voice-modes">
          <button type="button" className={`insp-chip${mode === 'clip' ? ' active' : ''}`}
            title="The clip selected on the timeline: its whole file" onClick={() => setMode('clip')}>
            Selected clip
          </button>
          <button type="button" className={`insp-chip${mode === 'media' ? ' active' : ''}`}
            title="An audio or video item of the media pool" onClick={() => setMode('media')}>
            Media pool
          </button>
          <button type="button" className={`insp-chip${mode === 'recording' ? ' active' : ''}`}
            title="Record a line from the microphone" onClick={() => setMode('recording')}>
            Microphone
          </button>
        </div>
        {mode === 'clip' && (clipSource
          ? <div className="voice-source">{clipSource.label}</div>
          : <div className="audio-hint">Select an audio or video clip on the timeline: its whole file is converted.</div>)}
        {mode === 'media' && (media.length ? (
          <select className="audio-select" value={mediaId} aria-label="Media item"
            onChange={(e) => { setMediaId(e.target.value); setResult(null); }}>
            {!media.some((m) => m.id === mediaId) && <option value={mediaId}>(pick an item)</option>}
            {media.map((m) => (
              <option key={m.id} value={m.id}>
                {m.kind === 'audio' ? 'Audio' : 'Video'} · {m.name}{m.duration ? ` · ${m.duration.toFixed(1)} s` : ''}
              </option>
            ))}
          </select>
        ) : (
          <div className="audio-hint">No audio or video in the media pool yet: import some, or record one.</div>
        ))}
        {mode === 'recording' && (
          <>
            <Recorder busy={uploading} useLabel="Upload and use as the source" onUse={uploadRecording}
              hint="Record the line to re-voice (3-60 s). It goes into the media pool, then converts like any item." />
            {recorded && (
              <div className="voice-source">
                Source: {recorded.name} · {Number(recorded.seconds || 0).toFixed(1)} s, in the media pool
              </div>
            )}
          </>
        )}
        {mediaNote && <div className="audio-note">{mediaNote}</div>}
      </div>

      <div className="audio-card">
        <div className="audio-card-title">Voice</div>
        <VoiceSettings choice={choice} onChange={setChoice} voices={voices} presets={presets} defs={defs}
          disabled={running} fxHint={FX_HINT} />
      </div>

      <div className="voice-row">
        <button type="button" className="audio-btn primary" disabled={!source || running || uploading}
          title="Convert the whole file; its length stays the same" onClick={() => void convert()}>
          {running ? 'Converting…' : 'Convert'}
        </button>
        <span className="audio-hint">
          {source ? `${source.label} into ${choiceLabel(choice ?? emptyChoice(presets))}` : 'Pick a source first.'}
        </span>
      </div>
      <JobProgress watch={watch} label="Convert" />

      {result && (
        <div className="audio-card voice-result">
          <div className="audio-card-title">
            Converted <span className="audio-badge">{Number(result.out.seconds || 0).toFixed(1)} s</span>
          </div>
          <div className="text-xs text-muted">
            {result.source} in {result.who}: {fileName(result.out.path)}, added to the media pool.
          </div>
          <audio className="audio-preview" controls src={result.url} />
          <div className="voice-row">
            <button type="button" className="audio-btn" disabled={!!downloading} onClick={() => void download('wav')}>
              {downloading === 'wav' ? 'Preparing…' : 'Download WAV'}
            </button>
            <button type="button" className="audio-btn" disabled={!!downloading} onClick={() => void download('mp3')}>
              {downloading === 'mp3' ? 'Preparing…' : 'Download MP3'}
            </button>
          </div>
        </div>
      )}
      {note && <div className="audio-note">{note}</div>}
      <LipsyncCard projectId={projectId} />
    </div>
  );
}

// -- lip-sync (read-only): the last voice-map apply's per-speaker check, and a re-check ----------

const corrText = (n: number | null | undefined) =>
  typeof n === 'number' && Number.isFinite(n) ? n.toFixed(2) : '–';

function lipsyncTip(s: { corr_original: number | null; corr_converted: number | null; lag_frames: number | null }) {
  return `Mouth-voice correlation: original ${corrText(s.corr_original)}, converted ${corrText(s.corr_converted)}`
    + `; lag ${s.lag_frames ?? '–'} frames`;
}

/** The chip for one speaker's worst span: green ok, amber check, muted when not checkable. */
function LipsyncChip({ summary }: { summary: LipsyncSummary | null | undefined }) {
  if (!summary) {
    return (
      <span className="voice-lipsync-chip muted" title="No lip-sync check for this speaker (no video or no GPU landmarks)">
        not checked
      </span>
    );
  }
  if (summary.flag) {
    const text = summary.reason === 'lag' ? `check: lag ${summary.lag_frames ?? '?'} frames` : 'check: weak match';
    return <span className="voice-lipsync-chip warn" title={lipsyncTip(summary)}>{text}</span>;
  }
  if (summary.reason === 'no_face' || summary.reason === 'short') {
    return (
      <span className="voice-lipsync-chip muted" title={lipsyncTip(summary)}>
        {summary.reason === 'no_face' ? 'no face' : 'too short'}
      </span>
    );
  }
  return <span className="voice-lipsync-chip ok" title={lipsyncTip(summary)}>ok</span>;
}

function LipsyncCard({ projectId }: { projectId: string }) {
  const [conversions, setConversions] = useState<VoiceConversion[]>([]);
  const [watch, setWatch] = useState<JobWatch | null>(null);
  const job = useJob(watch);
  const [check, setCheck] = useState<LipsyncCheckResult | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  // The project route does not return `metadata`; the applied voice map's record
  // (`settings.voice_map_applied`) carries the same `voice_conversions`, with `lipsync`.
  useEffect(() => {
    let live = true;
    getProject(projectId).then((p: any) => {
      const record = p?.settings?.voice_map_applied;
      const list = Array.isArray(record?.voice_conversions) ? record.voice_conversions : [];
      if (live) setConversions(list as VoiceConversion[]);
    }).catch(() => { if (live) setConversions([]); });
    return () => { live = false; };
  }, [projectId]);

  if (!conversions.length) return null;

  // = the backend's `lipsync_flags`: flagged spans across every speaker.
  const flags = conversions.reduce(
    (n, c) => n + (c.lipsync?.spans || []).filter((s) => s.flag).length, 0);
  const running = jobRunning(job);

  const recheck = async () => {
    setNote(null);
    setCheck(null);
    try {
      const { job_id: jobId } = await startLipsyncCheck(projectId);
      const w = watchJob(projectId, jobId);
      setWatch(w);
      void w.done.then((final) => {
        if (final.status === 'completed' && final.result && alive.current) {
          setCheck(final.result as unknown as LipsyncCheckResult);
        }
      });
    } catch (err) {
      setNote(startError(err, 'Lip-sync check'));
    }
  };

  return (
    <div className="audio-card voice-lipsync">
      <div className="audio-card-title">Lip-sync</div>
      {flags > 0 && (
        <div className="voice-lipsync-flags">{flags} span{flags === 1 ? '' : 's'} flagged</div>
      )}
      {conversions.map((c, i) => (
        <div key={`${c.speaker}-${c.voice}-${i}`} className="voice-lipsync-row">
          <span className="voice-lipsync-name" title={`${c.speaker} in ${c.voice}`}>
            {c.speaker} · {c.voice}
          </span>
          <span className="voice-lipsync-secs">{Number(c.seconds || 0).toFixed(1)} s</span>
          <LipsyncChip summary={c.lipsync} />
        </div>
      ))}
      <div className="voice-row">
        <button type="button" className="audio-btn" disabled={running}
          title="Check the converted spans against the speaker's mouth again (no re-conversion)"
          onClick={() => void recheck()}>
          {running ? 'Checking…' : 'Re-check lip-sync'}
        </button>
      </div>
      <JobProgress watch={watch} label="Lip-sync check" />
      {check && (
        <div className="voice-lipsync-spans">
          <div className="text-xs text-muted">
            {check.lipsync_flags > 0
              ? `${check.lipsync_flags} span${check.lipsync_flags === 1 ? '' : 's'} flagged`
              : 'No span flagged'}
          </div>
          {(check.spans || []).map((s: LipsyncSpan, i: number) => (
            <div key={i} className={`voice-lipsync-span${s.flag ? ' flagged' : ''}`} title={lipsyncTip(s)}>
              {formatSeconds(s.start_s)}-{formatSeconds(s.end_s)} · corr {corrText(s.corr_converted)}
              {' '}(was {corrText(s.corr_original)}) · lag {s.lag_frames ?? '–'}
              {s.reason ? ` · ${s.reason.replace('_', ' ')}` : ''}
            </div>
          ))}
        </div>
      )}
      {note && <div className="audio-note">{note}</div>}
    </div>
  );
}
