/**
 * Batch (VOICE_CHANGER_PLAN.md §2.7, SkinWalker's Batch tab): several media-pool items into one
 * voice + preset in one run. The `batch` job converts them one after another into new media-pool
 * items; its message reads "k/n <media_id>: ..." while item k runs, which gives each row its own
 * progress, and its result says per item what came out (or why not). Every output plays here and
 * downloads as WAV or MP3.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MediaEntry, listMedia } from '../../hooks/api';
import { EffectDef, audioFileUrl, errorText } from '../../hooks/audioApi';
import {
  BatchItemResult, BatchResult, PresetsResponse, VoiceChoice, VoiceEntry, batchConvert, choiceLabel,
  defaultChoice, downloadError, downloadMedia, emptyChoice, fileName, fileStem, mediaPoolChanged,
  notifyMediaPoolChanged, rememberChoice, startError, timeStamp,
} from '../../hooks/voiceApi';
import JobProgress, { JobWatch, jobRunning, useJob, watchJob } from './JobProgress';
import { VoiceSettings } from './PresetPicker';

const BATCH_MESSAGE = /^(\d+)\/(\d+)\s+(\S+)(?::\s*(.*))?$/;

function batchPosition(message?: string): { index: number; total: number; detail: string } | null {
  const m = BATCH_MESSAGE.exec(message || '');
  return m ? { index: Number(m[1]) - 1, total: Number(m[2]), detail: m[4] || '' } : null;
}

export default function BatchTab({ projectId, voices, presets, defs }: {
  projectId: string;
  voices: VoiceEntry[];
  presets: PresetsResponse | null;
  defs: Record<string, EffectDef>;
}) {
  const [media, setMedia] = useState<MediaEntry[]>([]);
  const [mediaNote, setMediaNote] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [choice, setChoice] = useState<VoiceChoice | null>(null);
  const [watch, setWatch] = useState<JobWatch | null>(null);
  const job = useJob(watch);
  const [runIds, setRunIds] = useState<string[]>([]);
  const [results, setResults] = useState<Record<string, BatchItemResult> | null>(null);
  const [stamp, setStamp] = useState('');
  const [playing, setPlaying] = useState<{ id: string; url: string } | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

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
    if (choice) return;
    const pick = defaultChoice(presets, voices);
    if (pick) setChoice(pick);
  }, [choice, presets, voices]);

  const running = jobRunning(job);
  const names = useMemo(() => {
    const out: Record<string, string> = {};
    for (const m of media) out[m.id] = m.name;
    return out;
  }, [media]);

  const toggle = (id: string, on: boolean) => {
    setSelected((current) => {
      const next = new Set(current);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });
  };

  const run = async () => {
    const ids = media.filter((m) => selected.has(m.id)).map((m) => m.id);
    if (!ids.length) return;
    const c = choice ?? emptyChoice(presets);
    setNote(null);
    setResults(null);
    setPlaying(null);
    setRunIds(ids);
    try {
      const { job_id: jobId } = await batchConvert(projectId, {
        media_ids: ids, voice: c.voice || null, preset: c.preset, params: c.params,
      });
      rememberChoice(c);
      const w = watchJob(projectId, jobId);
      setWatch(w);
      void w.done.then((final) => {
        if (final.status === 'completed') notifyMediaPoolChanged(projectId);
        if (!alive.current) return;
        const items = (final.result as unknown as BatchResult | null)?.items;
        if (final.status === 'completed' && Array.isArray(items)) {
          const byId: Record<string, BatchItemResult> = {};
          for (const item of items) if (item && item.media_id) byId[item.media_id] = item;
          setResults(byId);
          setStamp(timeStamp());
        }
      });
    } catch (err) {
      setRunIds([]);
      setNote(startError(err, 'Batch'));
    }
  };

  /** A row's state in the run: queued, converting (with its share of the progress), done, failed. */
  const rowStatus = (id: string): { text: string; tone: string } | null => {
    const k = runIds.indexOf(id);
    if (k < 0) return null;
    const result = results?.[id];
    if (result) {
      return result.error
        ? { text: `failed: ${result.error}`, tone: 'failed' }
        : { text: `done: ${fileName(result.path || '')}`, tone: 'applied' };
    }
    if (!job) return { text: 'queued', tone: 'pending' };
    if (job.status === 'failed') return { text: 'not converted', tone: 'failed' };
    if (job.status === 'completed') return { text: 'no result', tone: 'failed' };
    const at = batchPosition(job.message);
    if (!at) return { text: 'queued', tone: 'pending' };
    if (k < at.index) return { text: 'done', tone: 'applied' };
    if (k > at.index) return { text: 'queued', tone: 'pending' };
    const n = Math.max(1, at.total);
    const share = Math.round(Math.min(1, Math.max(0, job.progress * n - k)) * 100);
    return { text: `converting${at.detail ? `: ${at.detail}` : ''} · ${share}%`, tone: 'running' };
  };

  const download = async (result: BatchItemResult, format: 'wav' | 'mp3') => {
    if (!result.path) return;
    setDownloading(`${result.media_id}:${format}`);
    setNote(null);
    try {
      await downloadMedia(result.path, format, `${fileStem(result.path)}_${stamp || timeStamp()}`);
    } catch (err) {
      if (alive.current) setNote(downloadError(err));
    } finally {
      if (alive.current) setDownloading(null);
    }
  };

  const count = media.filter((m) => selected.has(m.id)).length;
  const okCount = results ? Object.values(results).filter((r) => !r.error).length : 0;
  const failCount = results ? Object.values(results).filter((r) => r.error).length : 0;

  return (
    <div className="voice-tab">
      <div className="audio-card">
        <div className="audio-card-title">
          Files <span className="audio-badge">{count} selected</span>
          <span className="voice-spacer" />
          <button type="button" className="audio-btn" disabled={running || media.length === 0}
            onClick={() => setSelected(new Set(media.map((m) => m.id)))}>
            All
          </button>
          <button type="button" className="audio-btn" disabled={running || count === 0}
            onClick={() => setSelected(new Set())}>
            None
          </button>
        </div>
        {media.length === 0 ? (
          <div className="audio-hint">No audio or video in the media pool yet.</div>
        ) : (
          <div className="voice-batch-list">
            {media.map((m) => {
              const status = rowStatus(m.id);
              const result = results?.[m.id];
              return (
                <div key={m.id} className="voice-batch-item">
                  <label className="voice-batch-name" title={m.path}>
                    <input type="checkbox" disabled={running} checked={selected.has(m.id)}
                      onChange={(e) => toggle(m.id, e.target.checked)} />
                    <span className="audio-badge">{m.kind}</span>
                    <span className="voice-ellipsis">{m.name}</span>
                    {m.duration > 0 && <span className="audio-unit">{m.duration.toFixed(1)} s</span>}
                  </label>
                  {status && (
                    <span className={`voice-status ${status.tone}`} title={status.text}>{status.text}</span>
                  )}
                  {result && !result.error && result.path && (
                    <span className="voice-row">
                      <button type="button" className="audio-btn"
                        onClick={() => setPlaying(playing?.id === m.id ? null : { id: m.id, url: audioFileUrl(result.path || '') })}>
                        {playing?.id === m.id ? 'Close' : 'Play'}
                      </button>
                      <button type="button" className="audio-btn" disabled={!!downloading}
                        onClick={() => void download(result, 'wav')}>
                        {downloading === `${m.id}:wav` ? '…' : 'WAV'}
                      </button>
                      <button type="button" className="audio-btn" disabled={!!downloading}
                        onClick={() => void download(result, 'mp3')}>
                        {downloading === `${m.id}:mp3` ? '…' : 'MP3'}
                      </button>
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        )}
        {playing && (
          <>
            <div className="text-xs text-muted">{names[playing.id] || playing.id}, converted</div>
            <audio className="audio-preview" controls autoPlay src={playing.url} />
          </>
        )}
        {mediaNote && <div className="audio-note">{mediaNote}</div>}
      </div>

      <div className="audio-card">
        <div className="audio-card-title">Voice for every file</div>
        <VoiceSettings choice={choice} onChange={setChoice} voices={voices} presets={presets} defs={defs}
          disabled={running}
          fxHint="The converted files carry no FX; add them as a clip rack in the Audio panel where a file is used." />
      </div>

      <div className="voice-row">
        <button type="button" className="audio-btn primary" disabled={count === 0 || running} onClick={() => void run()}>
          {running ? 'Converting…' : `Run on ${count} file${count === 1 ? '' : 's'}`}
        </button>
        <span className="audio-hint">
          {count ? `${count} file${count === 1 ? '' : 's'} into ${choiceLabel(choice ?? emptyChoice(presets))}, one after another; each becomes a new media-pool item.` : 'Tick the files to convert.'}
        </span>
      </div>
      <JobProgress watch={watch} label="Batch" />
      {results && (
        <div className="text-xs">
          {okCount} converted{failCount ? `, ${failCount} failed` : ''}: the new items are in the media pool.
        </div>
      )}
      {note && <div className="audio-note">{note}</div>}
    </div>
  );
}
