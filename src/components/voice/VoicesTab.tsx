/**
 * Voices (VOICE_CHANGER_PLAN.md §2.7): the reference-clip library in `data/voices`, shared by
 * every project -- list with kind, length and consent note; add one by recording from the
 * microphone or uploading a file (`POST /api/audio/voices`, multipart file + name + kind +
 * consent; 3-60 s, ffmpeg converts a browser recording); rename; delete (confirmed). It also shows
 * the seed-vc worker (running on which device, idle for how long) with Stop, which unloads the
 * model and frees the GPU. A real person's voice needs their consent: the rule is on screen.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { audioFileUrl, errorText, routeMissing } from '../../hooks/audioApi';
import {
  PresetsResponse, VoiceEntry, WorkerStatus, addVoice, deleteVoice, getVoiceWorker, renameVoice,
  stopVoiceWorker, voiceInLibrary, voiceNameProblem,
} from '../../hooks/voiceApi';
import Recorder, { RecordedClip } from './Recorder';

const KINDS = ['custom', 'woman', 'man', 'child', 'demon', 'monster', 'ghost'];
const WORKER_POLL_MS = 5000;
const MISSING = 'The voices library is not available on this backend yet (it predates the voice routes; restart BuzzEdit to load them).';

interface PendingClip {
  blob: Blob;
  filename: string;
  label: string;
}

const secs = (s: number | null | undefined) => (typeof s === 'number' && Number.isFinite(s) ? `${s.toFixed(1)} s` : '');

/** A file name as a library name: allowed characters only, starting with a letter or digit. */
function suggestName(filename: string): string {
  const dot = filename.lastIndexOf('.');
  const stem = dot > 0 ? filename.slice(0, dot) : filename;
  return stem.replace(/[^A-Za-z0-9 _.-]+/g, '_').replace(/^[^A-Za-z0-9]+/, '').replace(/[.]+$/, '').slice(0, 63);
}

export default function VoicesTab({ voices, loaded, missing, loadNote, presets, active, onChanged }: {
  voices: VoiceEntry[];
  loaded: boolean;
  /** the voices route is absent */
  missing: boolean;
  loadNote: string | null;
  presets: PresetsResponse | null;
  /** the tab is on screen: the worker's status is polled */
  active: boolean;
  /** the library changed: reload it (and the presets' voice_in_library) */
  onChanged: () => Promise<void> | void;
}) {
  const [mode, setMode] = useState<'record' | 'upload'>('record');
  const [clip, setClip] = useState<PendingClip | null>(null);
  const [name, setName] = useState('');
  const [kind, setKind] = useState('custom');
  const [consent, setConsent] = useState('');
  const [saving, setSaving] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [busyName, setBusyName] = useState<string | null>(null);
  const [playing, setPlaying] = useState<{ name: string; url: string } | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [worker, setWorker] = useState<WorkerStatus | null>(null);
  const [workerNote, setWorkerNote] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  // -- the worker ---------------------------------------------------------------------------------
  useEffect(() => {
    if (!active) return undefined;
    let live = true;
    const poll = async () => {
      try {
        const status = await getVoiceWorker();
        if (live) {
          setWorker(status);
          setWorkerNote((current) => (current && current.startsWith('Stopped') ? current : null));
        }
      } catch (err) {
        if (live) setWorkerNote(routeMissing(err) ? 'The worker status is not available on this backend yet.' : `Worker: ${errorText(err)}`);
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), WORKER_POLL_MS);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [active]);

  const stopWorker = async () => {
    setStopping(true);
    setWorkerNote(null);
    try {
      const status = await stopVoiceWorker();
      if (alive.current) {
        setWorker(status);
        setWorkerNote('Stopped: the model is unloaded and the GPU is free.');
      }
    } catch (err) {
      if (alive.current) setWorkerNote(`Stop failed: ${errorText(err)}`);
    } finally {
      if (alive.current) setStopping(false);
    }
  };

  // -- add a voice ------------------------------------------------------------------------------
  const takeRecording = (rec: RecordedClip) => {
    setClip({ blob: rec.blob, filename: rec.filename, label: `microphone recording, ${rec.seconds.toFixed(1)} s` });
    setNote(null);
    return true;
  };

  const pickFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setClip({ blob: file, filename: file.name, label: file.name });
    if (!name.trim()) setName(suggestName(file.name));
    setNote(null);
  };

  const nameProblem = clip ? voiceNameProblem(name) : null;
  const nameTaken = !!name.trim() && voiceInLibrary(voices, name.trim());

  const save = async () => {
    if (!clip) return;
    const problem = voiceNameProblem(name);
    if (problem) {
      setNote(problem);
      return;
    }
    setSaving(true);
    setNote(null);
    try {
      const res = await addVoice(clip.blob, { name: name.trim(), kind, consent: consent.trim(), filename: clip.filename });
      await onChanged();
      if (alive.current) {
        setClip(null);
        setName('');
        setConsent('');
        setNote(`Saved the voice "${res.voice.name}" (${secs(res.voice.seconds)}).`);
      }
    } catch (err) {
      if (alive.current) setNote(routeMissing(err) ? MISSING : `Could not save the voice: ${errorText(err)}`);
    } finally {
      if (alive.current) setSaving(false);
    }
  };

  // -- rename, delete, play ---------------------------------------------------------------------
  const rename = async (voice: VoiceEntry, next: string) => {
    setRenaming(null);
    const target = next.trim();
    if (!target || target === voice.name) return;
    const problem = voiceNameProblem(target);
    if (problem) {
      setNote(problem);
      return;
    }
    setBusyName(voice.name);
    setNote(null);
    try {
      await renameVoice(voice.name, target);
      await onChanged();
      if (alive.current) setNote(`Renamed "${voice.name}" to "${target}".`);
    } catch (err) {
      if (alive.current) setNote(`Could not rename: ${errorText(err)}`);
    } finally {
      if (alive.current) setBusyName(null);
    }
  };

  const remove = async (voice: VoiceEntry) => {
    if (!window.confirm(`Delete the voice "${voice.name}"? Its reference clip is removed from the library.`)) return;
    setBusyName(voice.name);
    setNote(null);
    try {
      await deleteVoice(voice.name);
      if (alive.current && playing?.name === voice.name) setPlaying(null);
      await onChanged();
      if (alive.current) setNote(`Deleted "${voice.name}".`);
    } catch (err) {
      if (alive.current) setNote(`Could not delete: ${errorText(err)}`);
    } finally {
      if (alive.current) setBusyName(null);
    }
  };

  /** Which preset voices exist: "woman (Woman, Ghost)" and the ones still to record. */
  const presetVoices = useMemo(() => {
    const byVoice = new Map<string, string[]>();
    for (const p of Object.values(presets?.presets || {})) {
      if (!p.voice) continue;
      byVoice.set(p.voice, [...(byVoice.get(p.voice) || []), p.name]);
    }
    return Array.from(byVoice.entries()).map(([voice, names]) => ({ voice, names, present: voiceInLibrary(voices, voice) }));
  }, [presets, voices]);

  const workerLine = worker?.running
    ? `Worker: running on ${worker.device || 'an unknown device'}${typeof worker.idle_s === 'number' ? ` · idle ${Math.round(worker.idle_s)} s` : ''} (it unloads by itself after 10 min without work)`
    : worker
      ? 'Worker: idle, no model loaded. The first conversion loads seed-vc on the GPU (about 30 s; the very first run downloads its checkpoints).'
      : 'Worker: …';

  return (
    <div className="voice-tab">
      <div className="audio-card voice-worker">
        <div className="voice-row">
          <span className={`voice-dot${worker?.running ? ' on' : ''}`} />
          <span className="audio-inline">{workerLine}</span>
          <span className="voice-spacer" />
          <button type="button" className="audio-btn" disabled={!worker?.running || stopping}
            title="Quit after the running job and unload the model (frees the GPU)" onClick={() => void stopWorker()}>
            {stopping ? 'Stopping…' : 'Stop'}
          </button>
        </div>
        {workerNote && <div className="audio-hint">{workerNote}</div>}
      </div>

      <div className="voice-consent">
        <strong>Consent.</strong> A real person's voice needs their consent. Record your own voice or one you have
        permission to use, and say so in the consent note (for example "my own voice" or "consent from Asha,
        2026-10-09").
      </div>

      <div className="audio-card">
        <div className="audio-card-title">Add a voice</div>
        <div className="fx-scope voice-modes">
          <button type="button" className={`insp-chip${mode === 'record' ? ' active' : ''}`} onClick={() => setMode('record')}>
            Record
          </button>
          <button type="button" className={`insp-chip${mode === 'upload' ? ' active' : ''}`} onClick={() => setMode('upload')}>
            Upload a file
          </button>
        </div>
        {mode === 'record' ? (
          <Recorder useLabel="Use this take" onUse={takeRecording}
            hint="5-20 s of dry speech works best (no music, no reverb); the library takes 3-60 s." />
        ) : (
          <div className="voice-row">
            <button type="button" className="audio-btn" onClick={() => fileInput.current?.click()}>Choose a file…</button>
            <span className="audio-hint">Any audio (or video with sound) ffmpeg reads; 3-60 s.</span>
            <input ref={fileInput} type="file" hidden accept="audio/*,video/*,.wav,.mp3,.m4a,.flac,.ogg,.opus,.webm,.aac"
              onChange={pickFile} />
          </div>
        )}
        {clip && (
          <>
            <div className="voice-source">
              Clip: {clip.label}
              <button type="button" className="audio-btn" disabled={saving} onClick={() => setClip(null)}>Clear</button>
            </div>
            <label className="voice-field">
              <span className="voice-label">Name</span>
              <input className="audio-text" value={name} maxLength={63} placeholder="e.g. demon"
                onChange={(e) => setName(e.target.value)} />
            </label>
            {presetVoices.some((p) => !p.present) && (
              <div className="audio-hint">
                Presets look their voice up by name: {presetVoices.map((p) => `${p.voice} (${p.names.join(', ')})${p.present ? '' : ' missing'}`).join(' · ')}.
              </div>
            )}
            <label className="voice-field">
              <span className="voice-label">Kind</span>
              <select className="audio-select" value={kind} onChange={(e) => setKind(e.target.value)}>
                {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
            </label>
            <label className="voice-field">
              <span className="voice-label">Consent note</span>
              <input className="audio-text" value={consent} placeholder="my own voice / consent from <name>, <date>"
                onChange={(e) => setConsent(e.target.value)} />
            </label>
            {name.trim() && nameProblem && <div className="audio-hint">{nameProblem}</div>}
            {nameTaken && <div className="audio-hint">A voice named "{name.trim()}" exists: rename or delete it first.</div>}
            <div className="voice-row">
              <button type="button" className="audio-btn primary" disabled={saving || !!nameProblem || nameTaken}
                onClick={() => void save()}>
                {saving ? 'Saving…' : 'Save to the library'}
              </button>
            </div>
          </>
        )}
      </div>

      <div className="audio-card">
        <div className="audio-card-title">Library <span className="audio-badge">{voices.length}</span></div>
        {missing ? (
          <div className="audio-note">{MISSING}</div>
        ) : !loaded ? (
          <div className="audio-hint">Loading the voices…</div>
        ) : voices.length === 0 ? (
          <div className="audio-hint">No voices yet: record or upload a reference above.</div>
        ) : (
          <div className="voice-list">
            {voices.map((v) => {
              const busy = busyName === v.name;
              return (
                <div key={v.name} className="voice-item">
                  <div className="voice-row">
                    {renaming === v.name ? (
                      <input className="audio-text" autoFocus defaultValue={v.name} maxLength={63}
                        onBlur={(e) => void rename(v, e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                          if (e.key === 'Escape') setRenaming(null);
                        }} />
                    ) : (
                      <span className="voice-item-name" title={v.path}>{v.name}</span>
                    )}
                    <span className="audio-badge">{v.kind || 'custom'}</span>
                    <span className="audio-unit">{secs(v.seconds)}</span>
                    <span className="voice-spacer" />
                    <button type="button" className="audio-btn"
                      onClick={() => setPlaying(playing?.name === v.name ? null : { name: v.name, url: audioFileUrl(v.path) })}>
                      {playing?.name === v.name ? 'Close' : 'Play'}
                    </button>
                    <button type="button" className="audio-btn" disabled={busy} onClick={() => setRenaming(v.name)}>Rename</button>
                    <button type="button" className="audio-btn danger" disabled={busy} onClick={() => void remove(v)}>Delete</button>
                  </div>
                  <div className="voice-item-sub">
                    {v.consent ? `Consent: ${v.consent}` : 'No consent note'}
                    {v.created ? ` · added ${String(v.created).slice(0, 10)}` : ''}
                    {v.source ? ` · from ${v.source}` : ''}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {playing && <audio className="audio-preview" controls autoPlay src={playing.url} />}
        {loadNote && <div className="audio-note">{loadNote}</div>}
      </div>
      {note && <div className="audio-note">{note}</div>}
    </div>
  );
}
