/**
 * Microphone recording for the Voice panel: MediaRecorder (webm/opus in Chromium) with a live
 * level meter from an AnalyserNode and a length guard (3-60 s by default, the voices library's
 * limits). The take is played back before it is used; "Use" hands the Blob to the tab, which
 * uploads it (a reference voice, or a media-pool item to convert). The browser's voice
 * processing (echo cancellation, noise suppression, auto gain) is off: a reference clip and a
 * line to convert should be the voice as it is.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export interface RecordedClip {
  blob: Blob;
  seconds: number;
  mimeType: string;
  /** a file name with the container's extension, e.g. recording-20261009-142233.webm */
  filename: string;
}

type Phase = 'idle' | 'starting' | 'recording' | 'recorded';

const MIME_TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg', 'audio/mp4'];

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') return undefined;
  return MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
}

function extensionOf(mimeType: string): string {
  if (mimeType.includes('ogg')) return 'ogg';
  if (mimeType.includes('mp4')) return 'm4a';
  return 'webm';
}

function stamp(date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}-${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
}

const toDb = (amplitude: number) => (amplitude > 0 ? 20 * Math.log10(amplitude) : -120);
/** -60..0 dBFS onto 0..100 % of the meter. */
const meterPercent = (db: number) => Math.min(100, Math.max(0, ((db + 60) / 60) * 100));

function micError(err: unknown): string {
  const name = err instanceof DOMException ? err.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') return 'Microphone access was refused. Allow it for BuzzEdit and try again.';
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No microphone found.';
  if (name === 'NotReadableError') return 'The microphone is busy in another application.';
  return `Could not start the microphone: ${err instanceof Error ? err.message : String(err)}`;
}

export default function Recorder({ minSeconds = 3, maxSeconds = 60, useLabel = 'Use recording', busy = false, hint, onUse }: {
  minSeconds?: number;
  maxSeconds?: number;
  useLabel?: string;
  /** the tab is uploading: the buttons wait */
  busy?: boolean;
  hint?: string;
  /** true = the take was used (the recorder empties); false or a throw = keep it to retry */
  onUse: (clip: RecordedClip) => Promise<boolean> | boolean;
}) {
  const [phase, setPhase] = useState<Phase>('idle');
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState({ rms: -120, peak: -120 });
  const [error, setError] = useState<string | null>(null);
  const [clip, setClip] = useState<(RecordedClip & { url: string }) | null>(null);

  const alive = useRef(true);
  const streamRef = useRef<MediaStream | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const rafRef = useRef<number | null>(null);
  const capRef = useRef<number | null>(null);
  const startedAt = useRef(0);
  const stoppedAt = useRef(0);
  const urlRef = useRef<string | null>(null);

  const releaseInput = useCallback(() => {
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    if (capRef.current != null) window.clearTimeout(capRef.current);
    capRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    const context = contextRef.current;
    contextRef.current = null;
    if (context && context.state !== 'closed') void context.close().catch(() => {});
  }, []);

  const dropClip = useCallback(() => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
    setClip(null);
  }, []);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      const recorder = recorderRef.current;
      recorderRef.current = null;
      if (recorder && recorder.state !== 'inactive') {
        recorder.onstop = null;
        try { recorder.stop(); } catch { /* already stopped */ }
      }
      releaseInput();
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    };
  }, [releaseInput]);

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === 'inactive') return;
    stoppedAt.current = performance.now();
    try { recorder.stop(); } catch { /* already stopped */ }
  }, []);

  const finish = useCallback((mimeType: string) => {
    const seconds = Math.min(maxSeconds, Math.max(0, (stoppedAt.current - startedAt.current) / 1000));
    releaseInput();
    recorderRef.current = null;
    const blob = new Blob(chunksRef.current, { type: mimeType });
    chunksRef.current = [];
    if (!alive.current) return;
    setLevel({ rms: -120, peak: -120 });
    if (seconds < minSeconds || blob.size === 0) {
      setPhase('idle');
      setError(`Too short: ${seconds.toFixed(1)} s. Record at least ${minSeconds} s.`);
      return;
    }
    const url = URL.createObjectURL(blob);
    urlRef.current = url;
    setClip({ blob, seconds, mimeType, filename: `recording-${stamp()}.${extensionOf(mimeType)}`, url });
    setPhase('recorded');
  }, [maxSeconds, minSeconds, releaseInput]);

  const start = async () => {
    setError(null);
    dropClip();
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('This window cannot record audio (no microphone API).');
      return;
    }
    setPhase('starting');
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      });
    } catch (err) {
      if (alive.current) {
        setPhase('idle');
        setError(micError(err));
      }
      return;
    }
    if (!alive.current) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    streamRef.current = stream;

    let analyser: AnalyserNode | null = null;
    try {
      const context = new AudioContext();
      contextRef.current = context;
      analyser = context.createAnalyser();
      analyser.fftSize = 2048;
      context.createMediaStreamSource(stream).connect(analyser);
      void context.resume().catch(() => {});
    } catch {
      analyser = null; // no meter, the recording still works
    }

    const mimeType = pickMimeType();
    let recorder: MediaRecorder;
    try {
      recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    } catch (err) {
      releaseInput();
      setPhase('idle');
      setError(`Could not record: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    chunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => finish(recorder.mimeType || mimeType || 'audio/webm');
    recorderRef.current = recorder;
    recorder.start(250);
    startedAt.current = performance.now();
    stoppedAt.current = startedAt.current;
    setElapsed(0);
    setPhase('recording');
    // The guard: stop at the limit even when the window is in the background (no frames then).
    capRef.current = window.setTimeout(stop, maxSeconds * 1000);

    const samples = analyser ? new Float32Array(analyser.fftSize) : null;
    const frame = () => {
      if (!recorderRef.current || recorderRef.current.state !== 'recording') return;
      const seconds = (performance.now() - startedAt.current) / 1000;
      if (analyser && samples) {
        analyser.getFloatTimeDomainData(samples);
        let peak = 0;
        let sum = 0;
        for (let i = 0; i < samples.length; i += 1) {
          const v = samples[i];
          const a = Math.abs(v);
          if (a > peak) peak = a;
          sum += v * v;
        }
        setLevel({ rms: toDb(Math.sqrt(sum / samples.length)), peak: toDb(peak) });
      }
      setElapsed(seconds);
      if (seconds >= maxSeconds) {
        stop();
        return;
      }
      rafRef.current = requestAnimationFrame(frame);
    };
    rafRef.current = requestAnimationFrame(frame);
  };

  const use = async () => {
    if (!clip) return;
    setError(null);
    try {
      const used = await onUse({ blob: clip.blob, seconds: clip.seconds, mimeType: clip.mimeType, filename: clip.filename });
      if (used && alive.current) {
        dropClip();
        setPhase('idle');
      }
    } catch (err) {
      if (alive.current) setError(err instanceof Error ? err.message : String(err));
    }
  };

  const recording = phase === 'recording';
  const tooShort = recording && elapsed < minSeconds;

  return (
    <div className="voice-recorder">
      <div className="voice-row">
        {recording ? (
          <button type="button" className="audio-btn primary" onClick={stop}
            title={tooShort ? `Keep going: at least ${minSeconds} s` : 'Stop and listen back'}>
            Stop
          </button>
        ) : (
          <button type="button" className="audio-btn" disabled={phase === 'starting' || busy}
            onClick={() => void start()} title={`Record ${minSeconds}-${maxSeconds} s from the microphone`}>
            <span className="voice-rec-dot" />{phase === 'starting' ? 'Starting…' : clip ? 'Record again' : 'Record'}
          </button>
        )}
        <span className={`voice-rec-time${recording ? ' live' : ''}`}>
          {recording ? `${elapsed.toFixed(1)} s / ${maxSeconds} s` : clip ? `${clip.seconds.toFixed(1)} s` : `${minSeconds}-${maxSeconds} s`}
        </span>
        {tooShort && <span className="audio-hint">at least {minSeconds} s</span>}
      </div>
      {recording && (
        <div className="voice-row" title="Input level: the bar is RMS, the tick the peak (dBFS)">
          <div className="voice-meter">
            <div className="voice-meter-mask" style={{ left: `${meterPercent(level.rms)}%` }} />
            <div className="voice-meter-peak" style={{ left: `${meterPercent(level.peak)}%` }} />
          </div>
          <span className="audio-unit">{level.rms > -120 ? `${level.rms.toFixed(0)} dBFS` : '–'}</span>
          {level.peak > -1 && <span className="voice-clip">too loud</span>}
        </div>
      )}
      {clip && phase === 'recorded' && (
        <>
          <audio className="audio-preview" controls src={clip.url} />
          <div className="voice-row">
            <button type="button" className="audio-btn primary" disabled={busy} onClick={() => void use()}>
              {busy ? 'Working…' : useLabel}
            </button>
            <button type="button" className="audio-btn" disabled={busy}
              onClick={() => { dropClip(); setPhase('idle'); }}>
              Discard
            </button>
          </div>
        </>
      )}
      {hint && phase !== 'recorded' && <div className="audio-hint">{hint}</div>}
      {error && <div className="audio-note">{error}</div>}
    </div>
  );
}
