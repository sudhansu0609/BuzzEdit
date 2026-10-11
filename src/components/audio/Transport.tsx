/**
 * Transport of the live engine (AUDIO_SUITE_PLAN.md §2.3): Play / Pause, Loop the range, the
 * position the server reports (`onState`, about four times a second) and A/B -- the whole rack
 * bypassed against the processed sound. The position arrives through a Signal, so only this
 * strip repaints while the audio plays.
 */

import { useEffect, useState } from 'react';
import { Pause, Play, Repeat } from 'lucide-react';
import type { Signal } from '../../hooks/audioApi';

export interface TransportRange {
  start_s: number;
  end_s: number;
  origin: 'selection' | 'clip' | 'programme';
}

export type LiveStatus = 'idle' | 'connecting' | 'live' | 'off';

export interface PlayState {
  t: number;
  playing: boolean;
}

export function formatTime(seconds: number): string {
  const t = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(2).padStart(5, '0')}`;
}

const ORIGIN: Record<TransportRange['origin'], string> = {
  selection: 'Selection',
  clip: 'Clip',
  programme: 'Whole programme',
};

const STATUS: Record<LiveStatus, string> = {
  idle: 'Idle',
  connecting: 'Connecting…',
  live: 'Live',
  off: 'Offline',
};

export default function Transport({
  position, playing, loopOn, ab, range, status, note,
  onPlayPause, onLoop, onAB, onReconnect,
}: {
  position: Signal<PlayState>;
  playing: boolean;
  loopOn: boolean;
  /** true = B: every effect bypassed */
  ab: boolean;
  range: TransportRange | null;
  status: LiveStatus;
  note: string | null;
  onPlayPause: () => void;
  onLoop: () => void;
  onAB: () => void;
  onReconnect: () => void;
}) {
  const [t, setT] = useState(0);
  useEffect(() => position.subscribe((state) => setT(state.t)), [position]);
  const live = status === 'live';
  const span = range ? `${formatTime(range.start_s)}–${formatTime(range.end_s)}` : '';

  return (
    <div className="audio-transport">
      <button className={`audio-tbtn primary${playing ? ' on' : ''}`} disabled={!live} onClick={onPlayPause}
        title={playing ? 'Pause (Space)' : 'Play (Space)'}>
        {playing ? <Pause size={14} /> : <Play size={14} />}
      </button>
      <button className={`audio-tbtn${loopOn ? ' on' : ''}`} disabled={!live || !range} onClick={onLoop}
        title={range ? `Loop ${ORIGIN[range.origin].toLowerCase()} ${span}` : 'Nothing to loop'}>
        <Repeat size={14} />
      </button>
      <button className={`audio-tbtn ab${ab ? ' on' : ''}`} disabled={!live} onClick={onAB}
        title={ab ? 'B: the rack is bypassed. Click to hear it processed (A).' : 'A: processed. Click to bypass the whole rack (B).'}>
        {ab ? 'B' : 'A'}
      </button>
      <span className="audio-time">{formatTime(t)}</span>
      <span className="audio-range" title="What Play, Loop, Analyze and the process effects use">
        {range ? `${ORIGIN[range.origin]} ${span}` : 'No range'}
      </span>
      <span className={`audio-live ${status}`} title={note ?? undefined}>{STATUS[status]}</span>
      {status === 'off' && (
        <button className="audio-btn" onClick={onReconnect}>Reconnect</button>
      )}
      {note && <div className="audio-transport-note">{note}</div>}
    </div>
  );
}
