/**
 * Live audio engine, browser side (AUDIO_SUITE_PLAN.md §2.3 / §2.6, brief L).
 *
 * Opens `WS /api/audio/<project>/live?scope=<scope>`, plays the PCM it receives through an
 * AudioWorklet ring buffer (`pcm-player.worklet.js`) at 48 kHz and grants the server credits
 * so that outstanding + buffered stays near 150 ms (output starts once 100 ms are queued).
 * Knob moves go out as `set` messages, at most one per 30 ms per parameter; the server applies
 * them to the running ffmpeg live or by a restart at the playhead. Meters (`meters.ts`) are
 * computed from the frames as they are enqueued. No UI here: the Audio panel drives it.
 */

import { API_BASE } from '../hooks/api';
import { LoudnessMeter, type MeterReading } from './meters';

export type { MeterReading } from './meters';

/** An effect of a rack, as `PUT /api/audio/{project}/rack` and the live `rack` message carry it. */
export interface LiveEffect {
  id: string;
  effect: string;
  params?: Record<string, unknown>;
  enabled?: boolean;
  start_s?: number | null;
  end_s?: number | null;
  preset?: string | null;
}

const SAMPLE_RATE = 48000;
const TOTAL_TARGET = Math.round(0.15 * SAMPLE_RATE); // outstanding credits + buffered frames
const PREBUFFER = Math.round(0.1 * SAMPLE_RATE); // frames queued before output starts
const MIN_GRANT = 512; // frames; smaller top-ups wait for the next one
const SET_INTERVAL_MS = 30; // per parameter
const METER_INTERVAL_MS = 50;

interface PendingSet {
  last: number;
  value?: unknown;
  timer?: ReturnType<typeof setTimeout>;
}

/** Messages to the worklet (`pcm-player.worklet.js`, plain JS, so the protocol is typed here). */
export type WorkletInMessage =
  | { type: 'push'; data: Float32Array } // interleaved L/R, transferred
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'flush' }
  | { type: 'drain' } // end of stream expected: running dry is not an underrun
  | { type: 'config'; prebuffer: number };

/** The worklet's report: every 512 frames played, after every push/flush, once per underrun. */
export interface WorkletStatus {
  type: 'status';
  buffered: number;
  written: number;
  played: number;
  underruns: number;
  overflows: number;
}

export class LiveEngine {
  onState: ((t: number, playing: boolean) => void) | null = null;
  onMeters: ((meters: MeterReading) => void) | null = null;
  onError: ((message: string) => void) | null = null;

  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private ws: WebSocket | null = null;
  private meter = new LoudnessMeter(SAMPLE_RATE, 2);
  private credited = 0; // frames granted to the server, ever
  private received = 0; // frames received, ever
  private posted = 0; // frames handed to the worklet, ever
  private workletBuffered = 0;
  private workletWritten = 0;
  private discarding = 0; // seeks still waiting for their `restarted`
  private ended = false;
  private closing = false;
  private lastMeters = 0;
  private sets = new Map<string, PendingSet>();
  private t = 0;
  private playing = false;
  /** Underruns the worklet has counted since connect (0 is the target over 60 s of play). */
  underruns = 0;

  /** Load the player and open the socket; resolves once the socket is open. */
  async connect(projectId: string, scope = 'A1', rack?: LiveEffect[]): Promise<void> {
    this.disconnect();
    this.closing = false;
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: 'interactive' });
    this.ctx = ctx;
    // Plain JS, copied verbatim by Vite in dev and build alike.
    await ctx.audioWorklet.addModule(new URL('./pcm-player.worklet.js', import.meta.url).href);
    const node = new AudioWorkletNode(ctx, 'pcm-player', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });
    node.port.onmessage = (event: MessageEvent) => this.onWorklet(event.data as WorkletStatus);
    node.connect(ctx.destination);
    this.node = node;
    this.post({ type: 'config', prebuffer: PREBUFFER });

    const base = API_BASE || window.location.origin;
    const url =
      base.replace(/^http/, 'ws') +
      `/api/audio/${encodeURIComponent(projectId)}/live?scope=${encodeURIComponent(scope)}`;
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') this.onJson(event.data);
      else this.onPcm(event.data as ArrayBuffer);
    };
    ws.onclose = () => {
      if (!this.closing) this.onError?.('live audio connection closed');
    };
    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = () => {
        this.onError?.('live audio connection failed');
        reject(new Error('live audio connection failed'));
      };
    });
    if (rack) this.replaceRack(rack);
  }

  async play(): Promise<void> {
    if (this.ctx?.state === 'suspended') await this.ctx.resume();
    if (this.ended) {
      // the server starts again from the top (or the loop start): drop the old tail
      this.post({ type: 'flush' });
      this.ended = false;
    }
    this.post({ type: 'play' });
    this.send({ type: 'play' });
    this.topUp();
  }

  pause(): void {
    this.send({ type: 'pause' });
    this.post({ type: 'pause' });
  }

  seek(t: number): void {
    this.jump();
    this.send({ type: 'seek', t });
  }

  loop(a: number, b: number): void {
    this.jump();
    this.send({ type: 'loop', a, b });
  }

  loopOff(): void {
    this.send({ type: 'loop', off: true });
  }

  /** A knob moved: at most one message per 30 ms per parameter, the latest value always sent. */
  set(effectId: string, param: string, value: unknown): void {
    const key = `${effectId}\u0000${param}`;
    const now = performance.now();
    const entry = this.sets.get(key) ?? { last: -Infinity };
    this.sets.set(key, entry);
    entry.value = value;
    if (entry.timer !== undefined) return; // a send is already scheduled; it takes the latest value
    const wait = entry.last + SET_INTERVAL_MS - now;
    const flush = () => {
      entry.timer = undefined;
      entry.last = performance.now();
      this.send({ type: 'set', effect_id: effectId, param, value: entry.value });
    };
    if (wait <= 0) flush();
    else entry.timer = setTimeout(flush, wait);
  }

  bypass(effectId: string, on: boolean): void {
    this.send({ type: 'bypass', effect_id: effectId, on });
  }

  replaceRack(effects: LiveEffect[]): void {
    for (const entry of this.sets.values()) if (entry.timer !== undefined) clearTimeout(entry.timer);
    this.sets.clear();
    this.send({ type: 'rack', effects });
  }

  disconnect(): void {
    this.closing = true;
    for (const entry of this.sets.values()) if (entry.timer !== undefined) clearTimeout(entry.timer);
    this.sets.clear();
    if (this.ws) {
      this.ws.onmessage = this.ws.onclose = this.ws.onerror = this.ws.onopen = null;
      if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) this.ws.close();
      this.ws = null;
    }
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
      this.node = null;
    }
    if (this.ctx) {
      void this.ctx.close();
      this.ctx = null;
    }
    this.credited = this.received = this.posted = 0;
    this.workletBuffered = this.workletWritten = 0;
    this.discarding = 0;
    this.ended = false;
    this.playing = false;
    this.meter.reset();
  }

  // -- internals ---------------------------------------------------------------------------

  /** A seek or a new loop: what is queued or still in flight belongs to the old position. */
  private jump(): void {
    this.discarding++;
    this.ended = false;
    this.post({ type: 'flush' });
    this.meter.reset();
  }

  private post(msg: WorkletInMessage, transfer: Transferable[] = []): void {
    this.node?.port.postMessage(msg, transfer);
  }

  private send(msg: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  private bufferedFrames(): number {
    // what the worklet reported, plus what was posted to it since
    return this.workletBuffered + Math.max(0, this.posted - this.workletWritten);
  }

  private topUp(): void {
    const outstanding = this.credited - this.received;
    const deficit = TOTAL_TARGET - (outstanding + this.bufferedFrames());
    if (deficit >= MIN_GRANT && this.ws?.readyState === WebSocket.OPEN) {
      this.credited += deficit;
      this.send({ type: 'credits', frames: deficit });
    }
  }

  private onPcm(buffer: ArrayBuffer): void {
    const s16 = new Int16Array(buffer);
    const frames = s16.length >> 1;
    this.received += frames;
    if (this.discarding > 0 || !this.node) {
      this.topUp();
      return;
    }
    const f32 = new Float32Array(s16.length);
    for (let i = 0; i < s16.length; i++) f32[i] = s16[i] / 32768;
    this.meter.push(f32);
    this.posted += frames;
    this.post({ type: 'push', data: f32 }, [f32.buffer]);
    const now = performance.now();
    if (now - this.lastMeters >= METER_INTERVAL_MS) {
      this.lastMeters = now;
      this.onMeters?.(this.meter.read());
    }
    this.topUp();
  }

  private onJson(text: string): void {
    let msg: { type?: string; t?: number; playing?: boolean; message?: string };
    try {
      msg = JSON.parse(text);
    } catch {
      return;
    }
    switch (msg.type) {
      case 'state':
        this.t = msg.t ?? this.t;
        this.playing = !!msg.playing;
        this.onState?.(this.t, this.playing);
        break;
      case 'restarted':
        if (this.discarding > 0) this.discarding--;
        this.t = msg.t ?? this.t;
        this.onState?.(this.t, this.playing);
        break;
      case 'ended':
        this.discarding = 0; // a seek past the end ends without a `restarted`
        this.ended = true;
        this.playing = false;
        this.post({ type: 'drain' });
        this.onState?.(this.t, false);
        break;
      case 'error':
        this.discarding = 0; // a failed restart sends no `restarted`
        this.onError?.(msg.message ?? 'live audio error');
        break;
    }
  }

  private onWorklet(status: WorkletStatus): void {
    if (status.type !== 'status') return;
    this.workletBuffered = status.buffered;
    this.workletWritten = status.written;
    this.underruns = status.underruns;
    this.topUp();
  }
}
