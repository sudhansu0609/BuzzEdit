/**
 * AudioWorklet side of the live audio engine (AUDIO_SUITE_PLAN.md §2.3, brief L).
 *
 * Plain JavaScript on purpose: LiveEngine.ts loads it with
 * `new URL('./pcm-player.worklet.js', import.meta.url)`, which Vite copies verbatim as an
 * asset in dev and build alike, so it must run as written. The message protocol is typed in
 * LiveEngine.ts (`WorkletInMessage`, `WorkletStatus`).
 *
 * A stereo ring buffer that LiveEngine fills with the PCM coming off the socket and the audio
 * thread plays 128 frames at a time. Output starts once `prebuffer` frames are queued (and
 * again after an underrun); paused, it outputs silence and keeps what it holds.
 *
 * Messages in:  {type:'push', data: Float32Array interleaved L/R} (transferred),
 *               {type:'play'} | {type:'pause'} | {type:'flush'} | {type:'drain'} (end of stream
 *               expected: running dry is not an underrun) | {type:'config', prebuffer}.
 * Messages out: {type:'status', buffered, written, played, underruns, overflows} every 512
 *               frames played, after every push/flush and once per underrun.
 */

const CAPACITY = 96000; // frames: 2 s at 48 kHz, far above the ~150 ms the credits allow
const STATUS_EVERY = 512; // frames played between status messages

class PcmPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring = new Float32Array(CAPACITY * 2);
    this.readPos = 0; // frames
    this.writePos = 0; // frames
    this.buffered = 0; // frames
    this.written = 0; // frames ever accepted
    this.played = 0; // frames ever played
    this.underruns = 0;
    this.overflows = 0;
    this.playing = false;
    this.started = false;
    this.draining = false;
    this.prebuffer = 4800;
    this.sinceStatus = 0;
    this.port.onmessage = (event) => this.onMessage(event.data);
  }

  onMessage(msg) {
    switch (msg.type) {
      case 'push':
        if (msg.data) this.push(msg.data);
        break;
      case 'play':
        this.playing = true;
        break;
      case 'pause':
        this.playing = false;
        break;
      case 'flush':
        this.readPos = this.writePos = this.buffered = 0;
        this.started = false;
        this.draining = false;
        this.status();
        break;
      case 'drain':
        this.draining = true;
        break;
      case 'config':
        if (typeof msg.prebuffer === 'number') this.prebuffer = Math.max(0, msg.prebuffer | 0);
        break;
    }
  }

  push(data) {
    const offered = data.length >> 1;
    const frames = Math.min(offered, CAPACITY - this.buffered);
    if (frames < offered) this.overflows++;
    for (let i = 0; i < frames; i++) {
      const w = ((this.writePos + i) % CAPACITY) * 2;
      this.ring[w] = data[i * 2];
      this.ring[w + 1] = data[i * 2 + 1];
    }
    this.writePos = (this.writePos + frames) % CAPACITY;
    this.buffered += frames;
    this.written += offered;
    this.status();
  }

  status() {
    this.port.postMessage({
      type: 'status',
      buffered: this.buffered,
      written: this.written,
      played: this.played,
      underruns: this.underruns,
      overflows: this.overflows,
    });
  }

  process(_inputs, outputs) {
    const out = outputs[0];
    const left = out[0];
    const right = out[1] || out[0];
    const n = left.length;
    // At the end of the stream whatever is left plays, even below the prebuffer.
    const need = this.draining ? 1 : Math.max(1, this.prebuffer);
    if (!this.started && this.buffered >= need) this.started = true;
    if (!this.playing || !this.started) {
      left.fill(0);
      if (right !== left) right.fill(0);
      return true;
    }
    const take = Math.min(n, this.buffered);
    for (let i = 0; i < take; i++) {
      const r = ((this.readPos + i) % CAPACITY) * 2;
      left[i] = this.ring[r];
      right[i] = this.ring[r + 1];
    }
    if (take < n) {
      left.fill(0, take);
      if (right !== left) right.fill(0, take);
      if (!this.draining) this.underruns++;
      this.started = false; // re-prebuffer before playing on
    }
    this.readPos = (this.readPos + take) % CAPACITY;
    this.buffered -= take;
    this.played += take;
    this.sinceStatus += take;
    if (this.sinceStatus >= STATUS_EVERY || take < n) {
      this.sinceStatus = 0;
      this.status();
    }
    return true;
  }
}

registerProcessor('pcm-player', PcmPlayer);
