/**
 * Level meters for the live audio engine (AUDIO_SUITE_PLAN.md §2.3, brief L).
 *
 * Fed with the interleaved float PCM the player enqueues. `read()` returns the sample peak
 * and RMS (dBFS) of everything pushed since the previous read, plus ITU-R BS.1770-4 loudness:
 * K-weighting (high shelf + RLB high-pass) per channel, mean square summed over L/R with
 * weight 1.0, L = -0.691 + 10·log10(Σ), over the last 400 ms (momentary) and 3 s (short-term),
 * updated every 100 ms block. Silence, or less than 400 ms of audio, reads -Infinity.
 */

export interface MeterReading {
  /** dBFS, sample peak since the previous read */
  peak: number;
  /** dBFS, RMS over both channels since the previous read */
  rms: number;
  /** LUFS, 400 ms window */
  momentaryLufs: number;
  /** LUFS, 3 s window (all available blocks until 3 s have passed) */
  shortTermLufs: number;
}

interface Biquad {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

/**
 * The two K-weighting stages for any sample rate, computed from the analog prototype as
 * libebur128 does; at 48 kHz they are the coefficients tabulated in BS.1770-4
 * (stage 1 b0 = 1.53512485958697, stage 2 a1 = -1.99004745483398).
 */
export function kWeighting(rate: number): [Biquad, Biquad] {
  let f0 = 1681.974450955533;
  const G = 3.999843853973347;
  let Q = 0.7071752369554196;
  let K = Math.tan((Math.PI * f0) / rate);
  const Vh = Math.pow(10, G / 20);
  const Vb = Math.pow(Vh, 0.4996667741545416);
  let a0 = 1 + K / Q + K * K;
  const shelf: Biquad = {
    b0: (Vh + (Vb * K) / Q + K * K) / a0,
    b1: (2 * (K * K - Vh)) / a0,
    b2: (Vh - (Vb * K) / Q + K * K) / a0,
    a1: (2 * (K * K - 1)) / a0,
    a2: (1 - K / Q + K * K) / a0,
  };
  f0 = 38.13547087602444;
  Q = 0.5003270373238773;
  K = Math.tan((Math.PI * f0) / rate);
  a0 = 1 + K / Q + K * K;
  const highpass: Biquad = {
    b0: 1,
    b1: -2,
    b2: 1,
    a1: (2 * (K * K - 1)) / a0,
    a2: (1 - K / Q + K * K) / a0,
  };
  return [shelf, highpass];
}

const toDb = (x: number): number => (x > 0 ? 20 * Math.log10(x) : -Infinity);
const toLufs = (meanSquare: number): number =>
  meanSquare > 0 ? -0.691 + 10 * Math.log10(meanSquare) : -Infinity;

const MOMENTARY_BLOCKS = 4; // 4 x 100 ms
const SHORT_TERM_BLOCKS = 30; // 30 x 100 ms

export class LoudnessMeter {
  private readonly channels: number;
  private readonly stages: [Biquad, Biquad];
  private readonly blockLen: number;
  /** transposed direct form II state: [stage1 z1, stage1 z2, stage2 z1, stage2 z2] per channel */
  private state: Float64Array;
  private blockSum: Float64Array;
  private blockFill = 0;
  /** channel-summed mean square of each finished 100 ms block, newest last */
  private blocks: number[] = [];
  private peak = 0;
  private sumSquares = 0;
  private samples = 0;

  constructor(sampleRate = 48000, channels = 2) {
    this.channels = channels;
    this.stages = kWeighting(sampleRate);
    this.blockLen = Math.round(sampleRate / 10);
    this.state = new Float64Array(4 * channels);
    this.blockSum = new Float64Array(channels);
  }

  /** Forget the history (a seek is a discontinuity). */
  reset(): void {
    this.state.fill(0);
    this.blockSum.fill(0);
    this.blockFill = 0;
    this.blocks = [];
    this.peak = 0;
    this.sumSquares = 0;
    this.samples = 0;
  }

  /** Interleaved samples, `channels` per frame, -1..1. */
  push(interleaved: Float32Array): void {
    const ch = this.channels;
    const [s1, s2] = this.stages;
    const z = this.state;
    const frames = Math.floor(interleaved.length / ch);
    for (let f = 0; f < frames; f++) {
      for (let c = 0; c < ch; c++) {
        const x = interleaved[f * ch + c];
        const ax = Math.abs(x);
        if (ax > this.peak) this.peak = ax;
        this.sumSquares += x * x;
        const o = c * 4;
        const y1 = s1.b0 * x + z[o];
        z[o] = s1.b1 * x - s1.a1 * y1 + z[o + 1];
        z[o + 1] = s1.b2 * x - s1.a2 * y1;
        const y2 = s2.b0 * y1 + z[o + 2];
        z[o + 2] = s2.b1 * y1 - s2.a1 * y2 + z[o + 3];
        z[o + 3] = s2.b2 * y1 - s2.a2 * y2;
        this.blockSum[c] += y2 * y2;
      }
      this.samples += ch;
      if (++this.blockFill === this.blockLen) {
        let energy = 0;
        for (let c = 0; c < ch; c++) energy += this.blockSum[c] / this.blockLen;
        this.blocks.push(energy);
        if (this.blocks.length > SHORT_TERM_BLOCKS) this.blocks.shift();
        this.blockSum.fill(0);
        this.blockFill = 0;
      }
    }
  }

  /** The reading now; peak and RMS restart from here. */
  read(): MeterReading {
    const n = this.blocks.length;
    const mean = (count: number): number => {
      let s = 0;
      for (let i = n - count; i < n; i++) s += this.blocks[i];
      return s / count;
    };
    const reading: MeterReading = {
      peak: toDb(this.peak),
      rms: this.samples > 0 ? toDb(Math.sqrt(this.sumSquares / this.samples)) : -Infinity,
      momentaryLufs: n >= MOMENTARY_BLOCKS ? toLufs(mean(MOMENTARY_BLOCKS)) : -Infinity,
      shortTermLufs: n >= MOMENTARY_BLOCKS ? toLufs(mean(n)) : -Infinity,
    };
    this.peak = 0;
    this.sumSquares = 0;
    this.samples = 0;
    return reading;
  }
}
