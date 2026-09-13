/**
 * DSP engine for the voice-perception analyzer.
 * Pure TypeScript — runs identically in the browser (live mic + uploads).
 *
 * Pipeline:
 *   PCM (any rate, mono) -> resample to 16 kHz -> frame (2048 samples / 128 ms)
 *   -> energy VAD (adaptive noise floor) -> segments
 *   -> per segment: RMS/dBFS, HF energy ratio (FFT), pitch (autocorrelation)
 *   -> distance band classification + speaker clustering by pitch
 *   -> 16-bit PCM WAV encoding for the ASR backend
 */

export const TARGET_RATE = 16000;
export const FRAME_SIZE = 2048; // 128 ms @ 16 kHz
export const MAX_SEGMENT_MS = 15000;
const MIN_SEGMENT_MS = 400;
const PREROLL_FRAMES = 3;
const START_CONFIRM_FRAMES = 2;
const SILENCE_END_MS = 1100;

/* ------------------------------------------------------------------ types */

export type DistanceBand = 'very-close' | 'close' | 'normal' | 'far' | 'very-far';

export const DISTANCE_META: Record<
  DistanceBand,
  { label: string; range: string; hint: string; tone: string }
> = {
  'very-close': {
    label: 'Very Close',
    range: '0–0.5 m',
    hint: 'Practically on top of the mic — full presence, crisp treble.',
    tone: 'emerald',
  },
  close: {
    label: 'Close',
    range: '0.5–1 m',
    hint: 'Conversational distance, directly addressing the mic.',
    tone: 'lime',
  },
  normal: {
    label: 'Normal',
    range: '1–3 m',
    hint: 'Across a desk — clear but with a touch of room.',
    tone: 'yellow',
  },
  far: {
    label: 'Far',
    range: '3–6 m',
    hint: 'Other side of the room — reverberant, treble fading.',
    tone: 'amber',
  },
  'very-far': {
    label: 'Very Far',
    range: '6 m+',
    hint: 'Barely reaching the mic — muffled, echo-heavy.',
    tone: 'orange',
  },
};

export interface AcousticProfile {
  dbfs: number; // average loudness, dB full scale
  peakDbfs: number;
  hfRatio: number; // 0..1, share of spectral energy above ~3.5 kHz
  pitchHz: number | null; // estimated fundamental (F0) of loudest voiced frame
  distanceBand: DistanceBand;
  durationMs: number;
}

export interface SpeakerInfo {
  id: number;
  label: string;
  pitchHz: number | null;
  utterances: number;
}

/* -------------------------------------------------------------- utilities */

export function frameRms(frame: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
  return Math.sqrt(sum / frame.length);
}

export function resampleLinear(input: Float32Array, fromRate: number, toRate = TARGET_RATE): Float32Array {
  if (fromRate === toRate) return input;
  const ratio = toRate / fromRate;
  const outLen = Math.max(1, Math.floor(input.length * ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i / ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(input.length - 1, i0 + 1);
    const t = pos - i0;
    out[i] = input[i0] * (1 - t) + input[i1] * t;
  }
  return out;
}

/** Mix multi-channel signal down to a mono Float32Array. */
export function mixdownToMono(channels: Float32Array[], length: number): Float32Array {
  if (channels.length === 1) return channels[0];
  const out = new Float32Array(length);
  for (let c = 0; c < channels.length; c++) {
    const ch = channels[c];
    for (let i = 0; i < length; i++) out[i] += ch[i];
  }
  for (let i = 0; i < length; i++) out[i] /= channels.length;
  return out;
}

/* ------------------------------------------------------------------- FFT */

const HANN_2048 = (() => {
  const w = new Float32Array(FRAME_SIZE);
  for (let i = 0; i < FRAME_SIZE; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FRAME_SIZE - 1)));
  return w;
})();

/** In-place iterative radix-2 FFT. Both arrays must be the same power-of-two length. */
export function fftInPlace(re: Float32Array, im: Float32Array): void {
  const n = re.length;

  // bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  // butterflies
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let j = 0; j < half; j++) {
        const aRe = re[i + j];
        const aIm = im[i + j];
        const bRe = re[i + j + half] * curRe - im[i + j + half] * curIm;
        const bIm = re[i + j + half] * curIm + im[i + j + half] * curRe;
        re[i + j] = aRe + bRe;
        im[i + j] = aIm + bIm;
        re[i + j + half] = aRe - bRe;
        im[i + j + half] = aIm - bIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

/** In-place inverse FFT via the conjugation trick: ifft(x) = conj(fft(conj(x)))/n. */
export function ifftInPlace(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fftInPlace(re, im);
  for (let i = 0; i < n; i++) {
    re[i] /= n;
    im[i] = -im[i] / n;
  }
}

/** FFT of a real signal. Input length must be a power of two. Returns n/2 magnitudes. */
export function fftMagnitudes(input: Float32Array): Float32Array {
  const n = input.length;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  for (let i = 0; i < n; i++) re[i] = input[i];
  fftInPlace(re, im);
  const mags = new Float32Array(n >> 1);
  for (let i = 0; i < n >> 1; i++) mags[i] = Math.hypot(re[i], im[i]);
  return mags;
}

/** Spectral energy above `hz` relative to total energy, for one 2048-sample frame. */
export function hfEnergyRatio(frame: Float32Array, sampleRate: number, hz = 3500): number {
  if (frame.length !== FRAME_SIZE) return 0;
  const windowed = new Float32Array(FRAME_SIZE);
  for (let i = 0; i < FRAME_SIZE; i++) windowed[i] = frame[i] * HANN_2048[i];
  const mags = fftMagnitudes(windowed);
  const binHz = sampleRate / FRAME_SIZE;
  const cutoffBin = Math.max(1, Math.ceil(hz / binHz));
  let hi = 0;
  let tot = 0;
  for (let b = 1; b < mags.length; b++) {
    const e = mags[b] * mags[b];
    tot += e;
    if (b >= cutoffBin) hi += e;
  }
  return tot > 0 ? hi / tot : 0;
}

/* ------------------------------------------------------------ pitch (F0) */

/** Autocorrelation pitch estimator. Returns Hz, or null for unvoiced/quiet frames. */
export function estimatePitch(frame: Float32Array, sampleRate: number): number | null {
  const n = frame.length;
  const x = new Float32Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += frame[i];
  mean /= n;
  let energy = 0;
  for (let i = 0; i < n; i++) {
    x[i] = frame[i] - mean;
    energy += x[i] * x[i];
  }
  if (energy / n < 1e-4) return null; // too quiet to judge voicing

  const minLag = Math.floor(sampleRate / 400); // 400 Hz ceiling
  const maxLag = Math.min(Math.floor(sampleRate / 70), n - 2); // 70 Hz floor
  let bestLag = -1;
  let bestCorr = 0;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let corr = 0;
    for (let i = 0; i < n - lag; i++) corr += x[i] * x[i + lag];
    const normalized = (corr * n) / (energy * (n - lag) || 1);
    if (normalized > bestCorr) {
      bestCorr = normalized;
      bestLag = lag;
    }
  }
  if (bestCorr < 0.35 || bestLag <= 0) return null;
  return sampleRate / bestLag;
}

/* ---------------------------------------------------------- distance band */

/**
 * Heuristic distance classification from loudness + treble presence.
 * Louder + crisper highs => closer. Mic gain varies across devices, so this is
 * explicitly an estimate, not a measurement.
 */
export function classifyDistance(dbfs: number, hfRatio: number): DistanceBand {
  if (dbfs >= -14) return hfRatio >= 0.03 ? 'very-close' : 'close';
  if (dbfs >= -22) return hfRatio >= 0.12 ? 'close' : 'normal';
  if (dbfs >= -30) return hfRatio >= 0.05 ? 'normal' : 'far';
  if (dbfs >= -38) return 'far';
  return 'very-far';
}

/* --------------------------------------------------------------- VAD / VAD */

export interface RawSegment {
  pcm: Float32Array; // 16 kHz mono
  durationMs: number;
}

/**
 * Energy-based voice activity segmenter with an adaptive noise floor,
 * pre-roll capture and trailing-silence trimming.
 */
export class VoiceSegmenter {
  private preroll: Float32Array[] = [];
  private frames: Float32Array[] = [];
  private speaking = false;
  private loudRun = 0;
  private silenceRun = 0;
  private noiseFloor = 0.003;
  private frameCount = 0;
  vadState: 'silence' | 'speech' = 'silence';

  feed(frame: Float32Array): RawSegment | null {
    const rms = frameRms(frame);
    if (!this.speaking) {
      // track ambient level only while nobody is talking
      this.noiseFloor = this.noiseFloor * 0.96 + Math.min(rms, 0.2) * 0.04;
    }
    const startThresh = Math.max(0.012, this.noiseFloor * 3.5);
    const endThresh = Math.max(0.005, this.noiseFloor * 2.0);

    if (!this.speaking) {
      this.preroll.push(frame);
      if (this.preroll.length > PREROLL_FRAMES) this.preroll.shift();
      if (rms > startThresh) this.loudRun++;
      else this.loudRun = 0;
      if (this.loudRun >= START_CONFIRM_FRAMES) {
        this.speaking = true;
        this.vadState = 'speech';
        this.frames = this.preroll.slice();
        this.silenceRun = 0;
        this.frameCount = this.frames.length;
        this.preroll = [];
      }
      return null;
    }

    this.frames.push(frame);
    this.frameCount++;
    if (rms > endThresh) this.silenceRun = 0;
    else this.silenceRun++;

    const silenceMs = (this.silenceRun * FRAME_SIZE * 1000) / TARGET_RATE;
    const segMs = (this.frameCount * FRAME_SIZE * 1000) / TARGET_RATE;
    if (silenceMs >= SILENCE_END_MS || segMs >= MAX_SEGMENT_MS) {
      return this.finalize();
    }
    return null;
  }

  /** Finalize the in-progress segment (used on stop / end of file). */
  flush(): RawSegment | null {
    if (!this.speaking) return null;
    return this.finalize();
  }

  reset(): void {
    this.preroll = [];
    this.frames = [];
    this.speaking = false;
    this.loudRun = 0;
    this.silenceRun = 0;
    this.frameCount = 0;
    this.vadState = 'silence';
    this.noiseFloor = 0.003;
  }

  private finalize(): RawSegment | null {
    // trim most of the trailing silence, keep a little natural tail
    const trailingSilent = Math.max(0, this.silenceRun - 2);
    let keep = this.frames.length;
    if (trailingSilent > 0) keep = Math.max(PREROLL_FRAMES + 1, this.frames.length - trailingSilent);
    const kept = this.frames.slice(0, keep);
    const pcm = concatFrames(kept);
    const durationMs = (kept.length * FRAME_SIZE * 1000) / TARGET_RATE;

    this.frames = [];
    this.preroll = [];
    this.speaking = false;
    this.loudRun = 0;
    this.silenceRun = 0;
    this.frameCount = 0;
    this.vadState = 'silence';

    if (durationMs < MIN_SEGMENT_MS) return null;
    return { pcm, durationMs };
  }
}

function concatFrames(frames: Float32Array[]): Float32Array {
  const total = frames.reduce((acc, f) => acc + f.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const f of frames) {
    out.set(f, off);
    off += f.length;
  }
  return out;
}

/** Cut a 16 kHz mono PCM stream into non-overlapping analysis frames. */
export function framePcm(pcm: Float32Array): Float32Array[] {
  const frames: Float32Array[] = [];
  for (let off = 0; off + FRAME_SIZE <= pcm.length; off += FRAME_SIZE) {
    frames.push(pcm.subarray(off, off + FRAME_SIZE));
  }
  return frames;
}

/* ------------------------------------------------------- segment analysis */

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export function analyzeSegment(pcm: Float32Array, sampleRate = TARGET_RATE): AcousticProfile {
  const frames = framePcm(pcm);
  let rmsSum = 0;
  let rmsCount = 0;
  let peak = 0;
  let hfSum = 0;
  const minFrameRms = 0.004;

  const byLoudness: { rms: number; frame: Float32Array }[] = [];
  for (const f of frames) {
    const r = frameRms(f);
    let p = 0;
    for (let i = 0; i < f.length; i++) {
      const a = Math.abs(f[i]);
      if (a > p) p = a;
    }
    if (p > peak) peak = p;
    if (r > minFrameRms) {
      rmsSum += r;
      rmsCount++;
      hfSum += hfEnergyRatio(f, sampleRate);
    }
    if (r > 0.01) byLoudness.push({ rms: r, frame: f });
  }

  const avgRms = rmsCount > 0 ? rmsSum / rmsCount : 0;
  const dbfs = avgRms > 0 ? 20 * Math.log10(avgRms) : -100;
  const peakDbfs = peak > 0 ? 20 * Math.log10(peak) : -100;
  const hfRatio = rmsCount > 0 ? hfSum / rmsCount : 0;

  // pitch: median across the strongest voiced frames — far more stable for
  // speaker clustering than a single-frame estimate
  byLoudness.sort((a, b) => b.rms - a.rms);
  const pitches = byLoudness
    .slice(0, 5)
    .map((c) => estimatePitch(c.frame, sampleRate))
    .filter((p): p is number => p != null);
  const pitchHz = pitches.length > 0 ? Math.round(median(pitches)) : null;
  const durationMs = (pcm.length * 1000) / sampleRate;

  return {
    dbfs: Math.round(dbfs * 10) / 10,
    peakDbfs: Math.round(peakDbfs * 10) / 10,
    hfRatio: Math.round(hfRatio * 1000) / 1000,
    pitchHz: pitchHz ? Math.round(pitchHz) : null,
    distanceBand: classifyDistance(dbfs, hfRatio),
    durationMs: Math.round(durationMs),
  };
}

/* -------------------------------------------------------- speaker tracker */

/**
 * Clusters utterances into speakers by pitch proximity (~30 Hz tolerance,
 * EMA-smoothed centroids). A heuristic — adult F0 ranges overlap, so results
 * are presented as estimates and the LLM layer can refine the WHO layer.
 */
export class SpeakerRegistry {
  private speakers: SpeakerInfo[] = [];

  assign(pitchHz: number | null): SpeakerInfo {
    if (pitchHz == null) {
      // unvoiced burst: attribute to the most recent speaker if one exists
      const last = this.speakers[this.speakers.length - 1];
      if (last) {
        last.utterances++;
        return { ...last };
      }
      const s: SpeakerInfo = { id: 1, label: 'Speaker 1', pitchHz: null, utterances: 1 };
      this.speakers.push(s);
      return { ...s };
    }

    let best: SpeakerInfo | null = null;
    let bestDiff = Infinity;
    for (const s of this.speakers) {
      if (s.pitchHz == null) continue;
      const d = Math.abs(s.pitchHz - pitchHz);
      if (d < bestDiff) {
        bestDiff = d;
        best = s;
      }
    }
    if (best && bestDiff <= 30) {
      best.pitchHz = Math.round(best.pitchHz! * 0.7 + pitchHz * 0.3);
      best.utterances++;
      return { ...best };
    }

    const s: SpeakerInfo = {
      id: this.speakers.length + 1,
      label: `Speaker ${this.speakers.length + 1}`,
      pitchHz,
      utterances: 1,
    };
    this.speakers.push(s);
    return { ...s };
  }

  list(): SpeakerInfo[] {
    return this.speakers.map((s) => ({ ...s }));
  }

  /** Associate a self-introduced name with a speaker's voice. */
  rename(id: number, name: string): void {
    const s = this.speakers.find((sp) => sp.id === id);
    if (s) s.label = `${name} (Speaker ${s.id})`;
  }

  labelOf(id: number): string {
    return this.speakers.find((s) => s.id === id)?.label ?? `Speaker ${id}`;
  }

  reset(): void {
    this.speakers = [];
  }
}

/* -------------------------------------------------------- noise suppression */

/** Result of cleaning one segment. */
export interface DenoiseResult {
  pcm: Float32Array;
  attenuationDb: number; // mean gain applied to attenuated bins (negative dB)
}

/**
 * Spectral noise suppression — the "attention filter".
 *
 * While nobody is talking it keeps an EMA of the room's power spectrum
 * (fan hum, street rumble, keyboard clatter). When a speech segment is
 * finalized, each STFT frame gets a per-frequency gain: anything that looks
 * like the learned noise is attenuated, everything else is kept. A
 * minimum-statistics fallback (per-bin minimum across the segment) lets
 * uploads that start talking immediately still be cleaned.
 *
 * STFT: Hann analysis window, 50% overlap, rectangular synthesis —
 * Hann at hop N/2 overlap-adds to exactly 1, so with unity gains the
 * round-trip is transparent.
 */
export class SpectralDenoiser {
  private noisePower: Float32Array | null = null; // per-bin power, bins 0..N/2
  private learnedFrames = 0;
  private lastAttenuationDb = 0;
  /** over-subtraction factor — how aggressively learned noise is removed */
  alpha = 2.0;
  /** spectral floor — a bin is never attenuated below 10% (dodges musical noise) */
  beta = 0.1;

  get ready(): boolean {
    return this.noisePower != null && this.learnedFrames >= 4;
  }

  /** Current ambient noise floor in dBFS (time-domain equivalent via Parseval). */
  get noiseFloorDb(): number {
    if (!this.noisePower) return -100;
    const n = FRAME_SIZE;
    let s = 0;
    for (let b = 0; b < this.noisePower.length; b++) s += this.noisePower[b]!;
    // Parseval + Hann window energy: mean time-domain power ≈ (16/3)·S/N²
    const meanTimePower = (16 / 3) * (s / (n * n));
    return meanTimePower > 0 ? 10 * Math.log10(meanTimePower) : -100;
  }

  get attenuationDb(): number {
    return this.lastAttenuationDb;
  }

  reset(): void {
    this.noisePower = null;
    this.learnedFrames = 0;
    this.lastAttenuationDb = 0;
  }

  /** Learn the ambient spectrum from one silent frame (raw PCM, 16 kHz, 2048 samples). */
  learn(frame: Float32Array): void {
    if (frame.length !== FRAME_SIZE) return;
    const re = new Float32Array(FRAME_SIZE);
    const im = new Float32Array(FRAME_SIZE);
    for (let i = 0; i < FRAME_SIZE; i++) re[i] = frame[i] * HANN_2048[i];
    fftInPlace(re, im);
    const half = FRAME_SIZE >> 1;
    if (!this.noisePower) this.noisePower = new Float32Array(half + 1);
    for (let b = 0; b <= half; b++) {
      const p = re[b] * re[b] + im[b] * im[b];
      this.noisePower[b] = this.noisePower[b]! * 0.88 + p * 0.12;
    }
    this.learnedFrames++;
  }

  /**
   * Clean a finalized speech segment with STFT spectral subtraction.
   * The input stays untouched; analysis (distance, pitch) keeps using the raw
   * signal so absolute loudness cues survive.
   */
  denoiseSegment(pcm: Float32Array): DenoiseResult {
    const N = FRAME_SIZE;
    const HOP = N >> 1;
    const half = N >> 1;
    if (pcm.length < N) return { pcm, attenuationDb: 0 };

    const totalFrames = Math.floor(pcm.length / HOP) + 1;

    // ---- noise estimate: streaming EMA, else per-bin minimum statistics ----
    let noise: Float32Array;
    if (this.ready) {
      noise = new Float32Array(half + 1);
      for (let b = 0; b <= half; b++) noise[b] = this.noisePower![b]!;
    } else {
      noise = new Float32Array(half + 1).fill(Infinity);
      const re = new Float32Array(N);
      const im = new Float32Array(N);
      for (let f = 0; f < totalFrames; f++) {
        const pos = f * HOP;
        re.fill(0);
        im.fill(0);
        for (let i = 0; i < N && pos + i < pcm.length; i++) re[i] = pcm[pos + i]! * HANN_2048[i]!;
        fftInPlace(re, im);
        for (let b = 0; b <= half; b++) {
          const p = re[b]! * re[b]! + im[b]! * im[b]!;
          if (p < noise[b]!) noise[b] = p;
        }
      }
      for (let b = 0; b <= half; b++) if (!Number.isFinite(noise[b]!)) noise[b] = 0;
    }

    // frequency-smooth the estimate (3-tap) to avoid isolated notch artifacts
    const smooth = new Float32Array(half + 1);
    for (let b = 0; b <= half; b++) {
      const a = noise[Math.max(0, b - 1)]!;
      const c = noise[Math.min(half, b + 1)]!;
      smooth[b] = 0.25 * a + 0.5 * noise[b]! + 0.25 * c;
    }
    noise = smooth;

    // ---- spectral subtraction, frame by frame ----
    const outLen = (totalFrames - 1) * HOP + N;
    const out = new Float32Array(outLen);
    const gains = new Float32Array(half + 1).fill(1); // temporal smoothing of gains
    let gainSumDb = 0;
    let gainCount = 0;

    const re = new Float32Array(N);
    const im = new Float32Array(N);
    for (let f = 0; f < totalFrames; f++) {
      const pos = f * HOP;
      re.fill(0);
      im.fill(0);
      for (let i = 0; i < N && pos + i < pcm.length; i++) re[i] = pcm[pos + i]! * HANN_2048[i]!;
      fftInPlace(re, im);

      for (let b = 0; b <= half; b++) {
        const mag = Math.hypot(re[b]!, im[b]!);
        const noiseMag = Math.sqrt(noise[b]!) * this.alpha;
        let g = mag > 1e-9 ? (mag - noiseMag) / mag : 1;
        if (g < this.beta) g = this.beta;
        if (g > 1) g = 1;
        gains[b] = gains[b]! * 0.55 + g * 0.45;
        const gg = gains[b]!;
        if (gg < 0.999) {
          gainSumDb += 20 * Math.log10(Math.max(1e-4, gg));
          gainCount++;
        }
        // apply to bin b and its mirror N-b so the spectrum stays conjugate-symmetric
        re[b] *= gg;
        im[b] *= gg;
        if (b > 0 && b < half) {
          re[N - b] *= gg;
          im[N - b] *= gg;
        }
      }

      ifftInPlace(re, im);
      // overlap-add: Hann at 50% overlap sums to exactly 1
      for (let i = 0; i < N && pos + i < outLen; i++) out[pos + i] += re[i]!;
    }

    this.lastAttenuationDb = gainCount > 0 ? Math.round((gainSumDb / gainCount) * 10) / 10 : 0;
    return { pcm: out.slice(0, pcm.length), attenuationDb: this.lastAttenuationDb };
  }
}

/* ----------------------------------------------------------- loudness prep */

/**
 * Scale a cleaned segment to a healthy listening level for the audio model.
 * Quiet/distant speech gets boosted so the ASR hears every word; already-loud
 * audio passes through untouched (never make things worse).
 */
export function normalizeLoudness(
  pcm: Float32Array,
  targetDbfs = -18,
  peakCeiling = 0.97,
): Float32Array {
  const rms = frameRms(pcm);
  if (rms <= 1e-5) return pcm;
  let gain = 10 ** (targetDbfs / 20) / rms;
  let peak = 0;
  for (let i = 0; i < pcm.length; i++) {
    const a = Math.abs(pcm[i]!);
    if (a > peak) peak = a;
  }
  if (peak > 0 && peak * gain > peakCeiling) gain = peakCeiling / peak;
  if (gain <= 1.0001) return pcm;
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i]! * gain;
  return out;
}

/* ------------------------------------------------------------ WAV encoding */

export function encodeWav(pcm: Float32Array, sampleRate = TARGET_RATE): ArrayBuffer {
  const dataBytes = pcm.length * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, dataBytes, true);
  let o = 44;
  for (let i = 0; i < pcm.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

export function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  const CHUNK = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    const slice = bytes.subarray(i, Math.min(i + CHUNK, bytes.length));
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}
