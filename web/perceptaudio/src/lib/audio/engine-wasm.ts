/**
 * Unified engine facade for PerceptAudio 2.0.
 *
 * Primary backend: the Rust DSP engine compiled to WebAssembly
 * (`public/wasm/percept_engine_bg.wasm`) — YIN pitch, twiddle-table FFT,
 * Catmull-Rom resampling, flatness-gated VAD, Wiener-style spectral
 * denoising, two-feature speaker clustering and spectrogram rendering.
 *
 * Fallback backend: the original TypeScript DSP (`./dsp`) — used only if the
 * WASM module fails to load (very old browsers, missing files). Both backends
 * implement the exact same interface, so the analyzer hook is backend-blind.
 */

import {
  type AcousticProfile,
  type DistanceBand,
  type SpeakerInfo,
  FRAME_SIZE,
  SpeakerRegistry,
  SpectralDenoiser,
  TARGET_RATE,
  VoiceSegmenter,
  analyzeSegment,
  encodeWav,
  estimatePitch,
  fftInPlace,
  frameRms,
  hfEnergyRatio,
  normalizeLoudness,
  resampleLinear,
} from './dsp';

export type { AcousticProfile, DistanceBand, SpeakerInfo };

export interface SpectroData {
  cols: number;
  rows: number;
  /** column-major: data[col * rows + row], normalized 0..1 */
  data: Float32Array;
}

export interface AnalyzeOutput {
  acoustics: AcousticProfile;
  speaker: SpeakerInfo;
  denoiseDb: number;
  /** denoised (unnormalized) PCM — for the louder second listen */
  cleanPcm: Float32Array;
  /** normalized 16-bit WAV, ready for the ASR */
  wav: Uint8Array;
  spectrogram: SpectroData;
  /** Rust (or JS) engine processing time */
  elapsedUs: number;
}

export interface FrameMetrics {
  dbfs: number;
  pitchHz: number | null;
  hfRatio: number;
  distanceBand: DistanceBand;
  vad: 'silence' | 'speech';
  noiseFloorDb: number;
  segmentReady: boolean;
  elapsedUs: number;
}

export interface EngineBackend {
  kind: 'rust' | 'ts';
  version: string;
  feedFrame(frame: Float32Array): FrameMetrics;
  takeSegment(): Float32Array | null;
  /** finalize any in-progress turn; true when a segment is ready to take */
  flush(): boolean;
  analyzeSegment(pcm: Float32Array, targetDbfs: number): AnalyzeOutput;
  renderWav(pcm: Float32Array, targetDbfs: number): Uint8Array;
  resample(input: Float32Array, fromRate: number, toRate: number): Float32Array;
  renameSpeaker(id: number, name: string): void;
  listSpeakers(): SpeakerInfo[];
  reset(): void;
}

export const ENGINE_VERSION = '2.0.0';
const WASM_URL = '/wasm/percept_engine.js';

/* --------------------------------------------------------------- wasm load */

interface WasmModule {
  default: (input?: unknown) => Promise<unknown>;
  PerceptEngine: new () => WasmEngine;
  resample: (input: Float32Array, from: number, to: number) => Float32Array;
  render_wav: (pcm: Float32Array, target: number, ceiling: number) => Uint8Array;
  version: () => string;
}

interface WasmEngine {
  feed_frame(frame: Float32Array): FrameMetrics;
  take_segment(): Float32Array | undefined;
  flush(): boolean;
  analyze_segment(pcm: Float32Array, targetDbfs: number): AnalyzeOutput;
  rename_speaker(id: number, name: string): void;
  list_speakers(): SpeakerInfo[];
  reset(): void;
}

let wasmPromise: Promise<WasmModule> | null = null;

async function loadWasm(): Promise<WasmModule> {
  if (!wasmPromise) {
    wasmPromise = (async () => {
      // loaded straight from /public — bypasses the bundler on purpose
      const mod = (await import(
        /* webpackIgnore: true */ /* turbopackIgnore: true */ WASM_URL
      )) as unknown as WasmModule;
      await mod.default(WASM_URL.replace(/\.js$/, '_bg.wasm'));
      return mod;
    })();
    wasmPromise.catch(() => {
      // allow a later retry after a failed load
      wasmPromise = null;
    });
  }
  return wasmPromise;
}

/* --------------------------------------------------------- rust backend */

class RustBackend implements EngineBackend {
  kind = 'rust' as const;
  version = ENGINE_VERSION;
  constructor(private eng: WasmEngine, private mod: WasmModule) {}

  feedFrame(frame: Float32Array): FrameMetrics {
    return this.eng.feed_frame(frame);
  }
  takeSegment(): Float32Array | null {
    return this.eng.take_segment() ?? null;
  }
  flush(): boolean {
    return this.eng.flush();
  }
  analyzeSegment(pcm: Float32Array, targetDbfs: number): AnalyzeOutput {
    return this.eng.analyze_segment(pcm, targetDbfs);
  }
  renderWav(pcm: Float32Array, targetDbfs: number): Uint8Array {
    return this.mod.render_wav(pcm, targetDbfs, 0.97);
  }
  resample(input: Float32Array, fromRate: number, toRate: number): Float32Array {
    return this.mod.resample(input, fromRate, toRate);
  }
  renameSpeaker(id: number, name: string): void {
    this.eng.rename_speaker(id, name);
  }
  listSpeakers(): SpeakerInfo[] {
    return this.eng.list_speakers() ?? [];
  }
  reset(): void {
    this.eng.reset();
  }
}

/* ------------------------------------------------------------ ts backend */

/** Log-band spectrogram mirroring the Rust engine's output shape. */
function tsSpectrogram(pcm: Float32Array, rows = 48, maxCols = 96): SpectroData {
  const N = 1024;
  const usable = Math.max(pcm.length, N);
  const cols = Math.min(maxCols, Math.max(1, Math.floor(usable / 1024)));
  const hop = Math.floor((usable - N) / cols) + 1;
  const minHz = 80;
  const maxHz = 7500;
  const minBin = Math.max(1, Math.floor(minHz / (TARGET_RATE / N)));
  const maxBin = Math.min(N / 2, Math.ceil(maxHz / (TARGET_RATE / N)));

  const edges = new Float32Array(rows + 1);
  for (let r = 0; r <= rows; r++) {
    edges[r] = minBin * Math.pow(maxBin / minBin, r / rows);
  }

  const raw = new Float32Array(cols * rows);
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  for (let c = 0; c < cols; c++) {
    const pos = Math.min(c * hop, Math.max(0, pcm.length - N));
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < N; i++) {
      if (pos + i < pcm.length) re[i] = pcm[pos + i];
    }
    // Hann + FFT
    for (let i = 0; i < N; i++) re[i] *= 0.5 * (1 - Math.cos((2 * Math.PI * i) / N));
    fftInPlace(re, im);
    for (let r = 0; r < rows; r++) {
      const b0 = Math.ceil(edges[r]);
      const b1 = Math.max(b0 + 1, Math.floor(edges[r + 1]));
      let sum = 0;
      for (let b = Math.min(b0, maxBin); b <= Math.min(b1, maxBin); b++) {
        sum += re[b] * re[b] + im[b] * im[b];
      }
      raw[c * rows + r] = sum;
    }
  }
  let maxDb = -Infinity;
  for (let i = 0; i < raw.length; i++) {
    raw[i] = 10 * Math.log10(Math.max(1e-12, raw[i]));
    if (raw[i] > maxDb) maxDb = raw[i];
  }
  for (let i = 0; i < raw.length; i++) {
    raw[i] = Math.min(1, Math.max(0, (raw[i] - maxDb + 70) / 70));
  }
  return { cols, rows, data: raw };
}

class TsBackend implements EngineBackend {
  kind = 'ts' as const;
  version = ENGINE_VERSION;
  private segmenter = new VoiceSegmenter();
  private registry = new SpeakerRegistry();
  private denoiser = new SpectralDenoiser();
  private smoothDbfs = -100;
  private smoothHf = 0;
  private smoothPitch: number | null = null;
  private pending: Float32Array | null = null;

  feedFrame(frame: Float32Array): FrameMetrics {
    const t0 = performance.now();
    const rms = frameRms(frame);
    let hfRaw = 0;
    if (rms > 0.008) {
      hfRaw = hfEnergyRatio(frame, TARGET_RATE);
      const p = estimatePitch(frame, TARGET_RATE);
      if (p) {
        this.smoothPitch = this.smoothPitch
          ? Math.round(this.smoothPitch * 0.6 + p * 0.4)
          : Math.round(p);
      }
    }
    this.smoothDbfs = this.smoothDbfs * 0.75 + (rms > 0 ? 20 * Math.log10(rms) : -100) * 0.25;
    this.smoothHf = this.smoothHf * 0.8 + hfRaw * 0.2;

    if (this.segmenter.vadState === 'silence' && rms < 0.04) {
      this.denoiser.learn(frame);
    }
    const seg = this.segmenter.feed(frame);
    if (seg) this.pending = seg.pcm;

    return {
      dbfs: Math.round(this.smoothDbfs * 10) / 10,
      pitchHz: this.smoothPitch,
      hfRatio: Math.round(this.smoothHf * 1000) / 1000,
      distanceBand: classifyDistanceTs(this.smoothDbfs, this.smoothHf),
      vad: this.segmenter.vadState,
      noiseFloorDb: Math.round(this.denoiser.noiseFloorDb * 10) / 10,
      segmentReady: this.pending != null,
      elapsedUs: (performance.now() - t0) * 1000,
    };
  }

  takeSegment(): Float32Array | null {
    const p = this.pending;
    this.pending = null;
    return p;
  }

  flush(): boolean {
    const seg = this.segmenter.flush();
    if (seg) {
      this.pending = seg.pcm;
      return true;
    }
    return false;
  }

  analyzeSegment(pcm: Float32Array, targetDbfs: number): AnalyzeOutput {
    const t0 = performance.now();
    const acoustics = analyzeSegment(pcm);
    const speaker = this.registry.assign(acoustics.pitchHz);
    const dn = this.denoiser.denoiseSegment(pcm);
    const normalized = normalizeLoudness(dn.pcm, targetDbfs);
    const wav = encodeWav(normalized);
    const spec = tsSpectrogram(dn.pcm);
    return {
      acoustics,
      speaker,
      denoiseDb: dn.attenuationDb,
      cleanPcm: dn.pcm,
      wav: new Uint8Array(wav),
      spectrogram: spec,
      elapsedUs: (performance.now() - t0) * 1000,
    };
  }

  renderWav(pcm: Float32Array, targetDbfs: number): Uint8Array {
    return new Uint8Array(encodeWav(normalizeLoudness(pcm, targetDbfs)));
  }

  resample(input: Float32Array, fromRate: number, toRate: number): Float32Array {
    return resampleLinear(input, fromRate, toRate);
  }

  renameSpeaker(id: number, name: string): void {
    this.registry.rename(id, name);
  }

  listSpeakers(): SpeakerInfo[] {
    return this.registry.list();
  }

  reset(): void {
    this.segmenter.reset();
    this.registry.reset();
    this.denoiser.reset();
    this.pending = null;
    this.smoothDbfs = -100;
    this.smoothHf = 0;
    this.smoothPitch = null;
  }
}

function classifyDistanceTs(dbfs: number, hf: number): DistanceBand {
  if (dbfs >= -14) return hf >= 0.03 ? 'very-close' : 'close';
  if (dbfs >= -22) return hf >= 0.12 ? 'close' : 'normal';
  if (dbfs >= -30) return hf >= 0.05 ? 'normal' : 'far';
  if (dbfs >= -38) return 'far';
  return 'very-far';
}

/* -------------------------------------------------------------- factory */

/**
 * Create the best available engine backend.
 * Resolves to the Rust/WASM engine, or the TypeScript fallback when the
 * module cannot load. Never rejects.
 */
export async function createEngine(): Promise<EngineBackend> {
  try {
    if (typeof WebAssembly === 'undefined') throw new Error('WebAssembly unavailable');
    const mod = await loadWasm();
    return new RustBackend(new mod.PerceptEngine(), mod);
  } catch (err) {
    console.warn(
      '[perceptaudio] Rust/WASM engine unavailable, falling back to TypeScript DSP:',
      err instanceof Error ? err.message : err,
    );
    return new TsBackend();
  }
}

/** Chunked ArrayBuffer→base64 (safe for multi-MB WAV payloads). */
export function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    const slice = bytes.subarray(i, Math.min(i + CHUNK, bytes.length));
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

export { FRAME_SIZE, TARGET_RATE };
