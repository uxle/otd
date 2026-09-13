/* tslint:disable */
/* eslint-disable */
/**
 * Engine version — surfaced in the UI's "Rust WASM" badge.
 */
export function version(): string;
/**
 * Catmull-Rom cubic resample. `resample(chunk, 48000, 16000)`.
 */
export function resample(input: Float32Array, from_rate: number, to_rate: number): Float32Array;
/**
 * Normalize a denoised segment to `target_dbfs` and encode 16-bit WAV.
 * Used for the second, louder listen (−12 dBFS retry).
 */
export function render_wav(pcm: Float32Array, target_dbfs: number, peak_ceiling: number): Uint8Array;
/**
 * The stateful perception engine.
 */
export class PerceptEngine {
  free(): void;
  /**
   * Feed one 2048-sample frame at 16 kHz. Returns live metrics JSON:
   * `{ dbfs, pitchHz, hfRatio, distanceBand, vad, noiseFloorDb,
   *    segmentReady, elapsedUs }`
   */
  feed_frame(frame: Float32Array): any;
  /**
   * Pull the pending raw segment PCM, if a turn just closed.
   * Returns `undefined` when nothing is pending.
   */
  take_segment(): Float32Array | undefined;
  /**
   * Current speaker roster for the UI.
   */
  list_speakers(): any;
  /**
   * Associate a self-introduced name with a speaker's voice.
   */
  rename_speaker(id: number, name: string): void;
  /**
   * Analyze a finished utterance. Returns an object:
   * `{ acoustics, speaker, denoiseDb, cleanPcm, wav, spectrogram, elapsedUs }`
   * where `wav` is a Uint8Array, `cleanPcm` a Float32Array and
   * `spectrogram` = `{ cols, rows, data: Float32Array }`.
   */
  analyze_segment(pcm: Float32Array, target_dbfs: number): any;
  constructor();
  /**
   * Finalize any in-progress turn (on stop / end of file).
   * Returns `true` when a segment became available via `take_segment`.
   */
  flush(): boolean;
  /**
   * Clear all session state (segmenter, speakers, noise profile).
   */
  reset(): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_perceptengine_free: (a: number, b: number) => void;
  readonly perceptengine_analyze_segment: (a: number, b: number, c: number, d: number) => any;
  readonly perceptengine_feed_frame: (a: number, b: number, c: number) => any;
  readonly perceptengine_flush: (a: number) => number;
  readonly perceptengine_list_speakers: (a: number) => any;
  readonly perceptengine_new: () => number;
  readonly perceptengine_rename_speaker: (a: number, b: number, c: number, d: number) => void;
  readonly perceptengine_reset: (a: number) => void;
  readonly perceptengine_take_segment: (a: number) => [number, number];
  readonly render_wav: (a: number, b: number, c: number, d: number) => [number, number];
  readonly resample: (a: number, b: number, c: number, d: number) => [number, number];
  readonly version: () => [number, number];
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_3: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
