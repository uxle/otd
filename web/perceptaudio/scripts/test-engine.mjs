// Node smoke test for the Rust→WASM PerceptAudio engine.
import { readFileSync } from 'node:fs';
import init from '/home/z/my-project/public/wasm/percept_engine.js';

const wasmBytes = readFileSync('/home/z/my-project/public/wasm/percept_engine_bg.wasm');
await init(wasmBytes);

const mod = await import('/home/z/my-project/public/wasm/percept_engine.js');
const { PerceptEngine, resample, render_wav, version } = mod;

console.log('version:', version());

// ---- resample: 48k -> 16k length check
const chunk = new Float32Array(4800).map((_, i) => Math.sin((2 * Math.PI * 220 * i) / 48000));
const rs = resample(chunk, 48000, 16000);
console.log('resample 4800@48k ->', rs.length, 'samples @16k (expect ~1600)');

// ---- synthetic "speech": voiced tones + silence around them
const SR = 16000;
const pcm = [];
const tone = (f0, ms) => {
  const n = Math.round((SR * ms) / 1000);
  let phase = 0;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    // gentle vibrato via incremental phase (no phase-modulation blowup)
    const inst = f0 * (1 + 0.008 * Math.sin(2 * Math.PI * 5 * t));
    phase += (2 * Math.PI * inst) / SR;
    const env = Math.min(1, t * 20) * Math.min(1, (n / SR - t) * 10);
    pcm.push(0.35 * env * Math.sin(phase) * (1 + 0.25 * Math.sin(2 * Math.PI * 2 * f0 * t)));
  }
};
const silence = (ms) => { for (let i = 0; i < Math.round((SR * ms) / 1000); i++) pcm.push((Math.random() - 0.5) * 0.002); };

silence(700); tone(150, 1500); silence(1400); tone(230, 1200); silence(700);

// ---- run through the engine frame by frame
const eng = new PerceptEngine();
const FRAME = 2048;
let segments = 0;
let lastMetrics = null;
let lastAnalyzed = null;
const frames = [];
for (let off = 0; off + FRAME <= pcm.length; off += FRAME) frames.push(pcm.slice(off, off + FRAME));
const rem = pcm.length % FRAME;
if (rem > 0) {
  const tail = pcm.slice(pcm.length - rem);
  while (tail.length < FRAME) tail.push(0);
  frames.push(tail);
}
for (const f of frames) {
  const m = eng.feed_frame(new Float32Array(f));
  lastMetrics = m;
  if (m.segmentReady) {
    const seg = eng.take_segment();
    const an = eng.analyze_segment(seg, -18);
    segments++;
    lastAnalyzed = an;
    console.log(`\nsegment #${segments}:`);
    console.log('  acoustics:', JSON.stringify(an.acoustics));
    console.log('  speaker:', JSON.stringify(an.speaker));
    console.log('  denoiseDb:', an.denoiseDb.toFixed(1), '| wav bytes:', an.wav.length, '| cleanPcm:', an.cleanPcm.length);
    console.log('  spectrogram:', an.spectrogram.cols, 'x', an.spectrogram.rows, '| rust elapsed:', an.elapsedUs.toFixed(0), 'us');
    const hdr = Buffer.from(an.wav.slice(0, 44));
    console.log('  wav RIFF:', hdr.slice(0, 4).toString(), '| WAVE:', hdr.slice(8, 12).toString(), '| rate:', hdr.readUInt32LE(24));
  }
}
// ---- flush any in-progress turn (stop / end-of-file path)
if (eng.flush()) {
  const seg = eng.take_segment();
  const an = eng.analyze_segment(seg, -18);
  segments++;
  lastAnalyzed = an;
  console.log(`\nsegment #${segments} (flushed):`);
  console.log('  acoustics:', JSON.stringify(an.acoustics));
  console.log('  speaker:', JSON.stringify(an.speaker));
  console.log('  rust elapsed:', an.elapsedUs.toFixed(0), 'us');
}
console.log('\nlast metrics:', JSON.stringify(lastMetrics));
console.log('segments found:', segments, '(expect 2)');

// ---- second-listen path
const louder = render_wav(new Float32Array(lastAnalyzed.cleanPcm), -12, 0.97);
console.log('render_wav @-12dBFS bytes:', louder.length);

// ---- speaker rename + reset
eng.rename_speaker(1, 'Maria');
eng.reset();
console.log('reset OK');

if (segments !== 2) {
  console.error('FAIL: expected 2 segments, got', segments);
  process.exit(1);
}
console.log('\nALL ENGINE CHECKS PASSED');
