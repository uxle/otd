# PerceptAudio 2.0 — Rust-Powered Voice-Perception Analyzer

A Next.js web app that listens like a person: it filters background noise, tracks who is
speaking and from how far, transcribes what they say, reads emotion and intent — and
answers out loud (Aria, a female voice) when you talk to her.

**What changed in 2.0:** the entire DSP pipeline now runs in **Rust compiled to
WebAssembly** — turn-taking VAD, YIN pitch tracking, distance bands, speaker clustering,
spectral noise suppression, resampling, WAV encoding and spectrogram rendering. The
browser keeps the mic and the paintbrush; Rust does the hearing.

Every utterance is broken into five perception layers:

| Layer | Question | How it is computed (v2.0 — Rust/WASM) |
|-------|----------|----------------------------------------|
| **WHO** | Who is speaking? | **YIN** pitch (F0) + spectral-centroid timbre, two-feature EMA centroids — each stable voice gets its own speaker label; names are learned when stated ("it's Maria") |
| **WHERE** | How far away are they? | Loudness (dBFS) + high-frequency energy ratio → distance bands: very-close (<0.5 m) / close / normal / far / very-far (6 m+) |
| **WHAT** | What was said? | Rust VAD segments speech turns → **Wiener-style spectral subtraction** strips learned background noise → loudness-normalized 16 kHz WAV → `/api/transcribe` (ASR audio model) |
| **HOW** | Emotional state? | `/api/analyze` (LLM) reads the transcript + acoustic measurements and returns emotion + evidence cue |
| **WHY** | Intent? | Same LLM call returns intent, whether the utterance is addressed *to the assistant*, and a situational awareness note |

## The Rust DSP engine (`engine-rust/`)

| Module | What it does | Upgrade over v1 (TypeScript) |
|--------|--------------|------------------------------|
| `fft.rs` | Radix-2 FFT, Hann window | Precomputed twiddle tables + bit-reversal permutation (v1 recomputed `cos/sin` per butterfly) |
| `pitch.rs` | YIN F0 estimator | v1 peak-picked the raw autocorrelation and suffered octave errors; YIN + parabolic interpolation tracks the true fundamental (measured: 151 Hz vs 75 Hz on the same 150 Hz tone) |
| `vad.rs` | Turn-taking segmenter | New **spectral-flatness start gate** — loud, flat broadband clatter no longer opens a turn |
| `denoise.rs` | Spectral subtraction | **Wiener-style gain** + **SNR-adaptive over-subtraction α (1.2–2.5)** — fewer musical-noise artifacts, gentler on clean speech, harder on noise |
| `speaker.rs` | Voice clustering | Two-feature centroids (pitch + timbre): same-pitch voices with different timbres no longer merge |
| `resample.rs` | 48/44.1 kHz → 16 kHz | Catmull-Rom cubic interpolation instead of linear (less aliasing) |
| `spectrogram.rs` | NEW — voiceprints | Log-band (48 bands, 80 Hz–7.5 kHz) spectrogram per utterance, painted under every card |
| `wav.rs` / `dsp.rs` | WAV encode, acoustics | Identical semantics to v1, measured in microseconds |

The engine is a single 105 KB `percept_engine_bg.wasm` in `public/wasm/`, loaded lazily
from the client. If WebAssembly is unavailable, the app transparently falls back to the
original TypeScript DSP (`src/lib/audio/dsp.ts`) — same pipeline, same behavior, and the
header badge switches from “Rust WASM engine” to “TS fallback” so you always know which
ear you are using.

## How it hears — in human terms

1. **Eardrum** — the mic. Loudness + treble crispness = distance.
2. **Attention** — while nobody talks, Rust learns the room's hum (fan, traffic, keyboard)
   and subtracts it spectrally; only voices reach the model.
3. **Turn-taking** — one speaker at a time; each voice keeps a pitch+timbre fingerprint.
4. **Understanding** — clean words go to the ASR at a healthy listening level
   (RMS normalized to −18 dBFS). If words come back thin, it **listens a second time,
   louder** (−12 dBFS) instead of guessing.
5. **Reply** — when an utterance is addressed to the assistant, Aria speaks the awareness
   note aloud via `/api/speak` (TTS). Side chatter stays silent — privacy by behavior.

---

## Requirements

- **Node.js 20+** or **Bun 1.3+** (app) · **Rust 1.85+** with the
  `wasm32-unknown-unknown` target + `wasm-bindgen-cli` (only to rebuild the engine)
- A modern browser (Chrome / Edge / Firefox / Safari)
- Microphone access requires the page to be served from `localhost` or **HTTPS**

## Install & run

```bash
# with bun (recommended, matches the lockfile)
bun install
bun run dev

# or with npm
npm install
npm run dev
```

Open <http://localhost:3000>.

### Rebuilding the Rust engine (optional)

The compiled WASM is committed under `public/wasm/`. To rebuild it after changing
`engine-rust/`:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.100   # must match the crate pin
bun run engine:build                               # cargo build + wasm-bindgen
bun run engine:test                                # Node smoke test (synthetic voices)
```

## Try it in 10 seconds (no mic needed)

1. Click **Analyze audio file** — or **drag & drop** a file onto the perception feed —
   and pick `sample-audio/test-speech.wav`.
2. The file is decoded, resampled in Rust, noise-filtered, VAD-segmented and pushed
   through the full pipeline — you will see utterance cards with speaker, distance,
   transcript, emotion, intent, a spectrogram voiceprint, and the Rust processing time.

With a mic: press **Start listening**, speak, pause ~1 s to close a turn. Talk *to* the
assistant ("hey, can you…") and Aria answers out loud. The speaker button next to the
mic mutes/unmutes her.

## Project structure

```
engine-rust/                       # Rust DSP engine (compiles to WASM)
├── Cargo.toml                     # pinned to wasm-bindgen 0.2.100
└── src/
    ├── lib.rs                     # wasm-bindgen API (PerceptEngine, resample, render_wav)
    ├── fft.rs                     # radix-2 FFT + Hann (precomputed tables)
    ├── pitch.rs                   # YIN F0 estimator
    ├── vad.rs                     # turn-taking segmenter + flatness gate
    ├── denoise.rs                 # Wiener-style spectral subtraction
    ├── speaker.rs                 # pitch+timbre voice clustering
    ├── resample.rs                # Catmull-Rom cubic SRC
    ├── spectrogram.rs             # log-band voiceprints
    ├── dsp.rs                     # RMS/peak/HF ratio, distance bands, normalization
    ├── wav.rs                     # 16-bit PCM WAV encoder
    └── engine.rs                  # state machine tying it all together

public/wasm/                       # generated: percept_engine.js + _bg.wasm (do not edit)

src/
├── app/
│   ├── page.tsx                  # studio UI: live meters, session insights, drag-drop feed
│   └── api/
│       ├── transcribe/route.ts   # ASR — audio_base64 → text        (z-ai-web-dev-sdk)
│       ├── analyze/route.ts      # perception — transcript+acoustics → HOW/WHY JSON
│       └── speak/route.ts        # TTS — text → WAV, Aria's voice   (z-ai-web-dev-sdk)
├── components/audio/
│   ├── visualizer.tsx            # wave / spectrum / scrolling spectrogram views
│   └── utterance-card.tsx        # WHO/WHERE/WHAT/HOW/WHY card + voiceprint + rust timing
├── hooks/
│   └── use-audio-analyzer.ts     # mic capture, engine pump, second-listen retry, TTS queue
└── lib/audio/
    ├── engine-wasm.ts            # engine facade: Rust/WASM primary, TS fallback
    └── dsp.ts                    # TypeScript fallback DSP (also re-exports shared types)
```

## The JS↔Rust API

```ts
const engine = new PerceptEngine()          // wasm-bindgen constructor
engine.feed_frame(Float32Array(2048))       // → { dbfs, pitchHz, hfRatio, distanceBand,
                                            //    vad, noiseFloorDb, segmentReady, elapsedUs }
engine.take_segment()                       // → Float32Array | undefined (raw 16 kHz PCM)
engine.analyze_segment(pcm, -18)            // → { acoustics, speaker, denoiseDb, cleanPcm,
                                            //    wav: Uint8Array, spectrogram, elapsedUs }
engine.render_wav(pcm, -12, 0.97)           // → Uint8Array (the second, louder listen)
engine.rename_speaker(id, "Maria"); engine.list_speakers(); engine.reset()
resample(chunk, 48000, 16000)               // → Float32Array (Catmull-Rom)
```

## Model provider note

`/api/transcribe`, `/api/analyze` and `/api/speak` call the `z-ai-web-dev-sdk` package
(ASR + chat + TTS). If that SDK is unavailable or unconfigured in your environment, those
endpoints return errors — but the client-side Rust DSP layer (waveform, VAD, noise
filter, distance estimate, speaker clustering, spectrograms) keeps working, and each
route file is a single self-contained function that is easy to swap for another provider.

## Honest limitations (by design, labeled in the UI)

- **Distance is a heuristic** — dBFS depends on mic gain and hardware; the app disables
  browser AGC/noise-suppression where possible and adapts its noise floor, but treat the
  distance band as an estimate, not a measurement.
- **Speaker identity is pitch+timbre clustering** — one voice with extreme prosody swings
  can split; two similar voices can still merge, though the timbre axis makes that rarer.
- **Noise suppression is spectral subtraction** — it removes steady background noise
  (hum, hiss, fans) very well; transient noises (door slams) mostly pass through, which
  is the safe direction for speech.
- **No guessing on unclear audio** — low-confidence turns surface as "no clear speech"
  (after a second, louder listen) instead of a fabricated transcript.

## Database (optional)

The Prisma/SQLite setup is scaffolded but unused by the app features. `.env` ships with a
portable relative `DATABASE_URL`. To activate the client: `bun run db:push`.

---

Built with Next.js 16 (App Router) · TypeScript · **Rust → WebAssembly (wasm-bindgen)** ·
Tailwind CSS 4 · shadcn/ui · Framer Motion.
