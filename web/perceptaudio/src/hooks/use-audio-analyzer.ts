'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  type EngineBackend,
  type SpectroData,
  createEngine,
  bytesToBase64,
  FRAME_SIZE,
} from '@/lib/audio/engine-wasm';
import {
  type AcousticProfile,
  type DistanceBand,
  type SpeakerInfo,
  DISTANCE_META,
  frameRms,
  mixdownToMono,
} from '@/lib/audio/dsp';

export interface PerceptionResult {
  emotion: string;
  emotionCue: string;
  intent: string;
  addressedToAssistant: boolean;
  speakerName: string | null;
  awarenessNote: string;
  summary: string;
}

export interface Utterance {
  id: string;
  seq: number;
  startedAt: number;
  durationMs: number;
  speaker: SpeakerInfo;
  acoustics: AcousticProfile;
  transcript: string | null;
  perception: PerceptionResult | null;
  status: 'transcribing' | 'analyzing' | 'done' | 'no-speech' | 'error';
  errorMessage?: string;
  source: 'live' | 'upload';
  /** mean dB cut the noise filter applied to this segment */
  denoiseDb?: number;
  /** true when the model got a second, louder listen */
  retried?: boolean;
  /** log-band spectrogram rendered by the engine */
  spectrogram?: SpectroData;
  /** engine processing time (Rust WASM) in microseconds */
  engineUs?: number;
}

export type EngineStatus = 'idle' | 'starting' | 'running' | 'error';
export type EngineKind = 'loading' | 'rust' | 'ts';

export interface LiveMetrics {
  dbfs: number;
  pitchHz: number | null;
  hfRatio: number;
  distanceBand: DistanceBand;
  vad: 'silence' | 'speech';
  noiseFloorDb: number;
}

const MAX_UTTERANCES = 80;
const IDLE_METRICS: LiveMetrics = {
  dbfs: -100,
  pitchHz: null,
  hfRatio: 0,
  distanceBand: 'far',
  vad: 'silence',
  noiseFloorDb: -100,
};

/**
 * Real-time voice perception engine.
 *
 * Live mic -> Rust/WASM VAD segmentation -> Rust/WASM acoustics + denoising ->
 * /api/transcribe (ASR) -> /api/analyze (LLM perception layers).
 * File uploads share the exact same pipeline.
 */
export function useAudioAnalyzer() {
  const [status, setStatus] = useState<EngineStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [speakers, setSpeakers] = useState<SpeakerInfo[]>([]);
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics>(IDLE_METRICS);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [engineKind, setEngineKind] = useState<EngineKind>('loading');
  /** median engine processing time of recent utterances, ms */
  const [engineTimingMs, setEngineTimingMs] = useState<number | null>(null);

  const engineRef = useRef<EngineBackend | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const pendingRef = useRef<Float32Array>(new Float32Array(0));
  const seqRef = useRef(0);
  const utterancesRef = useRef<Utterance[]>([]);
  const metricsRef = useRef<LiveMetrics>({ ...IDLE_METRICS });
  const metricsTimerRef = useRef<number | null>(null);
  const aliveRef = useRef(false);
  // serializes model calls so bursts of segments never trip rate limits
  const pipelineRef = useRef<Promise<unknown>>(Promise.resolve());
  const engineTimingsRef = useRef<number[]>([]);

  // Aria's voice — TTS playback, serialized so replies never talk over each other
  const [voiceOn, setVoiceOn] = useState(true);
  const voiceOnRef = useRef(true);
  const speakQueueRef = useRef<Promise<void>>(Promise.resolve());
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    voiceOnRef.current = voiceOn;
  }, [voiceOn]);

  useEffect(() => {
    utterancesRef.current = utterances;
  }, [utterances]);

  /* ------------------------------------------------- engine bootstrapping */

  useEffect(() => {
    let cancelled = false;
    void createEngine().then((eng) => {
      if (cancelled) return;
      engineRef.current = eng;
      setEngineKind(eng.kind);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const syncSpeakers = useCallback(() => {
    const eng = engineRef.current;
    if (eng) setSpeakers(eng.listSpeakers());
  }, []);

  /* --------------------------------------------------- Aria's voice (TTS) */

  const speak = useCallback((text: string) => {
    const t = text?.trim();
    if (!t || !voiceOnRef.current) return;
    speakQueueRef.current = speakQueueRef.current
      .then(async () => {
        try {
          const res = await fetch('/api/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: t.slice(0, 500) }),
          });
          if (!res.ok) return;
          const blob = await res.blob();
          if (blob.size < 200) return;
          const url = URL.createObjectURL(blob);
          await new Promise<void>((resolve) => {
            const audio = new Audio(url);
            currentAudioRef.current = audio;
            const done = () => {
              URL.revokeObjectURL(url);
              resolve();
            };
            audio.onended = done;
            audio.onerror = done;
            void audio.play().catch(done);
          });
        } catch {
          /* voice failures never block perception */
        }
      })
      .catch(() => {});
  }, []);

  const toggleVoice = useCallback(() => {
    setVoiceOn((v) => {
      const next = !v;
      voiceOnRef.current = next;
      if (!next) currentAudioRef.current?.pause();
      return next;
    });
  }, []);

  /* --------------------------------------------------------- utterance flow */

  const recordTiming = useCallback((us: number) => {
    const timings = engineTimingsRef.current;
    timings.push(us);
    if (timings.length > 20) timings.shift();
    const sorted = [...timings].sort((a, b) => a - b);
    setEngineTimingMs(sorted[Math.floor(sorted.length / 2)] / 1000);
  }, []);

  const processUtterance = useCallback(
    async (utt: Utterance, cleanPcm: Float32Array, wav: Uint8Array) => {
      const patch = (p: Partial<Utterance>) =>
        setUtterances((prev) => prev.map((u) => (u.id === utt.id ? { ...u, ...p } : u)));

      const transcribeOnce = async (targetDb: number, existing?: Uint8Array): Promise<string> => {
        const wavBytes =
          existing ??
          engineRef.current?.renderWav(cleanPcm, targetDb) ??
          new Uint8Array();
        const wavBase64 = bytesToBase64(wavBytes);
        const tres = await fetch('/api/transcribe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ audio_base64: wavBase64 }),
        });
        if (!tres.ok) {
          const e = await tres.json().catch(() => ({}));
          throw new Error(e?.error || `Transcription failed (${tres.status})`);
        }
        const { text } = (await tres.json()) as { text?: string };
        return (text ?? '').trim();
      };

      try {
        // 1) WHAT — transcription via the audio model, at a healthy listening level
        let text = await transcribeOnce(-18, wav);
        let words = text.split(/\s+/).filter(Boolean).length;

        // human second listen: words came back thin — lean in and try louder
        // instead of shrugging (never guess, ask the ear again)
        if (words < 2 && utt.durationMs > 700) {
          patch({ retried: true });
          const second = await transcribeOnce(-12);
          const secondWords = second.split(/\s+/).filter(Boolean).length;
          if (secondWords > words) {
            text = second;
            words = secondWords;
          }
        }

        if (!text) {
          patch({ status: 'no-speech' });
          return;
        }
        patch({ transcript: text, status: 'analyzing' });

        // 2) HOW + WHY — perception layers via the language model
        const history = utterancesRef.current
          .filter((u) => u.status === 'done' && u.transcript)
          .slice(0, 4)
          .map((u) => ({ speaker: u.speaker.label, text: u.transcript as string }))
          .reverse();

        const ares = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            transcript: text,
            acoustics: utt.acoustics,
            speaker: { label: utt.speaker.label, pitchHz: utt.speaker.pitchHz },
            history,
          }),
        });
        if (!ares.ok) {
          // transcript alone is still valuable — degrade gracefully
          patch({ status: 'done' });
          return;
        }
        const perception = (await ares.json()) as PerceptionResult;
        patch({ perception, status: 'done' });

        // Aria answers out loud — but only when the utterance was aimed at her
        // and only for live speech (uploaded files are for your eyes)
        if (
          utt.source === 'live' &&
          perception?.addressedToAssistant &&
          perception.awarenessNote
        ) {
          speak(perception.awarenessNote);
        }

        // 3) WHO — remember a name if this utterance introduced one
        if (perception.speakerName) {
          engineRef.current?.renameSpeaker(utt.speaker.id, perception.speakerName);
          syncSpeakers();
          const fresh = engineRef.current?.listSpeakers() ?? [];
          setUtterances((prev) =>
            prev.map((u) => {
              if (u.speaker.id !== utt.speaker.id) return u;
              const label = fresh.find((s) => s.id === utt.speaker.id)?.label ?? u.speaker.label;
              return { ...u, speaker: { ...u.speaker, label } };
            }),
          );
        }
      } catch (err) {
        patch({
          status: 'error',
          errorMessage: err instanceof Error ? err.message : 'Processing failed',
        });
      }
    },
    [speak, syncSpeakers],
  );

  const registerUtterance = useCallback(
    (rawPcm: Float32Array, source: 'live' | 'upload', startedAt?: number) => {
      const eng = engineRef.current;
      if (!eng || rawPcm.length < FRAME_SIZE) return;

      // one Rust call does everything: acoustics (WHERE), speaker (WHO),
      // denoise (attention), normalization + WAV (WHAT prep), spectrogram
      const analyzed = eng.analyzeSegment(rawPcm, -18);
      recordTiming(analyzed.elapsedUs);

      const seq = ++seqRef.current;
      const utt: Utterance = {
        id: uuidv4(),
        seq,
        startedAt: startedAt ?? Date.now(),
        durationMs: analyzed.acoustics.durationMs,
        speaker: analyzed.speaker,
        acoustics: analyzed.acoustics,
        transcript: null,
        perception: null,
        status: 'transcribing',
        source,
        denoiseDb: analyzed.denoiseDb,
        spectrogram: analyzed.spectrogram,
        engineUs: analyzed.elapsedUs,
      };
      setUtterances((prev) => [utt, ...prev].slice(0, MAX_UTTERANCES));
      syncSpeakers();
      // queue instead of firing in parallel — one model call at a time
      pipelineRef.current = pipelineRef.current
        .then(() => processUtterance(utt, analyzed.cleanPcm, analyzed.wav))
        .catch(() => {});
    },
    [processUtterance, recordTiming, syncSpeakers],
  );

  /* ---------------------------------------------------------- frame pump */

  /** Feed one 2048-sample 16 kHz frame; segments register under `source`. */
  const pushFrame = useCallback(
    (frame: Float32Array, source: 'live' | 'upload') => {
      const eng = engineRef.current;
      if (!eng) return;
      const m = eng.feedFrame(frame);
      metricsRef.current = {
        dbfs: m.dbfs,
        pitchHz: m.pitchHz,
        hfRatio: m.hfRatio,
        distanceBand: m.distanceBand,
        vad: m.vad,
        noiseFloorDb: m.noiseFloorDb,
      };
      if (m.segmentReady) {
        const seg = eng.takeSegment();
        if (seg) registerUtterance(seg, source);
      }
    },
    [registerUtterance],
  );

  /** Chunk (native rate) → resample in Rust → buffered 2048-sample frames. */
  const feedChunk = useCallback(
    (chunkNative: Float32Array, sampleRate: number) => {
      const eng = engineRef.current;
      if (!eng) return;
      const chunk = eng.resample(chunkNative, sampleRate, 16000);
      const merged = new Float32Array(pendingRef.current.length + chunk.length);
      merged.set(pendingRef.current);
      merged.set(chunk, pendingRef.current.length);
      pendingRef.current = merged;

      while (pendingRef.current.length >= FRAME_SIZE) {
        const frame = pendingRef.current.slice(0, FRAME_SIZE);
        pendingRef.current = pendingRef.current.subarray(FRAME_SIZE);
        pushFrame(frame, 'live');
      }
    },
    [pushFrame],
  );

  /* ---------------------------------------------------------- audio engine */

  const start = useCallback(async () => {
    if (aliveRef.current) return;
    setError(null);
    setStatus('starting');
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Microphone capture is not supported in this browser context.');
      }
      // AGC / noise suppression would destroy distance cues — keep them off.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
      });
      const ctx = new AudioContext();
      await ctx.resume();

      streamRef.current = stream;
      ctxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.72;
      source.connect(analyser);
      analyserRef.current = analyser;

      const processor = ctx.createScriptProcessor(4096, 1, 1);
      const mute = ctx.createGain();
      mute.gain.value = 0;
      source.connect(processor);
      processor.connect(mute);
      mute.connect(ctx.destination);

      processor.onaudioprocess = (ev: AudioProcessingEvent) => {
        feedChunk(ev.inputBuffer.getChannelData(0), ctx.sampleRate);
      };
      processorRef.current = processor;

      engineRef.current?.reset();
      pendingRef.current = new Float32Array(0);
      metricsRef.current = { ...IDLE_METRICS };
      aliveRef.current = true;
      setStatus('running');

      if (metricsTimerRef.current === null) {
        metricsTimerRef.current = window.setInterval(() => {
          setLiveMetrics({ ...metricsRef.current });
        }, 120);
      }
    } catch (err) {
      aliveRef.current = false;
      setStatus('error');
      const msg =
        err instanceof Error
          ? err.name === 'NotAllowedError'
            ? 'Microphone permission denied — allow mic access in your browser and try again.'
            : err.name === 'NotFoundError'
              ? 'No microphone found on this device.'
              : err.message
          : 'Failed to start microphone.';
      setError(msg);
    }
  }, [feedChunk]);

  const stop = useCallback(() => {
    const processor = processorRef.current;
    if (processor) {
      processor.onaudioprocess = null;
      processor.disconnect();
    }
    processorRef.current = null;
    analyserRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    void ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;

    const eng = engineRef.current;
    if (aliveRef.current && eng) {
      if (eng.flush()) {
        const seg = eng.takeSegment();
        if (seg) registerUtterance(seg, 'live');
      }
    }
    aliveRef.current = false;

    if (metricsTimerRef.current !== null) {
      window.clearInterval(metricsTimerRef.current);
      metricsTimerRef.current = null;
    }
    setLiveMetrics({ ...IDLE_METRICS });
    setStatus('idle');
  }, [registerUtterance]);

  /* ------------------------------------------------------------ file input */

  const analyzeFile = useCallback(
    async (file: File): Promise<number> => {
      if (uploadBusy) return 0;
      const eng = engineRef.current;
      if (!eng) throw new Error('Engine is still loading — try again in a second.');
      setUploadBusy(true);
      try {
        const ab = await file.arrayBuffer();
        const ctx = new AudioContext();
        let decoded: AudioBuffer;
        try {
          decoded = await ctx.decodeAudioData(ab);
        } finally {
          void ctx.close().catch(() => {});
        }
        const channels = Array.from({ length: decoded.numberOfChannels }, (_, c) =>
          decoded.getChannelData(c),
        );
        const mono = mixdownToMono(channels, decoded.length);
        const pcm16 = eng.resample(mono, decoded.sampleRate, 16000);

        // The file flows through the same engine — its quiet patches become
        // the learned noise profile, and speakers stay clustered together
        // with the live session (same registry, like the original engine).
        const before = seqRef.current;
        for (let off = 0; off + FRAME_SIZE <= pcm16.length; off += FRAME_SIZE) {
          pushFrame(pcm16.subarray(off, off + FRAME_SIZE), 'upload');
        }
        if (eng.flush()) {
          const seg = eng.takeSegment();
          if (seg) registerUtterance(seg, 'upload');
        }
        return seqRef.current - before;
      } finally {
        setUploadBusy(false);
      }
    },
    [pushFrame, registerUtterance, uploadBusy],
  );

  /* --------------------------------------------------------------- session */

  const clear = useCallback(() => {
    engineRef.current?.reset();
    seqRef.current = 0;
    engineTimingsRef.current = [];
    setEngineTimingMs(null);
    setUtterances([]);
    setSpeakers([]);
    setError(null);
  }, []);

  useEffect(() => {
    return () => {
      const processor = processorRef.current;
      if (processor) {
        processor.onaudioprocess = null;
        processor.disconnect();
      }
      streamRef.current?.getTracks().forEach((t) => t.stop());
      void ctxRef.current?.close().catch(() => {});
      if (metricsTimerRef.current !== null) window.clearInterval(metricsTimerRef.current);
    };
  }, []);

  return {
    status,
    error,
    utterances,
    speakers,
    liveMetrics,
    uploadBusy,
    analyserRef,
    distanceMeta: DISTANCE_META,
    voiceOn,
    toggleVoice,
    speak,
    start,
    stop,
    clear,
    analyzeFile,
    engineKind,
    engineTimingMs,
  };
}
