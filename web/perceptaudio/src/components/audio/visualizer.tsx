'use client';

import { useEffect, useRef, useState } from 'react';
import { AudioLines, BarChart3, Waves } from 'lucide-react';

type VizMode = 'wave' | 'spectrum' | 'spec';

interface VisualizerProps {
  analyserRef: React.RefObject<AnalyserNode | null>;
  active: boolean;
}

const SPEC_W = 256;
const SPEC_H = 64;
const SPEC_MIN_HZ = 80;
const SPEC_MAX_HZ = 7500;

/** dark → emerald → lime heat map for the spectrogram */
function heatColor(t: number): [number, number, number] {
  const stops: [number, number, number][] = [
    [10, 10, 14],
    [6, 78, 59],
    [16, 185, 129],
    [163, 230, 53],
    [253, 224, 71],
  ];
  const x = Math.min(0.999, Math.max(0, t)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

const MODES: { id: VizMode; label: string; icon: typeof Waves }[] = [
  { id: 'wave', label: 'Wave', icon: Waves },
  { id: 'spectrum', label: 'Spectrum', icon: BarChart3 },
  { id: 'spec', label: 'Spectrogram', icon: AudioLines },
];

/**
 * Live scope with three views: waveform, spectrum bars and a scrolling
 * log-frequency spectrogram. Draws a resting flatline when the engine is off.
 */
export function Visualizer({ analyserRef, active }: VisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const specXRef = useRef(0);
  const timeBufRef = useRef<Float32Array<ArrayBuffer> | null>(null);
  const freqBufRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const [mode, setMode] = useState<VizMode>('wave');

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!offscreenRef.current && typeof document !== 'undefined') {
      const off = document.createElement('canvas');
      off.width = SPEC_W;
      off.height = SPEC_H;
      offscreenRef.current = off;
      const octx = off.getContext('2d');
      if (octx) {
        octx.fillStyle = '#0a0a0c';
        octx.fillRect(0, 0, SPEC_W, SPEC_H);
      }
    }

    let raf = 0;

    const drawSpectrogram = (analyser: AnalyserNode, w: number, h: number) => {
      const off = offscreenRef.current;
      const octx = off?.getContext('2d');
      if (!off || !octx) return;

      if (!freqBufRef.current || freqBufRef.current.length !== analyser.frequencyBinCount) {
        freqBufRef.current = new Uint8Array(analyser.frequencyBinCount);
      }
      const freq = freqBufRef.current;
      analyser.getByteFrequencyData(freq);

      const binHz = analyser.context.sampleRate / analyser.fftSize;
      const minBin = Math.max(1, Math.floor(SPEC_MIN_HZ / binHz));
      const maxBin = Math.min(freq.length - 1, Math.ceil(SPEC_MAX_HZ / binHz));

      // advance one column
      specXRef.current = (specXRef.current + 1) % SPEC_W;
      const x = specXRef.current;
      for (let row = 0; row < SPEC_H; row++) {
        const t = row / (SPEC_H - 1);
        // log-spaced bins, high frequencies at the top
        const b = Math.floor(minBin * Math.pow(maxBin / minBin, 1 - t));
        const v = freq[b] / 255;
        const [r, g, bl] = heatColor(Math.pow(v, 1.4));
        octx.fillStyle = `rgb(${r},${g},${bl})`;
        octx.fillRect(x, SPEC_H - 1 - row, 1, 1);
      }
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, w, h);
    };

    const draw = () => {
      raf = requestAnimationFrame(draw);

      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const analyser = analyserRef.current;
      if (!analyser || !active) {
        ctx.fillStyle = '#09090b';
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = 'rgba(52, 211, 153, 0.35)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();
        return;
      }

      if (mode === 'spec') {
        drawSpectrogram(analyser, w, h);
        return;
      }

      // subtle grid
      ctx.strokeStyle = 'rgba(63, 63, 70, 0.35)';
      ctx.lineWidth = 1;
      for (let gx = 0; gx <= w; gx += Math.max(40, w / 12)) {
        ctx.beginPath();
        ctx.moveTo(Math.round(gx) + 0.5, 0);
        ctx.lineTo(Math.round(gx) + 0.5, h);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(0, Math.round(h / 2) + 0.5);
      ctx.lineTo(w, Math.round(h / 2) + 0.5);
      ctx.stroke();

      if (mode === 'wave') {
        const waveH = h * 0.42;
        const specTop = h * 0.52;
        const specH = h - specTop;

        if (!timeBufRef.current || timeBufRef.current.length !== analyser.fftSize) {
          timeBufRef.current = new Float32Array(analyser.fftSize);
        }
        const time = timeBufRef.current;
        analyser.getFloatTimeDomainData(time);
        ctx.strokeStyle = '#34d399';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        const step = Math.max(1, Math.floor(time.length / w));
        const amp = waveH / 2 - 2;
        for (let x = 0; x < w; x++) {
          const idx = Math.min(time.length - 1, x * step);
          let v = 0;
          // min-max decimation for a dense but honest trace
          for (let j = 0; j < step && idx + j < time.length; j++) {
            const a = Math.abs(time[idx + j]);
            if (a > v) v = a;
          }
          const y = waveH / 2 - Math.sign(time[idx] || 1) * v * amp * 2;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        if (!freqBufRef.current || freqBufRef.current.length !== analyser.frequencyBinCount) {
          freqBufRef.current = new Uint8Array(analyser.frequencyBinCount);
        }
        const freq = freqBufRef.current;
        analyser.getByteFrequencyData(freq);
        const bars = 56;
        const barW = w / bars;
        const binsPerBar = freq.length / bars;
        for (let b = 0; b < bars; b++) {
          let sum = 0;
          for (let k = 0; k < binsPerBar; k++) sum += freq[Math.floor(b * binsPerBar) + k] ?? 0;
          const avg = sum / binsPerBar / 255;
          const bh = Math.max(1, avg * avg * specH); // perceptual squash
          const y = h - bh;
          const grad = ctx.createLinearGradient(0, y, 0, h);
          grad.addColorStop(0, 'rgba(163, 230, 53, 0.95)');
          grad.addColorStop(1, 'rgba(16, 185, 129, 0.25)');
          ctx.fillStyle = grad;
          ctx.fillRect(b * barW + barW * 0.15, y, barW * 0.7, bh);
        }
      } else {
        // full-height spectrum
        if (!freqBufRef.current || freqBufRef.current.length !== analyser.frequencyBinCount) {
          freqBufRef.current = new Uint8Array(analyser.frequencyBinCount);
        }
        const freq = freqBufRef.current;
        analyser.getByteFrequencyData(freq);
        const bars = 72;
        const barW = w / bars;
        const binsPerBar = freq.length / bars;
        for (let b = 0; b < bars; b++) {
          let sum = 0;
          for (let k = 0; k < binsPerBar; k++) sum += freq[Math.floor(b * binsPerBar) + k] ?? 0;
          const avg = sum / binsPerBar / 255;
          const bh = Math.max(1, avg * avg * h);
          const y = h - bh;
          const grad = ctx.createLinearGradient(0, y, 0, h);
          grad.addColorStop(0, 'rgba(253, 224, 71, 0.95)');
          grad.addColorStop(0.5, 'rgba(163, 230, 53, 0.85)');
          grad.addColorStop(1, 'rgba(16, 185, 129, 0.3)');
          ctx.fillStyle = grad;
          ctx.fillRect(b * barW + barW * 0.12, y, barW * 0.76, bh);
        }
      }
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [analyserRef, active, mode]);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1" role="tablist" aria-label="Visualizer mode">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            role="tab"
            aria-selected={mode === m.id}
            onClick={() => setMode(m.id)}
            className={`flex items-center gap-1 rounded-md border px-2 py-0.5 text-[10px] font-medium transition-colors ${
              mode === m.id
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                : 'border-zinc-800 bg-zinc-900/60 text-zinc-500 hover:border-zinc-700 hover:text-zinc-300'
            }`}
          >
            <m.icon className="h-3 w-3" aria-hidden />
            {m.label}
          </button>
        ))}
      </div>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={
          active
            ? `Live microphone ${mode} visualization`
            : 'Audio visualizer idle'
        }
        className="h-28 w-full rounded-md border border-zinc-800 bg-zinc-950"
      />
      {mode === 'spec' && (
        <p className="text-[10px] text-zinc-600">
          Scrolling log-frequency spectrogram · 80 Hz – 7.5 kHz · brighter = more energy
        </p>
      )}
    </div>
  );
}
