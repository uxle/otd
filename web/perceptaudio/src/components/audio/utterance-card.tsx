'use client';

import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Bot, Cpu, FileAudio, Headphones, Mic, Quote, Sparkles, Volume2, XCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { DISTANCE_META, type DistanceBand } from '@/lib/audio/dsp';
import type { Utterance } from '@/hooks/use-audio-analyzer';

const TONE_CLASSES: Record<string, string> = {
  emerald: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
  lime: 'border-lime-500/40 bg-lime-500/10 text-lime-300',
  yellow: 'border-yellow-500/40 bg-yellow-500/10 text-yellow-300',
  amber: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
  orange: 'border-orange-500/40 bg-orange-500/10 text-orange-300',
};

const INTENT_TONE: Record<string, string> = {
  question: 'border-teal-500/40 bg-teal-500/10 text-teal-300',
  command: 'border-orange-500/40 bg-orange-500/10 text-orange-300',
  request: 'border-lime-500/40 bg-lime-500/10 text-lime-300',
  complaint: 'border-rose-500/40 bg-rose-500/10 text-rose-300',
  compliment: 'border-pink-500/40 bg-pink-500/10 text-pink-300',
  'casual-talk': 'border-zinc-600 bg-zinc-800/60 text-zinc-300',
  informational: 'border-cyan-500/40 bg-cyan-500/10 text-cyan-300',
};

function formatClock(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function emotionTone(emotion: string): string {
  const map: Record<string, string> = {
    calm: 'text-zinc-300',
    neutral: 'text-zinc-300',
    excited: 'text-lime-300',
    urgent: 'text-orange-300',
    frustrated: 'text-rose-300',
    sad: 'text-sky-300',
    sarcastic: 'text-fuchsia-300',
    whispering: 'text-purple-300',
  };
  return map[emotion] ?? 'text-zinc-300';
}

/** dark → emerald → lime heat map (mirrors the live spectrogram) */
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

function SpectrogramThumb({ spec }: { spec: NonNullable<Utterance['spectrogram']> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 300;
    const h = canvas.clientHeight || 40;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = '#0a0a0c';
    ctx.fillRect(0, 0, w, h);

    const { cols, rows, data } = spec;
    const colW = w / cols;
    const rowH = h / rows;
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        const v = data[c * rows + r] ?? 0;
        if (v <= 0.01) continue;
        const [cr, cg, cb] = heatColor(v);
        ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
        ctx.fillRect(c * colW, h - (r + 1) * rowH, Math.ceil(colW), Math.ceil(rowH));
      }
    }
  }, [spec]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Voice spectrogram of this utterance"
      className="mt-2 h-10 w-full rounded border border-zinc-800/80 bg-zinc-950"
    />
  );
}

export function UtteranceCard({ utt, speak }: { utt: Utterance; speak?: (text: string) => void }) {
  const meta = DISTANCE_META[utt.acoustics.distanceBand as DistanceBand];
  const perception = utt.perception;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: 'spring', stiffness: 320, damping: 26 }}
    >
      <Card className="border-zinc-800/90 bg-zinc-900/70 shadow-none backdrop-blur-sm transition-colors hover:border-zinc-700">
        <CardContent className="p-4">
          {/* header: WHO + WHERE */}
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="gap-1.5 border-emerald-500/40 bg-emerald-500/10 font-medium text-emerald-300">
              {utt.source === 'live' ? <Mic className="h-3 w-3" aria-hidden /> : <FileAudio className="h-3 w-3" aria-hidden />}
              {utt.speaker.label}
            </Badge>
            <Badge variant="outline" className={`gap-1.5 ${TONE_CLASSES[meta.tone]}`}>
              {meta.label} · {meta.range}
            </Badge>
            <span className="ml-auto flex items-center gap-2 font-mono text-[11px] text-zinc-500">
              {utt.retried && (
                <Badge
                  variant="outline"
                  className="gap-1 border-teal-500/40 bg-teal-500/10 px-1.5 py-0 text-[10px] text-teal-300"
                  title="Words came back thin, so Aria listened a second time, louder"
                >
                  <Headphones className="h-3 w-3" aria-hidden />
                  re-listened
                </Badge>
              )}
              {formatClock(utt.startedAt)} · {(utt.durationMs / 1000).toFixed(1)}s
            </span>
          </div>

          {/* WHAT */}
          <div className="mt-3">
            {utt.status === 'transcribing' && (
              <div className="flex items-center gap-2 text-sm text-zinc-400" aria-live="polite">
                <Sparkles className="h-4 w-4 animate-pulse text-emerald-400" aria-hidden />
                <span className="animate-pulse">Transcribing with the audio model…</span>
              </div>
            )}
            {utt.status === 'analyzing' && (
              <div className="space-y-2">
                <p className="text-base text-zinc-100">{utt.transcript}</p>
                <p className="flex items-center gap-2 text-xs text-zinc-400" aria-live="polite">
                  <Bot className="h-3.5 w-3.5 animate-pulse text-lime-400" aria-hidden />
                  <span className="animate-pulse">Reading emotion and intent…</span>
                </p>
              </div>
            )}
            {(utt.status === 'done' || utt.status === 'error') && utt.transcript && (
              <p className="text-base leading-relaxed text-zinc-100">
                <Quote className="mr-1.5 inline h-3.5 w-3.5 text-zinc-600" aria-hidden />
                {utt.transcript}
              </p>
            )}
            {utt.status === 'no-speech' && (
              <p className="text-sm italic text-zinc-500">
                No clear speech in this burst — most likely background noise or a distant sound.
              </p>
            )}
            {utt.status === 'error' && (
              <p className="mt-1 flex items-start gap-1.5 text-sm text-rose-400">
                <XCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                {utt.errorMessage ?? 'Processing failed'}
              </p>
            )}
          </div>

          {/* HOW + WHY */}
          {perception && (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <div className="rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                  HOW · emotional state
                </p>
                <p className={`text-sm font-medium ${emotionTone(perception.emotion)}`}>
                  {perception.emotion}
                </p>
                {perception.emotionCue && (
                  <p className="mt-0.5 text-xs leading-snug text-zinc-500">{perception.emotionCue}</p>
                )}
              </div>
              <div className="rounded-md border border-zinc-800 bg-zinc-950/60 px-3 py-2">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                  WHY · likely intent
                </p>
                <div className="mt-0.5 flex flex-wrap items-center gap-1.5">
                  <Badge
                    variant="outline"
                    className={INTENT_TONE[perception.intent] ?? INTENT_TONE['casual-talk']}
                  >
                    {perception.intent}
                  </Badge>
                  {perception.addressedToAssistant && (
                    <Badge variant="outline" className="gap-1 border-emerald-500/40 bg-emerald-500/10 text-emerald-300">
                      <Bot className="h-3 w-3" aria-hidden />
                      talking to you
                    </Badge>
                  )}
                </div>
                {perception.speakerName && (
                  <p className="mt-1 text-xs text-zinc-500">
                    introduced as <span className="text-zinc-300">{perception.speakerName}</span>
                  </p>
                )}
              </div>
            </div>
          )}

          {/* awareness note — Aria's situational reply */}
          {perception?.awarenessNote && (
            <div className="mt-3 flex items-start gap-2">
              <p className="flex-1 border-l-2 border-emerald-500/60 pl-3 text-sm italic leading-snug text-emerald-200/90">
                “{perception.awarenessNote}”
              </p>
              {speak && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 shrink-0 rounded-md text-zinc-500 hover:bg-zinc-800 hover:text-emerald-300"
                  onClick={() => speak(perception.awarenessNote)}
                  aria-label="Listen to Aria say this"
                  title="Listen to Aria say this"
                >
                  <Volume2 className="h-3.5 w-3.5" aria-hidden />
                </Button>
              )}
            </div>
          )}

          {/* voiceprint — the spectrogram the Rust engine rendered */}
          {utt.spectrogram && utt.spectrogram.cols > 0 && <SpectrogramThumb spec={utt.spectrogram} />}

          {/* acoustic measurements + engine timing */}
          <p className="mt-3 font-mono text-[11px] text-zinc-600">
            {utt.acoustics.dbfs.toFixed(1)} dBFS · peak {utt.acoustics.peakDbfs.toFixed(1)} · HF{' '}
            {(utt.acoustics.hfRatio * 100).toFixed(0)}% · F0{' '}
            {utt.acoustics.pitchHz ? `${utt.acoustics.pitchHz} Hz` : '—'}
            {utt.denoiseDb != null && utt.denoiseDb < -0.5
              ? ` · noise −${Math.abs(utt.denoiseDb).toFixed(1)} dB`
              : ''}
          </p>
          {utt.engineUs != null && (
            <p className="mt-1 flex items-center gap-1.5 font-mono text-[10px] text-teal-500/70">
              <Cpu className="h-3 w-3" aria-hidden />
              rust dsp {(utt.engineUs / 1000).toFixed(1)} ms
            </p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
