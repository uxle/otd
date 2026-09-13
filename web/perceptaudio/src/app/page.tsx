'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import {
  Activity,
  AudioWaveform,
  BarChart3,
  Cpu,
  Download,
  Ear,
  Filter,
  Gauge,
  MessageCircle,
  Mic,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Square,
  Upload,
  Users,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { useAudioAnalyzer, type Utterance } from '@/hooks/use-audio-analyzer';
import { UtteranceCard } from '@/components/audio/utterance-card';
import { Visualizer } from '@/components/audio/visualizer';
import { DISTANCE_META } from '@/lib/audio/dsp';

const TONE_DOT: Record<string, string> = {
  emerald: 'bg-emerald-400',
  lime: 'bg-lime-400',
  yellow: 'bg-yellow-400',
  amber: 'bg-amber-400',
  orange: 'bg-orange-400',
};

const TONE_BADGE: Record<string, string> = {
  emerald: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
  lime: 'border-lime-500/40 bg-lime-500/10 text-lime-300',
  yellow: 'border-yellow-500/40 bg-yellow-500/10 text-yellow-300',
  amber: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
  orange: 'border-orange-500/40 bg-orange-500/10 text-orange-300',
};

function dbfsToPercent(dbfs: number): number {
  return Math.min(100, Math.max(0, ((dbfs + 60) / 60) * 100));
}

interface SessionStats {
  totalMs: number;
  totalWords: number;
  bySpeaker: { label: string; ms: number }[];
  byEmotion: { emotion: string; count: number }[];
  byDistance: { band: string; count: number }[];
}

function computeStats(utterances: Utterance[]): SessionStats {
  const bySpeaker = new Map<string, number>();
  const byEmotion = new Map<string, number>();
  const byDistance = new Map<string, number>();
  let totalWords = 0;
  let totalMs = 0;

  for (const u of utterances) {
    totalMs += u.durationMs;
    bySpeaker.set(u.speaker.label, (bySpeaker.get(u.speaker.label) ?? 0) + u.durationMs);
    byDistance.set(u.acoustics.distanceBand, (byDistance.get(u.acoustics.distanceBand) ?? 0) + 1);
    if (u.perception) {
      byEmotion.set(u.perception.emotion, (byEmotion.get(u.perception.emotion) ?? 0) + 1);
    }
    if (u.transcript) {
      totalWords += u.transcript.split(/\s+/).filter(Boolean).length;
    }
  }
  return {
    totalMs,
    totalWords,
    bySpeaker: [...bySpeaker.entries()]
      .map(([label, ms]) => ({ label, ms }))
      .sort((a, b) => b.ms - a.ms),
    byEmotion: [...byEmotion.entries()]
      .map(([emotion, count]) => ({ emotion, count }))
      .sort((a, b) => b.count - a.count),
    byDistance: [...byDistance.entries()]
      .map(([band, count]) => ({ band, count }))
      .sort((a, b) => b.count - a.count),
  };
}

export default function Home() {
  const {
    status,
    error,
    utterances,
    speakers,
    liveMetrics,
    uploadBusy,
    analyserRef,
    voiceOn,
    toggleVoice,
    speak,
    start,
    stop,
    clear,
    analyzeFile,
    engineKind,
    engineTimingMs,
  } = useAudioAnalyzer();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const running = status === 'running';
  const liveMeta = DISTANCE_META[liveMetrics.distanceBand];
  const doneUtts = utterances.filter((u) => u.status === 'done');
  const avgDur =
    utterances.length > 0
      ? utterances.reduce((acc, u) => acc + u.durationMs, 0) / utterances.length / 1000
      : 0;
  const stats = useMemo(() => computeStats(utterances), [utterances]);

  const handleToggle = () => {
    if (running) {
      stop();
    } else {
      void start();
    }
  };

  const handleFile = useCallback(
    async (file: File) => {
      try {
        const found = await analyzeFile(file);
        if (found === 0) {
          toast({
            title: 'No speech detected',
            description: 'The file decoded fine, but no voice segments passed the activity gate.',
          });
        } else {
          toast({
            title: `Analyzing ${found} segment${found > 1 ? 's' : ''}`,
            description: `${file.name} — transcribing and reading perception layers.`,
          });
        }
      } catch (err) {
        toast({
          title: 'Could not decode audio',
          description: err instanceof Error ? err.message : 'Unsupported or corrupted audio file.',
          variant: 'destructive',
        });
      }
    },
    [analyzeFile, toast],
  );

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    await handleFile(file);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('audio/') && !/\.(wav|mp3|m4a|ogg|flac)$/i.test(file.name)) {
      toast({
        title: 'Not an audio file',
        description: `${file.name} does not look like audio. Drop a WAV, MP3, M4A, OGG or FLAC file.`,
        variant: 'destructive',
      });
      return;
    }
    await handleFile(file);
  };

  const exportSession = () => {
    const report = {
      exportedAt: new Date().toISOString(),
      app: 'PerceptAudio 2.0',
      engine: {
        kind: engineKind,
        version: '2.0.0',
        medianProcessingMs: engineTimingMs,
      },
      session: {
        utteranceCount: utterances.length,
        totalDurationMs: stats.totalMs,
        totalWords: stats.totalWords,
        speakers,
      },
      utterances: utterances.map((u) => ({
        seq: u.seq,
        startedAt: new Date(u.startedAt).toISOString(),
        source: u.source,
        speaker: u.speaker.label,
        acoustics: u.acoustics,
        denoiseDb: u.denoiseDb ?? null,
        retried: u.retried ?? false,
        transcript: u.transcript,
        perception: u.perception,
      })),
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `perceptaudio-session-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: 'Session exported', description: 'A JSON report of this perception session was downloaded.' });
  };

  const maxSpeakerMs = stats.bySpeaker[0]?.ms ?? 1;

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      {/* ------------------------------------------------------------ header */}
      <header className="border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center gap-3 px-4 py-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-emerald-500/30 bg-emerald-500/10">
            <AudioWaveform className="h-5 w-5 text-emerald-400" aria-hidden />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold leading-tight sm:text-lg">
              PerceptAudio <span className="font-normal text-emerald-400/90">2.0</span>
              <span className="hidden font-normal text-zinc-500 sm:inline"> — voice perception analyzer</span>
            </h1>
            <p className="hidden text-xs text-zinc-500 sm:block">
              Rust-powered ear: who is speaking · where they are · what, how and why
            </p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Badge
              variant="outline"
              className={
                engineKind === 'rust'
                  ? 'hidden gap-1.5 border-teal-500/40 bg-teal-500/10 text-teal-300 sm:flex'
                  : engineKind === 'ts'
                    ? 'hidden gap-1.5 border-amber-500/40 bg-amber-500/10 text-amber-300 sm:flex'
                    : 'hidden gap-1.5 border-zinc-700 bg-zinc-800/60 text-zinc-400 sm:flex'
              }
              title={
                engineKind === 'rust'
                  ? 'The DSP pipeline (VAD, YIN pitch, denoising, speaker clustering) runs in Rust compiled to WebAssembly'
                  : engineKind === 'ts'
                    ? 'WASM failed to load — running the TypeScript DSP fallback'
                    : 'Engine is loading'
              }
            >
              <Cpu className="h-3 w-3" aria-hidden />
              {engineKind === 'rust' && 'Rust WASM engine'}
              {engineKind === 'ts' && 'TS fallback'}
              {engineKind === 'loading' && 'Engine loading…'}
              {engineKind === 'rust' && engineTimingMs != null && (
                <span className="font-mono text-[10px] text-teal-400/70">
                  {engineTimingMs.toFixed(1)} ms
                </span>
              )}
            </Badge>
            {running ? (
              <Badge variant="outline" className="gap-1.5 border-emerald-500/40 bg-emerald-500/10 text-emerald-300">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
                </span>
                Live
              </Badge>
            ) : (
              <Badge variant="outline" className="border-zinc-700 bg-zinc-800/60 text-zinc-400">
                {status === 'starting' ? 'Starting…' : 'Standby'}
              </Badge>
            )}
          </div>
        </div>
      </header>

      {/* -------------------------------------------------------------- main */}
      <main className="mx-auto grid w-full max-w-7xl flex-1 grid-cols-1 gap-4 px-4 py-5 lg:grid-cols-[minmax(330px,400px)_1fr]">
        {/* left column — engine controls */}
        <div className="flex flex-col gap-4 lg:max-h-[calc(100vh-9rem)] lg:overflow-y-auto lg:pr-1 lg:scrollbar-slim">
          <Card className="border-zinc-800/90 bg-zinc-900/70 shadow-none">
            <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium">
                <Mic className="h-4 w-4 text-emerald-400" aria-hidden />
                Microphone
              </CardTitle>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-md text-zinc-500 hover:bg-zinc-800 hover:text-emerald-300"
                onClick={toggleVoice}
                aria-label={voiceOn ? 'Mute Aria\u2019s voice' : 'Unmute Aria\u2019s voice'}
                aria-pressed={voiceOn}
                title={voiceOn ? "Aria's voice on — she answers out loud when you talk to her" : "Aria's voice muted"}
              >
                {voiceOn ? (
                  <Volume2 className="h-4 w-4" aria-hidden />
                ) : (
                  <VolumeX className="h-4 w-4" aria-hidden />
                )}
              </Button>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={handleToggle}
                disabled={status === 'starting'}
                aria-pressed={running}
                className={`h-11 w-full gap-2 text-sm font-semibold ${
                  running
                    ? 'bg-rose-500/90 text-white hover:bg-rose-500'
                    : 'bg-emerald-500 text-zinc-950 hover:bg-emerald-400'
                }`}
              >
                {running ? (
                  <>
                    <Square className="h-4 w-4" aria-hidden /> Stop listening
                  </>
                ) : (
                  <>
                    <Mic className="h-4 w-4" aria-hidden /> Start listening
                  </>
                )}
              </Button>

              {error && (
                <p role="alert" className="rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs leading-snug text-rose-300">
                  {error}
                </p>
              )}

              <Visualizer analyserRef={analyserRef} active={running} />

              {/* level meter */}
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                    Loudness
                  </span>
                  <span className="font-mono text-xs text-zinc-400">
                    {running && liveMetrics.dbfs > -99 ? `${liveMetrics.dbfs.toFixed(1)} dBFS` : '— dBFS'}
                  </span>
                </div>
                <div
                  className="h-2 w-full overflow-hidden rounded-full bg-zinc-800"
                  role="meter"
                  aria-valuenow={Math.round(dbfsToPercent(liveMetrics.dbfs))}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-label="Microphone input level"
                >
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-emerald-500 via-lime-400 to-yellow-300 transition-[width] duration-150"
                    style={{ width: `${dbfsToPercent(running ? liveMetrics.dbfs : -100)}%` }}
                  />
                </div>
              </div>

              {/* live distance + vad + noise filter */}
              <div className="flex flex-wrap items-center gap-2">
                <Badge
                  variant="outline"
                  className={`gap-1.5 ${
                    running ? TONE_BADGE[liveMeta.tone] : 'border-zinc-700 bg-zinc-800/60 text-zinc-400'
                  }`}
                >
                  <Gauge className="h-3.5 w-3.5" aria-hidden />
                  {running ? `${liveMeta.label} · ${liveMeta.range}` : 'Distance: standby'}
                </Badge>
                <Badge
                  variant="outline"
                  className={
                    running && liveMetrics.vad === 'speech'
                      ? 'animate-pulse border-emerald-500/50 bg-emerald-500/15 text-emerald-300'
                      : 'border-zinc-700 bg-zinc-800/60 text-zinc-400'
                  }
                >
                  <Activity className="h-3.5 w-3.5" aria-hidden />
                  {running ? (liveMetrics.vad === 'speech' ? 'Speech detected' : 'Listening…') : 'VAD idle'}
                </Badge>
                <Badge
                  variant="outline"
                  className={
                    running && liveMetrics.noiseFloorDb > -95
                      ? 'gap-1.5 border-teal-500/40 bg-teal-500/10 text-teal-300'
                      : 'gap-1.5 border-zinc-700 bg-zinc-800/60 text-zinc-400'
                  }
                >
                  <Filter className="h-3.5 w-3.5" aria-hidden />
                  {running ? 'Noise filter on' : 'Filter idle'}
                </Badge>
              </div>

              {/* live readouts */}
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {[
                  { label: 'Pitch', value: liveMetrics.pitchHz ? `${liveMetrics.pitchHz} Hz` : '—' },
                  { label: 'Treble', value: `${(liveMetrics.hfRatio * 100).toFixed(0)}%` },
                  {
                    label: 'Noise floor',
                    value:
                      running && liveMetrics.noiseFloorDb > -95
                        ? `${liveMetrics.noiseFloorDb.toFixed(0)} dB`
                        : '—',
                  },
                  { label: 'Utterances', value: `${utterances.length}` },
                ].map((m) => (
                  <div key={m.label} className="rounded-md border border-zinc-800 bg-zinc-950/60 px-2.5 py-1.5 text-center">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">{m.label}</p>
                    <p className="font-mono text-sm text-zinc-200">{m.value}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* session insights */}
          <Card className="border-zinc-800/90 bg-zinc-900/70 shadow-none">
            <CardHeader className="flex-row items-center justify-between space-y-0 pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium">
                <BarChart3 className="h-4 w-4 text-lime-400" aria-hidden />
                Session insights
              </CardTitle>
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1.5 border-zinc-700 bg-zinc-900 px-2 text-[11px] text-zinc-300 hover:border-emerald-500/40 hover:bg-emerald-500/10 hover:text-emerald-300"
                onClick={exportSession}
                disabled={utterances.length === 0}
              >
                <Download className="h-3 w-3" aria-hidden />
                Export
              </Button>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Captured</p>
                  <p className="font-mono text-sm text-zinc-200">{utterances.length}</p>
                </div>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Analyzed</p>
                  <p className="font-mono text-sm text-zinc-200">{doneUtts.length}</p>
                </div>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Words</p>
                  <p className="font-mono text-sm text-zinc-200">{stats.totalWords}</p>
                </div>
              </div>

              {stats.bySpeaker.length > 0 && (
                <>
                  <Separator className="bg-zinc-800" />
                  <div>
                    <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                      Talk time by speaker
                    </p>
                    <ul className="space-y-1.5">
                      {stats.bySpeaker.map((s) => (
                        <li key={s.label} className="flex items-center gap-2">
                          <span className="w-28 shrink-0 truncate text-xs text-zinc-300">{s.label}</span>
                          <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-lime-400"
                              style={{ width: `${Math.max(4, (s.ms / maxSpeakerMs) * 100)}%` }}
                            />
                          </div>
                          <span className="w-12 shrink-0 text-right font-mono text-[10px] text-zinc-500">
                            {(s.ms / 1000).toFixed(1)}s
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </>
              )}

              {stats.byEmotion.length > 0 && (
                <>
                  <Separator className="bg-zinc-800" />
                  <div>
                    <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                      Emotional weather
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {stats.byEmotion.map((e) => (
                        <Badge key={e.emotion} variant="outline" className="border-zinc-700 bg-zinc-800/60 text-zinc-300">
                          {e.emotion}
                          <span className="ml-1 font-mono text-[10px] text-emerald-400/70">×{e.count}</span>
                        </Badge>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {stats.byDistance.length > 0 && (
                <>
                  <Separator className="bg-zinc-800" />
                  <div>
                    <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                      Distance footprint
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {stats.byDistance.map((d) => {
                        const meta = DISTANCE_META[d.band as keyof typeof DISTANCE_META];
                        return (
                          <Badge key={d.band} variant="outline" className={`gap-1.5 ${TONE_BADGE[meta?.tone ?? 'emerald']}`}>
                            <span className={`h-1.5 w-1.5 rounded-full ${TONE_DOT[meta?.tone ?? 'emerald']}`} aria-hidden />
                            {meta?.label ?? d.band}
                            <span className="font-mono text-[10px] opacity-70">×{d.count}</span>
                          </Badge>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}

              {utterances.length === 0 && (
                <p className="text-xs text-zinc-500">
                  Start a conversation and this panel fills up: talk time, emotional weather and the
                  session&apos;s distance footprint — exportable as a JSON report.
                </p>
              )}
            </CardContent>
          </Card>

          {/* how I hear — human terms */}
          <Card className="border-zinc-800/90 bg-zinc-900/70 shadow-none">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium">
                <Ear className="h-4 w-4 text-emerald-400" aria-hidden />
                How I hear — in human terms
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs leading-relaxed text-zinc-400">
                I&#8217;m <span className="text-emerald-300">Aria</span>. Between your voice and my answers,
                the same five things happen that a person&#8217;s ear and mind do:
              </p>
              <ol className="space-y-2.5">
                {[
                  {
                    icon: Ear,
                    title: 'Eardrum',
                    text: 'The microphone is my eardrum. Loudness and treble crispness tell me how far away you are — a close whisper or a voice across the room.',
                  },
                  {
                    icon: Filter,
                    title: 'Attention',
                    text: 'While nobody talks, I quietly learn the room\u2019s hum — fan, traffic, keyboard clatter — and subtract it. Only voices reach my understanding.',
                  },
                  {
                    icon: Users,
                    title: 'Turn-taking',
                    text: 'One speaker at a time gets the floor. Each voice carries a pitch fingerprint, so I notice who comes back — and I learn names when you offer them.',
                  },
                  {
                    icon: Sparkles,
                    title: 'Understanding',
                    text: 'Clean words go to the audio model at a healthy listening level. If they came back muffled or thin, I listen a second time, louder — I never guess.',
                  },
                  {
                    icon: MessageCircle,
                    title: 'Reply',
                    text: 'When you are actually talking to me, I answer out loud in my own voice. Side chatter I keep to myself — that\u2019s your privacy, not my business.',
                  },
                ].map((step, i) => (
                  <li key={step.title} className="flex gap-2.5">
                    <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-zinc-700 bg-zinc-800/60 font-mono text-[10px] text-zinc-400">
                      {i + 1}
                    </span>
                    <div className="min-w-0">
                      <p className="flex items-center gap-1.5 text-xs font-medium text-zinc-200">
                        <step.icon className="h-3.5 w-3.5 text-emerald-400/80" aria-hidden />
                        {step.title}
                      </p>
                      <p className="mt-0.5 text-xs leading-relaxed text-zinc-500">{step.text}</p>
                    </div>
                  </li>
                ))}
              </ol>
              <Separator className="bg-zinc-800" />
              <p className="flex items-start gap-2 text-[11px] leading-relaxed text-zinc-500">
                <Cpu className="mt-0.5 h-3.5 w-3.5 shrink-0 text-teal-400/80" aria-hidden />
                Everything acoustic — turn-taking, pitch, distance, noise removal, voiceprints — is
                computed by a <span className="text-teal-300">Rust DSP engine compiled to WebAssembly</span>:
                YIN pitch tracking, twiddle-table FFTs, Catmull-Rom resampling and Wiener-style
                spectral denoising, all in your browser.
              </p>
            </CardContent>
          </Card>

          {/* room snapshot */}
          <Card className="border-zinc-800/90 bg-zinc-900/70 shadow-none">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium">
                <Users className="h-4 w-4 text-lime-400" aria-hidden />
                Room snapshot
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {speakers.length === 0 ? (
                <p className="text-xs text-zinc-500">
                  No voices tracked yet. Distinct pitches will register as separate speakers.
                </p>
              ) : (
                <ul className="flex flex-wrap gap-1.5">
                  {speakers.map((s) => (
                    <li key={s.id}>
                      <Badge variant="outline" className="gap-1.5 border-emerald-500/40 bg-emerald-500/10 text-emerald-300">
                        {s.label}
                        <span className="font-mono text-[10px] text-emerald-400/70">
                          {s.pitchHz ? `≈${s.pitchHz}Hz` : '—'} · {s.utterances}×
                        </span>
                      </Badge>
                    </li>
                  ))}
                </ul>
              )}

              <Separator className="bg-zinc-800" />

              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Speakers</p>
                  <p className="font-mono text-sm text-zinc-200">{speakers.length}</p>
                </div>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Avg len</p>
                  <p className="font-mono text-sm text-zinc-200">{avgDur.toFixed(1)}s</p>
                </div>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Talk time</p>
                  <p className="font-mono text-sm text-zinc-200">{(stats.totalMs / 1000).toFixed(0)}s</p>
                </div>
              </div>

              <Separator className="bg-zinc-800" />

              <div className="flex flex-col gap-2 sm:flex-row">
                <Button
                  variant="outline"
                  className="h-10 flex-1 gap-2 border-zinc-700 bg-zinc-900 text-zinc-200 hover:border-zinc-600 hover:bg-zinc-800"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploadBusy}
                >
                  <Upload className="h-4 w-4" aria-hidden />
                  {uploadBusy ? 'Decoding…' : 'Analyze audio file'}
                </Button>
                <Button
                  variant="outline"
                  className="h-10 gap-2 border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-rose-500/40 hover:bg-rose-500/10 hover:text-rose-300"
                  onClick={clear}
                  disabled={utterances.length === 0 && speakers.length === 0}
                >
                  <RotateCcw className="h-4 w-4" aria-hidden />
                  Reset
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
                  className="hidden"
                  onChange={handleFileInput}
                  aria-label="Upload an audio file for perception analysis"
                />
              </div>
            </CardContent>
          </Card>

          {/* privacy + legend */}
          <Card className="border-zinc-800/90 bg-zinc-900/70 shadow-none">
            <CardContent className="space-y-3 pt-4">
              <div className="flex gap-2.5">
                <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-hidden />
                <p className="text-xs leading-relaxed text-zinc-400">
                  The mic streams only inside your browser. DSP — level, pitch, distance, noise removal —
                  runs locally. Audio leaves the page <span className="text-zinc-200">only after an
                  utterance ends</span>, and only to transcribe it. Background noise is learned and
                  erased in your browser, never uploaded. Nothing from side chatter is stored.
                </p>
              </div>
              <Separator className="bg-zinc-800" />
              <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                Distance bands (estimated from loudness + treble)
              </p>
              <ul className="space-y-1.5">
                {Object.entries(DISTANCE_META).map(([key, m]) => (
                  <li key={key} className="flex items-center gap-2 text-xs text-zinc-400">
                    <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${TONE_DOT[m.tone]}`} aria-hidden />
                    <span className="text-zinc-300">{m.label}</span>
                    <span className="font-mono text-[10px] text-zinc-500">{m.range}</span>
                    <span className="hidden truncate text-zinc-600 sm:inline">{m.hint}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>

        {/* right column — perception feed */}
        <section
          className="flex min-w-0 flex-col"
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
        >
          <div className="mb-3 flex items-center gap-2">
            <Ear className="h-4 w-4 text-emerald-400" aria-hidden />
            <h2 className="text-sm font-semibold text-zinc-200">Perception feed</h2>
            <Badge variant="outline" className="ml-auto border-zinc-700 bg-zinc-800/60 font-mono text-zinc-400">
              {utterances.length} event{utterances.length === 1 ? '' : 's'}
            </Badge>
          </div>

          <div
            className={`relative flex-1 rounded-lg transition-colors ${
              dragActive ? 'ring-2 ring-emerald-500/60 ring-offset-2 ring-offset-zinc-950' : ''
            }`}
          >
            {dragActive && (
              <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-lg border-2 border-dashed border-emerald-500/50 bg-emerald-500/5">
                <p className="flex items-center gap-2 text-sm font-medium text-emerald-300">
                  <Upload className="h-4 w-4" aria-hidden />
                  Drop an audio file to analyze it
                </p>
              </div>
            )}
            <ScrollArea className="h-[70vh] rounded-lg border border-zinc-800/90 bg-zinc-950/40 lg:h-[calc(100vh-12rem)]">
              <div className="space-y-3 p-1.5">
                {utterances.length === 0 ? (
                  <div className="flex h-64 flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-zinc-800 text-center">
                    <Ear className="h-8 w-8 text-zinc-700" aria-hidden />
                    <div>
                      <p className="text-sm text-zinc-400">Nothing heard yet</p>
                      <p className="mt-1 text-xs text-zinc-600">
                        Start the mic and speak — or drop / upload an audio file. Every utterance
                        lands here as a WHO · WHERE · WHAT · HOW · WHY card.
                      </p>
                    </div>
                  </div>
                ) : (
                  <AnimatePresence initial={false}>
                    {utterances.map((utt: Utterance) => (
                      <UtteranceCard key={utt.id} utt={utt} speak={speak} />
                    ))}
                  </AnimatePresence>
                )}
              </div>
            </ScrollArea>
          </div>
        </section>
      </main>

      {/* ------------------------------------------------------------ footer */}
      <footer className="mt-auto border-t border-zinc-800/80 bg-zinc-950">
        <div className="mx-auto w-full max-w-7xl px-4 py-3">
          <p className="text-center text-[11px] text-zinc-600">
            PerceptAudio 2.0 · DSP in Rust/WebAssembly (YIN · FFT · spectral denoising · speaker
            clustering) · Aria answers via TTS · distance &amp; speaker identity are heuristic
            estimates · transcription and perception via z-ai models
          </p>
        </div>
      </footer>
    </div>
  );
}
