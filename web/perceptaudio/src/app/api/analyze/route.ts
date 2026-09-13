import { NextRequest, NextResponse } from 'next/server';
import ZAI from 'z-ai-web-dev-sdk';

export const runtime = 'nodejs';
export const maxDuration = 60;

const EMOTIONS = [
  'calm',
  'excited',
  'urgent',
  'frustrated',
  'sad',
  'sarcastic',
  'whispering',
  'neutral',
] as const;

const INTENTS = [
  'question',
  'command',
  'request',
  'complaint',
  'compliment',
  'casual-talk',
  'informational',
] as const;

const SYSTEM_PROMPT = `You are the perception module of a voice-aware assistant. You receive ONE transcribed utterance together with its acoustic measurements, and you infer the pragmatic layers the audio alone cannot carry.

Analyze strictly and only the data provided — never invent extra speakers, words or events.

Return ONLY minified JSON, no markdown fences, no commentary, matching exactly:
{
  "emotion": one of [${EMOTIONS.join(', ')}],
  "emotionCue": "one short sentence of evidence from the text or acoustics",
  "intent": one of [${INTENTS.join(', ')}],
  "addressedToAssistant": true when the utterance reads as directed AT the assistant (question, command, request), false for side conversation,
  "speakerName": string or null — set ONLY if this utterance introduces a name (e.g. "it's Maria"),
  "awarenessNote": "one warm, situational sentence the assistant could say, weaving in distance/emotion when relevant (e.g. inviting a far speaker closer, acknowledging urgency)",
  "summary": "max 8 words capturing WHAT was said"
}

Guidance:
- distanceBand "far"/"very-far" + soft dbfs => awarenessNote should invite them closer or to speak up.
- very-close + hushed/short words + emotion cue => possibly whispering.
- Untranscribable or empty-feeling text => keep neutral, summary "unclear speech".
- speakerName must be null unless explicitly stated in the transcript.`;

interface AnalyzeBody {
  transcript?: string;
  acoustics?: {
    dbfs?: number;
    peakDbfs?: number;
    hfRatio?: number;
    pitchHz?: number | null;
    distanceBand?: string;
    durationMs?: number;
  };
  speaker?: { label?: string; pitchHz?: number | null };
  history?: { speaker: string; text: string }[];
}

function parseJsonLoose(raw: string): Record<string, unknown> | null {
  const cleaned = raw
    .replace(/```json/gi, '')
    .replace(/```/g, '')
    .trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) return null;
  try {
    return JSON.parse(cleaned.slice(start, end + 1));
  } catch {
    return null;
  }
}

/** Run an async task with backoff retries on rate-limit (429) errors. */
async function withRateLimitRetry<T>(task: () => Promise<T>, retries = 2): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await task();
    } catch (err) {
      const rateLimited = err instanceof Error && err.message.includes('429');
      if (!rateLimited || attempt >= retries) throw err;
      await new Promise((r) => setTimeout(r, 900 * (attempt + 1)));
    }
  }
}

/**
 * POST /api/analyze
 * Body: { transcript, acoustics, speaker, history }
 * Returns the five-layer perception: HOW (emotion), WHY (intent + addressed),
 * plus an awareness note the UI can surface directly.
 */
export async function POST(req: NextRequest) {
  try {
    const body = (await req.json().catch(() => null)) as AnalyzeBody | null;
    if (!body || typeof body.transcript !== 'string' || body.transcript.length === 0) {
      return NextResponse.json({ error: 'transcript is required' }, { status: 400 });
    }

    const history = Array.isArray(body.history) ? body.history.slice(-4) : [];
    const payload = {
      transcript: body.transcript,
      acoustics: body.acoustics ?? {},
      speaker: body.speaker ?? {},
      recentHistory: history,
    };

    const zai = await ZAI.create();
    const completion = await withRateLimitRetry(() =>
      zai.chat.completions.create({
        messages: [
          { role: 'assistant', content: SYSTEM_PROMPT },
          { role: 'user', content: JSON.stringify(payload) },
        ],
        thinking: { type: 'disabled' },
      }),
    );

    const raw = completion.choices[0]?.message?.content ?? '';
    const parsed = parseJsonLoose(raw);
    if (!parsed) {
      console.error('[api/analyze] unparseable model output:', raw.slice(0, 300));
      return NextResponse.json({ error: 'Failed to parse perception output' }, { status: 502 });
    }

    // sanitize enums so the UI never renders junk
    const emotion = EMOTIONS.includes(parsed.emotion as (typeof EMOTIONS)[number])
      ? parsed.emotion
      : 'neutral';
    const intent = INTENTS.includes(parsed.intent as (typeof INTENTS)[number])
      ? parsed.intent
      : 'casual-talk';

    return NextResponse.json({
      emotion,
      emotionCue: typeof parsed.emotionCue === 'string' ? parsed.emotionCue : '',
      intent,
      addressedToAssistant: Boolean(parsed.addressedToAssistant),
      speakerName: typeof parsed.speakerName === 'string' ? parsed.speakerName : null,
      awarenessNote: typeof parsed.awarenessNote === 'string' ? parsed.awarenessNote : '',
      summary: typeof parsed.summary === 'string' ? parsed.summary : '',
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Perception analysis failed';
    console.error('[api/analyze]', message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
