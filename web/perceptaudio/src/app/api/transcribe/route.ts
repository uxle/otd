import { NextRequest, NextResponse } from 'next/server';
import ZAI from 'z-ai-web-dev-sdk';

export const runtime = 'nodejs';
export const maxDuration = 60;

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
 * POST /api/transcribe
 * Body: { audio_base64: string } — 16 kHz mono 16-bit WAV, base64-encoded.
 * Returns: { text: string } — transcription from the audio model (ASR).
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => null);
    const audioBase64 = body?.audio_base64;
    if (typeof audioBase64 !== 'string' || audioBase64.length === 0) {
      return NextResponse.json({ error: 'audio_base64 is required' }, { status: 400 });
    }

    const zai = await ZAI.create();
    const result = await withRateLimitRetry(() =>
      zai.audio.asr.create({ file_base64: audioBase64 }),
    );
    const text = (result?.text ?? '').trim();

    return NextResponse.json({ text });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Transcription failed';
    console.error('[api/transcribe]', message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
