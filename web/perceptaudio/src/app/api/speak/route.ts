import { NextRequest, NextResponse } from 'next/server';
import ZAI from 'z-ai-web-dev-sdk';

export const runtime = 'nodejs';
export const maxDuration = 60;

/**
 * POST /api/speak
 * Body: { text: string } — a short awareness note / reply.
 * Returns: WAV audio spoken by Aria (warm female voice, "tongtong").
 * Called only for utterances that were addressed to the assistant.
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => null);
    const text = typeof body?.text === 'string' ? body.text.trim().slice(0, 1024) : '';
    if (!text) {
      return NextResponse.json({ error: 'text is required' }, { status: 400 });
    }

    const zai = await ZAI.create();
    const response = await zai.audio.tts.create({
      input: text,
      voice: 'tongtong',
      speed: 1.0,
      response_format: 'wav',
      stream: false,
    });

    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(new Uint8Array(arrayBuffer));
    if (buffer.length < 100 || buffer.subarray(0, 4).toString('ascii') !== 'RIFF') {
      return NextResponse.json({ error: 'Empty or invalid speech audio' }, { status: 502 });
    }

    return new NextResponse(buffer, {
      status: 200,
      headers: {
        'Content-Type': 'audio/wav',
        'Content-Length': buffer.length.toString(),
        'Cache-Control': 'no-cache',
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Speech synthesis failed';
    console.error('[api/speak]', message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
