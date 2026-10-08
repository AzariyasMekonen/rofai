export type Lang = 'auto' | 'en' | 'am' | 'om';

export interface Turn {
  role: 'user' | 'assistant';
  content: string;
}

export interface Viseme {
  t: number;
  id: number;
}

export interface TtsResult {
  audio: string | null;
  mime: string | null;
  visemes: Viseme[];
  language: string;
}

export interface Health {
  ok: boolean;
  llm: boolean;
  speech: boolean;
  indexChunks: number;
  assistant: string;
  founder: string;
  school: string;
}

export interface ChatEvent {
  event: string;
  data: any;
}

const BASE: string = import.meta.env.VITE_API_BASE ?? '';

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return typeof body.detail === 'string' ? body.detail : res.statusText;
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function health(): Promise<Health> {
  const res = await fetch(`${BASE}/api/health`);
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export async function transcribe(wav: Blob, language: Lang): Promise<{ text: string; language: string | null }> {
  const form = new FormData();
  form.append('audio', wav, 'question.wav');
  // Speech recognition supports English and Amharic only
  form.append('language', language === 'om' ? 'auto' : language);
  const res = await fetch(`${BASE}/api/stt`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export async function synthesize(text: string, language: Lang): Promise<TtsResult> {
  const res = await fetch(`${BASE}/api/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, language }),
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export async function* chatStream(
  message: string,
  history: Turn[],
  language: Lang,
  signal: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, language }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorMessage(res));

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary: number;
    while ((boundary = buffer.indexOf('\n\n')) >= 0) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = 'message';
      const data: string[] = [];
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
      }
      if (data.length) yield { event, data: JSON.parse(data.join('\n')) };
    }
  }
}
