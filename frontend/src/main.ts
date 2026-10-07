import './style.css';
import { chatStream, health, synthesize, transcribe, type Lang, type Turn } from './api';
import { Recorder, SpeechPlayer } from './audio';
import { AvatarScene, type AvatarState } from './avatar';
import { LipSync } from './lipsync';

const MODEL_URL = '/models/tsehay.vrm';
const MAX_RECORD_MS = 25_000;
const KIOSK_IDLE_RESET_MS = 90_000;
const MAX_HISTORY_TURNS = 8;
const MIN_SENTENCE_CHARS = 20;
const SENTENCE_END = /[.!?።፧]["'”’)\]]?\s|\n/;

const SUGGESTIONS = [
  'How did the school begin?',
  'What were the first years like?',
  'What is the school proud of?',
  'ትምህርት ቤቱ እንዴት ተጀመረ?',
];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('avatar');
const userCaption = $<HTMLParagraphElement>('user-caption');
const answerCaption = $<HTMLParagraphElement>('answer-caption');
const notice = $<HTMLParagraphElement>('notice');
const statusText = $<HTMLSpanElement>('status-text');
const micBtn = $<HTMLButtonElement>('mic');
const micLabel = $<HTMLSpanElement>('mic-label');
const textForm = $<HTMLFormElement>('text-form');
const textInput = $<HTMLInputElement>('text-input');
const suggestions = $<HTMLElement>('suggestions');
const disclosure = $<HTMLElement>('disclosure');
const modelNote = $<HTMLParagraphElement>('model-note');

const kiosk = new URLSearchParams(location.search).get('mode') === 'kiosk';
// ?bust shows the stand-in bust even when a VRM file exists
const forceBust = new URLSearchParams(location.search).has('bust');
document.body.classList.toggle('kiosk', kiosk);

const avatar = new AvatarScene(canvas);
const player = new SpeechPlayer();
const lipSync = new LipSync();
const recorder = new Recorder();

let history: Turn[] = [];
let controller: AbortController | null = null;
let turnId = 0;
let recordTimer: number | undefined;
let idleTimer: number | undefined;

class SentenceSplitter {
  private buffer = '';

  push(text: string): string[] {
    this.buffer += text;
    const out: string[] = [];
    for (;;) {
      const match = SENTENCE_END.exec(this.buffer.slice(MIN_SENTENCE_CHARS));
      if (!match) break;
      const end = MIN_SENTENCE_CHARS + match.index + match[0].length;
      const sentence = this.buffer.slice(0, end).trim();
      if (sentence) out.push(sentence);
      this.buffer = this.buffer.slice(end);
    }
    return out;
  }

  flush(): string[] {
    const rest = this.buffer.trim();
    this.buffer = '';
    return rest ? [rest] : [];
  }
}

const STATE_LABELS: Record<AvatarState, string> = {
  idle: 'Ready',
  listening: 'Listening… tap again when done',
  thinking: 'Thinking…',
  speaking: 'Speaking',
};

function setState(state: AvatarState): void {
  avatar.state = state;
  document.body.dataset.state = state;
  statusText.textContent = STATE_LABELS[state];
  micLabel.textContent = state === 'listening' ? 'Stop recording' : 'Tap to ask';
}

function showNotice(message: string): void {
  notice.textContent = message;
  notice.hidden = !message;
}

function currentLang(): Lang {
  return (document.querySelector<HTMLInputElement>('input[name="lang"]:checked')?.value ?? 'auto') as Lang;
}

function stopEverything(): void {
  turnId++;
  controller?.abort();
  controller = null;
  player.stop();
}

async function ask(rawQuestion: string): Promise<void> {
  const question = rawQuestion.trim().slice(0, 500);
  if (!question) return;
  stopEverything();
  const myTurn = turnId;
  const lang = currentLang();
  controller = new AbortController();
  resetIdleTimer();

  userCaption.textContent = question;
  answerCaption.textContent = '';
  showNotice('');
  setState('thinking');

  const splitter = new SentenceSplitter();
  const speak = (sentence: string) => {
    const clip = synthesize(sentence, lang).catch((err: Error) => {
      if (myTurn === turnId) showNotice(`Voice unavailable: ${err.message}`);
      throw err;
    });
    player.enqueue(clip);
    if (myTurn === turnId) setState('speaking');
  };

  let answer = '';
  let failed = false;
  try {
    for await (const ev of chatStream(question, history, lang, controller.signal)) {
      if (myTurn !== turnId) return;
      if (ev.event === 'delta') {
        answer += ev.data.text;
        answerCaption.textContent = answer;
        splitter.push(ev.data.text).forEach(speak);
      } else if (ev.event === 'error') {
        failed = true;
        answerCaption.textContent = ev.data.message;
      }
    }
    splitter.flush().forEach(speak);
  } catch (err) {
    if (myTurn !== turnId || (err instanceof DOMException && err.name === 'AbortError')) return;
    failed = true;
    showNotice(err instanceof Error ? err.message : String(err));
  }

  if (!failed && answer) {
    history.push({ role: 'user', content: question }, { role: 'assistant', content: answer });
    history = history.slice(-MAX_HISTORY_TURNS * 2);
  }
  await player.whenIdle();
  if (myTurn === turnId) setState('idle');
}

async function finishRecording(): Promise<void> {
  window.clearTimeout(recordTimer);
  micBtn.setAttribute('aria-pressed', 'false');
  if (!recorder.recording) return;
  const wav = await recorder.stop();
  setState('thinking');
  try {
    const { text } = await transcribe(wav, currentLang());
    if (!text) {
      showNotice("I didn't catch that. Please tap and try again.");
      setState('idle');
      return;
    }
    await ask(text);
  } catch (err) {
    showNotice(err instanceof Error ? err.message : String(err));
    setState('idle');
  }
}

micBtn.addEventListener('click', async () => {
  await player.resume();
  if (recorder.recording) {
    await finishRecording();
    return;
  }
  stopEverything();
  try {
    await recorder.start();
  } catch (err) {
    showNotice(`Microphone not available: ${err instanceof Error ? err.message : String(err)}`);
    return;
  }
  showNotice('');
  micBtn.setAttribute('aria-pressed', 'true');
  setState('listening');
  recordTimer = window.setTimeout(finishRecording, MAX_RECORD_MS);
});

textForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  await player.resume();
  const q = textInput.value;
  textInput.value = '';
  await ask(q);
});

for (const q of SUGGESTIONS) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.textContent = q;
  btn.addEventListener('click', async () => {
    await player.resume();
    await ask(q);
  });
  suggestions.append(btn);
}

function resetIdleTimer(): void {
  if (!kiosk) return;
  window.clearTimeout(idleTimer);
  idleTimer = window.setTimeout(() => {
    stopEverything();
    history = [];
    userCaption.textContent = '';
    answerCaption.textContent = '';
    showNotice('');
    setState('idle');
  }, KIOSK_IDLE_RESET_MS);
}

if (kiosk) {
  for (const type of ['pointerdown', 'keydown']) document.addEventListener(type, resetIdleTimer);
  document.addEventListener(
    'pointerdown',
    () => document.documentElement.requestFullscreen?.().catch(() => undefined),
    { once: true },
  );
}

async function init(): Promise<void> {
  setState('idle');
  avatar.start((dt, time) => lipSync.update(player, dt, time));
  const [loaded] = await Promise.all([
    avatar.load(forceBust ? '/models/__none__.vrm' : MODEL_URL),
    health()
      .then((h) => {
        $('title').textContent = h.persona;
        if (h.school && h.school !== 'our school') {
          $('eyebrow').textContent = `${h.school} · 40th Anniversary`;
        }
        disclosure.textContent =
          `AI character based on ${h.persona}, created by the school for its 40th anniversary. ` +
          'Answers come from the school archives and may contain mistakes.';
        const missing = [!h.llm && 'language model', !h.speech && 'speech service'].filter(Boolean);
        if (missing.length) showNotice(`Server is missing configuration: ${missing.join(', ')}.`);
        else if (!h.indexChunks) showNotice('No archive loaded yet; run the ingest step on the server.');
      })
      .catch(() => showNotice('Cannot reach the server.')),
  ]);
  if (!loaded && !kiosk && !forceBust) {
    modelNote.hidden = false;
    modelNote.textContent = `Stand-in bust · add frontend/public${MODEL_URL}`;
  }
}

init();
