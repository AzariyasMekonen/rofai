import './call.css';
import { chatStream, health, synthesize, transcribe, type Lang, type Turn } from './api';
import { SpeechPlayer } from './audio';
import { AvatarScene, type AvatarState } from './avatar';
import { LipSync } from './lipsync';
import { VoiceDetector } from './vad';

const MODEL_URL = '/models/mekdes.vrm';
const KIOSK_IDLE_END_MS = 90_000;
const MAX_HISTORY_TURNS = 8;
const MIN_SENTENCE_CHARS = 20;
const SENTENCE_END = /[.!?።፧]["'”’)\]]?\s|\n/;

const SUGGESTIONS = [
  'How did the school begin?',
  'Who was the founder?',
  'What is the school proud of?',
  'ትምህርት ቤቱ እንዴት ተጀመረ?',
];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const userCaption = $<HTMLParagraphElement>('user-caption');
const answerCaption = $<HTMLParagraphElement>('answer-caption');
const notice = $<HTMLParagraphElement>('notice');
const statusText = $<HTMLSpanElement>('status-text');
const selfTile = document.querySelector<HTMLElement>('.self')!;
const muteBtn = $<HTMLButtonElement>('mute');
const keyboardBtn = $<HTMLButtonElement>('keyboard');
const textForm = $<HTMLFormElement>('text-form');
const textInput = $<HTMLInputElement>('text-input');
const timerEl = $<HTMLSpanElement>('timer');

const params = new URLSearchParams(location.search);
const kiosk = params.get('mode') === 'kiosk';
// ?bust shows the stand-in bust even when a VRM file exists
const forceBust = params.has('bust');
document.body.classList.toggle('kiosk', kiosk);

const avatar = new AvatarScene($<HTMLCanvasElement>('avatar'));
const player = new SpeechPlayer();
const lipSync = new LipSync();

let assistantName = 'Mekdes';
let history: Turn[] = [];
let controller: AbortController | null = null;
let turnId = 0;
let inCall = false;
let interrupting = false;
let callStarted = 0;
let clock: number | undefined;
let idleTimer: number | undefined;

const detector = new VoiceDetector({
  onSpeechStart() {
    resetIdleTimer();
    if (avatar.state === 'speaking' || avatar.state === 'thinking') {
      // Lower her voice right away; she stops only if the recognizer confirms real words
      interrupting = true;
      player.setVolume(0.2);
    } else {
      setState('listening');
    }
  },
  onUtterance(wav) {
    void handleUtterance(wav);
  },
  onDiscard() {
    if (interrupting) {
      interrupting = false;
      player.setVolume(1);
    } else if (avatar.state === 'listening') {
      setState('idle');
    }
  },
  onLevel(level) {
    selfTile.style.setProperty('--level', level.toFixed(2));
  },
});

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

function setState(state: AvatarState): void {
  avatar.state = state;
  document.body.dataset.state = state;
  detector.strict = state === 'speaking';
  statusText.textContent = {
    idle: 'Listening',
    listening: 'Hearing you…',
    thinking: 'Thinking…',
    speaking: `${assistantName} is speaking`,
  }[state];
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
  interrupting = false;
  controller?.abort();
  controller = null;
  player.stop();
  player.setVolume(1);
}

async function speakOnly(text: string, lang: Lang): Promise<void> {
  const myTurn = turnId;
  answerCaption.textContent = text;
  setState('speaking');
  player.enqueue(synthesize(text, lang));
  await player.whenIdle();
  if (myTurn === turnId && inCall) setState('idle');
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
    if (myTurn === turnId && !interrupting) setState('speaking');
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
  if (myTurn === turnId && inCall) setState('idle');
}

async function handleUtterance(wav: Blob): Promise<void> {
  if (!inCall) return;
  const wasInterrupting = interrupting;
  const myTurn = turnId;
  if (!wasInterrupting) setState('thinking');
  try {
    const { text } = await transcribe(wav, currentLang());
    if (!inCall) return;
    if (!text.trim()) {
      // Just noise: let her carry on, or go back to listening
      if (wasInterrupting) {
        interrupting = false;
        player.setVolume(1);
      } else if (myTurn === turnId) {
        setState('idle');
      }
      return;
    }
    await ask(text);
  } catch (err) {
    showNotice(err instanceof Error ? err.message : String(err));
    if (wasInterrupting) {
      interrupting = false;
      player.setVolume(1);
    } else if (myTurn === turnId) {
      setState('idle');
    }
  }
}

function greeting(lang: Lang): string {
  if (lang === 'am' && assistantName === 'Mekdes') {
    return 'ሰላም! እኔ መቅደስ ነኝ። ስለ ትምህርት ቤታችን የ40 ዓመት ታሪክ ጠይቁኝ።';
  }
  return `Hi, I'm ${assistantName}! Ask me anything about our school's forty years.`;
}

async function startCall(): Promise<void> {
  const startBtn = $<HTMLButtonElement>('start');
  startBtn.disabled = true;
  const label = startBtn.lastChild!;
  label.textContent = ' Allow the microphone…';
  await player.resume();
  try {
    await detector.start();
  } catch (err) {
    showNotice(`Microphone not available: ${err instanceof Error ? err.message : String(err)}`);
    return;
  } finally {
    startBtn.disabled = false;
    label.textContent = ' Start video call';
  }
  if (kiosk) document.documentElement.requestFullscreen?.().catch(() => undefined);
  inCall = true;
  document.body.dataset.call = 'on';
  showNotice('');
  callStarted = Date.now();
  clock = window.setInterval(updateClock, 1000);
  updateClock();
  resetIdleTimer();
  await speakOnly(greeting(currentLang()), currentLang());
}

async function endCall(): Promise<void> {
  inCall = false;
  stopEverything();
  await detector.stop();
  detector.setMuted(false);
  window.clearInterval(clock);
  window.clearTimeout(idleTimer);
  history = [];
  userCaption.textContent = '';
  answerCaption.textContent = '';
  showNotice('');
  muteBtn.setAttribute('aria-pressed', 'false');
  document.body.dataset.muted = 'false';
  document.body.dataset.call = 'off';
  setState('idle');
}

function updateClock(): void {
  const s = Math.floor((Date.now() - callStarted) / 1000);
  timerEl.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

function resetIdleTimer(): void {
  if (!kiosk || !inCall) return;
  window.clearTimeout(idleTimer);
  idleTimer = window.setTimeout(() => void endCall(), KIOSK_IDLE_END_MS);
}

$('start').addEventListener('click', () => void startCall());
$('end').addEventListener('click', () => void endCall());

muteBtn.addEventListener('click', () => {
  const muted = muteBtn.getAttribute('aria-pressed') !== 'true';
  muteBtn.setAttribute('aria-pressed', String(muted));
  muteBtn.setAttribute('aria-label', muted ? 'Unmute microphone' : 'Mute microphone');
  document.body.dataset.muted = String(muted);
  detector.setMuted(muted);
});

keyboardBtn.addEventListener('click', () => {
  textForm.hidden = !textForm.hidden;
  keyboardBtn.setAttribute('aria-pressed', String(!textForm.hidden));
  if (!textForm.hidden) textInput.focus();
});

textForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const q = textInput.value;
  textInput.value = '';
  void ask(q);
});

$('theme').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('theme', next);
});

for (const q of SUGGESTIONS) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.textContent = q;
  btn.addEventListener('click', () => void ask(q));
  $('suggestions').append(btn);
}

if (kiosk) {
  for (const type of ['pointerdown', 'keydown']) document.addEventListener(type, resetIdleTimer);
}

async function init(): Promise<void> {
  setState('idle');
  avatar.start((dt, time) => lipSync.update(player, dt, time));
  const [loaded] = await Promise.all([
    avatar.load(forceBust ? '/models/__none__.vrm' : MODEL_URL),
    health()
      .then((h) => {
        assistantName = h.assistant;
        document.title = `Video call with ${h.assistant} · 40th Anniversary`;
        $('title').textContent = h.assistant;
        $('lobby-name').textContent = h.assistant;
        const school = h.school && h.school !== 'our school' ? h.school : 'School';
        $('subtitle').textContent = `${school} guide · 40th Anniversary`;
        $('disclosure').textContent =
          `${h.assistant} is an AI guide created by the school for its 40th anniversary. ` +
          'Answers come from the school archives and may contain mistakes.';
        const missing = [!h.llm && 'language model', !h.speech && 'speech service'].filter(Boolean);
        if (missing.length) showNotice(`Server is missing configuration: ${missing.join(', ')}.`);
        else if (!h.indexChunks) showNotice('No archive loaded yet; run the ingest step on the server.');
      })
      .catch(() => showNotice('Cannot reach the server.')),
  ]);
  if (!loaded && !kiosk && !forceBust) {
    const note = $<HTMLParagraphElement>('model-note');
    note.hidden = false;
    note.textContent = `Stand-in · add frontend/public${MODEL_URL}`;
  }
}

void init();
