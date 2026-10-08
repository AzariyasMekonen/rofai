import type { TtsResult, Viseme } from './api';

export const TARGET_RATE = 16000;

export function downsample(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}

export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

export interface PlayingClip {
  start: number;
  visemes: Viseme[];
}

/** Plays synthesized sentences strictly in order while exposing timing for lip-sync. */
export class SpeechPlayer {
  readonly ctx = new AudioContext();
  private readonly analyser: AnalyserNode;
  private readonly gain: GainNode;
  private readonly samples: Float32Array<ArrayBuffer>;
  private queue: Promise<void> = Promise.resolve();
  private sources = new Set<AudioBufferSourceNode>();
  private generation = 0;
  current: PlayingClip | null = null;

  constructor() {
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.samples = new Float32Array(this.analyser.fftSize);
    this.gain = this.ctx.createGain();
    this.analyser.connect(this.gain).connect(this.ctx.destination);
  }

  setVolume(volume: number): void {
    this.gain.gain.setTargetAtTime(volume, this.ctx.currentTime, 0.05);
  }

  resume(): Promise<void> {
    return this.ctx.resume();
  }

  enqueue(clip: Promise<TtsResult>): Promise<void> {
    const gen = this.generation;
    this.queue = this.queue.then(async () => {
      const result = await clip.catch(() => null);
      if (gen !== this.generation || !result?.audio) return;
      const buffer = await this.ctx.decodeAudioData(base64ToArrayBuffer(result.audio));
      if (gen !== this.generation) return;
      await this.play(buffer, result.visemes);
    });
    return this.queue;
  }

  whenIdle(): Promise<void> {
    return this.queue;
  }

  stop(): void {
    this.generation++;
    for (const src of this.sources) src.stop();
    this.sources.clear();
    this.current = null;
    this.queue = Promise.resolve();
  }

  /** Current output loudness, 0..1. */
  level(): number {
    this.analyser.getFloatTimeDomainData(this.samples);
    let sum = 0;
    for (const s of this.samples) sum += s * s;
    const rms = Math.sqrt(sum / this.samples.length);
    return Math.min(1, Math.max(0, (rms - 0.01) * 6));
  }

  private play(buffer: AudioBuffer, visemes: Viseme[]): Promise<void> {
    return new Promise((resolve) => {
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(this.analyser);
      const start = this.ctx.currentTime + 0.05;
      const clip = { start, visemes };
      this.current = clip;
      src.onended = () => {
        this.sources.delete(src);
        if (this.current === clip) this.current = null;
        resolve();
      };
      this.sources.add(src);
      src.start(start);
    });
  }
}
