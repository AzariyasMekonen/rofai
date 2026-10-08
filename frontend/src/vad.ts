import { TARGET_RATE, downsample, encodeWav } from './audio';

const FRAME_MS = 20;
const PREROLL_FRAMES = 15;
const END_SILENCE_MS = 700;
const MIN_SPEECH_MS = 350;
const MAX_UTTERANCE_MS = 20_000;

export interface VoiceDetectorEvents {
  onSpeechStart(): void;
  onUtterance(wav: Blob): void;
  /** Speech started but was too short to be a real utterance (cough, bump, echo). */
  onDiscard(): void;
  onLevel(level: number): void;
}

/**
 * Keeps the microphone open for a whole call and cuts it into utterances with an
 * energy-based voice detector, so only speech is sent for recognition.
 */
export class VoiceDetector {
  muted = false;
  /** Raise the bar while the assistant is talking so her own voice leaking from speakers is less likely to count. */
  strict = false;

  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private frameSize = 960;
  private pending: number[] = [];
  private preroll: Float32Array[] = [];
  private speech: Float32Array[] = [];
  private inSpeech = false;
  private aboveMs = 0;
  private belowMs = 0;
  private speechMs = 0;
  private noiseFloor = 0.004;

  constructor(private readonly events: VoiceDetectorEvents) {}

  get active(): boolean {
    return this.ctx !== null;
  }

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule('/recorder-worklet.js');
    this.frameSize = Math.round((this.ctx.sampleRate * FRAME_MS) / 1000);
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'recorder');
    this.node.port.onmessage = (e: MessageEvent<Float32Array>) => this.push(e.data);
    source.connect(this.node);
    // Worklet writes no output, so this stays silent but keeps the node in the render graph
    this.node.connect(this.ctx.destination);
  }

  async stop(): Promise<void> {
    const ctx = this.ctx;
    this.ctx = null;
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.node = null;
    this.stream = null;
    this.reset();
    await ctx?.close();
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    this.stream?.getAudioTracks().forEach((t) => (t.enabled = !muted));
    if (muted) this.reset();
  }

  private reset(): void {
    this.pending = [];
    this.preroll = [];
    this.speech = [];
    this.inSpeech = false;
    this.aboveMs = this.belowMs = this.speechMs = 0;
  }

  private push(chunk: Float32Array): void {
    for (const s of chunk) this.pending.push(s);
    while (this.pending.length >= this.frameSize) {
      this.processFrame(Float32Array.from(this.pending.splice(0, this.frameSize)));
    }
  }

  private processFrame(frame: Float32Array): void {
    let sum = 0;
    for (const s of frame) sum += s * s;
    const rms = Math.sqrt(sum / frame.length);
    this.events.onLevel(Math.min(1, rms * 10));
    if (this.muted) return;

    if (!this.inSpeech) {
      // Track background noise: follow drops quickly, rises slowly
      const rate = rms < this.noiseFloor ? 0.1 : 0.005;
      this.noiseFloor += (rms - this.noiseFloor) * rate;
    }
    const threshold = Math.max(0.012, this.noiseFloor * 3) * (this.strict ? 2.2 : 1);

    if (!this.inSpeech) {
      this.preroll.push(frame);
      if (this.preroll.length > PREROLL_FRAMES) this.preroll.shift();
      this.aboveMs = rms > threshold ? this.aboveMs + FRAME_MS : 0;
      if (this.aboveMs >= (this.strict ? 260 : 120)) {
        this.inSpeech = true;
        this.speech = this.preroll;
        this.preroll = [];
        this.speechMs = this.aboveMs;
        this.belowMs = 0;
        this.events.onSpeechStart();
      }
      return;
    }

    this.speech.push(frame);
    this.speechMs += FRAME_MS;
    // Hysteresis: once talking, a quieter level still counts as speech
    this.belowMs = rms > threshold * 0.6 ? 0 : this.belowMs + FRAME_MS;
    if (this.belowMs >= END_SILENCE_MS || this.speechMs >= MAX_UTTERANCE_MS) this.finish();
  }

  private finish(): void {
    const frames = this.speech;
    const voicedMs = this.speechMs - this.belowMs;
    this.reset();
    if (!this.ctx) return;
    if (voicedMs < MIN_SPEECH_MS) {
      this.events.onDiscard();
      return;
    }
    const merged = new Float32Array(frames.length * this.frameSize);
    frames.forEach((f, i) => merged.set(f, i * this.frameSize));
    this.events.onUtterance(encodeWav(downsample(merged, this.ctx.sampleRate, TARGET_RATE), TARGET_RATE));
  }
}
