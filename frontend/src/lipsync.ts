import type { Viseme } from './api';
import type { SpeechPlayer } from './audio';

export interface MouthShape {
  aa: number;
  ih: number;
  ou: number;
  ee: number;
  oh: number;
}

const KEYS: (keyof MouthShape)[] = ['aa', 'ih', 'ou', 'ee', 'oh'];

// Azure viseme IDs (0-21) mapped onto the five VRM mouth expressions
const VISEME_SHAPES: Record<number, Partial<MouthShape>> = {
  0: {}, // silence / closed lips
  1: { aa: 0.65 }, // /æ/, /ʌ/, Amharic 1st order (ä)
  2: { aa: 1.0 }, // /ɑ/, Amharic 4th order (a)
  3: { oh: 0.85, aa: 0.2 }, // /ɔ/
  4: { ee: 0.7, aa: 0.15 }, // /e/, Amharic 5th order (e)
  5: { ih: 0.45, ou: 0.3 }, // /ɝ/
  6: { ee: 0.8, ih: 0.3 }, // /i/, Amharic 3rd order (i)
  7: { ou: 0.95 }, // /u/, /w/, Amharic 2nd order (u)
  8: { oh: 0.9, ou: 0.3 }, // /o/, Amharic 7th order (o)
  9: { aa: 0.7, ou: 0.4 }, // /aʊ/
  10: { oh: 0.65, ee: 0.35 }, // /ɔɪ/
  11: { aa: 0.7, ee: 0.3 }, // /aɪ/
  12: { aa: 0.35 }, // /h/
  13: { ou: 0.4, oh: 0.2 }, // /r/
  14: { aa: 0.3, ee: 0.2 }, // /l/
  15: { ih: 0.5, ee: 0.2 }, // /s/, /z/, Amharic sibilants
  16: { ou: 0.4, ih: 0.4 }, // /ʃ/, /tʃ/
  17: { ih: 0.35 }, // /θ/, /ð/
  18: { ih: 0.3, ou: 0.2 }, // /f/, /v/, Amharic ፈ, ቨ
  19: { ih: 0.35, ee: 0.15 }, // /d/, /t/, /n/, Amharic 6th order
  20: { aa: 0.35 }, // /k/, /g/
  21: {}, // /m/, /b/, /p/, bilabial closure (lips pressed together)
};

/** Continuous cosine interpolation between sequential viseme keyframes. */
function sampleViseme(visemes: Viseme[], ms: number): MouthShape {
  const out: MouthShape = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
  if (!visemes.length || ms < visemes[0].t) return out;

  let lo = 0;
  let hi = visemes.length - 1;
  let idx = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (visemes[mid].t <= ms) {
      idx = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  const curr = visemes[idx];
  const currShape = VISEME_SHAPES[curr.id] ?? {};

  // If past the last viseme, fade out to rest over 100ms
  if (idx >= visemes.length - 1) {
    const elapsed = ms - curr.t;
    const fade = Math.max(0, 1 - elapsed / 100);
    for (const k of KEYS) out[k] = (currShape[k] ?? 0) * fade;
    return out;
  }

  const next = visemes[idx + 1];
  const nextShape = VISEME_SHAPES[next.id] ?? {};
  const span = Math.max(1, next.t - curr.t);
  const rawProgress = Math.min(1, Math.max(0, (ms - curr.t) / span));
  // Smooth S-curve (Hermite smoothstep) transition between phonemes
  const t = rawProgress * rawProgress * (3 - 2 * rawProgress);

  for (const k of KEYS) {
    const v0 = currShape[k] ?? 0;
    const v1 = nextShape[k] ?? 0;
    out[k] = v0 + (v1 - v0) * t;
  }
  return out;
}

export class LipSync {
  private readonly shape: MouthShape = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };

  update(player: SpeechPlayer, dt: number, _time: number): MouthShape {
    let target: Partial<MouthShape> = {};
    const clip = player.current;

    if (clip) {
      const ms = (player.ctx.currentTime - clip.start) * 1000;
      const energy = player.level();
      const hasValidVisemes = clip.visemes.length > 0 && clip.visemes.some((v) => v.id > 0);

      if (ms >= 0) {
        if (hasValidVisemes) {
          const sampled = sampleViseme(clip.visemes, ms);
          // Modulate viseme shape by actual audio energy envelope.
          // Closes mouth naturally on pauses between words and clauses!
          const speechGate = Math.min(1.2, Math.max(0, (energy - 0.02) * 2.5));
          for (const key of KEYS) {
            target[key] = sampled[key] * speechGate;
          }
        } else {
          // Spectral formant fallback: drives jaw, rounding, and spread directly from audio bands
          const { low, mid, high } = player.vowelFeatures();
          if (energy > 0.02) {
            const jaw = Math.min(1, low * 1.8 + energy * 0.7);
            const spread = Math.min(1, mid * 1.6);
            const round = Math.min(1, Math.max(0, low * 1.3 - mid * 0.7));
            const fricative = Math.min(0.4, high * 1.1);

            target = {
              aa: jaw * Math.max(0, 1 - spread * 0.4 - round * 0.4),
              ee: spread * 0.75 * energy,
              ih: (spread * 0.35 + fricative * 0.5) * energy,
              oh: round * 0.7 * jaw,
              ou: round * 0.65 * energy,
            };
          }
        }
      }
    }

    // Critical dampening: fast enough for speech consonants, smooth enough to eliminate jitter
    const k = 1 - Math.exp(-dt * 22);
    for (const key of KEYS) {
      this.shape[key] += ((target[key] ?? 0) - this.shape[key]) * k;
    }

    return this.shape;
  }
}

