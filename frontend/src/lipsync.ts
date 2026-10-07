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
  0: {},
  1: { aa: 0.6 },
  2: { aa: 1.0 },
  3: { oh: 0.8 },
  4: { ee: 0.6 },
  5: { ih: 0.4, ou: 0.3 },
  6: { ih: 0.7 },
  7: { ou: 0.9 },
  8: { oh: 0.9 },
  9: { aa: 0.7, ou: 0.3 },
  10: { oh: 0.7, ih: 0.3 },
  11: { aa: 0.7, ih: 0.3 },
  12: { aa: 0.3 },
  13: { ou: 0.4 },
  14: { aa: 0.3, ih: 0.2 },
  15: { ih: 0.4 },
  16: { ou: 0.5 },
  17: { ih: 0.3 },
  18: { ih: 0.2 },
  19: { ih: 0.3 },
  20: { aa: 0.3 },
  21: {},
};

function visemeAt(visemes: Viseme[], ms: number): number {
  let lo = 0;
  let hi = visemes.length - 1;
  let found = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (visemes[mid].t <= ms) {
      found = visemes[mid].id;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

export class LipSync {
  private readonly shape: MouthShape = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };

  update(player: SpeechPlayer, dt: number, time: number): MouthShape {
    let target: Partial<MouthShape> = {};
    const clip = player.current;
    if (clip) {
      const ms = (player.ctx.currentTime - clip.start) * 1000;
      if (ms >= 0) {
        if (clip.visemes.length) {
          target = VISEME_SHAPES[visemeAt(clip.visemes, ms)] ?? {};
        } else {
          // No visemes (e.g. Amharic voices): drive the jaw from loudness with slight vowel variation
          const level = player.level();
          const wobble = 0.5 + 0.5 * Math.sin(time * 11);
          target = { aa: level * 0.8, oh: level * 0.3 * wobble, ih: level * 0.2 * (1 - wobble) };
        }
      }
    }
    const k = 1 - Math.exp(-dt * 18);
    for (const key of KEYS) this.shape[key] += ((target[key] ?? 0) - this.shape[key]) * k;
    return this.shape;
  }
}
