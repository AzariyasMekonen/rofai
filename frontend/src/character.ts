import type { MouthShape } from './lipsync';

/** Blink timing shared by every character type. */
export class Blinker {
  private next = 2 + Math.random() * 3;
  private phase = -1;

  value(dt: number): number {
    if (this.phase < 0) {
      this.next -= dt;
      if (this.next <= 0) this.phase = 0;
      return 0;
    }
    this.phase += dt / 0.15;
    if (this.phase >= 1) {
      this.phase = -1;
      this.next = 2.5 + Math.random() * 4;
      return 0;
    }
    return Math.sin(this.phase * Math.PI);
  }
}

export function mouthOpenness(m: MouthShape): number {
  return Math.min(1, m.aa + m.oh * 0.8 + m.ou * 0.5 + m.ih * 0.4 + m.ee * 0.4);
}
