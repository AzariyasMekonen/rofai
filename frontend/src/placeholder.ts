import * as THREE from 'three';
import type { AvatarState, Character } from './avatar';
import { Blinker, mouthOpenness } from './character';
import type { MouthShape } from './lipsync';

// Stylized stand-in bust modelled on her reference photo. Units are metres; she faces +Z.
const HEAD_Y = 1.55;
// Torso silhouette (radius, height), spun into a lathe and flattened front-to-back
const TORSO_PROFILE: [number, number][] = [
  [0.0, 1.405],
  [0.06, 1.4],
  [0.12, 1.38],
  [0.18, 1.345],
  [0.215, 1.3],
  [0.232, 1.24],
  [0.238, 1.1],
  [0.24, 0.9],
];
const TORSO_DEPTH = 0.58;

function torsoRadius(y: number): number {
  for (let i = 0; i < TORSO_PROFILE.length - 1; i++) {
    const [r0, y0] = TORSO_PROFILE[i];
    const [r1, y1] = TORSO_PROFILE[i + 1];
    if (y <= y0 && y >= y1) return r0 + ((y0 - y) / (y0 - y1)) * (r1 - r0);
  }
  return TORSO_PROFILE[TORSO_PROFILE.length - 1][0];
}

function tibebTexture(): THREE.CanvasTexture {
  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 256;
  const g = c.getContext('2d')!;
  g.fillStyle = '#f5b800';
  g.fillRect(0, 0, 64, 256);
  g.strokeStyle = '#1b1712';
  g.lineWidth = 4;
  g.beginPath();
  for (let y = 0; y <= 256; y += 32) {
    g.lineTo(10, y);
    g.lineTo(54, y + 16);
  }
  g.stroke();
  g.strokeStyle = '#c8102e';
  g.lineWidth = 3;
  for (let y = 16; y < 256; y += 32) {
    for (const [x, dy] of [
      [18, 8],
      [46, -8],
    ]) {
      g.beginPath();
      g.moveTo(x - 5, y + dy);
      g.lineTo(x + 5, y + dy);
      g.moveTo(x, y + dy - 5);
      g.lineTo(x, y + dy + 5);
      g.stroke();
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(1, 3);
  return tex;
}

/** A strip lying on the torso surface from `top` to `bottom` (x,y pairs). */
function bandGeometry(top: [number, number], bottom: [number, number], width: number, segments = 24): THREE.BufferGeometry {
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const y = top[1] + (bottom[1] - top[1]) * t;
    const cx = top[0] + (bottom[0] - top[0]) * t;
    for (const side of [-0.5, 0.5]) {
      const x = cx + side * width;
      const r = torsoRadius(y) + 0.004;
      positions.push(x, y, TORSO_DEPTH * Math.sqrt(Math.max(r * r - x * x, 0)));
      uvs.push(side + 0.5, t);
    }
    if (i < segments) {
      const a = i * 2;
      indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/** Deterministic pseudo-random so the hairstyle is identical on every load. */
function seeded(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function curlyHair(material: THREE.Material): THREE.InstancedMesh {
  const rand = seeded(1986);
  const curls = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 2), material, 260);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const v = new THREE.Vector3();
  const scale = new THREE.Vector3();
  let count = 0;
  while (count < curls.count) {
    const u = rand() * 2 - 1;
    const a = rand() * Math.PI * 2;
    v.set(Math.sqrt(1 - u * u) * Math.cos(a), u, Math.sqrt(1 - u * u) * Math.sin(a));
    // Leave the face clear and keep the bob above the jaw
    const frontFace = v.z > 0.25 && v.y < 0.62;
    const tooLow = v.y < -0.35 || (v.y < 0.05 && v.z > -0.2 && Math.abs(v.x) < 0.75);
    if (frontFace || tooLow) continue;
    const shell = 0.1 + rand() * 0.012;
    v.set(v.x * shell * 1.2, v.y * shell * 1.18 + 0.008, v.z * shell * 1.05);
    const s = 0.017 + rand() * 0.013;
    q.setFromEuler(e.set(rand() * 3, rand() * 3, rand() * 3));
    m.compose(v, q, scale.set(s, s, s));
    curls.setMatrixAt(count++, m);
  }
  return curls;
}

export class PlaceholderCharacter implements Character {
  readonly object = new THREE.Group();
  readonly focus = new THREE.Vector3(0, HEAD_Y - 0.03, 0);
  private readonly head = new THREE.Group();
  private readonly torso: THREE.Mesh;
  private readonly eyes: THREE.Group[] = [];
  private readonly mouthInner: THREE.Mesh;
  private readonly lips: THREE.Mesh;
  private readonly smile: THREE.Mesh;
  private readonly blinker = new Blinker();

  constructor() {
    const mat = (color: number, roughness: number, extra: THREE.MeshStandardMaterialParameters = {}) =>
      new THREE.MeshStandardMaterial({ color, roughness, ...extra });
    const skin = mat(0x7d4a31, 0.6);
    const skinShade = mat(0x6a3d29, 0.7);
    const hair = mat(0x15110f, 0.9);
    const netela = mat(0xfbf8f0, 0.95, { side: THREE.DoubleSide });
    const dark = mat(0x1a1210, 0.4);
    const white = mat(0xf4efe6, 0.3);

    const torsoGeo = new THREE.LatheGeometry(TORSO_PROFILE.map(([r, y]) => new THREE.Vector2(r, y)), 64);
    torsoGeo.scale(1, 1, TORSO_DEPTH);
    this.torso = new THREE.Mesh(torsoGeo, netela);

    // Gathered folds of the netela around the neck
    const cowl = new THREE.Group();
    for (const [radius, tube, y, tilt] of [
      [0.078, 0.03, 1.405, 0.12],
      [0.09, 0.026, 1.385, -0.08],
      [0.1, 0.022, 1.365, 0.18],
    ] as const) {
      const fold = new THREE.Mesh(new THREE.TorusGeometry(radius, tube, 16, 48), netela);
      fold.rotation.set(Math.PI / 2 + tilt, 0, tilt * 0.6);
      fold.scale.set(1.15, 0.85, 1);
      fold.position.y = y;
      cowl.add(fold);
    }

    // Tibeb border down her left shoulder (viewer's right)
    const band = new THREE.Mesh(
      bandGeometry([0.13, 1.36], [0.075, 0.9], 0.05),
      mat(0xffffff, 0.8, { map: tibebTexture(), side: THREE.DoubleSide }),
    );

    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.043, 0.05, 0.13, 32), skin);
    neck.position.y = 1.44;

    // Egg-shaped head with a tapered jaw
    const headGeo = new THREE.SphereGeometry(0.1, 64, 48);
    const p = headGeo.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i);
      const taper = y < 0 ? 1 - 0.28 * (-y / 0.1) ** 1.5 : 1;
      p.setXYZ(i, p.getX(i) * 0.9 * taper, y * 1.14, p.getZ(i) * 0.97 * (y < 0 ? 0.96 : 1));
    }
    headGeo.computeVertexNormals();
    this.head.add(new THREE.Mesh(headGeo, skin));
    this.head.position.y = HEAD_Y;

    for (const side of [-1, 1]) {
      const ear = new THREE.Mesh(new THREE.SphereGeometry(0.026, 16, 12), skinShade);
      ear.scale.set(0.45, 0.85, 0.6);
      ear.position.set(side * 0.088, -0.005, -0.005);
      const earring = new THREE.Mesh(new THREE.SphereGeometry(0.0065, 16, 12), mat(0xfaf6ee, 0.12, { metalness: 0.1 }));
      earring.position.set(side * 0.09, -0.032, 0.006);

      const eye = new THREE.Group();
      const sclera = new THREE.Mesh(new THREE.SphereGeometry(0.0135, 20, 14), white);
      sclera.scale.set(1.25, 0.8, 0.55);
      const iris = new THREE.Mesh(new THREE.SphereGeometry(0.0075, 16, 12), dark);
      iris.position.z = 0.0055;
      iris.scale.set(1, 1, 0.6);
      eye.add(sclera, iris);
      eye.position.set(side * 0.033, 0.012, 0.081);
      this.eyes.push(eye);

      const brow = new THREE.Mesh(new THREE.CapsuleGeometry(0.0028, 0.026, 4, 8), hair);
      brow.rotation.z = Math.PI / 2 - side * 0.12;
      brow.position.set(side * 0.034, 0.04, 0.084);

      this.head.add(ear, earring, eye, brow);
    }

    const nose = new THREE.Mesh(new THREE.SphereGeometry(0.017, 20, 14), skinShade);
    nose.scale.set(1.05, 0.95, 0.85);
    nose.position.set(0, -0.012, 0.09);
    const bridge = new THREE.Mesh(new THREE.CapsuleGeometry(0.007, 0.03, 4, 8), skin);
    bridge.position.set(0, 0.008, 0.088);

    this.lips = new THREE.Mesh(new THREE.SphereGeometry(0.02, 24, 16), mat(0xb5646c, 0.45));
    this.lips.position.set(0, -0.052, 0.079);
    this.mouthInner = new THREE.Mesh(new THREE.SphereGeometry(0.016, 20, 14), dark);
    this.mouthInner.position.set(0, -0.052, 0.084);
    const teeth = new THREE.Mesh(new THREE.SphereGeometry(0.011, 16, 12), white);
    teeth.scale.set(1.2, 0.35, 0.5);
    teeth.position.set(0, 0.006, 0.004);
    this.mouthInner.add(teeth);
    // Upturned arc under the lips for her resting smile
    this.smile = new THREE.Mesh(new THREE.TorusGeometry(0.026, 0.0024, 8, 32, Math.PI * 0.7), mat(0x8e4148, 0.5));
    this.smile.rotation.z = Math.PI + Math.PI * 0.15;
    this.smile.position.set(0, -0.042, 0.083);
    this.head.add(nose, bridge, this.lips, this.mouthInner, this.smile, curlyHair(hair));

    this.object.add(this.torso, cowl, band, neck, this.head);
  }

  update(dt: number, time: number, mouth: MouthShape, state: AvatarState): void {
    const open = mouthOpenness(mouth);
    this.mouthInner.scale.set(1 + mouth.ee * 0.35 - mouth.ou * 0.3, 0.12 + open * 0.95, 0.6);
    this.lips.scale.set(1.25 + mouth.ee * 0.3 - mouth.ou * 0.35, 0.32 + open * 0.75, 0.5);
    this.smile.visible = open < 0.15;

    const blink = this.blinker.value(dt);
    for (const eye of this.eyes) eye.scale.y = 1 - blink * 0.92;

    const breathe = Math.sin(time * 1.3) * 0.004;
    this.torso.scale.set(1 + breathe, 1, 1 + breathe);

    const tilt = state === 'thinking' ? 0.1 : state === 'listening' ? 0.05 : 0;
    const nod = state === 'speaking' ? Math.sin(time * 2.4) * 0.025 : state === 'thinking' ? -0.06 : 0;
    this.head.rotation.z = THREE.MathUtils.damp(this.head.rotation.z, tilt, 4, dt);
    this.head.rotation.x = THREE.MathUtils.damp(this.head.rotation.x, nod, 5, dt);
    this.head.rotation.y = Math.sin(time * 0.45) * 0.06;
  }
}
