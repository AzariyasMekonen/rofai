import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, type VRM } from '@pixiv/three-vrm';
import type { MouthShape } from './lipsync';

export type AvatarState = 'idle' | 'listening' | 'thinking' | 'speaking';

interface Character {
  object: THREE.Object3D;
  /** World-space point the camera should frame (roughly the face). */
  focus: THREE.Vector3;
  update(dt: number, time: number, mouth: MouthShape, state: AvatarState): void;
}

/** Blink timing shared by both character types. */
class Blinker {
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

class VrmCharacter implements Character {
  readonly object: THREE.Object3D;
  readonly focus = new THREE.Vector3();
  private readonly blinker = new Blinker();

  constructor(private readonly vrm: VRM, camera: THREE.Camera) {
    this.object = vrm.scene;
    if (vrm.lookAt) vrm.lookAt.target = camera;

    // Relax the T-pose: VRM normalized arms point along ±X, so rotate about Z to lower them
    const h = vrm.humanoid;
    h.getNormalizedBoneNode('leftUpperArm')?.rotation.set(0, 0, -1.2);
    h.getNormalizedBoneNode('rightUpperArm')?.rotation.set(0, 0, 1.2);
    h.getNormalizedBoneNode('leftLowerArm')?.rotation.set(0, -0.2, 0);
    h.getNormalizedBoneNode('rightLowerArm')?.rotation.set(0, 0.2, 0);

    vrm.update(0);
    h.getRawBoneNode('head')?.getWorldPosition(this.focus);
  }

  update(dt: number, time: number, mouth: MouthShape, state: AvatarState): void {
    const em = this.vrm.expressionManager;
    if (em) {
      em.setValue('aa', mouth.aa);
      em.setValue('ih', mouth.ih);
      em.setValue('ou', mouth.ou);
      em.setValue('ee', mouth.ee);
      em.setValue('oh', mouth.oh);
      em.setValue('blink', this.blinker.value(dt));
      // Her default expression is a gentle smile, softened while speaking so mouth shapes stay readable
      em.setValue('happy', state === 'speaking' ? 0.15 : 0.35);
    }

    const h = this.vrm.humanoid;
    const spine = h.getNormalizedBoneNode('spine');
    if (spine) spine.rotation.x = Math.sin(time * 1.2) * 0.015;
    const neck = h.getNormalizedBoneNode('neck');
    if (neck) {
      const tilt = state === 'thinking' ? 0.12 : state === 'listening' ? 0.06 : 0;
      neck.rotation.z = THREE.MathUtils.damp(neck.rotation.z, tilt, 4, dt);
      neck.rotation.y = Math.sin(time * 0.5) * 0.04;
      neck.rotation.x = state === 'speaking' ? Math.sin(time * 2.3) * 0.02 : 0;
    }

    this.vrm.update(dt);
  }
}

/** Simple stand-in bust used until the real VRM model is placed in public/models/. */
class PlaceholderCharacter implements Character {
  readonly object = new THREE.Group();
  readonly focus = new THREE.Vector3(0, 1.45, 0);
  private readonly mouth: THREE.Mesh;
  private readonly eyes: THREE.Mesh[] = [];
  private readonly head = new THREE.Group();
  private readonly blinker = new Blinker();

  constructor() {
    const skin = new THREE.MeshStandardMaterial({ color: 0x8d5a3b, roughness: 0.7 });
    const hair = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 });
    const cloth = new THREE.MeshStandardMaterial({ color: 0xf7f4ec, roughness: 0.9 });
    const tibeb = new THREE.MeshStandardMaterial({ color: 0xf2c200, roughness: 0.6 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x1a1a1a });

    const shawl = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.42, 0.55, 40), cloth);
    shawl.position.y = 1.05;
    const band = new THREE.Mesh(new THREE.TorusGeometry(0.33, 0.035, 12, 48), tibeb);
    band.position.y = 0.98;
    band.rotation.x = Math.PI / 2;
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.07, 0.14, 24), skin);
    neck.position.y = 1.36;

    this.head.position.y = 1.47;
    const face = new THREE.Mesh(new THREE.SphereGeometry(0.115, 40, 32), skin);
    face.scale.set(1, 1.15, 1);
    this.head.add(face);
    for (let i = 0; i < 14; i++) {
      const curl = new THREE.Mesh(new THREE.SphereGeometry(0.055, 16, 12), hair);
      const a = (i / 14) * Math.PI * 1.6 - Math.PI * 0.8;
      curl.position.set(Math.sin(a) * 0.105, 0.07 + Math.cos(a * 0.8) * 0.04, Math.cos(a) * -0.06 - 0.02);
      this.head.add(curl);
    }
    for (const x of [-0.04, 0.04]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), dark);
      eye.position.set(x, 0.02, 0.105);
      this.eyes.push(eye);
      this.head.add(eye);
    }
    this.mouth = new THREE.Mesh(new THREE.CircleGeometry(0.025, 24), dark);
    this.mouth.position.set(0, -0.055, 0.112);
    this.head.add(this.mouth);

    this.object.add(shawl, band, neck, this.head);
  }

  update(dt: number, time: number, mouth: MouthShape, state: AvatarState): void {
    const open = Math.min(1, mouth.aa + mouth.oh * 0.8 + mouth.ou * 0.5 + mouth.ih * 0.4 + mouth.ee * 0.4);
    this.mouth.scale.set(1 + mouth.ee * 0.4 - mouth.ou * 0.3, 0.15 + open * 0.9, 1);
    const blink = this.blinker.value(dt);
    for (const eye of this.eyes) eye.scale.y = 1 - blink * 0.9;
    const tilt = state === 'thinking' ? 0.12 : state === 'listening' ? 0.06 : 0;
    this.head.rotation.z = THREE.MathUtils.damp(this.head.rotation.z, tilt, 4, dt);
    this.head.rotation.y = Math.sin(time * 0.5) * 0.05;
  }
}

export class AvatarScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(28, 1, 0.1, 20);
  private readonly clock = new THREE.Clock();
  private character: Character | null = null;
  state: AvatarState = 'idle';

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a6d3b, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(0.6, 2.2, 2.0);
    this.scene.add(key);

    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  /** Loads the VRM model; on failure shows the placeholder and returns false. */
  async load(url: string): Promise<boolean> {
    try {
      const loader = new GLTFLoader();
      loader.register((parser) => new VRMLoaderPlugin(parser));
      const gltf = await loader.loadAsync(url);
      const vrm = gltf.userData.vrm as VRM | undefined;
      if (!vrm) throw new Error('file is not a VRM model');
      VRMUtils.rotateVRM0(vrm);
      this.setCharacter(new VrmCharacter(vrm, this.camera));
      return true;
    } catch (err) {
      console.warn(`Could not load ${url}; using placeholder.`, err);
      this.setCharacter(new PlaceholderCharacter());
      return false;
    }
  }

  start(getMouth: (dt: number, time: number) => MouthShape): void {
    this.renderer.setAnimationLoop(() => {
      const dt = Math.min(this.clock.getDelta(), 0.1);
      const time = this.clock.elapsedTime;
      this.character?.update(dt, time, getMouth(dt, time), this.state);
      this.renderer.render(this.scene, this.camera);
    });
  }

  private setCharacter(character: Character): void {
    if (this.character) this.scene.remove(this.character.object);
    this.character = character;
    this.scene.add(character.object);
    const f = character.focus;
    // Head-and-shoulders framing
    this.camera.position.set(f.x, f.y - 0.08, f.z + 1.25);
    this.camera.lookAt(f.x, f.y - 0.12, f.z);
  }

  private resize(): void {
    const { clientWidth: w, clientHeight: h } = this.canvas;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
