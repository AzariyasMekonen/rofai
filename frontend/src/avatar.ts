import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { VRMLoaderPlugin, VRMUtils, type VRM } from '@pixiv/three-vrm';
import { Blinker } from './character';
import type { MouthShape } from './lipsync';
import { PlaceholderCharacter } from './placeholder';

export type AvatarState = 'idle' | 'listening' | 'thinking' | 'speaking';

export interface Character {
  object: THREE.Object3D;
  /** World-space point the camera should frame (roughly the face). */
  focus: THREE.Vector3;
  update(dt: number, time: number, mouth: MouthShape, state: AvatarState): void;
}

class VrmCharacter implements Character {
  readonly object: THREE.Object3D;
  readonly focus = new THREE.Vector3();
  private readonly blinker = new Blinker();

  constructor(private readonly vrm: VRM, camera: THREE.Camera) {
    this.object = vrm.scene;
    if (vrm.lookAt) vrm.lookAt.target = camera;

    const h = vrm.humanoid;
    VrmCharacter.lowerArm(vrm, 'leftUpperArm', 'leftHand');
    VrmCharacter.lowerArm(vrm, 'rightUpperArm', 'rightHand');
    h.getNormalizedBoneNode('leftLowerArm')?.rotation.set(0, -0.2, 0);
    h.getNormalizedBoneNode('rightLowerArm')?.rotation.set(0, 0.2, 0);

    vrm.update(0);
    // Frame between the eyes and chin rather than the skull base
    const head = h.getRawBoneNode('head');
    head?.getWorldPosition(this.focus);
    this.focus.y += 0.05;
  }

  /** Rotate the arm out of the T-pose, picking the direction that actually puts the hand below the shoulder. */
  private static lowerArm(vrm: VRM, upper: 'leftUpperArm' | 'rightUpperArm', hand: 'leftHand' | 'rightHand'): void {
    const bone = vrm.humanoid.getNormalizedBoneNode(upper);
    const handNode = vrm.humanoid.getRawBoneNode(hand);
    const shoulder = vrm.humanoid.getRawBoneNode(upper);
    if (!bone || !handNode || !shoulder) return;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    bone.rotation.set(0, 0, 1.2);
    vrm.update(0);
    vrm.scene.updateMatrixWorld(true);
    if (handNode.getWorldPosition(a).y > shoulder.getWorldPosition(b).y) bone.rotation.set(0, 0, -1.2);
  }

  update(dt: number, time: number, mouth: MouthShape, state: AvatarState): void {

    const open = Math.min(1, mouth.aa + mouth.oh * 0.8 + mouth.ou * 0.5 + mouth.ih * 0.3 + mouth.ee * 0.3);

    const em = this.vrm.expressionManager;
    if (em) {
      em.setValue('aa', mouth.aa);
      em.setValue('ih', mouth.ih);
      em.setValue('ou', mouth.ou);
      em.setValue('ee', mouth.ee);
      em.setValue('oh', mouth.oh);
      em.setValue('blink', this.blinker.value(dt));
      // Gentle default smile, dynamically softened when speaking to prevent jaw distortion
      const smile = state === 'speaking' ? Math.max(0, 0.2 - open * 0.4) : 0.35;
      em.setValue('happy', smile);
    }

    const h = this.vrm.humanoid;
    const spine = h.getNormalizedBoneNode('spine');
    if (spine) spine.rotation.x = Math.sin(time * 1.2) * 0.015;
    const neck = h.getNormalizedBoneNode('neck');
    if (neck) {
      const tilt = state === 'thinking' ? 0.12 : state === 'listening' ? 0.06 : 0;
      neck.rotation.z = THREE.MathUtils.damp(neck.rotation.z, tilt, 4, dt);
      neck.rotation.y = Math.sin(time * 0.5) * 0.04;
      neck.rotation.x = state === 'speaking' ? Math.sin(time * 2.8) * 0.015 * Math.min(1, open * 1.5) : 0;
    }


    this.vrm.update(dt);
  }
}

export class AvatarScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(26, 1, 0.1, 20);
  private readonly timer = new THREE.Timer();
  private character: Character | null = null;
  state: AvatarState = 'idle';

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();

    const key = new THREE.DirectionalLight(0xfff1dc, 1.6);
    key.position.set(-0.8, 2.4, 2.2);
    const rim = new THREE.DirectionalLight(0xffd36b, 1.2);
    rim.position.set(1.2, 2.0, -1.5);
    this.scene.add(key, rim, new THREE.AmbientLight(0xffffff, 0.15));

    this.timer.connect(document);
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  /** Loads the VRM model; if it is missing or invalid, shows the stand-in bust and returns false. */
  async load(url: string): Promise<boolean> {
    try {
      const res = await fetch(url);
      // Dev servers answer unknown paths with index.html, so check the type as well as the status
      if (!res.ok || (res.headers.get('content-type') ?? '').includes('text/html')) {
        throw new Error(`${url} not found`);
      }
      const loader = new GLTFLoader();
      loader.register((parser) => new VRMLoaderPlugin(parser));
      const gltf = await loader.parseAsync(await res.arrayBuffer(), '');
      const vrm = gltf.userData.vrm as VRM | undefined;
      if (!vrm) throw new Error('file is not a VRM model');
      VRMUtils.rotateVRM0(vrm);
      this.setCharacter(new VrmCharacter(vrm, this.camera));
      return true;
    } catch (err) {
      console.info(`Using stand-in bust: ${err instanceof Error ? err.message : err}`);
      this.setCharacter(new PlaceholderCharacter());
      return false;
    }
  }

  start(getMouth: (dt: number, time: number) => MouthShape): void {
    this.renderer.setAnimationLoop((timestamp) => {
      this.timer.update(timestamp);
      const dt = Math.min(this.timer.getDelta(), 0.1);
      const time = this.timer.getElapsed();
      this.character?.update(dt, time, getMouth(dt, time), this.state);
      this.renderer.render(this.scene, this.camera);
    });
  }

  private setCharacter(character: Character): void {
    if (this.character) this.scene.remove(this.character.object);
    this.character = character;
    this.scene.add(character.object);
    this.frame();
  }

  /** Head-and-shoulders framing that adapts to portrait and landscape canvases. */
  private frame(): void {
    const f = this.character?.focus;
    if (!f) return;
    const distance = this.camera.aspect < 0.9 ? 1.45 : 1.2;
    this.camera.position.set(f.x, f.y + 0.01, f.z + distance);
    this.camera.lookAt(f.x, f.y - 0.08, f.z);
  }

  private resize(): void {
    const { clientWidth: w, clientHeight: h } = this.canvas;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.frame();
  }
}
