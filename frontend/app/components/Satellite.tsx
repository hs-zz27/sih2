'use client';

/**
 * The hero satellite (react-three-fiber).
 *
 * Built from primitives rather than a downloaded GLTF: the deck has to boot
 * with no network, and a 4 MB model fetched from a CDN is the one asset that
 * would not. Every material is procedural for the same reason — the
 * environment map and the solar-panel texture are drawn into a 2D canvas at
 * mount and uploaded as textures.
 *
 * Motion respects `prefers-reduced-motion`, and does so *reactively*: framer's
 * `useReducedMotion` subscribes to the media query, so turning the OS setting
 * on mid-session stops the rotation instead of leaving it running until the
 * next reload.
 */

import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { useReducedMotion } from 'framer-motion';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';

const CHAMP = 0xe8c39e;

/**
 * Seconds for one full circuit of the orbit.
 *
 * Slow on purpose. This is ambient: something you notice has moved when you
 * look back, not something that pulls your eye off the headline while you are
 * reading it.
 */
const ORBIT_SECONDS = 52;

/**
 * Half the rig's width in its own model units, before scaling.
 *
 * The wing pivot sits at x = 1.6, the panel is offset a further 1.55 and is
 * 3.1 wide, so the outer edge of a solar panel lands at 1.6 + 1.55 + 1.55.
 * The orbit is clamped by this times the current scale, because clamping the
 * centre point alone still lets a wing hang off the edge of the frame — which
 * is exactly what it did.
 */
const RIG_SPAN = 4.7;

/** How high the orbit rises and falls either side of its centre line. */
const ORBIT_RISE = 0.8;

function canvasTexture(
  width: number,
  height: number,
  draw: (ctx: CanvasRenderingContext2D) => void,
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  draw(canvas.getContext('2d')!);
  return new THREE.CanvasTexture(canvas);
}

/** A tiny equirectangular sky: sun above, terminator, earthshine below. */
function useEnvironment(): THREE.Texture {
  return useMemo(() => {
    const tex = canvasTexture(128, 64, (g) => {
      const grd = g.createLinearGradient(0, 0, 0, 64);
      grd.addColorStop(0.0, '#FFE9CF');
      grd.addColorStop(0.24, '#4A4048');
      grd.addColorStop(0.52, '#14121A');
      grd.addColorStop(0.78, '#2A2028');
      grd.addColorStop(1.0, '#6A4E38');
      g.fillStyle = grd;
      g.fillRect(0, 0, 128, 64);
      const sun = g.createRadialGradient(96, 12, 0, 96, 12, 26);
      sun.addColorStop(0, '#FFFFFF');
      sun.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = sun;
      g.fillRect(0, 0, 128, 64);
    });
    tex.mapping = THREE.EquirectangularReflectionMapping;
    return tex;
  }, []);
}

function Rig({ still }: { still: boolean }) {
  const envMap = useEnvironment();
  const rig = useRef<THREE.Group>(null);
  const body = useRef<THREE.Group>(null);
  const wings = useRef<THREE.Group[]>([]);
  const pings = useRef<THREE.Mesh[]>([]);
  const stars = useRef<THREE.Points>(null);
  const pointer = useRef({ x: 0, y: 0 });
  const { size, camera, scene } = useThree();

  useEffect(() => {
    scene.environment = envMap;
    return () => {
      scene.environment = null;
    };
  }, [scene, envMap]);

  const materials = useMemo(() => {
    const panel = canvasTexture(256, 128, (g) => {
      g.fillStyle = '#20202C';
      g.fillRect(0, 0, 256, 128);
      g.strokeStyle = 'rgba(232,195,158,0.30)';
      g.lineWidth = 1;
      for (let x = 0; x <= 256; x += 16) {
        g.beginPath();
        g.moveTo(x + 0.5, 0);
        g.lineTo(x + 0.5, 128);
        g.stroke();
      }
      for (let y = 0; y <= 128; y += 16) {
        g.beginPath();
        g.moveTo(0, y + 0.5);
        g.lineTo(256, y + 0.5);
        g.stroke();
      }
    });
    return {
      metal: new THREE.MeshStandardMaterial({
        color: 0x38323f,
        metalness: 0.62,
        roughness: 0.34,
        envMap,
        envMapIntensity: 1.3,
      }),
      foil: new THREE.MeshStandardMaterial({
        color: 0x9a7150,
        metalness: 0.75,
        roughness: 0.26,
        envMap,
        envMapIntensity: 1.6,
      }),
      glass: new THREE.MeshStandardMaterial({
        map: panel,
        metalness: 0.55,
        roughness: 0.3,
        envMap,
        envMapIntensity: 1.2,
      }),
      dish: new THREE.MeshStandardMaterial({
        color: 0x565060,
        metalness: 0.5,
        roughness: 0.45,
        side: THREE.DoubleSide,
        envMap,
        envMapIntensity: 1.2,
      }),
      tip: new THREE.MeshStandardMaterial({
        color: CHAMP,
        metalness: 0.6,
        roughness: 0.25,
        envMap,
        envMapIntensity: 1.6,
      }),
    };
  }, [envMap]);

  const starPositions = useMemo(() => {
    const positions: number[] = [];
    for (let i = 0; i < 240; i++) {
      const r = 26 + Math.random() * 22;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions.push(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        -Math.abs(r * Math.cos(phi)) * 0.7,
      );
    }
    return new Float32Array(positions);
  }, []);

  // The rig crosses the hero rather than hovering in one spot, so it reads as
  // something in orbit rather than a model on a turntable. It still climbs a
  // little as it goes, and sits higher on a narrow viewport where the copy
  // runs full width.
  const wide = size.width > 900;
  const baseY = wide ? 1.05 : 1.85;
  const scale = wide ? 0.62 : 0.42;

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      pointer.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      pointer.current.y = (e.clientY / window.innerHeight) * 2 - 1;
    };
    window.addEventListener('pointermove', onMove, { passive: true });
    return () => window.removeEventListener('pointermove', onMove);
  }, []);

  /**
   * Half the visible world width at the rig's depth, from the camera itself.
   *
   * Derived rather than hardcoded: the hero is a wide, shallow band whose
   * aspect changes with the viewport, and a fixed number would either stop
   * the pass short of the edge on a wide monitor or drag it far off-screen on
   * a narrow one.
   */
  const halfWidth = () => {
    const cam = camera as THREE.PerspectiveCamera;
    return Math.tan(((cam.fov * Math.PI) / 180) / 2) * cam.position.z * cam.aspect;
  };

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (!still) {
      if (body.current) {
        body.current.rotation.y = t * 0.055 + Math.sin(t * 0.19) * 0.16;
        body.current.rotation.x = Math.sin(t * 0.23) * 0.05;
      }
      wings.current.forEach((w, i) => {
        if (w) w.rotation.x = Math.sin(t * 0.15 + i * 0.5) * 0.2;
      });
      // Downlink pings: three rings expanding out of the dish, on a loop.
      pings.current.forEach((ring, i) => {
        if (!ring) return;
        const p = (t * 0.34 + i / 3) % 1;
        const s = 1 + p * 3.6;
        ring.scale.set(s, s, s);
        ring.position.z = 1.8 + p * 1.5;
        (ring.material as THREE.MeshBasicMaterial).opacity = 0.3 * Math.pow(1 - p, 1.7);
      });
      if (stars.current) stars.current.rotation.y = t * 0.006;

      if (rig.current) {
        /* A closed orbit inside the right-hand region, not a pass that exits.
           A traverse has to leave the frame to loop, which reads as the
           satellite disappearing rather than as motion; an ellipse never
           stops and never goes anywhere it cannot be seen. Both limits are
           derived from the frustum, so the path resizes with the viewport
           instead of being tuned to one window. */
        const half = halfWidth();
        const right = half - RIG_SPAN * scale;
        // On a wide viewport the headline owns the left column, so the orbit
        // starts right of centre. On a narrow one the copy runs full width
        // and there is no such column to keep clear.
        const left = wide ? half * 0.06 : -half * 0.3;
        const centre = (right + left) / 2;
        const radius = Math.max(0.4, (right - left) / 2);

        const theta = (t / ORBIT_SECONDS) * Math.PI * 2;
        rig.current.position.x = centre + Math.cos(theta) * radius;
        rig.current.position.y =
          baseY + Math.sin(theta) * ORBIT_RISE + Math.sin(t * 0.42) * 0.12;

        rig.current.rotation.x = 0.22 + Math.sin(t * 0.35) * 0.03 - pointer.current.y * 0.07;
        rig.current.rotation.z = 0.14 + Math.cos(t * 0.28) * 0.025;
      }
      camera.position.x += (pointer.current.x * 0.9 - camera.position.x) * 0.03;
      camera.position.y += (-pointer.current.y * 0.35 - camera.position.y) * 0.03;
      camera.lookAt(0, 0, 0);
    } else if (rig.current) {
      // Reduced motion: parked on the orbit, no travel at all.
      const half = halfWidth();
      const right = half - RIG_SPAN * scale;
      const left = wide ? half * 0.06 : -half * 0.3;
      rig.current.position.x = (right + left) / 2;
      rig.current.position.y = baseY + ORBIT_RISE;
    }
  });

  return (
    <>
      <ambientLight color={0x3e3945} intensity={0.7} />
      <directionalLight color={0xffe6c8} intensity={2.1} position={[6, 6.5, 6]} />
      <directionalLight color={0x8fa0c8} intensity={0.85} position={[-7, -3, -5]} />

      <points ref={stars}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[starPositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          color={0xf2ede6}
          size={0.13}
          transparent
          opacity={0.45}
          sizeAttenuation
        />
      </points>

      <group
        ref={rig}
        position={[0, baseY, 0]}
        rotation={[0.22, -0.5, 0.14]}
        scale={scale}
      >
        <group ref={body}>
          {/* bus */}
          <mesh material={materials.metal}>
            <boxGeometry args={[1.5, 1.5, 2.1]} />
          </mesh>
          <mesh position={[0, -0.34, 0]} material={materials.foil}>
            <boxGeometry args={[1.54, 0.42, 2.14]} />
          </mesh>

          {/* solar wings, one arm and one panel each side */}
          {[-1, 1].map((side, i) => (
            <group key={side}>
              <mesh
                position={[side * 1.15, 0, 0]}
                rotation={[0, 0, Math.PI / 2]}
                material={materials.metal}
              >
                <cylinderGeometry args={[0.045, 0.045, 0.9, 10]} />
              </mesh>
              <group
                ref={(el) => {
                  if (el) wings.current[i] = el;
                }}
                position={[side * 1.6, 0, 0]}
              >
                <mesh position={[side * 1.55, 0, 0]} material={materials.glass}>
                  <boxGeometry args={[3.1, 0.05, 1.35]} />
                </mesh>
              </group>
            </group>
          ))}

          {/* downlink dish and its pings */}
          <mesh
            position={[0, -0.35, 1.5]}
            rotation={[Math.PI * 0.62, 0, 0]}
            material={materials.dish}
          >
            <sphereGeometry args={[0.62, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2.6]} />
          </mesh>
          {[0, 1, 2].map((i) => (
            <mesh
              key={i}
              ref={(el) => {
                if (el) pings.current[i] = el;
              }}
              position={[0, -0.35, 1.8]}
            >
              <ringGeometry args={[0.26, 0.29, 48]} />
              <meshBasicMaterial
                color={CHAMP}
                transparent
                opacity={0}
                side={THREE.DoubleSide}
                blending={THREE.AdditiveBlending}
                depthWrite={false}
              />
            </mesh>
          ))}

          {/* magnetometer boom */}
          <mesh
            position={[0.45, 1.6, -0.3]}
            rotation={[0, 0, -0.16]}
            material={materials.foil}
          >
            <cylinderGeometry args={[0.028, 0.028, 1.7, 8]} />
          </mesh>
          <mesh position={[0.32, 2.42, -0.3]} material={materials.tip}>
            <sphereGeometry args={[0.085, 14, 10]} />
          </mesh>
        </group>
      </group>
    </>
  );
}

export default function Satellite() {
  const reduce = useReducedMotion();
  const [onScreen, setOnScreen] = useState(true);
  const holder = useRef<HTMLDivElement>(null);

  /**
   * Stop rendering once the hero has scrolled away.
   *
   * A WebGL scene driving 60 frames a second competes with the compositor for
   * every one of them, and it was still doing that from behind three
   * screenfuls of deck. Parking the frameloop while the hero is off screen
   * gives the scroll the main thread back, and the scene picks up exactly
   * where it left off when the hero returns — `elapsedTime` keeps running, so
   * the orbit does not jump.
   */
  useEffect(() => {
    const el = holder.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => setOnScreen(entry.isIntersecting),
      { rootMargin: '120px' },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={holder} style={{ position: 'absolute', inset: 0 }}>
      <Canvas
        aria-hidden="true"
        camera={{ fov: 38, position: [0, 0, 12], near: 0.1, far: 200 }}
        dpr={[1, 2]}
        gl={{ alpha: true, antialias: true }}
        // A still scene needs one frame, not sixty per second — and a scene
        // nobody can see needs none at all.
        frameloop={reduce || !onScreen ? 'demand' : 'always'}
      >
        <Rig still={Boolean(reduce)} />
      </Canvas>
    </div>
  );
}
