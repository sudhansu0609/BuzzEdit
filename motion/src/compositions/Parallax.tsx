// M6 2.5D parallax (MOTION_GRAPHICS_PLAN.md): presentation/motion_parallax.py
// picks a share of already-generated stills with real depth contrast, runs
// Depth Anything V2 Small (ONNX) over each once, and hands this composition
// the still itself plus a small displacement grid sampled from that depth map
// (`depthGrid`, row-major, row 0 = the BOTTOM of the image -- already flipped
// on the Python side to match @react-three/fiber's PlaneGeometry vertex order,
// so no flip is needed here; see motion_parallax.py::_depth_grid). A small
// virtual camera (@remotion/three's ThreeCanvas, a drop-in for
// react-three-fiber's Canvas synced to Remotion's own frame clock) dollies in
// 3-5% and drifts sideways a little over the displaced plane -- the near
// (bright) parts of the depth map swim past faster than the far (dark) parts,
// which is the whole 2.5D illusion. The plane is rendered 8% larger than the
// frame and the displacement kept modest (MOTION_GRAPHICS_PLAN.md M6) so the
// small camera move never pans the plane's own edge into view.
//
// This is not a `BUILDERS`-registered beat kind like the rest of M1/M2/M7:
// presentation/motion_parallax.py applies it after stills already exist, to a
// share of them (`parallax_share`), not from a script directive -- see
// director.py's hook right before `placement.place_broll`.
import React, { useEffect, useLayoutEffect, useMemo, useState } from "react";
import {
  AbsoluteFill, Easing, continueRender, delayRender, interpolate,
  useCurrentFrame, useVideoConfig,
} from "remotion";
import { ThreeCanvas } from "@remotion/three";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { Theme, withTheme } from "../theme";

export type ParallaxProps = {
  // A data: URI (the still is not under motion/public/, so it travels as a
  // prop rather than a staticFile -- see motion_parallax.py::_data_uri).
  imageSrc?: string;
  // Flat (gridSize+1)^2 array, 0..1, bottom-to-top row-major (see above).
  depthGrid?: number[];
  gridSize?: number;
  // A per-beat seed (hash of the beat id) -- only picks the drift's left/right
  // sign, so back-to-back parallax shots do not all drift the same way.
  seed?: number;
  dollyPercent?: number;
  driftPercent?: number;
  theme?: Partial<Theme>;
};

const FLAT_GRID_SIZE = 8;
const FLAT_DEPTH_GRID = new Array((FLAT_GRID_SIZE + 1) * (FLAT_GRID_SIZE + 1)).fill(0.5);
// A 2x2 dark-grey PNG -- `--selftest` renders every registered composition
// with no real still on hand; this is enough for the scene to mount and
// encode.
const BLANK_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPU0NBgYGBgYgADAAXKAHxvK9IEAAAAAElFTkSuQmCC";

export const parallaxDefaultProps: ParallaxProps = {
  imageSrc: BLANK_IMAGE,
  depthGrid: FLAT_DEPTH_GRID,
  gridSize: FLAT_GRID_SIZE,
  seed: 0,
  dollyPercent: 4,
  driftPercent: 2,
  theme: {},
};

// World units -- arbitrary, only their ratios to each other matter. The
// camera's vertical FOV is derived from these so the *unscaled* plane exactly
// fills the frame at rest; PLANE_MARGIN then grows the plane (not the frame)
// by 8%, which is where the dolly/drift's safety margin comes from.
const CAMERA_DISTANCE = 6;
const PLANE_HEIGHT = 2;
const PLANE_MARGIN = 1.08;
// How far the near/far ends of the depth range are pushed toward/away from
// the camera, in the same world units as PLANE_HEIGHT -- kept modest (6% of
// the plane's own height) so a displaced edge vertex never reaches outside
// the 8% margin above. Tuned against real renders (MOTION_GRAPHICS_PLAN.md M6).
const DEPTH_STRENGTH = PLANE_HEIGHT * 0.06;

function useImageTexture(src: string): THREE.Texture | null {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  const [handle] = useState<number>(() => delayRender("parallax-image"));
  useEffect(() => {
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const tex = new THREE.Texture(img);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      setTexture(tex);
      continueRender(handle);
    };
    img.onerror = () => {
      // No image -- the plane still renders (a flat theme-coloured card),
      // which is a visibly wrong result but never a crashed render; the
      // Python side already falls back to the ordinary Ken Burns still on any
      // render failure, so this path only matters for a malformed data URI.
      if (!cancelled) continueRender(handle);
    };
    img.src = src;
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);
  return texture;
}

function buildGeometry(depthGrid: number[], gridSize: number, aspect: number): THREE.PlaneGeometry {
  const geometry = new THREE.PlaneGeometry(PLANE_HEIGHT * aspect, PLANE_HEIGHT, gridSize, gridSize);
  const position = geometry.attributes.position as THREE.BufferAttribute;
  // A mismatched grid (a bad prop) still renders -- a trailing/missing cell
  // just reads as mid-depth (0.5) -- rather than throwing.
  for (let i = 0; i < position.count; i++) {
    const raw = i < depthGrid.length ? depthGrid[i] : 0.5;
    position.setZ(i, (raw - 0.5) * 2 * DEPTH_STRENGTH);
  }
  position.needsUpdate = true;
  return geometry;
}

const CameraRig: React.FC<{
  frame: number;
  durationInFrames: number;
  fov: number;
  dollyPercent: number;
  driftPercent: number;
  direction: number;
}> = ({ frame, durationInFrames, fov, dollyPercent, driftPercent, direction }) => {
  const { camera } = useThree();
  const t = interpolate(frame, [0, Math.max(1, durationInFrames - 1)], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.ease),
  });
  const z = CAMERA_DISTANCE * (1 - (Math.max(0, dollyPercent) / 100) * t);
  const x = direction * (Math.max(0, driftPercent) / 100) * CAMERA_DISTANCE * t;
  useLayoutEffect(() => {
    camera.position.set(x, 0, z);
    camera.lookAt(0, 0, 0);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.fov = fov;
      camera.updateProjectionMatrix();
    }
  });
  return null;
};

export const Parallax: React.FC<ParallaxProps> = (props) => {
  const frame = useCurrentFrame();
  const { width, height, durationInFrames } = useVideoConfig();
  const theme = withTheme(props.theme);
  const imageSrc = props.imageSrc || BLANK_IMAGE;
  const gridSize = props.gridSize && props.gridSize > 0 ? props.gridSize : FLAT_GRID_SIZE;
  const depthGrid = props.depthGrid && props.depthGrid.length ? props.depthGrid : FLAT_DEPTH_GRID;
  const aspect = width / Math.max(1, height);

  const texture = useImageTexture(imageSrc);
  const geometry = useMemo(() => buildGeometry(depthGrid, gridSize, aspect),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [depthGrid, gridSize, aspect]);
  const fov = useMemo(() => {
    const rad = 2 * Math.atan((PLANE_HEIGHT / 2) / CAMERA_DISTANCE);
    return (rad * 180) / Math.PI;
  }, []);
  // Only the drift's left/right sign varies by seed -- everything else about
  // the move is fixed, so every parallax shot reads the same way.
  const direction = (Math.abs(Math.round(props.seed || 0)) % 2 === 0) ? 1 : -1;

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg }}>
      <ThreeCanvas width={width} height={height}>
        <CameraRig
          frame={frame}
          durationInFrames={durationInFrames}
          fov={fov}
          dollyPercent={props.dollyPercent ?? 4}
          driftPercent={props.driftPercent ?? 2}
          direction={direction}
        />
        <mesh geometry={geometry} scale={[PLANE_MARGIN, PLANE_MARGIN, 1]}>
          {texture ? (
            <meshBasicMaterial map={texture} toneMapped={false} />
          ) : (
            <meshBasicMaterial color={theme.bg} />
          )}
        </mesh>
      </ThreeCanvas>
    </AbsoluteFill>
  );
};
