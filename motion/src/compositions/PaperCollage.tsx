// `collage` beat -> presentation/motion_collage.py::build_paper_collage ->
// this composition (MOTION_GRAPHICS_PLAN.md M5). 2-4 already-generated
// stills, each either a clean cutout (RGBA PNG; drawn with a white-outline +
// drop-shadow "sticker" look) or -- when the matting model found no usable
// subject -- a plain photo (`cutout: false`, drawn as a polaroid card
// instead), step in one by one onto an off-white paper background, landing
// by ~40% of the clip (Architecture) with a slight rotation, then settle
// into a small stop-motion "on twos" jitter. 1-2 hand-drawn arrows/circles
// (`@remotion/paths`' `evolvePath`, the same technique MotionChart's line
// draw uses) point at a couple of the cutouts, and the caption is a
// torn-paper sticker label in the creator's handwriting font
// (fonts.ts::handFontStack -- the same face StickScene's caption uses).
import React, { useMemo } from "react";
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { evolvePath } from "@remotion/paths";
import { fontFaceCss, handFontStack } from "../fonts";
import { Theme, withTheme } from "../theme";

export type CollageImage = {
  // A data: URI -- the still lives outside motion/public/, so it travels as
  // a prop rather than through staticFile (the way Parallax's still does).
  src: string;
  // True: an RGBA cutout (presentation/motion_collage.py's IS-Net matte) --
  // drawn as a die-cut sticker. False: a plain rectangular photo (the mask
  // was missing or unusable) -- drawn as a polaroid card instead.
  cutout: boolean;
  aspect: number; // width / height
};

export type PaperCollageProps = {
  images?: CollageImage[];
  caption?: string;
  // A faint engineering-paper grid under the fibre texture, for the
  // explainer art style (presentation/motion_collage.py::build_paper_collage).
  grid?: boolean;
  seed?: number;
  theme?: Partial<Theme>;
};

// A 2x2 dark-grey PNG -- `--selftest` renders every registered composition
// with no real stills on hand; this is enough for the scene to mount.
const BLANK_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPU0NBgYGBgYgADAAXKAHxvK9IEAAAAAElFTkSuQmCC";

export const paperCollageDefaultProps: PaperCollageProps = {
  images: [
    { src: BLANK_IMAGE, cutout: false, aspect: 1.5 },
    { src: BLANK_IMAGE, cutout: false, aspect: 0.9 },
    { src: BLANK_IMAGE, cutout: false, aspect: 1.2 },
  ],
  caption: "",
  grid: false,
  seed: 0,
  theme: {},
};

// --- deterministic "hand-drawn" jitter --------------------------------------
// Its own small copy of the idea StickScene.tsx's boilJitter already uses
// (a seeded offset re-rolled every 2 frames = stop-motion "on twos") --
// kept local since that file is edited in parallel elsewhere.
function hashUnit(n: number): number {
  const s = Math.sin(n) * 43758.5453123;
  return s - Math.floor(s);
}
function jitter(seed: number, bucket: number, amp: number): { dx: number; dy: number; dr: number } {
  const h1 = hashUnit(seed * 12.9898 + bucket * 78.233);
  const h2 = hashUnit(seed * 39.346 + bucket * 11.135 + 7.5);
  const h3 = hashUnit(seed * 93.989 + bucket * 4.141 + 3.1);
  return { dx: (h1 - 0.5) * 2 * amp, dy: (h2 - 0.5) * 2 * amp, dr: (h3 - 0.5) * 2 * (amp * 0.6) };
}

// --- layout: one scrapbook template per count, varied a little by seed -----
// Review (MOTION_GRAPHICS_PLAN.md M5): a real render + frame inspection
// found the items clustered small in the top half with the bottom half bare
// paper. These boxes run 35-50% of the frame height -- 2 items largest
// (near 50%), 4 items smallest (near 35%) -- and their centres are spread
// across the frame by thirds (left/mid/right column, top/mid row), with
// every bottom-row box kept clear of the caption's own strip
// (TornCaption sits roughly in the bottom 19% -- y 0.81-0.94).
type Slot = { cx: number; cy: number; boxW: number; boxH: number; rot: number };
const LAYOUTS: Record<number, Slot[]> = {
  2: [
    { cx: 0.28, cy: 0.46, boxW: 0.36, boxH: 0.50, rot: -6 },
    { cx: 0.72, cy: 0.46, boxW: 0.36, boxH: 0.50, rot: 5 },
  ],
  3: [
    { cx: 0.22, cy: 0.26, boxW: 0.32, boxH: 0.40, rot: -7 },
    { cx: 0.62, cy: 0.60, boxW: 0.34, boxH: 0.38, rot: 6 },
    { cx: 0.80, cy: 0.42, boxW: 0.28, boxH: 0.36, rot: -4 },
  ],
  4: [
    { cx: 0.22, cy: 0.24, boxW: 0.30, boxH: 0.36, rot: -8 },
    { cx: 0.76, cy: 0.22, boxW: 0.30, boxH: 0.36, rot: 5 },
    { cx: 0.26, cy: 0.60, boxW: 0.30, boxH: 0.36, rot: 6 },
    { cx: 0.72, cy: 0.58, boxW: 0.30, boxH: 0.36, rot: -5 },
  ],
};

function slotsFor(n: number, seed: number): Slot[] {
  const base = LAYOUTS[Math.max(2, Math.min(4, n))] || LAYOUTS[3];
  return base.map((s, i) => {
    const h1 = hashUnit(seed * 7.13 + i * 2.7);
    const h2 = hashUnit(seed * 3.31 + i * 5.9 + 1.1);
    return { ...s, cx: s.cx + (h1 - 0.5) * 0.03, cy: s.cy + (h2 - 0.5) * 0.03, rot: s.rot + (h1 - 0.5) * 6 };
  });
}

function fitSize(boxW: number, boxH: number, aspect: number): { w: number; h: number } {
  let w = boxW;
  let h = boxW / Math.max(0.2, aspect);
  if (h > boxH) {
    h = boxH;
    w = boxH * Math.max(0.2, aspect);
  }
  return { w, h };
}

function polaroidSize(boxW: number, boxH: number, aspect: number) {
  const border = Math.max(8, Math.min(boxW, boxH) * 0.05);
  const bottom = border * 2.4;
  const availW = Math.max(10, boxW - border * 2);
  const availH = Math.max(10, boxH - border - bottom);
  const { w: imgW, h: imgH } = fitSize(availW, availH, aspect);
  return { imgW, imgH, border, cardW: imgW + border * 2, cardH: imgH + border + bottom };
}

// The white sticker outline is baked into the cutout PNG itself
// (presentation/motion_collage.py::_add_outline) -- a real render of it
// timed out at 1920x1080 when this was instead a ring of ~10 stacked
// `drop-shadow()` filters recomputed by Chromium every frame over a full-size
// image (measured 2026-10-28). Only the floating soft shadow is still a
// (single, cheap) filter here.
function stickerShadow(): string {
  return "drop-shadow(0 10px 16px rgba(20,14,4,0.38))";
}

// --- paper background: off-white, a fibre-noise texture, an optional light
// engineering-paper grid (explainer art style) ------------------------------
const PAPER_BG = "#F2E9D2";
const PaperTexture: React.FC<{ grid?: boolean }> = ({ grid }) => (
  <>
    <AbsoluteFill style={{ opacity: 0.1, mixBlendMode: "multiply" }}>
      <svg width="100%" height="100%">
        <filter id="collage-fibre">
          <feTurbulence type="fractalNoise" baseFrequency="0.018" numOctaves="3" seed={11} />
          <feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.55 0" />
        </filter>
        <rect width="100%" height="100%" fill="#7A6A42" filter="url(#collage-fibre)" />
      </svg>
    </AbsoluteFill>
    {grid && (
      <AbsoluteFill style={{ opacity: 0.16 }}>
        <svg width="100%" height="100%">
          <pattern id="collage-grid" width={48} height={48} patternUnits="userSpaceOnUse">
            <path d="M 48 0 L 0 0 0 48" fill="none" stroke="#6F8FA8" strokeWidth={1} />
          </pattern>
          <rect width="100%" height="100%" fill="url(#collage-grid)" />
        </svg>
      </AbsoluteFill>
    )}
  </>
);

// --- doodled arrows/circles pointing at a couple of the cutouts ------------
function circlePath(cx: number, cy: number, rx: number, ry: number, seed: number): string {
  const steps = 20;
  const pts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const a = (i / steps) * Math.PI * 2 - Math.PI / 2;
    const wobble = 1 + (hashUnit(seed * 5.1 + i * 1.7) - 0.5) * 0.12;
    const x = cx + Math.cos(a) * rx * wobble;
    const y = cy + Math.sin(a) * ry * wobble;
    pts.push(`${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`);
  }
  return pts.join(" ");
}

function arrowPath(x1: number, y1: number, x2: number, y2: number, seed: number): string {
  const mx = (x1 + x2) / 2 + (hashUnit(seed * 3.3) - 0.5) * 40;
  const my = (y1 + y2) / 2 + (hashUnit(seed * 7.7) - 0.5) * 40;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 16;
  const a1 = angle + Math.PI * 0.8, a2 = angle - Math.PI * 0.8;
  const hx1 = x2 + Math.cos(a1) * headLen, hy1 = y2 + Math.sin(a1) * headLen;
  const hx2 = x2 + Math.cos(a2) * headLen, hy2 = y2 + Math.sin(a2) * headLen;
  return `M ${x1.toFixed(1)} ${y1.toFixed(1)} Q ${mx.toFixed(1)} ${my.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)} `
       + `M ${hx1.toFixed(1)} ${hy1.toFixed(1)} L ${x2.toFixed(1)} ${y2.toFixed(1)} L ${hx2.toFixed(1)} ${hy2.toFixed(1)}`;
}

// An item's own on-screen bounding box (the actual fitted/polaroid size,
// not its layout slot -- a slot is often bigger than what lands inside it).
type ItemRect = { cx: number; cy: number; w: number; h: number };

// Where a ray from `rect`'s centre, aimed at (dirX, dirY), crosses the
// rectangle's own edge -- used both to make the arrow's tip stop just short
// of a specific item (Review finding 3) and, implicitly, to keep it aimed
// from the correct side.
function boxEdgePoint(rect: ItemRect, dirX: number, dirY: number): { x: number; y: number } {
  const halfW = rect.w / 2, halfH = rect.h / 2;
  const len = Math.hypot(dirX, dirY) || 1;
  const ux = dirX / len, uy = dirY / len;
  const tx = Math.abs(ux) > 1e-6 ? halfW / Math.abs(ux) : Infinity;
  const ty = Math.abs(uy) > 1e-6 ? halfH / Math.abs(uy) : Infinity;
  const t = Math.min(tx, ty);
  return { x: rect.cx + ux * t, y: rect.cy + uy * t };
}

// Review (MOTION_GRAPHICS_PLAN.md M5): a real render + frame inspection
// found the circle sized off a slot far bigger than what actually landed in
// it (loose, not hugging anything) and the arrow starting off-frame and
// landing on top of its target instead of short of it. Both now work off
// the items' own real on-screen rects: the circle is that item's bbox
// +6-10%, and the arrow starts near the caption (or, with none, between the
// two items it runs between) and stops 10-30px short of its target's edge.
const Doodles: React.FC<{
  itemRects: ItemRect[]; width: number; height: number; frame: number;
  landFrame: number; durationInFrames: number; accent: string; seed: number; hasCaption: boolean;
}> = ({ itemRects, width, height, frame, landFrame, durationInFrames, accent, seed, hasCaption }) => {
  const count = itemRects.length;
  const picks = Math.min(2, count);
  // TornCaption's own geometry: bottom: height*0.06, ~0.13 of the height
  // tall (padding + one line) -- its approximate on-screen centre.
  const captionPoint = { x: width * 0.5, y: height * 0.875 };
  const items: React.ReactNode[] = [];
  for (let k = 0; k < picks; k++) {
    const startF = landFrame + k * 10;
    const endF = Math.min(durationInFrames - 6, startF + 26);
    const progress = interpolate(frame, [startF, endF], [0, 1],
      { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
    let d: string;
    if (k % 2 === 0) {
      const rect = itemRects[0];
      const margin = 1.06 + hashUnit(seed * 5.1 + k) * 0.04; // +6-10%
      d = circlePath(rect.cx, rect.cy, (rect.w / 2) * margin, (rect.h / 2) * margin, seed + k);
    } else {
      const rect = itemRects[Math.min(count - 1, 1)];
      const other = itemRects[0];
      const start = hasCaption ? captionPoint : { x: (rect.cx + other.cx) / 2, y: (rect.cy + other.cy) / 2 };
      const edge = boxEdgePoint(rect, start.x - rect.cx, start.y - rect.cy);
      const gap = 10 + hashUnit(seed * 9.3 + k) * 20; // 10-30px short of the edge
      const ex = edge.x - rect.cx, ey = edge.y - rect.cy;
      const elen = Math.hypot(ex, ey) || 1;
      const tipX = edge.x - (ex / elen) * gap;
      const tipY = edge.y - (ey / elen) * gap;
      d = arrowPath(start.x, start.y, tipX, tipY, seed + k);
    }
    items.push(
      <path key={k} d={d} stroke={accent} strokeWidth={5} fill="none"
        strokeLinecap="round" strokeLinejoin="round" {...evolvePath(progress, d)} />,
    );
  }
  return <>{items}</>;
};

// --- caption: a torn-paper sticker label in the handwriting font -----------
const TORN_CLIP = "polygon(0% 8%, 4% 0%, 9% 6%, 15% 1%, 21% 7%, 28% 0%, 35% 5%, 42% 1%, 50% 6%, 58% 0%,"
  + " 65% 5%, 72% 1%, 79% 6%, 86% 0%, 92% 5%, 100% 0%, 100% 92%, 95% 100%, 88% 94%, 81% 100%, 74% 95%,"
  + " 66% 100%, 59% 94%, 51% 100%, 44% 95%, 36% 100%, 29% 94%, 22% 100%, 14% 95%, 7% 100%, 0% 94%)";

const TornCaption: React.FC<{
  text: string; width: number; height: number; frame: number; fps: number; landFrame: number;
}> = ({ text, width, height, frame, fps, landFrame }) => {
  const local = Math.max(0, frame - landFrame * 0.9);
  const a = spring({ frame: local, fps, config: { damping: 14, mass: 0.6, stiffness: 120 } });
  return (
    <div style={{
      position: "absolute", left: "50%", bottom: height * 0.06,
      transform: `translateX(-50%) scale(${interpolate(a, [0, 1], [0.8, 1])}) rotate(-2deg)`,
      opacity: a, maxWidth: width * 0.64,
      background: "#FBF6E8", clipPath: TORN_CLIP,
      padding: `${height * 0.035}px ${width * 0.045}px`,
      boxShadow: "0 10px 18px rgba(20,14,4,0.3)",
    }}>
      <div style={{
        fontFamily: handFontStack(), fontWeight: 400, fontSize: height * 0.05,
        color: "#2A2012", textAlign: "center",
      }}>
        {text}
      </div>
    </div>
  );
};

// --- the composition ---------------------------------------------------------
export const PaperCollage: React.FC<PaperCollageProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const images = (props.images && props.images.length >= 2
    ? props.images : paperCollageDefaultProps.images!).slice(0, 4);
  const seed = props.seed || 0;
  const slots = useMemo(() => slotsFor(images.length, seed), [images.length, seed]);
  const caption = (props.caption || "").trim();
  // Each item's own real on-screen rect (not its layout slot, which is
  // usually bigger than what actually fits inside it) -- the doodles hug
  // and aim at these (Review finding 3).
  const itemRects = useMemo(() => images.map((img, i) => {
    const slot = slots[i];
    const boxW = slot.boxW * width, boxH = slot.boxH * height;
    const cx = slot.cx * width, cy = slot.cy * height;
    if (img.cutout) {
      const { w, h } = fitSize(boxW, boxH, img.aspect || 1);
      return { cx, cy, w, h };
    }
    const pz = polaroidSize(boxW, boxH, img.aspect || 1);
    return { cx, cy, w: pz.cardW, h: pz.cardH };
  }), [images, slots, width, height]);
  // The doodles are a red-pen mark on paper, not a themed accent -- some
  // art styles' own accent is white (explainer's card colour, meant for a
  // dark card elsewhere) and would be invisible against this composition's
  // always-off-white paper (found by a real render + frame inspection,
  // 2026-10-28).
  const doodleColor = "#B6392B";

  // Landed by ~40% of the clip (Architecture); each image's own entrance
  // starts a little after the previous one, every one settled well before
  // the last image's own spring finishes.
  const landFrame = Math.max(10, Math.round(durationInFrames * 0.4));
  const perImage = (landFrame * 0.7) / Math.max(1, images.length);

  return (
    <AbsoluteFill style={{ backgroundColor: PAPER_BG, overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>
      <PaperTexture grid={props.grid} />
      <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
        <Doodles itemRects={itemRects} width={width} height={height} frame={frame}
          landFrame={landFrame} durationInFrames={durationInFrames} accent={doodleColor}
          seed={seed} hasCaption={!!caption} />
      </svg>
      {images.map((img, i) => {
        const slot = slots[i];
        const boxW = slot.boxW * width, boxH = slot.boxH * height;
        const startFrame = i * perImage;
        const local = frame - startFrame;
        const p = local < 0 ? 0 : spring({ frame: local, fps, config: { damping: 13, mass: 0.6, stiffness: 140 } });
        const settled = p >= 0.999;
        const bucket = Math.floor(frame / 2);
        const idle = settled ? jitter(i * 977 + seed, bucket, 2.4) : { dx: 0, dy: 0, dr: 0 };
        const dropIn = interpolate(p, [0, 1], [-height * 0.22, 0], { easing: Easing.out(Easing.cubic) });
        const scaleIn = interpolate(p, [0, 1], [1.3, 1]);
        const rotIn = slot.rot + (i % 2 === 0 ? 20 : -20) * (1 - p);
        const opacity = interpolate(p, [0, 0.4], [0, 1], { extrapolateRight: "clamp" });
        const left = slot.cx * width + idle.dx;
        const top = slot.cy * height + dropIn + idle.dy;
        const rotate = rotIn + idle.dr;

        if (img.cutout) {
          const { w, h } = fitSize(boxW, boxH, img.aspect || 1);
          return (
            <div key={i} style={{
              position: "absolute", left, top, width: w, height: h,
              transform: `translate(-50%, -50%) rotate(${rotate}deg) scale(${scaleIn})`,
              opacity,
            }}>
              <img src={img.src} style={{ width: "100%", height: "100%", objectFit: "contain", filter: stickerShadow() }} />
            </div>
          );
        }
        const pz = polaroidSize(boxW, boxH, img.aspect || 1);
        return (
          <div key={i} style={{
            position: "absolute", left, top, width: pz.cardW, height: pz.cardH,
            transform: `translate(-50%, -50%) rotate(${rotate}deg) scale(${scaleIn})`,
            opacity, background: "#FFFFFF", borderRadius: 4, boxSizing: "border-box",
            boxShadow: "0 12px 20px rgba(20,14,4,0.38)", padding: pz.border,
          }}>
            <img src={img.src} style={{ width: pz.imgW, height: pz.imgH, objectFit: "cover", display: "block" }} />
          </div>
        );
      })}
      {caption && <TornCaption text={caption} width={width} height={height} frame={frame} fps={fps} landFrame={landFrame} />}
    </AbsoluteFill>
  );
};
