// `callout` beat -> presentation/motion_ui.py::build_callout -> this
// composition (MOTION_GRAPHICS_PLAN.md M10). Directive `[callout: <image
// prompt> | mark=circle|arrow|magnify | at=<3x3 region> | label=<short
// text> | seconds=N]` (presentation/script.py::directive_beats): the image
// prompt is generated through the normal ComfyUI still path first
// (presentation/motion_ui.py::expand_callouts, mirroring motion_collage.py::
// expand_collages's own pattern for paper_collage), then this draws a
// hand-drawn circle or arrow (`@remotion/paths` evolvePath, the same
// technique PaperCollage's doodles use) or slides a magnifier lens showing a
// 2x zoom onto the chosen region of a 3x3 grid; the rest of the frame dims
// slightly and the whole scene pushes in slowly toward that region, landing
// by ~40% of the clip (Architecture).
import React from "react";
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { evolvePath } from "@remotion/paths";
import { fontFaceCss, fontStack } from "../fonts";
import { Theme, withTheme } from "../theme";

export type CalloutMark = "circle" | "arrow" | "magnify";
export type CalloutRegion =
  | "top_left" | "top" | "top_right"
  | "left" | "center" | "right"
  | "bottom_left" | "bottom" | "bottom_right";

// Where the mark should actually land, in 0..1 fractions of the frame --
// `rx`/`ry` the half-extent of whatever it is aimed at (also 0..1 fractions,
// of width/height respectively). `motion_ui.py::aim_callout` builds this
// from a detected face or the nearest salient object's centroid; the 3x3
// `at` region alone only picks a cell, which lands on whatever happens to
// be in it (MOTION_GRAPHICS_PLAN.md M10 review -- a circle on empty floor
// beside the lamp it was meant to mark, a magnifier on the wall instead of
// a face).
export type CalloutTarget = { x: number; y: number; rx: number; ry: number };

export type CalloutProps = {
  // A data: URI -- the still lives outside motion/public/, so it travels as
  // a prop (mirrors PaperCollage's own CollageImage.src).
  image?: string;
  mark?: CalloutMark;
  at?: CalloutRegion;
  label?: string | null;
  // Absent (or null) falls back to the bare cell centre, exactly as before
  // -- the aim step found nothing to lock onto (no face, nothing salient).
  target?: CalloutTarget | null;
  theme?: Partial<Theme>;
};

// A 2x2 dark-grey PNG -- `--selftest` renders every registered composition
// with no real still on hand; this is enough for the scene to mount (the
// same trick PaperCollage.tsx uses for its own default props).
const BLANK_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPU0NBgYGBgYgADAAXKAHxvK9IEAAAAAElFTkSuQmCC";

export const calloutDefaultProps: CalloutProps = {
  image: BLANK_IMAGE,
  mark: "circle",
  at: "top_right",
  label: "Look here",
  theme: {},
};

const MARKS: CalloutMark[] = ["circle", "arrow", "magnify"];
const REGION_CELL: Record<CalloutRegion, [number, number]> = {
  top_left: [0, 0], top: [1, 0], top_right: [2, 0],
  left: [0, 1], center: [1, 1], right: [2, 1],
  bottom_left: [0, 2], bottom: [1, 2], bottom_right: [2, 2],
};

// --- a small, local copy of PaperCollage.tsx's own "hand-drawn" helpers ----
// (that file is edited in parallel elsewhere, per MOTION_GRAPHICS_PLAN.md --
// copied, not imported, the way PaperCollage.tsx itself already copies
// StickScene's own boilJitter idea rather than importing it).
function hashUnit(n: number): number {
  const s = Math.sin(n) * 43758.5453123;
  return s - Math.floor(s);
}
function circlePath(cx: number, cy: number, rx: number, ry: number, seed: number): string {
  const steps = 24;
  const pts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const a = (i / steps) * Math.PI * 2 - Math.PI / 2;
    const wobble = 1 + (hashUnit(seed * 5.1 + i * 1.7) - 0.5) * 0.1;
    const x = cx + Math.cos(a) * rx * wobble;
    const y = cy + Math.sin(a) * ry * wobble;
    pts.push(`${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`);
  }
  return pts.join(" ");
}
function arrowPath(x1: number, y1: number, x2: number, y2: number, seed: number): string {
  const mx = (x1 + x2) / 2 + (hashUnit(seed * 3.3) - 0.5) * 30;
  const my = (y1 + y2) / 2 + (hashUnit(seed * 7.7) - 0.5) * 30;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 20;
  const a1 = angle + Math.PI * 0.8, a2 = angle - Math.PI * 0.8;
  const hx1 = x2 + Math.cos(a1) * headLen, hy1 = y2 + Math.sin(a1) * headLen;
  const hx2 = x2 + Math.cos(a2) * headLen, hy2 = y2 + Math.sin(a2) * headLen;
  return `M ${x1.toFixed(1)} ${y1.toFixed(1)} Q ${mx.toFixed(1)} ${my.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)} `
       + `M ${hx1.toFixed(1)} ${hy1.toFixed(1)} L ${x2.toFixed(1)} ${y2.toFixed(1)} L ${hx2.toFixed(1)} ${hy2.toFixed(1)}`;
}

// --- the composition ---------------------------------------------------------

export const Callout: React.FC<CalloutProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const image = props.image || calloutDefaultProps.image!;
  const mark: CalloutMark = props.mark && MARKS.includes(props.mark) ? props.mark : "circle";
  const region: CalloutRegion = props.at && REGION_CELL[props.at] ? props.at : "center";
  const label = (props.label || "").trim();

  const [col, row] = REGION_CELL[region];
  const cellW = width / 3, cellH = height / 3;
  const x0 = col * cellW, y0 = row * cellH, x1 = x0 + cellW, y1 = y0 + cellH;
  const cx = x0 + cellW / 2, cy = y0 + cellH / 2;

  // The mark's own geometry targets the aimed point (a face or salient
  // object inside this cell) when one was found; the dim frame and the
  // slow push-in below still key off the plain cell, unchanged. Every
  // derived value below falls back to exactly its old cell-based formula
  // when `target` is absent, so a render with no aim (or an old cached
  // job) looks identical to before.
  const target = props.target || null;
  const mx = target ? target.x * width : cx;
  const my = target ? target.y * height : cy;
  const markRX = target ? Math.max(24, target.rx * width) : cellW * 0.42;
  const markRY = target ? Math.max(24, target.ry * height) : cellH * 0.42;
  // Circle: an ellipse around the target's own bbox, padded 10%.
  const circleRX = target ? markRX * 1.1 : markRX;
  const circleRY = target ? markRY * 1.1 : markRY;
  // Arrow: stop 20px short of the bbox edge, not a fixed share of the cell.
  const arrowEndRX = target ? Math.max(0, markRX - 20) : cellW * 0.14;
  const arrowEndRY = target ? Math.max(0, markRY - 20) : cellH * 0.14;
  // Magnifier: sized off the bbox (padded a third) rather than the cell.
  const magnifyBaseR = target ? Math.max(markRX, markRY) * 1.3 : Math.min(cellW, cellH) * 0.56;

  const landFrame = Math.max(1, Math.min(durationInFrames - 1, Math.round(durationInFrames * 0.4)));
  // Slow push-in toward the marked region over the whole clip (Architecture).
  const zoom = interpolate(frame, [0, Math.max(1, durationInFrames - 1)], [1, 1.14],
    { easing: Easing.inOut(Easing.ease) });
  const dimT = interpolate(frame, [0, Math.max(1, Math.round(fps * 0.8))], [0, 0.45],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const markStart = Math.round(fps * 0.3);
  const markEnd = Math.max(markStart + 1, Math.min(landFrame, markStart + Math.round(fps * 1.1)));
  const markProgress = interpolate(frame, [markStart, markEnd], [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });

  const seed = (col + 1) * 7 + (row + 1) * 13;

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>
      <AbsoluteFill style={{ transform: `scale(${zoom})`, transformOrigin: `${(cx / width) * 100}% ${(cy / height) * 100}%` }}>
        <AbsoluteFill style={{ backgroundImage: `url(${image})`, backgroundSize: "cover", backgroundPosition: "center" }} />
        <DimFrame x0={x0} y0={y0} x1={x1} y1={y1} width={width} height={height} opacity={dimT} />
        {mark === "magnify" ? (
          <MagnifyLens image={image} width={width} height={height} cx={mx} cy={my}
            cellW={cellW} cellH={cellH} baseR={magnifyBaseR} frame={frame} fps={fps} theme={theme} startFrame={markStart} />
        ) : (
          <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
            {mark === "circle" ? (
              <CirclePathMark cx={mx} cy={my} rx={circleRX} ry={circleRY} progress={markProgress}
                accent={theme.accent} seed={seed} />
            ) : (
              <ArrowPathMark width={width} height={height} cx={mx} cy={my} cellW={cellW} cellH={cellH}
                rx={arrowEndRX} ry={arrowEndRY} progress={markProgress} accent={theme.accent} seed={seed} />
            )}
          </svg>
        )}
        {label && (
          <LabelSticker text={label} x0={x0} y0={y0} width={width} height={height}
            frame={frame} start={markStart + Math.round(fps * 0.4)} fps={fps} theme={theme} />
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const DimFrame: React.FC<{ x0: number; y0: number; x1: number; y1: number; width: number; height: number; opacity: number; }>
  = ({ x0, y0, x1, y1, width, height, opacity }) => {
  const bg = `rgba(0,0,0,${opacity})`;
  return (
    <>
      {y0 > 0 && <div style={{ position: "absolute", left: 0, top: 0, width, height: y0, background: bg }} />}
      {y1 < height && <div style={{ position: "absolute", left: 0, top: y1, width, height: height - y1, background: bg }} />}
      {x0 > 0 && <div style={{ position: "absolute", left: 0, top: y0, width: x0, height: y1 - y0, background: bg }} />}
      {x1 < width && <div style={{ position: "absolute", left: x1, top: y0, width: width - x1, height: y1 - y0, background: bg }} />}
    </>
  );
};

const CirclePathMark: React.FC<{ cx: number; cy: number; rx: number; ry: number; progress: number; accent: string; seed: number; }>
  = ({ cx, cy, rx, ry, progress, accent, seed }) => {
  const d = circlePath(cx, cy, rx, ry, seed);
  return <path d={d} stroke={accent} strokeWidth={6} fill="none" strokeLinecap="round" strokeLinejoin="round"
    {...evolvePath(progress, d)} />;
};

const ArrowPathMark: React.FC<{
  width: number; height: number; cx: number; cy: number; cellW: number; cellH: number;
  rx: number; ry: number; progress: number; accent: string; seed: number;
}> = ({ width, height, cx, cy, cellW, cellH, rx, ry, progress, accent, seed }) => {
  // The arrow runs in from whichever side of the frame the region is nearer
  // to (a diagonal for the dead-centre region, which is equidistant from
  // every side); where it starts is still a fixed share of the grid cell
  // (just the visual sweep-in distance), but it now stops `rx`/`ry` short of
  // centre -- the caller works that out from the target's own bbox (20px
  // short of its edge) rather than a fixed share of the cell, which could be
  // far bigger or smaller than whatever is actually inside it
  // (MOTION_GRAPHICS_PLAN.md M10 review).
  let dirX = cx < width / 2 ? -1 : cx > width / 2 ? 1 : 0;
  let dirY = cy < height / 2 ? -1 : cy > height / 2 ? 1 : 0;
  if (dirX === 0 && dirY === 0) { dirX = 1; dirY = -1; }
  const startX = cx + dirX * cellW * 1.1;
  const startY = cy + dirY * cellH * 1.1;
  const endX = cx - dirX * rx;
  const endY = cy - dirY * ry;
  const d = arrowPath(startX, startY, endX, endY, seed);
  return <path d={d} stroke={accent} strokeWidth={7} fill="none" strokeLinecap="round" strokeLinejoin="round"
    {...evolvePath(progress, d)} />;
};

const MagnifyLens: React.FC<{
  image: string; width: number; height: number; cx: number; cy: number; cellW: number; cellH: number;
  baseR: number; frame: number; fps: number; theme: Theme; startFrame: number;
}> = ({ image, width, height, cx, cy, cellW, cellH, baseR, frame, fps, theme, startFrame }) => {
  const since = Math.max(0, frame - startFrame);
  const pop = spring({ frame: since, fps, config: { damping: 14, mass: 0.7, stiffness: 130 } });
  // A region touching an edge (or two, at a corner) would otherwise push the
  // lens -- sized off `baseR` (the target's own bbox, padded a third, or the
  // cell when there was nothing to aim at) -- past the frame and clip its
  // own border there (found by a real render + frame inspection, a top-row
  // region); clamped to how much room the lens's own settled centre (cx,cy)
  // actually has on every side.
  const edgeRoom = Math.min(cx, width - cx, cy, height - cy) - 6;
  const lensR = Math.min(baseR, Math.max(24, edgeRoom));
  const zoomFactor = 2;
  // The classic CSS image-magnifier trick: a circular window onto a second
  // copy of the same background, scaled `zoomFactor`x and offset so the
  // point under the lens's own centre lands centred inside it.
  const bgW = width * zoomFactor, bgH = height * zoomFactor;
  const bgX = -(cx * zoomFactor - lensR);
  const bgY = -(cy * zoomFactor - lensR);
  const slideFrom = { x: cx + cellW * 0.6, y: cy - cellH * 0.6 };
  const left = interpolate(pop, [0, 1], [slideFrom.x, cx]);
  const top = interpolate(pop, [0, 1], [slideFrom.y, cy]);
  return (
    <div style={{
      position: "absolute", left, top, width: lensR * 2, height: lensR * 2,
      transform: `translate(-50%, -50%) scale(${Math.max(0.2, pop)})`,
      borderRadius: "50%", overflow: "hidden", opacity: Math.min(1, pop + 0.2),
      border: `${Math.max(3, lensR * 0.045)}px solid ${theme.fg}`,
      boxShadow: "0 14px 30px rgba(0,0,0,0.5)",
    }}>
      <div style={{
        position: "absolute", left: bgX, top: bgY, width: bgW, height: bgH,
        backgroundImage: `url(${image})`, backgroundSize: `${bgW}px ${bgH}px`, backgroundPosition: "0 0",
      }} />
    </div>
  );
};

const LabelSticker: React.FC<{
  text: string; x0: number; y0: number; width: number; height: number;
  frame: number; start: number; fps: number; theme: Theme;
}> = ({ text, x0, y0, width, height, frame, start, fps, theme }) => {
  const since = Math.max(0, frame - start);
  const pop = spring({ frame: since, fps, config: { damping: 14, mass: 0.6, stiffness: 150 } });
  const opacity = frame < start ? 0 : Math.min(1, pop + 0.15);
  const y1 = y0 + height / 3;
  // Below the cell when there's room, else above it -- always clear of the
  // top/bottom frame edge.
  const below = y1 + height * 0.1 <= height;
  const top = below ? y1 + height * 0.02 : Math.max(height * 0.02, y0 - height * 0.09);
  const left = Math.min(Math.max(x0, width * 0.02), width * 0.98 - width * 0.3);
  return (
    <div style={{
      position: "absolute", left, top, opacity, transform: `scale(${Math.max(0.5, pop)})`,
      transformOrigin: "left top", background: theme.accent, color: theme.bg,
      fontFamily: fontStack(theme.fontBody), fontWeight: 700, fontSize: Math.max(14, height * 0.028),
      padding: `${height * 0.012}px ${height * 0.018}px`, borderRadius: 999,
      boxShadow: "0 8px 16px rgba(0,0,0,0.35)", maxWidth: width * 0.3,
    }}>
      {text}
    </div>
  );
};
