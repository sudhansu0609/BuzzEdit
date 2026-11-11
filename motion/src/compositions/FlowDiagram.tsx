// `flow_diagram` beat -> presentation/motion_diagrams.py::build_flow_diagram
// -> this composition (MOTION_GRAPHICS_PLAN.md M9). Directive `[flow: step A
// ; step B ; step C | layout=chain|cycle|branch | seconds=N]`
// (presentation/script.py::directive_beats). 2-6 rounded boxes (theme panel/
// fg/accent) connect with arrows that draw in order (`@remotion/paths`
// evolvePath, the same technique MotionChart's line mode and MotionMap's
// route use); each box pops in the instant its own incoming arrow finishes.
// `chain` runs left-to-right (wrapping to two rows above 3 boxes), `cycle`
// puts the boxes on a circle with outward-bulging arrows that close the loop
// back onto the first box, `branch` fans every arrow out from the first box
// to the rest. The box the build has just passed flashes theme.accent, then
// settles with an accent2 border marking it as already visited. Key build
// lands by ~40% of the clip (Architecture); a slow scale drift keeps the
// rest of the hold from reading as frozen.
import React, { useMemo } from "react";
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { evolvePath } from "@remotion/paths";
import { fontFaceCss, fontStack } from "../fonts";
import { Theme, withTheme } from "../theme";

export type FlowDiagramLayout = "chain" | "cycle" | "branch";
export type FlowDiagramProps = {
  steps?: string[];
  layout?: FlowDiagramLayout;
  theme?: Partial<Theme>;
};

const VALID_LAYOUTS: FlowDiagramLayout[] = ["chain", "cycle", "branch"];

export const flowDiagramDefaultProps: FlowDiagramProps = {
  steps: ["Collect data", "Clean it up", "Train the model", "Ship it"],
  layout: "chain",
  theme: {},
};

type Point = { x: number; y: number };
type Arrow = { from: number; to: number };

// Chain layout: wraps to two rows once there are more than 3 steps (M9b
// review), so a 4-6 step chain can size each box by height as well as width
// instead of sitting as a single thin horizontal strip.
function chainRowCounts(n: number): number[] {
  if (n <= 3) return [n];
  const firstRow = Math.ceil(n / 2);
  return [firstRow, n - firstRow];
}

// How far a `cycle` layout's boxes sit from the loop's centre, as a fraction
// of the frame's own *height* (M9b review) -- the oval is always wider than
// it is tall, and height was the dimension starving the boxes. Capped at
// 0.32 rather than the review's 0.35-0.4: at 0.37 the top/bottom boxes,
// once tall enough for the review's own >=40px text floor (`CYCLE_BOX_H_FRAC`
// below), landed with a negative top edge and were clipped by the frame --
// a real bug caught by rendering and inspecting a frame, not just by eye on
// the formula. 0.32 is the largest radius that still leaves every box fully
// on screen with both floors in force.
const CYCLE_RADIUS_FRAC = 0.32;
// A `cycle` box's own height, as a fraction of the frame's height -- floored
// so `Box`'s own `h * 0.12` minimum font size (`FlowDiagram.tsx`'s shared
// text-fit) never drops below the review's >=40px ask.
const CYCLE_BOX_H_FRAC = 0.315;
// `chain`: the widest row of boxes together fills about this much of the
// frame's width; the gap between adjacent boxes is this fraction of a box's
// own width (room enough for an arrow and its head to read clearly).
const CHAIN_WIDTH_FILL = 0.8;
const CHAIN_GAP_RATIO = 0.35;

function layoutPositions(layout: FlowDiagramLayout, n: number, width: number, height: number, boxW: number): Point[] {
  const points: Point[] = [];
  if (layout === "cycle") {
    const cx = width / 2, cy = height / 2;
    const rx = width * 0.33, ry = height * CYCLE_RADIUS_FRAC;
    for (let i = 0; i < n; i++) {
      const angle = -Math.PI / 2 + (i / n) * Math.PI * 2;
      points.push({ x: cx + rx * Math.cos(angle), y: cy + ry * Math.sin(angle) });
    }
    return points;
  }
  if (layout === "branch") {
    points.push({ x: width / 2, y: height * 0.26 });
    const restN = n - 1;
    const left = width * 0.14, right = width * 0.86;
    for (let i = 0; i < restN; i++) {
      const x = restN > 1 ? left + (right - left) * (i / (restN - 1)) : width / 2;
      points.push({ x, y: height * 0.72 });
    }
    return points;
  }
  // chain -- left to right, wrapping to two rows above 3 steps; every box
  // shares `boxW` (computed by the caller to fill ~`CHAIN_WIDTH_FILL` of the
  // frame's width), spaced by `CHAIN_GAP_RATIO` of that width, each row
  // centred on its own span so a shorter second row still reads as balanced.
  const counts = chainRowCounts(n);
  const rows = counts.length;
  const gap = boxW * CHAIN_GAP_RATIO;
  for (let r = 0; r < rows; r++) {
    const count = counts[r];
    const y = rows === 1 ? height * 0.5 : (r === 0 ? height * 0.27 : height * 0.73);
    const span = count * boxW + (count - 1) * gap;
    const rowLeft = (width - span) / 2 + boxW / 2;
    for (let c = 0; c < count; c++) {
      const x = count > 1 ? rowLeft + (boxW + gap) * c : width / 2;
      points.push({ x, y });
    }
  }
  return points;
}

function arrowsFor(layout: FlowDiagramLayout, n: number): Arrow[] {
  if (layout === "branch") {
    const list: Arrow[] = [];
    for (let i = 1; i < n; i++) list.push({ from: 0, to: i });
    return list;
  }
  if (layout === "cycle") {
    const list: Arrow[] = [];
    for (let i = 0; i < n; i++) list.push({ from: i, to: (i + 1) % n });
    return list;
  }
  const list: Arrow[] = [];
  for (let i = 0; i < n - 1; i++) list.push({ from: i, to: i + 1 });
  return list;
}

// A ray from (cx,cy) through (dx,dy) hits this axis-aligned box's edge here --
// used to start/end an arrow exactly at a box's border rather than its centre.
function edgePoint(cx: number, cy: number, dx: number, dy: number, halfW: number, halfH: number): Point {
  const safeDx = dx === 0 ? 1e-6 : dx;
  const safeDy = dy === 0 ? 1e-6 : dy;
  const scale = Math.min(halfW / Math.abs(safeDx), halfH / Math.abs(safeDy));
  return { x: cx + dx * scale, y: cy + dy * scale };
}

// --- text fit: shrink the font until the step reads in at most 2 lines ----
// (a character-count width estimate, not a canvas measurement -- the fit
// only has to avoid clipping/overlap, not be exact, and this way never races
// the custom @font-face files loading in the renderer).
const AVG_CHAR_WIDTH_EM = 0.56;

function wrapAtSize(words: string[], size: number, maxWidth: number): string[] {
  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    const attempt = current ? `${current} ${word}` : word;
    const w = attempt.length * size * AVG_CHAR_WIDTH_EM;
    if (!current || w <= maxWidth) {
      current = attempt;
    } else {
      lines.push(current);
      current = word;
    }
  }
  if (current) lines.push(current);
  return lines;
}

function fitLines(text: string, maxWidth: number, maxSize: number, minSize: number): { lines: string[]; size: number } {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return { lines: [""], size: maxSize };
  for (let size = maxSize; size >= minSize; size -= 1) {
    const lines = wrapAtSize(words, size, maxWidth);
    if (lines.length <= 2) return { lines, size };
  }
  const mid = Math.max(1, Math.ceil(words.length / 2));
  return { lines: [words.slice(0, mid).join(" "), words.slice(mid).join(" ")], size: minSize };
}

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex) || [];
  return { r: parseInt(m[1] || "00", 16), g: parseInt(m[2] || "00", 16), b: parseInt(m[3] || "00", 16) };
}

function mixColor(a: string, b: string, t: number): string {
  const pa = hexToRgb(a), pb = hexToRgb(b);
  const clampT = Math.max(0, Math.min(1, t));
  const r = Math.round(pa.r + (pb.r - pa.r) * clampT);
  const g = Math.round(pa.g + (pb.g - pa.g) * clampT);
  const bl = Math.round(pa.b + (pb.b - pa.b) * clampT);
  return `rgb(${r},${g},${bl})`;
}

export const FlowDiagram: React.FC<FlowDiagramProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const layout: FlowDiagramLayout = VALID_LAYOUTS.includes(props.layout as FlowDiagramLayout)
    ? (props.layout as FlowDiagramLayout) : "chain";
  const steps = useMemo(() => {
    const list = (props.steps || []).filter((s) => s && s.trim()).slice(0, 6);
    return list.length >= 2 ? list : flowDiagramDefaultProps.steps!;
  }, [props.steps]);
  const n = steps.length;

  // Every box has landed and every arrow has drawn by ~40% of the clip; the
  // rest is a slow live drift (Architecture). `landFrame` is the *budget*
  // the sequential build below aims to fit inside -- the drift itself only
  // starts once the build has actually finished (`schedule.buildEnd`), never
  // mid-build, even for a build that overruns the budget on a 6-box layout.
  const landFrame = Math.max(1, Math.min(durationInFrames - 1, Math.round(durationInFrames * 0.4)));

  // Box size (M9b review): `chain` fills `CHAIN_WIDTH_FILL` of the frame's
  // width across its widest row and takes a generous, fixed share of the
  // height so its text is never small; `cycle` sizes from the loop's own
  // radius so adjacent boxes clear each other, floored so the text still
  // reads at >=40px; `branch` is unchanged.
  let boxW: number;
  let boxH: number;
  if (layout === "cycle") {
    const r = height * CYCLE_RADIUS_FRAC;
    const chord = n > 1 ? 2 * r * Math.sin(Math.PI / n) : r;
    boxW = Math.max(width * 0.14, Math.min(width * 0.27, chord * 0.82));
    boxH = Math.max(height * CYCLE_BOX_H_FRAC, boxW * 0.56);
  } else if (layout === "branch") {
    const cols = Math.max(2, n - 1);
    boxW = Math.min(width * 0.24, ((width * 0.82) / cols) * 0.82);
    boxH = boxW * 0.56;
  } else {
    const maxCols = Math.max(...chainRowCounts(n));
    boxW = (width * CHAIN_WIDTH_FILL) / (maxCols + (maxCols - 1) * CHAIN_GAP_RATIO);
    boxH = height * 0.34;
  }
  const strokeW = Math.max(5, Math.round(Math.min(width, height) * 0.0075));

  const points = useMemo(() => layoutPositions(layout, n, width, height, boxW), [layout, n, width, height, boxW]);
  const arrows = useMemo(() => arrowsFor(layout, n), [layout, n]);

  // Strictly sequential build: box 0 pops and fully lands, *then* its
  // outgoing arrow draws, *then* the box it points at pops and lands, and so
  // on -- one step at a time, never two overlapping, so an arrow can never
  // start drawing while its own source box is still mid-entrance (the old
  // per-index 0.85 overlap let that happen, leaving a stray line stub
  // reaching out of a box that hadn't visually arrived yet). `steps` walks
  // `arrows` in order and inserts one "box" step the first time that arrow's
  // target is reached -- for `branch`, every arrow shares the same
  // already-landed source (box 0), so each later arrow simply starts the
  // instant the previous box finishes landing, still never overlapping
  // anything; for `cycle`, the loop-closing arrow (last -> 0) draws after
  // the last new box lands and adds no further box step, since box 0 is
  // already on screen.
  type FlowStep = { kind: "box"; box: number } | { kind: "arrow"; arrow: number };
  const buildSteps = useMemo(() => {
    const landed = new Set<number>([0]);
    const list: FlowStep[] = [{ kind: "box", box: 0 }];
    arrows.forEach((a, i) => {
      list.push({ kind: "arrow", arrow: i });
      if (!landed.has(a.to)) {
        landed.add(a.to);
        list.push({ kind: "box", box: a.to });
      }
    });
    return list;
  }, [arrows]);

  // A box gets a short, fixed settle time (enough for its entrance spring to
  // visibly land) before the next step may start; whatever's left of the
  // budget splits evenly across the arrow draws.
  const boxDur = Math.max(4, Math.round(fps * 0.22));
  const boxStepCount = buildSteps.filter((s) => s.kind === "box").length;
  const arrowStepCount = Math.max(1, buildSteps.filter((s) => s.kind === "arrow").length);
  const arrowDur = Math.max(4, (landFrame - boxStepCount * boxDur) / arrowStepCount);

  const schedule = useMemo(() => {
    const arrowWindow: { start: number; dur: number }[] = new Array(arrows.length);
    const boxArrival = new Array(n).fill(0);
    let t = 0;
    for (const step of buildSteps) {
      if (step.kind === "box") {
        boxArrival[step.box] = t;
        t += boxDur;
      } else {
        arrowWindow[step.arrow] = { start: t, dur: arrowDur };
        t += arrowDur;
      }
    }
    return { arrowWindow, boxArrival, buildEnd: t };
  }, [buildSteps, boxDur, arrowDur, n, arrows.length]);

  const arrival = schedule.boxArrival;

  const driftStart = Math.max(1, Math.min(durationInFrames - 1, schedule.buildEnd));
  const driftEnd = Math.max(driftStart + 1, durationInFrames);
  const driftT = interpolate(frame, [driftStart, driftEnd], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.ease),
  });
  const driftScale = 1 + driftT * 0.02;

  const circleCenter = { x: width / 2, y: height / 2 };

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>
      <AbsoluteFill style={{ transform: `scale(${driftScale})`, transformOrigin: "center" }}>
        <svg width={width} height={height} style={{ position: "absolute" }}>
          {arrows.map((a, i) => {
            const { start, dur } = schedule.arrowWindow[i];
            const progress = interpolate(frame, [start, start + dur], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
            });
            let outward: Point | undefined;
            if (layout === "cycle") {
              const mx = (points[a.from].x + points[a.to].x) / 2;
              const my = (points[a.from].y + points[a.to].y) / 2;
              const ndx = mx - circleCenter.x, ndy = my - circleCenter.y;
              const nlen = Math.max(1, Math.hypot(ndx, ndy));
              const bulge = Math.min(width, height) * 0.11;
              outward = { x: (ndx / nlen) * bulge, y: (ndy / nlen) * bulge };
            }
            return (
              <ArrowPath key={i} from={points[a.from]} to={points[a.to]} boxW={boxW} boxH={boxH}
                        progress={progress} color={theme.accent2} strokeW={strokeW}
                        curved={layout === "cycle"} outward={outward} />
            );
          })}
        </svg>
        {points.map((p, i) => (
          <Box key={i} text={steps[i] || ""} x={p.x} y={p.y} w={boxW} h={boxH}
              arrival={arrival[i]} theme={theme} fps={fps} />
        ))}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const ArrowPath: React.FC<{
  from: Point; to: Point; boxW: number; boxH: number; progress: number;
  color: string; strokeW: number; curved: boolean; outward?: Point;
}> = ({ from, to, boxW, boxH, progress, color, strokeW, curved, outward }) => {
  const dx = to.x - from.x, dy = to.y - from.y;
  const start = edgePoint(from.x, from.y, dx, dy, boxW / 2, boxH / 2);
  const end = edgePoint(to.x, to.y, -dx, -dy, boxW / 2, boxH / 2);
  let d: string;
  let tangentAngle: number;
  if (curved && outward) {
    const mx = (start.x + end.x) / 2 + outward.x;
    const my = (start.y + end.y) / 2 + outward.y;
    d = `M ${start.x},${start.y} Q ${mx},${my} ${end.x},${end.y}`;
    tangentAngle = (Math.atan2(end.y - my, end.x - mx) * 180) / Math.PI;
  } else {
    d = `M ${start.x},${start.y} L ${end.x},${end.y}`;
    tangentAngle = (Math.atan2(end.y - start.y, end.x - start.x) * 180) / Math.PI;
  }
  const headOpacity = interpolate(progress, [0.9, 1], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });
  const bodyOpacity = interpolate(progress, [0, 0.03], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });
  return (
    <g opacity={bodyOpacity}>
      <path d={d} stroke={color} strokeWidth={strokeW} fill="none" strokeLinecap="round"
            {...evolvePath(progress, d)} />
      <g transform={`translate(${end.x},${end.y}) rotate(${tangentAngle})`} opacity={headOpacity}>
        <path d={`M ${-strokeW * 2.6},${-strokeW * 1.6} L ${strokeW},0 L ${-strokeW * 2.6},${strokeW * 1.6} Z`}
              fill={color} />
      </g>
    </g>
  );
};

const Box: React.FC<{
  text: string; x: number; y: number; w: number; h: number;
  arrival: number; theme: Theme; fps: number;
}> = ({ text, x, y, w, h, arrival, theme, fps }) => {
  const frame = useCurrentFrame();
  const since = frame - arrival;
  const entrance = spring({ frame: since, fps, config: { damping: 14, mass: 0.7, stiffness: 150 } });
  const opacity = since < 0 ? 0 : Math.min(1, entrance + 0.001);
  const scale = interpolate(entrance, [0, 1], [0.5, 1]);
  // A brief accent flash right as the build passes this box, settling back
  // to the panel fill with an accent2 border marking it already visited.
  const highlightT = interpolate(since, [0, fps * 0.18, fps * 1.0], [0, 1, 0], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.quad),
  });
  const fill = mixColor(theme.panel, theme.accent, highlightT);
  const borderColor = since >= 0 ? theme.accent2 : theme.border;
  const { lines, size } = useMemo(
    () => fitLines(text, w * 0.82, Math.max(10, Math.round(h * 0.26)), Math.max(8, Math.round(h * 0.12))),
    [text, w, h],
  );

  return (
    <div style={{
      position: "absolute", left: x - w / 2, top: y - h / 2, width: w, height: h,
      opacity, transform: `scale(${scale})`, transformOrigin: "center",
      display: "flex", alignItems: "center", justifyContent: "center",
      borderRadius: h * 0.22, background: fill,
      border: `${Math.max(2, h * 0.025)}px solid ${borderColor}`,
      boxShadow: "0 6px 18px rgba(0,0,0,0.25)",
    }}>
      <div style={{ textAlign: "center", lineHeight: 1.2 }}>
        {lines.map((line, i) => (
          <div key={i} style={{
            fontFamily: fontStack(theme.fontBody), fontWeight: 700, fontSize: size, color: theme.fg,
          }}>
            {line}
          </div>
        ))}
      </div>
    </div>
  );
};
