// M3 (MOTION_GRAPHICS_PLAN.md): five transparent transition stingers --
// ShapeWipe, Iris, Ink, Bars, PaperTear -- rendered as ProRes 4444 overlays
// and placed by presentation/motion_transitions.py centred on a cut.
// ShapeWipe, Iris, Ink and Bars are driven by the same `coverage()` curve:
// 0 at the first frame, 1 at the exact middle frame, 0 at the last frame
// (Easing.inOut, clamped) -- so whatever shape a composition draws, it is
// guaranteed to fully cover the canvas at the midpoint and leave nothing
// behind at either end. PaperTear (M3b: a sliding, torn sheet rather than a
// shape that grows) instead uses `useSlideT()`, the same landmark frames and
// easing but monotonic (-1 -> 0 -> +1), which lands it centred -- and so
// still fully covering -- at the exact same midpoint. Every shape is drawn
// oversized relative to the canvas so the covered state has margin on every
// side; only the shape (never a full-bleed AbsoluteFill background) carries
// colour, so the root stays transparent for the alpha render.
import React, { useMemo } from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { Theme, withTheme } from "../theme";

export type TransitionStingerProps = { theme?: Partial<Theme> };

/** 0 -> 1 -> 0 across the clip, landing exactly 1 at the middle frame. */
function useCoverage(): number {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const mid = Math.max(1, Math.round(durationInFrames / 2));
  const last = Math.max(mid + 1, durationInFrames - 1);
  return interpolate(frame, [0, mid, last], [0, 1, 0], {
    easing: Easing.inOut(Easing.cubic),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
}

export const transitionStingerDefaultProps: TransitionStingerProps = { theme: {} };

// --- ShapeWipe: a diamond (rotated square) grows from the centre -----------

export const ShapeWipe: React.FC<TransitionStingerProps> = (props) => {
  const theme = withTheme(props.theme);
  const { width, height } = useVideoConfig();
  const coverage = useCoverage();
  // Oversized so the diamond's corner-to-corner reach at coverage=1 clears
  // the canvas's own corners with margin.
  const side = Math.hypot(width, height) * 1.3;
  const scale = coverage;
  if (scale <= 0.001) return <AbsoluteFill />;
  return (
    <AbsoluteFill>
      <div
        style={{
          position: "absolute", left: "50%", top: "50%",
          width: side, height: side,
          background: theme.accent,
          transform: `translate(-50%, -50%) rotate(45deg) scale(${scale})`,
        }}
      />
    </AbsoluteFill>
  );
};

// --- Iris: a circle grows from a point ---------------------------------

export const Iris: React.FC<TransitionStingerProps> = (props) => {
  const theme = withTheme(props.theme);
  const { width, height } = useVideoConfig();
  const coverage = useCoverage();
  const maxR = (Math.hypot(width, height) / 2) * 1.15;
  const r = coverage * maxR;
  if (r <= 0.5) return <AbsoluteFill />;
  return (
    <AbsoluteFill>
      <div
        style={{
          position: "absolute", left: "50%", top: "50%",
          width: r * 2, height: r * 2, borderRadius: "50%",
          background: theme.accent2,
          transform: "translate(-50%, -50%)",
        }}
      />
    </AbsoluteFill>
  );
};

// --- Ink: a metaball splash in the theme accent colour, with tendrils ------
// M3b (review): read as flat black circles -- `fill={theme.bg}` instead of
// the accent colour, and six discs centred dead-on on the canvas instead of
// an organic splash. Now two off-centre anchors (never dead centre) carry
// the main blot, and a handful of smaller, farther, later-arriving blobs
// along a few fixed directions from each anchor read as tendrils once
// blurred and thresholded together -- they reach out as `coverage` climbs
// past their own `growAt` and (since it is the same symmetric curve) pull
// back in first as it recedes, the way a real ink spread's thin fingers are
// the first part to vanish.

type InkBlob = { dx: number; dy: number; k: number; growAt?: number };

const INK_BLOBS: InkBlob[] = [
  // The two anchors the splash actually grows from -- off-centre, one
  // bigger than the other, so it never reads as a disc from dead centre.
  { dx: -0.08, dy: -0.05, k: 1.0 },
  { dx: 0.1, dy: 0.06, k: 0.85 },
  // Tendrils reaching out from the primary anchor.
  { dx: -0.2, dy: -0.16, k: 0.42, growAt: 0.25 },
  { dx: -0.26, dy: 0.04, k: 0.32, growAt: 0.4 },
  { dx: -0.12, dy: 0.22, k: 0.28, growAt: 0.5 },
  // Tendrils reaching out from the secondary anchor.
  { dx: 0.24, dy: 0.18, k: 0.36, growAt: 0.3 },
  { dx: 0.2, dy: -0.12, k: 0.28, growAt: 0.45 },
  { dx: 0.05, dy: 0.24, k: 0.24, growAt: 0.55 },
];

export const Ink: React.FC<TransitionStingerProps> = (props) => {
  const theme = withTheme(props.theme);
  const { width, height } = useVideoConfig();
  const coverage = useCoverage();
  // Comfortably bigger than a dead-centre blot would need, since the two
  // anchors are off-centre -- the farthest canvas corner from the smaller
  // one is still well inside this radius at coverage=1.
  const maxR = Math.hypot(width, height) * 0.68;
  const r = coverage * maxR;
  if (r <= 0.5) return <AbsoluteFill />;
  const blur = Math.max(1, width * 0.02);
  return (
    <AbsoluteFill>
      <svg width={width} height={height} style={{ position: "absolute", top: 0, left: 0 }}>
        <filter id="inkBlot">
          <feGaussianBlur in="SourceGraphic" stdDeviation={blur} result="blur" />
          {/* Pushes the blurred alpha gradient to a hard edge -- several soft
             circles read as one organic blot instead of overlapping discs.
             Identity on R/G/B (M3b: a zeroed colour row painted every blob
             solid black regardless of `fill` -- the same "flat black
             circles" defect this composition was meant to fix, just moved
             into the filter) -- only alpha is thresholded, so the blot keeps
             `theme.accent`. */}
          <feColorMatrix in="blur" type="matrix"
            values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 24 -10" result="solid" />
        </filter>
        <g filter="url(#inkBlot)">
          {INK_BLOBS.map((b, i) => {
            const grow = b.growAt
              ? interpolate(coverage, [b.growAt, Math.min(1, b.growAt + 0.35)], [0, 1], {
                  extrapolateLeft: "clamp", extrapolateRight: "clamp",
                })
              : 1;
            return (
              <circle key={i}
                cx={width / 2 + b.dx * width}
                cy={height / 2 + b.dy * height}
                r={Math.max(0, r * b.k * grow)}
                fill={theme.accent}
              />
            );
          })}
        </g>
      </svg>
    </AbsoluteFill>
  );
};

// --- Bars: vertical blinds close to a solid wall and open again -----------

const BARS_COUNT = 7;

export const Bars: React.FC<TransitionStingerProps> = (props) => {
  const theme = withTheme(props.theme);
  const coverage = useCoverage();
  if (coverage <= 0.001) return <AbsoluteFill />;
  return (
    <AbsoluteFill style={{ flexDirection: "row" }}>
      {Array.from({ length: BARS_COUNT }).map((_, i) => (
        <div
          key={i}
          style={{
            flex: 1, height: "100%",
            background: i % 2 === 0 ? theme.accent : theme.accent2,
            transform: `scaleY(${coverage})`,
            transformOrigin: "center",
          }}
        />
      ))}
    </AbsoluteFill>
  );
};

// --- PaperTear: a paper sheet slides across, torn on its leading edge, ----
// --- then torn away on its trailing edge -----------------------------------
// M3b (review): read as a plain white square growing from the centre, not a
// tear. A paper sheet, oversized so its straight top/bottom clear the frame
// with margin, now *slides* horizontally instead of scaling: its right
// (leading) edge is ragged while it slides in from the left and fully
// covers the frame at the midpoint, then it keeps sliding the same
// direction and its left (trailing) edge -- torn with a different, unrelated
// jag pattern -- rips it away off the right, uncovering the frame from the
// left side first. A faint fibre-texture layer and a drop shadow that
// follows the torn silhouette (CSS `filter` applied after `clipPath`) sell
// it as paper rather than a flat card.
// FX1 (review): at normal (1080p) size the wide, shallow 14-tooth comb read
// as a near-straight edge with a stray wobble, not a tear. Both torn edges
// are now a dense, irregular deckle (teeth ~16-20px apart, each one 8-20px
// deep at 1080p, scaled to size) and carry a torn-fibre rim: a second,
// inset copy of the same jagged silhouette shades the paper just past the
// rim with a soft inner shadow, leaving the outer 3-6px (at 1080p) of actual
// paper colour reading paler by contrast, tooth by tooth. The drop shadow
// still comes from the one true outer silhouette, so it follows the jagged
// shape exactly as before.

/** A deterministic jagged polygon, torn on its left and right (vertical)
 * edges and straight top/bottom -- the two edges carry unrelated jag
 * patterns (different seeds) so the sheet's leading tear while sliding in
 * never matches the tear that rips it away on the other side. `insetPx`
 * (also in the 1080p-scaled unit) shrinks the same torn silhouette inward by
 * a fixed amount on both edges -- used to clip the torn-fibre rim's own
 * inner shading, so it tracks every tooth of the outer tear. */
function tornSheetPolygon(sheetHeight: number, refHeight: number, seedLeft: number,
                          seedRight: number, insetPx = 0): string {
  const rnd = (seed: number, i: number) => {
    const x = Math.sin(i * 12.9898 + seed * 78.233) * 43758.5453;
    return x - Math.floor(x);
  };
  // 1080p-scaled so a smaller preview render keeps proportionally the same
  // deckled look instead of a comb of huge teeth relative to its own frame.
  const unit = refHeight / 1080;
  const toothMin = 8 * unit;
  const toothMax = 20 * unit;
  const teeth = Math.max(16, Math.round(sheetHeight / (18 * unit)));
  const jagAt = (seed: number, i: number) => {
    const mag = toothMin + rnd(seed, i * 2) * (toothMax - toothMin);
    const sign = rnd(seed, i * 2 + 1) < 0.5 ? -1 : 1;
    return sign * mag;
  };
  const right: string[] = [];
  const left: string[] = [];
  for (let i = 0; i <= teeth; i++) {
    const yp = `${(i / teeth) * 100}%`;
    const rightJag = jagAt(seedRight, i) - insetPx;
    const leftJag = jagAt(seedLeft, i + 100) + insetPx;
    right.push(`calc(100% + ${rightJag}px) ${yp}`);
    left.push(`${leftJag}px ${yp}`);
  }
  return `polygon(${right.join(",")}, ${left.reverse().join(",")})`;
}

/** -1 at the first frame, 0 (centred, full cover) exactly at the midpoint,
 * +1 at the last frame -- unlike `useCoverage` this is monotonic (a slide,
 * not a grow-then-shrink), but it shares the same landmark frames and
 * per-segment easing, so the sheet is still centred at the exact midpoint
 * and moving slowest right around it. */
function useSlideT(): number {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const mid = Math.max(1, Math.round(durationInFrames / 2));
  const last = Math.max(mid + 1, durationInFrames - 1);
  return interpolate(frame, [0, mid, last], [-1, 0, 1], {
    easing: Easing.inOut(Easing.cubic),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
}

export const PaperTear: React.FC<TransitionStingerProps> = (props) => {
  const theme = withTheme(props.theme);
  const { width, height } = useVideoConfig();
  const slideT = useSlideT();
  const sheetW = width * 1.5;
  const sheetH = height * 1.3;
  // Torn-fibre rim width -- 3-6px at 1080p, scaled for other render sizes.
  const rimPx = Math.max(1.5, 4.5 * (height / 1080));
  const outerClip = useMemo(() => tornSheetPolygon(sheetH, height, 7, 23),
    [sheetH, height]);
  const innerClip = useMemo(() => tornSheetPolygon(sheetH, height, 7, 23, rimPx),
    [sheetH, height, rimPx]);
  // The sheet travels its own width (plus a margin) end to end so both the
  // leading and the trailing torn edges clear the frame before/after.
  const travel = sheetW * 0.85;
  const x = slideT * travel;
  // Fully off-screen on either side -- nothing to draw (same bail-out
  // `useCoverage`-driven stingers get from `coverage <= 0.001`).
  if (Math.abs(x) > width / 2 + sheetW / 2) return <AbsoluteFill />;
  return (
    <AbsoluteFill>
      <div
        style={{
          position: "absolute", left: "50%", top: "50%",
          width: sheetW, height: sheetH,
          transform: `translate(calc(-50% + ${x}px), -50%)`,
          clipPath: outerClip,
          filter: "drop-shadow(0 0 18px rgba(0,0,0,0.45))",
        }}
      >
        {/* Flat, untouched paper colour -- the base, and (where the inset
            layer below does not reach) the rim itself. `box-shadow` shades
            an element's own rectangular box, not its `clip-path` silhouette,
            so an inset shadow here would land on the sheet's real (offscreen)
            rectangular edges, nowhere near the torn silhouette -- the rim
            instead has to be a second, smaller fill confined to its own
            inset `clip-path`, below. */}
        <div style={{ position: "absolute", inset: 0, background: theme.panel }} />
        {/* Torn-fibre rim: the fibre texture is confined to an inset copy of
            the same jagged silhouette (clipped `rimPx` in from every tooth),
            leaving the true outer rim -- flat, untouched paper colour with
            no darkening multiply-blend noise over it -- reading paler by
            contrast with the slightly toned interior next to it, following
            every tooth of the tear. */}
        <div style={{ position: "absolute", inset: 0, clipPath: innerClip }}>
          <svg width="100%" height="100%" style={{ position: "absolute", inset: 0 }}>
            <filter id="paperFibre">
              <feTurbulence type="fractalNoise" baseFrequency="0.012 0.06" numOctaves={2} seed={4} result="noise" />
              <feColorMatrix in="noise" type="matrix"
                values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.18 0" />
            </filter>
            <rect width="100%" height="100%" filter="url(#paperFibre)" style={{ mixBlendMode: "multiply" }} />
          </svg>
        </div>
      </div>
    </AbsoluteFill>
  );
};
