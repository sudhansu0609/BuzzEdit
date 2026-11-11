// `chart` beat -> presentation/motion_data.py::build_chart -> this
// composition (MOTION_GRAPHICS_PLAN.md M2). One composition, four modes --
// mirroring charts.py's own dispatch exactly (`render_chart_assets`): bars
// grow staggered with their value counting up, a line draws with points
// popping as it reaches them, pie slices sweep in, and a `years` beat (the
// same `render_timeline_strip` case) is a ruler with the active year large.
// Key values land by ~40% of the clip, then drift slowly (Architecture).
import React, { useMemo } from "react";
import {
  AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig,
} from "remotion";
import { evolvePath } from "@remotion/paths";
import { makePie } from "@remotion/shapes";
import { fontFaceCss, fontStack } from "../fonts";
import { Theme, withTheme } from "../theme";

export type MotionChartMode = "bar" | "line" | "pie" | "years";
export type MotionChartProps = {
  mode?: MotionChartMode;
  labels?: string[];
  values?: number[];
  unit?: string;
  title?: string | null;
  years?: string[];
  active?: string;
  theme?: Partial<Theme>;
};

export const motionChartDefaultProps: MotionChartProps = {
  mode: "bar",
  labels: ["2022", "2023", "2024"],
  values: [40, 65, 82],
  unit: "%",
  title: "Motion Chart Demo",
  theme: {},
};

function formatValue(value: number, unit: string): string {
  const rounded = Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
  const withCommas = rounded.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  if (unit === "%") return `${withCommas}%`;
  return unit ? `${withCommas} ${unit}` : withCommas;
}

const PIE_FACTORS = [0, 0.35, -0.25, 0.55, -0.45, 0.2, -0.15];
function tint(hex: string, factor: number): string {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex) || [];
  let [r, g, b] = [m[1], m[2], m[3]].map((h) => parseInt(h || "00", 16));
  if (factor >= 0) {
    r = r + (255 - r) * factor; g = g + (255 - g) * factor; b = b + (255 - b) * factor;
  } else {
    r = r * (1 + factor); g = g * (1 + factor); b = b * (1 + factor);
  }
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  return `rgb(${clamp(r)},${clamp(g)},${clamp(b)})`;
}

export const MotionChart: React.FC<MotionChartProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const mode: MotionChartMode = props.mode && ["bar", "line", "pie", "years"].includes(props.mode)
    ? props.mode : "bar";
  const landFrame = Math.max(1, Math.min(durationInFrames - 1, Math.round(durationInFrames * 0.4)));

  const titleEl = props.title ? (
    <div style={{
      position: "absolute", top: height * 0.06, left: width * 0.08, right: width * 0.08,
      fontFamily: fontStack(theme.fontHead), fontWeight: 700, fontSize: height * 0.06,
      color: theme.fg,
    }}>
      {props.title}
    </div>
  ) : null;

  let body: React.ReactNode = null;
  if (mode === "years") {
    body = <YearsStrip years={props.years || []} active={props.active} theme={theme}
                        frame={frame} landFrame={landFrame} width={width} height={height} fps={fps} />;
  } else if (mode === "pie") {
    body = <PieBody labels={props.labels || []} values={props.values || []} unit={props.unit || ""}
                     theme={theme} frame={frame} landFrame={landFrame} width={width} height={height} />;
  } else if (mode === "line") {
    body = <LineBody labels={props.labels || []} values={props.values || []} unit={props.unit || ""}
                      theme={theme} frame={frame} landFrame={landFrame} width={width} height={height} fps={fps} />;
  } else {
    body = <BarBody labels={props.labels || []} values={props.values || []} unit={props.unit || ""}
                     theme={theme} frame={frame} landFrame={landFrame} width={width} height={height} fps={fps} />;
  }

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>
      {titleEl}
      {body}
    </AbsoluteFill>
  );
};

// --- bar ----------------------------------------------------------------

const BarBody: React.FC<{
  labels: string[]; values: number[]; unit: string; theme: Theme;
  frame: number; landFrame: number; width: number; height: number; fps: number;
}> = ({ labels, values, unit, theme, frame, landFrame, width, height, fps }) => {
  const count = Math.max(1, values.length);
  const peak = Math.max(1e-6, ...values.map((v) => Math.abs(v)));
  const maxIndex = values.reduce((best, v, i) => (Math.abs(v) > Math.abs(values[best]) ? i : best), 0);
  const top = height * 0.26;
  const areaH = height * 0.62;
  const rowH = areaH / count;
  const barH = Math.min(rowH * 0.5, height * 0.1);
  const perBar = landFrame / (count + 1);

  return (
    <div style={{ position: "absolute", top, left: width * 0.08, right: width * 0.08, bottom: height * 0.1 }}>
      {values.map((value, i) => {
        const start = Math.round(i * perBar * 0.7);
        const t = interpolate(frame, [start, start + fps * 0.55], [0, 1], {
          extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
        });
        const widthPct = t * (Math.abs(value) / peak) * 100;
        const shown = value * t;
        const color = i === maxIndex ? theme.accent : theme.accent2;
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", height: rowH }}>
            <div style={{
              width: width * 0.16, textAlign: "right", paddingRight: width * 0.02,
              fontFamily: fontStack(theme.fontBody), fontSize: height * 0.034, color: theme.fg,
              opacity: Math.min(1, t * 3),
            }}>
              {labels[i] ?? ""}
            </div>
            <div style={{ flex: 1, height: barH, position: "relative" }}>
              <div style={{
                width: `${widthPct}%`, height: "100%", background: color,
                borderRadius: barH / 2, transition: "none",
              }} />
            </div>
            <div style={{
              width: width * 0.14, paddingLeft: width * 0.018,
              fontFamily: fontStack(theme.fontHead), fontWeight: 700, fontSize: height * 0.036,
              color: theme.fg, opacity: Math.min(1, t * 3),
            }}>
              {formatValue(shown, unit)}
            </div>
          </div>
        );
      })}
    </div>
  );
};

// --- line -----------------------------------------------------------------

const LineBody: React.FC<{
  labels: string[]; values: number[]; unit: string; theme: Theme;
  frame: number; landFrame: number; width: number; height: number; fps: number;
}> = ({ labels, values, unit, theme, frame, landFrame, width, height, fps }) => {
  const count = Math.max(1, values.length);
  const left = width * 0.1, right = width * 0.9, top = height * 0.3, bottom = height * 0.78;
  let vmax = Math.max(...values), vmin = Math.min(...values);
  if (vmax === vmin) { vmax += 1; vmin -= 1; }
  const span = vmax - vmin;
  vmax += span * 0.15;
  if (vmin < 0) vmin -= span * 0.15;
  const xAt = (i: number) => (count === 1 ? left : left + (right - left) * (i / (count - 1)));
  const yAt = (v: number) => bottom - (bottom - top) * ((v - vmin) / Math.max(1e-6, vmax - vmin));
  const points = values.map((v, i) => [xAt(i), yAt(v)] as [number, number]);
  const lineD = points.length > 1 ? `M ${points.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" L ")}` : "";
  const progress = interpolate(frame, [Math.round(landFrame * 0.1), landFrame], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });
  const areaD = points.length > 1
    ? `${lineD} L ${points[points.length - 1][0]},${bottom} L ${points[0][0]},${bottom} Z` : "";
  const peakIndex = values.reduce((best, v, i) => (v > values[best] ? i : best), 0);

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ position: "absolute" }}>
      {[0, 1, 2, 3, 4].map((g) => {
        const gy = top + (bottom - top) * (g / 4);
        return <line key={g} x1={left} x2={right} y1={gy} y2={gy} stroke={theme.border} strokeWidth={1} />;
      })}
      {areaD && (
        <path d={areaD} fill={tint(theme.accent2, 0.78)} opacity={progress}
              clipPath={`inset(0 ${(1 - progress) * 100}% 0 0)`} />
      )}
      {lineD && <path d={lineD} stroke={theme.accent} strokeWidth={4} fill="none" {...evolvePath(progress, lineD)} />}
      {points.map(([x, y], i) => {
        const threshold = count === 1 ? 0 : i / (count - 1);
        const show = progress >= threshold - 0.02;
        const pop = spring({ frame: show ? frame - landFrame * threshold * 0.1 - (landFrame * threshold) : -999,
          fps, config: { damping: 12 } });
        if (!show) return null;
        return (
          <g key={i} opacity={Math.min(1, pop + 0.4)}>
            <circle cx={x} cy={y} r={7} fill={theme.panel} stroke={theme.accent} strokeWidth={3} />
            {i === peakIndex && (
              <text x={x} y={y - 18} textAnchor="middle" fontFamily={fontStack(theme.fontHead)}
                    fontWeight={700} fontSize={height * 0.035} fill={theme.fg}>
                {formatValue(values[i], unit)}
              </text>
            )}
            {labels[i] && (
              <text x={x} y={bottom + height * 0.045} textAnchor="middle" fontFamily={fontStack(theme.fontBody)}
                    fontSize={height * 0.028} fill={theme.muted}>
                {labels[i]}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
};

// --- pie --------------------------------------------------------------------

const PieBody: React.FC<{
  labels: string[]; values: number[]; unit: string; theme: Theme;
  frame: number; landFrame: number; width: number; height: number;
}> = ({ labels, values, unit, theme, frame, landFrame, width, height }) => {
  const total = values.reduce((s, v) => s + Math.abs(v), 0) || 1;
  const accentIndex = values.reduce((best, v, i) => (Math.abs(v) > Math.abs(values[best]) ? i : best), 0);
  let factorI = 0;
  const colours = values.map((_, i) => {
    if (i === accentIndex) return theme.accent;
    const c = tint(theme.accent2, PIE_FACTORS[factorI % PIE_FACTORS.length]);
    factorI += 1;
    return c;
  });
  const radius = Math.min(width * 0.26, height * 0.3);
  const cx = width * 0.3, cy = height * 0.56;
  let cumulative = 0;
  const perSlice = landFrame / (values.length + 1);

  return (
    <>
      <svg width={width} height={height} style={{ position: "absolute" }}>
        <g transform={`translate(${cx - radius}, ${cy - radius})`}>
          {values.map((value, i) => {
            const share = Math.abs(value) / total;
            const start = Math.round(i * perSlice * 0.8);
            const t = interpolate(frame, [start, start + perSlice * 1.4], [0, 1], {
              extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
            });
            const pie = makePie({ radius, progress: share * t, rotation: cumulative * Math.PI * 2 });
            cumulative += share;
            return <path key={i} d={pie.path} fill={colours[i]} stroke={theme.bg} strokeWidth={2} />;
          })}
        </g>
      </svg>
      <div style={{ position: "absolute", left: width * 0.58, top: height * 0.28, right: width * 0.08 }}>
        {values.map((value, i) => {
          const start = Math.round(i * perSlice * 0.8);
          const t = interpolate(frame, [start, start + perSlice], [0, 1], {
            extrapolateLeft: "clamp", extrapolateRight: "clamp",
          });
          return (
            <div key={i} style={{
              display: "flex", alignItems: "center", marginBottom: height * 0.02, opacity: t,
              transform: `translateX(${(1 - t) * width * 0.03}px)`,
            }}>
              <div style={{ width: height * 0.03, height: height * 0.03, background: colours[i], marginRight: width * 0.012 }} />
              <div style={{ fontFamily: fontStack(theme.fontBody), fontSize: height * 0.032, color: theme.fg }}>
                {(labels[i] || "")} — {((Math.abs(value) / total) * 100).toFixed(0)}%
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
};

// --- years ruler --------------------------------------------------------

const YearsStrip: React.FC<{
  years: string[]; active?: string; theme: Theme;
  frame: number; landFrame: number; width: number; height: number; fps: number;
}> = ({ years, active, theme, frame, landFrame, width, height, fps }) => {
  const ordered = Array.from(new Set(years));
  const count = Math.max(1, ordered.length);
  const cy = height * 0.56;
  const left = width * 0.1, right = width * 0.9;
  const lineT = interpolate(frame, [0, landFrame * 0.5], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });
  const perTick = (landFrame * 0.7) / (count + 1);

  return (
    <svg width={width} height={height} style={{ position: "absolute" }}>
      <line x1={left} x2={left + (right - left) * lineT} y1={cy} y2={cy}
            stroke={theme.border} strokeWidth={4} />
      {ordered.map((year, i) => {
        const x = left + (right - left) * (count > 1 ? i / (count - 1) : 0.5);
        const isActive = active !== undefined && String(year) === String(active);
        const start = Math.round(i * perTick);
        const pop = spring({ frame: frame - start, fps, config: { damping: 12 } });
        const r = (isActive ? 15 : 8) * Math.max(0, Math.min(1, pop));
        return (
          <g key={i} opacity={Math.max(0, Math.min(1, pop))}>
            <circle cx={x} cy={cy} r={r} fill={isActive ? theme.accent : theme.accent2} />
            <text x={x} y={cy + r + height * 0.05} textAnchor="middle"
                  fontFamily={fontStack(isActive ? theme.fontHead : theme.fontBody)}
                  fontWeight={isActive ? 700 : 400} fontSize={isActive ? height * 0.045 : height * 0.03}
                  fill={isActive ? theme.fg : theme.muted}>
              {year}
            </text>
          </g>
        );
      })}
    </svg>
  );
};
