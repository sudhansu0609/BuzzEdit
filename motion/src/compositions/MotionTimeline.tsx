// `timeline` beat -> presentation/motion_data.py::build_timeline -> this
// composition (MOTION_GRAPHICS_PLAN.md M2). Mirrors graphics.py::render_timeline's
// own layout (a big headline, a caption, then a ruler of ticks with the
// active one picked out) but animated: the headline pops in first, the ruler
// draws left to right, ticks pop in along it staggered, and a marker ring
// slides from the first tick to the active one, landing by ~40% of the clip
// and pulsing gently for the rest (the "live drift" every other composition
// here uses).
import { evolvePath, getLength } from "@remotion/paths";
import React, { useMemo } from "react";
import {
  AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig,
} from "remotion";
import { fontFaceCss, fontStack } from "../fonts";
import { Theme, withTheme } from "../theme";

export type MotionTimelineProps = {
  ticks?: string[];
  active?: number;
  headline?: string;
  caption?: string;
  theme?: Partial<Theme>;
};

export const motionTimelineDefaultProps: MotionTimelineProps = {
  ticks: ["1947", "1984", "2001", "2016", "2024"],
  active: 4,
  headline: "Five decades of recorded sightings",
  caption: "Collected from police and newspaper archives",
  theme: {},
};

export const MotionTimeline: React.FC<MotionTimelineProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const ticks = props.ticks && props.ticks.length ? props.ticks : motionTimelineDefaultProps.ticks!;
  const count = Math.max(1, ticks.length);
  const activeIndex = Math.max(0, Math.min(count - 1, props.active ?? count - 1));
  const headline = props.headline || "";
  const caption = props.caption || "";

  const landFrame = Math.max(1, Math.min(durationInFrames - 1, Math.round(durationInFrames * 0.4)));
  const driftEnd = Math.max(landFrame + 1, durationInFrames);
  const driftT = interpolate(frame, [landFrame, driftEnd], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.ease),
  });

  const cy = height * 0.74;
  const left = width * 0.1, right = width * 0.9;
  const lineD = `M ${left},${cy} L ${right},${cy}`;
  const lineLen = getLength(lineD);
  const drawT = interpolate(frame, [0, Math.round(landFrame * 0.55)], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });

  const xAt = (i: number) => left + (right - left) * (count > 1 ? i / (count - 1) : 0.5);
  const perTick = (landFrame * 0.7) / (count + 1);

  const markerT = interpolate(frame, [Math.round(landFrame * 0.5), landFrame], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.quad),
  });
  const markerStartX = xAt(0);
  const markerEndX = xAt(activeIndex);
  const markerX = markerStartX + (markerEndX - markerStartX) * markerT;
  const pulseCycle = fps * 1.6;
  const pulseT = frame > landFrame ? ((frame - landFrame) % pulseCycle) / pulseCycle : 0;

  const headlinePop = spring({ frame: frame - Math.round(fps * 0.1), fps,
    config: { damping: 13, mass: 0.7, stiffness: 150 } });
  const headlineScale = interpolate(headlinePop, [0, 1], [0.85, 1]);
  const captionT = interpolate(frame, [Math.round(landFrame * 0.45), Math.round(landFrame * 0.8)], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });

  const headlineSize = useMemo(() => {
    const words = headline.split(/\s+/).filter(Boolean).length;
    const base = height * 0.085;
    return words > 8 ? base * 0.72 : words > 5 ? base * 0.85 : base;
  }, [headline, height]);

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>

      <div style={{
        position: "absolute", top: height * 0.1, left: width * 0.08, right: width * 0.08,
        textAlign: "center", opacity: Math.min(1, headlinePop + 0.2),
        transform: `scale(${headlineScale}) translateY(${(1 - Math.min(1, headlinePop)) * height * 0.03}px)`,
      }}>
        <div style={{
          fontFamily: fontStack(theme.fontHead), fontWeight: 700, color: theme.fg,
          fontSize: headlineSize, lineHeight: 1.12,
        }}>
          {headline}
        </div>
      </div>

      {caption && (
        <div style={{
          position: "absolute", top: height * 0.4, left: width * 0.1, right: width * 0.1,
          textAlign: "center", opacity: captionT,
          transform: `translateY(${(1 - captionT) * height * 0.02}px)`,
          fontFamily: fontStack(theme.fontBody), fontSize: height * 0.034, color: theme.muted,
        }}>
          {caption}
        </div>
      )}

      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}
           style={{ position: "absolute", transform: `translateY(${driftT * height * 0.006}px)` }}>
        <path d={lineD} stroke={theme.border} strokeWidth={Math.max(2, height / 400)} fill="none"
              {...evolvePath(drawT, lineD)} />

        {ticks.map((label, i) => {
          const x = xAt(i);
          const isActive = i === activeIndex;
          const start = Math.round(i * perTick);
          const pop = spring({ frame: frame - start, fps, config: { damping: 12 } });
          const clamped = Math.max(0, Math.min(1, pop));
          const r = (isActive ? height * 0.016 : height * 0.009) * clamped;
          return (
            <g key={i} opacity={clamped}>
              <circle cx={x} cy={cy} r={r} fill={isActive ? theme.accent : theme.accent2} />
              <text x={x} y={cy + height * 0.05} textAnchor="middle"
                    fontFamily={fontStack(isActive ? theme.fontHead : theme.fontBody)}
                    fontWeight={isActive ? 700 : 400}
                    fontSize={isActive ? height * 0.04 : height * 0.028}
                    fill={isActive ? theme.fg : theme.muted}>
                {label}
              </text>
            </g>
          );
        })}

        {lineLen > 0 && (
          <g opacity={markerT > 0.01 ? 1 : 0}>
            <circle cx={markerX} cy={cy} r={height * 0.026 + pulseT * height * 0.012}
                    fill="none" stroke={theme.accent} strokeWidth={2}
                    opacity={Math.max(0, 0.6 * (1 - pulseT))} />
            <circle cx={markerX} cy={cy} r={height * 0.013} fill={theme.accent}
                    stroke={theme.bg} strokeWidth={1.5} />
          </g>
        )}
      </svg>
    </AbsoluteFill>
  );
};
