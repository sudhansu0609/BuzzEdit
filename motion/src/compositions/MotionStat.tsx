// `stat_callout` beat -> presentation/motion_data.py::build_stat -> this
// composition (MOTION_GRAPHICS_PLAN.md M2). Mirrors cards.py's own counter
// (`item.text.counter`: 0 -> `data["value"]`, `prefix`/`suffix`/`decimals`,
// landing at 0.4 of the beat's duration) -- a big number counts up, then its
// label fades in below it, with the same "key info by ~40%, slow drift
// after" shape every other M1/M2 composition uses.
import React from "react";
import {
  AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig,
} from "remotion";
import { fontFaceCss, fontStack } from "../fonts";
import { Theme, withTheme } from "../theme";

export type MotionStatProps = {
  value?: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  label?: string;
  theme?: Partial<Theme>;
};

export const motionStatDefaultProps: MotionStatProps = {
  value: 2847,
  prefix: "",
  suffix: "+",
  decimals: 0,
  label: "Reported sightings since 1990",
  theme: {},
};

function formatCount(value: number, decimals: number): string {
  const fixed = value.toFixed(Math.max(0, decimals));
  const [whole, frac] = fixed.split(".");
  const withCommas = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return frac ? `${withCommas}.${frac}` : withCommas;
}

export const MotionStat: React.FC<MotionStatProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const target = typeof props.value === "number" ? props.value : motionStatDefaultProps.value!;
  const prefix = props.prefix || "";
  const suffix = props.suffix || "";
  const decimals = Math.max(0, props.decimals ?? 0);
  const label = props.label || "";

  const landFrame = Math.max(1, Math.min(durationInFrames - 1, Math.round(durationInFrames * 0.4)));
  const driftEnd = Math.max(landFrame + 1, durationInFrames);
  const driftT = interpolate(frame, [landFrame, driftEnd], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.ease),
  });

  // The whole block pops in first, then the number counts up to its target
  // by landFrame -- the pop and the count overlap a little so the number is
  // never seen sitting at a cold "0" before it starts climbing.
  const entrance = spring({ frame, fps, config: { damping: 13, mass: 0.7, stiffness: 150 } });
  const countStart = Math.round(fps * 0.1);
  const countT = interpolate(frame, [countStart, landFrame], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });
  const shown = target * countT;

  const labelT = interpolate(frame, [Math.round(landFrame * 0.55), landFrame + Math.round(fps * 0.35)],
    [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const underlineT = interpolate(frame, [landFrame, landFrame + fps * 0.5], [0, 100], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.quad),
  });
  // A slow pulse behind the number once it has landed -- never finishes, so a
  // long hold on the stat never looks static (same idea as MotionMap's pin).
  const pulseCycle = fps * 2.2;
  const pulseT = frame > landFrame ? ((frame - landFrame) % pulseCycle) / pulseCycle : 0;
  const glowOpacity = frame > landFrame ? 0.35 * (1 - Math.abs(pulseT - 0.5) * 2) : 0;

  const scale = interpolate(entrance, [0, 1], [0.7, 1]) * (1 + driftT * 0.02);

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div style={{ position: "relative", transform: `scale(${scale})`, textAlign: "center",
                     opacity: Math.min(1, entrance + 0.2) }}>
          <div style={{
            position: "absolute", left: "50%", top: "50%", width: height * 0.9, height: height * 0.9,
            transform: "translate(-50%, -62%)", borderRadius: "50%",
            background: `radial-gradient(circle, ${theme.accent} 0%, transparent 70%)`,
            opacity: glowOpacity,
          }} />
          <div style={{
            fontFamily: fontStack(theme.fontHead), fontWeight: 700, color: theme.fg,
            fontSize: height * 0.16, lineHeight: 1, whiteSpace: "nowrap",
            paintOrder: "stroke", WebkitTextStroke: `${height * 0.004}px ${theme.bg}`,
          }}>
            <span style={{ color: theme.accent }}>{prefix}</span>
            {formatCount(shown, decimals)}
            <span style={{ color: theme.accent }}>{suffix}</span>
          </div>
          <div style={{ position: "relative", marginTop: height * 0.02 }}>
            <div style={{
              fontFamily: fontStack(theme.fontBody), fontWeight: 600, color: theme.muted,
              fontSize: height * 0.042, opacity: labelT, maxWidth: width * 0.7,
              transform: `translateY(${(1 - labelT) * height * 0.03}px)`,
            }}>
              {label}
            </div>
            <div style={{
              height: Math.max(3, height * 0.006), width: `${underlineT}%`, maxWidth: "100%",
              background: theme.accent2, margin: `${height * 0.012}px auto 0`, borderRadius: 3,
            }} />
          </div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
