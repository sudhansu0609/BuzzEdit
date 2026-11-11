// M1 (MOTION_GRAPHICS_PLAN.md): the one motion composition that ships with the
// engine itself. Directive `[kinetic: text with *marks* | style=slam|stack|
// reveal|typewriter]` (presentation/script.py::directive_beats) -> a beat ->
// presentation/motion.py builds these props. Words spring in; a `*marked*`
// word lands in theme.accent with an extra punch and an underline sweep. The
// background carries a slow gradient drift and, for noir/grain themes, a
// faint film-grain texture; a paper-textured theme (hand-drawn art styles)
// gets a soft paper-blotch overlay instead of a flat colour.
import React, { useMemo } from "react";
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { fontFaceCss, fontStack } from "../fonts";
import { Theme, withTheme } from "../theme";

export type KineticWord = { text: string; mark?: boolean };
export type KineticTextStyle = "slam" | "stack" | "reveal" | "typewriter";

export type KineticTextProps = {
  words?: KineticWord[];
  text?: string;
  style?: KineticTextStyle;
  theme?: Partial<Theme>;
};

const VALID_STYLES: KineticTextStyle[] = ["slam", "stack", "reveal", "typewriter"];

/** "This is *huge*" -> [{text:"This"},{text:"is"},{text:"huge",mark:true}].
 * Mirrors presentation/motion.py::_parse_marks -- kept here too so a
 * hand-written `text` prop (gallery samples, tests) works without the
 * Python side. */
export function parseMarks(text: string): KineticWord[] {
  const out: KineticWord[] = [];
  for (const raw of text.trim().split(/\s+/).filter(Boolean)) {
    const mark = raw.length > 2 && raw.startsWith("*") && raw.endsWith("*");
    out.push({ text: mark ? raw.slice(1, -1) : raw, mark });
  }
  return out;
}

export const kineticTextDefaultProps: KineticTextProps = {
  words: [{ text: "Kinetic" }, { text: "Text" }, { text: "Demo", mark: true }],
  style: "slam",
  theme: {},
};

export const KineticText: React.FC<KineticTextProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const words = useMemo(() => {
    if (props.words && props.words.length) return props.words;
    if (props.text) return parseMarks(props.text);
    return kineticTextDefaultProps.words!;
  }, [props.words, props.text]);
  const style: KineticTextStyle = VALID_STYLES.includes(props.style as KineticTextStyle)
    ? (props.style as KineticTextStyle)
    : "slam";

  // Every word has landed by ~40% of the clip; the rest is a slow live drift.
  // Clamped strictly below durationInFrames so the drift's interpolate() below
  // always gets a non-degenerate range, even on a 1-second selftest clip.
  const landFrame = Math.max(1, Math.min(durationInFrames - 1, Math.round(durationInFrames * 0.4)));
  const perWord = words.length > 0 ? landFrame / (words.length + 1) : landFrame;

  const driftEnd = Math.max(landFrame + 1, durationInFrames);
  const driftT = interpolate(frame, [landFrame, driftEnd], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.ease),
  });
  const driftScale = 1 + driftT * 0.035;
  const driftPanPx = driftT * Math.max(6, width * 0.01);

  // A slow background gradient drift over the whole duration -- never
  // finishes, so a long hold never looks static.
  const bgT = (frame / Math.max(1, durationInFrames)) % 1;
  const angle = 115 + bgT * 35;
  const bgShade = theme.noir ? theme.panel : theme.accent2;

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>
      <AbsoluteFill
        style={{
          backgroundImage: `linear-gradient(${angle}deg, ${theme.bg} 0%, ${bgShade} 100%)`,
          opacity: 0.9,
        }}
      />
      {theme.grain && <GrainOverlay seed={1} opacity={0.08} />}
      {theme.paper && <PaperOverlay fg={theme.muted} />}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          padding: `0 ${width * 0.08}px`,
          transform: `scale(${driftScale}) translateX(${driftPanPx}px)`,
        }}
      >
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: style === "stack" ? "flex-start" : "center",
            flexDirection: style === "stack" ? "column" : "row",
            justifyContent: "center",
            columnGap: width * 0.018,
            rowGap: height * 0.02,
          }}
        >
          {words.map((word, index) => (
            <Word
              key={`${index}-${word.text}`}
              word={word}
              index={index}
              landStartFrame={Math.round((index + 1) * perWord * 0.6)}
              theme={theme}
              style={style}
              fps={fps}
              widthPx={width}
              heightPx={height}
            />
          ))}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const Word: React.FC<{
  word: KineticWord;
  index: number;
  landStartFrame: number;
  theme: Theme;
  style: KineticTextStyle;
  fps: number;
  widthPx: number;
  heightPx: number;
}> = ({ word, index, landStartFrame, theme, style, fps, widthPx, heightPx }) => {
  const frame = useCurrentFrame();
  const local = frame - landStartFrame;
  const size = Math.round(heightPx * 0.09);
  const color = word.mark ? theme.accent : theme.fg;

  const entrance = spring({
    frame: local,
    fps,
    config: { damping: 14, mass: 0.6, stiffness: 140 },
  });

  let transform = "none";
  let opacity = entrance;
  let clipPath: string | undefined;

  if (style === "slam") {
    const scale = interpolate(entrance, [0, 1], [2.4, 1]);
    transform = `scale(${scale})`;
  } else if (style === "stack") {
    const y = interpolate(entrance, [0, 1], [heightPx * 0.05, 0]);
    transform = `translateY(${y}px)`;
  } else if (style === "reveal") {
    const reveal = interpolate(local, [0, fps * 0.5], [0, 100], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: Easing.out(Easing.cubic),
    });
    clipPath = `inset(0 ${100 - reveal}% 0 0)`;
    opacity = local < 0 ? 0 : 1;
  } else {
    // typewriter: the word pops in essentially instantly, one after another --
    // the stagger between words (landStartFrame) reads as the typing.
    opacity = local < 0 ? 0 : 1;
    transform = `translateY(${interpolate(Math.min(1, Math.max(0, local / (fps * 0.12))), [0, 1], [heightPx * 0.015, 0])}px)`;
  }

  // The punch + underline sweep: a marked word gets a brief extra pop right
  // as it lands, and a bar sweeps in under it just after.
  const punch = word.mark
    ? interpolate(local, [0, fps * 0.12, fps * 0.3], [1, 1.16, 1], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      })
    : 1;
  const underline = word.mark
    ? interpolate(local, [fps * 0.1, fps * 0.45], [0, 100], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
        easing: Easing.out(Easing.quad),
      })
    : 0;

  return (
    <div
      style={{
        position: "relative",
        display: "inline-block",
        opacity,
        transform: `${transform} scale(${punch})`,
        transformOrigin: "center",
        clipPath,
      }}
    >
      <span
        style={{
          fontFamily: fontStack(theme.fontHead),
          fontWeight: 700,
          fontSize: size,
          color,
          lineHeight: 1.05,
          whiteSpace: "pre",
        }}
      >
        {word.text}
      </span>
      {word.mark && (
        <div
          style={{
            position: "absolute",
            left: 0,
            bottom: -size * 0.12,
            height: Math.max(3, size * 0.07),
            width: `${underline}%`,
            background: theme.accent2,
            borderRadius: 3,
          }}
        />
      )}
    </div>
  );
};

/** A faint, static film-grain texture (SVG fractal noise) -- cheap to render
 * and reads as texture rather than as a photograph, which is the point for a
 * noir/horror theme. */
const GrainOverlay: React.FC<{ seed: number; opacity: number }> = ({ seed, opacity }) => (
  <AbsoluteFill style={{ opacity, mixBlendMode: "overlay" }}>
    <svg width="100%" height="100%">
      <filter id={`grain-${seed}`}>
        <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" seed={seed} stitchTiles="stitch" />
        <feColorMatrix type="saturate" values="0" />
      </filter>
      <rect width="100%" height="100%" filter={`url(#grain-${seed})`} />
    </svg>
  </AbsoluteFill>
);

/** A soft, mottled paper look for hand-drawn art styles (stick figure,
 * cartoon, painterly, illustrated) -- blotches rather than a flat fill. */
const PaperOverlay: React.FC<{ fg: string }> = ({ fg }) => (
  <AbsoluteFill style={{ opacity: 0.12, mixBlendMode: "multiply" }}>
    <svg width="100%" height="100%">
      <filter id="paper">
        <feTurbulence type="fractalNoise" baseFrequency="0.015" numOctaves="3" seed={7} />
        <feColorMatrix type="matrix" values={`0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.6 0`} />
      </filter>
      <rect width="100%" height="100%" fill={fg} filter="url(#paper)" />
    </svg>
  </AbsoluteFill>
);
