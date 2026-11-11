// `icon_list` beat -> presentation/motion_diagrams.py::build_icon_list ->
// this composition (MOTION_GRAPHICS_PLAN.md M9). Directive `[icons: label ;
// label ; ... | mode=list|pros_cons | seconds=N]`
// (presentation/script.py::directive_beats). `list` mode is 2-6 rows (a
// 2-column grid above 4) of a bundled Lucide icon in an accent circle plus a
// label, staggered pop-ins with a small spring overshoot; `pros_cons` splits
// the same items into two columns by the `+`/`-` prefix the directive's
// labels carried (presentation/motion_diagrams.py strips it before this),
// each row's icon badged with a green tick (pro) or red cross (con). Icon
// choice travels as a Lucide icon slug per item (`icon`); `iconComponentFor`
// resolves it through lucide-react's own dynamic `icons` map, defaulting to
// a neutral dot for an unknown name. Key build lands by ~40% of the clip
// (Architecture); a slow vertical drift keeps the rest of the hold alive.
import React, { useMemo } from "react";
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Check, icons as LUCIDE_ICONS, X } from "lucide-react";
import { fontFaceCss, fontStack } from "../fonts";
import { Theme, withTheme } from "../theme";

export type IconListMode = "list" | "pros_cons";
export type IconListItem = { label: string; icon?: string; side?: "pro" | "con" | null };
export type IconListProps = {
  items?: IconListItem[];
  mode?: IconListMode;
  theme?: Partial<Theme>;
};

export const iconListDefaultProps: IconListProps = {
  items: [
    { label: "Fast setup", icon: "zap" },
    { label: "Lower cost", icon: "wallet" },
    { label: "Great support", icon: "users" },
  ],
  mode: "list",
  theme: {},
};

const GOOD_COLOR = "#2FAE60";
const BAD_COLOR = "#D0433A";

// --- text fit: shrink the font until a label reads in at most 2 lines ------
// (a character-count width estimate, not a canvas measurement, the same
// technique FlowDiagram's own box text uses -- a longer label like "Takes
// longer to learn" used to hit the fixed font size + `-webkit-line-clamp: 2`
// combination's own ellipsis and lose its last word; this always wraps to
// fit instead of ever truncating).
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

function fitLabel(text: string, maxWidth: number, maxSize: number, minSize: number): { lines: string[]; size: number } {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return { lines: [""], size: maxSize };
  for (let size = maxSize; size >= minSize; size -= 1) {
    const lines = wrapAtSize(words, size, maxWidth);
    if (lines.length <= 2) return { lines, size };
  }
  const mid = Math.max(1, Math.ceil(words.length / 2));
  return { lines: [words.slice(0, mid).join(" "), words.slice(mid).join(" ")], size: minSize };
}

function pascalCase(slug: string): string {
  return slug.split("-").filter(Boolean).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join("");
}

function iconComponentFor(name?: string): React.ComponentType<{
  color?: string; size?: number; strokeWidth?: number;
}> {
  const key = pascalCase(name || "circle-dot");
  const found = (LUCIDE_ICONS as Record<string, any>)[key];
  return found || LUCIDE_ICONS.CircleDot;
}

export const IconList: React.FC<IconListProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const mode: IconListMode = props.mode === "pros_cons" ? "pros_cons" : "list";
  const items = useMemo(() => {
    const list = (props.items || []).filter((it) => it && it.label && it.label.trim()).slice(0, 6);
    return list.length >= 2 ? list : iconListDefaultProps.items!;
  }, [props.items]);

  const landFrame = Math.max(1, Math.min(durationInFrames - 1, Math.round(durationInFrames * 0.4)));
  const driftEnd = Math.max(landFrame + 1, durationInFrames);
  const driftT = interpolate(frame, [landFrame, driftEnd], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.ease),
  });
  const driftPanPx = driftT * Math.max(4, height * 0.01);

  const body = mode === "pros_cons"
    ? <ProsConsBody items={items} theme={theme} landFrame={landFrame} fps={fps} width={width} height={height} />
    : <ListBody items={items} theme={theme} landFrame={landFrame} fps={fps} width={width} height={height} />;

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>
      <AbsoluteFill style={{ transform: `translateY(${driftPanPx}px)` }}>
        {body}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// --- list mode: 2-6 rows, a 2-column grid above 4 --------------------------

const ListBody: React.FC<{
  items: IconListItem[]; theme: Theme; landFrame: number; fps: number; width: number; height: number;
}> = ({ items, theme, landFrame, fps, width, height }) => {
  const n = items.length;
  const cols = n > 4 ? 2 : 1;
  const rows = Math.ceil(n / cols);
  const perItem = landFrame / (n + 1);
  const cellW = width * (cols === 2 ? 0.42 : 0.72);
  const cellH = (height * 0.76) / rows;

  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", display: "flex" }}>
      <div style={{
        display: "grid", gridTemplateColumns: `repeat(${cols}, ${cellW}px)`,
        gridAutoRows: `${cellH}px`, columnGap: width * 0.04, rowGap: cellH * 0.08,
      }}>
        {items.map((item, i) => {
          const start = Math.round(i * perItem * 0.8);
          return <IconRow key={i} item={item} start={start} theme={theme} fps={fps} cellH={cellH} cellW={cellW} />;
        })}
      </div>
    </AbsoluteFill>
  );
};

const IconRow: React.FC<{
  item: IconListItem; start: number; theme: Theme; fps: number; cellH: number; cellW: number;
}> = ({ item, start, theme, fps, cellH, cellW }) => {
  const frame = useCurrentFrame();
  const since = frame - start;
  // Low damping -> a small, deliberate overshoot as each row pops in.
  const pop = spring({ frame: since, fps, config: { damping: 9, mass: 0.5, stiffness: 160 } });
  const opacity = since < 0 ? 0 : Math.min(1, pop + 0.15);
  const scale = Math.max(0, pop);
  const Icon = iconComponentFor(item.icon);
  const circle = cellH * 0.6;
  const gap = cellH * 0.16;
  const textMaxW = Math.max(40, cellW - circle - gap) * 0.96;
  const { lines, size } = useMemo(
    () => fitLabel(item.label, textMaxW, Math.max(10, Math.round(cellH * 0.24)), Math.max(8, Math.round(cellH * 0.13))),
    [item.label, textMaxW, cellH],
  );

  return (
    <div style={{
      display: "flex", alignItems: "center", gap, opacity,
      transform: `scale(${scale})`, transformOrigin: "left center",
    }}>
      <div style={{
        width: circle, height: circle, borderRadius: circle / 2, background: theme.accent,
        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
      }}>
        <Icon color={theme.bg} size={circle * 0.56} strokeWidth={2.4} />
      </div>
      <div style={{ lineHeight: 1.15 }}>
        {lines.map((line, i) => (
          <div key={i} style={{
            fontFamily: fontStack(theme.fontBody), fontWeight: 600, color: theme.fg, fontSize: size,
          }}>
            {line}
          </div>
        ))}
      </div>
    </div>
  );
};

// --- pros_cons mode: two columns, tick/cross badges -------------------------

const ProsConsBody: React.FC<{
  items: IconListItem[]; theme: Theme; landFrame: number; fps: number; width: number; height: number;
}> = ({ items, theme, landFrame, fps, width, height }) => {
  const pros = items.filter((it) => it.side !== "con");
  const cons = items.filter((it) => it.side === "con");
  const total = Math.max(1, pros.length + cons.length);
  const perItem = landFrame / (total + 1);
  const colW = width * 0.4;
  const rowH = (height * 0.74) / Math.max(1, Math.max(pros.length, cons.length));

  let order = 0;
  const prosRows = pros.map((item) => {
    const start = Math.round(order * perItem * 0.8);
    order += 1;
    return { item, start };
  });
  const consRows = cons.map((item) => {
    const start = Math.round(order * perItem * 0.8);
    order += 1;
    return { item, start };
  });

  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", display: "flex", flexDirection: "row" }}>
      <div style={{
        width: colW, display: "flex", flexDirection: "column", justifyContent: "center",
        gap: rowH * 0.12, paddingRight: width * 0.02,
      }}>
        {prosRows.map(({ item, start }, i) => (
          <BadgeRow key={i} item={item} start={start} theme={theme} fps={fps} rowH={rowH}
                    colW={colW - width * 0.02} good />
        ))}
      </div>
      <div style={{ width: 2, height: height * 0.68, background: theme.border, opacity: 0.6 }} />
      <div style={{
        width: colW, display: "flex", flexDirection: "column", justifyContent: "center",
        gap: rowH * 0.12, paddingLeft: width * 0.02,
      }}>
        {consRows.map(({ item, start }, i) => (
          <BadgeRow key={i} item={item} start={start} theme={theme} fps={fps} rowH={rowH}
                    colW={colW - width * 0.02} good={false} />
        ))}
      </div>
    </AbsoluteFill>
  );
};

const BadgeRow: React.FC<{
  item: IconListItem; start: number; theme: Theme; fps: number; rowH: number; colW: number; good: boolean;
}> = ({ item, start, theme, fps, rowH, colW, good }) => {
  const frame = useCurrentFrame();
  const since = frame - start;
  const pop = spring({ frame: since, fps, config: { damping: 9, mass: 0.5, stiffness: 160 } });
  const opacity = since < 0 ? 0 : Math.min(1, pop + 0.15);
  const scale = Math.max(0, pop);
  const Icon = iconComponentFor(item.icon);
  const circle = rowH * 0.5;
  const badge = rowH * 0.3;
  const gap = rowH * 0.14;
  const textMaxW = Math.max(40, colW - circle - gap) * 0.96;
  const { lines, size } = useMemo(
    () => fitLabel(item.label, textMaxW, Math.max(10, Math.round(rowH * 0.22)), Math.max(8, Math.round(rowH * 0.12))),
    [item.label, textMaxW, rowH],
  );

  return (
    <div style={{
      display: "flex", alignItems: "center", gap, opacity,
      transform: `scale(${scale})`, transformOrigin: "left center",
    }}>
      <div style={{
        width: circle, height: circle, borderRadius: circle / 2, background: theme.panel,
        border: `2px solid ${theme.border}`, display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0, position: "relative",
      }}>
        <Icon color={theme.fg} size={circle * 0.56} strokeWidth={2.2} />
        <div style={{
          position: "absolute", right: -badge * 0.28, bottom: -badge * 0.28, width: badge, height: badge,
          borderRadius: badge / 2, background: good ? GOOD_COLOR : BAD_COLOR,
          display: "flex", alignItems: "center", justifyContent: "center", border: `2px solid ${theme.bg}`,
        }}>
          {good
            ? <Check color="#fff" size={badge * 0.62} strokeWidth={3} />
            : <X color="#fff" size={badge * 0.62} strokeWidth={3} />}
        </div>
      </div>
      <div style={{ lineHeight: 1.15 }}>
        {lines.map((line, i) => (
          <div key={i} style={{
            fontFamily: fontStack(theme.fontBody), fontWeight: 600, color: theme.fg, fontSize: size,
          }}>
            {line}
          </div>
        ))}
      </div>
    </div>
  );
};
