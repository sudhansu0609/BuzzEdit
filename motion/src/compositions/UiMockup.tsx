// `ui_mockup` beat -> presentation/motion_ui.py::build_ui_mockup -> this
// composition (MOTION_GRAPHICS_PLAN.md M10). Directive `[ui: <variant> |
// text=<body> | name=<display name> | handle=<@handle> | meta=<e.g. 2h .
// 14.2K likes> | seconds=N]` (presentation/script.py::directive_beats). Five
// generic variants -- no real platform logos, colours or names:
//   post:     round avatar w/ initials, name/handle, body text, like/reply/
//             share counts animating up.
//   forum:    title, upvote arrow + count, 1-2 nested reply bubbles from
//             `text` split on " // ".
//   chat:     a phone with bubbles from `text` split on " // " (a leading
//             ">" marks the other side), typing dots before each, timestamps.
//   search:   a search bar where `text` types itself letter by letter, then
//             3 suggestion rows drop down (from `meta` split on " ; ").
//   headline: a news strip with `text` stamped in, `name` as the outlet.
// The card floats on the theme background with a slow push-in; key content
// lands by ~40% of the duration (Architecture). Engagement/upvote counts and
// suggestion rows a bare directive does not supply are generic placeholder
// chrome, seeded off the beat's own text so a re-render never flickers.
import React, { useMemo } from "react";
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { ChevronUp, Heart, MessageCircle, Search, Share2 } from "lucide-react";
import { fontFaceCss, fontStack } from "../fonts";
import { Theme, withTheme } from "../theme";

export type UiVariant = "post" | "forum" | "chat" | "search" | "headline";
export type UiMockupProps = {
  variant?: UiVariant;
  text?: string;
  name?: string;
  handle?: string;
  meta?: string;
  theme?: Partial<Theme>;
};

const VARIANTS: UiVariant[] = ["post", "forum", "chat", "search", "headline"];

export const uiMockupDefaultProps: UiMockupProps = {
  variant: "post",
  text: "Finally wrapped up the big project we've been building all month.",
  name: "Rhea Kapoor",
  handle: "@rhea",
  meta: "2h . 14.2K likes",
  theme: {},
};

// --- small shared helpers ---------------------------------------------------

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

// A character-estimate text-fit (the same technique FlowDiagram/IconList's
// own box/label text uses) generalised to any number of lines, bounded by a
// height rather than a fixed line cap -- so a longer body never clips.
function fitBlock(text: string, maxWidth: number, maxHeight: number, maxSize: number,
                  minSize: number, lineHeight = 1.3): { lines: string[]; size: number } {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length === 0) return { lines: [""], size: maxSize };
  for (let size = maxSize; size >= minSize; size -= 1) {
    const lines = wrapAtSize(words, size, maxWidth);
    if (lines.length * size * lineHeight <= maxHeight) return { lines, size };
  }
  const size = minSize;
  return { lines: wrapAtSize(words, size, maxWidth), size };
}

function formatCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1).replace(/\.0$/, "")}K`;
  return Math.max(0, Math.round(n)).toString();
}

function parseMetaCount(meta: string): number | null {
  if (!meta) return null;
  const km = meta.match(/(\d+(?:\.\d+)?)\s*([KM])\b/);
  if (km) return parseFloat(km[1]) * (km[2] === "M" ? 1_000_000 : 1000);
  const bare = meta.match(/\b(\d{2,})\b/);
  return bare ? parseInt(bare[1], 10) : null;
}

function stringHash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h;
}
function hashUnit(n: number): number {
  const s = Math.sin(n) * 43758.5453123;
  return s - Math.floor(s);
}
function syntheticCount(seed: number, min: number, max: number): number {
  return Math.round(min + hashUnit(seed) * (max - min));
}

function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return "U";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

// --- the composition ---------------------------------------------------------

export const UiMockup: React.FC<UiMockupProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const variant: UiVariant = props.variant && VARIANTS.includes(props.variant) ? props.variant : "post";
  const text = (props.text || "").trim() || uiMockupDefaultProps.text!;
  const name = (props.name || "").trim();
  const handle = (props.handle || "").trim();
  const meta = (props.meta || "").trim();
  const seed = useMemo(() => stringHash(`${variant}:${name}:${text}`), [variant, name, text]);

  const landFrame = Math.max(1, Math.min(durationInFrames - 1, Math.round(durationInFrames * 0.4)));
  const entrance = spring({ frame, fps, config: { damping: 15, mass: 0.8, stiffness: 140 } });
  const opacity = Math.min(1, entrance + 0.15);
  const popScale = interpolate(entrance, [0, 1], [0.88, 1]);
  const pushIn = interpolate(frame, [0, Math.max(1, durationInFrames - 1)], [1, 1.035],
    { easing: Easing.inOut(Easing.ease) });
  const translateY = interpolate(entrance, [0, 1], [height * 0.04, 0]);

  const shared = { theme, frame, fps, landFrame, seed, width, height };
  let body: React.ReactNode;
  switch (variant) {
    case "forum": body = <ForumCard {...shared} text={text} name={name} meta={meta} />; break;
    case "chat": body = <ChatPhone {...shared} text={text} name={name} meta={meta} />; break;
    case "search": body = <SearchBar {...shared} text={text} meta={meta} />; break;
    case "headline": body = <Headline {...shared} text={text} name={name} meta={meta} />; break;
    default: body = <PostCard {...shared} text={text} name={name} handle={handle} meta={meta} />;
  }

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", display: "flex" }}>
        <div style={{ transform: `translateY(${translateY}px) scale(${popScale * pushIn})`, opacity }}>
          {body}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

type Shared = { theme: Theme; frame: number; fps: number; landFrame: number; seed: number; width: number; height: number; };

// --- post --------------------------------------------------------------------

const EngagementItem: React.FC<{ Icon: React.ComponentType<any>; value: number; countT: number; theme: Theme; size: number; }>
  = ({ Icon, value, countT, theme, size }) => (
  <div style={{ display: "flex", alignItems: "center", gap: size * 0.3 }}>
    <Icon color={theme.muted} size={size} strokeWidth={2} />
    <span style={{ fontFamily: fontStack(theme.fontBody), color: theme.muted, fontSize: size * 0.8, fontWeight: 600 }}>
      {formatCompact(value * countT)}
    </span>
  </div>
);

const PostCard: React.FC<Shared & { text: string; name: string; handle: string; meta: string; }> = (
  { theme, text, name, handle, meta, frame, fps, landFrame, seed, width, height },
) => {
  // 55-65% of the frame width (MOTION_GRAPHICS_PLAN.md M10 review -- 0.46
  // read as a small, hard-to-read card in a real render), with body/name/
  // count sizes floored in absolute px so they stay legible at 1080p
  // regardless of how long the text is (fitBlock still shrinks past that
  // floor only for a block too long to fit at all).
  const cardW = width * 0.6;
  const avatarD = cardW * 0.11;
  const displayName = name || "Someone";
  const displayHandle = handle ? (handle.startsWith("@") ? handle : `@${handle}`) : "";
  const { lines, size } = fitBlock(text, cardW * 0.9, height * 0.22,
    Math.max(40, cardW * 0.036), Math.max(28, cardW * 0.025));
  const likeTarget = parseMetaCount(meta) ?? syntheticCount(seed, 40, 900);
  const replyTarget = syntheticCount(seed + 101, 2, 60);
  const shareTarget = syntheticCount(seed + 202, 1, 30);
  const countT = interpolate(frame, [Math.round(fps * 0.15), landFrame], [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const iconSize = Math.max(35, cardW * 0.032);

  return (
    <div style={{
      width: cardW, background: theme.panel, borderRadius: cardW * 0.03,
      border: `1px solid ${theme.border}`, padding: cardW * 0.05,
      boxShadow: "0 24px 48px rgba(0,0,0,0.35)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: cardW * 0.03 }}>
        <div style={{
          width: avatarD, height: avatarD, borderRadius: "50%", background: theme.accent,
          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
        }}>
          <span style={{ fontFamily: fontStack(theme.fontBody), fontWeight: 700, color: theme.bg, fontSize: avatarD * 0.42 }}>
            {initials(displayName)}
          </span>
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontFamily: fontStack(theme.fontBody), fontWeight: 700, color: theme.fg, fontSize: Math.max(34, cardW * 0.032) }}>
            {displayName}{displayHandle && <span style={{ color: theme.muted, fontWeight: 400 }}> {displayHandle}</span>}
          </div>
          {meta && (
            <div style={{ fontFamily: fontStack(theme.fontBody), color: theme.muted, fontSize: cardW * 0.024 }}>{meta}</div>
          )}
        </div>
      </div>
      <div style={{ marginTop: cardW * 0.045, lineHeight: 1.3 }}>
        {lines.map((line, i) => (
          <div key={i} style={{ fontFamily: fontStack(theme.fontBody), color: theme.fg, fontSize: size, fontWeight: 500 }}>
            {line}
          </div>
        ))}
      </div>
      <div style={{
        marginTop: cardW * 0.05, display: "flex", gap: cardW * 0.09,
        borderTop: `1px solid ${theme.border}`, paddingTop: cardW * 0.035,
      }}>
        <EngagementItem Icon={Heart} value={likeTarget} countT={countT} theme={theme} size={iconSize} />
        <EngagementItem Icon={MessageCircle} value={replyTarget} countT={countT} theme={theme} size={iconSize} />
        <EngagementItem Icon={Share2} value={shareTarget} countT={countT} theme={theme} size={iconSize} />
      </div>
    </div>
  );
};

// --- forum -------------------------------------------------------------------

const ReplyBubble: React.FC<{ text: string; depth: number; theme: Theme; cardW: number; frame: number; fps: number; start: number; }>
  = ({ text, depth, theme, cardW, frame, fps, start }) => {
  const since = frame - start;
  const pop = spring({ frame: Math.max(0, since), fps, config: { damping: 14, mass: 0.6, stiffness: 150 } });
  const opacity = since < 0 ? 0 : Math.min(1, pop + 0.1);
  const { lines, size } = fitBlock(text, cardW * 0.62 - depth * cardW * 0.06, cardW * 0.12, cardW * 0.022, cardW * 0.016);
  return (
    <div style={{
      marginLeft: depth * cardW * 0.08, paddingLeft: cardW * 0.025, borderLeft: `2px solid ${theme.border}`,
      opacity, transform: `translateX(${(1 - Math.max(0, pop)) * 10}px)`,
    }}>
      {lines.map((l, i) => (
        <div key={i} style={{ fontFamily: fontStack(theme.fontBody), color: theme.fg, fontSize: size }}>{l}</div>
      ))}
    </div>
  );
};

const ForumCard: React.FC<Shared & { text: string; name: string; meta: string; }> = (
  { theme, text, name, meta, frame, fps, landFrame, seed, width },
) => {
  // Same 55-65% band and absolute px floors as PostCard (MOTION_GRAPHICS_PLAN.md
  // M10 review) -- the title is this card's own "body" text.
  const cardW = width * 0.6;
  const segments = text.split("//").map((s) => s.trim()).filter(Boolean);
  const title = segments[0] || "Discussion thread";
  const replies = segments.slice(1, 3);
  const upvotes = parseMetaCount(meta) ?? syntheticCount(seed, 5, 400);
  const countT = interpolate(frame, [Math.round(fps * 0.15), landFrame], [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const { lines: titleLines, size: titleSize } = fitBlock(title, cardW * 0.72, cardW * 0.18,
    Math.max(40, cardW * 0.036), Math.max(28, cardW * 0.025));

  return (
    <div style={{
      width: cardW, background: theme.panel, borderRadius: cardW * 0.025, border: `1px solid ${theme.border}`,
      padding: cardW * 0.045, boxShadow: "0 24px 48px rgba(0,0,0,0.35)",
    }}>
      <div style={{ display: "flex", gap: cardW * 0.03 }}>
        <div style={{
          display: "flex", flexDirection: "column", alignItems: "center", background: theme.bg,
          borderRadius: cardW * 0.02, padding: `${cardW * 0.015}px ${cardW * 0.018}px`, flexShrink: 0,
        }}>
          <ChevronUp color={theme.accent} size={cardW * 0.035} strokeWidth={2.6} />
          <span style={{ fontFamily: fontStack(theme.fontBody), fontWeight: 700, color: theme.fg, fontSize: Math.max(28, cardW * 0.026) }}>
            {formatCompact(upvotes * countT)}
          </span>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          {name && (
            <div style={{ fontFamily: fontStack(theme.fontBody), color: theme.muted, fontSize: cardW * 0.02, marginBottom: cardW * 0.01 }}>
              Posted by {name}
            </div>
          )}
          {titleLines.map((l, i) => (
            <div key={i} style={{ fontFamily: fontStack(theme.fontHead), fontWeight: 700, color: theme.fg, fontSize: titleSize }}>
              {l}
            </div>
          ))}
        </div>
      </div>
      {replies.length > 0 && (
        <div style={{ marginTop: cardW * 0.04, display: "flex", flexDirection: "column", gap: cardW * 0.025 }}>
          {replies.map((reply, i) => (
            <ReplyBubble key={i} text={reply} depth={i} theme={theme} cardW={cardW} frame={frame} fps={fps}
              start={landFrame * 0.5 + i * fps * 0.3} />
          ))}
        </div>
      )}
    </div>
  );
};

// --- chat --------------------------------------------------------------------

const ChatBubble: React.FC<{
  side: "me" | "other"; text: string; theme: Theme; fps: number; frame: number; start: number; phoneW: number; index: number;
}> = ({ side, text, theme, fps, frame, start, phoneW, index }) => {
  const typingDur = fps * 0.5;
  const since = frame - start;
  const showTyping = since >= 0 && since < typingDur;
  const bubbleSince = since - typingDur;
  const pop = spring({ frame: Math.max(0, bubbleSince), fps, config: { damping: 14, mass: 0.6, stiffness: 160 } });
  const visible = bubbleSince >= 0;
  const opacity = visible ? Math.min(1, pop + 0.15) : 0;
  const align = side === "me" ? "flex-end" : "flex-start";
  const bg = side === "me" ? theme.accent : theme.panel;
  const color = side === "me" ? theme.bg : theme.fg;
  // Bubble text floored at >=34px (MOTION_GRAPHICS_PLAN.md M10 review --
  // 0.052/0.036 of a phone already narrow-by-design read as tiny bubbles).
  const { lines, size } = fitBlock(text, phoneW * 0.6, phoneW * 0.4, Math.max(34, phoneW * 0.08), Math.max(24, phoneW * 0.055));
  const minutes = 10 + index * 2;
  const timestamp = `${String(9 + Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: align }}>
      {showTyping && (
        <div style={{ display: "flex", gap: 4, background: theme.panel, borderRadius: 14, padding: `${phoneW * 0.025}px ${phoneW * 0.035}px` }}>
          {[0, 1, 2].map((d) => (
            <div key={d} style={{
              width: 6, height: 6, borderRadius: 3, background: theme.muted,
              opacity: 0.4 + 0.6 * Math.abs(Math.sin((frame - start) * 0.6 + d)),
            }} />
          ))}
        </div>
      )}
      {visible && (
        <>
          <div style={{
            background: bg, color, borderRadius: 14, padding: `${phoneW * 0.025}px ${phoneW * 0.04}px`,
            opacity, transform: `scale(${Math.max(0.6, pop)})`,
            transformOrigin: side === "me" ? "right bottom" : "left bottom", maxWidth: phoneW * 0.68,
          }}>
            {lines.map((l, i) => (
              <div key={i} style={{ fontFamily: fontStack(theme.fontBody), fontSize: size, lineHeight: 1.25 }}>{l}</div>
            ))}
          </div>
          <div style={{ fontFamily: fontStack(theme.fontBody), color: theme.muted, fontSize: phoneW * 0.032, marginTop: 2, opacity }}>
            {timestamp}
          </div>
        </>
      )}
    </div>
  );
};

const ChatPhone: React.FC<Shared & { text: string; name: string; meta: string; }> = (
  { theme, text, name, meta, frame, fps, landFrame, height },
) => {
  const phoneH = height * 0.82;
  const phoneW = phoneH * 0.52;
  const raw = text.split("//").map((s) => s.trim()).filter(Boolean);
  const segments = raw.length ? raw : [text];
  const bubbles = segments.slice(0, 6).map((s) => (
    s.startsWith(">") ? { side: "other" as const, text: s.slice(1).trim() } : { side: "me" as const, text: s }
  ));
  const perBubble = Math.max(fps * 0.6, landFrame / Math.max(1, bubbles.length + 1));
  const contactName = name || "Contact";
  const status = meta || "Online";

  return (
    <div style={{
      width: phoneW, height: phoneH, background: theme.bg, border: `3px solid ${theme.border}`,
      borderRadius: phoneW * 0.12, boxShadow: "0 30px 60px rgba(0,0,0,0.45)",
      overflow: "hidden", display: "flex", flexDirection: "column",
    }}>
      <div style={{ height: phoneH * 0.03, display: "flex", justifyContent: "center", alignItems: "flex-end", paddingBottom: 4 }}>
        <div style={{ width: phoneW * 0.28, height: phoneH * 0.015, background: theme.border, borderRadius: 8 }} />
      </div>
      <div style={{
        display: "flex", alignItems: "center", gap: phoneW * 0.06,
        padding: `${phoneH * 0.015}px ${phoneW * 0.08}px`, borderBottom: `1px solid ${theme.border}`,
      }}>
        <div style={{
          width: phoneW * 0.14, height: phoneW * 0.14, borderRadius: "50%", background: theme.accent,
          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
        }}>
          <span style={{ fontFamily: fontStack(theme.fontBody), fontWeight: 700, color: theme.bg, fontSize: phoneW * 0.07 }}>
            {initials(contactName)}
          </span>
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontFamily: fontStack(theme.fontBody), fontWeight: 700, color: theme.fg, fontSize: phoneW * 0.07 }}>
            {contactName}
          </div>
          <div style={{ fontFamily: fontStack(theme.fontBody), color: theme.muted, fontSize: phoneW * 0.05 }}>{status}</div>
        </div>
      </div>
      <div style={{
        flex: 1, display: "flex", flexDirection: "column", justifyContent: "flex-end",
        gap: phoneH * 0.018, padding: `${phoneH * 0.02}px ${phoneW * 0.07}px`, overflow: "hidden",
      }}>
        {bubbles.map((b, i) => (
          <ChatBubble key={i} side={b.side} text={b.text} theme={theme} fps={fps} frame={frame}
            start={i * perBubble} phoneW={phoneW} index={i} />
        ))}
      </div>
    </div>
  );
};

// --- search --------------------------------------------------------------------

const SuggestionRow: React.FC<{ text: string; theme: Theme; frame: number; fps: number; start: number; barW: number; first: boolean; }>
  = ({ text, theme, frame, fps, start, barW, first }) => {
  const since = frame - start;
  const t = spring({ frame: Math.max(0, since), fps, config: { damping: 16, mass: 0.6, stiffness: 160 } });
  const opacity = since < 0 ? 0 : Math.min(1, t + 0.2);
  const { lines, size } = fitBlock(text, barW * 0.82, barW * 0.12, Math.max(34, barW * 0.028), Math.max(24, barW * 0.018));
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: barW * 0.025, padding: `${barW * 0.018}px ${barW * 0.03}px`,
      opacity, transform: `translateY(${(1 - Math.max(0, t)) * -6}px)`,
      borderTop: first ? "none" : `1px solid ${theme.border}`,
    }}>
      <Search color={theme.muted} size={barW * 0.03} strokeWidth={2} />
      <div style={{ fontFamily: fontStack(theme.fontBody), color: theme.fg, fontSize: size }}>{lines[0]}</div>
    </div>
  );
};

const SearchBar: React.FC<Shared & { text: string; meta: string; }> = ({ theme, text, meta, frame, fps, landFrame, width }) => {
  // ~70% of the frame width (MOTION_GRAPHICS_PLAN.md M10 review -- 0.52 read
  // as a small bar with small text).
  const barW = width * 0.7;
  const typeEnd = Math.min(landFrame, Math.round(fps * 2));
  const typeT = interpolate(frame, [0, Math.max(1, typeEnd)], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const typeChars = Math.max(0, Math.round(text.length * typeT));
  const shown = text.slice(0, typeChars);
  const cursorOn = Math.floor(frame / 10) % 2 === 0;
  const suggestions = (meta ? meta.split(";").map((s) => s.trim()).filter(Boolean)
    : [`${text} near me`, `${text} reviews`, `${text} today`]).slice(0, 3);
  const dropStart = typeEnd + Math.round(fps * 0.2);
  // The dropdown panel itself (not just its rows) fades in just ahead of the
  // first suggestion -- otherwise an empty bordered box sits under the bar
  // for the whole typing animation, which reads as a stray blank card.
  const panelOpacity = interpolate(frame, [Math.max(0, dropStart - fps * 0.25), dropStart], [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <div style={{ width: barW }}>
      <div style={{
        display: "flex", alignItems: "center", gap: barW * 0.025, background: theme.panel,
        border: `1px solid ${theme.border}`, borderRadius: 999, padding: `${barW * 0.022}px ${barW * 0.03}px`,
      }}>
        <Search color={theme.muted} size={barW * 0.045} strokeWidth={2.2} />
        <div style={{
          fontFamily: fontStack(theme.fontBody), color: theme.fg, fontSize: Math.max(44, barW * 0.035), flex: 1,
          whiteSpace: "nowrap", overflow: "hidden",
        }}>
          {shown}{cursorOn && <span style={{ color: theme.accent }}>|</span>}
        </div>
      </div>
      <div style={{
        marginTop: barW * 0.02, background: theme.panel, border: `1px solid ${theme.border}`,
        borderRadius: barW * 0.03, overflow: "hidden", opacity: panelOpacity,
      }}>
        {suggestions.map((s, i) => (
          <SuggestionRow key={i} text={s} theme={theme} frame={frame} fps={fps}
            start={dropStart + i * fps * 0.18} barW={barW} first={i === 0} />
        ))}
      </div>
    </div>
  );
};

// --- headline ------------------------------------------------------------

const Headline: React.FC<Shared & { text: string; name: string; meta: string; }> = ({ theme, text, name, meta, width }) => {
  const cardW = width * 0.68;
  const outlet = name || "Daily Desk";
  const kicker = meta || "BREAKING";
  const { lines, size } = fitBlock(text, cardW * 0.94, cardW * 0.42, cardW * 0.075, cardW * 0.04, 1.08);

  return (
    <div style={{ width: cardW, textAlign: "center" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: cardW * 0.015, marginBottom: cardW * 0.025 }}>
        <div style={{ width: cardW * 0.28, height: 2, background: theme.border }} />
        <span style={{
          fontFamily: fontStack(theme.fontBody), color: theme.fg, fontWeight: 700,
          fontSize: cardW * 0.026, letterSpacing: 2,
        }}>
          {outlet.toUpperCase()}
        </span>
        <div style={{ width: cardW * 0.28, height: 2, background: theme.border }} />
      </div>
      <div style={{
        display: "inline-block", background: theme.accent, color: theme.bg, fontFamily: fontStack(theme.fontBody),
        fontWeight: 700, fontSize: cardW * 0.026, padding: `${cardW * 0.01}px ${cardW * 0.025}px`,
        borderRadius: 4, marginBottom: cardW * 0.02, letterSpacing: 1.5,
      }}>
        {kicker.toUpperCase()}
      </div>
      <div>
        {lines.map((l, i) => (
          <div key={i} style={{ fontFamily: fontStack(theme.fontHead), fontWeight: 700, color: theme.fg, fontSize: size, lineHeight: 1.15 }}>
            {l}
          </div>
        ))}
      </div>
    </div>
  );
};
