// `map` beat -> presentation/motion_data.py::build_map -> this composition
// (MOTION_GRAPHICS_PLAN.md M2). An orthographic globe turns from a neutral
// starting view to the named place and zooms in, by ~40% of the clip
// (Architecture: "key information lands by ~40% ... slow live drift after").
// A country/state place fills with the highlight colour; every place gets a
// pin (the first with a pulse ring) and a label that slides in. A `route`
// beat (several places in order) also draws a dashed line between them with
// a dot travelling along it.
//
// The world outline is `public/world/land-110m.json` (world-atlas, copied in
// by scripts/copy_world_atlas.mjs on every `npm install`) -- the one network
// fetch Remotion compositions commonly do from their own bundle, blocked on
// with delayRender/continueRender so the browser never captures a frame
// before the land shape is ready. The highlighted country/state polygon
// itself comes from Python already clipped to just that place and rounded
// (maps.motion_feature) -- small enough to travel as an ordinary prop.
import { geoOrthographic, geoPath } from "d3-geo";
import React, { useEffect, useMemo, useState } from "react";
import {
  AbsoluteFill, Easing, continueRender, delayRender, interpolate, spring,
  staticFile, useCurrentFrame, useVideoConfig,
} from "remotion";
import { evolvePath, getLength, getPointAtLength } from "@remotion/paths";
import { feature } from "topojson-client";
import { fontFaceCss, fontStack } from "../fonts";
import { Theme, withTheme } from "../theme";

export type MotionPlace = { name: string; lat: number; lon: number; kind?: string };
// A country/state outline or a state/province line inside the current view
// (presentation/motion_data.py::_motion_borders) -- same shape as `highlight`,
// just many of them, and `state` marks the thinner/fainter province lines so
// they read as a clean explainer map instead of a blank beige field (M2b).
export type MotionBorder = { rings: number[][][]; state?: boolean };
export type MotionMapProps = {
  places?: MotionPlace[];
  highlight?: { rings: number[][][] } | null;
  borders?: MotionBorder[];
  neighbours?: MotionPlace[];
  label?: string;
  route?: boolean;
  theme?: Partial<Theme>;
};

export const motionMapDefaultProps: MotionMapProps = {
  places: [{ name: "Jaipur", lat: 26.92, lon: 75.81, kind: "city" }],
  highlight: null,
  borders: [],
  neighbours: [],
  label: "Jaipur",
  route: false,
  theme: {},
};

// A fixed, neutral starting view (mid-Atlantic-ish) -- every render turns
// from the same place, so the "turns to the place" motion always reads the
// same way regardless of where the target is.
const START_ROTATE: [number, number] = [20, -10];

type LandFC = GeoJSON.FeatureCollection | null;

let landCache: LandFC = null;
let landPromise: Promise<LandFC> | null = null;

function useLand(): LandFC {
  const [land, setLand] = useState<LandFC>(landCache);
  const [handle] = useState<number | null>(() => (landCache ? null : delayRender("motion-map-land")));
  useEffect(() => {
    if (landCache) return;
    if (!landPromise) {
      landPromise = fetch(staticFile("world/land-110m.json"))
        .then((r) => r.json())
        .then((topo: any) => feature(topo, topo.objects.land) as unknown as GeoJSON.FeatureCollection)
        .catch(() => null);
    }
    landPromise.then((f) => {
      landCache = f;
      setLand(f);
      if (handle !== null) continueRender(handle);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return land;
}

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "#000000") || [];
  return [parseInt(m[1] || "00", 16), parseInt(m[2] || "00", 16), parseInt(m[3] || "00", 16)];
}

// How many degrees the zoomed-in view should frame, mirroring
// maps.py::_span_for exactly (same constants, same padding factors) so the
// motion render and the PIL fallback zoom to the same place the same way --
// without this a route between nearby cities (or a single city) stays at
// the "turns to the place" globe scale and its pins/labels end up crowded
// on top of each other.
const SPAN_CITY_DEG = 9.0;
const SPAN_STATE_DEG = 14.0;
const SPAN_COUNTRY_MIN_DEG = 12.0;
const MAX_SPAN_DEG = 140.0;

function spanDegFor(places: MotionPlace[], highlight: MotionMapProps["highlight"], aspect: number): number {
  if (highlight && highlight.rings.length) {
    let lon0 = Infinity, lat0 = Infinity, lon1 = -Infinity, lat1 = -Infinity;
    for (const ring of highlight.rings) {
      for (const [lon, lat] of ring) {
        if (lon < lon0) lon0 = lon;
        if (lon > lon1) lon1 = lon;
        if (lat < lat0) lat0 = lat;
        if (lat > lat1) lat1 = lat;
      }
    }
    const cos = Math.max(0.2, Math.cos(((lat0 + lat1) / 2) * Math.PI / 180));
    const minSpan = places[0]?.kind === "country" ? SPAN_COUNTRY_MIN_DEG : SPAN_STATE_DEG;
    const span = Math.max(minSpan, (lon1 - lon0) * cos * 1.35, (lat1 - lat0) * 1.35 * aspect);
    return Math.min(MAX_SPAN_DEG, span);
  }
  if (places.length <= 1) {
    return SPAN_CITY_DEG * Math.sqrt(aspect / (16 / 9));
  }
  const lons = places.map((p) => p.lon), lats = places.map((p) => p.lat);
  const cos = Math.max(0.2, Math.cos((lats.reduce((a, b) => a + b, 0) / lats.length) * Math.PI / 180));
  const span = Math.max(SPAN_CITY_DEG, (Math.max(...lons) - Math.min(...lons)) * cos * 1.8,
    (Math.max(...lats) - Math.min(...lats)) * 1.8 * aspect);
  return Math.min(MAX_SPAN_DEG, span);
}

function mix(a: string, b: string, t: number): string {
  const [ar, ag, ab] = hexToRgb(a);
  const [br, bg, bb] = hexToRgb(b);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${bl})`;
}

export const MotionMap: React.FC<MotionMapProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const land = useLand();
  const places = props.places && props.places.length ? props.places : motionMapDefaultProps.places!;
  const isRoute = Boolean(props.route) && places.length > 1;
  // M2b (review): a route's `label` is the beat's own caption ("Jaipur to
  // Agra"), not a place name -- it replaced the primary pin's own name
  // instead of sitting beside the map. Every pin now shows its own place
  // name; the caption becomes a small title chip, and only when it actually
  // says something the place names don't.
  const placeNameSet = useMemo(
    () => new Set(places.map((p) => p.name.trim().toLowerCase())),
    [places],
  );
  const chipText = props.label && !placeNameSet.has(props.label.trim().toLowerCase())
    ? props.label : null;

  const target: [number, number] = useMemo(() => {
    const lon = places.reduce((s, p) => s + p.lon, 0) / places.length;
    const lat = places.reduce((s, p) => s + p.lat, 0) / places.length;
    return [-lon, -lat];
  }, [places]);

  const landFrame = Math.max(1, Math.min(durationInFrames - 1, Math.round(durationInFrames * 0.4)));
  const turnT = interpolate(frame, [0, landFrame], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });
  // A slow continuing drift after landing -- never finishes, so a long hold
  // never looks frozen (same idea as KineticText's background drift).
  const driftT = interpolate(frame, [landFrame, Math.max(landFrame + 1, durationInFrames)], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });

  const rotation: [number, number, number] = [
    START_ROTATE[0] + (target[0] - START_ROTATE[0]) * turnT + driftT * 2.5,
    START_ROTATE[1] + (target[1] - START_ROTATE[1]) * turnT,
    0,
  ];
  const minDim = Math.min(width, height);
  // The globe starts at a fixed, neutral full-earth scale and turns in to a
  // scale that frames every place with room around it (spanDegFor), not a
  // flat zoom factor -- a route between nearby cities needs far more zoom
  // than a single country to keep its pins and labels apart on screen.
  const startScale = minDim * 0.46;
  const spanDeg = useMemo(() => spanDegFor(places, props.highlight ?? null, width / height),
    [places, props.highlight, width, height]);
  const zoomedScale = Math.min(minDim * 14, (minDim * 0.6 / spanDeg) * (180 / Math.PI));
  const targetScale = Math.max(startScale, zoomedScale);
  const scale = startScale + (targetScale - startScale) * turnT;

  const projection = useMemo(
    () => geoOrthographic().clipAngle(90).translate([width / 2, height / 2]),
    [width, height],
  );
  projection.rotate(rotation).scale(scale);
  const path = useMemo(() => geoPath(projection), [projection]);

  const sphereD = path({ type: "Sphere" } as any) || "";
  const landD = land ? path(land as any) || "" : "";
  const highlightGeometry = props.highlight && props.highlight.rings.length
    ? ({ type: "Polygon", coordinates: props.highlight.rings } as GeoJSON.Polygon)
    : null;
  const highlightD = highlightGeometry ? path(highlightGeometry) || "" : "";
  const highlightColor = mix(theme.land, theme.accent, 0.45);

  // Country borders + state/province lines (M2b: the clean map had none of
  // either, reading as a blank beige field). Each travels as the same
  // `{rings}` shape `highlight` already uses, so it is projected the same
  // way -- `path` itself reflects the latest rotate/scale every frame
  // because `projection` is mutated in place above, exactly like sphereD/
  // landD/highlightD already rely on.
  const borderPaths = (props.borders || []).map((b) => ({
    d: b.rings.length ? path({ type: "Polygon", coordinates: b.rings } as any) || "" : "",
    state: Boolean(b.state),
  }));
  const neighbourPoints = (props.neighbours || [])
    .map((n) => ({ n, pt: projection([n.lon, n.lat]) }))
    .filter((e): e is { n: MotionPlace; pt: [number, number] } => e.pt !== null);

  const projected = places.map((p) => projection([p.lon, p.lat]));

  const routeD = useMemo(() => {
    if (!isRoute) return "";
    const pts = projected.filter((p): p is [number, number] => p !== null);
    if (pts.length < 2) return "";
    return `M ${pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" L ")}`;
  }, [isRoute, projected]);
  const routeLen = routeD ? getLength(routeD) : 0;
  const routeT = interpolate(frame, [landFrame, Math.max(landFrame + 1, durationInFrames * 0.85)], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.quad),
  });
  const travelling = routeD && routeLen > 0 ? getPointAtLength(routeD, routeLen * routeT) : null;

  const pinEntrance = spring({ frame: Math.max(0, frame - Math.round(landFrame * 0.75)), fps,
    config: { damping: 11, mass: 0.6, stiffness: 160 } });
  const labelT = interpolate(frame, [landFrame, landFrame + fps * 0.5], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });
  // A pulse ring that loops continuously once the pin has landed.
  const pulseCycle = fps * 1.4;
  const pulseT = frame > landFrame ? ((frame - landFrame) % pulseCycle) / pulseCycle : 0;

  // M2b (review): the route line and the secondary pins read as too thin at
  // 1080p. Sized off `minDim` so a smaller render (gallery previews, tests
  // at 320x180) still gets a visible line, but 1080p always clears the
  // floor: >= 6px stroke with a soft glow, >= 18px pins.
  const routeStroke = Math.max(6, minDim * 0.0056);
  const routeGlow = routeStroke * 2.4;

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg, overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <path d={sphereD} fill={theme.water} stroke={theme.border} strokeWidth={1} />
        {landD && <path d={landD} fill={theme.land} stroke={theme.border} strokeWidth={1} />}
        {/* Country borders (M2b: the clean map had none, reading as a blank
           beige field). Drawn before the highlight so the highlighted
           place's own tint and outline still sit on top of it. */}
        {borderPaths.filter((b) => !b.state && b.d).map((b, i) => (
          <path key={`border-${i}`} d={b.d} fill="none" stroke={theme.border}
                strokeWidth={1.6} strokeOpacity={0.9} />
        ))}
        {highlightD && <path d={highlightD} fill={highlightColor} stroke={theme.accent} strokeWidth={2} />}
        {/* State/province lines -- thinner and fainter than a country border,
           only sent once zoomed in enough to read (motion_data.py's own
           span <= 40 gate, mirroring maps.py's PIL fallback). */}
        {borderPaths.filter((b) => b.state && b.d).map((b, i) => (
          <path key={`state-${i}`} d={b.d} fill="none" stroke={theme.border}
                strokeWidth={1} strokeOpacity={0.5} />
        ))}
        {/* Neighbour cities give the zoomed-in place context, same as the
           PIL map's own dots (maps.render_map). */}
        {neighbourPoints.map(({ n, pt }, i) => (
          <g key={`neighbour-${n.name}-${i}`} opacity={0.85}>
            <circle cx={pt[0]} cy={pt[1]} r={Math.max(3, minDim * 0.0035)} fill={theme.muted} />
            <text x={pt[0] + minDim * 0.012} y={pt[1] + minDim * 0.006}
                  fontFamily={fontStack(theme.fontBody)} fontWeight={500}
                  fontSize={minDim * 0.016} fill={theme.muted}
                  style={{ paintOrder: "stroke", stroke: theme.bg, strokeWidth: 3 }}>
              {n.name}
            </text>
          </g>
        ))}
        {theme.noir && <path d={sphereD} fill="black" opacity={0.18} />}

        {isRoute && routeD && (
          <>
            <clipPath id="motionMapRouteReveal">
              <path d={routeD} stroke="white" strokeWidth={40} fill="none"
                    strokeLinecap="round" strokeLinejoin="round"
                    {...evolvePath(routeT, routeD)} />
            </clipPath>
            <filter id="motionMapRouteGlow" x="-60%" y="-60%" width="220%" height="220%">
              <feGaussianBlur stdDeviation={routeGlow * 0.5} />
            </filter>
            {/* A soft glow under the crisp line -- same clip-reveal, wider
               and blurred, at low opacity -- so the route reads clearly
               against both the pale "clean" look and the dark "noir" one. */}
            <path d={routeD} stroke={theme.accent2} strokeWidth={routeGlow} fill="none"
                  strokeLinecap="round" opacity={0.5} filter="url(#motionMapRouteGlow)"
                  clipPath="url(#motionMapRouteReveal)" />
            <path d={routeD} stroke={theme.accent2} strokeWidth={routeStroke} fill="none"
                  strokeDasharray={`${routeStroke * 4.5} ${routeStroke * 3}`} strokeLinecap="round"
                  clipPath="url(#motionMapRouteReveal)" />
            {travelling && (
              <circle cx={travelling.x} cy={travelling.y} r={routeStroke * 1.3} fill={theme.accent2}
                      stroke={theme.bg} strokeWidth={2} />
            )}
          </>
        )}

        {projected.map((pt, i) => {
          if (!pt) return null;
          const isPrimary = i === 0;
          const stopT = i / Math.max(1, places.length - 1);
          const reveal = isPrimary ? pinEntrance
            : (isRoute && routeLen > 0
                ? interpolate(routeT, [Math.max(0, stopT - 0.08), stopT], [0, 1], {
                    extrapolateLeft: "clamp", extrapolateRight: "clamp",
                  })
                : pinEntrance);
          if (reveal <= 0.01) return null;
          // >= 18px across at 1080p even settled (no spring overshoot needed).
          const r = (isPrimary ? 16 : 10) * Math.min(1.3, Math.max(0, reveal));
          return (
            <g key={places[i].name + i} opacity={Math.min(1, reveal)}>
              {isPrimary && (
                <circle cx={pt[0]} cy={pt[1]} r={r + 10 + pulseT * 26}
                        fill="none" stroke={theme.accent} strokeWidth={2}
                        opacity={Math.max(0, 0.55 * (1 - pulseT))} />
              )}
              <circle cx={pt[0]} cy={pt[1]} r={r} fill={theme.accent} stroke={theme.fg} strokeWidth={1.5} />
              {(
                // Every stop gets its own place name (M2b review: only the
                // first stop was labelled, and with the beat's caption text
                // instead of the city name -- that caption is now a
                // separate title chip, see below).
                <text
                  x={isPrimary && isRoute ? pt[0] : pt[0] + r + 10}
                  y={isPrimary && isRoute ? pt[1] - r - 14 : pt[1] + 5}
                  textAnchor={isPrimary && isRoute ? "middle" : "start"}
                  fontFamily={fontStack(theme.fontHead)} fontWeight={700}
                  fontSize={isPrimary ? minDim * 0.045 : minDim * 0.028}
                  fill={theme.fg}
                  opacity={isPrimary ? labelT : Math.min(1, reveal)}
                  style={{
                    paintOrder: "stroke",
                    stroke: theme.bg,
                    strokeWidth: isPrimary ? 6 : 4,
                    transform: isPrimary && isRoute
                      ? `translateY(${(1 - labelT) * minDim * 0.03}px)`
                      : `translateX(${(1 - (isPrimary ? labelT : reveal)) * -minDim * 0.03}px)`,
                  }}>
                  {places[i].name}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {/* The beat's own caption ("Jaipur to Agra") as a small title chip,
         top-left -- only when it says something none of the place names
         already say (M2b review). */}
      {chipText && (
        <div
          style={{
            position: "absolute", top: minDim * 0.05, left: minDim * 0.05,
            padding: `${minDim * 0.014}px ${minDim * 0.024}px`,
            background: theme.panel, color: theme.fg,
            fontFamily: fontStack(theme.fontHead), fontWeight: 700,
            fontSize: minDim * 0.026, borderRadius: minDim * 0.01,
            opacity: labelT,
            boxShadow: "0 2px 10px rgba(0,0,0,0.35)",
          }}
        >
          {chipText}
        </div>
      )}
    </AbsoluteFill>
  );
};
