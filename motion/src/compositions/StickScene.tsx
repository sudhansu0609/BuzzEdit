// `stick` beat -> presentation/motion_stick.py::build_stick_scene -> this
// composition (MOTION_GRAPHICS_PLAN.md M7). An SVG-rigged stick figure (head
// circle, torso line, 2-segment arms and legs) acts out one of twelve poses
// -- idle, walk, run, point, shrug, think, cheer, fall, sit, scared, swim,
// climb (an unknown action falls back to idle) -- with a simple line-drawn
// prop and a whiteboard/paper/night/underwater ground. Every stroke "boils"
// (a seeded 1-2px jitter recomputed every 2 frames), the hand-drawn way a
// whiteboard doodle never holds perfectly still. The key pose lands by ~40%
// of the clip (Architecture) for the one-shot gestures; the locomotion poses
// (walk/run/swim/climb) move across the frame the whole time instead. The
// `scene` text from the directive never appears on screen -- Python already
// turned it into just the `ground` prop (presentation/motion_stick.py).
import React, { useMemo } from "react";
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { fontFaceCss, handFontStack } from "../fonts";
import { Theme, withTheme } from "../theme";
// The ONE shared vocabulary (STICK_WORLD_PLAN.md "The shared vocabulary"):
// body/head/held/hair piece names and the `eras` table -- also read from
// Python (presentation/motion_stick.py::_load_vocab) against the same file.
// Remotion's webpack bundler imports JSON natively; tsconfig.json carries
// `resolveJsonModule` for `tsc --noEmit`.
import stickWorldData from "../stick_world.json";

export type StickAction =
  | "idle" | "walk" | "run" | "point" | "shrug" | "think" | "cheer"
  | "fall" | "sit" | "scared" | "swim" | "climb";
export type StickProp =
  | "none" | "ship" | "wave" | "house" | "tree" | "phone" | "money" | "car"
  | "mountain" | "door" | "table" | "book" | "sun" | "moon" | "rain";
export type StickGround = "whiteboard" | "paper" | "night" | "underwater";

// Stick World (STICK_WORLD_PLAN.md SW1): a figure dressed for a period and,
// optionally, standing over a generated picture instead of today's plain
// ground. All four fields are optional and additive -- a beat that sets
// none of them renders exactly as it did before this section existed.
export type StickWorldPerson = { outfit?: string };

export type StickSceneProps = {
  action?: StickAction;
  prop?: StickProp;
  figures?: number;
  ground?: StickGround;
  caption?: string;
  theme?: Partial<Theme>;
  era?: string;                      // stick_world.json eras key: palette + default outfits
  people?: StickWorldPerson[];       // 1-3 outfit specs; overrides `figures` when given
  background?: string;               // data: URI (JPEG) of the generated scene picture
  bgLuma?: number;                   // 0-255 mean luma of the picture's lower-middle band
};

const ACTIONS: StickAction[] = [
  "idle", "walk", "run", "point", "shrug", "think", "cheer",
  "fall", "sit", "scared", "swim", "climb",
];
const PROP_NAMES: StickProp[] = [
  "none", "ship", "wave", "house", "tree", "phone", "money", "car",
  "mountain", "door", "table", "book", "sun", "moon", "rain",
];
const GROUNDS: StickGround[] = ["whiteboard", "paper", "night", "underwater"];

// Shared by Ground (where the line is drawn) and the figure/prop grounding
// math below, so a standing figure's feet and a house's doorstep both land
// exactly on the line the eye actually sees.
const GROUND_LINE_FRAC = 0.86;

// Local-space y (as a fraction of the prop's own size `s`, same units as
// PropArt's P() helper) of each grounded prop's lowest drawn point, so the
// scene can sit it exactly on the ground line instead of floating partway
// up the frame (review finding 7). sun/moon/rain are sky props and are
// never grounded.
const PROP_GROUND_OFFSET: Partial<Record<StickProp, number>> = {
  house: 0.56, tree: 0.5, car: 0.2, mountain: 0.3, ship: 0.35,
  phone: 0.4, money: 0.18, door: 0.5, table: 0.4, book: 0.22, wave: 0.18,
};

export const stickSceneDefaultProps: StickSceneProps = {
  action: "cheer",
  prop: "none",
  figures: 1,
  ground: "whiteboard",
  caption: "",
  theme: {},
};

// --- hand-drawn "boiling" jitter --------------------------------------------
// A deterministic pseudo-random offset, re-rolled every 2 frames (the bucket),
// so every stroke wobbles the way a hand-held marker redraws a line instead
// of holding dead still. seed identifies the point (so a joint's two line
// ends boil independently); bucket is floor(frame/2).
function hashUnit(n: number): number {
  const s = Math.sin(n) * 43758.5453123;
  return s - Math.floor(s);
}
function boilJitter(seed: number, bucket: number, amp = 1.6): { dx: number; dy: number } {
  const h1 = hashUnit(seed * 12.9898 + bucket * 78.233);
  const h2 = hashUnit(seed * 39.346 + bucket * 11.135 + 7.5);
  return { dx: (h1 - 0.5) * 2 * amp, dy: (h2 - 0.5) * 2 * amp };
}

const BoilLine: React.FC<{
  x1: number; y1: number; x2: number; y2: number; seed: number; frame: number;
  stroke: string; strokeWidth: number;
}> = ({ x1, y1, x2, y2, seed, frame, stroke, strokeWidth }) => {
  const bucket = Math.floor(frame / 2);
  const j1 = boilJitter(seed, bucket);
  const j2 = boilJitter(seed + 1000, bucket);
  return (
    <line x1={x1 + j1.dx} y1={y1 + j1.dy} x2={x2 + j2.dx} y2={y2 + j2.dy}
      stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" />
  );
};

const BoilCircle: React.FC<{
  cx: number; cy: number; r: number; seed: number; frame: number;
  stroke: string; strokeWidth: number; fill?: string;
}> = ({ cx, cy, r, seed, frame, stroke, strokeWidth, fill = "none" }) => {
  const bucket = Math.floor(frame / 2);
  const j = boilJitter(seed, bucket);
  const rj = boilJitter(seed + 2000, bucket, 1.0).dx;
  return (
    <circle cx={cx + j.dx} cy={cy + j.dy} r={Math.max(2, r + rj)}
      stroke={stroke} strokeWidth={strokeWidth} fill={fill} />
  );
};

const BoilPolyline: React.FC<{
  points: Array<[number, number]>; seedBase: number; frame: number;
  stroke: string; strokeWidth: number; closed?: boolean;
}> = ({ points, seedBase, frame, stroke, strokeWidth, closed }) => {
  const seq = closed ? [...points, points[0]] : points;
  return (
    <>
      {seq.slice(0, -1).map((p, i) => (
        <BoilLine key={i} x1={p[0]} y1={p[1]} x2={seq[i + 1][0]} y2={seq[i + 1][1]}
          seed={seedBase + i * 3} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />
      ))}
    </>
  );
};

// --- pose: joint angles per action, eased to a key pose by ~40% ------------
// Angle convention: 0deg points straight down (resting arm/leg at the side),
// positive rotates toward the figure's front/right (toXY below). Locomotion
// poses (walk/run/swim/climb) cycle continuously for the whole clip instead
// of landing once; the rest ease to their key pose by `land` and either hold
// (fall/sit) or keep a small live loop afterwards (point/shrug/think/cheer/
// idle/scared) -- "keeps moving after" (Architecture).
export type PoseAngles = {
  lean: number;
  armA: number; armAElbow: number;
  armB: number; armBElbow: number;
  legA: number; legAKnee: number;
  legB: number; legBKnee: number;
  bob: number;      // added to the hip's screen Y
  rotate: number;   // whole-figure rotation around the hip anchor (fall/swim)
  headTilt: number; // head-only rotation around the neck base (shrug's "head slightly tilted")
  shoulderLift: number; // raises the shoulder/arm/head assembly only, feet stay planted (shrug)
};

// armA/armB default to a visible rest angle (not 0 = straight down) so a
// resting or near-zero-swing arm never retraces the torso's own line --
// that was the "armless" bug (review finding 1). The shoulder-width offset
// in computeJoints already tilts the torso itself ~10deg off vertical
// (atan(shoulderHalf/torso)), so a rest angle too close to that same ~10deg
// reads as one unbroken line from hip through shoulder to hand -- a second,
// subtler "armless" look (M7b polish review). 24deg keeps a clear ~14deg
// kink at the shoulder in every pose that inherits this default (idle, the
// un-pointed/un-raised arm in point/think/sit), the torso visibly a
// separate segment from the arm in every one of them.
const BASE_POSE: PoseAngles = {
  lean: 0, armA: 24, armAElbow: 0, armB: -24, armBElbow: 0,
  legA: 3, legAKnee: 0, legB: -3, legBKnee: 0, bob: 0, rotate: 0, headTilt: 0, shoulderLift: 0,
};

function computePose(
  action: StickAction, frame: number, fps: number, durationInFrames: number, phaseOffsetFrames: number,
): PoseAngles {
  const land = Math.max(1, Math.min(durationInFrames - 1, Math.round(durationInFrames * 0.4)));
  const f = frame + phaseOffsetFrames;
  const t = frame / fps;
  const approach = (target: number) => interpolate(frame, [0, land], [0, target], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });

  switch (action) {
    case "walk": {
      const phase = (f / fps) * 1.45 * 2 * Math.PI;
      // armA/armB keep an outward base bigger than the swing amplitude, so
      // each arm's angle never crosses zero (and so its own shoulder-offset
      // origin) into the other arm's path -- the old symmetric +-24deg swing
      // crossed every half-cycle, drawing a big X across the chest instead
      // of two separate swinging arms (review finding 1). The floor (18deg)
      // also stays clear of the torso's own ~10deg shoulder-offset tilt, so
      // even the low point of the swing reads as a separate arm and not a
      // continuation of the torso's own line (M7b polish review).
      return {
        ...BASE_POSE,
        armA: 30 + 12 * Math.sin(phase), armB: -30 - 12 * Math.sin(phase),
        legA: -26 * Math.sin(phase), legB: 26 * Math.sin(phase),
        legAKnee: 22 * Math.max(0, Math.sin(phase + Math.PI / 2)),
        legBKnee: 22 * Math.max(0, Math.sin(phase + Math.PI / 2 + Math.PI)),
        bob: Math.abs(Math.sin(phase)) * 4,
      };
    }
    case "run": {
      const phase = (f / fps) * 2.3 * 2 * Math.PI;
      // Same no-crossing bias as walk, scaled up for run's bigger swing; the
      // floor (35deg) and the smaller elbow fold (-25 not -40) keep the
      // forearm itself from folding back past the upper arm's own angle at
      // the low point of the swing, which used to hook the hand back in
      // toward the hip instead of reading as a bent running arm (M7b
      // polish review).
      return {
        ...BASE_POSE, lean: 10,
        armA: 65 + 30 * Math.sin(phase), armB: -65 - 30 * Math.sin(phase), armAElbow: -25, armBElbow: -25,
        legA: -48 * Math.sin(phase), legB: 48 * Math.sin(phase),
        legAKnee: 55 * Math.max(0, Math.sin(phase + Math.PI / 2)),
        legBKnee: 55 * Math.max(0, Math.sin(phase + Math.PI / 2 + Math.PI)),
        bob: Math.abs(Math.sin(phase)) * 9,
      };
    }
    case "point": {
      const a = approach(1);
      return {
        ...BASE_POSE, lean: 6 * a,
        armA: 10 + 75 * a, armAElbow: -5 * a,
        bob: frame > land ? Math.sin(t * 2 * Math.PI * 0.6) * 1.5 : 0,
      };
    }
    case "shrug": {
      const a = approach(1);
      const settle = frame > land ? 0.8 + 0.1 * Math.sin(((frame - land) / fps) * 2 * Math.PI * 1.4) : a;
      // Shoulders raised (lifted via bob), upper arms down-and-OUT (small
      // positive/negative lean away from the body, never crossing centre),
      // forearms bent a further 75deg outward so each ends up roughly
      // horizontal pointing away from the torso -- two separate "wings",
      // not the inward-crossing diamond the old +-95deg elbow made.
      return {
        ...BASE_POSE,
        armA: 14 * settle, armAElbow: 76 * settle,
        armB: -14 * settle, armBElbow: -76 * settle,
        lean: 2 * settle,
        shoulderLift: 13 * settle,
        headTilt: 9 * settle,
      };
    }
    case "think": {
      const a = approach(1);
      const nod = frame > land ? Math.sin(((frame - land) / fps) * 2 * Math.PI * 0.4) * 4 : 0;
      return {
        ...BASE_POSE,
        armA: 20 * a, armAElbow: 110 * a,
        armB: -22,
        lean: 4 * a + nod,
      };
    }
    case "cheer": {
      const a = approach(1);
      const bounce = frame > land ? Math.abs(Math.sin(((frame - land) / fps) * 2 * Math.PI * 1.6)) : 0;
      // Target ~135deg from straight-down (45deg off vertical) with a
      // straight elbow: a wide V well clear of the head on both sides and
      // above it, instead of the old near-vertical ~170deg arms that ran up
      // the sides of the head circle like antennae.
      return {
        ...BASE_POSE,
        armA: 10 + 125 * a, armAElbow: 0,
        armB: -10 - 125 * a, armBElbow: 0,
        legAKnee: 18 * bounce, legBKnee: 18 * bounce,
        bob: -bounce * 10,
      };
    }
    case "fall": {
      const a = interpolate(frame, [0, land], [0, 1], {
        extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.in(Easing.cubic),
      });
      const twitch = frame > land ? Math.sin(((frame - land) / fps) * 2 * Math.PI * 2.5) * 3 : 0;
      // hipDrop used to approximate the sinking-to-the-ground motion by eye;
      // the renderer now grounds every rotated/bent pose exactly (its lowest
      // joint lands on the ground line), so rotate is all this needs.
      return {
        ...BASE_POSE,
        armA: 10 + 70 * a, armB: -10 - 40 * a,
        legAKnee: 20 * a, legBKnee: 10 * a,
        rotate: 86 * a + twitch,
      };
    }
    case "sit": {
      const a = approach(1);
      return {
        ...BASE_POSE,
        legA: 70 * a, legAKnee: -78 * a,
        legB: 74 * a, legBKnee: -82 * a,
        lean: 6 * a,
      };
    }
    case "scared": {
      const shiver = Math.sin(t * 2 * Math.PI * 7) * 3;
      return {
        ...BASE_POSE,
        armA: 55 + shiver, armAElbow: 90, armB: -55 - shiver, armBElbow: -90,
        legA: 8, legB: -8, lean: -4 + shiver * 0.5,
      };
    }
    case "swim": {
      // A gentle alternating freestyle stroke -- kept to a modest arc (not a
      // full swing past the torso) so the rotated, horizontal figure reads as
      // a swimmer rather than a tangle of overlapping limbs.
      const phase = (f / fps) * 1.5 * 2 * Math.PI;
      return {
        ...BASE_POSE, rotate: 84,
        armA: 30 * Math.sin(phase), armAElbow: -18 - 14 * Math.max(0, Math.sin(phase)),
        armB: -30 * Math.sin(phase + Math.PI), armBElbow: 18 + 14 * Math.max(0, Math.sin(phase + Math.PI)),
        legA: -20 * Math.sin(phase * 1.3), legB: 20 * Math.sin(phase * 1.3),
      };
    }
    case "climb": {
      const phase = (f / fps) * 1.0 * 2 * Math.PI;
      return {
        ...BASE_POSE, lean: 16,
        armA: -120 + 50 * Math.sin(phase), armB: -40 - 50 * Math.sin(phase),
        legA: 30 * Math.sin(phase + Math.PI), legAKnee: 40 * Math.max(0, Math.sin(phase + Math.PI)),
        legB: 30 * Math.sin(phase), legBKnee: 40 * Math.max(0, Math.sin(phase)),
      };
    }
    case "idle":
    default: {
      const breathe = Math.sin(t * 2 * Math.PI * 0.5);
      // armA/armB keep BASE_POSE's own rest angle (not a smaller ~10deg --
      // that landed almost exactly on the torso's own ~10deg shoulder-offset
      // tilt and read as one unbroken line from hip to hand, M7b polish
      // review); only the small breathing sway rides on top of it.
      return { ...BASE_POSE, armA: 24 + breathe, armB: -24 - breathe, bob: breathe * 2 };
    }
  }
}

// Dressed figures keep every gesture inside their own space (SW1b): a pointing
// or reaching arm is held to a relaxed angle, so it never stretches through
// the neighbouring figure; waving up, climbing, swimming and falling are left
// alone.
const FREE_ARM_ACTIONS = new Set<StickAction>(["cheer", "climb", "swim", "fall"]);
function calmPose(pose: PoseAngles, action: StickAction): PoseAngles {
  if (FREE_ARM_ACTIONS.has(action)) return pose;
  const capUp = (a: number) => Math.max(-40, Math.min(40, a));
  // the forearm may fold inward freely but never reach out past ~40deg
  const fa = Math.min(pose.armA + pose.armAElbow, 40);
  const fb = Math.max(pose.armB + pose.armBElbow, -40);
  const armA = capUp(pose.armA), armB = capUp(pose.armB);
  return { ...pose, armA, armB, armAElbow: fa - armA, armBElbow: fb - armB };
}

// Horizontal/vertical travel across the frame for the locomotion poses; every
// other action stays put (its own bob/hipDrop/rotate already carries the
// motion). Clamped by the caller so a figure is never pushed off-frame.
function travelFor(
  action: StickAction, frame: number, durationInFrames: number, width: number, height: number,
): { dx: number; dy: number } {
  const progress = durationInFrames > 1 ? frame / (durationInFrames - 1) : 0;
  switch (action) {
    case "walk":
      return { dx: interpolate(progress, [0, 1], [-width * 0.28, width * 0.28]), dy: 0 };
    case "run":
      return { dx: interpolate(progress, [0, 1], [-width * 0.35, width * 0.35]), dy: 0 };
    case "swim":
      return { dx: interpolate(progress, [0, 1], [-width * 0.28, width * 0.28]), dy: 0 };
    case "climb":
      return {
        dx: interpolate(progress, [0, 1], [-width * 0.06, width * 0.1]),
        dy: interpolate(progress, [0, 1], [height * 0.14, -height * 0.14]),
      };
    default:
      return { dx: 0, dy: 0 };
  }
}

// --- the rig: head circle, torso line, 2-segment arms and legs -------------
const toXY = (deg: number, len: number) => {
  const a = (deg * Math.PI) / 180;
  return { dx: len * Math.sin(a), dy: len * Math.cos(a) };
};

// Rotates a local point (hip-relative) by the whole-figure `rotate` degrees --
// the mirror, in plain JS, of the `<g rotate(...)>` the SVG itself applies --
// so the renderer can find a rotated pose's own lowest point (see
// `lowestLocalY` below) before it ever draws anything.
function rotateLocal(p: { x: number; y: number }, deg: number): { x: number; y: number } {
  const a = (deg * Math.PI) / 180;
  const cos = Math.cos(a), sin = Math.sin(a);
  return { x: p.x * cos - p.y * sin, y: p.x * sin + p.y * cos };
}

export type Joints = ReturnType<typeof computeJoints>;

// Every joint in hip-relative local space (hip at 0,0), plus the limb
// lengths the caller needs for grounding math. Shared by Figure (drawing)
// and lowestLocalY (grounding), so the two can never disagree about where a
// pose's hands/feet/head actually are.
function computeJoints(scale: number, pose: PoseAngles) {
  // Fractions of standing figure height `scale` (head+neck+torso+legs sums
  // to 1.0*scale when lean=0): headR 0.17 diameter (review finding 4).
  const headR = scale * 0.085;
  const neck = scale * 0.02;
  const torso = scale * 0.33;
  const upperArm = scale * 0.17;
  const foreArm = scale * 0.15;
  const upperLeg = scale * 0.24;
  const lowerLeg = scale * 0.24;
  // Shoulders (and hips) have real width, so a resting or mid-swing arm/leg
  // at angle 0 is a separate parallel line next to the torso, never the
  // torso's own line redrawn -- the root fix for "armless" poses (finding 1).
  const shoulderHalf = scale * 0.06;
  const hipHalf = scale * 0.045;

  const hip = { x: 0, y: 0 };
  const shVec = toXY(180 - pose.lean, torso);
  const shoulderC = { x: hip.x + shVec.dx, y: hip.y + shVec.dy - pose.shoulderLift };
  const hdVec = toXY(180 - pose.lean, neck + headR);
  const head = { x: shoulderC.x + hdVec.dx, y: shoulderC.y + hdVec.dy };
  // perpendicular to the torso, so the shoulder offset stays "sideways"
  // even when the torso itself leans (run/climb).
  const perp = { x: Math.cos((pose.lean * Math.PI) / 180), y: Math.sin((pose.lean * Math.PI) / 180) };
  const shoulderA = { x: shoulderC.x + perp.x * shoulderHalf, y: shoulderC.y + perp.y * shoulderHalf };
  const shoulderB = { x: shoulderC.x - perp.x * shoulderHalf, y: shoulderC.y - perp.y * shoulderHalf };
  const hipA = { x: hip.x + perp.x * hipHalf, y: hip.y + perp.y * hipHalf };
  const hipB = { x: hip.x - perp.x * hipHalf, y: hip.y - perp.y * hipHalf };

  const eA = toXY(pose.armA, upperArm);
  const elbowA = { x: shoulderA.x + eA.dx, y: shoulderA.y + eA.dy };
  const hA = toXY(pose.armA + pose.armAElbow, foreArm);
  const handA = { x: elbowA.x + hA.dx, y: elbowA.y + hA.dy };

  const eB = toXY(pose.armB, upperArm);
  const elbowB = { x: shoulderB.x + eB.dx, y: shoulderB.y + eB.dy };
  const hB = toXY(pose.armB + pose.armBElbow, foreArm);
  const handB = { x: elbowB.x + hB.dx, y: elbowB.y + hB.dy };

  const kA = toXY(pose.legA, upperLeg);
  const kneeA = { x: hipA.x + kA.dx, y: hipA.y + kA.dy };
  const fA = toXY(pose.legA + pose.legAKnee, lowerLeg);
  const footA = { x: kneeA.x + fA.dx, y: kneeA.y + fA.dy };

  const kB = toXY(pose.legB, upperLeg);
  const kneeB = { x: hipB.x + kB.dx, y: hipB.y + kB.dy };
  const fB = toXY(pose.legB + pose.legBKnee, lowerLeg);
  const footB = { x: kneeB.x + fB.dx, y: kneeB.y + fB.dy };

  return {
    headR, hip, shoulderA, shoulderB, head, elbowA, handA, elbowB, handB,
    hipA, hipB, kneeA, footA, kneeB, footB, legLen: upperLeg + lowerLeg,
  };
}

// The lowest (max-Y, after the figure's own `rotate`) point among the feet,
// hands and head -- used to plant every grounded pose exactly on the ground
// line regardless of how bent (sit) or rotated (fall) it is, instead of the
// old fixed-pixel `hipDrop` guess (review finding 8).
function lowestLocalY(joints: Joints, rotateDeg: number): number {
  const candidates = [joints.footA, joints.footB, joints.handA, joints.handB,
    { x: joints.head.x, y: joints.head.y + joints.headR }];
  return Math.max(...candidates.map((p) => rotateLocal(p, rotateDeg).y));
}

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// --- Stick World: dressing the rig (STICK_WORLD_PLAN.md SW1) ---------------
// `stick_world.json`'s typed shape, read once at module scope. Python
// (presentation/motion_stick.py) reads the same file for the same categories
// and `eras` table -- change the vocabulary only there, never here.
type StickWorldEra = { label: string; people: string[]; palette: string[]; bg: string };
type StickWorldVocab = { body: string[]; head: string[]; held: string[]; hair: string[]; eras: Record<string, StickWorldEra> };
const STICK_WORLD = stickWorldData as unknown as StickWorldVocab;
const ERAS = STICK_WORLD.eras;

type OutfitCategory = "body" | "head" | "held" | "hair";
type ResolvedOutfit = Partial<Record<OutfitCategory, string>>;

// Every piece name -> its one category, so an outfit spec's tokens (order-
// free, `+`-joined) can be sorted onto the rig regardless of the order they
// were written in.
const PIECE_CATEGORY: Record<string, OutfitCategory> = (() => {
  const map: Record<string, OutfitCategory> = {};
  (["body", "head", "held", "hair"] as const).forEach((cat) => {
    for (const name of STICK_WORLD[cat]) map[name] = cat;
  });
  return map;
})();

// `spec` (already normalised by presentation/motion_stick.py when it comes
// from a real beat) or, with none, `era`'s own default for this figure
// (contract: "era alone fills each figure from the era's people") -- this is
// a second, defensive pass at that same fallback so StickScene also dresses
// a figure correctly when driven directly (a sample job, a gallery props
// file) with just `era` and no `people` at all. An unknown token is dropped;
// a second token of a category already kept is dropped too, same rule as
// the Python side. null (not `{}`) only when there is no spec and no era --
// "a figure with no outfit and no era is the plain stick figure of today".
function resolveOutfit(spec: string | undefined, era: string | undefined, index: number): ResolvedOutfit | null {
  const eraRow = era ? ERAS[era] : undefined;
  const raw = spec || (eraRow && eraRow.people.length ? eraRow.people[index % eraRow.people.length] : undefined);
  if (!raw) return null;
  const out: ResolvedOutfit = {};
  for (const token of raw.split("+")) {
    const name = token.trim();
    if (!name) continue;
    const category = PIECE_CATEGORY[name];
    if (!category || out[category]) continue;
    out[category] = name;
  }
  return out;
}

// A skirt/robe/sleeve hangs a fixed length regardless of the current pose's
// own leg bend (sit/run fold the legs; the garment does not fold with them) --
// mirrors computeJoints' own torso(0.33) + upperLeg(0.24) + lowerLeg(0.24)
// fractions of `scale`; change one, check the other.
// (hip to foot = upperLeg 0.24 + lowerLeg 0.24: `hem` is a share of the LEG, so 0.9 stops just above the feet)
const STANDING_LEN_FRAC = 0.24 + 0.24;

type BodyStyle = {
  hem: number;        // share of the leg length (STANDING_LEN_FRAC) the garment hangs from the hip
  sleeve: number;      // 0 sleeveless, 0.42 short (to the elbow), 1 long (to the wrist)
  widthMul: number;    // hem half-width, as a multiple of the rig's own hip half-width
  wrap?: boolean;       // a diagonal sash/strap accent, shoulder to the opposite hip
  collarV?: boolean;    // a V notch at the neckline
  belt?: boolean;       // a horizontal accent partway down the hem
  buttons?: boolean;    // a few small dots down the centre front
  straps?: boolean;     // vertical accents over the shoulders (overalls, epaulettes)
  plates?: boolean;     // two horizontal accents suggesting segmented armour/plating
  hemZigzag?: boolean;  // a few short notches hanging off the hem (fur, fringe)
};

// One row per `body` vocabulary piece (31) -- a hem length/width, a sleeve
// length and a few accent flags stand in for real costume cut; "flat fill
// from the era palette" plus the hand-drawn boiling outline carries the rest.
const BODY_STYLES: Record<string, BodyStyle> = {
  fur_tunic: { hem: 0.45, sleeve: 0, widthMul: 1.7, hemZigzag: true },
  kilt: { hem: 0.25, sleeve: 0, widthMul: 1.6 },
  dress: { hem: 0.9, sleeve: 0.42, widthMul: 1.9 },
  chiton: { hem: 0.85, sleeve: 0, widthMul: 1.8, wrap: true },
  toga: { hem: 0.85, sleeve: 0, widthMul: 1.9, wrap: true },
  tunic: { hem: 0.45, sleeve: 0.42, widthMul: 1.7 },
  dhoti: { hem: 0.55, sleeve: 0, widthMul: 1.6 },
  saree: { hem: 0.95, sleeve: 0.42, widthMul: 1.9, wrap: true },
  lehenga: { hem: 0.95, sleeve: 0.42, widthMul: 2.2 },
  angrakha: { hem: 0.65, sleeve: 1, widthMul: 1.8, collarV: true },
  sherwani: { hem: 0.6, sleeve: 1, widthMul: 1.7, buttons: true, collarV: true },
  kurta: { hem: 0.55, sleeve: 1, widthMul: 1.7 },
  hanfu: { hem: 0.85, sleeve: 1, widthMul: 2.0, wrap: true },
  kimono: { hem: 0.85, sleeve: 1, widthMul: 2.0, wrap: true, belt: true },
  robe: { hem: 0.9, sleeve: 1, widthMul: 1.9, belt: true },
  armour: { hem: 0.35, sleeve: 0.42, widthMul: 1.7, plates: true },
  gown: { hem: 0.98, sleeve: 1, widthMul: 2.2 },
  doublet: { hem: 0.3, sleeve: 1, widthMul: 1.6, buttons: true },
  frock_coat: { hem: 0.65, sleeve: 1, widthMul: 1.7, buttons: true, collarV: true },
  vest: { hem: 0.3, sleeve: 0, widthMul: 1.5, buttons: true },
  suit: { hem: 0.35, sleeve: 1, widthMul: 1.6, buttons: true, collarV: true },
  trench_coat: { hem: 0.7, sleeve: 1, widthMul: 1.7, belt: true, collarV: true },
  uniform: { hem: 0.35, sleeve: 1, widthMul: 1.6, buttons: true, straps: true },
  lab_coat: { hem: 0.75, sleeve: 1, widthMul: 1.7, buttons: true },
  shirt: { hem: 0.3, sleeve: 1, widthMul: 1.5, collarV: true },
  tshirt: { hem: 0.28, sleeve: 0.42, widthMul: 1.5 },
  hoodie: { hem: 0.32, sleeve: 1, widthMul: 1.6 },
  overalls: { hem: 0.5, sleeve: 0, widthMul: 1.6, straps: true },
  flapper_dress: { hem: 0.42, sleeve: 0, widthMul: 1.7, hemZigzag: true },
  jumpsuit: { hem: 0.5, sleeve: 1, widthMul: 1.6, belt: true },
  spacesuit: { hem: 0.5, sleeve: 1, widthMul: 1.8, plates: true },
};

// --- drawing helpers for dressed figures (SW1b) -----------------------------
type Pt = { x: number; y: number };
type Seg = { p: Pt; c?: Pt; c2?: Pt };
type XY = [number, number];

// Straight / quadratic / cubic segments sampled into one outline, so a drape,
// a dome or a robe can be written as a few curves and still be boiled.
function sampleOutline(start: Pt, segs: Seg[], n = 7): XY[] {
  const out: XY[] = [[start.x, start.y]];
  let cur = start;
  for (const s of segs) {
    if (!s.c) {
      out.push([s.p.x, s.p.y]);
    } else {
      for (let i = 1; i <= n; i++) {
        const t = i / n, u = 1 - t;
        if (s.c2) {
          out.push([
            u * u * u * cur.x + 3 * u * u * t * s.c.x + 3 * u * t * t * s.c2.x + t * t * t * s.p.x,
            u * u * u * cur.y + 3 * u * u * t * s.c.y + 3 * u * t * t * s.c2.y + t * t * t * s.p.y,
          ]);
        } else {
          out.push([
            u * u * cur.x + 2 * u * t * s.c.x + t * t * s.p.x,
            u * u * cur.y + 2 * u * t * s.c.y + t * t * s.p.y,
          ]);
        }
      }
    }
    cur = s.p;
  }
  return out;
}

function ellipsePts(cx: number, cy: number, rx: number, ry: number, n = 20): XY[] {
  return Array.from({ length: n }, (_, i) => {
    const a = (i / n) * Math.PI * 2;
    return [cx + Math.cos(a) * rx, cy + Math.sin(a) * ry] as XY;
  });
}

function ringsToD(rings: XY[][]): string {
  return rings.map((r) => "M " + r.map((p) => p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" L ") + " Z").join(" ");
}

function shade(hex: string, amt: number): string {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  const mix = (c: number) => Math.round(amt < 0 ? c * (1 + amt) : c + (255 - c) * amt);
  const r = mix((n >> 16) & 255), g = mix((n >> 8) & 255), b = mix(n & 255);
  return "#" + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
}

// One closed shape (or rings with holes), hand-drawn: every point wobbles a
// little per 2-frame bucket like BoilLine's ends. `part` draws only the fill,
// only the outline, or both, so details can sit between the two.
const BoilShape: React.FC<{
  rings: XY[][]; seed: number; frame: number; fill?: string; stroke?: string; strokeWidth?: number;
  part?: "fill" | "line" | "both"; opacity?: number; amp?: number;
}> = ({ rings, seed, frame, fill = "none", stroke = "none", strokeWidth = 2, part = "both", opacity, amp = 0.9 }) => {
  const bucket = Math.floor(frame / 2);
  const wobbled = rings.map((r, ri) => r.map((p, i) => {
    const j = boilJitter(seed + ri * 500 + i * 3, bucket, amp);
    return [p[0] + j.dx, p[1] + j.dy] as XY;
  }));
  return (
    <path d={ringsToD(wobbled)} fillRule="evenodd" strokeLinejoin="round" strokeLinecap="round" opacity={opacity}
      fill={part === "line" ? "none" : fill} stroke={part === "fill" ? "none" : stroke} strokeWidth={strokeWidth} />
  );
};

// A thick coloured stroke with a thin outline behind it (sleeves, horns, the
// headphone band): outline first, colour on top.
const Tube: React.FC<{
  d: string; fill: string; stroke: string; width: number; outline: number; opacity?: number;
}> = ({ d, fill, stroke, width, outline, opacity }) => (
  <g opacity={opacity} fill="none" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} stroke={stroke} strokeWidth={width + outline * 2} />
    <path d={d} stroke={fill} strokeWidth={width} />
  </g>
);

// A drape: narrow at the shoulders, a slight waist, flaring to a curved hem.
// Built from the rig's own shoulders / hip, so it follows every pose; the hem
// continues the torso's lean and keeps a pose-independent length
// (STANDING_LEN_FRAC). Drawn AFTER the torso and leg lines so it covers them.
const Garment: React.FC<{
  joints: Joints; pose: PoseAngles; scale: number; style: BodyStyle;
  fill: string; stroke: string; strokeWidth: number; seedBase: number; frame: number;
}> = ({ joints, pose, scale, style, fill, stroke, strokeWidth, seedBase, frame }) => {
  const { hip, shoulderA, shoulderB, hipA } = joints;
  const hipHalf = dist(hip, hipA);
  const hemLen = scale * STANDING_LEN_FRAC * style.hem;
  const leanRad = (pose.lean * Math.PI) / 180;
  const perp = { x: Math.cos(leanRad), y: Math.sin(leanRad) };
  const dv = toXY(-pose.lean, 1);
  const hemHalf = hipHalf * style.widthMul * 1.5;
  const at = (side: number, across: number, down: number): Pt => ({
    x: hip.x + perp.x * side * across + dv.dx * down,
    y: hip.y + perp.y * side * across + dv.dy * down,
  });
  const shoulderMid = { x: (shoulderA.x + shoulderB.x) / 2, y: (shoulderA.y + shoulderB.y) / 2 };
  const torsoLen = dist(hip, shoulderMid);
  const sag = Math.min(hemLen * 0.12, scale * 0.025);
  const hemA = at(1, hemHalf, hemLen);
  const hemB = at(-1, hemHalf, hemLen);
  const outline = sampleOutline(shoulderA, [
    { c: at(1, hipHalf * 0.95, hemLen * 0.1), p: hemA },
    { c: at(0, 0, hemLen + sag * 2), p: hemB },
    { c: at(-1, hipHalf * 0.95, hemLen * 0.1), p: shoulderB },
    { p: shoulderA },
  ]);
  outline.pop();
  const dark = shade(fill, -0.28);
  const ink = shade(fill, -0.6);
  const clipId = "gc" + Math.round(seedBase);
  const band = (a: Pt, b: Pt, w: number, color = dark, key?: string) => (
    <line key={key} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={color} strokeWidth={w} strokeLinecap="round" />
  );
  const thin = Math.max(1.5, strokeWidth * 0.55);

  // fur / fringe hem: little teeth hanging under the hem, part of the shape
  let teeth: XY[] | null = null;
  if (style.hemZigzag) {
    const n = 5;
    const dip = Math.max(scale * 0.025, hemLen * 0.12);
    const pts: XY[] = [];
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      const base = at(1 - 2 * t, hemHalf, hemLen + sag * (1 - Math.abs(1 - 2 * t) ** 2) * 2);
      pts.push([base.x, base.y]);
      if (i < n) {
        const tm = (i + 0.5) / n;
        const tip = at(1 - 2 * tm, hemHalf, hemLen + dip + sag * 2);
        pts.push([tip.x, tip.y]);
      }
    }
    pts.push([hemB.x, hemB.y - 1], [hemA.x, hemA.y - 1]);
    teeth = pts;
  }

  return (
    <>
      {teeth && (
        <>
          <BoilShape rings={[teeth]} seed={seedBase + 70} frame={frame} fill={fill} part="fill" />
          <BoilShape rings={[teeth]} seed={seedBase + 70} frame={frame} stroke={stroke} strokeWidth={strokeWidth} part="line" />
        </>
      )}
      <clipPath id={clipId}><polygon points={outline.map((p) => p.join(",")).join(" ")} /></clipPath>
      <BoilShape rings={[outline]} seed={seedBase} frame={frame} fill={fill} part="fill" />
      <g clipPath={"url(#" + clipId + ")"}>
        {style.wrap && band(shoulderA, at(-1, hipHalf * 0.8, hemLen * 0.12), scale * 0.034)}
        {style.straps && (
          <>
            {band(shoulderA, at(1, hipHalf * 0.9, 0), scale * 0.02, dark, "s1")}
            {band(shoulderB, at(-1, hipHalf * 0.9, 0), scale * 0.02, dark, "s2")}
          </>
        )}
        {style.belt && band(at(1, hemHalf * 1.4, -torsoLen * 0.28), at(-1, hemHalf * 1.4, -torsoLen * 0.28), scale * 0.022)}
        {style.plates && [0.62, 0.3].map((f, i) => band(at(1, hemHalf * 1.4, -torsoLen * f), at(-1, hemHalf * 1.4, -torsoLen * f), scale * 0.014, dark, "p" + i))}
        {style.collarV && (() => {
          const tip = { x: shoulderMid.x + dv.dx * torsoLen * 0.2, y: shoulderMid.y + dv.dy * torsoLen * 0.2 };
          return (
            <>
              {band(shoulderA, tip, thin, ink, "c1")}
              {band(shoulderB, tip, thin, ink, "c2")}
            </>
          );
        })()}
        {style.buttons && [0.34, 0.55, 0.76].map((f, i) => {
          const p = { x: shoulderMid.x + dv.dx * torsoLen * f, y: shoulderMid.y + dv.dy * torsoLen * f };
          return <circle key={i} cx={p.x} cy={p.y} r={Math.max(1.6, strokeWidth * 0.4)} fill={ink} />;
        })}
      </g>
      <BoilShape rings={[outline]} seed={seedBase} frame={frame} stroke={stroke} strokeWidth={strokeWidth} part="line" />
    </>
  );
};

// A sleeve covers the arm from the shoulder to the elbow (short) or the wrist
// (long): one tube along the arm, drawn over the garment edge. The forearm
// stick is drawn by the caller wherever no sleeve covers it.
const SleeveArm: React.FC<{
  shoulder: Pt; elbow: Pt; hand: Pt; sleeve: number; fill: string; stroke: string; strokeWidth: number;
}> = ({ shoulder, elbow, hand, sleeve, fill, stroke, strokeWidth }) => {
  if (sleeve <= 0) return null;
  const long = sleeve >= 0.99;
  const end = long ? { x: elbow.x + (hand.x - elbow.x) * 0.86, y: elbow.y + (hand.y - elbow.y) * 0.86 } : elbow;
  const d = "M " + shoulder.x + " " + shoulder.y + " L " + elbow.x + " " + elbow.y + (long ? " L " + end.x + " " + end.y : "");
  return <Tube d={d} fill={fill} stroke={stroke} width={strokeWidth * 2.3} outline={strokeWidth * 0.5} />;
};


type HeadStyle = {
  shape: "band" | "dome" | "brim" | "peak" | "veil" | "tall" | "bubble" | "cups" | "cap";
  wide?: boolean; tall?: boolean; crest?: boolean; horns?: boolean; flap?: boolean;
  ribbon?: boolean; badge?: boolean; flat?: boolean; strap?: boolean; wrapLines?: boolean; knot?: boolean; accentDots?: boolean;
};

// One row per `head` vocabulary piece (32) -- a shape primitive plus a few
// accent flags; HEADWEAR shapes share code the way the body pieces share
// `Garment`, rather than 32 bespoke drawings.
const HEAD_STYLES: Record<string, HeadStyle> = {
  headband: { shape: "band" },
  nemes: { shape: "band", flap: true },
  laurel: { shape: "band", accentDots: true },
  greek_helmet: { shape: "dome", crest: true },
  roman_helmet: { shape: "dome", crest: true, flap: true },
  crown: { shape: "band", horns: true },
  turban: { shape: "dome", wide: true, wrapLines: true },
  mughal_turban: { shape: "dome", wide: true, wrapLines: true, accentDots: true },
  gandhi_cap: { shape: "cap" },
  pith_helmet: { shape: "dome", wide: true },
  hood: { shape: "veil" },
  veil: { shape: "veil" },
  dupatta: { shape: "veil", accentDots: true },
  knight_helmet: { shape: "dome", tall: true, strap: true },
  viking_helmet: { shape: "dome", horns: true },
  kabuto: { shape: "dome", crest: true, wide: true },
  beret: { shape: "cap", flat: true },
  tricorn: { shape: "brim", wide: true },
  bandana: { shape: "band", knot: true },
  top_hat: { shape: "tall" },
  bonnet: { shape: "dome", ribbon: true },
  deerstalker: { shape: "cap", flap: true },
  cowboy_hat: { shape: "brim", wide: true, tall: true },
  fedora: { shape: "brim" },
  flat_cap: { shape: "peak" },
  cloche: { shape: "dome" },
  military_helmet: { shape: "dome", strap: true },
  police_cap: { shape: "peak", badge: true },
  cap: { shape: "peak" },
  hard_hat: { shape: "dome", wide: true },
  headphones: { shape: "cups" },
  astronaut_helmet: { shape: "bubble" },
};

// Headwear sits ON the head: every shape's lower edge stays above the eye line
// (about -0.4 head radii from the centre); only cloth (hood, veil, nemes) and
// cheek flaps hang beside the face, never across it. All coordinates are in
// head radii from the head centre, y down.
const Headwear: React.FC<{
  head: { x: number; y: number }; headR: number; style: HeadStyle;
  fill: string; stroke: string; accent: string; strokeWidth: number; seedBase: number; frame: number;
}> = ({ head, headR, style, fill, stroke, accent, strokeWidth, seedBase, frame }) => {
  const R = headR;
  const sw = strokeWidth;
  const P = (x: number, y: number): Pt => ({ x: head.x + x * R, y: head.y + y * R });
  const dark = shade(fill, -0.3);
  const ink = shade(fill, -0.6);
  const nodes: React.ReactNode[] = [];
  let count = 0;
  const idBase = "hw" + Math.round(seedBase);

  const solid = (pts: XY[], color: string, details?: React.ReactNode, opacity?: number) => {
    const k = count++;
    const id = idBase + "_" + k;
    const s = seedBase + 7 + k * 41;
    nodes.push(
      <React.Fragment key={id}>
        <clipPath id={id}><polygon points={pts.map((p) => p.join(",")).join(" ")} /></clipPath>
        <BoilShape rings={[pts]} seed={s} frame={frame} fill={color} part="fill" opacity={opacity} />
        {details && <g clipPath={"url(#" + id + ")"}>{details}</g>}
        <BoilShape rings={[pts]} seed={s} frame={frame} stroke={stroke} strokeWidth={sw} part="line" />
      </React.Fragment>,
    );
  };
  const tube = (d: string, color: string, width: number, outline = sw * 0.5) => {
    nodes.push(<Tube key={idBase + "_t" + count++} d={d} fill={color} stroke={stroke} width={width} outline={outline} />);
  };
  const line = (a: Pt, c: Pt | null, b: Pt, color: string, width: number) => {
    const d = c ? "M " + a.x + " " + a.y + " Q " + c.x + " " + c.y + " " + b.x + " " + b.y : "M " + a.x + " " + a.y + " L " + b.x + " " + b.y;
    return <path key={idBase + "_l" + count++} d={d} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" />;
  };
  // a rounded dome: base chord at baseY, top at peakY, optional upward bow of the lower edge
  const dome = (w: number, peakY: number, baseY: number, bowY?: number): XY[] => {
    const cy = (8 * peakY - 2 * baseY) / 6;
    const pts = sampleOutline(P(-w, baseY), [
      { c: P(-w * 1.12, cy), c2: P(w * 1.12, cy), p: P(w, baseY) },
      { c: P(0, bowY === undefined ? baseY : bowY), p: P(-w, baseY) },
    ]);
    pts.pop();
    return pts;
  };
  const cheek = (side: number) => sampleOutline(P(side * 0.98, -0.45), [
    { p: P(side * 1.1, 0.5) }, { p: P(side * 0.8, 0.5) }, { p: P(side * 0.86, -0.35) },
  ]);

  switch (style.shape) {
    case "band": {
      if (style.horns) {
        // crown: a band with points
        const pts = sampleOutline(P(-0.88, -0.42), [
          { p: P(-0.9, -1.2) }, { p: P(-0.45, -0.88) }, { p: P(0, -1.38) },
          { p: P(0.45, -0.88) }, { p: P(0.9, -1.2) }, { p: P(0.88, -0.42) },
        ]);
        solid(pts, fill, (
          <>
            {line(P(-1, -0.58), null, P(1, -0.58), dark, R * 0.16)}
            {[-0.5, 0, 0.5].map((k, i) => <circle key={i} cx={P(k, -0.6).x} cy={P(k, -0.6).y} r={Math.max(2, R * 0.1)} fill={accent} />)}
          </>
        ));
      } else if (style.flap) {
        // nemes: a head-cloth with a face opening and two tails
        const pts = sampleOutline(P(-1.0, -0.35), [
          { c: P(-1.3, -1.7), c2: P(1.3, -1.7), p: P(1.0, -0.35) },
          { p: P(1.3, 1.25) }, { p: P(0.9, 1.25) }, { p: P(0.9, -0.05) },
          { c: P(0, -0.7), p: P(-0.9, -0.05) },
          { p: P(-0.9, 1.25) }, { p: P(-1.3, 1.25) },
        ]);
        solid(pts, fill, (
          <>
            {[-0.62, -0.95, -1.28].map((y, i) => line(P(-1.5, y), P(0, y - 0.16), P(1.5, y), accent, R * 0.1))}
            {[0.1, 0.5, 0.9].map((y, i) => line(P(-1.4, y), null, P(-0.8, y), accent, R * 0.08))}
            {[0.1, 0.5, 0.9].map((y, i) => line(P(0.8, y), null, P(1.4, y), accent, R * 0.08))}
          </>
        ));
      } else if (style.accentDots) {
        // laurel: leaves along the top of the head
        const leaf = "#5E8C3A";
        for (let i = 0; i < 9; i++) {
          const a = ((-152 + i * (124 / 8)) * Math.PI) / 180;
          const c = P(Math.cos(a) * 0.98, Math.sin(a) * 0.98);
          const deg = (a * 180) / Math.PI + 90;
          nodes.push(
            <ellipse key={idBase + "_lf" + i} cx={c.x} cy={c.y} rx={R * 0.24} ry={R * 0.11} fill={leaf} stroke={stroke}
              strokeWidth={Math.max(1.2, sw * 0.5)} transform={"rotate(" + deg + " " + c.x + " " + c.y + ")"} />,
          );
        }
      } else {
        // headband / bandana: a thin band round the top of the head
        const pts = sampleOutline(P(-0.82, -0.64), [
          { c: P(0, -0.7), p: P(0.82, -0.64) },
          { p: P(0.93, -0.42) },
          { c: P(0, -0.34), p: P(-0.93, -0.42) },
        ]);
        solid(pts, fill);
        if (style.knot) {
          nodes.push(
            <path key={idBase + "_kn"} d={"M " + P(-0.93, -0.52).x + " " + P(-0.93, -0.52).y + " l " + (-R * 0.3) + " " + (-R * 0.18) + " m " + (R * 0.3) + " " + (R * 0.18) + " l " + (-R * 0.32) + " " + (R * 0.2)}
              fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round" />,
          );
        }
      }
      break;
    }
    case "cap": {
      if (style.flat) {
        // beret
        solid(ellipsePts(head.x + R * 0.12, head.y - R * 0.98, R * 1.05, R * 0.4, 26), fill);
        nodes.push(line(P(0.12, -1.38), null, P(0.12, -1.55), stroke, sw * 1.1));
      } else {
        if (style.flap) { solid(cheek(-1), fill); solid(cheek(1), fill); }
        solid(dome(0.92, -1.14, -0.45), fill, line(P(-1.2, -0.56), null, P(1.2, -0.56), dark, R * 0.18));
      }
      break;
    }
    case "peak": {
      solid(dome(1.0, -1.1, -0.42), fill, line(P(-1.2, -0.54), null, P(1.2, -0.54), dark, R * 0.16));
      solid(sampleOutline(P(0.05, -0.52), [{ c: P(0.8, -0.66), p: P(1.5, -0.46) }, { c: P(0.8, -0.32), p: P(0.05, -0.4) }]), dark);
      if (style.badge) nodes.push(<circle key={idBase + "_bd"} cx={P(0, -0.8).x} cy={P(0, -0.8).y} r={Math.max(2.5, R * 0.12)} fill={accent} stroke={stroke} strokeWidth={sw * 0.5} />);
      break;
    }
    case "brim": {
      if (style.wide && !style.tall) {
        // tricorn
        const pts = sampleOutline(P(-1.45, -0.42), [
          { c: P(-0.9, -1.05), p: P(0, -1.38) },
          { c: P(0.9, -1.05), p: P(1.45, -0.42) },
          { c: P(0, -0.28), p: P(-1.45, -0.42) },
        ]);
        pts.pop();
        solid(pts, fill, (
          <>
            {line(P(-1.5, -0.5), P(-0.9, -1.1), P(0, -1.4), accent, R * 0.12)}
            {line(P(0, -1.4), P(0.9, -1.1), P(1.5, -0.5), accent, R * 0.12)}
          </>
        ));
      } else {
        solid(ellipsePts(head.x, head.y - R * 0.46, R * (style.wide ? 1.55 : 1.3), R * 0.17, 26), fill);
        solid(dome(0.72, style.tall ? -1.6 : -1.2, -0.5), fill, line(P(-1, -0.66), null, P(1, -0.66), dark, R * 0.16));
      }
      break;
    }
    case "tall": {
      solid(ellipsePts(head.x, head.y - R * 0.46, R * 1.25, R * 0.16, 26), fill);
      const pts = sampleOutline(P(-0.7, -0.5), [
        { p: P(-0.7, -1.95) }, { c: P(0, -2.15), p: P(0.7, -1.95) }, { p: P(0.7, -0.5) }, { c: P(0, -0.4), p: P(-0.7, -0.5) },
      ]);
      pts.pop();
      solid(pts, fill, line(P(-1, -0.76), null, P(1, -0.76), ink, R * 0.22));
      break;
    }
    case "veil": {
      const outer = sampleOutline(P(-1.2, 0.3), [
        { c: P(-1.45, -1.7), c2: P(1.45, -1.7), p: P(1.2, 0.3) },
        { c: P(1.3, 0.9), p: P(1.5, 1.7) },
        { c: P(0, 1.95), p: P(-1.5, 1.7) },
        { c: P(-1.3, 0.9), p: P(-1.2, 0.3) },
      ]);
      outer.pop();
      const inner = ellipsePts(head.x, head.y + R * 0.02, R * 0.88, R * 0.98, 24);
      const k = count++;
      nodes.push(
        <React.Fragment key={idBase + "_v" + k}>
          <BoilShape rings={[outer, inner]} seed={seedBase + 9} frame={frame} fill={fill} part="fill" opacity={0.95} />
          <BoilShape rings={[outer]} seed={seedBase + 9} frame={frame} stroke={stroke} strokeWidth={sw} part="line" />
          <BoilShape rings={[inner]} seed={seedBase + 19} frame={frame} stroke={stroke} strokeWidth={sw * 0.8} part="line" />
          {style.accentDots && [-0.6, -0.2, 0.2, 0.6].map((x, i) => (
            <circle key={i} cx={P(x * 1.5, 1.45).x} cy={P(x * 1.5, 1.45).y} r={Math.max(2, R * 0.09)} fill={accent} />
          ))}
        </React.Fragment>,
      );
      break;
    }
    case "bubble": {
      const ring = ellipsePts(head.x, head.y, R * 1.42, R * 1.42, 30);
      nodes.push(
        <React.Fragment key={idBase + "_bb"}>
          <BoilShape rings={[ring]} seed={seedBase + 5} frame={frame} fill="#FFFFFF" part="fill" opacity={0.22} />
          <BoilShape rings={[ring]} seed={seedBase + 5} frame={frame} stroke={stroke} strokeWidth={sw} part="line" />
        </React.Fragment>,
      );
      nodes.push(line(P(-0.95, -0.85), P(-0.7, -1.2), P(-0.25, -1.3), "#FFFFFF", sw * 0.9));
      break;
    }
    case "cups": {
      tube("M " + P(-1.02, -0.25).x + " " + P(-1.02, -0.25).y + " C " + P(-1.12, -1.55).x + " " + P(-1.12, -1.55).y + " " + P(1.12, -1.55).x + " " + P(1.12, -1.55).y + " " + P(1.02, -0.25).x + " " + P(1.02, -0.25).y, fill, sw * 1.5);
      solid(ellipsePts(head.x - R * 1.03, head.y, R * 0.24, R * 0.4, 16), fill);
      solid(ellipsePts(head.x + R * 1.03, head.y, R * 0.24, R * 0.4, 16), fill);
      break;
    }
    case "dome":
    default: {
      const turban = !!style.wrapLines;
      const w = style.wide ? 1.12 : 0.99;
      const peak = style.tall ? -1.78 : turban ? -1.5 : -1.26;
      const brimmed = !!style.wide && !turban && !style.crest;
      if (style.horns) {
        const horn = "#EDE6D2";
        tube("M " + P(-0.95, -0.75).x + " " + P(-0.95, -0.75).y + " Q " + P(-1.75, -0.85).x + " " + P(-1.75, -0.85).y + " " + P(-1.55, -1.85).x + " " + P(-1.55, -1.85).y, horn, R * 0.26);
        tube("M " + P(0.95, -0.75).x + " " + P(0.95, -0.75).y + " Q " + P(1.75, -0.85).x + " " + P(1.75, -0.85).y + " " + P(1.55, -1.85).x + " " + P(1.55, -1.85).y, horn, R * 0.26);
      }
      if (style.crest && !style.wide) {
        solid(sampleOutline(P(-0.5, -1.15), [{ c: P(-0.55, -2.15), c2: P(0.5, -2.35), p: P(0.8, -1.55) }, { c: P(0.35, -1.3), p: P(0.4, -1.15) }]), accent);
      }
      if (style.crest && style.wide) {
        tube("M " + P(-0.15, -1.1).x + " " + P(-0.15, -1.1).y + " L " + P(-0.8, -2.0).x + " " + P(-0.8, -2.0).y, accent, R * 0.16);
        tube("M " + P(0.15, -1.1).x + " " + P(0.15, -1.1).y + " L " + P(0.8, -2.0).x + " " + P(0.8, -2.0).y, accent, R * 0.16);
      }
      if (brimmed) solid(ellipsePts(head.x, head.y - R * 0.42, R * 1.5, R * 0.13, 24), fill);
      if (turban && style.accentDots) {
        // plume behind the jewel
        tube("M " + P(0.1, -1.0).x + " " + P(0.1, -1.0).y + " Q " + P(0.45, -1.9).x + " " + P(0.45, -1.9).y + " " + P(0.1, -2.25).x + " " + P(0.1, -2.25).y, accent, R * 0.12);
      }
      const body = dome(w, peak, -0.4, turban ? -0.58 : undefined);
      const details = (
        <>
          {turban && (
            <>
              {line(P(-1.3, -0.62), P(0, -1.12), P(1.3, -0.62), stroke, sw * 0.6)}
              {line(P(-1.0, -0.85), P(-0.1, -1.55), P(0.8, -1.55), stroke, sw * 0.6)}
              {line(P(-0.2, -0.55), P(0.7, -1.05), P(1.3, -0.95), stroke, sw * 0.6)}
            </>
          )}
          {brimmed && line(P(0, peak + 0.08), null, P(0, -0.5), dark, sw * 0.9)}
          {style.tall && line(P(0, peak + 0.08), null, P(0, -0.5), dark, sw * 0.9)}
        </>
      );
      solid(body, fill, details);
      if (turban && style.accentDots) {
        nodes.push(<circle key={idBase + "_jw"} cx={P(0.05, -0.82).x} cy={P(0.05, -0.82).y} r={Math.max(2.5, R * 0.14)} fill={accent} stroke={stroke} strokeWidth={sw * 0.6} />);
      }
      if (style.flap || (style.crest && !style.wide)) { solid(cheek(-1), fill); solid(cheek(1), fill); }
      if (style.strap) {
        nodes.push(line(P(-0.93, -0.3), P(-1.0, 0.7), P(-0.4, 0.97), ink, sw * 0.8));
        nodes.push(line(P(0.93, -0.3), P(1.0, 0.7), P(0.4, 0.97), ink, sw * 0.8));
      }
      break;
    }
  }

  return <>{nodes}</>;
};


type HairStyle = { kind: "hair" | "bun" | "facial" | "none"; length: number; down?: boolean; flare?: boolean };

// One row per `hair` vocabulary piece (7, including `bald`).
const HAIR_STYLES: Record<string, HairStyle> = {
  short: { kind: "hair", length: 0.6 },
  long_hair: { kind: "hair", length: 1.6, down: true },
  bun: { kind: "bun", length: 0 },
  bob: { kind: "hair", length: 0.9, down: true, flare: true },
  bald: { kind: "none", length: 0 },
  beard: { kind: "facial", length: 0.55 },
  moustache: { kind: "facial", length: 0.22 },
};

// Hair is a filled shape, never loose lines: a cap over the top of the head
// (front layer, above the eye line), long/bob hair as panels behind the head
// (back layer), a bun on top, a beard hanging from the chin, a moustache above
// the mouth line.
const Hair: React.FC<{
  head: { x: number; y: number }; headR: number; style: HairStyle; color: string;
  stroke: string; strokeWidth: number; seedBase: number; frame: number; layer: "back" | "front";
}> = ({ head, headR, style, color, stroke, strokeWidth, seedBase, frame, layer }) => {
  if (style.kind === "none") return null;
  const R = headR;
  const P = (x: number, y: number): Pt => ({ x: head.x + x * R, y: head.y + y * R });
  const outline = Math.max(1.6, strokeWidth * 0.8);
  const shape = (pts: XY[], s: number) => (
    <>
      <BoilShape rings={[pts]} seed={seedBase + s} frame={frame} fill={color} part="fill" />
      <BoilShape rings={[pts]} seed={seedBase + s} frame={frame} stroke={stroke} strokeWidth={outline} part="line" />
    </>
  );
  const capPts = () => {
    const pts = sampleOutline(P(-0.98, -0.12), [
      { c: P(-1.2, -1.45), c2: P(1.2, -1.45), p: P(0.98, -0.12) },
      { c: P(0, -0.66), p: P(-0.98, -0.12) },
    ]);
    pts.pop();
    return pts;
  };

  if (style.kind === "bun") {
    if (layer === "back") return null;
    return (
      <>
        <BoilShape rings={[ellipsePts(head.x, head.y - R * 1.28, R * 0.4, R * 0.36, 16)]} seed={seedBase + 3} frame={frame} fill={color} part="fill" />
        <BoilShape rings={[ellipsePts(head.x, head.y - R * 1.28, R * 0.4, R * 0.36, 16)]} seed={seedBase + 3} frame={frame} stroke={stroke} strokeWidth={outline} part="line" />
        {shape(capPts(), 10)}
      </>
    );
  }
  if (style.kind === "facial") {
    if (layer === "back") return null;
    if (style.length > 0.3) {
      // beard: hangs from the cheeks round the chin, clear of the mouth
      const depth = 1.0 + style.length * 1.0;
      const pts = sampleOutline(P(-0.9, 0.34), [
        { c: P(-1.0, 1.15), c2: P(-0.4, depth), p: P(0, depth) },
        { c: P(0.4, depth), c2: P(1.0, 1.15), p: P(0.9, 0.34) },
        { c: P(0.55, 0.62), p: P(0, 0.7) },
        { c: P(-0.55, 0.62), p: P(-0.9, 0.34) },
      ]);
      pts.pop();
      return shape(pts, 4);
    }
    // moustache: a short thick curve between nose and mouth
    const d = "M " + P(-0.46, 0.27).x + " " + P(-0.46, 0.27).y + " Q " + P(-0.22, 0.1).x + " " + P(-0.22, 0.1).y + " " + P(0, 0.22).x + " " + P(0, 0.22).y
      + " Q " + P(0.22, 0.1).x + " " + P(0.22, 0.1).y + " " + P(0.46, 0.27).x + " " + P(0.46, 0.27).y;
    return <Tube d={d} fill={color} stroke={stroke} width={Math.max(3, R * 0.16)} outline={Math.max(0.8, strokeWidth * 0.3)} />;
  }
  // "hair": a cap on top; long / bob also fall behind the head
  if (layer === "back") {
    if (!style.down) return null;
    const depth = style.length > 1 ? 2.3 : 1.0;
    const pts = sampleOutline(P(-0.97, -0.3), [
      { c: P(-1.5, 0.4), c2: P(-1.45, depth * 0.8), p: P(-1.05, depth) },
      { c: P(0, depth + 0.15), p: P(1.05, depth) },
      { c: P(1.45, depth * 0.8), c2: P(1.5, 0.4), p: P(0.97, -0.3) },
    ]);
    return shape(pts, 5);
  }
  return shape(capPts(), 6);
};

// Held items (STICK_WORLD_PLAN.md SW1): one in the leading hand, anchored at
// `handA` and rotated to the forearm's own angle (the same toXY convention
// pose angles use -- 0deg points straight down), so it swings with the arm
// through every pose instead of sitting fixed on the frame. A "rod" extends
// past the hand along that angle; an "object" sits just past the fist.
// A rod keeps its own hand-anchored origin (y=0 at the hand, extending out);
// every other (compact) item is drawn with its own small internal offset
// from the hand for clearance and gets pulled back in by HELD_OBJECT_PULL
// below, or it reads as dropped near the hand rather than held in it (a real
// render + frame inspection, STICK_WORLD_PLAN.md SW1).
const ROD_NAMES = new Set([
  "club", "spear", "staff", "sceptre", "sword", "axe", "torch", "cane", "umbrella", "wrench", "bow", "lasso",
]);

// Rods that stand upright in the hand (the rest hang along the forearm).
const UPRIGHT_RODS = new Set(["club", "spear", "staff", "sceptre", "sword", "axe", "torch", "bow"]);

const HeldArt: React.FC<{
  hand: { x: number; y: number }; angle: number; s: number;
  fill: string; stroke: string; accent: string; frame: number; name: string;
}> = ({ hand, angle, s, fill, stroke, accent, frame, name }) => {
  const sw = Math.max(2, s * 0.09);
  const isRod = ROD_NAMES.has(name);
  // a rod held upright (angle ~180) stands beside the figure, taller than a hanging one
  const len = s * (isRod && angle > 150 ? 2.7 : 1.7);
  const HELD_OBJECT_PULL = s * 0.24;
  const rod = (tip?: React.ReactNode) => (
    <>
      <line x1={0} y1={0} x2={0} y2={len} stroke={fill} strokeWidth={sw * 1.3} strokeLinecap="round" />
      <BoilLine x1={0} y1={0} x2={0} y2={len} seed={1} frame={frame} stroke={stroke} strokeWidth={sw * 0.5} />
      {tip}
    </>
  );
  const node = (() => {
    switch (name) {
      case "club":
        return rod(<BoilCircle cx={0} cy={len} r={s * 0.3} seed={2} frame={frame} stroke={stroke} strokeWidth={sw * 0.5} fill={fill} />);
      case "spear":
        return rod(<>
          <polygon points={`${-s * 0.13},${len * 0.9} 0,${len * 1.16} ${s * 0.13},${len * 0.9}`} fill="#D5D9DE" />
          <BoilPolyline closed points={[[-s * 0.13, len * 0.9], [0, len * 1.16], [s * 0.13, len * 0.9]]} seedBase={2} frame={frame} stroke={stroke} strokeWidth={sw * 0.5} />
        </>);
      case "staff":
        return rod();
      case "sceptre":
        return rod(<BoilCircle cx={0} cy={len * 0.82} r={s * 0.22} seed={2} frame={frame} stroke={stroke} strokeWidth={sw * 0.5} fill={accent} />);
      case "sword":
        return rod(<>
          <line x1={-s * 0.22} y1={len * 0.78} x2={s * 0.22} y2={len * 0.78} stroke={stroke} strokeWidth={sw * 0.5} />
          <BoilPolyline points={[[-s * 0.1, len], [0, len * 1.12], [s * 0.1, len]]} seedBase={2} frame={frame} stroke={stroke} strokeWidth={sw * 0.5} />
        </>);
      case "axe":
        return rod(<BoilPolyline closed points={[[0, len * 0.92], [s * 0.4, len * 0.82], [s * 0.3, len * 1.05]]} seedBase={2} frame={frame} stroke={stroke} strokeWidth={sw * 0.5} />);
      case "torch":
        return rod(<BoilPolyline closed points={[[-s * 0.14, len * 0.95], [0, len * 1.2], [s * 0.14, len * 0.95]]} seedBase={2} frame={frame} stroke={stroke} strokeWidth={sw * 0.5} />);
      case "cane":
        return rod(<path d={`M 0 ${len * 0.05} Q ${s * 0.3} ${len * 0.02} ${s * 0.3} ${len * 0.22}`} fill="none" stroke={stroke} strokeWidth={sw * 0.5} />);
      case "umbrella":
        return rod(<path d={`M ${-s * 0.4} ${len * 0.1} Q 0 ${-s * 0.05} ${s * 0.4} ${len * 0.1}`} fill="none" stroke={accent} strokeWidth={sw * 0.5} />);
      case "wrench":
        return rod(<>
          <line x1={-s * 0.16} y1={len * 0.95} x2={0} y2={len * 1.1} stroke={stroke} strokeWidth={sw * 0.5} />
          <line x1={s * 0.16} y1={len * 0.95} x2={0} y2={len * 1.1} stroke={stroke} strokeWidth={sw * 0.5} />
        </>);
      case "bow":
        return (
          <>
            <path d={`M ${-s * 0.3} ${len * 0.1} Q ${s * 0.32} ${len * 0.55} ${-s * 0.3} ${len}`} fill="none" stroke={stroke} strokeWidth={sw * 0.5} />
            <line x1={-s * 0.3} y1={len * 0.1} x2={-s * 0.3} y2={len} stroke={fill} strokeWidth={sw * 0.4} />
          </>
        );
      case "lasso":
        return rod(<BoilCircle cx={0} cy={len * 0.75} r={s * 0.32} seed={2} frame={frame} stroke={stroke} strokeWidth={sw * 0.5} fill="none" />);
      // -- compact objects, held just past the fist ---------------------
      case "lantern":
        return <><rect x={-s * 0.16} y={s * 0.32} width={s * 0.32} height={s * 0.3} fill={fill} /><path d={`M ${-s * 0.08} ${s * 0.32} Q 0 ${s * 0.18} ${s * 0.08} ${s * 0.32}`} fill="none" stroke={stroke} strokeWidth={sw * 0.4} /><BoilCircle cx={0} cy={s * 0.47} r={s * 0.07} seed={3} frame={frame} stroke={stroke} strokeWidth={sw * 0.3} fill={accent} /></>;
      case "scroll":
        return <><rect x={-s * 0.14} y={s * 0.25} width={s * 0.28} height={s * 0.5} fill={fill} /><line x1={0} y1={s * 0.25} x2={0} y2={s * 0.75} stroke={stroke} strokeWidth={sw * 0.4} /></>;
      case "shield":
        return <><ellipse cx={0} cy={s * 0.5} rx={s * 0.26} ry={s * 0.34} fill={fill} /><line x1={0} y1={s * 0.2} x2={0} y2={s * 0.8} stroke={accent} strokeWidth={sw * 0.5} /></>;
      case "quill":
        return <><line x1={-s * 0.1} y1={s * 0.75} x2={s * 0.22} y2={s * 0.2} stroke={fill} strokeWidth={sw * 0.5} /><BoilPolyline points={[[s * 0.22, s * 0.2], [s * 0.34, s * 0.22], [s * 0.26, s * 0.32]]} seedBase={3} frame={frame} stroke={stroke} strokeWidth={sw * 0.4} /></>;
      case "book":
        return <><rect x={-s * 0.22} y={s * 0.35} width={s * 0.44} height={s * 0.34} fill={fill} /><line x1={0} y1={s * 0.35} x2={0} y2={s * 0.69} stroke={accent} strokeWidth={sw * 0.4} /></>;
      case "fan":
        return <>{[-0.3, 0, 0.3].map((k, i) => <line key={i} x1={0} y1={s * 0.75} x2={k * s * 0.4} y2={s * 0.25} stroke={fill} strokeWidth={sw * 0.5} />)}</>;
      case "magnifier":
        return <><BoilCircle cx={0} cy={s * 0.35} r={s * 0.18} seed={3} frame={frame} stroke={stroke} strokeWidth={sw * 0.4} fill="none" /><line x1={s * 0.12} y1={s * 0.48} x2={s * 0.3} y2={s * 0.68} stroke={stroke} strokeWidth={sw * 0.4} /></>;
      case "pipe":
        return <><line x1={-s * 0.1} y1={s * 0.6} x2={s * 0.15} y2={s * 0.6} stroke={fill} strokeWidth={sw * 0.5} /><line x1={s * 0.15} y1={s * 0.6} x2={s * 0.15} y2={s * 0.4} stroke={fill} strokeWidth={sw * 0.5} /></>;
      case "briefcase":
        return <><rect x={-s * 0.26} y={s * 0.4} width={s * 0.52} height={s * 0.32} fill={fill} /><path d={`M ${-s * 0.1} ${s * 0.4} Q 0 ${s * 0.28} ${s * 0.1} ${s * 0.4}`} fill="none" stroke={stroke} strokeWidth={sw * 0.4} /></>;
      case "newspaper":
        return <><rect x={-s * 0.2} y={s * 0.3} width={s * 0.4} height={s * 0.42} fill={fill} />{[0.42, 0.52, 0.62].map((y, i) => <line key={i} x1={-s * 0.14} y1={s * y} x2={s * 0.14} y2={s * y} stroke={accent} strokeWidth={sw * 0.3} />)}</>;
      case "notepad":
        return <><rect x={-s * 0.16} y={s * 0.35} width={s * 0.32} height={s * 0.36} fill={fill} />{[0.45, 0.55].map((y, i) => <line key={i} x1={-s * 0.1} y1={s * y} x2={s * 0.1} y2={s * y} stroke={accent} strokeWidth={sw * 0.3} />)}</>;
      case "phone":
        return <rect x={-s * 0.1} y={s * 0.35} width={s * 0.2} height={s * 0.34} fill={fill} rx={s * 0.03} />;
      case "laptop":
        return <><rect x={-s * 0.22} y={s * 0.55} width={s * 0.44} height={s * 0.08} fill={fill} /><rect x={-s * 0.2} y={s * 0.28} width={s * 0.4} height={s * 0.27} fill={fill} /></>;
      case "tablet":
        return <rect x={-s * 0.16} y={s * 0.3} width={s * 0.32} height={s * 0.42} fill={fill} rx={s * 0.02} />;
      case "coffee":
        return <><rect x={-s * 0.14} y={s * 0.4} width={s * 0.28} height={s * 0.28} fill={fill} /><path d={`M ${s * 0.14} ${s * 0.45} Q ${s * 0.26} ${s * 0.5} ${s * 0.14} ${s * 0.6}`} fill="none" stroke={stroke} strokeWidth={sw * 0.4} /></>;
      case "microphone":
        return <><BoilCircle cx={0} cy={s * 0.35} r={s * 0.14} seed={3} frame={frame} stroke={stroke} strokeWidth={sw * 0.4} fill={fill} /><line x1={0} y1={s * 0.49} x2={0} y2={s * 0.75} stroke={stroke} strokeWidth={sw * 0.4} /></>;
      default:
        return null;
    }
  })();
  if (!node) return null;
  // SVG's own `rotate()` spins the opposite way from this file's `toXY`
  // convention (the reason `rotateLocal` above exists as a hand-written
  // mirror of it, rather than every rig angle just trusting the transform) --
  // `angle` here is a toXY-convention forearm angle, so the transform needs
  // the negated value to extend the item the same way the arm itself bends.
  return (
    <g transform={`translate(${hand.x} ${hand.y}) rotate(${-angle})`}>
      <g transform={isRod ? undefined : `translate(0, ${-HELD_OBJECT_PULL})`}>{node}</g>
    </g>
  );
};

const EXPRESSIONS = { neutral: "neutral", smile: "smile", open: "open", raised: "raised" } as const;
type Expression = keyof typeof EXPRESSIONS;

// Dot eyes + a mouth stroke, turned toward `faceTurn` (-1 left, 0 centre,
// +1 right -- the direction of travel for locomotion poses, per review
// finding 5). Drawn in head-local space so it inherits the figure's own
// rotate/translate for free (swim's sideways head included).
const Face: React.FC<{
  head: { x: number; y: number }; headR: number; expression: Expression; faceTurn: number;
  seedBase: number; frame: number; stroke: string; strokeWidth: number;
}> = ({ head, headR, expression, faceTurn, seedBase, frame, stroke, strokeWidth }) => {
  const eyeY = head.y - headR * 0.08;
  const eyeDx = headR * 0.38 + faceTurn * headR * 0.12;
  const eyeR = Math.max(1.5, headR * 0.1);
  const mouthY = head.y + headR * 0.42;
  const mouthCx = head.x + faceTurn * headR * 0.15;
  const fw = Math.max(1.2, strokeWidth * 0.6);
  return (
    <>
      <BoilCircle cx={head.x - eyeDx} cy={eyeY} r={eyeR} seed={seedBase + 40} frame={frame} stroke={stroke} strokeWidth={fw} fill={stroke} />
      <BoilCircle cx={head.x + eyeDx} cy={eyeY} r={eyeR} seed={seedBase + 41} frame={frame} stroke={stroke} strokeWidth={fw} fill={stroke} />
      {expression === "open" ? (
        <BoilCircle cx={mouthCx} cy={mouthY} r={headR * 0.17} seed={seedBase + 42} frame={frame} stroke={stroke} strokeWidth={fw} />
      ) : expression === "smile" ? (
        <BoilPolyline
          points={[[mouthCx - headR * 0.3, mouthY - headR * 0.03], [mouthCx, mouthY + headR * 0.22], [mouthCx + headR * 0.3, mouthY - headR * 0.03]]}
          seedBase={seedBase + 43} frame={frame} stroke={stroke} strokeWidth={fw}
        />
      ) : (
        <BoilLine x1={mouthCx - headR * 0.26} y1={mouthY} x2={mouthCx + headR * 0.26} y2={mouthY}
          seed={seedBase + 44} frame={frame} stroke={stroke} strokeWidth={fw} />
      )}
      {expression === "raised" && (
        <>
          <BoilLine x1={head.x - eyeDx - headR * 0.22} y1={eyeY - headR * 0.3} x2={head.x - eyeDx + headR * 0.18} y2={eyeY - headR * 0.42}
            seed={seedBase + 45} frame={frame} stroke={stroke} strokeWidth={fw} />
          <BoilLine x1={head.x + eyeDx - headR * 0.18} y1={eyeY - headR * 0.42} x2={head.x + eyeDx + headR * 0.22} y2={eyeY - headR * 0.3}
            seed={seedBase + 46} frame={frame} stroke={stroke} strokeWidth={fw} />
        </>
      )}
    </>
  );
};

const Figure: React.FC<{
  anchorX: number; anchorY: number; rotate: number; scale: number;
  pose: PoseAngles; seedBase: number; frame: number; stroke: string;
  expression: Expression; faceTurn: number;
  hasHat?: boolean; hasHair?: boolean;
  // Stick World (SW1): a dressed figure, with no outfit (the old plain
  // stick figure) leaving everything below exactly as it was.
  outfit?: ResolvedOutfit | null; fill?: string; accent?: string; halo?: string;
}> = ({ anchorX, anchorY, rotate, scale, pose, seedBase, frame, stroke, expression, faceTurn, hasHat, hasHair, outfit, fill, accent, halo }) => {
  const strokeWidth = Math.max(2.5, scale * 0.011);
  const j = computeJoints(scale, pose);
  const { headR, hip, shoulderA, shoulderB, head, elbowA, handA, elbowB, handB, hipA, hipB, kneeA, footA, kneeB, footB } = j;
  const headPivot = { x: (shoulderA.x + shoulderB.x) / 2, y: (shoulderA.y + shoulderB.y) / 2 };
  const bodyStyle = outfit?.body ? BODY_STYLES[outfit.body] : undefined;
  const headStyle = outfit?.head ? HEAD_STYLES[outfit.head] : undefined;
  const hairStyle = outfit?.hair ? HAIR_STYLES[outfit.hair] : undefined;
  const garmentFill = fill || stroke;
  // A soft colour glow in the opposite polarity to `stroke`, so the figure
  // reads against a generated picture regardless of what is directly behind
  // any one limb (STICK_WORLD_PLAN.md SW1) -- cheap at this size (a thin
  // rig, not a full-frame image; contrast PaperCollage's own note on why it
  // keeps to a single filter over a full still).
  const haloStyle: React.CSSProperties | undefined = halo
    ? { filter: `drop-shadow(0 0 2px ${halo}) drop-shadow(0 0 2px ${halo})` }
    : undefined;

  const dressed = !!outfit;
  const ink = "#1A1A1A";
  const skin = "#F3E3C8";
  const arm = (shoulder: Pt, elbow: Pt, hand: Pt, sleeve: number, k: number) => (
    <>
      {sleeve <= 0 && <BoilLine x1={shoulder.x} y1={shoulder.y} x2={elbow.x} y2={elbow.y} seed={seedBase + k} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />}
      {sleeve < 0.99 && <BoilLine x1={elbow.x} y1={elbow.y} x2={hand.x} y2={hand.y} seed={seedBase + k + 1} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />}
      {bodyStyle && sleeve > 0 && (
        <SleeveArm shoulder={shoulder} elbow={elbow} hand={hand} sleeve={sleeve} fill={garmentFill} stroke={stroke} strokeWidth={strokeWidth} />
      )}
      {dressed && <circle cx={hand.x} cy={hand.y} r={strokeWidth * 1.15} fill={skin} stroke={stroke} strokeWidth={strokeWidth * 0.6} />}
    </>
  );
  const sleeveLen = bodyStyle ? bodyStyle.sleeve : 0;
  const heldName = outfit?.held;
  const heldAngle = heldName && UPRIGHT_RODS.has(heldName) ? 174 : heldName && ROD_NAMES.has(heldName) ? 6 : pose.armA + pose.armAElbow;
  const hairColor = ["#2E2622", "#6B5B4B", "#8A8A86"][Math.round(seedBase / 100) % 3];

  return (
    <g transform={`translate(${anchorX} ${anchorY}) rotate(${rotate})`} style={haloStyle}>
      {/* hair behind the head */}
      {dressed && hairStyle && (
        <Hair head={head} headR={headR} style={hairStyle} color={hairColor} stroke={stroke} strokeWidth={strokeWidth} seedBase={seedBase + 300} frame={frame} layer="back" />
      )}
      {/* legs and torso first: a garment is drawn over them */}
      <BoilLine x1={hip.x} y1={hip.y} x2={shoulderA.x} y2={shoulderA.y} seed={seedBase + 1} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />
      <BoilLine x1={hip.x} y1={hip.y} x2={shoulderB.x} y2={shoulderB.y} seed={seedBase + 1.5} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />
      <BoilLine x1={hip.x} y1={hip.y} x2={hipA.x} y2={hipA.y} seed={seedBase + 5.5} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />
      <BoilLine x1={hip.x} y1={hip.y} x2={hipB.x} y2={hipB.y} seed={seedBase + 5.7} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />
      <BoilLine x1={hipA.x} y1={hipA.y} x2={kneeA.x} y2={kneeA.y} seed={seedBase + 6} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />
      <BoilLine x1={kneeA.x} y1={kneeA.y} x2={footA.x} y2={footA.y} seed={seedBase + 7} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />
      <BoilLine x1={hipB.x} y1={hipB.y} x2={kneeB.x} y2={kneeB.y} seed={seedBase + 8} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />
      <BoilLine x1={kneeB.x} y1={kneeB.y} x2={footB.x} y2={footB.y} seed={seedBase + 9} frame={frame} stroke={stroke} strokeWidth={strokeWidth} />
      {bodyStyle && (
        <Garment joints={j} pose={pose} scale={scale} style={bodyStyle} fill={garmentFill} stroke={stroke} strokeWidth={strokeWidth} seedBase={seedBase + 100} frame={frame} />
      )}
      {/* arms over the garment */}
      {arm(shoulderA, elbowA, handA, sleeveLen, 2)}
      {arm(shoulderB, elbowB, handB, sleeveLen, 4)}
      <g transform={`rotate(${pose.headTilt} ${headPivot.x} ${headPivot.y})`}>
        <BoilCircle cx={head.x} cy={head.y} r={headR} seed={seedBase + 10} frame={frame} stroke={stroke} strokeWidth={strokeWidth} fill={dressed ? skin : "none"} />
        <Face head={head} headR={headR} expression={expression} faceTurn={faceTurn} seedBase={seedBase} frame={frame} stroke={dressed ? ink : stroke} strokeWidth={strokeWidth} />
        {outfit ? (
          <>
            {hairStyle && (
              <Hair head={head} headR={headR} style={hairStyle} color={hairColor} stroke={stroke} strokeWidth={strokeWidth} seedBase={seedBase + 300} frame={frame} layer="front" />
            )}
            {headStyle && (
              <Headwear head={head} headR={headR} style={headStyle} fill={garmentFill} stroke={stroke} accent={accent || stroke} strokeWidth={strokeWidth} seedBase={seedBase + 200} frame={frame} />
            )}
          </>
        ) : (
          <>
            {hasHat && (
              <BoilPolyline
                points={[[head.x - headR * 1.1, head.y - headR * 0.3], [head.x, head.y - headR * 1.7], [head.x + headR * 1.1, head.y - headR * 0.3]]}
                seedBase={seedBase + 20} frame={frame} stroke={stroke} strokeWidth={strokeWidth}
              />
            )}
            {hasHair && [-0.6, 0, 0.6].map((k, i) => (
              <BoilLine key={i} x1={head.x + k * headR} y1={head.y - headR * 0.9}
                x2={head.x + k * headR * 1.3} y2={head.y - headR * 1.5}
                seed={seedBase + 30 + i} frame={frame} stroke={stroke} strokeWidth={Math.max(1.5, strokeWidth * 0.7)} />
            ))}
          </>
        )}
      </g>
      {outfit?.held && (
        <HeldArt hand={handA} angle={heldAngle} s={scale * 0.22} fill={garmentFill} stroke={stroke} accent={accent || stroke} frame={frame} name={outfit.held} />
      )}
    </g>
  );
};

// --- ground: whiteboard, paper, night or underwater -------------------------
const TextureOverlay: React.FC<{ dark: boolean }> = ({ dark }) => (
  <AbsoluteFill style={{ opacity: dark ? 0.18 : 0.1, mixBlendMode: dark ? "screen" : "multiply" }}>
    <svg width="100%" height="100%">
      <filter id="stick-paper">
        <feTurbulence type="fractalNoise" baseFrequency="0.016" numOctaves="3" seed={4} />
        <feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.6 0" />
      </filter>
      <rect width="100%" height="100%" fill={dark ? "#FFFFFF" : "#000000"} filter="url(#stick-paper)" />
    </svg>
  </AbsoluteFill>
);

const Ground: React.FC<{ ground: StickGround; theme: Theme; width: number; height: number; frame: number }> = (
  { ground, theme, width, height, frame },
) => {
  if (ground === "underwater") {
    const baseY = height * 0.84;
    return (
      <AbsoluteFill style={{ background: "linear-gradient(180deg, #164258 0%, #0A2434 100%)" }}>
        <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
          {[0.18, 0.32, 0.47, 0.62, 0.77].map((frac, i) => {
            const y = height * frac;
            const phase = frame * 0.03 + i * 0.9;
            const pts: Array<[number, number]> = Array.from({ length: 14 }, (_, k) => {
              const x = (width / 13) * k;
              return [x, y + Math.sin((k / 13) * 3 * Math.PI + phase) * (14 + i * 2)];
            });
            const d = pts.map((p, k) => `${k === 0 ? "M" : "L"} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
            return <path key={i} d={d} stroke="#6FC3E0" strokeWidth={2.4} fill="none" opacity={0.55} />;
          })}
        </svg>
        <AbsoluteFill style={{
          top: baseY, height: height - baseY,
          background: "linear-gradient(180deg, rgba(10,36,52,0) 0%, rgba(6,20,30,0.55) 100%)",
        }} />
      </AbsoluteFill>
    );
  }
  const bg = ground === "night" ? "#15151D" : theme.paper ? "#F4ECD8" : "#FDFDFB";
  const groundColor = ground === "night" ? "#4A4A58" : "#B9B2A0";
  // A few overlapping boiled segments instead of one dead-straight <line> --
  // "slightly sketchy", the way the rest of the scene already wobbles
  // (review finding 7), not a second hand-drawn pass over it.
  const gy = height * GROUND_LINE_FRAC;
  const segs: Array<[number, number, number, number]> = [
    [width * 0.05, gy, width * 0.32, gy + 1],
    [width * 0.3, gy - 1, width * 0.62, gy],
    [width * 0.6, gy, width * 0.95, gy - 1],
  ];
  return (
    <AbsoluteFill style={{ background: bg }}>
      {(ground === "paper" || ground === "night") && <TextureOverlay dark={ground === "night"} />}
      <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
        {segs.map(([x1, y1, x2, y2], i) => (
          <BoilLine key={i} x1={x1} y1={y1} x2={x2} y2={y2} seed={9000 + i} frame={frame}
            stroke={groundColor} strokeWidth={2.5} />
        ))}
      </svg>
    </AbsoluteFill>
  );
};

// --- props: simple line-drawn SVG, unknown -> none --------------------------
// `accent` colours exactly one small sub-shape per prop (the theme's single
// accent colour, review finding 7) -- everything else stays line art.
const PropArt: React.FC<{ name: StickProp; ox: number; oy: number; s: number; stroke: string; accent: string; frame: number }> = (
  { name, ox, oy, s, stroke, accent, frame },
) => {
  const sw = Math.max(2.5, s * 0.0185);
  const P = (pts: Array<[number, number]>): Array<[number, number]> => pts.map(([x, y]) => [ox + x * s, oy + y * s]);
  switch (name) {
    case "house":
      return (
        <>
          <BoilPolyline points={P([[-0.5, 0], [0, -0.62], [0.5, 0]])} seedBase={1} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilPolyline points={P([[-0.42, 0], [0.42, 0], [0.42, 0.56], [-0.42, 0.56]])} closed seedBase={10} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilPolyline points={P([[-0.08, 0.56], [0.08, 0.56], [0.08, 0.3], [-0.08, 0.3]])} closed seedBase={20} frame={frame} stroke={accent} strokeWidth={sw} />
        </>
      );
    case "tree":
      return (
        <>
          <BoilLine x1={ox} y1={oy + 0.5 * s} x2={ox} y2={oy - 0.05 * s} seed={1} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilCircle cx={ox} cy={oy - 0.32 * s} r={0.33 * s} seed={2} frame={frame} stroke={stroke} strokeWidth={sw} />
        </>
      );
    case "car":
      return (
        <>
          <BoilPolyline points={P([[-0.5, 0.2], [0.5, 0.2], [0.5, -0.05], [-0.5, -0.05]])} closed seedBase={1} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilPolyline points={P([[-0.25, -0.05], [-0.14, -0.3], [0.14, -0.3], [0.25, -0.05]])} seedBase={10} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilCircle cx={ox - 0.3 * s} cy={oy + 0.2 * s} r={0.11 * s} seed={20} frame={frame} stroke={accent} strokeWidth={sw} />
          <BoilCircle cx={ox + 0.3 * s} cy={oy + 0.2 * s} r={0.11 * s} seed={21} frame={frame} stroke={accent} strokeWidth={sw} />
        </>
      );
    case "mountain":
      return (
        <>
          <BoilPolyline points={P([[-0.55, 0.3], [-0.12, -0.5], [0.3, 0.3]])} seedBase={1} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilPolyline points={P([[-0.05, 0.3], [0.32, -0.38], [0.6, 0.3]])} seedBase={10} frame={frame} stroke={stroke} strokeWidth={sw} />
        </>
      );
    case "ship":
      return (
        <>
          <BoilPolyline points={P([[-0.5, 0.15], [-0.35, 0.35], [0.35, 0.35], [0.5, 0.15]])} seedBase={1} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilLine x1={ox} y1={oy + 0.15 * s} x2={ox} y2={oy - 0.55 * s} seed={10} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilPolyline points={P([[0, -0.55], [0.3, -0.42], [0, -0.3]])} seedBase={20} frame={frame} stroke={accent} strokeWidth={sw} />
        </>
      );
    case "wave":
      return (
        <BoilPolyline
          points={Array.from({ length: 7 }, (_, i) => (
            [ox + (-0.5 + i / 6) * s, oy + Math.sin((i / 6) * 2 * Math.PI + frame * 0.05) * 0.18 * s] as [number, number]
          ))}
          seedBase={1} frame={frame} stroke={stroke} strokeWidth={sw}
        />
      );
    case "phone":
      return (
        <>
          <BoilPolyline points={P([[-0.18, -0.4], [0.18, -0.4], [0.18, 0.4], [-0.18, 0.4]])} closed seedBase={1} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilLine x1={ox - 0.06 * s} y1={oy + 0.32 * s} x2={ox + 0.06 * s} y2={oy + 0.32 * s} seed={10} frame={frame} stroke={accent} strokeWidth={sw} />
        </>
      );
    case "money":
      return (
        <>
          <BoilPolyline points={P([[-0.32, -0.18], [0.32, -0.18], [0.32, 0.18], [-0.32, 0.18]])} closed seedBase={1} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilCircle cx={ox} cy={oy} r={0.12 * s} seed={10} frame={frame} stroke={accent} strokeWidth={sw} />
        </>
      );
    case "door":
      return (
        <>
          <BoilPolyline points={P([[-0.22, -0.5], [0.22, -0.5], [0.22, 0.5], [-0.22, 0.5]])} closed seedBase={1} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilCircle cx={ox + 0.14 * s} cy={oy + 0.05 * s} r={0.03 * s} seed={10} frame={frame} stroke={accent} strokeWidth={sw} />
        </>
      );
    case "table":
      return (
        <>
          <BoilLine x1={ox - 0.45 * s} y1={oy - 0.2 * s} x2={ox + 0.45 * s} y2={oy - 0.2 * s} seed={1} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilLine x1={ox - 0.35 * s} y1={oy - 0.2 * s} x2={ox - 0.35 * s} y2={oy + 0.4 * s} seed={10} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilLine x1={ox + 0.35 * s} y1={oy - 0.2 * s} x2={ox + 0.35 * s} y2={oy + 0.4 * s} seed={11} frame={frame} stroke={stroke} strokeWidth={sw} />
        </>
      );
    case "book":
      return (
        <>
          <BoilPolyline points={P([[-0.3, -0.22], [0.3, -0.22], [0.3, 0.22], [-0.3, 0.22]])} closed seedBase={1} frame={frame} stroke={stroke} strokeWidth={sw} />
          <BoilLine x1={ox} y1={oy - 0.22 * s} x2={ox} y2={oy + 0.22 * s} seed={10} frame={frame} stroke={accent} strokeWidth={sw} />
        </>
      );
    case "sun":
      return (
        <>
          <BoilCircle cx={ox} cy={oy} r={0.22 * s} seed={1} frame={frame} stroke={accent} strokeWidth={sw} />
          {Array.from({ length: 8 }, (_, k) => {
            const a = (k / 8) * Math.PI * 2;
            const x1 = ox + Math.cos(a) * 0.3 * s, y1 = oy + Math.sin(a) * 0.3 * s;
            const x2 = ox + Math.cos(a) * 0.45 * s, y2 = oy + Math.sin(a) * 0.45 * s;
            return <BoilLine key={k} x1={x1} y1={y1} x2={x2} y2={y2} seed={10 + k} frame={frame} stroke={stroke} strokeWidth={sw} />;
          })}
        </>
      );
    case "moon":
      return (
        <>
          <BoilPolyline
            points={Array.from({ length: 9 }, (_, k) => {
              const a = (-70 + k * 17.5) * (Math.PI / 180);
              return [ox + Math.sin(a) * 0.3 * s, oy - Math.cos(a) * 0.3 * s] as [number, number];
            })}
            seedBase={1} frame={frame} stroke={stroke} strokeWidth={sw}
          />
          <BoilPolyline
            points={Array.from({ length: 7 }, (_, k) => {
              const a = (-50 + k * 16.6) * (Math.PI / 180);
              return [ox + 0.08 * s + Math.sin(a) * 0.2 * s, oy - Math.cos(a) * 0.2 * s] as [number, number];
            })}
            seedBase={20} frame={frame} stroke={stroke} strokeWidth={sw}
          />
        </>
      );
    case "rain":
      return (
        <>
          {Array.from({ length: 6 }, (_, k) => {
            const x = ox + (-0.4 + k * 0.16) * s;
            return (
              <BoilLine key={k} x1={x} y1={oy - 0.25 * s} x2={x - 0.08 * s} y2={oy + 0.25 * s}
                seed={1 + k} frame={frame} stroke={stroke} strokeWidth={sw * 0.8} />
            );
          })}
        </>
      );
    default:
      return null;
  }
};

// --- caption: handwriting font, top of frame --------------------------------
const Caption: React.FC<{
  text: string; theme: Theme; color: string; width: number; height: number; frame: number; fps: number;
}> = ({ text, color, width, height, frame, fps }) => {
  const a = spring({ frame, fps, config: { damping: 14, mass: 0.6, stiffness: 120 } });
  return (
    <div style={{
      position: "absolute", top: height * 0.06, left: 0, width: "100%", textAlign: "center",
      opacity: a, transform: `scale(${interpolate(a, [0, 1], [0.85, 1])}) rotate(-1.5deg)`,
      fontFamily: handFontStack(), fontWeight: 400, fontSize: height * 0.065,
      color, padding: `0 ${width * 0.08}px`,
    }}>
      {text}
    </div>
  );
};

// Expression per action (review finding 5): everything not listed is neutral.
const EXPRESSION_FOR: Partial<Record<StickAction, Expression>> = {
  cheer: "smile", walk: "smile", run: "smile",
  fall: "open", scared: "open",
  shrug: "raised", think: "raised",
};
// "turned toward the direction of travel" only means something for the four
// locomotion poses -- everyone else faces forward, centred.
const TRAVEL_ACTIONS = new Set<StickAction>(["walk", "run", "swim", "climb"]);
// climb and swim deliberately move away from the ground line (up a wall /
// across water); every other pose is planted on it exactly (finding 8).
const AIRBORNE_ACTIONS = new Set<StickAction>(["climb", "swim"]);

// --- the composition ---------------------------------------------------------
export const StickScene: React.FC<StickSceneProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const theme = withTheme(props.theme);
  const action: StickAction = ACTIONS.includes(props.action as StickAction) ? (props.action as StickAction) : "idle";
  const prop: StickProp = PROP_NAMES.includes(props.prop as StickProp) ? (props.prop as StickProp) : "none";
  const ground: StickGround = GROUNDS.includes(props.ground as StickGround) ? (props.ground as StickGround) : "whiteboard";
  // `people` overrides `figures` when given (contract); era/people/background
  // are all optional and additive -- a beat setting none of them (figuresCount
  // from `figures` as before) renders exactly as it did before this section.
  const figuresCount = props.people && props.people.length
    ? Math.max(1, Math.min(3, props.people.length))
    : Math.max(1, Math.min(3, Math.round(props.figures || 1)));
  const caption = (props.caption || "").trim();
  const era = props.era && ERAS[props.era] ? props.era : undefined;
  const overBg = !!props.background;
  const bgLuma = typeof props.bgLuma === "number" ? props.bgLuma : 128;

  const groundY = height * GROUND_LINE_FRAC;
  // 9:16 sizes and places figures from the frame's own short side, so a
  // portrait canvas never cuts one off sideways (STICK_WORLD_PLAN.md SW1) --
  // for 16:9 the short side is the height, same as before this section.
  const shortSide = Math.min(width, height);
  // Standing height ~50-52% of the short side for one figure, ~47% for 2-3
  // sharing the frame (review finding 4's 45-55% band).
  const dressedScene = !!era || !!(props.people && props.people.length);
  // a portrait canvas has no room for three full-size figures side by side
  const scale = shortSide * (figuresCount > 1 ? (dressedScene && width < height ? 0.36 : 0.47) : 0.52);
  const legLen = scale * 0.48; // climb/swim only -- see AIRBORNE_ACTIONS below
  // Whiteboard/paper needs a dark marker line, night/underwater a light one;
  // over a generated `background` picture the pick instead follows its own
  // measured `bgLuma` (<110 = a dark picture -> light strokes), with `halo`
  // carrying the opposite polarity so a figure reads against a locally
  // lighter or darker patch of the same picture either way. Never theme.fg
  // blindly in the ground case, or a light or dark theme could draw an
  // invisible figure against its own ground.
  const stroke = overBg
    ? (bgLuma < 110 ? "#F3EAD3" : "#1A1A1A")
    : ground === "whiteboard" || ground === "paper"
      ? "#1A1A1A"
      : ground === "underwater" ? "#EAF6FB" : (theme.fg || "#F3EAD3");
  const halo = overBg ? (bgLuma < 110 ? "#15151D" : "#FFFFFF") : undefined;
  const propStroke = ground === "underwater" ? "#DCEFF6" : stroke;
  const accent = theme.accent || "#E0393E";
  // Flat garment fill, one colour per figure, from the era's own palette;
  // with no era (an outfit given directly, with no era prop) a themed
  // fallback keeps every dressed figure readable rather than invisible.
  const palette = era ? ERAS[era].palette : [accent, theme.accent2 || accent, theme.muted || accent];
  // The picture fills the frame with a slow push (STICK_WORLD_PLAN.md SW1).
  const bgPush = interpolate(frame, [0, Math.max(1, durationInFrames - 1)], [1.0, 1.05], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });

  const travel = useMemo(
    () => travelFor(action, frame, durationInFrames, width, height),
    [action, frame, durationInFrames, width, height],
  );

  // Figure(s) sit left-of-centre and the prop on the right third when the
  // two share the frame, instead of a dead-centre figure and an edge-hugging
  // prop (review finding 7).
  const figureBaseX = prop !== "none" ? width * 0.38 : width * 0.5;
  const propGroundFrac = PROP_GROUND_OFFSET[prop];
  const propS = height * 0.3;
  const propOx = width * 0.72;
  const propOy = propGroundFrac != null ? groundY - propGroundFrac * propS : height * 0.26;

  const expression: Expression = EXPRESSION_FOR[action] || "neutral";
  const faceTurn = TRAVEL_ACTIONS.has(action) ? 1 : 0;
  const airborne = AIRBORNE_ACTIONS.has(action);

  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      <style>{fontFaceCss()}</style>
      {props.background ? (
        // Today's ground line is hidden over a generated picture -- the
        // picture IS the ground (STICK_WORLD_PLAN.md SW1).
        <img src={props.background} style={{
          position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover",
          transform: `scale(${bgPush})`, transformOrigin: "center center",
        }} />
      ) : (
        <Ground ground={ground} theme={theme} width={width} height={height} frame={frame} />
      )}
      {prop !== "none" && (
        <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
          <PropArt name={prop} ox={propOx} oy={propOy} s={propS} stroke={propStroke} accent={accent} frame={frame} />
        </svg>
      )}
      <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
        {Array.from({ length: figuresCount }, (_, i) => {
          const pose = dressedScene
            ? calmPose(computePose(action, frame, fps, durationInFrames, i * 6), action)
            : computePose(action, frame, fps, durationInFrames, i * 6);
          const spacing = figuresCount > 1 ? Math.max(width * 0.19, scale * (dressedScene ? 0.66 : 0)) : 0;
          const rawX = figureBaseX + (i - (figuresCount - 1) / 2) * spacing + travel.dx;
          const figScale = scale * (i === 0 ? 1 : i === 1 ? 0.92 : 1.06);
          // Grounded poses are planted exactly on the ground line by their
          // own lowest joint (feet, or hands/head once rotated/bent); climb
          // and swim keep the old fixed-leg-length anchor since they are
          // meant to float above/along the line, not touch it.
          const groundOffset = airborne ? legLen : lowestLocalY(computeJoints(figScale, pose), pose.rotate);
          const rawY = groundY - groundOffset + pose.bob + travel.dy;
          const anchorX = Math.min(width * 0.92, Math.max(width * 0.08, rawX));
          const anchorY = Math.min(height * 0.95, Math.max(height * 0.22, rawY));
          const outfit = resolveOutfit(props.people?.[i]?.outfit, era, i);
          const fill = palette[i % palette.length];
          return (
            <React.Fragment key={i}>
              {overBg && (
                <ellipse cx={anchorX} cy={groundY} rx={figScale * 0.22} ry={figScale * 0.055}
                  fill="rgba(0,0,0,0.35)" style={{ filter: "blur(4px)" }} />
              )}
              <Figure anchorX={anchorX} anchorY={anchorY} rotate={pose.rotate}
                scale={figScale} pose={pose} seedBase={i * 100} frame={frame} stroke={stroke}
                expression={expression} faceTurn={faceTurn}
                hasHat={i === 1} hasHair={i === 2}
                outfit={outfit} fill={fill} accent={accent} halo={halo} />
            </React.Fragment>
          );
        })}
      </svg>
      {caption && <Caption text={caption} theme={theme} color={stroke} width={width} height={height} frame={frame} fps={fps} />}
    </AbsoluteFill>
  );
};
