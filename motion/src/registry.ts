// Every composition the engine can render, one line each (MOTION_GRAPHICS_PLAN.md
// Architecture). M2 adds MotionMap/MotionChart/MotionStat/MotionTimeline, M3
// adds the shape-transition stingers -- each a new line here, each composition
// in its own file under src/compositions/. Root.tsx just maps over this list.
import type { ComponentType } from "react";
import { KineticText, kineticTextDefaultProps, KineticTextProps } from "./compositions/KineticText";
import { MotionMap, motionMapDefaultProps, MotionMapProps } from "./compositions/MotionMap";
import { MotionChart, motionChartDefaultProps, MotionChartProps } from "./compositions/MotionChart";
import { MotionStat, motionStatDefaultProps, MotionStatProps } from "./compositions/MotionStat";
import { MotionTimeline, motionTimelineDefaultProps, MotionTimelineProps } from "./compositions/MotionTimeline";
import {
  Bars, Ink, Iris, PaperTear, ShapeWipe,
  transitionStingerDefaultProps, TransitionStingerProps,
} from "./compositions/TransitionStinger";
import { StickScene, stickSceneDefaultProps, StickSceneProps } from "./compositions/StickScene";
import { Parallax, parallaxDefaultProps, ParallaxProps } from "./compositions/Parallax";
import { PaperCollage, paperCollageDefaultProps, PaperCollageProps } from "./compositions/PaperCollage";
import { FlowDiagram, flowDiagramDefaultProps, FlowDiagramProps } from "./compositions/FlowDiagram";
import { IconList, iconListDefaultProps, IconListProps } from "./compositions/IconList";
import { UiMockup, uiMockupDefaultProps, UiMockupProps } from "./compositions/UiMockup";
import { Callout, calloutDefaultProps, CalloutProps } from "./compositions/Callout";

export type CompositionEntry<P = any> = {
  id: string;
  component: ComponentType<P>;
  defaultWidth: number;
  defaultHeight: number;
  defaultFps: number;
  defaultDurationInFrames: number;
  defaultProps: P;
};

// M3's five transparent transition stingers (0.8s @ 30fps = 24 frames) share
// one props shape (TransitionStingerProps); presentation/motion_transitions.py
// picks the composition id from the `[fx: <name> dur=0.8]` directive's name.
const STINGER_DEFAULTS = {
  defaultWidth: 1920, defaultHeight: 1080, defaultFps: 30, defaultDurationInFrames: 24,
  defaultProps: transitionStingerDefaultProps as TransitionStingerProps,
};

export const COMPOSITIONS: CompositionEntry[] = [
  {
    id: "KineticText",
    component: KineticText,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 120,
    defaultProps: kineticTextDefaultProps as KineticTextProps,
  },
  { id: "ShapeWipe", component: ShapeWipe, ...STINGER_DEFAULTS },
  { id: "Iris", component: Iris, ...STINGER_DEFAULTS },
  { id: "Ink", component: Ink, ...STINGER_DEFAULTS },
  { id: "Bars", component: Bars, ...STINGER_DEFAULTS },
  { id: "PaperTear", component: PaperTear, ...STINGER_DEFAULTS },
  {
    id: "MotionMap",
    component: MotionMap,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 150,
    defaultProps: motionMapDefaultProps as MotionMapProps,
  },
  {
    id: "MotionChart",
    component: MotionChart,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 150,
    defaultProps: motionChartDefaultProps as MotionChartProps,
  },
  {
    id: "MotionStat",
    component: MotionStat,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 120,
    defaultProps: motionStatDefaultProps as MotionStatProps,
  },
  {
    id: "MotionTimeline",
    component: MotionTimeline,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 150,
    defaultProps: motionTimelineDefaultProps as MotionTimelineProps,
  },
  {
    id: "StickScene",
    component: StickScene,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 180,
    defaultProps: stickSceneDefaultProps as StickSceneProps,
  },
  {
    id: "Parallax",
    component: Parallax,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 120,
    defaultProps: parallaxDefaultProps as ParallaxProps,
  },
  {
    id: "PaperCollage",
    component: PaperCollage,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 210,
    defaultProps: paperCollageDefaultProps as PaperCollageProps,
  },
  {
    id: "FlowDiagram",
    component: FlowDiagram,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 150,
    defaultProps: flowDiagramDefaultProps as FlowDiagramProps,
  },
  {
    id: "IconList",
    component: IconList,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 150,
    defaultProps: iconListDefaultProps as IconListProps,
  },
  {
    id: "UiMockup",
    component: UiMockup,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 150,
    defaultProps: uiMockupDefaultProps as UiMockupProps,
  },
  {
    id: "Callout",
    component: Callout,
    defaultWidth: 1920,
    defaultHeight: 1080,
    defaultFps: 30,
    defaultDurationInFrames: 150,
    defaultProps: calloutDefaultProps as CalloutProps,
  },
];
