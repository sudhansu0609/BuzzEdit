// Registers every entry of registry.ts as a Composition. calculateMetadata
// reads width/height/fps/durationInFrames back out of the per-render props
// render.mjs builds (job.width/height/fps/frames merged into inputProps) --
// this is how one registered composition serves any size/length job rather
// than needing a fixed one per render (MOTION_GRAPHICS_PLAN.md Architecture).
import React from "react";
import { Composition } from "remotion";
import { COMPOSITIONS } from "./registry";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {COMPOSITIONS.map((entry) => (
        <Composition
          key={entry.id}
          id={entry.id}
          component={entry.component}
          width={entry.defaultWidth}
          height={entry.defaultHeight}
          fps={entry.defaultFps}
          durationInFrames={entry.defaultDurationInFrames}
          defaultProps={entry.defaultProps}
          calculateMetadata={async ({ props }) => {
            const p = (props || {}) as Record<string, unknown>;
            return {
              width: Number(p.width) || entry.defaultWidth,
              height: Number(p.height) || entry.defaultHeight,
              fps: Number(p.fps) || entry.defaultFps,
              durationInFrames: Number(p.durationInFrames) || entry.defaultDurationInFrames,
            };
          }}
        />
      ))}
    </>
  );
};
