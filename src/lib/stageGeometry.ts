/**
 * Where a clip's picture actually lands on the canvas, and how to get back.
 *
 * The inspector exposes `scale`, `pos_x` and `pos_y` as three numbers, which is
 * a poor way to lay two clips out side by side — you nudge a slider, render, and
 * look. This turns the same three numbers into a rectangle you can drag, so the
 * arithmetic has to agree with the renderer *exactly* or the box lies about
 * where the picture will be.
 *
 * There are two placements, and they do not share a formula:
 *
 * **Overlay** (V2 and above, `build_overlay_transform`): the clip is fitted into
 * a box of `canvas * scale` and composited at
 * `x = (W-w)/2 + pos_x*W/2`. `pos_x = 1` therefore shifts it half a canvas from
 * centre, which for a small overlay hangs half of it off the edge — deliberate,
 * and how you park a picture-in-picture partly out of frame.
 *
 * **Canvas** (V1, `build_canvas_transform`): the clip is fitted into the same
 * box, then *padded* up to the canvas at
 * `place = (canvas - canvas*scale)/2 * (1 + pos)`. Here `pos = 1` is flush with
 * the edge, and the travel shrinks to nothing as the picture fills the frame —
 * which is right, because at `scale >= 1` there is nowhere left to move it and
 * the pan becomes a crop offset into the zoom instead.
 *
 * Note the renderer measures its room from the *box*, not from the fitted
 * picture, so a clip whose aspect differs from the canvas stops slightly short
 * of the edge. That is mirrored here rather than corrected: the box has to match
 * what gets drawn, not what would have been tidier.
 */

export interface Rect {
  /** All four in 0..1 of the canvas; left/top may be negative for an overlay
   *  deliberately hanging off the edge. */
  left: number;
  top: number;
  width: number;
  height: number;
}

export type Placement = 'canvas' | 'overlay';

export function placementFor(track: string): Placement {
  return (track || '').toUpperCase() === 'V1' ? 'canvas' : 'overlay';
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** The fitted picture size, as a fraction of the canvas, for a given scale. */
function fitted(scale: number, sourceAspect: number, canvasAspect: number) {
  const s = Math.max(0.05, scale);
  // `force_original_aspect_ratio=decrease` into a box of (s, s): whichever side
  // binds first decides, exactly as ffmpeg does it.
  if (sourceAspect >= canvasAspect) {
    return { width: s, height: s * (canvasAspect / sourceAspect) };
  }
  return { width: s * (sourceAspect / canvasAspect), height: s };
}

export interface TransformValues {
  scale?: number;
  pos_x?: number;
  pos_y?: number;
}

/** The rectangle the renderer will draw this clip into. */
export function rectFor(
  transform: TransformValues | null | undefined,
  placement: Placement,
  sourceAspect: number,
  canvasAspect: number,
): Rect {
  const scale = Math.max(0.05, transform?.scale ?? 1);
  const posX = clamp(transform?.pos_x ?? 0, -1, 1);
  const posY = clamp(transform?.pos_y ?? 0, -1, 1);
  const size = fitted(scale, sourceAspect, canvasAspect);

  if (placement === 'overlay') {
    return {
      left: (1 - size.width) / 2 + posX / 2,
      top: (1 - size.height) / 2 + posY / 2,
      width: size.width,
      height: size.height,
    };
  }

  // Canvas: at scale >= 1 the picture covers the frame and the pan is a crop
  // into it, so the visible rectangle is simply the whole canvas.
  if (scale >= 1) return { left: 0, top: 0, width: 1, height: 1 };

  // Room measured from the box, as the renderer does.
  return {
    left: ((1 - scale) / 2) * (1 + posX),
    top: ((1 - scale) / 2) * (1 + posY),
    width: size.width,
    height: size.height,
  };
}

/**
 * The inverse: the transform that puts the picture where this rectangle is.
 *
 * `width` alone decides the scale — the box keeps the source's aspect while it
 * is dragged, so height carries no extra information and letting both speak
 * would let rounding pull them apart.
 */
export function transformFor(
  rect: Rect,
  placement: Placement,
  sourceAspect: number,
  canvasAspect: number,
): Required<TransformValues> {
  // Undo the fit: the width the user dragged is the *picture*, and scale is the
  // box it was fitted into.
  const scale = clamp(
    sourceAspect >= canvasAspect ? rect.width : rect.width * (canvasAspect / sourceAspect),
    0.05, 4,
  );
  const size = fitted(scale, sourceAspect, canvasAspect);

  if (placement === 'overlay') {
    return {
      scale,
      pos_x: clamp(2 * rect.left + size.width - 1, -1, 1),
      pos_y: clamp(2 * rect.top + size.height - 1, -1, 1),
    };
  }

  if (scale >= 1) {
    return { scale, pos_x: clamp(rect.left, -1, 1), pos_y: clamp(rect.top, -1, 1) };
  }
  const roomX = (1 - scale) / 2;
  const roomY = (1 - scale) / 2;
  return {
    scale,
    // No room means no meaningful position; keep it centred rather than
    // dividing by zero and flinging the clip to an edge.
    pos_x: roomX > 1e-6 ? clamp(rect.left / roomX - 1, -1, 1) : 0,
    pos_y: roomY > 1e-6 ? clamp(rect.top / roomY - 1, -1, 1) : 0,
  };
}

/** Ready-made layouts, because "two clips, side by side" should be one click. */
export interface StagePreset {
  id: string;
  label: string;
  hint: string;
  /** Rectangles in placement order; a clip past the end keeps what it has. */
  slots: Rect[];
}

export const STAGE_PRESETS: StagePreset[] = [
  {
    id: 'side_by_side', label: 'Side by side',
    hint: 'Two clips filling half the frame each',
    slots: [
      { left: 0, top: 0.25, width: 0.5, height: 0.5 },
      { left: 0.5, top: 0.25, width: 0.5, height: 0.5 },
    ],
  },
  {
    id: 'stacked', label: 'Stacked',
    hint: 'Two clips, one above the other — the vertical cut',
    slots: [
      { left: 0.25, top: 0, width: 0.5, height: 0.5 },
      { left: 0.25, top: 0.5, width: 0.5, height: 0.5 },
    ],
  },
  {
    id: 'pip', label: 'Picture in picture',
    hint: 'Full frame with a small inset in the corner',
    slots: [
      { left: 0, top: 0, width: 1, height: 1 },
      { left: 0.66, top: 0.66, width: 0.3, height: 0.3 },
    ],
  },
  {
    id: 'grid', label: 'Quad',
    hint: 'Four clips in a grid',
    slots: [
      { left: 0, top: 0.25, width: 0.5, height: 0.25 },
      { left: 0.5, top: 0.25, width: 0.5, height: 0.25 },
      { left: 0, top: 0.5, width: 0.5, height: 0.25 },
      { left: 0.5, top: 0.5, width: 0.5, height: 0.25 },
    ],
  },
];
