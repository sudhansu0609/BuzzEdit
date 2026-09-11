/**
 * Direct manipulation of what is on the canvas.
 *
 * Placing two clips side by side used to mean typing numbers into the inspector
 * — a scale, an x, a y, per clip — rendering, and looking to see whether you had
 * guessed right. This draws a box around each clip that is live at the playhead
 * and lets you drag and resize it on the picture itself, which is the only way
 * laying out several pieces of media is anything but arithmetic.
 *
 * The boxes are computed by `lib/stageGeometry`, which mirrors the renderer's
 * placement formulas exactly, so where the box sits is where the picture lands.
 * Corner handles keep the source's aspect ratio, because a clip's frame is never
 * stretched by the renderer — it is fitted — so a box of a different shape could
 * not be honoured and would only mislead.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { setTransform } from '../hooks/api';
import {
  Rect, STAGE_PRESETS, placementFor, rectFor, transformFor,
} from '../lib/stageGeometry';

type Handle = 'move' | 'nw' | 'ne' | 'sw' | 'se';

interface StageClip {
  id: string;
  track: string;
  label: string;
  rect: Rect;
  sourceAspect: number;
  /** Carries a Ken Burns move, which placement has to stop. */
  animated: boolean;
}

interface Props {
  /** Programme time, so the same clips are boxed as are on screen. */
  programmeTime: number;
  /** The picture's rectangle on screen — the stage is letterboxed inside its
   *  container, so this is not the container's own box. */
  pictureRect: { left: number; top: number; width: number; height: number } | null;
}

const MIN_SIZE = 0.05;

export default function StageLayout({ programmeTime, pictureRect }: Props) {
  const { project, updateProject, selectedClipId, setSelectedClip, setError } = useProjectStore();
  const timeline = (project as any)?.timeline;
  const projectId = project?.id;
  const fps = (timeline?.fps_num ?? 30) / (timeline?.fps_den ?? 1) || 30;
  const canvasAspect = (timeline?.width ?? 1920) / (timeline?.height ?? 1080);

  const [drag, setDrag] = useState<
    { id: string; handle: Handle; startX: number; startY: number; origin: Rect } | null>(null);
  // What the boxes show mid-drag. The timeline round-trip is far too slow to
  // drive a drag, so the gesture runs locally and only the settled result is
  // sent.
  const [live, setLive] = useState<Record<string, Rect>>({});
  const pending = useRef<Record<string, Rect>>({});
  const flushTimer = useRef<any>(null);

  const frame = Math.round(programmeTime * fps);

  const clips = useMemo<StageClip[]>(() => {
    const items: any[] = timeline?.items ?? [];
    return items
      .filter((i) => (i.kind ?? 'media') === 'media'
        && (i.track || '').toUpperCase().startsWith('V')
        && i.enabled !== false
        && !timeline?.tracks?.[i.track]?.hidden
        && i.timeline_start_frame <= frame && frame < i.timeline_end_frame)
      .map((i) => {
        const source = timeline?.sources?.[i.source_id];
        const sourceAspect = source?.width && source?.height
          ? source.width / source.height : canvasAspect;
        const placement = placementFor(i.track);
        const t = i.transform;
        return {
          id: i.id,
          track: i.track,
          label: i.label || i.track,
          sourceAspect,
          animated: !!t && (
            (t.scale_end != null && t.scale_end !== t.scale)
            || (t.pos_x_end != null && t.pos_x_end !== t.pos_x)
            || (t.pos_y_end != null && t.pos_y_end !== t.pos_y)),
          rect: live[i.id] ?? rectFor(i.transform, placement, sourceAspect, canvasAspect),
        };
      })
      // Top track last, so the box you are most likely to want is on top of the
      // stack and takes the click.
      .sort((a, b) => parseInt(a.track.replace(/\D/g, '') || '0', 10)
        - parseInt(b.track.replace(/\D/g, '') || '0', 10));
  }, [timeline, frame, canvasAspect, live]);

  // Anything the drag left behind is dropped once the saved timeline comes back,
  // so the boxes go back to being a view of the project rather than of the
  // gesture.
  useEffect(() => { setLive({}); }, [timeline?.revision]);

  const commit = useCallback(() => {
    if (!projectId) return;
    const changes = pending.current;
    pending.current = {};
    Object.entries(changes).forEach(([id, rect]) => {
      const clip = clips.find((c) => c.id === id);
      if (!clip) return;
      const values = transformFor(
        rect, placementFor(clip.track), clip.sourceAspect, canvasAspect);
      setTransform(projectId, id, {
        ...values,
        // A clip with a zoom move takes the renderer's animated path, which
        // pins it full-canvas at 0,0 and throws the placement away — so a box
        // dragged onto a Ken Burns clip would have looked like it did nothing.
        // Placing a clip is a decision to hold it still.
        scale_end: null, pos_x_end: null, pos_y_end: null,
      } as any)
        .then((res: any) => { if (res?.timeline) updateProject({ timeline: res.timeline }); })
        .catch((err: any) => setError(err.message || 'Could not move the clip'));
    });
  }, [projectId, clips, canvasAspect, updateProject, setError]);

  const schedule = useCallback((id: string, rect: Rect) => {
    pending.current[id] = rect;
    if (flushTimer.current) clearTimeout(flushTimer.current);
    flushTimer.current = setTimeout(commit, 180);
  }, [commit]);

  const onPointerDown = useCallback((e: React.PointerEvent, clip: StageClip, handle: Handle) => {
    if (e.button !== 0 || !pictureRect) return;
    e.preventDefault();
    e.stopPropagation();
    setSelectedClip(clip.id);
    setDrag({ id: clip.id, handle, startX: e.clientX, startY: e.clientY, origin: clip.rect });
    (e.target as Element).setPointerCapture?.(e.pointerId);
  }, [pictureRect, setSelectedClip]);

  useEffect(() => {
    if (!drag || !pictureRect) return;

    const move = (e: PointerEvent) => {
      const dx = (e.clientX - drag.startX) / pictureRect.width;
      const dy = (e.clientY - drag.startY) / pictureRect.height;
      const o = drag.origin;
      let next: Rect;

      if (drag.handle === 'move') {
        next = { ...o, left: o.left + dx, top: o.top + dy };
      } else {
        // Corner resize about the opposite corner, driven by the horizontal
        // travel alone so the box cannot be pulled out of the source's shape.
        const grow = drag.handle === 'se' || drag.handle === 'ne' ? dx : -dx;
        const width = Math.max(MIN_SIZE, Math.min(4, o.width + grow));
        const height = o.height * (width / o.width);
        const anchorRight = drag.handle === 'nw' || drag.handle === 'sw';
        const anchorBottom = drag.handle === 'nw' || drag.handle === 'ne';
        next = {
          width,
          height,
          left: anchorRight ? o.left + o.width - width : o.left,
          top: anchorBottom ? o.top + o.height - height : o.top,
        };
      }
      setLive((prev) => ({ ...prev, [drag.id]: next }));
      schedule(drag.id, next);
    };

    const up = () => {
      setDrag(null);
      if (flushTimer.current) clearTimeout(flushTimer.current);
      commit();
    };

    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
  }, [drag, pictureRect, schedule, commit]);

  const applyPreset = useCallback((presetId: string) => {
    const preset = STAGE_PRESETS.find((p) => p.id === presetId);
    if (!preset || !projectId) return;
    preset.slots.forEach((slot, index) => {
      const clip = clips[index];
      if (!clip) return;
      setLive((prev) => ({ ...prev, [clip.id]: slot }));
      pending.current[clip.id] = slot;
    });
    if (flushTimer.current) clearTimeout(flushTimer.current);
    flushTimer.current = setTimeout(commit, 60);
  }, [clips, projectId, commit]);

  if (!pictureRect) return null;

  const px = (rect: Rect) => ({
    left: pictureRect.left + rect.left * pictureRect.width,
    top: pictureRect.top + rect.top * pictureRect.height,
    width: rect.width * pictureRect.width,
    height: rect.height * pictureRect.height,
  });

  return (
    <>
      <div className="stage-boxes" style={{ position: 'absolute', inset: 0, zIndex: 7 }}>
        {clips.map((clip) => {
          const box = px(clip.rect);
          const selected = clip.id === selectedClipId;
          return (
            <div key={clip.id}
              className={`stage-box ${selected ? 'selected' : ''}`}
              style={{ left: box.left, top: box.top, width: box.width, height: box.height }}
              onPointerDown={(e) => onPointerDown(e, clip, 'move')}
              title={`${clip.track} · drag to move, corners to resize`
                + (clip.animated ? ' · placing this clip will hold its zoom move still' : '')}>
              <span className="stage-box-tag">{clip.label}</span>
              {(['nw', 'ne', 'sw', 'se'] as Handle[]).map((h) => (
                <span key={h} className={`stage-handle ${h}`}
                  onPointerDown={(e) => onPointerDown(e, clip, h)} />
              ))}
            </div>
          );
        })}
      </div>

      <div className="stage-presets">
        {clips.length < 2 ? (
          <span className="fx-pill">
            Only one clip here — put another on V2 to lay them out together
          </span>
        ) : STAGE_PRESETS.map((p) => (
          <button key={p.id} className="btn btn-xs" title={p.hint}
            onClick={() => applyPreset(p.id)}>{p.label}</button>
        ))}
      </div>
    </>
  );
}
