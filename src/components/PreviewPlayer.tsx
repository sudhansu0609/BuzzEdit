import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import { useProjectStore } from '../hooks/store';
import {
  API_BASE, previewFrameUrl, getPreviewProxy, buildPreviewProxy, ProxyStatus,
} from '../hooks/api';
import StageLayout from './StageLayout';

/**
 * `fx` is the dressed programme — grade, atmosphere, captions, B-roll, bars —
 * which none of the other three modes can show. `cut` plays the source file and
 * skips the struck words, so it is honest about the edit and silent about
 * everything else; `result` is the last full render and may be well behind the
 * timeline. FX sits between them: always current, and dressed.
 */
type PreviewMode = 'cut' | 'result' | 'source' | 'fx';

export default function PreviewPlayer() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const { project, currentTime, setCurrentTime, isPlaying, setIsPlaying, isScrubbing } = useProjectStore();
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [videoReady, setVideoReady] = useState(false);
  // Path to a seek-friendly H.264 proxy of the source, once the backend has
  // built one. iPhone 10-bit HEVC cannot be seeked in the browser, so the live
  // "Cut" preview plays straight through the removed fumbles on the raw file.
  // The proxy is what makes the cuts actually skip. null = use the raw source.
  const [proxyPath, setProxyPath] = useState<string | null>(null);
  const [proxyBuilding, setProxyBuilding] = useState(false);

  const sourceVideo = project?.source_video || project?.sourceVideo;
  const renderedOutput: string | undefined =
    (project as any)?.output_path || (project as any)?.rendered_video || project?.outputPath;
  // Bumps every time the render is regenerated (the file's mtime). Appended to
  // the Rendered stream URL so the browser refetches instead of replaying the
  // previous render — without it a fresh zoom/B-roll pass writes the same
  // filename and the preview keeps showing the stale, flat video.
  const renderVersion = (project as any)?.output_version;
  const timeline = (project as any)?.timeline;

  // The kept source ranges (seconds), straight from the V1 segments. Playing
  // these in order — skipping the gaps between them — reconstructs the CURRENT
  // cut live from the source, so it always matches the transcript strikes with
  // no re-render. This is what fixes "the strikes aren't reflected in the cut":
  // the rendered file can be stale after an edit, but this never is.
  const keptIntervals = useMemo<Array<[number, number]>>(() => {
    const items: any[] = Array.isArray(timeline?.items) ? timeline.items : [];
    const fps = (timeline?.fps_num ?? 30) / (timeline?.fps_den ?? 1) || 30;
    return items
      .filter((i) => i.track === 'V1' && i.source_end_frame > i.source_start_frame)
      .map((i) => [i.source_start_frame / fps, i.source_end_frame / fps] as [number, number])
      .sort((a, b) => a[0] - b[0]);
  }, [timeline]);

  const hasCut = keptIntervals.length > 0;

  // Edit-time mapping. The cut preview plays the SOURCE file and skips gaps, so
  // video.currentTime is source time — but the user is watching the *edit*, and
  // the seek bar/clock must run on edit time or a 3:15 source cut down to 1:45
  // still reads "/ 03:15" and looks like the cuts were never applied.
  const cutDuration = useMemo(
    () => keptIntervals.reduce((acc, [s, e]) => acc + (e - s), 0), [keptIntervals]);

  const srcToEdit = useCallback((t: number) => {
    let acc = 0;
    for (const [s, e] of keptIntervals) {
      if (t >= e) { acc += e - s; continue; }
      if (t > s) acc += t - s;
      break;
    }
    return acc;
  }, [keptIntervals]);

  const editToSrc = useCallback((editTime: number) => {
    let remaining = Math.max(0, editTime);
    for (let i = 0; i < keptIntervals.length; i++) {
      const [s, e] = keptIntervals[i];
      if (remaining <= e - s || i === keptIntervals.length - 1) {
        return Math.min(s + remaining, e);
      }
      remaining -= e - s;
    }
    return keptIntervals[0]?.[0] ?? 0;
  }, [keptIntervals]);

  // Ask the backend for a seek-friendly proxy of the source. HEVC/10-bit phone
  // footage can't be seeked in Chromium, so without this the cut preview plays
  // through every removed region. Polls while the transcode runs, then swaps the
  // preview over to the proxy. Plain H.264 reports `not_needed` and we stay on
  // the original. Cancelled cleanly when the source changes.
  useEffect(() => {
    setProxyPath(null);
    setProxyBuilding(false);
    if (!sourceVideo) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const res = await fetch(
          `${API_BASE}/api/media/proxy?path=${encodeURIComponent(sourceVideo)}`);
        if (cancelled) return;
        const data = await res.json();
        if (cancelled) return;
        if (data.status === 'ready' && data.path) {
          setProxyPath(data.path);
          setProxyBuilding(false);
        } else if (data.status === 'building') {
          setProxyBuilding(true);
          timer = setTimeout(poll, 2500);
        } else {
          // not_needed / none / error — the raw source is what we preview.
          setProxyBuilding(false);
        }
      } catch {
        if (!cancelled) setProxyBuilding(false);
      }
    };
    poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [sourceVideo]);

  // Default: the live cut when we have a timeline, else the source.
  const [mode, setMode] = useState<PreviewMode>('cut');
  useEffect(() => {
    setMode(hasCut ? 'cut' : (renderedOutput ? 'result' : 'source'));
  }, [hasCut, renderedOutput, project?.id]);

  // --- the dressed preview ------------------------------------------------
  // Programme time, which is what the renderer counts in and what the FX routes
  // expect. The store's currentTime is SOURCE time (the cut preview plays the
  // original file and jumps the gaps), so everything crossing into FX converts
  // here and everything leaving it converts back — the store's meaning never
  // changes, and the timeline and panels keep working exactly as before.
  const programOffset = ((timeline?.program_offset_frames ?? 0)
    / ((timeline?.fps_num ?? 30) / (timeline?.fps_den ?? 1) || 30));
  const revision: number | undefined = timeline?.revision;
  const programmeTime = srcToEdit(currentTime) + programOffset;
  const programmeDuration = cutDuration + programOffset;

  const [proxy, setProxy] = useState<ProxyStatus>({ state: 'none', stale: true });
  const [fxFrame, setFxFrame] = useState<string | null>(null);
  const [fxLoading, setFxLoading] = useState(false);
  const [fxEmpty, setFxEmpty] = useState(false);
  const stageRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const pictureRef = useRef<HTMLImageElement | HTMLVideoElement | null>(null);

  // Direct layout of the clips on the canvas. Only offered on the dressed
  // preview: on the cut preview the stage shows the source file alone, so a box
  // drawn round an overlay would be floating over a picture that does not
  // contain it.
  const [layoutMode, setLayoutMode] = useState(false);
  // The picture's rectangle inside the container. The stage is letterboxed, so
  // this is emphatically not the container's own box — the boxes would be
  // stretched across the black bars.
  const [pictureRect, setPictureRect] = useState<
    { left: number; top: number; width: number; height: number } | null>(null);


  // The proxy is only worth playing when it was built from the edit you are
  // looking at; a stale one is offered, but labelled.
  const proxyReady = proxy.state === 'ready' && !!proxy.path;
  const useProxyVideo = mode === 'fx' && proxyReady;

  useEffect(() => {
    if (!layoutMode) { setPictureRect(null); return; }
    const measure = () => {
      const el = pictureRef.current;
      const box = containerRef.current;
      if (!el || !box) return;
      const a = el.getBoundingClientRect();
      const b = box.getBoundingClientRect();
      if (a.width < 2 || a.height < 2) return;
      setPictureRect({ left: a.left - b.left, top: a.top - b.top, width: a.width, height: a.height });
    };
    measure();
    const observer = new ResizeObserver(measure);
    if (containerRef.current) observer.observe(containerRef.current);
    if (pictureRef.current) observer.observe(pictureRef.current);
    window.addEventListener('resize', measure);
    return () => { observer.disconnect(); window.removeEventListener('resize', measure); };
  }, [layoutMode, fxFrame, useProxyVideo, mode]);

  const refreshProxy = useCallback(async () => {
    if (!project?.id) return;
    try {
      setProxy(await getPreviewProxy(project.id));
    } catch {
      setProxy({ state: 'none', stale: true });
    }
  }, [project?.id]);

  useEffect(() => { refreshProxy(); }, [refreshProxy, revision]);

  // Poll only while something is actually being built.
  useEffect(() => {
    if (proxy.state !== 'building') return;
    const timer = setInterval(refreshProxy, 2500);
    return () => clearInterval(timer);
  }, [proxy.state, refreshProxy]);

  const startProxy = useCallback(async () => {
    if (!project?.id) return;
    setProxy({ state: 'building', elapsed: 0 });
    try {
      await buildPreviewProxy(project.id, 540);
    } catch {
      setProxy({ state: 'error', detail: 'Could not start the build' });
    }
  }, [project?.id]);

  // The still: fetched when the playhead settles, not while it moves. A frame
  // costs about half a second of ffmpeg, so firing on every scrub tick would
  // queue up work for positions the user has already left — and the answer would
  // arrive after they had moved on anyway.
  useEffect(() => {
    if (mode !== 'fx' || useProxyVideo || !project?.id) return;
    let cancelled = false;
    const width = Math.min(1280, Math.max(320,
      Math.round((stageRef.current?.clientWidth || 640) * (window.devicePixelRatio || 1))));
    const url = previewFrameUrl(project.id, programmeTime, width, revision);

    const timer = setTimeout(() => {
      setFxLoading(true);
      // Decoded off-screen first, so the frame on screen is replaced rather than
      // blanked — scrubbing through a cached run should not strobe.
      const img = new Image();
      img.onload = () => {
        if (cancelled) return;
        setFxFrame(url); setFxEmpty(false); setFxLoading(false);
      };
      img.onerror = () => {
        if (cancelled) return;
        // 204: nothing on V1 there — a gap, or past the end.
        setFxEmpty(true); setFxLoading(false);
      };
      img.src = url;
    }, 140);

    return () => { cancelled = true; clearTimeout(timer); };
  }, [mode, useProxyVideo, project?.id, programmeTime, revision]);

  // Cut and Source modes play the source itself — through the proxy when one is
  // ready, so seeks (and therefore the gap-skipping that IS the cut) work.
  // Rendered mode plays the finished file, which is already H.264 and seekable.
  const previewSource = proxyPath || sourceVideo;
  const activeSrc = useProxyVideo
    ? proxy.path!
    : (mode === 'result' && renderedOutput) ? renderedOutput : previewSource;
  // FX-still mode has no video at all; the stage is an <img>.
  const showsVideo = mode !== 'fx' || useProxyVideo;

  // Refs so the (stable) timeupdate handler reads the latest values.
  const scrubRef = useRef(isScrubbing); scrubRef.current = isScrubbing;
  const modeRef = useRef(mode); modeRef.current = mode;
  const intervalsRef = useRef(keptIntervals); intervalsRef.current = keptIntervals;
  const proxyRef = useRef(useProxyVideo); proxyRef.current = useProxyVideo;
  const offsetRef = useRef(programOffset); offsetRef.current = programOffset;

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !activeSrc) return;
    setVideoReady(false);
    setVideoError(null);

    const handleTimeUpdate = () => {
      if (scrubRef.current) return;
      // The proxy runs on programme time; the store speaks source time. Convert
      // on the way out so the timeline and transcript stay where they were.
      setCurrentTime(proxyRef.current
        ? editToSrc(video.currentTime - offsetRef.current)
        : video.currentTime);
    };
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);
    const handleLoadedData = () => {
      setVideoReady(true);
      // Start the live cut at the first kept moment. The proxy needs no such
      // nudge — every frame in it is kept material already.
      if (modeRef.current === 'cut' && !proxyRef.current && intervalsRef.current.length) {
        const first = intervalsRef.current[0][0];
        if (video.currentTime < first) video.currentTime = first;
      }
    };
    const handleError = () => setVideoError('Failed to stream video');

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('ended', handleEnded);
    video.addEventListener('loadeddata', handleLoadedData);
    video.addEventListener('error', handleError);

    // Cache-bust only the rendered file: every re-render overwrites the same
    // path, so a stable URL would keep the stale render on screen. The source
    // clip never changes, so it needs no token.
    const bust = useProxyVideo
      ? `&v=${encodeURIComponent(String(proxy.mtime ?? 0))}`
      : (mode === 'result' && renderedOutput && renderVersion != null)
        ? `&v=${encodeURIComponent(String(renderVersion))}` : '';
    video.src = `${API_BASE}/api/media/stream?path=${encodeURIComponent(activeSrc)}${bust}`;
    video.load();

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('loadeddata', handleLoadedData);
      video.removeEventListener('error', handleError);
    };
  }, [activeSrc, mode, renderedOutput, renderVersion, useProxyVideo, proxy.mtime,
      editToSrc, setCurrentTime, setIsPlaying]);

function updateVideoTransform(video: HTMLVideoElement | null, t: number, timeline: any, mode: PreviewMode) {
  if (!video) return;
  if (mode !== 'cut' || !timeline?.items) {
    if (video.style.transform !== '') video.style.transform = '';
    if (video.style.opacity !== '') video.style.opacity = '';
    return;
  }
  const fps = (timeline.fps_num ?? 30) / (timeline.fps_den ?? 1) || 30;
  const items: any[] = timeline.items;
  let activeTransform: any = null;
  let segStart = 0;
  let segEnd = 0;

  for (const item of items) {
    if (item.track === 'V1' && item.enabled !== false && item.kind === 'media') {
      const s = item.source_start_frame / fps;
      const e = item.source_end_frame / fps;
      if (t >= s && t <= e + 0.05) {
        activeTransform = item.transform;
        segStart = s;
        segEnd = e;
        break;
      }
    }
  }

  if (!activeTransform) {
    if (video.style.transform !== '') video.style.transform = '';
    if (video.style.opacity !== '') video.style.opacity = '';
    return;
  }

  const duration = Math.max(0.001, segEnd - segStart);
  const p = Math.max(0, Math.min(1, (t - segStart) / duration));
  const easeP = p * p * (3 - 2 * p);
  const scaleStart = activeTransform.scale ?? 1.0;
  const scaleEnd = activeTransform.scale_end ?? scaleStart;
  const scale = scaleStart + (scaleEnd - scaleStart) * easeP;
  const posXStart = activeTransform.pos_x ?? 0.0;
  const posXEnd = activeTransform.pos_x_end ?? posXStart;
  const posX = posXStart + (posXEnd - posXStart) * easeP;
  const posYStart = activeTransform.pos_y ?? 0.0;
  const posYEnd = activeTransform.pos_y_end ?? posYStart;
  const posY = posYStart + (posYEnd - posYStart) * easeP;

  const flipH = activeTransform.flip_h ? -1 : 1;
  const flipV = activeTransform.flip_v ? -1 : 1;
  const rotate = activeTransform.rotation ?? 0;
  const opacity = activeTransform.opacity ?? 1;

  video.style.transform = `scale(${scale * flipH}, ${scale * flipV}) translate(${posX * 10}%, ${posY * 10}%) rotate(${rotate}deg)`;
  video.style.opacity = `${opacity}`;
  video.style.transformOrigin = 'center center';
}

  // Frame-accurate gap skipping for the live cut. This used to hang off the
  // `timeupdate` event, which browsers fire only ~4 times a second — so up to a
  // quarter second of every removed fumble played before the jump, and a short
  // cut could play through entirely. The strikes looked "not applied" even
  // though the EDL was right. An rAF loop checks every screen frame (~16ms).
  useEffect(() => {
    const video = videoRef.current;
    if (!video || mode !== 'cut' || keptIntervals.length === 0) return;
    let raf = 0;
    const tick = () => {
      if (!video.paused && !video.seeking && !scrubRef.current) {
        const iv = intervalsRef.current;
        const t = video.currentTime;
        // First kept interval that hasn't fully played yet.
        const idx = iv.findIndex(([, e]) => t < e - 0.005);
        if (idx === -1) {
          // Past the last kept range — the cut is over.
          video.pause();
          video.currentTime = iv[0][0];
          setCurrentTime(iv[0][0]);
        } else if (t < iv[idx][0] - 0.005) {
          // Inside a removed gap — jump to the next kept range.
          video.currentTime = iv[idx][0];
        }
      }
      updateVideoTransform(video, video.currentTime, timeline, modeRef.current);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mode, keptIntervals, setCurrentTime, timeline]);

  useEffect(() => {
    const target = useProxyVideo ? programmeTime : currentTime;
    if (videoRef.current && Math.abs(videoRef.current.currentTime - target) > 0.3) {
      videoRef.current.currentTime = target;
    }
    // The proxy already has every transform baked in; re-applying the CSS
    // approximation on top would double every zoom.
    updateVideoTransform(videoRef.current, currentTime, timeline,
                         useProxyVideo ? 'result' : mode);
  }, [currentTime, programmeTime, useProxyVideo, timeline, mode]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying) video.pause();
    else video.play().catch(() => {});
  }, [isPlaying]);

  const handleSeek = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    // FX-still mode has no video element at all, so the bar drives the store
    // directly and the frame follows the playhead.
    if (modeRef.current === 'fx' && !proxyRef.current) {
      setCurrentTime(editToSrc(x * cutDuration));
      return;
    }
    const video = videoRef.current;
    if (!video || !video.duration) return;
    // In cut mode the bar spans the EDIT, so a click lands on the kept material
    // it points at — never inside a removed region.
    const newTime = (modeRef.current === 'cut' && intervalsRef.current.length)
      ? editToSrc(x * cutDuration)
      : x * video.duration;
    video.currentTime = newTime;
    setCurrentTime(proxyRef.current
      ? editToSrc(newTime - offsetRef.current) : newTime);
  }, [setCurrentTime, editToSrc, cutDuration]);

  const toggleMute = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !isMuted;
    setIsMuted(!isMuted);
  }, [isMuted]);

  const handleVolumeChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;
    const vol = parseFloat(e.target.value);
    video.volume = vol;
    setVolume(vol);
    if (vol === 0) setIsMuted(true);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // The cut preview reports edit time: total = kept material only, and the
  // playhead position counts only kept seconds behind it.
  const isCutMode = mode === 'cut' && hasCut;
  const isFx = mode === 'fx';
  const duration = isFx ? (programmeDuration || 60)
    : isCutMode ? cutDuration : (videoRef.current?.duration || 60);
  const displayTime = isFx ? programmeTime
    : isCutMode ? srcToEdit(currentTime) : currentTime;

  const MODES: Array<{ id: PreviewMode; label: string; show: boolean; title: string }> = [
    { id: 'cut', label: 'Cut', show: hasCut,
      title: 'Live preview of the current edit — plays the kept words, skips the struck ones (always matches the transcript)' },
    { id: 'result', label: 'Rendered', show: !!renderedOutput,
      title: 'The last rendered file (B-roll, zoom, captions baked in — may be behind recent edits)' },
    { id: 'source', label: 'Source', show: !!sourceVideo,
      title: 'The untouched original clip' },
    { id: 'fx', label: '✦ FX', show: hasCut,
      title: 'The dressed programme — grade, effects, captions, B-roll and bars, '
        + 'composed by the renderer itself. Exact, and always current.' },
  ];

  return (
    <div className="preview-player">
      <div className="video-container" ref={containerRef} style={{ position: 'relative' }}>
        {(hasCut || renderedOutput) && (
          <div style={{ position: 'absolute', top: 8, left: 8, zIndex: 5, display: 'flex', gap: 4 }}>
            {MODES.filter((m) => m.show).map((m) => (
              <button
                key={m.id}
                className={`btn btn-sm ${mode === m.id ? 'btn-primary' : ''}`}
                onClick={() => setMode(m.id)}
                title={m.title}
              >{m.label}</button>
            ))}
          </div>
        )}
        {mode === 'fx' && (
          <div className="fx-stage-bar">
            {useProxyVideo ? (
              <>
                <span className={`fx-pill ${proxy.stale ? 'warn' : 'ok'}`}>
                  {proxy.stale ? '✦ proxy · behind the edit' : '✦ proxy · current'}
                </span>
                <button className="btn btn-xs" onClick={startProxy}
                  title="Re-encode the dressed programme from the timeline as it stands now">
                  Rebuild
                </button>
              </>
            ) : proxy.state === 'building' ? (
              <span className="fx-pill">
                <span className="spinner" style={{ width: 10, height: 10 }} />
                Building FX preview… {proxy.elapsed ? `${Math.round(proxy.elapsed)}s` : ''}
              </span>
            ) : (
              <>
                <span className={`fx-pill ${fxLoading ? '' : 'ok'}`}>
                  {fxLoading ? '✦ composing…' : '✦ exact frame'}
                </span>
                <button className="btn btn-xs" onClick={startProxy}
                  title="Encode the whole dressed programme small, so it plays and scrubs live with the effects moving. Roughly 2.5× realtime to build.">
                  Build moving preview
                </button>
              </>
            )}
            {proxy.state === 'error' && (
              <span className="fx-pill warn" title={proxy.detail}>build failed</span>
            )}
            <button className={`btn btn-xs ${layoutMode ? 'btn-primary' : ''}`}
              onClick={() => setLayoutMode((v) => !v)}
              title="Move and resize the clips on the canvas by dragging them, so several can share the frame">
              ⬚ Layout
            </button>
          </div>
        )}
        {mode === 'fx' && layoutMode && (
          <StageLayout programmeTime={programmeTime} pictureRect={pictureRect} />
        )}
        {mode === 'fx' && !useProxyVideo ? (
          <div className="fx-stage" ref={stageRef}>
            {fxFrame && !fxEmpty && (
              <img className="video-element" src={fxFrame} alt="Composed preview frame"
                ref={(el) => { pictureRef.current = el; }} />
            )}
            {fxEmpty && (
              <div className="video-placeholder">
                <span>Nothing on V1 at this point in the programme</span>
              </div>
            )}
            {!fxFrame && !fxEmpty && (
              <div className="video-loading">
                <div className="spinner" />
                <span>Composing the dressed frame…</span>
              </div>
            )}
          </div>
        ) : activeSrc ? (
          <>
            <video className="video-element" onClick={togglePlay}
              ref={(el) => { videoRef.current = el; pictureRef.current = el; }} />
            {!videoReady && !videoError && (
              <div className="video-loading">
                <div className="spinner" />
                <span>Loading video stream...</span>
              </div>
            )}
            {videoError && (
              <div className="video-error">
                <span>{videoError}</span>
              </div>
            )}
            {proxyBuilding && mode !== 'result' && (
              <div style={{
                position: 'absolute', bottom: 8, left: 8, zIndex: 5,
                display: 'flex', alignItems: 'center', gap: 6,
                background: 'rgba(0,0,0,0.65)', color: '#fff',
                padding: '4px 8px', borderRadius: 4, fontSize: 11,
              }}>
                <div className="spinner" style={{ width: 12, height: 12 }} />
                <span>Preparing preview — cuts will skip cleanly once ready</span>
              </div>
            )}
          </>
        ) : (
          <div className="video-placeholder">
            <span>No video loaded</span>
          </div>
        )}
      </div>

      <div className="player-controls">
        <div className="seek-bar" onClick={handleSeek}>
          <div className="seek-progress" style={{ width: `${Math.min(100, (displayTime / (duration || 1)) * 100)}%` }} />
        </div>

        <div className="controls-row">
          <button className="btn btn-icon" onClick={togglePlay}>
            {isPlaying ? '⏸' : '▶'}
          </button>

          <span className="time-display font-mono text-xs">
            {formatTime(displayTime)} / {formatTime(duration)}
          </span>

          <div className="volume-control">
            <button className="btn btn-icon" onClick={toggleMute}>
              {isMuted ? '🔇' : volume < 0.5 ? '🔉' : '🔊'}
            </button>
            <input
              type="range" min="0" max="1" step="0.01"
              value={volume} onChange={handleVolumeChange} className="volume-slider"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
