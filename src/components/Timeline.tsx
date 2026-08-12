import { useRef, useEffect, useState, useCallback } from 'react';
import { useProjectStore } from '../hooks/store';

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 5;

export default function Timeline() {
  const containerRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const { project, currentTime, setCurrentTime, selectedClipId, setSelectedClip, zoom, setZoom, showSegments } = useProjectStore();
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(0);
  const [scrollStart, setScrollStart] = useState(0);

  const duration = project?.transcript?.length ? Math.max(...project.transcript.map(s => s.end)) : 60;
  const pixelsPerSecond = 50 * zoom;
  const trackWidth = duration * pixelsPerSecond;

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollLeft = currentTime * pixelsPerSecond - containerRef.current.clientWidth / 2;
    }
  }, [currentTime, pixelsPerSecond]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom(Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom + delta)));
  }, [zoom, setZoom]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
      setIsDragging(true);
      setDragStart(e.clientX);
      setScrollStart(containerRef.current?.scrollLeft || 0);
    }
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging && containerRef.current) {
      const dx = e.clientX - dragStart;
      containerRef.current.scrollLeft = scrollStart - dx;
    }
  }, [isDragging, dragStart, scrollStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleTrackClick = useCallback((e: React.MouseEvent) => {
    if (isDragging) return;
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left + (containerRef.current?.scrollLeft || 0);
    const time = Math.max(0, Math.min(duration, x / pixelsPerSecond));
    setCurrentTime(time);
  }, [isDragging, duration, pixelsPerSecond, setCurrentTime]);

  const handleSegmentClick = useCallback((segmentStart: number, segmentEnd: number) => {
    setCurrentTime(segmentStart);
  }, [setCurrentTime]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const frames = Math.floor((seconds % 1) * 30);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`;
  };

  const getTimeMarkers = () => {
    const markers = [];
    const interval = zoom < 0.5 ? 30 : zoom < 1 ? 10 : zoom < 2 ? 5 : 1;
    for (let t = 0; t <= duration; t += interval) {
      markers.push(t);
    }
    return markers;
  };

  if (!project) return null;

  return (
    <div className="timeline-container">
      <div className="timeline-header">
        <div className="timeline-timecode">{formatTime(currentTime)}</div>
        <div className="timeline-zoom">
          <button className="btn btn-sm" onClick={() => setZoom(Math.max(MIN_ZOOM, zoom - 0.2))}>-</button>
          <span className="text-xs text-muted">{Math.round(zoom * 100)}%</span>
          <button className="btn btn-sm" onClick={() => setZoom(Math.min(MAX_ZOOM, zoom + 0.2))}>+</button>
        </div>
        <div className="timeline-duration">{formatTime(duration)}</div>
      </div>

      <div
        ref={containerRef}
        className="timeline-scroll"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div
          ref={trackRef}
          className="timeline-track"
          style={{ width: `${trackWidth}px` }}
          onClick={handleTrackClick}
        >
          <div className="timeline-ruler">
            {getTimeMarkers().map(t => (
              <div
                key={t}
                className="timeline-marker"
                style={{ left: `${t * pixelsPerSecond}px` }}
              >
                <span className="marker-label">{formatTime(t)}</span>
              </div>
            ))}
          </div>

          <div className="timeline-segments">
            {showSegments && project.detectedSegments?.map((segment, i) => (
              <div
                key={i}
                className={`timeline-segment segment-${segment.segmentType}`}
                style={{
                  left: `${segment.start * pixelsPerSecond}px`,
                  width: `${(segment.end - segment.start) * pixelsPerSecond}px`,
                }}
                onClick={() => handleSegmentClick(segment.start, segment.end)}
              >
                <span className="segment-label">{segment.label}</span>
              </div>
            ))}
          </div>

          <div className="timeline-waveform">
            {Array.from({ length: Math.min(200, Math.ceil(duration * zoom)) }).map((_, i) => {
              const height = 10 + Math.random() * 40;
              return (
                <div
                  key={i}
                  className="waveform-bar"
                  style={{
                    height: `${height}%`,
                    left: `${(i / 200) * 100}%`,
                  }}
                />
              );
            })}
          </div>

          <div
            className="timeline-playhead"
            style={{ left: `${currentTime * pixelsPerSecond}px` }}
          />
        </div>
      </div>
    </div>
  );
}
