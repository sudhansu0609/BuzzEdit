import { useCallback, useRef, useEffect } from 'react';
import { useProjectStore } from '../hooks/store';

export default function TranscriptEditor() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { project, currentTime, setCurrentTime, updateProject } = useProjectStore();

  useEffect(() => {
    if (!containerRef.current || !project?.transcript) return;
    const activeSegment = project.transcript.find(
      seg => currentTime >= seg.start && currentTime <= seg.end
    );
    if (activeSegment) {
      const element = containerRef.current.querySelector(`[data-segment-id="${activeSegment.id}"]`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [currentTime, project?.transcript]);

  const handleSegmentClick = useCallback((start: number, end: number) => {
    setCurrentTime(start);
  }, [setCurrentTime]);

  const handleTextChange = useCallback((id: number, newText: string) => {
    if (!project?.transcript) return;
    const updated = project.transcript.map(seg =>
      seg.id === id ? { ...seg, text: newText } : seg
    );
    updateProject({ transcript: updated });
  }, [project?.transcript, updateProject]);

  const handleDeleteSegment = useCallback((id: number) => {
    if (!project?.transcript) return;
    const updated = project.transcript.filter(seg => seg.id !== id);
    updateProject({ transcript: updated });
  }, [project?.transcript, updateProject]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'var(--success)';
    if (confidence >= 0.7) return 'var(--warning)';
    return 'var(--danger)';
  };

  if (!project) return null;

  if (!project.transcript || project.transcript.length === 0) {
    return (
      <div className="transcript-empty">
        <span>No transcript available</span>
        <span className="text-muted text-sm">Run transcription to generate transcript</span>
      </div>
    );
  }

  return (
    <div className="transcript-editor" ref={containerRef}>
      <div className="transcript-header">
        <span className="text-sm font-weight-semibold">{project.transcript.length} segments</span>
        <span className="text-xs text-muted">Click to jump, edit to modify</span>
      </div>
      <div className="transcript-list">
        {project.transcript.map((segment) => {
          const isActive = currentTime >= segment.start && currentTime <= segment.end;
          return (
            <div
              key={segment.id}
              data-segment-id={segment.id}
              className={`transcript-segment ${isActive ? 'active' : ''}`}
              onClick={() => handleSegmentClick(segment.start, segment.end)}
            >
              <div className="segment-timecode font-mono text-xs">
                {formatTime(segment.start)}
              </div>
              <div className="segment-content">
                <textarea
                  className="segment-text"
                  value={segment.text}
                  onChange={(e) => handleTextChange(segment.id, e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                  rows={Math.max(1, Math.ceil(segment.text.length / 60))}
                />
                <div className="segment-meta">
                  <span
                    className="confidence-badge"
                    style={{ color: getConfidenceColor(segment.confidence) }}
                  >
                    {Math.round(segment.confidence * 100)}%
                  </span>
                  <button
                    className="btn btn-sm btn-danger delete-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteSegment(segment.id);
                    }}
                  >
                    ✕
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
