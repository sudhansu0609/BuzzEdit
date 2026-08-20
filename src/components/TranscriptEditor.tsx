import { useCallback, useRef, useEffect, useState, useMemo } from 'react';
import { useProjectStore } from '../hooks/store';
import type { TranscriptSegment } from '../hooks/store';
import { toggleWordApi } from '../hooks/api';
import { reasonTooltip } from '../lib/editReasons';

type ViewMode = 'hinglish' | 'native' | 'english';

/** One word of the EDL, as the backend stores it. */
interface TimelineWord {
  id: string;
  text: string;
  start_frame: number;
  end_frame: number;
  enabled: boolean;
  disfluency: boolean;
  reason?: string | null;
  candidate?: boolean;
}

// The backend may store `transcript` either as a bare segment array or as a
// TranscriptionResult ({ segments, language, duration }). Normalize to an array.
function getSegments(transcript: unknown): TranscriptSegment[] {
  if (Array.isArray(transcript)) return transcript as TranscriptSegment[];
  if (transcript && typeof transcript === 'object' && Array.isArray((transcript as any).segments)) {
    return (transcript as any).segments as TranscriptSegment[];
  }
  return [];
}

// Field shown/edited for a given view. Hinglish/primary lives on `text`.
const FIELD_BY_MODE: Record<ViewMode, keyof TranscriptSegment> = {
  hinglish: 'text',
  native: 'text_native',
  english: 'text_english',
};

export default function TranscriptEditor() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { project, currentTime, setCurrentTime, updateProject } = useProjectStore();
  const [viewMode, setViewMode] = useState<ViewMode>('hinglish');
  // The auto-edit decides word by word, so reviewing it needs a word-by-word
  // view. The segment view stays for reading and correcting the text itself.
  const [wordView, setWordView] = useState(true);
  const [busyWord, setBusyWord] = useState<string | null>(null);

  const segments = useMemo(() => getSegments(project?.transcript), [project?.transcript]);

  const timeline = (project as any)?.timeline;
  const words: TimelineWord[] = useMemo(
    () => (Array.isArray(timeline?.words) ? timeline.words : []), [timeline]);
  const fps = (timeline?.fps_num ?? 30) / (timeline?.fps_den ?? 1);
  const wordTime = useCallback((frame: number) => frame / (fps || 30), [fps]);

  // Putting a word back, or taking one out, is a real edit: the backend rebuilds
  // V1/A1 around the new decision and hands back the whole timeline, which is
  // what the timeline panel and preview then read.
  const handleToggleWord = useCallback(async (word: TimelineWord) => {
    if (!project?.id || busyWord) return;
    setBusyWord(word.id);
    try {
      const res: any = await toggleWordApi(project.id, word.id, !word.enabled);
      if (res?.timeline) updateProject({ timeline: res.timeline } as any);
    } catch (err) {
      console.error('Could not change that word:', err);
    } finally {
      setBusyWord(null);
    }
  }, [project?.id, busyWord, updateProject]);

  const cutCount = words.filter(w => !w.enabled).length;

  // Only offer views that actually have content in this transcript.
  const availableModes = useMemo(() => {
    const modes: ViewMode[] = ['hinglish'];
    if (segments.some(s => s.text_native && s.text_native !== s.text)) modes.push('native');
    if (segments.some(s => s.text_english && s.text_english !== s.text)) modes.push('english');
    return modes;
  }, [segments]);

  const effectiveMode = availableModes.includes(viewMode) ? viewMode : 'hinglish';
  const getText = (seg: TranscriptSegment) =>
    (seg[FIELD_BY_MODE[effectiveMode]] as string | undefined) ?? seg.text;

  useEffect(() => {
    if (!containerRef.current || segments.length === 0) return;
    const activeSegment = segments.find(
      seg => currentTime >= seg.start && currentTime <= seg.end
    );
    if (activeSegment) {
      const element = containerRef.current.querySelector(`[data-segment-id="${activeSegment.id}"]`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [currentTime, segments]);

  const handleSegmentClick = useCallback((start: number, end: number) => {
    setCurrentTime(start);
  }, [setCurrentTime]);

  // Write a new segment array back to the store, preserving whether the source
  // transcript was a bare array or a { segments, ... } object.
  const writeSegments = useCallback((next: TranscriptSegment[]) => {
    const t = project?.transcript as any;
    if (t && !Array.isArray(t) && Array.isArray(t.segments)) {
      updateProject({ transcript: { ...t, segments: next } as any });
    } else {
      updateProject({ transcript: next });
    }
  }, [project?.transcript, updateProject]);

  const handleTextChange = useCallback((id: number, newText: string) => {
    const field = FIELD_BY_MODE[effectiveMode];
    const updated = segments.map(seg =>
      seg.id === id ? { ...seg, [field]: newText } : seg
    );
    writeSegments(updated);
  }, [segments, effectiveMode, writeSegments]);

  const handleDeleteSegment = useCallback((id: number) => {
    writeSegments(segments.filter(seg => seg.id !== id));
  }, [segments, writeSegments]);

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

  if (segments.length === 0 && words.length === 0) {
    return (
      <div className="transcript-empty">
        <span>No transcript available</span>
        <span className="text-muted text-sm">Run transcription to generate transcript</span>
      </div>
    );
  }

  const showWords = wordView && words.length > 0;

  const MODE_LABELS: Record<ViewMode, string> = {
    hinglish: 'Hinglish',
    native: 'Native',
    english: 'English',
  };

  return (
    <div className="transcript-editor" ref={containerRef}>
      <div className="transcript-header">
        <span className="text-sm font-weight-semibold">
          {showWords ? `${cutCount} of ${words.length} words cut` : `${segments.length} segments`}
        </span>
        {words.length > 0 && (
          <div style={{ display: 'flex', gap: 4 }}>
            <button
              className={`btn btn-sm ${showWords ? 'btn-primary' : ''}`}
              onClick={() => setWordView(true)}
              title="Review and overrule the auto-edit word by word"
            >
              Cuts
            </button>
            <button
              className={`btn btn-sm ${!showWords ? 'btn-primary' : ''}`}
              onClick={() => setWordView(false)}
              title="Read and correct the transcript text"
            >
              Text
            </button>
          </div>
        )}
        {!showWords && availableModes.length > 1 && (
          <div className="transcript-lang-toggle" style={{ display: 'flex', gap: 4 }}>
            {availableModes.map(mode => (
              <button
                key={mode}
                className={`btn btn-sm ${effectiveMode === mode ? 'btn-primary' : ''}`}
                onClick={() => setViewMode(mode)}
              >
                {MODE_LABELS[mode]}
              </button>
            ))}
          </div>
        )}
        <span className="text-xs text-muted">
          {showWords ? 'Click a word to cut or restore it · double-click to jump there'
                     : 'Click to jump, edit to modify'}
        </span>
      </div>

      {showWords && (
        <div className="transcript-words">
          {words.map((word) => {
            const start = wordTime(word.start_frame);
            const isActive = currentTime >= start && currentTime < wordTime(word.end_frame);
            const title = word.enabled
              ? `${formatTime(start)} — click to cut this word`
              : `${formatTime(start)} — ${reasonTooltip(word.reason)}. Click to put it back.`;
            return (
              <span
                key={word.id}
                className={[
                  'transcript-word',
                  word.enabled ? 'is-kept' : 'is-cut',
                  word.candidate ? 'is-uncertain' : '',
                  isActive ? 'is-playing' : '',
                  busyWord === word.id ? 'is-busy' : '',
                ].filter(Boolean).join(' ')}
                title={title}
                onClick={() => handleToggleWord(word)}
                onDoubleClick={(e) => { e.stopPropagation(); setCurrentTime(start); }}
              >
                {word.text}
              </span>
            );
          })}
        </div>
      )}

      {!showWords && (
      <div className="transcript-list">
        {segments.map((segment) => {
          const isActive = currentTime >= segment.start && currentTime <= segment.end;
          const displayText = getText(segment);
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
                  value={displayText}
                  onChange={(e) => handleTextChange(segment.id, e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                  rows={Math.max(1, Math.ceil(displayText.length / 60))}
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
      )}
    </div>
  );
}
