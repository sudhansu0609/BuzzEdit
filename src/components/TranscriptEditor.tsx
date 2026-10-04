import { useCallback, useRef, useEffect, useState, useMemo } from 'react';
import { useProjectStore } from '../hooks/store';
import type { TranscriptSegment } from '../hooks/store';
import { toggleWordApi, replaceWords, getProject } from '../hooks/api';
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
  const searchInputRef = useRef<HTMLInputElement>(null);
  const { project, currentTime, setCurrentTime, updateProject, setProject } = useProjectStore();
  const [viewMode, setViewMode] = useState<ViewMode>('hinglish');
  // The auto-edit decides word by word, so reviewing it needs a word-by-word
  // view. The segment view stays for reading and correcting the text itself.
  const [wordView, setWordView] = useState(true);
  const [busyWord, setBusyWord] = useState<string | null>(null);

  // --- Search + replace (F3) ---
  const [searchTerm, setSearchTerm] = useState('');
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);
  const [showReplace, setShowReplace] = useState(false);
  const [replaceTerm, setReplaceTerm] = useState('');
  const [busyReplace, setBusyReplace] = useState(false);
  const [replacedMsg, setReplacedMsg] = useState<string | null>(null);

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

  // Matches are always found against the word list — it's the one place with
  // real per-word timing, so it's what a jump (in either view) lands on.
  const matches = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    if (!term) return [] as TimelineWord[];
    // Either script counts — the native view shows `word_native`.
    return words.filter(w => w.text.toLowerCase().includes(term)
      || ((w as any).word_native || '').toLowerCase().includes(term));
  }, [words, searchTerm]);
  const matchIds = useMemo(() => new Set(matches.map(m => m.id)), [matches]);
  const currentMatchId = matches[currentMatchIndex]?.id ?? null;

  useEffect(() => {
    setCurrentMatchIndex(0);
  }, [searchTerm]);

  useEffect(() => {
    if (currentMatchIndex >= matches.length && matches.length > 0) {
      setCurrentMatchIndex(matches.length - 1);
    }
  }, [matches, currentMatchIndex]);

  // Which segment a match falls in, so segment view can highlight it too —
  // the textarea it renders into can't host inline <mark>s, so the whole
  // segment is what lights up.
  const matchSegmentIds = useMemo(() => {
    const set = new Set<number>();
    matches.forEach(w => {
      const t = wordTime(w.start_frame);
      const seg = segments.find(s => t >= s.start && t <= s.end);
      if (seg) set.add(seg.id);
    });
    return set;
  }, [matches, segments, wordTime]);

  const currentMatchSegmentId = useMemo(() => {
    const m = matches[currentMatchIndex];
    if (!m) return null;
    const t = wordTime(m.start_frame);
    return segments.find(s => t >= s.start && t <= s.end)?.id ?? null;
  }, [matches, currentMatchIndex, segments, wordTime]);

  const showWords = wordView && words.length > 0;

  const jumpToMatch = useCallback((index: number) => {
    if (matches.length === 0) return;
    const wrapped = ((index % matches.length) + matches.length) % matches.length;
    setCurrentMatchIndex(wrapped);
    const match = matches[wrapped];
    setCurrentTime(wordTime(match.start_frame));
    requestAnimationFrame(() => {
      if (!containerRef.current) return;
      const el = showWords
        ? containerRef.current.querySelector(`[data-word-id="${match.id}"]`)
        : (() => {
            const t = wordTime(match.start_frame);
            const seg = segments.find(s => t >= s.start && t <= s.end);
            return seg ? containerRef.current!.querySelector(`[data-segment-id="${seg.id}"]`) : null;
          })();
      el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  }, [matches, setCurrentTime, wordTime, segments, showWords]);

  const handleNextMatch = useCallback(() => jumpToMatch(currentMatchIndex + 1), [jumpToMatch, currentMatchIndex]);
  const handlePrevMatch = useCallback(() => jumpToMatch(currentMatchIndex - 1), [jumpToMatch, currentMatchIndex]);

  const handleSearchKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) handlePrevMatch(); else handleNextMatch();
    }
  }, [handleNextMatch, handlePrevMatch]);

  // Ctrl+F focuses the search box for as long as this panel is mounted,
  // without touching the global command dispatcher (no command owns that chord).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'f') {
        e.preventDefault();
        searchInputRef.current?.focus();
        searchInputRef.current?.select();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const refreshTranscriptFromServer = useCallback(async () => {
    if (!project?.id) return;
    try {
      const fresh = await getProject(project.id);
      if (fresh) setProject(fresh);
    } catch (err) {
      console.error('Could not refresh the transcript:', err);
    }
  }, [project?.id, setProject]);

  const handleReplaceOne = useCallback(async () => {
    const match = matches[currentMatchIndex];
    if (!project?.id || !match || !searchTerm.trim() || busyReplace) return;
    setBusyReplace(true);
    try {
      const res: any = await replaceWords(project.id, searchTerm.trim(), replaceTerm, { wordIds: [match.id], wholeWord: false });
      if (res?.timeline) updateProject({ timeline: res.timeline } as any);
      await refreshTranscriptFromServer();
      setReplacedMsg(`Replaced ${res?.replaced ?? 1} word${(res?.replaced ?? 1) === 1 ? '' : 's'}`);
    } catch (err) {
      console.error('Could not replace that word:', err);
    } finally {
      setBusyReplace(false);
      setTimeout(() => setReplacedMsg(null), 2500);
    }
  }, [project?.id, matches, currentMatchIndex, searchTerm, replaceTerm, busyReplace, updateProject, refreshTranscriptFromServer]);

  const handleReplaceAll = useCallback(async () => {
    if (!project?.id || !searchTerm.trim() || busyReplace || matches.length === 0) return;
    setBusyReplace(true);
    try {
      // Exactly the highlighted words, with the same substring match the
      // search used — what you see highlighted is what gets replaced.
      const res: any = await replaceWords(project.id, searchTerm.trim(), replaceTerm,
        { wordIds: matches.map(m => m.id), wholeWord: false });
      if (res?.timeline) updateProject({ timeline: res.timeline } as any);
      await refreshTranscriptFromServer();
      setReplacedMsg(`Replaced ${res?.replaced ?? 0} word${(res?.replaced ?? 0) === 1 ? '' : 's'}`);
    } catch (err) {
      console.error('Could not replace those words:', err);
    } finally {
      setBusyReplace(false);
      setTimeout(() => setReplacedMsg(null), 2500);
    }
  }, [project?.id, matches, searchTerm, replaceTerm, busyReplace, updateProject, refreshTranscriptFromServer]);

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

      <div className="transcript-search">
        <input
          ref={searchInputRef}
          type="text"
          className="transcript-search-input"
          placeholder="Search transcript… (Ctrl+F)"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onKeyDown={handleSearchKeyDown}
        />
        <span className="text-xs text-muted transcript-search-count">
          {searchTerm.trim() ? (matches.length > 0 ? `${currentMatchIndex + 1} of ${matches.length}` : '0 of 0') : ''}
        </span>
        <button className="btn btn-sm" disabled={matches.length === 0} onClick={handlePrevMatch} title="Previous match (Shift+Enter)">▲</button>
        <button className="btn btn-sm" disabled={matches.length === 0} onClick={handleNextMatch} title="Next match (Enter)">▼</button>
        <button className={`btn btn-sm ${showReplace ? 'btn-primary' : ''}`} onClick={() => setShowReplace(v => !v)} title="Replace">🔁 Replace</button>
        {replacedMsg && <span className="text-xs text-muted transcript-replaced-msg">{replacedMsg}</span>}
        {showReplace && (
          <>
            <input
              type="text"
              className="transcript-search-input"
              placeholder="Replace with…"
              value={replaceTerm}
              onChange={(e) => setReplaceTerm(e.target.value)}
            />
            <button className="btn btn-sm" disabled={busyReplace || !searchTerm.trim() || !currentMatchId} onClick={handleReplaceOne} title="Replace the current match only">Replace</button>
            <button className="btn btn-sm" disabled={busyReplace || !searchTerm.trim()} onClick={handleReplaceAll} title="Replace every match in the transcript">Replace all</button>
          </>
        )}
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
                data-word-id={word.id}
                className={[
                  'transcript-word',
                  word.enabled ? 'is-kept' : 'is-cut',
                  word.candidate ? 'is-uncertain' : '',
                  isActive ? 'is-playing' : '',
                  busyWord === word.id ? 'is-busy' : '',
                  matchIds.has(word.id) ? 'is-match' : '',
                  word.id === currentMatchId ? 'is-current-match' : '',
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
              className={[
                'transcript-segment',
                isActive ? 'active' : '',
                matchSegmentIds.has(segment.id) ? 'has-match' : '',
                segment.id === currentMatchSegmentId ? 'current-match' : '',
              ].filter(Boolean).join(' ')}
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
