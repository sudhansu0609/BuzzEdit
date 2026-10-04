import { create } from 'zustand';

export interface Clip {
  id: string;
  startTime: number;
  endTime: number;
  clipType: 'speech' | 'silence' | 'fumble' | 'broll' | 'intro' | 'outro' | 'transition';
  label?: string;
  transcript?: string;
  confidence?: number;
}

export interface DetectedSegment {
  start: number;
  end: number;
  segmentType: string;
  confidence: number;
  label?: string;
}

export interface TranscriptSegment {
  id: number;
  start: number;
  end: number;
  text: string;            // primary display text (Hinglish for Indic speech, else native)
  text_native?: string;    // original script (e.g. Devanagari)
  text_english?: string;   // English translation
  confidence: number;
}

export interface Project {
  id: string;
  name: string;
  sourceVideo: string;
  source_video?: string;
  timeline?: any;
  clips: Clip[];
  transcript?: TranscriptSegment[];
  detectedSegments: DetectedSegment[];
  outputPath?: string;
  status: 'draft' | 'transcribed' | 'analyzed' | 'rendered' | 'error';
  settings: Record<string, any>;
}

export interface PipelineState {
  project: Project | null;
  isProcessing: boolean;
  progress: number;
  progressMessage: string;
  error: string | null;
  selectedClipId: string | null;
  /** Full selection, primary clip first. Compounding needs more than one clip. */
  selectedClipIds: string[];
  currentTime: number;
  isPlaying: boolean;
  isScrubbing: boolean;
  zoom: number;
  showTranscript: boolean;
  showSegments: boolean;

  setProject: (project: Project | null) => void;
  updateProject: (updates: Partial<Project>) => void;
  setProcessing: (v: boolean) => void;
  setProgress: (p: number, label?: string) => void;
  setError: (e: string | null) => void;
  setSelectedClip: (id: string | null) => void;
  toggleSelectedClip: (id: string) => void;
  /** Replace the whole selection; `primary` (default: the last id) drives the inspector. */
  setSelection: (ids: string[], primary?: string | null) => void;
  setCurrentTime: (t: number) => void;
  setIsPlaying: (v: boolean) => void;
  setScrubbing: (v: boolean) => void;
  setZoom: (z: number) => void;
  setShowTranscript: (v: boolean) => void;
  setShowSegments: (v: boolean) => void;
  reset: () => void;
}

export const useProjectStore = create<PipelineState>((set) => ({
  project: null,
  isProcessing: false,
  progress: 0,
    progressMessage: 'Idle',
  error: null,
  selectedClipId: null,
  selectedClipIds: [],
  currentTime: 0,
  isPlaying: false,
  isScrubbing: false,
  zoom: 1,
  showTranscript: true,
  showSegments: false,

  setProject: (project) => set({ project, error: null }),
  updateProject: (updates) =>
    set((s) => ({ project: s.project ? { ...s.project, ...updates } : null })),
  setProcessing: (v) => set({ isProcessing: v }),
  setProgress: (p, label) => set({ progress: p, progressMessage: label || 'Processing...' }),
  setError: (e) => set({ error: e }),
  setSelectedClip: (id) => set({ selectedClipId: id, selectedClipIds: id ? [id] : [] }),
  toggleSelectedClip: (id) =>
    set((s) => {
      const next = s.selectedClipIds.includes(id)
        ? s.selectedClipIds.filter((x) => x !== id)
        : [...s.selectedClipIds, id];
      // The primary selection drives the inspector, so keep it on a clip that
      // is still selected.
      return { selectedClipIds: next, selectedClipId: next[next.length - 1] ?? null };
    }),
  setSelection: (ids, primary) =>
    set(() => {
      const unique = Array.from(new Set(ids));
      const lead = primary && unique.includes(primary) ? primary : unique[unique.length - 1] ?? null;
      return { selectedClipIds: unique, selectedClipId: lead };
    }),
  setCurrentTime: (t) => set({ currentTime: t }),
  setIsPlaying: (v) => set({ isPlaying: v }),
  setScrubbing: (v) => set({ isScrubbing: v }),
  setZoom: (z) => set({ zoom: z }),
  setShowTranscript: (v) => set({ showTranscript: v }),
  setShowSegments: (v) => set({ showSegments: v }),
  reset: () => set({
    project: null,
    isProcessing: false,
    progress: 0,
  progressMessage: 'Idle',
    error: null,
    selectedClipId: null,
    selectedClipIds: [],
    currentTime: 0,
    isPlaying: false,
    zoom: 1,
  }),
}));
