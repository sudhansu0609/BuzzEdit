export interface WordItem {
  id: string;
  text: string;
  start_frame: number;
  end_frame: number;
  enabled: boolean;
  disfluency: boolean;
  speaker?: string;
}

export interface TimelineItem {
  id: string;
  track: 'V1' | 'A1' | 'V2' | 'CAP';
  source_id: string;
  source_start_frame: number;
  source_end_frame: number;
  timeline_start_frame: number;
  timeline_end_frame: number;
  enabled: boolean;
  anchor_word_id?: string;
}

export interface Timeline {
  version: string;
  fps_num: number;
  fps_den: number;
  duration_frames: number;
  sources: Record<string, any>;
  words: WordItem[];
  items: TimelineItem[];
  revision: number;
}

export function frameToSeconds(frame: number, fpsNum = 30, fpsDen = 1): number {
  const fps = fpsNum / fpsDen;
  return frame / fps;
}

export function secondsToFrame(seconds: number, fpsNum = 30, fpsDen = 1): number {
  const fps = fpsNum / fpsDen;
  return Math.max(0, Math.round(seconds * fps));
}
