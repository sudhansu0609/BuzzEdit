import { useEffect, useState, useCallback } from 'react';
import { useProjectStore } from './store';

declare global {
  interface Window {
    electronAPI?: {
      /** The backend URL the shell actually took — see electron/preload.cjs. */
      apiUrl?: string | null;
      /** The ComfyUI URL the shell actually took. */
      comfyUrl?: string | null;
      dialogOpenFile: (opts: any) => Promise<any>;
      dialogSaveFile: (opts: any) => Promise<any>;
      windowMinimize: () => Promise<void>;
      windowMaximize: () => Promise<void>;
      windowClose: () => Promise<void>;
      getApiUrl: () => Promise<string>;
      getPorts?: () => Promise<{
        apiUrl: string; comfyUrl: string; backendPort: number; comfyPort: number;
      }>;
      openPath: (filePath: string) => Promise<string>;
      showItemInFolder: (filePath: string) => Promise<void>;
    };
  }
}

/**
 * Where the backend is, decided once at load.
 *
 * GUARDIAN_PLAN.md section 11: no port is written down here, because the
 * backend may well not be on 8099 — if something else held it the shell stepped
 * forward, and the number is only known at launch.
 *
 *  1. Electron hands the real URL to the preload as a command-line switch. That
 *     is the production path, and the only one that works from `file://`.
 *  2. `VITE_BUZZEDIT_API` for a browser pointed straight at a backend.
 *  3. Otherwise relative, i.e. `window.location.origin` — correct both when the
 *     bundle is served by the backend itself and under `vite dev`, whose proxy
 *     (vite.config.ts) forwards to the port it was handed at launch.
 */
export const API_BASE: string =
  (typeof window !== 'undefined' ? window.electronAPI?.apiUrl : null)
  || (import.meta.env?.VITE_BUZZEDIT_API as string | undefined)
  || '';

/** The ComfyUI URL, for the places the UI names it. Empty when unknown. */
export const COMFY_BASE: string =
  (typeof window !== 'undefined' ? window.electronAPI?.comfyUrl : null)
  || (import.meta.env?.VITE_COMFYUI_URL as string | undefined)
  || '';

/** Strips the scheme from a base URL, for display: `http://host:port` -> `host:port`. */
export function hostPort(url: string): string {
  return url.replace(/^https?:\/\//, '').replace(/\/+$/, '');
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export async function uploadVideo(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/projects/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function importVideoPath(filePath: string): Promise<any> {
  return api('/api/projects/import_path', {
    method: 'POST',
    body: JSON.stringify({ path: filePath }),
  });
}

export async function transcribeProject(projectId: string, language?: string): Promise<any> {
  return api('/api/transcription/transcribe', {
    method: 'POST',
    // Omit `language` to let the backend auto-detect the spoken language
    // (Hindi -> Hinglish, etc.) instead of forcing English.
    body: JSON.stringify({ project_id: projectId, model: 'large-v3', language: language ?? null }),
  });
}

export async function analyzeProject(projectId: string): Promise<any> {
  return api('/api/analysis/analyze', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, remove_silence: true, remove_fumbles: true }),
  });
}

export async function renderProject(projectId: string): Promise<any> {
  return api('/api/rendering/render', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  });
}

export async function autoEditProject(projectId: string): Promise<any> {
  return api(`/api/rendering/${projectId}/auto_edit`, {
    method: 'POST',
  });
}

export async function getProject(projectId: string): Promise<any> {
  return api(`/api/projects/${projectId}`);
}

export async function listProjects(): Promise<{ projects: Array<{ id: string; name: string; status: string; source_video?: string }> }> {
  return api('/api/projects/list');
}

export async function getTranscript(projectId: string): Promise<any> {
  return api(`/api/transcription/${projectId}/transcript`);
}

export async function getSegments(projectId: string): Promise<any> {
  return api(`/api/analysis/${projectId}/segments`);
}

export async function getTimeline(projectId: string): Promise<any> {
  return api(`/api/timeline/${projectId}`);
}

export async function generateTimeline(projectId: string): Promise<any> {
  return api(`/api/timeline/${projectId}/generate`, { method: 'POST' });
}

export async function toggleWordApi(projectId: string, wordId: string, enabled: boolean): Promise<any> {
  return api(`/api/timeline/${projectId}/toggle_word`, {
    method: 'POST',
    body: JSON.stringify({ word_id: wordId, enabled }),
  });
}

/** Find/replace across the transcript. Omit `wordIds` (or pass null) to match
 *  every occurrence; pass one or more word ids to scope the replacement to
 *  specific words (e.g. "replace this one match only"). */
export async function replaceWords(
  projectId: string,
  find: string,
  replace: string,
  opts?: { wordIds?: string[] | null; matchCase?: boolean; wholeWord?: boolean },
): Promise<any> {
  return api(`/api/transcription/${projectId}/replace_words`, {
    method: 'POST',
    body: JSON.stringify({
      find,
      replace,
      word_ids: opts?.wordIds ?? null,
      match_case: opts?.matchCase ?? false,
      whole_word: opts?.wholeWord ?? false,
    }),
  });
}

// --- Multi-track clip editing ---

export async function splitClip(projectId: string, itemId: string, atFrame: number): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/split`, {
    method: 'POST',
    body: JSON.stringify({ item_id: itemId, at_frame: atFrame }),
  });
}

export async function deleteClip(projectId: string, itemId: string): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/delete`, {
    method: 'POST',
    body: JSON.stringify({ item_id: itemId }),
  });
}

/** Split every listed clip the frame falls inside (linked partners included). */
export async function splitClips(projectId: string, itemIds: string[], atFrame: number): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/split_many`, {
    method: 'POST',
    body: JSON.stringify({ item_ids: itemIds, at_frame: atFrame }),
  });
}

/** Delete a selection with its linked partners; `ripple` closes the holes. */
export async function deleteClips(projectId: string, itemIds: string[], ripple = false): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/delete_many`, {
    method: 'POST',
    body: JSON.stringify({ item_ids: itemIds, ripple }),
  });
}

/** Slide a selection (and linked partners) by a frame delta, optionally re-homing some clips. */
export async function moveClips(
  projectId: string, itemIds: string[], deltaFrames: number, trackMap?: Record<string, string>,
  linked = true,
): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/move_many`, {
    method: 'POST',
    body: JSON.stringify({ item_ids: itemIds, delta_frames: deltaFrames, track_map: trackMap ?? {}, linked }),
  });
}

/** Trim the start or end of the listed clips to a frame — "trim to playhead". */
export async function trimClips(
  projectId: string, itemIds: string[], edge: 'start' | 'end', atFrame: number, ripple = false,
): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/trim_many`, {
    method: 'POST',
    body: JSON.stringify({ item_ids: itemIds, edge, at_frame: atFrame, ripple }),
  });
}

export async function linkClips(projectId: string, itemIds: string[]): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/link`, {
    method: 'POST',
    body: JSON.stringify({ item_ids: itemIds }),
  });
}

export async function unlinkClips(projectId: string, itemIds: string[]): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/unlink`, {
    method: 'POST',
    body: JSON.stringify({ item_ids: itemIds }),
  });
}

/** Paste copied clip snapshots so the earliest starts at `atFrame`. */
export async function pasteClips(
  projectId: string, items: unknown[], atFrame: number, insert = false,
): Promise<{ item_ids: string[]; timeline: any }> {
  return api(`/api/timeline/${projectId}/clip/paste`, {
    method: 'POST',
    body: JSON.stringify({ items, at_frame: atFrame, insert }),
  });
}

/** Ripple-delete the empty stretch of a track under a frame. */
export async function closeGap(projectId: string, track: string, atFrame: number): Promise<any> {
  return api(`/api/timeline/${projectId}/track/close_gap`, {
    method: 'POST',
    body: JSON.stringify({ track, at_frame: atFrame }),
  });
}

/** Put a whole earlier timeline back — the undo/redo path. */
export async function replaceTimeline(projectId: string, timeline: unknown): Promise<any> {
  return api(`/api/timeline/${projectId}/replace`, {
    method: 'POST',
    body: JSON.stringify({ timeline }),
  });
}

export async function moveClip(projectId: string, itemId: string, timelineStartFrame: number, track?: string): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/move`, {
    method: 'POST',
    body: JSON.stringify({ item_id: itemId, timeline_start_frame: timelineStartFrame, track: track ?? null }),
  });
}

export async function trimClip(
  projectId: string, itemId: string, edge: 'start' | 'end', timelineFrame: number, ripple = false,
  linked = true,
): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/trim`, {
    method: 'POST',
    body: JSON.stringify({ item_id: itemId, edge, timeline_frame: timelineFrame, ripple, linked }),
  });
}

/** Ripple-cut a programme-frame range out of V1/A1 (the auto-edited spine),
 *  shifting overlays to close the gap. */
export async function cutProgramRange(projectId: string, startFrame: number, endFrame: number): Promise<any> {
  return api(`/api/timeline/${projectId}/program/cut`, {
    method: 'POST',
    body: JSON.stringify({ start_frame: startFrame, end_frame: endFrame }),
  });
}

/** Undo every ripple cut made on the programme via `cutProgramRange`. */
export async function restoreProgramCuts(projectId: string): Promise<any> {
  return api(`/api/timeline/${projectId}/program/restore_cuts`, {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export async function detachAudio(projectId: string, itemId: string, track?: string): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/detach_audio`, {
    method: 'POST',
    body: JSON.stringify({ item_id: itemId, track: track ?? null }),
  });
}

export async function compoundClips(projectId: string, itemIds: string[], label?: string): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/compound`, {
    method: 'POST',
    body: JSON.stringify({ item_ids: itemIds, label: label ?? null }),
  });
}

export async function uncompoundClip(projectId: string, itemId: string): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/uncompound`, {
    method: 'POST',
    body: JSON.stringify({ item_id: itemId }),
  });
}

export async function setClipFlags(
  projectId: string,
  itemId: string,
  flags: {
    enabled?: boolean; locked?: boolean; mute?: boolean; volume?: number; label?: string;
    loop?: boolean; audio_fade_in?: number; audio_fade_out?: number; duck?: number;
  },
): Promise<any> {
  return api(`/api/timeline/${projectId}/clip/flags`, {
    method: 'POST',
    body: JSON.stringify({ item_id: itemId, ...flags }),
  });
}

// --- Transform / colour / text ---

export interface ClipTransform {
  crop_left: number; crop_top: number; crop_right: number; crop_bottom: number;
  scale: number; pos_x: number; pos_y: number; rotation: number; opacity: number;
  flip_h: boolean; flip_v: boolean;
  scale_end?: number | null; pos_x_end?: number | null; pos_y_end?: number | null;
}

/** One colour-balance wheel: an r/g/b push for a single tonal range. */
export interface ColorWheel { r: number; g: number; b: number }

export interface ClipColor {
  preset?: string | null;
  // correction
  exposure: number;
  brightness: number; contrast: number; gamma: number;
  temperature: number; tint: number;
  shadows: number; highlights: number;
  // grade
  lift: ColorWheel; midtones: ColorWheel; gain: ColorWheel;
  saturation: number; vibrance: number; hue: number;
  // look
  lut_file?: string | null; lut_strength: number;
  sharpen: number; denoise: number; vignette: number;
  fade_in: number; fade_out: number;
}

export interface ClipChroma {
  enabled: boolean;
  key_type: 'chroma' | 'color';
  color: string;
  similarity: number;
  blend: number;
  choke: number;
  feather: number;
  spill: number;
  spill_expand: number;
  show_matte: boolean;
}

export interface TextStyle {
  font_family: string; font_file?: string | null; font_size: number;
  color: string; opacity: number; bold: boolean; italic: boolean;
  line_spacing: number; align: 'left' | 'center' | 'right';
  pos_x: number; pos_y: number;
  stroke_width: number; stroke_color: string;
  shadow_x: number; shadow_y: number; shadow_color: string;
  box: boolean; box_color: string; box_padding: number;
  animation: 'none' | 'fade' | 'pop' | 'slide-up'; animation_duration: number;
}

/** Omit `itemId` to target the whole program instead of one clip. */
export async function setTransform(
  projectId: string,
  itemId: string | null,
  updates: Partial<ClipTransform>,
  reset = false,
): Promise<any> {
  return api(`/api/timeline/${projectId}/transform`, {
    method: 'POST',
    body: JSON.stringify({ item_id: itemId, updates, reset }),
  });
}

/** Omit `itemId` to grade the whole program instead of one clip. */
export async function setColor(
  projectId: string,
  itemId: string | null,
  updates: Partial<ClipColor>,
  preset?: string | null,
  reset = false,
): Promise<any> {
  return api(`/api/timeline/${projectId}/color`, {
    method: 'POST',
    body: JSON.stringify({ item_id: itemId, updates, preset: preset ?? null, reset }),
  });
}

export async function setChroma(
  projectId: string,
  itemId: string,
  updates: Partial<ClipChroma>,
  reset = false,
): Promise<any> {
  return api(`/api/timeline/${projectId}/chroma`, {
    method: 'POST',
    body: JSON.stringify({ item_id: itemId, updates, reset }),
  });
}

// --- Programme audio master (voice chain: EQ, de-ess, compress, reverb...) ---

/** Mirrors backend timeline/schema.py AudioMaster. */
export interface AudioMaster {
  voice_gain_db: number;
  voice_enhance: string;
  voice_denoise: number;
  voice_deess: number;
  voice_compress: number;
  voice_eq_preset: string;
  voice_saturation: number;
  voice_reverb: string;
  voice_fx: Array<Record<string, any>>;
  loudness_lufs: number | null;
  true_peak_db: number;
  origin?: string | null;
}

export async function getAudioMaster(projectId: string): Promise<any> {
  return api(`/api/timeline/${projectId}/audio_master`);
}

export async function updateAudioMaster(projectId: string, updates: Record<string, any>): Promise<any> {
  return api(`/api/timeline/${projectId}/audio_master`, {
    method: 'POST',
    body: JSON.stringify({ updates }),
  });
}

export async function addTextClip(
  projectId: string,
  opts: {
    content?: string; track?: string; timelineStartFrame?: number;
    durationSeconds?: number; preset?: string; style?: Partial<TextStyle>;
  } = {},
): Promise<any> {
  return api(`/api/timeline/${projectId}/text/add`, {
    method: 'POST',
    body: JSON.stringify({
      content: opts.content ?? 'Text',
      track: opts.track ?? null,
      timeline_start_frame: opts.timelineStartFrame ?? 0,
      duration_seconds: opts.durationSeconds ?? 3,
      preset: opts.preset ?? null,
      style: opts.style ?? {},
    }),
  });
}

export async function updateTextClip(
  projectId: string,
  itemId: string,
  opts: { content?: string; preset?: string | null; style?: Partial<TextStyle> } = {},
): Promise<any> {
  return api(`/api/timeline/${projectId}/text/update`, {
    method: 'POST',
    body: JSON.stringify({
      item_id: itemId,
      content: opts.content ?? null,
      preset: opts.preset ?? null,
      style: opts.style ?? {},
    }),
  });
}

// --- Generated captions & intros ---

export async function generateCaptions(projectId: string, preset: string, style?: Partial<TextStyle>): Promise<any> {
  return api(`/api/timeline/${projectId}/captions/generate`, {
    method: 'POST',
    body: JSON.stringify({ preset, style: style ?? {} }),
  });
}

export async function clearCaptions(projectId: string): Promise<any> {
  return api(`/api/timeline/${projectId}/captions/clear`, { method: 'POST' });
}

export async function applyIntro(projectId: string, preset: string, title: string, subtitle = ''): Promise<any> {
  return api(`/api/timeline/${projectId}/intro/apply`, {
    method: 'POST',
    body: JSON.stringify({ preset, title, subtitle }),
  });
}

export async function clearIntro(projectId: string): Promise<any> {
  return api(`/api/timeline/${projectId}/intro/clear`, { method: 'POST' });
}

// --- Fonts & presets ---

export interface FontFamily {
  family: string;
  regular: string | null;
  bold: string | null;
  italic: string | null;
  bold_italic: string | null;
  styles: string[];
}

export interface PresetEntry {
  id: string;
  label: string;
  values?: Record<string, number>;
  style?: Record<string, any>;
  duration?: number;
  background?: string | null;
  words_per_caption?: number;
  uppercase?: boolean;
}

export interface PresetCatalogue {
  color: PresetEntry[];
  text: PresetEntry[];
  caption: PresetEntry[];
  intro: PresetEntry[];
  transition: PresetEntry[];
  effect: PresetEntry[];
  aspect: PresetEntry[];
  transition_catalogue: { group: string; transitions: string[] }[];
  /** Every atmosphere effect the renderer can build, including the ones with no
   *  named preset (glitch, VHS, shake, flash, flicker). */
  effect_catalogue: { id: string; label: string; description: string }[];
}

// --- Transitions, atmosphere effects and cinematic bars ---

export interface AtmosphereEffect {
  type: string;
  intensity: number;
  speed: number;
  color?: string | null;
  enabled: boolean;
}

/** Omit `itemId` to set the transition used at every cut. */
export async function setTransition(
  projectId: string,
  itemId: string | null,
  opts: { preset?: string; updates?: Record<string, any>; reset?: boolean } = {},
): Promise<any> {
  return api(`/api/timeline/${projectId}/transition`, {
    method: 'POST',
    body: JSON.stringify({
      item_id: itemId,
      preset: opts.preset ?? null,
      updates: opts.updates ?? {},
      reset: opts.reset ?? false,
    }),
  });
}

/** `itemId` targets one adjustment layer; omit it for the whole programme. */
export async function addEffect(projectId: string, preset?: string, updates?: Record<string, any>,
                                itemId?: string | null): Promise<any> {
  return api(`/api/timeline/${projectId}/effects/add`, {
    method: 'POST',
    body: JSON.stringify({ preset: preset ?? null, updates: updates ?? {}, item_id: itemId ?? null }),
  });
}

export async function updateEffect(projectId: string, index: number, updates: Record<string, any>,
                                   itemId?: string | null): Promise<any> {
  return api(`/api/timeline/${projectId}/effects/update`, {
    method: 'POST',
    body: JSON.stringify({ index, updates, item_id: itemId ?? null }),
  });
}

export async function removeEffect(projectId: string, index: number,
                                   itemId?: string | null): Promise<any> {
  return api(`/api/timeline/${projectId}/effects/remove`, {
    method: 'POST',
    body: JSON.stringify({ index, updates: {}, item_id: itemId ?? null }),
  });
}

/** An adjustment layer: a clip with no picture that treats every layer below it. */
export async function addAdjustment(
  projectId: string,
  opts: { track?: string; timelineStartFrame?: number; durationSeconds?: number; preset?: string } = {},
): Promise<any> {
  return api(`/api/timeline/${projectId}/adjustment/add`, {
    method: 'POST',
    body: JSON.stringify({
      track: opts.track ?? null,
      timeline_start_frame: opts.timelineStartFrame ?? 0,
      duration_seconds: opts.durationSeconds ?? 5,
      preset: opts.preset ?? null,
    }),
  });
}

// --- Dressed preview (what the render will actually look like) ---

/**
 * URL of the fully composed frame at `seconds` of programme time.
 *
 * `revision` is part of the URL rather than a cache-buster bolted on: the
 * backend serves these `immutable`, so a frame the user scrubs back over is
 * fetched once for the life of that edit and comes from the browser cache
 * afterwards. Bumping the revision simply names a different resource.
 */
export function previewFrameUrl(
  projectId: string, seconds: number, width: number, revision: number | undefined,
): string {
  const t = Math.max(0, seconds).toFixed(2);
  return `${API_BASE}/api/timeline/${projectId}/preview/frame`
    + `?t=${t}&w=${Math.round(width)}&r=${revision ?? 0}`;
}

export interface ProxyStatus {
  state: 'none' | 'building' | 'ready' | 'error';
  path?: string;
  revision?: number | null;
  stale?: boolean;
  elapsed?: number;
  detail?: string;
  size_bytes?: number;
  mtime?: number;
}

export async function getPreviewProxy(projectId: string): Promise<ProxyStatus> {
  return api(`/api/timeline/${projectId}/preview/proxy`);
}

export async function buildPreviewProxy(projectId: string, height = 540): Promise<ProxyStatus> {
  return api(`/api/timeline/${projectId}/preview/proxy`, {
    method: 'POST',
    body: JSON.stringify({ height }),
  });
}

export async function setAspectBars(projectId: string, ratio: number | null): Promise<any> {
  return api(`/api/timeline/${projectId}/aspect`, {
    method: 'POST',
    body: JSON.stringify({ ratio }),
  });
}

export async function listFonts(refresh = false): Promise<{ count: number; families: FontFamily[] }> {
  return api(`/api/presets/fonts${refresh ? '?refresh=true' : ''}`);
}

export async function listPresets(): Promise<PresetCatalogue> {
  return api('/api/presets/');
}

// --- Reference-video style matching ---

export interface StyleProfile {
  id: string;
  name: string;
  source_path: string;
  duration: number;
  created_at: string;
  pacing: {
    shots: number; cuts_per_minute: number; median_shot_seconds: number;
    p25_shot_seconds: number; p75_shot_seconds: number;
  };
  rhythm: { bpm: number; beat_alignment: number; confidence: number; cuts_to_music: boolean };
  motion: {
    zoom_share: number; pan_share: number; mean_zoom_ratio: number;
    mean_pan_fraction: number; mean_move_seconds: number;
  };
  look: { description: string; stats: Record<string, any> };
  captions: {
    present: boolean; pos_y: number; size_fraction: number;
    coverage: number; boxed: boolean; confidence: number;
  };
  transitions: Record<string, { count: number; mean_duration: number; share: number }>;
  confidence: Record<string, string>;
  notes: string[];
}

export interface StyleApplyParts {
  look?: boolean;
  motion?: boolean;
  captions?: boolean;
  transitions?: boolean;
}

export async function listStyleProfiles(): Promise<{ profiles: StyleProfile[] }> {
  return api('/api/style/');
}

export async function analyzeStyle(path: string, name?: string): Promise<{ profile: StyleProfile }> {
  return api('/api/style/analyze', {
    method: 'POST',
    body: JSON.stringify({ path, name: name ?? null }),
  });
}

export async function deleteStyleProfile(profileId: string): Promise<any> {
  return api(`/api/style/${profileId}`, { method: 'DELETE' });
}

export async function applyStyleProfile(
  projectId: string,
  profileId: string,
  parts: StyleApplyParts = {},
): Promise<{ report: Record<string, any>; timeline: any }> {
  return api('/api/style/apply', {
    method: 'POST',
    body: JSON.stringify({
      project_id: projectId,
      profile_id: profileId,
      look: parts.look ?? true,
      motion: parts.motion ?? true,
      captions: parts.captions ?? true,
      transitions: parts.transitions ?? true,
    }),
  });
}

export interface WaveformData {
  peaks: number[];
  points_per_second: number;
  duration: number;
}

export async function getWaveform(projectId: string, sourceId: string): Promise<WaveformData> {
  return api(`/api/timeline/${projectId}/waveform/${sourceId}`);
}

/** Layout of a source's filmstrip: `columns` frames spread evenly over `duration`
 *  seconds. `columns: 0` means the source has no picture to show. */
export interface FilmstripData {
  columns: number;
  tile_width: number;
  tile_height: number;
  duration: number;
}

export async function getFilmstrip(projectId: string, sourceId: string): Promise<FilmstripData> {
  return api(`/api/timeline/${projectId}/filmstrip/${sourceId}`);
}

/** The strip image itself. One URL per source, so every clip cut from the same
 *  footage shares a single decoded image in the browser cache. */
export function filmstripImageUrl(projectId: string, sourceId: string): string {
  return `${API_BASE}/api/timeline/${projectId}/filmstrip/${sourceId}/image`;
}

export async function addMedia(projectId: string, path: string, opts?: { track?: string; timelineStartFrame?: number }): Promise<any> {
  return api(`/api/timeline/${projectId}/add_media`, {
    method: 'POST',
    body: JSON.stringify({
      path,
      track: opts?.track ?? null,
      timeline_start_frame: opts?.timelineStartFrame ?? 0,
    }),
  });
}

export interface TrackState {
  hidden: boolean;
  locked: boolean;
  muted: boolean;
}

export async function addTrack(projectId: string, kind: 'V' | 'A' | 'T'): Promise<any> {
  return api(`/api/timeline/${projectId}/add_track`, {
    method: 'POST',
    body: JSON.stringify({ kind }),
  });
}

/** Hide, lock or mute a whole track. Omitted flags are left as they are. */
export async function setTrackFlags(
  projectId: string,
  track: string,
  flags: { hidden?: boolean; locked?: boolean; muted?: boolean },
): Promise<any> {
  return api(`/api/timeline/${projectId}/track/flags`, {
    method: 'POST',
    body: JSON.stringify({ track, ...flags }),
  });
}

export async function deleteTrack(projectId: string, track: string): Promise<any> {
  return api(`/api/timeline/${projectId}/track/delete`, {
    method: 'POST',
    body: JSON.stringify({ track }),
  });
}

// --- Media pool ---

export interface MediaEntry {
  id: string;
  name: string;
  path: string;
  source_path?: string;
  kind: 'video' | 'audio' | 'image';
  duration: number;
  width?: number;
  height?: number;
  fps?: number;
  has_audio?: boolean;
  size_bytes?: number;
  linked?: boolean;
  missing?: boolean;
  has_thumb?: boolean;
  /** Set once the clip has been re-aimed at the lens; original_path is what revert restores. */
  eye_contact?: { original_path: string; output_path: string; report?: EyeContactReport };
}

export interface MediaListing {
  media: MediaEntry[];
  counts: Record<string, number>;
  total: number;
}

export function mediaThumbUrl(projectId: string, mediaId: string): string {
  return `${API_BASE}/api/projects/${projectId}/media/${mediaId}/thumb`;
}

/** Range-served stream of a file on disk, for hover previews and the player. */
export function mediaStreamUrl(path: string): string {
  return `${API_BASE}/api/media/stream?path=${encodeURIComponent(path)}`;
}

export async function listMedia(
  projectId: string,
  opts?: { kind?: string; q?: string },
): Promise<MediaListing> {
  const params = new URLSearchParams();
  if (opts?.kind) params.set('kind', opts.kind);
  if (opts?.q) params.set('q', opts.q);
  const qs = params.toString();
  return api(`/api/projects/${projectId}/media${qs ? `?${qs}` : ''}`);
}

/** Paths may be files or folders; folders are walked for media on the backend. */
export async function importMedia(
  projectId: string,
  paths: string[],
  copy = false,
): Promise<{ added: MediaEntry[]; skipped: string[]; media: MediaEntry[] }> {
  return api(`/api/projects/${projectId}/media/import`, {
    method: 'POST',
    body: JSON.stringify({ paths, copy }),
  });
}

export async function removeMedia(
  projectId: string,
  mediaId: string,
  deleteFile = false,
): Promise<{ media: MediaEntry[] }> {
  return api(`/api/projects/${projectId}/media/${mediaId}?delete_file=${deleteFile}`, {
    method: 'DELETE',
  });
}

export async function renameMedia(projectId: string, mediaId: string, name: string): Promise<{ media: MediaEntry[] }> {
  return api(`/api/projects/${projectId}/media/${mediaId}`, {
    method: 'PATCH',
    body: JSON.stringify({ name }),
  });
}

export async function relinkMedia(projectId: string, mediaId: string, path: string): Promise<{ media: MediaEntry[] }> {
  return api(`/api/projects/${projectId}/media/${mediaId}/relink`, {
    method: 'POST',
    body: JSON.stringify({ path }),
  });
}

// --- Eye contact (teleprompter gaze correction) ---

export interface EyeContactReport {
  frames: number;
  face_found_pct: number;
  corrected_pct: number;
  prompter_offset_deg: number;
  symmetry_offset_deg: number;
  reading_sweep_deg: number;
  median_shift_px: number;
  seconds: number;
}

export interface EyeContactJob {
  id: string;
  media_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  stage: string;
  result: EyeContactReport | null;
  error: string | null;
}

/** The speaker's prompter setup: saved once, pre-filled for every run. */
export interface EyeContactSetup {
  prompter_side: 'left' | 'right' | 'auto';
  /** Degrees between prompter and lens; 0 = work it out from the two distances. */
  angle_deg: number;
  prompter_cm: number;
  camera_cm: number;
  steadiness: number;
  aim_deg: number;
  quality: 'standard' | 'high' | 'max';
}

export async function getEyeContactSetup(): Promise<EyeContactSetup> {
  return api(`/api/eyecontact/settings`);
}

export async function saveEyeContactSetup(setup: Partial<EyeContactSetup>): Promise<EyeContactSetup> {
  return api(`/api/eyecontact/settings`, { method: 'PUT', body: JSON.stringify(setup) });
}

export async function eyeContactAvailable(): Promise<{ available: boolean; reason: string }> {
  return api(`/api/eyecontact/available`);
}

export async function startEyeContact(projectId: string, mediaId: string,
  opts: Partial<EyeContactSetup> = {}): Promise<{ job_id: string }> {
  return api(`/api/eyecontact/projects/${projectId}/media/${mediaId}`, {
    method: 'POST',
    body: JSON.stringify(opts),
  });
}

/** A short corrected stretch of a clip: both files come from the same cut, frame for frame. */
export interface EyeContactPreview extends EyeContactReport {
  before_path: string;
  after_path: string;
  start_s: number;
  duration_s: number;
}

/** Correct `durationS` seconds from `startS` without touching the project; poll with getEyeContactStatus. */
export async function startEyeContactPreview(projectId: string, mediaId: string, setup: EyeContactSetup,
  startS: number, durationS = 10): Promise<{ job_id: string }> {
  return api(`/api/eyecontact/projects/${projectId}/media/${mediaId}/preview`, {
    method: 'POST',
    body: JSON.stringify({ ...setup, start_s: startS, duration_s: durationS }),
  });
}

export async function getEyeContactStatus(jobId: string): Promise<EyeContactJob> {
  return api(`/api/eyecontact/status/${jobId}`);
}

export async function cancelEyeContact(jobId: string): Promise<{ status: string }> {
  return api(`/api/eyecontact/cancel/${jobId}`, { method: 'POST' });
}

export async function revertEyeContact(projectId: string, mediaId: string): Promise<{ media: MediaEntry[] }> {
  return api(`/api/eyecontact/projects/${projectId}/media/${mediaId}/revert`, { method: 'POST' });
}

export async function getMediaWaveform(projectId: string, mediaId: string): Promise<WaveformData> {
  return api(`/api/projects/${projectId}/media/${mediaId}/waveform`);
}

// --- LM Studio (LLM) ---

export interface LlmStatus {
  server_up: boolean;
  model_loaded: boolean;
  loaded_models: string[];
  cli_available: boolean;
  autostart_enabled: boolean;
  ready?: boolean;
}

export async function getLlmStatus(): Promise<LlmStatus> {
  return api('/api/llm/status');
}

export interface LlmModel {
  id: string;
  state?: string;   // "loaded" | "not-loaded"
  type?: string;    // "llm" | "vlm"
  [k: string]: any;
}

export interface LlmModelList {
  models: LlmModel[];
  selected: string;
  default: string;
}

/** Every text model LM Studio knows about, plus the currently-selected id. */
export async function getLlmModels(): Promise<LlmModelList> {
  return api('/api/llm/models');
}

export async function ensureLlm(): Promise<LlmStatus> {
  return api('/api/llm/ensure', { method: 'POST' });
}

export async function getRenderStatus(jobId: string): Promise<any> {
  return api(`/api/rendering/status/${jobId}`);
}

export async function getTranscriptionStatus(jobId: string): Promise<any> {
  return api(`/api/transcription/status/${jobId}`);
}

export async function deleteProject(projectId: string): Promise<void> {
  await api(`/api/projects/${projectId}`, { method: 'DELETE' });
}

export async function updateSettings(projectId: string, settings: Record<string, any>): Promise<any> {
  return api(`/api/projects/${projectId}/settings`, {
    method: 'PUT',
    body: JSON.stringify(settings),
  });
}

export async function getComfyUIStatus(): Promise<any> {
  return api('/api/comfyui/status');
}

export interface ComfyModelList {
  checkpoints: string[];
  diffusion_models: string[];
  loras: string[];
  vae: string[];
}

/** Model files ComfyUI can load, per loader type (empty if ComfyUI offline). */
export async function getComfyUIModels(): Promise<ComfyModelList> {
  return api('/api/comfyui/models');
}

export interface ComfyWorkflowDetail {
  file: string;
  valid: boolean;
  error?: string;
  nodes?: number;
  output_kind?: 'image' | 'video';
  can_prompt?: boolean;
  bindings?: Record<string, any>;
  roles_ok?: Record<string, boolean>;
}

export interface ComfyWorkflowList {
  workflows: ComfyWorkflowDetail[];
  roles: string[];
  /** Which file each role currently resolves to (null = unset). */
  selection: Record<string, string | null>;
  /** Which app-setting key names the workflow for each role. */
  settings_key: Record<string, string>;
}

/**
 * Every workflow file on disk, its detected bindings, the roles each can serve,
 * and the current per-role selection. Drives the Generation settings dropdowns.
 */
export async function getComfyUIWorkflows(): Promise<ComfyWorkflowList> {
  return api('/api/comfyui/workflows');
}

export async function triggerThumbnail(projectId: string, title?: string): Promise<any> {
  return api('/api/agents/thumbnail', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, title }),
  });
}

export async function triggerCaptions(projectId: string): Promise<any> {
  return api('/api/agents/captions', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  });
}

export async function getSchedulerStatus(): Promise<any> {
  return api('/api/scheduler/jobs');
}

export async function enqueueJob(projectId: string, priority: string = 'normal', opts?: any): Promise<any> {
  return api('/api/scheduler/enqueue', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, priority, ...opts }),
  });
}

// --- The presentation pass and its ComfyUI workflows ---

export interface WorkflowInfo {
  file: string;
  valid: boolean;
  error?: string;
  nodes?: number;
  output_kind?: 'image' | 'video';
  can_prompt?: boolean;
  bindings?: Record<string, string[]>;
  roles_ok?: Record<string, boolean>;
}

export async function listGenerationWorkflows(): Promise<{
  workflows: WorkflowInfo[];
  selected: Record<string, string | null>;
  roles: string[];
  settings_keys: Record<string, string>;
}> {
  return api('/api/presentation/workflows');
}

export async function runPresentationPass(projectId: string, settings: any = {}): Promise<any> {
  return api(`/api/presentation/${projectId}/run`, {
    method: 'POST',
    body: JSON.stringify({ settings }),
  });
}

/** Start the presentation pass (B-roll + zoom + captions) as a polled job. */
export async function startPresentationPass(projectId: string, settings: any = {}): Promise<{ status: string; job_id: string }> {
  return api(`/api/presentation/${projectId}/start`, {
    method: 'POST',
    body: JSON.stringify({ settings }),
  });
}

export async function getPresentationStatus(jobId: string): Promise<any> {
  return api(`/api/presentation/status/${jobId}`);
}

export async function getPresentationReport(projectId: string): Promise<any> {
  return api(`/api/presentation/${projectId}/report`);
}

export interface ScriptStatus {
  status: 'none' | 'stored' | 'aligned';
  text?: string;
  token_count?: number;
  aligned_words?: number;
  spelling_fixed?: number;
  paragraphs?: number;
  directives?: { kind: string; arg: string; at: number }[];
}

/** Store the speaker's script and align it to the transcript (fixes caption
    spelling; its paragraphs and [map:/sfx:/broll:…] directions feed the pass). */
export async function setProjectScript(projectId: string, text: string): Promise<ScriptStatus> {
  return api(`/api/projects/${projectId}/script`, {
    method: 'PUT',
    body: JSON.stringify({ text }),
  });
}

export async function getProjectScript(projectId: string): Promise<ScriptStatus> {
  return api(`/api/projects/${projectId}/script`);
}

export async function deleteProjectScript(projectId: string): Promise<{ status: string }> {
  return api(`/api/projects/${projectId}/script`, { method: 'DELETE' });
}

/** BuzzcafAI's brands -- distinct from this app's own `GENRE_OPTIONS`. Only
    used to pick which brand voice/genre map `plan_visuals` asks BuzzcafAI to
    use (backend/integrations/buzzedit_settings.py on BuzzcafAI's side). */
export const BUZZCAF_BRAND_OPTIONS = [
  'Beyond3Baje', 'Raat3Baje', 'Khayal3Baje', 'Life3Baje', 'Originals',
] as const;

export interface PlanVisualsResult {
  status: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  settings: Record<string, any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  visual_plan: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  skipped_beats?: any[];
  script: { text: string; directives: { kind: string; arg: string; at: number }[]; paragraphs: number };
}

/** Ask BuzzcafAI to plan the visuals for this project (from its stored script,
    else its transcript) and apply the annotated script it hands back -- the
    round trip happens server-side, on `POST /api/agents/plan_visuals`, so the
    browser never talks to BuzzcafAI directly. */
export async function planVisualsWithBuzzcaf(projectId: string, brand: string): Promise<PlanVisualsResult> {
  return api(`/api/agents/plan_visuals`, {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, brand }),
  });
}

export interface ShotPrompt {
  id: string;
  topic: string;
  kind: string;
  start_s: number;
  end_s: number;
  image_prompt?: string | null;
  video_prompt?: string | null;
  popup_text?: string | null;
}

/** The prompts the last B-roll run used, one per generated shot. */
export async function getShotPlan(projectId: string): Promise<{
  source: string; thumbnail_title?: string | null; count: number; prompts: ShotPrompt[];
}> {
  return api(`/api/presentation/${projectId}/shot_plan`);
}

export async function cancelSchedulerJob(jobId: string): Promise<any> {
  return api(`/api/scheduler/jobs/${jobId}`, {
    method: 'DELETE',
  });
}

export async function clearCompletedJobs(): Promise<any> {
  return api('/api/scheduler/clear_completed', {
    method: 'POST',
  });
}

export async function pauseScheduler(): Promise<any> {
  return api('/api/scheduler/pause', {
    method: 'POST',
  });
}

export async function resumeScheduler(): Promise<any> {
  return api('/api/scheduler/resume', {
    method: 'POST',
  });
}

export async function getGpuStatus(): Promise<any> {
  return api('/api/system/gpu');
}

export async function getSystemPaths(): Promise<any> {
  return api('/api/system/paths');
}

export async function getRecoveryState(): Promise<any> {
  return api('/api/settings/recovery_state');
}

export async function setLastProject(projectId: string | null): Promise<any> {
  return api('/api/settings/last_project', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  });
}

export async function autoSaveProject(projectId: string, projectData: Record<string, any>): Promise<any> {
  return api('/api/settings/auto_save', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, project_data: projectData }),
  });
}

export async function getAppSettings(): Promise<any> {
  return api('/api/settings/');
}

export async function updateAppSettings(updates: Record<string, any>): Promise<any> {
  return api('/api/settings/', {
    method: 'PUT',
    body: JSON.stringify(updates),
  });
}

// --- BuzzEdit-side presentation defaults & overrides (beat BuzzcafStudio) ---

export type PresentationOverrideMode = 'off' | 'fields' | 'all';

export interface PresentationOverridesState {
  presentation_overrides: Record<string, any>;
  override_mode: PresentationOverrideMode;
}

export async function getPresentationOverrides(): Promise<PresentationOverridesState> {
  return api('/api/settings/presentation_overrides');
}

export async function setPresentationOverrides(
  overrides: Record<string, any>,
  mode: PresentationOverrideMode,
): Promise<PresentationOverridesState> {
  return api('/api/settings/presentation_overrides', {
    method: 'PUT',
    body: JSON.stringify({ presentation_overrides: overrides, override_mode: mode }),
  });
}

// --- free stock fallback keys (Pexels / Pixabay) -----------------------------
// See presentation.stock: used only when ComfyUI cannot make a visual for a
// beat and the project's "Allow free stock" is on. Keys are masked server-side
// — GET/PUT never return one in full.

export interface StockKeysState {
  pexels_api_key: string;
  pexels_api_key_set: boolean;
  pixabay_api_key: string;
  pixabay_api_key_set: boolean;
}

export async function getStockKeys(): Promise<StockKeysState> {
  return api('/api/settings/stock_keys');
}

export async function setStockKeys(
  keys: { pexels_api_key?: string; pixabay_api_key?: string },
): Promise<StockKeysState> {
  return api('/api/settings/stock_keys', {
    method: 'PUT',
    body: JSON.stringify(keys),
  });
}

/** A short WAV of the project's own voice run through one named preset — for
 *  auditioning a voice preset without a full render. */
export async function audioPreview(
  projectId: string, preset: string, startS = 0, seconds = 20,
): Promise<{ path: string; preset: string; start_s: number; seconds: number }> {
  return api(`/api/presentation/${projectId}/audio_preview`, {
    method: 'POST',
    body: JSON.stringify({ preset, start_s: startS, seconds }),
  });
}

export async function clearVram(): Promise<any> {
  return api('/api/system/clear_vram', {
    method: 'POST',
  });
}

export async function exportFilmoraXml(projectId: string): Promise<any> {
  return api('/api/advanced/export_filmora', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  });
}

export async function applyColorGrading(projectId: string, preset: string): Promise<any> {
  return api('/api/advanced/color_grade', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, preset }),
  });
}

export async function convertToShorts(projectId: string): Promise<any> {
  return api('/api/advanced/shorts', {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId }),
  });
}

export function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(async <T>(fn: () => Promise<T>): Promise<T | null> => {
    setLoading(true);
    setError(null);
    try {
      const result = await fn();
      return result;
    } catch (err: any) {
      setError(err.message || 'Unknown error');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { loading, error, execute, setError };
}

const VIDEO_EXTS = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v', 'mpg', 'mpeg', 'wmv', 'flv', 'mts', 'm2ts'];
const AUDIO_EXTS = ['mp3', 'wav', 'aac', 'm4a', 'flac', 'ogg', 'opus', 'wma', 'aiff'];
const IMAGE_EXTS = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif', 'tif', 'tiff'];

export const MEDIA_FILTERS = [
  { name: 'All media', extensions: [...VIDEO_EXTS, ...AUDIO_EXTS, ...IMAGE_EXTS] },
  { name: 'Video', extensions: VIDEO_EXTS },
  { name: 'Audio & music', extensions: AUDIO_EXTS },
  { name: 'Images', extensions: IMAGE_EXTS },
];

export function useElectron() {
  const pickFile = useCallback(async (filters?: any) => {
    if (window.electronAPI) {
      const result = await window.electronAPI.dialogOpenFile({
        properties: ['openFile'],
        filters: filters || [{ name: 'Video', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm'] }],
      });
      return result.filePaths?.[0];
    }
    return null;
  }, []);

  const pickFiles = useCallback(async (filters?: any): Promise<string[]> => {
    if (window.electronAPI) {
      const result = await window.electronAPI.dialogOpenFile({
        properties: ['openFile', 'multiSelections'],
        filters: filters || MEDIA_FILTERS,
      });
      return result.filePaths || [];
    }
    return [];
  }, []);

  /** Pick one or more folders; the backend walks them for media. */
  const pickFolders = useCallback(async (): Promise<string[]> => {
    if (window.electronAPI) {
      const result = await window.electronAPI.dialogOpenFile({
        properties: ['openDirectory', 'multiSelections'],
      });
      return result.filePaths || [];
    }
    return [];
  }, []);

  const saveFile = useCallback(async (defaultPath?: string) => {
    if (window.electronAPI) {
      const result = await window.electronAPI.dialogSaveFile({
        defaultPath: defaultPath || 'output.mp4',
        filters: [{ name: 'Video', extensions: ['mp4'] }],
      });
      return result.filePath;
    }
    return null;
  }, []);

  return { pickFile, pickFiles, pickFolders, saveFile };
}
