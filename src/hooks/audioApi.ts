/**
 * Typed clients for the audio suite's routes (AUDIO_SUITE_PLAN.md §2.6, brief A4).
 *
 * Everything lives under `${API_BASE}/api/audio`. The catalogue, presets and racks come from
 * `routes/audio.py`; analysis, the noise print and the process lane from
 * `routes/audio_process.py`. A backend started before those routes landed answers FastAPI's
 * bare `{"detail": "Not Found"}`: `routeMissing()` tells that apart from a route's own 404
 * ("No noise print captured…"), so the panel can say "not available yet" instead of failing.
 */

import { API_BASE, mediaStreamUrl } from './api';

// -- catalogue -----------------------------------------------------------------------------

export type ParamType = 'number' | 'choice' | 'bool' | 'curve' | 'bands';
export type CurvePoint = [number, number];
export type BandValue = Record<string, number | string | boolean>;

/** One parameter of an effect, in Audition's UI units. */
export interface ParamDef {
  key: string;
  label: string;
  type: ParamType;
  /** number: the value range; curve: the y axis */
  min: number | null;
  max: number | null;
  step: number | null;
  default: unknown;
  unit: string;
  choices?: Array<string | number>;
  /** every ffmpeg option it drives takes a runtime command (else the live engine restarts) */
  live: boolean;
  /** curve only: the x axis */
  x_min?: number;
  x_max?: number;
  x_unit?: string;
  x_log?: boolean;
  /** bands only: the ParamDefs of one band */
  fields?: ParamDef[];
}

export interface EffectDef {
  id: string;
  name: string;
  category: string;
  /** "ffmpeg" = a rack effect, heard live; "process" = rendered into a new version */
  engine: string;
  description: string;
  params: ParamDef[];
  presets: Record<string, Record<string, unknown>>;
  windowable: boolean;
}

export interface Catalogue {
  effects: EffectDef[];
  categories: string[];
}

/** An effect in a rack, as `PUT /{project}/rack` stores it (assignable to LiveEngine's LiveEffect). */
export interface AudioEffect {
  id: string;
  effect: string;
  params: Record<string, unknown>;
  enabled: boolean;
  start_s?: number | null;
  end_s?: number | null;
  preset?: string | null;
}

/** `{effect: {preset name: params}}` */
export type UserPresets = Record<string, Record<string, Record<string, unknown>>>;

export interface SourceAudioState {
  audio_override: string | null;
  history_len: number;
}

export interface RacksResponse {
  racks: Record<string, AudioEffect[]>;
  items: Record<string, AudioEffect[]>;
  sources: Record<string, SourceAudioState>;
}

export interface RackResponse {
  scope: string;
  effects: AudioEffect[];
  revision: number;
}

/** A whole saved rack (`data/audio_rack_presets.json`); applying it gives every effect a fresh id. */
export interface RackPreset {
  effects: AudioEffect[];
  created?: string;
  note?: string;
}

/** `{preset name: RackPreset}` */
export type RackPresets = Record<string, RackPreset>;

export interface RackPreview {
  path: string;
  seconds: number;
}

// -- analysis, noise, processes ------------------------------------------------------------

export interface LoudnessSeries {
  t: number[];
  momentary_lufs: Array<number | null>;
  short_term_lufs: Array<number | null>;
}

export interface Spectrum {
  freqs_hz: number[];
  mean_db: number[];
}

export interface AnalyzeResult {
  source_id?: string;
  start_s: number;
  end_s: number;
  peak_dbfs: number | null;
  rms_dbfs: number | null;
  dc_offset: number | null;
  noise_floor_dbfs: number | null;
  integrated_lufs: number | null;
  lra_lu: number | null;
  true_peak_dbtp: number | null;
  loudness_series?: LoudnessSeries | null;
  spectrum_png?: string | null;
  spectrum?: Spectrum | null;
}

export interface NoiseDetectResult {
  source_id?: string;
  start_s: number;
  end_s: number;
  floor_dbfs: number | null;
  method?: string;
}

export interface NoisePrint {
  source_id?: string;
  start_s: number;
  end_s: number;
  floor_dbfs?: number | null;
  freqs_hz: number[];
  mean_db: number[];
  bands_db?: number[];
  band_floor_db?: number | null;
  created?: string;
}

export interface ProcessJob {
  id?: string;
  status: string; // pending | running | completed | failed
  progress: number;
  message?: string;
  result?: Record<string, unknown> | null;
  error?: string | null;
}

export interface ProcessPreview {
  path: string;
  seconds?: number;
  start_s?: number;
  source_id?: string;
}

export interface AudioVersion {
  path: string;
  effect: string;
  params?: Record<string, unknown>;
  start_s?: number | null;
  end_s?: number | null;
  created?: string;
}

export interface VersionsResult {
  source_id: string;
  audio_override: string | null;
  versions: AudioVersion[];
}

export interface UndoResult {
  source_id: string;
  audio_override: string | null;
  history_len: number;
  revision?: number;
  undone?: AudioVersion | null;
}

// -- transport -----------------------------------------------------------------------------

/** A failed call: the HTTP status (0 = no answer) and FastAPI's `detail`. */
export class AudioApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(status ? `API ${status}: ${detail}` : detail);
    this.status = status;
    this.detail = detail;
  }
}

/** The route itself is absent (a backend from before the audio suite), not a 404 it raised. */
export function routeMissing(err: unknown): boolean {
  return err instanceof AudioApiError
    && ((err.status === 404 && err.detail === 'Not Found') || err.status === 405);
}

/** Readable text for a failed call. */
export function errorText(err: unknown): string {
  if (err instanceof AudioApiError) return err.detail || err.message;
  if (err instanceof Error) return err.message;
  return String(err);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/audio${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
  } catch (err) {
    throw new AudioApiError(0, `backend unreachable (${errorText(err)})`);
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    let detail = text || res.statusText;
    try {
      const body = JSON.parse(text);
      if (body && body.detail !== undefined) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* not JSON: keep the text */
    }
    throw new AudioApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body ?? {}) });

const pid = (projectId: string) => encodeURIComponent(projectId);

// -- catalogue and presets -------------------------------------------------------------------

export function getEffects(): Promise<Catalogue> {
  return request<Catalogue>('/effects');
}

export function getPresets(): Promise<{ presets: UserPresets }> {
  return request('/presets');
}

export function savePreset(effect: string, name: string, params: Record<string, unknown>): Promise<{ presets: UserPresets }> {
  return post('/presets', { effect, name, params });
}

export function deletePreset(effect: string, name: string): Promise<{ presets: UserPresets }> {
  return request(`/presets/${encodeURIComponent(effect)}/${encodeURIComponent(name)}`, { method: 'DELETE' });
}

// -- racks -----------------------------------------------------------------------------------

export function getRacks(projectId: string): Promise<RacksResponse> {
  return request(`/${pid(projectId)}/racks`);
}

/** `scope` = "A1", "MASTER" or "item:<id>"; replaces that rack. */
export function putRack(projectId: string, scope: string, effects: AudioEffect[]): Promise<RackResponse> {
  return request(`/${pid(projectId)}/rack`, { method: 'PUT', body: JSON.stringify({ scope, effects }) });
}

export function previewRack(projectId: string, body: {
  scope: string; source_id?: string; start_s: number; seconds: number; effects?: AudioEffect[];
}): Promise<RackPreview> {
  return post(`/${pid(projectId)}/preview`, body);
}

// -- rack presets (whole racks) --------------------------------------------------------------

export function getRackPresets(): Promise<{ presets: RackPresets }> {
  return request('/rack_presets');
}

/** Saves (or replaces) the rack preset `name`. */
export function saveRackPreset(name: string, effects: AudioEffect[], note?: string): Promise<{ presets: RackPresets }> {
  return post('/rack_presets', { name, effects, note: note ?? null });
}

export function deleteRackPreset(name: string): Promise<{ presets: RackPresets }> {
  return request(`/rack_presets/${encodeURIComponent(name)}`, { method: 'DELETE' });
}

/** Replaces the `scope` rack with the preset's effects under fresh ids; answers as `putRack`. */
export function applyRackPreset(projectId: string, scope: string, name: string): Promise<RackResponse> {
  return post(`/${pid(projectId)}/rack/apply_preset`, { scope, name });
}

// -- analysis and noise ----------------------------------------------------------------------

/** The process catalogue: the route returns `{id: EffectDef}`; a list is accepted too. */
export async function getProcesses(): Promise<EffectDef[]> {
  const raw = await request<unknown>('/processes');
  if (Array.isArray(raw)) return raw as EffectDef[];
  if (raw && typeof raw === 'object') {
    const effects = (raw as { effects?: unknown }).effects;
    if (Array.isArray(effects)) return effects as EffectDef[];
    return Object.values(raw as Record<string, EffectDef>).filter((d) => d && typeof d === 'object' && 'id' in d);
  }
  return [];
}

export function analyzeRange(projectId: string, body: {
  source_id?: string; start_s: number; end_s: number;
}): Promise<AnalyzeResult> {
  return post(`/${pid(projectId)}/analyze`, body);
}

export function detectNoise(projectId: string, sourceId?: string): Promise<NoiseDetectResult> {
  return post(`/${pid(projectId)}/noise/detect`, { source_id: sourceId ?? null });
}

export function captureNoisePrint(projectId: string, body: {
  source_id?: string; start_s: number; end_s: number;
}): Promise<NoisePrint> {
  return post(`/${pid(projectId)}/noise/print`, body);
}

export function getNoisePrint(projectId: string): Promise<NoisePrint> {
  return request(`/${pid(projectId)}/noise/print`);
}

// -- process lane ----------------------------------------------------------------------------

export function startProcess(projectId: string, body: {
  source_id?: string; effect: string; params: Record<string, unknown>;
  start_s?: number | null; end_s?: number | null;
}): Promise<{ job_id: string }> {
  return post(`/${pid(projectId)}/process`, body);
}

export function getProcessJob(projectId: string, jobId: string): Promise<ProcessJob> {
  return request(`/${pid(projectId)}/process/${encodeURIComponent(jobId)}`);
}

export function previewProcess(projectId: string, body: {
  source_id?: string; effect: string; params: Record<string, unknown>; start_s: number; seconds: number;
}): Promise<ProcessPreview> {
  return post(`/${pid(projectId)}/process/preview`, body);
}

export function undoProcess(projectId: string, sourceId?: string): Promise<UndoResult> {
  return post(`/${pid(projectId)}/undo`, { source_id: sourceId ?? null });
}

export function getVersions(projectId: string, sourceId?: string): Promise<VersionsResult> {
  const query = sourceId ? `?source_id=${encodeURIComponent(sourceId)}` : '';
  return request(`/${pid(projectId)}/versions${query}`);
}

// -- helpers ---------------------------------------------------------------------------------

/** A new rack effect id, `fx_<8 hex>`: `[A-Za-z0-9_]+` and unique in the project. */
export function newEffectId(taken?: Iterable<string>): string {
  const used = new Set(taken ?? []);
  for (;;) {
    const bytes = new Uint8Array(4);
    crypto.getRandomValues(bytes);
    const id = `fx_${Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')}`;
    if (!used.has(id)) return id;
  }
}

/** A deep copy of a JSON-like value (params hold arrays for curves and bands). */
export function cloneValue<T>(value: T): T {
  return value === undefined ? value : JSON.parse(JSON.stringify(value));
}

/** Every parameter at its catalogue default. */
export function defaultParams(def: EffectDef): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const p of def.params || []) out[p.key] = cloneValue(p.default);
  return out;
}

/** A rendered file (preview WAV, spectrum PNG) through the media route, never cached. */
export function audioFileUrl(path: string): string {
  return `${mediaStreamUrl(path)}&v=${Date.now()}`;
}

/** Tiny pub/sub for the live engine's fast events, so only the widgets that show them repaint. */
export class Signal<T> {
  private listeners = new Set<(value: T) => void>();

  subscribe(fn: (value: T) => void): () => void {
    this.listeners.add(fn);
    return () => { this.listeners.delete(fn); };
  }

  emit(value: T): void {
    this.listeners.forEach((fn) => fn(value));
  }
}
