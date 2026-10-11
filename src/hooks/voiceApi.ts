/**
 * Typed clients for the Voice panel (VOICE_CHANGER_PLAN.md §2.7, brief S2b).
 *
 * The voice routes live under `${API_BASE}/api/audio` (`backend/routes/audio_voice.py`, briefs S1
 * and S2a); their jobs run in the process lane's registry and are polled on
 * `GET /api/audio/{project}/process/{job_id}` (see `components/voice/JobProgress.tsx`). A fresh
 * microphone recording reaches the media pool through `POST /api/projects/{project}/media/upload`
 * and any converted file leaves through `GET /api/media/download?path&format&name`.
 *
 * Errors are `AudioApiError`s (the audio suite's), so `routeMissing()` tells a backend that
 * predates a route apart from the route's own 404 and the panel can say "not available yet".
 *
 * Besides the clients this module carries the few things the Timeline and the Media pool share
 * with the panel: `openVoicePanel()` (show the panel on a tab, optionally with a media-pool item
 * as the Convert source), `mediaPoolChanged` (a conversion or an upload added an item: refresh the
 * pool) and `convertVoiceFromSelection()` (the range band's "Convert voice…").
 */

import { API_BASE, getTimeline } from './api';
import { AudioApiError, AudioEffect, Signal, cloneValue, errorText, routeMissing } from './audioApi';
import { useLayoutStore } from './layout';
import { useProjectStore } from './store';

// -- types -------------------------------------------------------------------------------------

/** A reference clip in `data/voices` (`GET /voices`). */
export interface VoiceEntry {
  name: string;
  kind: string | null;
  consent: string | null;
  created: string | null;
  seconds: number | null;
  source: string | null;
  path: string;
}

/** The voice_conversion values a request may set beside its voice and preset. */
export interface VoiceParams {
  match_pitch?: boolean;
  pitch_shift?: number;
  steps?: number;
  cfg_rate?: number;
}

/** seed-vc's settings (`presets.json` engine_defaults, a preset's engine block). */
export interface EngineSettings {
  diffusion_steps?: number;
  length_adjust?: number;
  inference_cfg_rate?: number;
  f0_condition?: boolean;
  auto_f0_adjust?: boolean;
  pitch_shift?: number;
  [key: string]: unknown;
}

/** A preset's creature-FX intent: pitch_semitones, formant, drive, reverb, sub_octave. */
export type FxIntent = Record<string, number>;

export interface VoicePreset {
  name: string;
  /** the stem of its reference clip in the library */
  voice: string;
  voice_in_library: boolean;
  engine: EngineSettings;
  fx: FxIntent;
  /** the intent as rack effects (only catalogue ids that exist) */
  fx_rack: AudioEffect[];
  params: VoiceParams & { preset?: string };
}

export interface PresetsResponse {
  engine_defaults: EngineSettings;
  presets: Record<string, VoicePreset>;
}

export type RegionStatus = 'pending' | 'applied' | 'failed';

/** One `voice_regions` entry: programme seconds, what the timeline shows. */
export interface VoiceRegion {
  id: string;
  start_s: number;
  end_s: number;
  voice: string;
  preset?: string | null;
  params: VoiceParams;
  fx: AudioEffect[];
  status: RegionStatus;
  error?: string;
  applied_version?: string;
}

export interface RegionsResponse {
  regions: VoiceRegion[];
}

/** The regions apply job's result. */
export interface RegionsApplyResult {
  applied: string[];
  failed: Array<{ id: string; error: string }>;
  version: string | null;
}

/** The convert_file job's result: the new media-pool item. */
export interface ConvertFileResult {
  media_id: string;
  path: string;
  seconds: number;
}

export interface BatchItemResult {
  media_id: string;
  out_media_id?: string;
  path?: string;
  error?: string;
}

export interface BatchResult {
  items: BatchItemResult[];
}

export interface WorkerStatus {
  running: boolean | null;
  device: string | null;
  idle_s: number | null;
}

/** `POST /api/projects/{project}/media/upload`. */
export interface UploadedMedia {
  media_id: string;
  path: string;
  name: string;
  kind: string;
  seconds: number;
}

export interface SpeakerSpan {
  speaker: string | null;
  start_s: number;
  end_s: number;
  text?: string;
}

export interface SpeakerSpansResponse {
  has_script: boolean;
  spans: SpeakerSpan[];
  speakers: string[];
}

export type VoiceMapEntry = VoiceParams & { voice?: string; preset?: string | null };

/** What a job's poll answers (`processes.JOBS`). */
export interface JobState {
  id: string;
  status: string; // pending | running | completed | failed
  progress: number;
  message?: string;
  result?: Record<string, unknown> | null;
  error?: string | null;
}

// -- transport ---------------------------------------------------------------------------------

async function detailOf(res: Response): Promise<string> {
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
  return detail;
}

/** `url` from `/api/...`; a FormData body goes without a JSON content type (the browser sets the
 *  multipart boundary). */
async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const isForm = typeof FormData !== 'undefined' && init?.body instanceof FormData;
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${url}`, {
      ...init,
      headers: isForm ? init?.headers : { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
  } catch (err) {
    throw new AudioApiError(0, `backend unreachable (${errorText(err)})`);
  }
  if (!res.ok) throw new AudioApiError(res.status, await detailOf(res));
  return res.json() as Promise<T>;
}

const audio = (path: string) => `/api/audio${path}`;
const pid = (projectId: string) => encodeURIComponent(projectId);
const send = (method: string, body?: unknown): RequestInit => ({ method, body: JSON.stringify(body ?? {}) });

// -- voices library, presets, worker (S1) -------------------------------------------------------

export function listVoices(): Promise<{ voices: VoiceEntry[] }> {
  return call(audio('/voices'));
}

/** Multipart: the clip (a browser recording in webm/opus is fine, ffmpeg converts), its name,
 *  kind and consent note. The backend refuses < 3 s, > 60 s and a taken name. */
export function addVoice(file: Blob, opts: { name: string; kind?: string; consent?: string; filename?: string }): Promise<{ voice: VoiceEntry }> {
  const form = new FormData();
  form.append('file', file, opts.filename || (file instanceof File ? file.name : 'recording.webm'));
  form.append('name', opts.name);
  form.append('kind', opts.kind || 'custom');
  form.append('consent', opts.consent || '');
  return call(audio('/voices'), { method: 'POST', body: form });
}

export function renameVoice(name: string, newName: string): Promise<{ voice: VoiceEntry }> {
  return call(audio(`/voices/${encodeURIComponent(name)}`), send('PUT', { name: newName }));
}

export function deleteVoice(name: string): Promise<{ deleted: string }> {
  return call(audio(`/voices/${encodeURIComponent(name)}`), { method: 'DELETE' });
}

export function getVoicePresets(): Promise<PresetsResponse> {
  return call(audio('/voice/presets'));
}

export function getVoiceWorker(): Promise<WorkerStatus> {
  return call(audio('/voice/worker'));
}

/** Quit after the running job, then offload the GPU. */
export function stopVoiceWorker(): Promise<WorkerStatus> {
  return call(audio('/voice/worker/stop'), send('POST'));
}

// -- project: speaker spans, voice map, one range (S1) ------------------------------------------

export function getSpeakerSpans(projectId: string): Promise<SpeakerSpansResponse> {
  return call(audio(`/${pid(projectId)}/speaker_spans`));
}

export function getVoiceMap(projectId: string): Promise<{ voice_map: Record<string, VoiceMapEntry> }> {
  return call(audio(`/${pid(projectId)}/voice_map`));
}

export function putVoiceMap(projectId: string, voiceMap: Record<string, VoiceMapEntry>): Promise<{ voice_map: Record<string, VoiceMapEntry> }> {
  return call(audio(`/${pid(projectId)}/voice_map`), send('PUT', { voice_map: voiceMap }));
}

export function applyVoiceMap(projectId: string): Promise<{ job_id: string }> {
  return call(audio(`/${pid(projectId)}/voice/apply_map`), send('POST'));
}

/** One range in a voice as a process-lane job (a new version; Undo pops it). `program_range`:
 *  start_s/end_s are programme seconds, mapped onto the source pieces across cuts. */
export function convertRange(projectId: string, body: {
  source_id?: string | null; start_s: number; end_s: number; program_range?: boolean;
  voice?: string | null; preset?: string | null; params: VoiceParams;
}): Promise<{ job_id: string }> {
  return call(audio(`/${pid(projectId)}/voice/convert`), send('POST', body));
}

// -- voice regions (S2a) ------------------------------------------------------------------------

export function getRegions(projectId: string): Promise<RegionsResponse> {
  return call(audio(`/${pid(projectId)}/voice/regions`));
}

/** Replace the regions: the backend checks them, clamps them to the programme and trims overlaps
 *  (a later region starts where the one before ends), and answers what it stored. */
export function putRegions(projectId: string, regions: VoiceRegion[]): Promise<RegionsResponse> {
  return call(audio(`/${pid(projectId)}/voice/regions`), send('PUT', { regions }));
}

/** Every pending region (or the listed ones) in ONE new version; result `RegionsApplyResult`. */
export function applyRegions(projectId: string, ids?: string[]): Promise<{ job_id: string }> {
  return call(audio(`/${pid(projectId)}/voice/regions/apply`), send('POST', ids && ids.length ? { ids } : {}));
}

/** Undo the last apply: the version is popped, its tagged rack FX removed, its regions pending. */
export function undoRegions(projectId: string): Promise<RegionsResponse> {
  return call(audio(`/${pid(projectId)}/voice/regions/undo`), send('POST'));
}

// -- whole files (S2a) --------------------------------------------------------------------------

/** A whole media-pool item (or a file path) into a new 48 kHz WAV registered in the media pool;
 *  result `ConvertFileResult`. */
export function convertFile(projectId: string, body: {
  media_id?: string | null; path?: string | null; voice?: string | null; preset?: string | null;
  params: VoiceParams; name?: string | null;
}): Promise<{ job_id: string }> {
  return call(audio(`/${pid(projectId)}/voice/convert_file`), send('POST', body));
}

/** Several media-pool items in one voice, one after another; result `BatchResult`. The job's
 *  message reads "k/n <media_id>: ..." while item k runs. */
export function batchConvert(projectId: string, body: {
  media_ids: string[]; voice?: string | null; preset?: string | null; params: VoiceParams;
}): Promise<{ job_id: string }> {
  return call(audio(`/${pid(projectId)}/voice/batch`), send('POST', body));
}

// -- media upload and download ------------------------------------------------------------------

/** A browser recording (webm/ogg, transcoded to WAV by the backend) or any audio/video file into
 *  the project's media pool. */
export function uploadMedia(projectId: string, file: Blob, opts: { name?: string; filename?: string } = {}): Promise<UploadedMedia> {
  const form = new FormData();
  form.append('file', file, opts.filename || (file instanceof File ? file.name : 'recording.webm'));
  if (opts.name) form.append('name', opts.name);
  return call(`/api/projects/${pid(projectId)}/media/upload`, { method: 'POST', body: form });
}

/** The upload route answered 404/405: the backend predates it. */
export function uploadMissing(err: unknown): boolean {
  return err instanceof AudioApiError && (err.status === 404 || err.status === 405);
}

export function mediaDownloadUrl(path: string, format: 'wav' | 'mp3', name?: string): string {
  const query = new URLSearchParams({ path, format });
  if (name) query.set('name', name);
  return `${API_BASE}/api/media/download?${query.toString()}`;
}

/** Fetch the file as WAV or MP3 and hand it to the browser's download (Electron asks where to
 *  save it). Fetched first, so a missing route or file is a message, not a page. */
export async function downloadMedia(path: string, format: 'wav' | 'mp3', name?: string): Promise<void> {
  let res: Response;
  try {
    res = await fetch(mediaDownloadUrl(path, format, name));
  } catch (err) {
    throw new AudioApiError(0, `backend unreachable (${errorText(err)})`);
  }
  if (!res.ok) throw new AudioApiError(res.status, await detailOf(res));
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${safeFileName(name || fileStem(path))}.${format}`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

/** Readable text for a failed download. */
export function downloadError(err: unknown): string {
  if (routeMissing(err)) return 'The download route is not available on this backend yet (restart BuzzEdit to load it).';
  return `Download failed: ${errorText(err)}`;
}

// -- small helpers ------------------------------------------------------------------------------

export function fileName(path: string): string {
  return String(path || '').split(/[\\/]/).pop() || String(path || '');
}

export function fileStem(path: string): string {
  const base = fileName(path);
  const dot = base.lastIndexOf('.');
  return dot > 0 ? base.slice(0, dot) : base;
}

export function safeFileName(name: string): string {
  return String(name || '').replace(/[\\/:*?"<>|\r\n]+/g, '_').trim() || 'voice';
}

/** `YYYYMMDD-HHMMSS`, for timestamped output names (SkinWalker named its outputs so). */
export function timeStamp(date = new Date()): string {
  const p = (n: number) => String(n).padStart(2, '0');
  return `${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}-${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
}

/** Windows paths compare case-insensitively and with either slash. */
export function samePath(a: string | null | undefined, b: string | null | undefined): boolean {
  if (!a || !b) return false;
  const norm = (p: string) => p.replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
  return norm(a) === norm(b);
}

/** The voices library's own name rule (`voice_conversion._check_name`). */
export function voiceNameProblem(name: string): string | null {
  const clean = name.trim();
  if (!clean) return 'Give the voice a name.';
  if (!/^[A-Za-z0-9][A-Za-z0-9 _.-]{0,62}$/.test(clean) || clean.endsWith('.') || clean.toLowerCase().endsWith('.wav')) {
    return "A voice name is 1-63 letters, digits, spaces, '_', '-' or '.', starting with a letter or digit.";
  }
  return null;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
export const PITCH_RANGE: [number, number] = [-12, 12];
export const STEPS_RANGE: [number, number] = [10, 50];

/** m:ss.cc, the Audio panel's time format. */
export function formatSeconds(seconds: number): string {
  const t = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(2).padStart(5, '0')}`;
}

// -- voice choice: a voice, an optional preset, its params and FX -------------------------------

/** What a conversion runs with: the preset's values, then `params`, then `voice` (the backend
 *  resolves it the same way). `fx` is the preset's FX rack (regions put it on the A1 rack). */
export interface VoiceChoice {
  voice: string;
  preset: string | null;
  params: VoiceParams;
  fx: AudioEffect[];
}

/** The engine defaults as voice_conversion params. */
export function engineParams(defaults?: EngineSettings | null): VoiceParams {
  const d = defaults || {};
  return {
    match_pitch: !!d.f0_condition,
    pitch_shift: clamp(Math.round(Number(d.pitch_shift) || 0), PITCH_RANGE[0], PITCH_RANGE[1]),
    steps: clamp(Math.round(Number(d.diffusion_steps) || 25), STEPS_RANGE[0], STEPS_RANGE[1]),
    cfg_rate: typeof d.inference_cfg_rate === 'number' ? d.inference_cfg_rate : 0.7,
  };
}

/** A preset fills the voice, the params and the FX. */
export function presetChoice(preset: VoicePreset): VoiceChoice {
  const p = preset.params || {};
  return {
    voice: preset.voice || '',
    preset: preset.name,
    params: {
      match_pitch: !!p.match_pitch,
      pitch_shift: typeof p.pitch_shift === 'number' ? p.pitch_shift : 0,
      steps: typeof p.steps === 'number' ? p.steps : 25,
      ...(typeof p.cfg_rate === 'number' ? { cfg_rate: p.cfg_rate } : {}),
    },
    fx: cloneValue(preset.fx_rack || []),
  };
}

export function emptyChoice(presets?: PresetsResponse | null): VoiceChoice {
  return { voice: '', preset: null, params: engineParams(presets?.engine_defaults), fx: [] };
}

const CHOICE_KEY = 'buzzedit.voice.default.v1';

/** The panel remembers the last voice/preset picked; it is the default for a new region. */
export function rememberChoice(choice: { voice: string; preset: string | null }): void {
  try {
    localStorage.setItem(CHOICE_KEY, JSON.stringify({ voice: choice.voice || '', preset: choice.preset || null }));
  } catch {
    /* a full or disabled localStorage must not break the panel */
  }
}

function storedChoice(): { voice: string; preset: string | null } | null {
  try {
    const raw = localStorage.getItem(CHOICE_KEY);
    if (!raw) return null;
    const saved = JSON.parse(raw);
    return {
      voice: typeof saved?.voice === 'string' ? saved.voice : '',
      preset: typeof saved?.preset === 'string' ? saved.preset : null,
    };
  } catch {
    return null;
  }
}

export const voiceInLibrary = (voices: VoiceEntry[], name: string | null | undefined) =>
  !!name && voices.some((v) => v.name.toLowerCase() === String(name).toLowerCase());

/** The default voice/preset: the one picked last, else the first preset whose voice is in the
 *  library, else the first library voice, else the first preset; null when there is neither. */
export function defaultChoice(presets: PresetsResponse | null, voices: VoiceEntry[]): VoiceChoice | null {
  const table = presets?.presets || {};
  const stored = storedChoice();
  if (stored?.preset && table[stored.preset]) {
    const choice = presetChoice(table[stored.preset]);
    if (stored.voice) choice.voice = stored.voice;
    return choice;
  }
  if (stored?.voice && voiceInLibrary(voices, stored.voice)) {
    return { ...emptyChoice(presets), voice: stored.voice };
  }
  const ready = Object.values(table).find((p) => p.voice_in_library);
  if (ready) return presetChoice(ready);
  if (voices.length) return { ...emptyChoice(presets), voice: voices[0].name };
  const first = Object.values(table)[0];
  return first ? presetChoice(first) : null;
}

/** "Demon (demon)", "woman": who a choice speaks as. */
export function choiceLabel(choice: { voice?: string | null; preset?: string | null }): string {
  if (choice.preset) return `${choice.preset}${choice.voice ? ` (${choice.voice})` : ''}`;
  return choice.voice || 'no voice';
}

// -- regions helpers ----------------------------------------------------------------------------

export function newRegionId(taken: Iterable<string> = []): string {
  const used = new Set(taken);
  for (;;) {
    const bytes = new Uint8Array(4);
    crypto.getRandomValues(bytes);
    const id = `vr_${Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')}`;
    if (!used.has(id)) return id;
  }
}

export function regionFromChoice(choice: VoiceChoice, startS: number, endS: number, taken: Iterable<string> = []): VoiceRegion {
  return {
    id: newRegionId(taken),
    start_s: startS,
    end_s: endS,
    voice: choice.voice || '',
    preset: choice.preset,
    params: { ...choice.params },
    fx: cloneValue(choice.fx || []),
    status: 'pending',
  };
}

/** Mirror the stored regions into the store's project settings: the editor's autosave writes the
 *  whole settings back, so a stale copy there would undo them. */
export function syncRegionsToStore(projectId: string, regions: VoiceRegion[]): void {
  const state = useProjectStore.getState();
  if (state.project?.id !== projectId) return;
  state.updateProject({ settings: { ...(state.project.settings || {}), voice_regions: regions } });
}

/** Fetch the stored regions and mirror them into the store. */
export async function reloadRegions(projectId: string): Promise<VoiceRegion[]> {
  const res = await getRegions(projectId);
  const regions = res.regions || [];
  syncRegionsToStore(projectId, regions);
  return regions;
}

/** An apply or an undo changed the A1 rack and the source's version: pull the timeline in, so the
 *  autosave never writes an older one back. */
export async function refreshTimeline(projectId: string): Promise<void> {
  try {
    const tl = await getTimeline(projectId);
    const state = useProjectStore.getState();
    if (tl && state.project?.id === projectId) state.updateProject({ timeline: tl });
  } catch {
    /* the next timeline load picks it up */
  }
}

/** Readable text for a failed regions call. */
export function regionsError(err: unknown): string {
  if (routeMissing(err)) return 'Voice regions are not available on this backend yet (restart BuzzEdit to load the voice routes).';
  return errorText(err);
}

/** Readable text for a conversion that could not start: a missing route, no engine (503), a
 *  render or another conversion running (409), a bad voice or input (422). */
export function startError(err: unknown, what: string): string {
  if (routeMissing(err)) return `${what} is not available on this backend yet (restart BuzzEdit to load the voice routes).`;
  if (err instanceof AudioApiError && err.status === 503) return `The voice engine is not available: ${err.detail}`;
  if (err instanceof AudioApiError && err.status === 409) return `Not now: ${err.detail}`;
  return `${what} failed: ${errorText(err)}`;
}

/** Add a pending region over `range` (programme seconds) with `choice` (default: the default
 *  voice/preset) and say what the backend made of it (trimmed, or covered by another region). */
export async function addRegionFromSelection(
  projectId: string,
  range: { start_s: number; end_s: number },
  choice?: VoiceChoice | null,
): Promise<{ regions: VoiceRegion[]; note: string }> {
  let pick = choice ?? null;
  if (!pick) {
    const [presets, voices] = await Promise.all([
      getVoicePresets().catch(() => null),
      listVoices().then((r) => r.voices || []).catch(() => [] as VoiceEntry[]),
    ]);
    pick = defaultChoice(presets, voices);
  }
  if (!pick) {
    throw new Error('No voice or preset yet: record or upload a voice on the Voices tab first.');
  }
  const current = (await getRegions(projectId)).regions || [];
  const region = regionFromChoice(pick, range.start_s, range.end_s, current.map((r) => r.id));
  const res = await putRegions(projectId, [...current, region]);
  const regions = res.regions || [];
  syncRegionsToStore(projectId, regions);
  const stored = regions.find((r) => r.id === region.id);
  const who = choiceLabel(pick);
  let note: string;
  if (!stored) {
    note = 'Nothing added: the selection lies inside an existing region or outside the programme.';
  } else if (Math.abs(stored.start_s - range.start_s) > 0.001 || Math.abs(stored.end_s - range.end_s) > 0.001) {
    note = `Added ${formatSeconds(stored.start_s)}–${formatSeconds(stored.end_s)} in ${who}, trimmed to the programme and the region before it.`;
  } else {
    note = `Added ${formatSeconds(stored.start_s)}–${formatSeconds(stored.end_s)} in ${who}. Set its voice, preset and params below, then Apply pending.`;
  }
  return { regions, note };
}

// -- panel requests and the media pool ----------------------------------------------------------

export type VoiceTab = 'convert' | 'batch' | 'regions' | 'voices';

export interface VoicePanelRequest {
  tab?: VoiceTab;
  /** the Convert tab takes this media-pool item as its source */
  mediaId?: string;
  /** a message for the requested tab */
  note?: string;
  /** the regions changed outside the panel: reload them */
  reloadRegions?: boolean;
}

let pendingRequest: VoicePanelRequest | null = null;

/** Fired when a request is waiting; the panel takes it (it may mount only after the request). */
export const voicePanelRequests = new Signal<void>();

/** Show the Voice panel (hidden until asked for) and hand it a request. */
export function openVoicePanel(request: VoicePanelRequest = {}): void {
  pendingRequest = { ...(pendingRequest ?? {}), ...request };
  useLayoutStore.getState().toggleVisible('voice', true);
  voicePanelRequests.emit();
  window.requestAnimationFrame(() => {
    document.querySelector<HTMLElement>('.voice-panel')?.focus({ preventScroll: true });
  });
}

export function takeVoicePanelRequest(): VoicePanelRequest | null {
  const request = pendingRequest;
  pendingRequest = null;
  return request;
}

/** A project id whose media pool gained an item (a conversion, an upload). */
export const mediaPoolChanged = new Signal<string>();

export function notifyMediaPoolChanged(projectId: string): void {
  mediaPoolChanged.emit(projectId);
}

/** The timeline range band's "Convert voice…": a pending region over the audio range with the
 *  default voice/preset, then the Voice panel on its Regions tab. */
export async function convertVoiceFromSelection(): Promise<void> {
  const { project, audioSelection } = useProjectStore.getState();
  openVoicePanel({ tab: 'regions' });
  if (!project?.id) return;
  if (!audioSelection || !(audioSelection.end_s > audioSelection.start_s)) {
    openVoicePanel({ tab: 'regions', note: 'No audio range: Shift+drag on an audio lane or the ruler first.' });
    return;
  }
  try {
    const { note } = await addRegionFromSelection(project.id, audioSelection);
    openVoicePanel({ tab: 'regions', reloadRegions: true, note });
  } catch (err) {
    openVoicePanel({ tab: 'regions', reloadRegions: true, note: `Could not add the region: ${regionsError(err)}` });
  }
}
