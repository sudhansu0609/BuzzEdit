import { useCallback, useEffect, useState, ReactNode } from 'react';
import {
  getAppSettings, updateAppSettings,
  getLlmStatus, ensureLlm, getLlmModels, LlmModelList, LlmStatus,
  getComfyUIStatus, listGenerationWorkflows, getComfyUIModels, ComfyModelList,
  getGpuStatus, getSystemPaths, clearVram, COMFY_BASE, hostPort,
  getPresentationOverrides, setPresentationOverrides, PresentationOverrideMode,
  getStockKeys, setStockKeys, StockKeysState,
  getAutoCutEditor, setAutoCutEditor, AutoCutEditorState,
  audioPreview, mediaStreamUrl,
} from '../hooks/api';
import { useProjectStore } from '../hooks/store';
import {
  useAppearanceStore, ACCENT_SWATCHES, THEMES, DensityId,
} from '../hooks/appearance';
import { COMMANDS, COMMAND_GROUPS, useCommandStore } from '../hooks/commands';
import { eventToChord, chordLabel } from '../hooks/shortcuts';

export type PrefTab = 'general' | 'models' | 'presentation' | 'shortcuts' | 'appearance' | 'diagnostics';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  initialTab?: PrefTab;
}

const TABS: { id: PrefTab; label: string; icon: string }[] = [
  { id: 'general', label: 'General', icon: '⚙️' },
  { id: 'models', label: 'AI Models', icon: '🧠' },
  { id: 'presentation', label: 'Presentation', icon: '🎬' },
  { id: 'shortcuts', label: 'Shortcuts', icon: '⌨️' },
  { id: 'appearance', label: 'Appearance', icon: '🎨' },
  { id: 'diagnostics', label: 'Diagnostics', icon: '🩺' },
];

const RESOLUTIONS = [
  { label: '4K (3840×2160)', value: '3840x2160' },
  { label: '1080p (1920×1080)', value: '1920x1080' },
  { label: '720p (1280×720)', value: '1280x720' },
  { label: 'Shorts (1080×1920)', value: '1080x1920' },
];
const FORMATS = ['mp4', 'mov', 'mkv', 'webm'];

export default function Preferences({ isOpen, onClose, initialTab = 'general' }: Props) {
  const [tab, setTab] = useState<PrefTab>(initialTab);

  useEffect(() => { if (isOpen) setTab(initialTab); }, [isOpen, initialTab]);

  if (!isOpen) return null;

  return (
    <div className="pref-overlay" onMouseDown={onClose}>
      <div className="pref-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="pref-header">
          <h3>Preferences</h3>
          <button className="btn btn-sm" onClick={onClose}>✕</button>
        </div>
        <div className="pref-body">
          <nav className="pref-tabs">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`pref-tab ${tab === t.id ? 'active' : ''}`}
                onClick={() => setTab(t.id)}
              >
                <span>{t.icon}</span> {t.label}
              </button>
            ))}
          </nav>
          <div className="pref-content">
            {tab === 'general' && <GeneralTab />}
            {tab === 'models' && <ModelsTab />}
            {tab === 'presentation' && <PresentationTab />}
            {tab === 'shortcuts' && <ShortcutsTab />}
            {tab === 'appearance' && <AppearanceTab />}
            {tab === 'diagnostics' && <DiagnosticsTab />}
          </div>
        </div>
      </div>
    </div>
  );
}

// --- General -----------------------------------------------------------------

function GeneralTab() {
  const [settings, setSettings] = useState<Record<string, any>>({});
  const [saved, setSaved] = useState(false);

  useEffect(() => { getAppSettings().then(setSettings).catch(() => {}); }, []);

  const save = useCallback(async (patch: Record<string, any>) => {
    setSettings((s) => ({ ...s, ...patch }));
    try {
      await updateAppSettings(patch);
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
    } catch { /* ignore */ }
  }, []);

  const num = (k: string, d: number) => (settings[k] ?? d);

  return (
    <div className="pref-section">
      <h4>Editing defaults <span className="pref-hint">applied to new projects</span>
        {saved && <span className="pref-saved">Saved ✓</span>}</h4>

      <Field label="Default output resolution">
        <select className="pref-input" value={settings.default_resolution ?? '1920x1080'}
          onChange={(e) => save({ default_resolution: e.target.value })}>
          {RESOLUTIONS.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
        </select>
      </Field>

      <Field label="Default output format">
        <select className="pref-input" value={settings.default_format ?? 'mp4'}
          onChange={(e) => save({ default_format: e.target.value })}>
          {FORMATS.map((f) => <option key={f} value={f}>{f.toUpperCase()}</option>)}
        </select>
      </Field>

      <Field label={`Fumble aggressiveness — ${Math.round(num('default_fumble_aggressiveness', 0.5) * 100)}%`}>
        <input type="range" min={0} max={1} step={0.05}
          value={num('default_fumble_aggressiveness', 0.5)}
          onChange={(e) => save({ default_fumble_aggressiveness: parseFloat(e.target.value) })} />
      </Field>

      <Field label={`Max pause kept — ${num('default_max_pause_seconds', 0.4).toFixed(2)}s`}>
        <input type="range" min={0.15} max={1.2} step={0.05}
          value={num('default_max_pause_seconds', 0.4)}
          onChange={(e) => save({ default_max_pause_seconds: parseFloat(e.target.value) })} />
      </Field>

      <h4 style={{ marginTop: 20 }}>Autosave</h4>
      <Field label="Autosave enabled">
        <input type="checkbox" checked={settings.auto_save_enabled ?? true}
          onChange={(e) => save({ auto_save_enabled: e.target.checked })} />
      </Field>
      <Field label={`Interval — ${num('auto_save_interval_seconds', 30)}s`}>
        <input type="range" min={10} max={120} step={5}
          value={num('auto_save_interval_seconds', 30)}
          onChange={(e) => save({ auto_save_interval_seconds: parseInt(e.target.value, 10) })} />
      </Field>
    </div>
  );
}

// --- AI Models ---------------------------------------------------------------

function ModelsTab() {
  const [llm, setLlm] = useState<LlmStatus | null>(null);
  const [models, setModels] = useState<LlmModelList | null>(null);
  const [starting, setStarting] = useState(false);
  const [comfy, setComfy] = useState<any>(null);
  const [wf, setWf] = useState<Awaited<ReturnType<typeof listGenerationWorkflows>> | null>(null);
  const [gen, setGen] = useState<Record<string, any>>({});
  const [comfyModels, setComfyModels] = useState<ComfyModelList | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    getLlmStatus().then(setLlm).catch(() => {});
    getLlmModels().then(setModels).catch(() => {});
    getComfyUIStatus().then(setComfy).catch(() => {});
    listGenerationWorkflows().then(setWf).catch(() => {});
    getAppSettings().then(setGen).catch(() => {});
    getComfyUIModels().then(setComfyModels).catch(() => {});
  }, []);

  const saveGen = useCallback(async (patch: Record<string, any>) => {
    setGen((g) => ({ ...g, ...patch }));
    await updateAppSettings(patch).catch(() => {});
    setNote('Generation settings saved.');
    setTimeout(() => setNote(null), 1500);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const chooseModel = async (id: string) => {
    setModels((m) => (m ? { ...m, selected: id } : m));
    await updateAppSettings({ llm_model: id }).catch(() => {});
    setNote('LLM model saved — used on the next AI edit.');
    setTimeout(() => setNote(null), 2000);
  };

  const startLlm = async () => {
    setStarting(true);
    try { setLlm(await ensureLlm()); await refresh(); } finally { setStarting(false); }
  };

  const chooseWorkflow = async (role: string, file: string) => {
    const key = wf?.settings_keys?.[role];
    if (!key) return;
    setWf((w) => (w ? { ...w, selected: { ...w.selected, [role]: file } } : w));
    await updateAppSettings({ [key]: file }).catch(() => {});
    setNote(`Workflow for ${role} saved.`);
    setTimeout(() => setNote(null), 2000);
  };

  const llmReady = !!llm?.server_up && !!llm?.model_loaded;

  return (
    <div className="pref-section">
      {note && <div className="pref-saved-banner">{note}</div>}

      <h4>LM Studio (LLM)</h4>
      <div className="pref-status">
        <span className={`dot ${llmReady ? 'ok' : llm?.server_up ? 'warn' : 'bad'}`} />
        {starting ? 'Starting…'
          : llmReady ? 'Server up · model loaded'
          : llm?.server_up ? 'Server up · no model loaded'
          : 'LM Studio offline'}
        <button className="btn btn-sm" onClick={startLlm} disabled={starting || llmReady}
          style={{ marginLeft: 'auto' }}>Start &amp; load</button>
      </div>

      <Field label="Model used for AI edits">
        <select className="pref-input" value={models?.selected ?? 'auto'}
          onChange={(e) => chooseModel(e.target.value)}>
          <option value="auto">Local model (auto-detect)</option>
          {/* Keep a pinned model selectable even if the server is down / it's not listed. */}
          {models && models.selected !== 'auto'
            && !models.models.some((m) => m.id === models.selected) && (
            <option value={models.selected}>{models.selected} (saved)</option>
          )}
          {models?.models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.id}{m.state === 'loaded' ? ' — loaded' : ''}{m.type === 'vlm' ? ' (vision)' : ''}
            </option>
          ))}
        </select>
      </Field>
      <p className="pref-note">
        Default is <strong>auto</strong> — the local model LM Studio already has loaded (nothing runs
        in the cloud). Pin a specific model only if you want to: the fluency pass benefits from a
        larger one (~26B removes fumbles a 7.5B misses), but larger models need more VRAM.
      </p>

      <h4 style={{ marginTop: 22 }}>ComfyUI (image / video generation)</h4>
      <div className="pref-status">
        <span className={`dot ${comfy?.connected ? 'ok' : 'warn'}`} />
        {comfy?.connected ? `Online · ${comfy?.url}` : 'Offline (generation falls back or is skipped)'}
      </div>

      {wf?.roles.map((role) => {
        const usable = wf.workflows.filter((w) => w.valid && w.roles_ok?.[role]);
        return (
          <Field key={role} label={role.replace('_', ' ')}>
            <select className="pref-input" value={wf.selected[role] ?? ''}
              onChange={(e) => chooseWorkflow(role, e.target.value)}>
              <option value="">— none —</option>
              {/* Selected file always listed even if it can't serve the role, so it's visible. */}
              {wf.selected[role] && !usable.some((w) => w.file === wf.selected[role]) && (
                <option value={wf.selected[role]!}>{wf.selected[role]} (check compatibility)</option>
              )}
              {usable.map((w) => (
                <option key={w.file} value={w.file}>{w.file} · {w.output_kind}</option>
              ))}
            </select>
          </Field>
        );
      })}
      <p className="pref-note">
        Drop an API-format workflow (ComfyUI → “Export (API)”) into the app’s <code>workflows/</code>
        folder and it appears here. Roles without a compatible workflow are skipped during the
        presentation pass.
      </p>

      <h4 style={{ marginTop: 22 }}>Image generation</h4>
      <p className="pref-note" style={{ marginTop: 0 }}>
        Applied on top of the selected B-roll workflow. Leave a field blank to keep the workflow’s
        own value.
      </p>

      <Field label="Resolution">
        <select className="pref-input"
          value={sizeToStr(gen.gen_image_size) || '1920x1080'}
          onChange={(e) => saveGen({ gen_image_size: strToSize(e.target.value) })}>
          {IMAGE_SIZES.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </Field>

      <Field label="Upscale (Lanczos resize)">
        <select className="pref-input" value={String(gen.gen_upscale ?? 1)}
          onChange={(e) => saveGen({ gen_upscale: parseFloat(e.target.value) })}>
          <option value="1">Off (1×)</option>
          <option value="1.5">1.5×</option>
          <option value="2">2×</option>
          <option value="3">3×</option>
          <option value="4">4×</option>
        </select>
      </Field>

      <Field label="Steps (blank = workflow default)">
        <input className="pref-input" type="number" min={1} max={100}
          value={gen.gen_steps ?? ''} placeholder="workflow default"
          onChange={(e) => saveGen({ gen_steps: e.target.value === '' ? null : parseInt(e.target.value, 10) })} />
      </Field>

      <Field label="CFG (blank = workflow default)">
        <input className="pref-input" type="number" min={0} max={30} step={0.5}
          value={gen.gen_cfg ?? ''} placeholder="workflow default"
          onChange={(e) => saveGen({ gen_cfg: e.target.value === '' ? null : parseFloat(e.target.value) })} />
      </Field>

      <Field label="B-roll model override">
        <select className="pref-input" value={gen.gen_broll_model ?? ''}
          onChange={(e) => saveGen({ gen_broll_model: e.target.value || null })}>
          <option value="">— use the workflow’s own model —</option>
          {comfyModels && comfyModels.diffusion_models.length > 0 && (
            <optgroup label="Diffusion models (UNET)">
              {comfyModels.diffusion_models.map((m) => <option key={m} value={m}>{m}</option>)}
            </optgroup>
          )}
          {comfyModels && comfyModels.checkpoints.length > 0 && (
            <optgroup label="Checkpoints">
              {comfyModels.checkpoints.map((m) => <option key={m} value={m}>{m}</option>)}
            </optgroup>
          )}
        </select>
      </Field>
      <p className="pref-note">
        The override only works with a model of the <em>same family</em> the workflow was built for
        (a UNET workflow like Z-Image needs a diffusion model; a checkpoint workflow needs a
        checkpoint). Leave it on “workflow’s own model” unless you know they match.
      </p>
    </div>
  );
}

const IMAGE_SIZES = ['1024x1024', '1280x720', '1344x768', '1536x864', '1920x1080', '832x1216', '1080x1920'];

function sizeToStr(v: any): string | null {
  return Array.isArray(v) && v.length === 2 ? `${v[0]}x${v[1]}` : null;
}
function strToSize(s: string): number[] {
  const [w, h] = s.split('x').map((n) => parseInt(n, 10));
  return [w, h];
}

// --- Presentation defaults & overrides (BuzzcafStudio contract) --------------

// The 18 genres presentation/genre.py styles (GENRE_STYLES keys), in the same
// order — "general" first, since that's the identity genre.
const GENRE_OPTIONS = [
  'general', 'horror', 'true_crime', 'comedy', 'gaming', 'tech', 'science_education',
  'finance', 'motivational', 'health_fitness', 'cooking', 'travel', 'devotional',
  'news', 'vlog', 'documentary', 'mystery', 'geopolitics',
];
const DENSITY_OPTIONS = ['calm', 'balanced', 'busy', 'max'] as const;
const VOICE_PRESET_OPTIONS = [
  'studio_mic', 'broadcast', 'warm_radio', 'rap_vocal', 'horror_intimate',
  'clean', 'podcast', 'light',
];
const SOURCE_OPTIONS = ['auto', 'library', 'generated', 'synth', 'off'];

const PRESENTATION_FIELDS = [
  'density', 'genre', 'genre_secondary', 'target_coverage', 'video_broll_share',
  'punch_rate_per_minute', 'text_fx_per_minute', 'zoom_depth', 'voice_preset',
  'sfx_source', 'music_source',
] as const;
type PresentationField = typeof PRESENTATION_FIELDS[number];

const PRESENTATION_DEFAULTS: Record<PresentationField, any> = {
  density: 'balanced', genre: 'general', genre_secondary: '',
  target_coverage: 0.5, video_broll_share: 0.2, punch_rate_per_minute: 1.5,
  text_fx_per_minute: 2.0, zoom_depth: 0.10, voice_preset: 'studio_mic',
  sfx_source: 'auto', music_source: 'auto',
};

function PresentationTab() {
  const { project } = useProjectStore();
  const [mode, setMode] = useState<PresentationOverrideMode>('off');
  const [locked, setLocked] = useState<Set<PresentationField>>(new Set());
  const [values, setValues] = useState<Record<PresentationField, any>>({ ...PRESENTATION_DEFAULTS });
  const [loaded, setLoaded] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [stockKeys, setStockKeysState] = useState<StockKeysState | null>(null);
  const [pexelsInput, setPexelsInput] = useState('');
  const [pixabayInput, setPixabayInput] = useState('');
  const [savingKeys, setSavingKeys] = useState(false);
  const [keysNote, setKeysNote] = useState<string | null>(null);

  useEffect(() => {
    getStockKeys().then(setStockKeysState).catch(() => {});
    getAutoCutEditor().then(setEditorState).catch(() => {});
  }, []);

  // The AI editor (auto-cut planner): which planner cuts, and the Claude proxy key it needs.
  const [editorState, setEditorState] = useState<AutoCutEditorState | null>(null);
  const [editorKeyInput, setEditorKeyInput] = useState('');
  const [editorUrlInput, setEditorUrlInput] = useState('');
  const saveEditor = async (body: { planner?: 'editor' | 'classic'; base_url?: string; api_key?: string }) => {
    setSavingKeys(true);
    try {
      setEditorState(await setAutoCutEditor(body));
      setEditorKeyInput('');
      setEditorUrlInput('');
      setKeysNote('Saved.');
      setTimeout(() => setKeysNote(null), 1500);
    } catch (e: any) {
      setKeysNote(`Save failed: ${e.message}`);
    } finally {
      setSavingKeys(false);
    }
  };

  const saveStockKey = async (field: 'pexels_api_key' | 'pixabay_api_key', value: string) => {
    if (!value) return;
    setSavingKeys(true);
    try {
      const next = await setStockKeys({ [field]: value });
      setStockKeysState(next);
      if (field === 'pexels_api_key') setPexelsInput(''); else setPixabayInput('');
      setKeysNote('Saved.');
      setTimeout(() => setKeysNote(null), 1500);
    } catch (e: any) {
      setKeysNote(`Save failed: ${e.message}`);
    } finally {
      setSavingKeys(false);
    }
  };

  useEffect(() => {
    getPresentationOverrides().then((state) => {
      const saved = state.presentation_overrides || {};
      const nextValues = { ...PRESENTATION_DEFAULTS };
      const nextLocked = new Set<PresentationField>();
      for (const field of PRESENTATION_FIELDS) {
        if (field in saved) {
          nextValues[field] = saved[field];
          nextLocked.add(field);
        }
      }
      setMode(state.override_mode || 'off');
      setValues(nextValues);
      setLocked(nextLocked);
      setLoaded(true);
    }).catch(() => setLoaded(true));
  }, []);

  const persist = useCallback(async (
    nextValues: Record<PresentationField, any>,
    nextLocked: Set<PresentationField>,
    nextMode: PresentationOverrideMode,
  ) => {
    const overrides: Record<string, any> = {};
    for (const field of PRESENTATION_FIELDS) {
      if (nextMode === 'all' || (nextMode === 'fields' && nextLocked.has(field))) {
        overrides[field] = nextValues[field];
      }
    }
    try {
      await setPresentationOverrides(overrides, nextMode);
      setNote('Saved.');
      setTimeout(() => setNote(null), 1500);
    } catch (e: any) {
      setNote(`Save failed: ${e.message}`);
    }
  }, []);

  const setValue = (field: PresentationField, value: any) => {
    const next = { ...values, [field]: value };
    setValues(next);
    persist(next, locked, mode);
  };

  const toggleLock = (field: PresentationField) => {
    const next = new Set(locked);
    if (next.has(field)) next.delete(field); else next.add(field);
    setLocked(next);
    persist(values, next, mode);
  };

  const changeMode = (next: PresentationOverrideMode) => {
    setMode(next);
    persist(values, locked, next);
  };

  const runPreview = async () => {
    if (!project) return;
    setPreviewing(true);
    setPreviewError(null);
    setPreviewUrl(null);
    try {
      const res = await audioPreview(project.id, values.voice_preset);
      setPreviewUrl(mediaStreamUrl(res.path));
    } catch (e: any) {
      setPreviewError(e.message || 'Preview failed');
    } finally {
      setPreviewing(false);
    }
  };

  if (!loaded) return <div className="pref-section">Loading…</div>;

  return (
    <div className="pref-section">
      {note && <div className="pref-saved-banner">{note}</div>}
      <h4>Presentation defaults &amp; overrides
        <span className="pref-hint">for BuzzcafStudio-enqueued jobs</span>
      </h4>
      <p className="pref-note">
        BuzzcafStudio sends its own density, genre and voice settings with every overnight job.
        <strong> Off</strong> leaves them alone; <strong>Fields</strong> overrides only the fields
        locked below; <strong>All</strong> overrides every field on this page, whatever Studio sent.
      </p>

      <Field label="Override mode">
        <div className="seg-row">
          {(['off', 'fields', 'all'] as PresentationOverrideMode[]).map((m) => (
            <button key={m} type="button" className={`seg ${mode === m ? 'active' : ''}`}
              onClick={() => changeMode(m)}>{m}</button>
          ))}
        </div>
      </Field>

      <LockableField label="Density" field="density" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <select className="pref-input" value={values.density}
          onChange={(e) => setValue('density', e.target.value)}>
          {DENSITY_OPTIONS.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
      </LockableField>

      <LockableField label="Genre" field="genre" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <select className="pref-input" value={values.genre}
          onChange={(e) => setValue('genre', e.target.value)}>
          {GENRE_OPTIONS.map((g) => <option key={g} value={g}>{g}</option>)}
        </select>
      </LockableField>

      <LockableField label="Secondary genre" field="genre_secondary" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <select className="pref-input" value={values.genre_secondary}
          onChange={(e) => setValue('genre_secondary', e.target.value)}>
          <option value="">— none —</option>
          {GENRE_OPTIONS.map((g) => <option key={g} value={g}>{g}</option>)}
        </select>
      </LockableField>

      <LockableField label={`Coverage — ${Math.round(values.target_coverage * 100)}%`}
        field="target_coverage" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <input type="range" min={0} max={0.8} step={0.01} value={values.target_coverage}
          onChange={(e) => setValue('target_coverage', parseFloat(e.target.value))} />
      </LockableField>

      <LockableField label={`Video share — ${Math.round(values.video_broll_share * 100)}%`}
        field="video_broll_share" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <input type="range" min={0} max={1} step={0.01} value={values.video_broll_share}
          onChange={(e) => setValue('video_broll_share', parseFloat(e.target.value))} />
      </LockableField>

      <LockableField label={`Punch-ins / min — ${Number(values.punch_rate_per_minute).toFixed(1)}`}
        field="punch_rate_per_minute" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <input type="range" min={0} max={5} step={0.1} value={values.punch_rate_per_minute}
          onChange={(e) => setValue('punch_rate_per_minute', parseFloat(e.target.value))} />
      </LockableField>

      <LockableField label={`Text FX / min — ${Number(values.text_fx_per_minute).toFixed(1)}`}
        field="text_fx_per_minute" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <input type="range" min={0} max={6} step={0.1} value={values.text_fx_per_minute}
          onChange={(e) => setValue('text_fx_per_minute', parseFloat(e.target.value))} />
      </LockableField>

      <LockableField label={`Zoom depth — ${Number(values.zoom_depth).toFixed(2)}`}
        field="zoom_depth" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <input type="range" min={0.04} max={0.3} step={0.01} value={values.zoom_depth}
          onChange={(e) => setValue('zoom_depth', parseFloat(e.target.value))} />
      </LockableField>

      <LockableField label="Voice preset" field="voice_preset" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flex: 1 }}>
          <select className="pref-input" value={values.voice_preset}
            onChange={(e) => setValue('voice_preset', e.target.value)}>
            {VOICE_PRESET_OPTIONS.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
          <button className="btn btn-sm" disabled={!project || previewing} onClick={runPreview}
            title={project ? "Render 20s of this project's voice through the preset"
                            : 'Open a project to preview'}>
            {previewing ? 'Rendering…' : '▶ Preview 20s'}
          </button>
        </div>
      </LockableField>
      {previewUrl && <audio controls src={previewUrl} style={{ width: '100%', marginTop: 6 }} />}
      {previewError && (
        <p className="pref-note" style={{ color: 'var(--danger)' }}>{previewError}</p>
      )}

      <LockableField label="SFX source" field="sfx_source" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <select className="pref-input" value={values.sfx_source}
          onChange={(e) => setValue('sfx_source', e.target.value)}>
          {SOURCE_OPTIONS.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </LockableField>

      <LockableField label="Music source" field="music_source" mode={mode} locked={locked} onToggleLock={toggleLock}>
        <select className="pref-input" value={values.music_source}
          onChange={(e) => setValue('music_source', e.target.value)}>
          {SOURCE_OPTIONS.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </LockableField>

      <h4>AI auto-cut</h4>
      <p className="pref-note">
        The AI editor reads the whole recording three times through your Claude proxy
        ({editorState?.model ?? 'claude-opus-5-5'}, {editorState?.effort ?? 'medium'} effort) and cuts only
        where all three agree. Without a proxy key, cuts fall back to the classic planner.
      </p>
      <Field label="Planner">
        <select className="pref-input" value={editorState?.planner ?? 'editor'} disabled={savingKeys}
          onChange={(e) => saveEditor({ planner: e.target.value as 'editor' | 'classic' })}>
          <option value="editor">AI editor (Claude)</option>
          <option value="classic">Classic (rules + local model)</option>
        </select>
      </Field>
      <Field label="Claude proxy URL">
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flex: 1 }}>
          <input type="text" className="pref-input" value={editorUrlInput}
            placeholder={editorState?.base_url ?? 'http://127.0.0.1:8787/v1'}
            onChange={(e) => setEditorUrlInput(e.target.value)} />
          <button className="btn btn-sm" disabled={!editorUrlInput || savingKeys}
            onClick={() => saveEditor({ base_url: editorUrlInput })}>
            Save
          </button>
        </div>
      </Field>
      <Field label="Claude proxy key">
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flex: 1 }}>
          <input type="password" className="pref-input" value={editorKeyInput}
            placeholder={editorState?.key_configured
              ? (editorState.key_source === 'env' ? 'Set by CLAUDE_API_KEY' : 'Saved')
              : 'Not set: cuts use the classic planner'}
            onChange={(e) => setEditorKeyInput(e.target.value)} />
          <button className="btn btn-sm" disabled={!editorKeyInput || savingKeys}
            onClick={() => saveEditor({ api_key: editorKeyInput })}>
            Save
          </button>
        </div>
      </Field>

      <h4>Free stock fallback</h4>
      <p className="pref-note">
        Free sources only; used only when ComfyUI can&apos;t make a visual and
        &quot;Allow free stock&quot; is on.
      </p>
      {keysNote && <div className="pref-saved-banner">{keysNote}</div>}

      <Field label="Pexels API key">
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flex: 1 }}>
          <input type="password" className="pref-input" value={pexelsInput}
            placeholder={stockKeys?.pexels_api_key_set ? `Saved: ${stockKeys.pexels_api_key}` : 'Not set'}
            onChange={(e) => setPexelsInput(e.target.value)} />
          <button className="btn btn-sm" disabled={!pexelsInput || savingKeys}
            onClick={() => saveStockKey('pexels_api_key', pexelsInput)}>
            Save
          </button>
        </div>
      </Field>

      <Field label="Pixabay API key">
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flex: 1 }}>
          <input type="password" className="pref-input" value={pixabayInput}
            placeholder={stockKeys?.pixabay_api_key_set ? `Saved: ${stockKeys.pixabay_api_key}` : 'Not set'}
            onChange={(e) => setPixabayInput(e.target.value)} />
          <button className="btn btn-sm" disabled={!pixabayInput || savingKeys}
            onClick={() => saveStockKey('pixabay_api_key', pixabayInput)}>
            Save
          </button>
        </div>
      </Field>
    </div>
  );
}

function LockableField({ label, field, mode, locked, onToggleLock, children }: {
  label: string;
  field: PresentationField;
  mode: PresentationOverrideMode;
  locked: Set<PresentationField>;
  onToggleLock: (field: PresentationField) => void;
  children: ReactNode;
}) {
  const isLocked = locked.has(field);
  const active = mode === 'all' || (mode === 'fields' && isLocked);
  return (
    <label className="pref-field">
      <span className="pref-field-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        {label}
        {mode === 'fields' && (
          <button
            type="button"
            className={`lock-toggle ${isLocked ? 'active' : ''}`}
            title={isLocked ? 'BuzzEdit wins on this field' : 'BuzzcafStudio wins on this field'}
            onClick={(e) => { e.preventDefault(); onToggleLock(field); }}
          >
            {isLocked ? '🔒 BuzzEdit wins' : '🔓 Studio wins'}
          </button>
        )}
      </span>
      <span className="pref-field-control" style={{ opacity: mode === 'off' ? 0.6 : active ? 1 : 0.85 }}>
        {children}
      </span>
    </label>
  );
}

// --- Shortcuts ---------------------------------------------------------------

function ShortcutsTab() {
  const chordFor = useCommandStore((s) => s.chordFor);
  const setBinding = useCommandStore((s) => s.setBinding);
  const resetBinding = useCommandStore((s) => s.resetBinding);
  const resetAll = useCommandStore((s) => s.resetAll);
  useCommandStore((s) => s.keymap); // re-render on change
  const [capturing, setCapturing] = useState<string | null>(null);

  useEffect(() => {
    if (!capturing) return;
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === 'Escape') { setCapturing(null); return; }
      if (['Control', 'Alt', 'Shift', 'Meta'].includes(e.key)) return; // wait for the real key
      setBinding(capturing, eventToChord(e));
      setCapturing(null);
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [capturing, setBinding]);

  return (
    <div className="pref-section">
      <h4>Keyboard shortcuts
        <button className="btn btn-sm" style={{ marginLeft: 'auto' }} onClick={resetAll}>
          Reset all
        </button>
      </h4>
      <p className="pref-note">Click a shortcut to record a new key combination. Esc cancels.</p>
      {COMMAND_GROUPS.map((group) => (
        <div key={group} className="shortcut-group">
          <div className="shortcut-group-title">{group}</div>
          {COMMANDS.filter((c) => c.group === group).map((cmd) => {
            const chord = chordFor(cmd.id);
            return (
              <div key={cmd.id} className="shortcut-row">
                <span className="shortcut-label">{cmd.label}</span>
                <button
                  className={`shortcut-chord ${capturing === cmd.id ? 'capturing' : ''}`}
                  onClick={() => setCapturing(cmd.id)}
                >
                  {capturing === cmd.id ? 'Press keys…' : chordLabel(chord) || 'Unassigned'}
                </button>
                <button className="shortcut-reset" title="Reset to default"
                  onClick={() => resetBinding(cmd.id)}>↺</button>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

// --- Appearance --------------------------------------------------------------

function AppearanceTab() {
  const { accent, theme, density, accentFollowsTheme, set, reset } = useAppearanceStore();

  const densities: { id: DensityId; label: string }[] = [
    { id: 'comfortable', label: 'Comfortable' },
    { id: 'compact', label: 'Compact' },
  ];
  const dark = THEMES.filter((t) => t.mode === 'dark');
  const light = THEMES.filter((t) => t.mode === 'light');

  // The card is a miniature of the real thing — page, panel and a line of text —
  // rather than three abstract dots, because what you are choosing between is
  // how much contrast the editor has, and that is only legible as a layout.
  const card = (t: typeof THEMES[number]) => (
    <button key={t.id} title={t.note}
      className={`theme-card ${theme === t.id ? 'active' : ''}`}
      onClick={() => set({ theme: t.id })}>
      <span className="theme-swatch" style={{ background: t.preview[0] }}>
        <span className="theme-swatch-panel" style={{ background: t.preview[1] }}>
          <span className="theme-swatch-line" style={{ background: t.preview[2] }} />
          <span className="theme-swatch-line short" style={{ background: t.preview[2] }} />
        </span>
        <span className="theme-swatch-dot" style={{ background: t.accent }} />
      </span>
      <span className="theme-card-name">{t.label}</span>
      <span className="theme-card-note">{t.note}</span>
    </button>
  );

  return (
    <div className="pref-section">
      <h4>Theme</h4>
      <div className="theme-group-label">Dark</div>
      <div className="theme-grid">{dark.map(card)}</div>
      <div className="theme-group-label" style={{ marginTop: 12 }}>Light</div>
      <div className="theme-grid">{light.map(card)}</div>

      <h4 style={{ marginTop: 20 }}>Accent colour</h4>
      <div className="swatch-row">
        {ACCENT_SWATCHES.map((s) => (
          <button key={s.value} title={s.label}
            className={`swatch ${!accentFollowsTheme && accent.toLowerCase() === s.value.toLowerCase() ? 'active' : ''}`}
            style={{ background: s.value }} onClick={() => set({ accent: s.value })} />
        ))}
        <label className="swatch-custom" title="Custom colour">
          <input type="color" value={accent} onChange={(e) => set({ accent: e.target.value })} />
        </label>
      </div>
      <label className="pref-inline-check"
        title="Each palette was drawn around its own accent — Dracula's violet, Nord's frost blue. Untick to keep one colour across every theme.">
        <input type="checkbox" checked={accentFollowsTheme}
          onChange={(e) => set({ accentFollowsTheme: e.target.checked })} />
        Let the theme choose the accent
      </label>

      <h4 style={{ marginTop: 20 }}>Density</h4>
      <div className="seg-row">
        {densities.map((d) => (
          <button key={d.id} className={`seg ${density === d.id ? 'active' : ''}`}
            onClick={() => set({ density: d.id })}>{d.label}</button>
        ))}
      </div>

      <button className="btn btn-sm" style={{ marginTop: 20 }} onClick={reset}>
        Reset appearance
      </button>
    </div>
  );
}

// --- Diagnostics (relocated from SetupWizard) --------------------------------

function DiagnosticsTab() {
  const [gpu, setGpu] = useState<any>(null);
  const [paths, setPaths] = useState<any>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchDiag = useCallback(async () => {
    setLoading(true);
    try {
      const [g, p] = await Promise.all([getGpuStatus(), getSystemPaths()]);
      setGpu(g); setPaths(p);
    } catch { /* ignore */ } finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchDiag(); }, [fetchDiag]);

  const flush = async () => {
    try {
      setMsg('Flushing CUDA cache…');
      const res = await clearVram();
      setGpu(res.gpu);
      setMsg('VRAM cache cleared.');
      setTimeout(() => setMsg(null), 2500);
    } catch (e: any) { setMsg(`Clear VRAM failed: ${e.message}`); }
  };

  return (
    <div className="pref-section">
      {msg && <div className="pref-saved-banner">{msg}</div>}
      <div className="diag-card">
        <div className="diag-row">
          <span style={{ fontWeight: 600 }}>🎮 {gpu?.device_name || 'Detecting GPU…'}</span>
          <span className="text-xs text-muted">
            {gpu?.free_vram_mb || 0} MB free / {gpu?.total_vram_mb || 0} MB
          </span>
        </div>
        {gpu && gpu.total_vram_mb > 0 && (
          <div className="diag-meter">
            <div style={{ width: `${Math.min(100, (gpu.allocated_vram_mb / gpu.total_vram_mb) * 100)}%` }} />
          </div>
        )}
        <div className="diag-row">
          <span className="text-xs text-muted">CUDA memory</span>
          <button className="btn btn-sm btn-secondary" onClick={flush}>🧹 Flush VRAM</button>
        </div>
      </div>

      <div className="diag-card">
        <h4 style={{ margin: '0 0 8px' }}>Services</h4>
        <DiagLine label="FFmpeg (NVENC)" ok={paths?.ffmpeg_available}
          okText="Available" badText="Not found" />
        <DiagLine label={`ComfyUI bridge${COMFY_BASE ? ` (${hostPort(COMFY_BASE)})` : ''}`} ok={paths?.comfyui_connected}
          okText="Online" badText="Offline (fallback)" warn />
        <DiagLine label="ComfyUI input folder" ok={paths?.comfyui_input_dir_exists}
          okText="Exists" badText="Missing" />
      </div>

      <button className="btn btn-secondary" onClick={fetchDiag} disabled={loading}>🔄 Refresh</button>
    </div>
  );
}

function DiagLine({ label, ok, okText, badText, warn }:
  { label: string; ok?: boolean; okText: string; badText: string; warn?: boolean }) {
  return (
    <div className="diag-row" style={{ fontSize: 13 }}>
      <span>{label}</span>
      <span style={{ color: ok ? '#10b981' : warn ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>
        {ok ? `✓ ${okText}` : `${warn ? '⚠' : '✕'} ${badText}`}
      </span>
    </div>
  );
}

// --- shared ------------------------------------------------------------------

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="pref-field">
      <span className="pref-field-label">{label}</span>
      <span className="pref-field-control">{children}</span>
    </label>
  );
}
