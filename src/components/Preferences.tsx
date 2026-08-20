import { useCallback, useEffect, useState, ReactNode } from 'react';
import {
  getAppSettings, updateAppSettings,
  getLlmStatus, ensureLlm, getLlmModels, LlmModelList, LlmStatus,
  getComfyUIStatus, listGenerationWorkflows, getComfyUIModels, ComfyModelList,
  getGpuStatus, getSystemPaths, clearVram,
} from '../hooks/api';
import {
  useAppearanceStore, ACCENT_SWATCHES, ThemeId, DensityId,
} from '../hooks/appearance';
import { COMMANDS, COMMAND_GROUPS, useCommandStore } from '../hooks/commands';
import { eventToChord, chordLabel } from '../hooks/shortcuts';

export type PrefTab = 'general' | 'models' | 'shortcuts' | 'appearance' | 'diagnostics';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  initialTab?: PrefTab;
}

const TABS: { id: PrefTab; label: string; icon: string }[] = [
  { id: 'general', label: 'General', icon: '⚙️' },
  { id: 'models', label: 'AI Models', icon: '🧠' },
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
  const { accent, theme, density, set, reset } = useAppearanceStore();

  const themes: { id: ThemeId; label: string }[] = [
    { id: 'dark', label: 'Dark' },
    { id: 'midnight', label: 'Midnight' },
    { id: 'light', label: 'Light' },
  ];
  const densities: { id: DensityId; label: string }[] = [
    { id: 'comfortable', label: 'Comfortable' },
    { id: 'compact', label: 'Compact' },
  ];

  return (
    <div className="pref-section">
      <h4>Accent colour</h4>
      <div className="swatch-row">
        {ACCENT_SWATCHES.map((s) => (
          <button key={s.value} title={s.label}
            className={`swatch ${accent.toLowerCase() === s.value.toLowerCase() ? 'active' : ''}`}
            style={{ background: s.value }} onClick={() => set({ accent: s.value })} />
        ))}
        <label className="swatch-custom" title="Custom colour">
          <input type="color" value={accent} onChange={(e) => set({ accent: e.target.value })} />
        </label>
      </div>

      <h4 style={{ marginTop: 20 }}>Theme</h4>
      <div className="seg-row">
        {themes.map((t) => (
          <button key={t.id} className={`seg ${theme === t.id ? 'active' : ''}`}
            onClick={() => set({ theme: t.id })}>{t.label}</button>
        ))}
      </div>

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
        <DiagLine label="ComfyUI bridge (127.0.0.1:8188)" ok={paths?.comfyui_connected}
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
