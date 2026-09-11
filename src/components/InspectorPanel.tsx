import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import {
  addTextClip, applyIntro, clearCaptions, clearIntro, generateCaptions,
  listFonts, listPresets, setChroma, setColor, setTransform, updateTextClip,
  analyzeStyle, applyStyleProfile, deleteStyleProfile, listStyleProfiles,
  setTransition, addEffect, updateEffect, removeEffect, setAspectBars,
  useElectron, MEDIA_FILTERS, AtmosphereEffect,
  ClipChroma, ClipColor, ClipTransform, ColorWheel, FontFamily, PresetCatalogue,
  StyleProfile, StyleApplyParts, TextStyle,
} from '../hooks/api';
import {
  ColorField, PresetChips, Row, Section, Slider, toHex,
} from './controls';

const DEFAULT_TRANSFORM: ClipTransform = {
  crop_left: 0, crop_top: 0, crop_right: 0, crop_bottom: 0,
  scale: 1, pos_x: 0, pos_y: 0, rotation: 0, opacity: 1,
  flip_h: false, flip_v: false,
  scale_end: null, pos_x_end: null, pos_y_end: null,
};

const NEUTRAL_WHEEL: ColorWheel = { r: 0, g: 0, b: 0 };

const DEFAULT_COLOR: ClipColor = {
  preset: null,
  exposure: 0, brightness: 0, contrast: 1, gamma: 1,
  temperature: 0, tint: 0, shadows: 0, highlights: 0,
  lift: NEUTRAL_WHEEL, midtones: NEUTRAL_WHEEL, gain: NEUTRAL_WHEEL,
  saturation: 1, vibrance: 0, hue: 0,
  lut_file: null, lut_strength: 1,
  sharpen: 0, denoise: 0, vignette: 0, fade_in: 0, fade_out: 0,
};

const DEFAULT_CHROMA: ClipChroma = {
  enabled: true, key_type: 'chroma', color: '#00ff00',
  similarity: 0.18, blend: 0.08, choke: 0, feather: 0,
  spill: 0, spill_expand: 0, show_matte: false,
};

/** The two screens anyone actually shoots against. */
const KEY_PRESETS: { id: string; label: string; color: string }[] = [
  { id: 'green', label: 'Green screen', color: '#00ff00' },
  { id: 'blue', label: 'Blue screen', color: '#0000ff' },
];

// -- small controls ---------------------------------------------------------

/**
 * One colour-balance wheel, as three sliders.
 *
 * A round trackpad looks more like a grading suite, but it can only express two
 * of the three axes at once and it is far harder to set an exact value with.
 * Three labelled sliders say what they do and can be typed at.
 */
function Wheel({ label, value, onChange, hint }: {
  label: string; value: ColorWheel; onChange: (v: Partial<ColorWheel>) => void; hint?: string;
}) {
  const channels: { key: keyof ColorWheel; name: string; low: string; high: string }[] = [
    { key: 'r', name: 'R', low: 'cyan', high: 'red' },
    { key: 'g', name: 'G', low: 'magenta', high: 'green' },
    { key: 'b', name: 'B', low: 'yellow', high: 'blue' },
  ];
  const neutral = value.r === 0 && value.g === 0 && value.b === 0;
  return (
    <div className="insp-wheel" title={hint}>
      <div className="insp-wheel-head">
        <span className="insp-label">{label}</span>
        {!neutral && (
          <button className="btn btn-xs"
            onClick={() => onChange({ r: 0, g: 0, b: 0 })} title="Back to neutral">Reset</button>
        )}
      </div>
      {channels.map((c) => (
        <label key={c.key} className={`insp-row wheel-${c.key}`}
          title={`${c.low} ← → ${c.high}`}>
          <span className="insp-label">{c.name}</span>
          <div className="insp-control">
            <input type="range" min={-1} max={1} step={0.01} value={value[c.key]}
              onChange={(e) => onChange({ [c.key]: parseFloat(e.target.value) } as Partial<ColorWheel>)} />
            <span className="insp-value">{value[c.key].toFixed(2)}</span>
          </div>
        </label>
      ))}
    </div>
  );
}

/** How much a measured signal can be trusted — shown so a guess never reads as a fact. */
function Confidence({ level }: { level?: string }) {
  if (!level || level === 'none') return null;
  return <span className={`insp-confidence ${level}`}>{level}</span>;
}

function StyleSummary({ profile }: { profile: StyleProfile }) {
  const { pacing, rhythm, motion, look, captions, confidence } = profile;
  const rows: [string, React.ReactNode, string | undefined][] = [
    ['Pace', `${pacing.cuts_per_minute.toFixed(1)} cuts/min · ${pacing.median_shot_seconds.toFixed(1)}s shots`, confidence.pacing],
    ['Rhythm', rhythm.cuts_to_music
      ? `cut to music · ${rhythm.bpm.toFixed(0)} BPM`
      : rhythm.bpm > 0 ? `${rhythm.bpm.toFixed(0)} BPM, not cut to it` : 'no musical pulse', confidence.rhythm],
    ['Movement', motion.zoom_share > 0
      ? `${Math.round(motion.zoom_share * 100)}% of shots zoom ~${Math.round(motion.mean_zoom_ratio * 100)}%`
      : 'locked off', confidence.motion],
    ['Look', look.description, confidence.look],
    ['Captions', captions.present
      ? `at ${Math.round(((captions.pos_y + 1) / 2) * 100)}% height${captions.boxed ? ', boxed' : ''}`
      : 'none detected', confidence.captions],
  ];
  return (
    <div className="insp-style-summary">
      {rows.map(([label, value, level]) => (
        <div key={label} className="insp-style-row">
          <span className="insp-label">{label}</span>
          <span className="insp-style-value">{value}</span>
          <Confidence level={level} />
        </div>
      ))}
      {profile.notes.map((note) => (
        <div key={note} className="insp-style-note text-xs">{note}</div>
      ))}
    </div>
  );
}

// -- panel ------------------------------------------------------------------

export default function InspectorPanel() {
  const { project, updateProject, selectedClipId, setError, currentTime } = useProjectStore();
  const [fonts, setFonts] = useState<FontFamily[]>([]);
  const [presets, setPresets] = useState<PresetCatalogue | null>(null);
  const [busy, setBusy] = useState(false);
  const [captionPreset, setCaptionPreset] = useState('youtube_shorts');
  const [introPreset, setIntroPreset] = useState('title_card');
  const [introTitle, setIntroTitle] = useState('');
  const [introSubtitle, setIntroSubtitle] = useState('');
  const [styles, setStyles] = useState<StyleProfile[]>([]);
  const [styleId, setStyleId] = useState<string>('');
  const [styleParts, setStyleParts] = useState<Required<StyleApplyParts>>({
    look: true, motion: true, captions: true, transitions: true,
  });
  const [styleReport, setStyleReport] = useState<string[] | null>(null);
  const { pickFile } = useElectron();

  const projectId = project?.id;
  const timeline = project?.timeline as any;
  const fps = timeline?.fps_num ? timeline.fps_num / (timeline.fps_den || 1) : 30;

  const item = useMemo(
    () => (timeline?.items || []).find((i: any) => i.id === selectedClipId) || null,
    [timeline, selectedClipId],
  );
  const isText = item?.kind === 'text';
  const isAdjustment = item?.kind === 'adjustment';
  const targetLabel = item
    ? `${item.kind === 'text' ? 'Text' : item.kind === 'compound' ? 'Compound'
        : item.kind === 'adjustment' ? 'Adjustment layer' : 'Clip'} · ${item.track}`
    : 'Whole program';

  useEffect(() => {
    listFonts().then((r) => setFonts(r.families)).catch(() => setFonts([]));
    listPresets().then(setPresets).catch(() => setPresets(null));
    refreshStyles();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshStyles = useCallback(async () => {
    try {
      const res = await listStyleProfiles();
      setStyles(res.profiles || []);
      setStyleId((current) => current || res.profiles?.[0]?.id || '');
    } catch {
      setStyles([]);
    }
  }, []);

  useEffect(() => {
    if (!introTitle && project?.name) setIntroTitle(project.name);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project?.name]);

  const run = useCallback(async (fn: () => Promise<any>) => {
    setBusy(true);
    try {
      const res = await fn();
      if (res?.timeline) updateProject({ timeline: res.timeline });
      return res;
    } catch (err: any) {
      setError(err.message || 'Inspector action failed');
    } finally {
      setBusy(false);
    }
  }, [updateProject, setError]);

  // Sliders fire continuously; mirror the change locally for instant feedback and
  // push the accumulated patch to the backend once the user settles.
  const [draftTransform, setDraftTransform] = useState<ClipTransform | null>(null);
  const [draftColor, setDraftColor] = useState<ClipColor | null>(null);
  const [draftChroma, setDraftChroma] = useState<ClipChroma | null>(null);
  const [draftStyle, setDraftStyle] = useState<TextStyle | null>(null);
  const [draftContent, setDraftContent] = useState<string | null>(null);
  const timer = useRef<any>(null);

  useEffect(() => {
    setDraftTransform(null); setDraftColor(null); setDraftChroma(null);
    setDraftStyle(null); setDraftContent(null);
  }, [selectedClipId]);

  const storedTransform: ClipTransform = {
    ...DEFAULT_TRANSFORM,
    ...((item ? item.transform : timeline?.master_transform) || {}),
  };
  const storedColor: ClipColor = {
    ...DEFAULT_COLOR,
    ...((item ? item.color : timeline?.master_color) || {}),
  };
  const storedStyle: TextStyle | null = isText ? { ...(item.text?.style || {}) } as TextStyle : null;

  const storedChroma: ClipChroma = { ...DEFAULT_CHROMA, ...((item?.chroma) || {}) };

  const transform = draftTransform ?? storedTransform;
  const color = draftColor ?? storedColor;
  const chroma = draftChroma ?? storedChroma;
  const style = draftStyle ?? storedStyle;
  // Keying only makes sense on an overlay: V1 is the programme track and there
  // is nothing behind it for the transparency to reveal. The backend refuses it
  // too, but a control that errors when you touch it is worse than no control.
  const canKey = !!item && !isText && !isAdjustment
    && /^V\d+$/i.test(item.track || '') && item.track.toUpperCase() !== 'V1';
  const keyed = !!item?.chroma?.enabled;

  const schedule = useCallback((fn: () => Promise<any>) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => run(fn), 220);
  }, [run]);

  const patchTransform = useCallback((updates: Partial<ClipTransform>) => {
    if (!projectId) return;
    const next = { ...transform, ...updates };
    setDraftTransform(next);
    schedule(() => setTransform(projectId, selectedClipId, updates));
  }, [projectId, selectedClipId, transform, schedule]);

  const patchColor = useCallback((updates: Partial<ClipColor>, preset?: string | null) => {
    if (!projectId) return;
    setDraftColor({ ...color, ...updates, ...(preset !== undefined ? { preset } : {}) });
    schedule(() => setColor(projectId, selectedClipId, updates, preset));
  }, [projectId, selectedClipId, color, schedule]);

  const patchChroma = useCallback((updates: Partial<ClipChroma>) => {
    if (!projectId || !selectedClipId) return;
    setDraftChroma({ ...chroma, ...updates });
    schedule(() => setChroma(projectId, selectedClipId, updates));
  }, [projectId, selectedClipId, chroma, schedule]);

  const patchStyle = useCallback((updates: Partial<TextStyle>) => {
    if (!projectId || !selectedClipId || !style) return;
    setDraftStyle({ ...style, ...updates });
    schedule(() => updateTextClip(projectId, selectedClipId, { style: updates }));
  }, [projectId, selectedClipId, style, schedule]);

  const applyColorPreset = useCallback((id: string) => {
    if (!projectId) return;
    setDraftColor(null);
    run(() => setColor(projectId, selectedClipId, {}, id === 'none' ? null : id, id === 'none'));
  }, [projectId, selectedClipId, run]);

  const applyTextPreset = useCallback((id: string) => {
    if (!projectId || !selectedClipId) return;
    setDraftStyle(null);
    run(() => updateTextClip(projectId, selectedClipId, { preset: id }));
  }, [projectId, selectedClipId, run]);

  const resetTransform = () => {
    if (!projectId) return;
    setDraftTransform(null);
    run(() => setTransform(projectId, selectedClipId, {}, true));
  };

  const kenBurns = (from: number, to: number) =>
    patchTransform({ scale: from, scale_end: to, pos_x: transform.pos_x, pos_y: transform.pos_y });

  const handleAnalyzeReference = useCallback(async () => {
    const path = await pickFile(MEDIA_FILTERS);
    if (!path) {
      if (!window.electronAPI) setError('Picking a reference video needs the desktop app.');
      return;
    }
    setBusy(true);
    setStyleReport(null);
    try {
      const res = await analyzeStyle(path);
      await refreshStyles();
      setStyleId(res.profile.id);
    } catch (err: any) {
      setError(err.message || 'Could not analyse that video');
    } finally {
      setBusy(false);
    }
  }, [pickFile, refreshStyles, setError]);

  /** Turn the backend's per-part report into plain sentences. */
  const handleApplyStyle = useCallback(async () => {
    if (!projectId || !styleId) return;
    setBusy(true);
    setStyleReport(null);
    try {
      const res = await applyStyleProfile(projectId, styleId, styleParts);
      if (res.timeline) updateProject({ timeline: res.timeline });
      const report = res.report || {};
      const lines: string[] = [];
      if (report.look) {
        lines.push(report.look.applied
          ? `Look matched: ${report.look.from} → ${report.look.to}`
          : `Look skipped — ${report.look.reason}`);
      }
      if (report.motion) {
        lines.push(report.motion.applied
          ? `${report.motion.moves} zoom moves placed to match ${report.motion.target_cuts_per_minute} cuts/min`
          : `Movement skipped — ${report.motion.reason}`);
      }
      if (report.transitions) {
        lines.push(report.transitions.applied
          ? `Programme fades set to ${report.transitions.fade_seconds}s`
          : `Transitions skipped — ${report.transitions.reason}`);
      }
      if (report.captions) {
        lines.push(report.captions.applied
          ? 'Caption placement saved — press Generate captions to use it'
          : `Captions skipped — ${report.captions.reason}`);
      }
      setStyleReport(lines);
    } catch (err: any) {
      setError(err.message || 'Could not apply that style');
    } finally {
      setBusy(false);
    }
  }, [projectId, styleId, styleParts, updateProject, setError]);

  const handleAddText = () =>
    projectId && run(() => addTextClip(projectId, {
      content: 'New text',
      timelineStartFrame: Math.round(currentTime * fps),
      durationSeconds: 3,
      preset: 'title_bold',
    }));

  if (!project) return null;
  if (!timeline) {
    return (
      <div className="inspector empty-hint">
        No timeline yet — run <strong>Transcribe</strong> first, then select a clip to style it.
      </div>
    );
  }

  const selectedStyle = styles.find((s) => s.id === styleId) || null;
  // An adjustment layer owns its effects; otherwise the panel edits the programme's.
  const effectOwner = isAdjustment ? selectedClipId : null;
  const effects: AtmosphereEffect[] = (isAdjustment ? item.atmosphere : timeline?.effects) ?? [];
  // A clip's own transition wins; otherwise the panel edits the programme default.
  const activeTransition =
    (item && !isText && !isAdjustment ? item.transition : timeline?.default_transition) || null;
  const transitionActive = presets?.transition.find(
    (p) => (p as any).type === activeTransition?.type
       && Math.abs(((p as any).duration ?? 0) - (activeTransition?.duration ?? -1)) < 0.001,
  )?.id ?? (activeTransition ? undefined : 'cut');
  const aspectActive = presets?.aspect.find((a) => {
    const ratio = (a as any).ratio ?? null;
    if (!timeline?.aspect_bars) return ratio === null;
    return ratio != null && Math.abs(ratio - timeline.aspect_bars) < 0.02;
  })?.id;
  const animatedZoom = transform.scale_end != null && transform.scale_end !== transform.scale;
  const animatedPan =
    (transform.pos_x_end != null && transform.pos_x_end !== transform.pos_x) ||
    (transform.pos_y_end != null && transform.pos_y_end !== transform.pos_y);

  return (
    <div className={`inspector ${busy ? 'busy' : ''}`}>
      <div className="insp-target">
        <span className="insp-target-name">{targetLabel}</span>
        <span className="text-xs text-muted">
          {item ? (item.label || item.id.slice(-6)) : 'Nothing selected — edits apply to the whole video'}
        </span>
      </div>

      {isAdjustment && (
        <div className="insp-note">
          Everything set below is applied to <strong>every layer beneath {item.track}</strong>,
          for as long as this clip runs. Move it to a higher track to reach more layers,
          or trim it to change how long the look lasts.
        </div>
      )}

      {isText && style && (
        <Section title="Text">
          <textarea className="insp-textarea" rows={3}
            value={draftContent ?? item.text?.content ?? ''}
            onChange={(e) => {
              const next = e.target.value;
              setDraftContent(next);
              if (projectId) schedule(() => updateTextClip(projectId, selectedClipId!, { content: next }));
            }} />

          {presets && <PresetChips items={presets.text} active={item.text?.preset} onPick={applyTextPreset} />}

          <Row label="Font">
            <select value={style.font_family}
              onChange={(e) => patchStyle({ font_family: e.target.value, font_file: null })}>
              {fonts.length === 0 && <option>{style.font_family}</option>}
              {fonts.map((f) => <option key={f.family} value={f.family}>{f.family}</option>)}
            </select>
          </Row>
          <Row label="Weight">
            <button className={`btn btn-xs ${style.bold ? 'btn-primary' : ''}`}
              onClick={() => patchStyle({ bold: !style.bold })}><b>B</b></button>
            <button className={`btn btn-xs ${style.italic ? 'btn-primary' : ''}`}
              onClick={() => patchStyle({ italic: !style.italic })}><i>I</i></button>
            <select value={style.align} onChange={(e) => patchStyle({ align: e.target.value as any })}>
              <option value="left">Left</option>
              <option value="center">Center</option>
              <option value="right">Right</option>
            </select>
          </Row>
          <Slider label="Size" value={style.font_size} min={12} max={220} step={1}
            onChange={(v) => patchStyle({ font_size: Math.round(v) })} format={(v) => `${Math.round(v)}px`} />
          <ColorField label="Fill" value={style.color} onChange={(v) => patchStyle({ color: v })} allowAlpha={false} />
          <Slider label="Opacity" value={style.opacity} min={0} max={1}
            onChange={(v) => patchStyle({ opacity: v })} format={(v) => `${Math.round(v * 100)}%`} />
          <Slider label="Outline" value={style.stroke_width} min={0} max={20} step={1}
            onChange={(v) => patchStyle({ stroke_width: Math.round(v) })} format={(v) => `${Math.round(v)}px`} />
          {style.stroke_width > 0 && (
            <ColorField label="Outline colour" value={style.stroke_color}
              onChange={(v) => patchStyle({ stroke_color: v })} />
          )}
          <Slider label="Shadow X" value={style.shadow_x} min={-20} max={20} step={1}
            onChange={(v) => patchStyle({ shadow_x: Math.round(v) })} format={(v) => `${Math.round(v)}`} />
          <Slider label="Shadow Y" value={style.shadow_y} min={-20} max={20} step={1}
            onChange={(v) => patchStyle({ shadow_y: Math.round(v) })} format={(v) => `${Math.round(v)}`} />
          {(style.shadow_x !== 0 || style.shadow_y !== 0) && (
            <ColorField label="Shadow colour" value={style.shadow_color}
              onChange={(v) => patchStyle({ shadow_color: v })} />
          )}
          <Row label="Background">
            <input type="checkbox" checked={style.box} onChange={(e) => patchStyle({ box: e.target.checked })} />
            <span className="text-xs text-muted">Box behind text</span>
          </Row>
          {style.box && (
            <>
              <ColorField label="Box colour" value={style.box_color} onChange={(v) => patchStyle({ box_color: v })} />
              <Slider label="Box padding" value={style.box_padding} min={0} max={60} step={1}
                onChange={(v) => patchStyle({ box_padding: Math.round(v) })} format={(v) => `${Math.round(v)}px`} />
            </>
          )}
          <Slider label="Position X" value={style.pos_x} min={-1} max={1}
            onChange={(v) => patchStyle({ pos_x: v })} />
          <Slider label="Position Y" value={style.pos_y} min={-1} max={1}
            onChange={(v) => patchStyle({ pos_y: v })} />
          <Slider label="Line spacing" value={style.line_spacing} min={0} max={60} step={1}
            onChange={(v) => patchStyle({ line_spacing: Math.round(v) })} format={(v) => `${Math.round(v)}`} />
          <Row label="Animation">
            <select value={style.animation} onChange={(e) => patchStyle({ animation: e.target.value as any })}>
              <option value="none">None</option>
              <option value="fade">Fade</option>
              <option value="pop">Pop</option>
              <option value="slide-up">Slide up</option>
            </select>
          </Row>
          {style.animation !== 'none' && (
            <Slider label="Anim. length" value={style.animation_duration} min={0.05} max={1.5}
              onChange={(v) => patchStyle({ animation_duration: v })} format={(v) => `${v.toFixed(2)}s`} />
          )}
        </Section>
      )}

      {!isText && (
        <Section title={isAdjustment ? 'Transform · reframe what is below' : 'Transform · crop, pan & zoom'}
          action={<button className="btn btn-xs" onClick={resetTransform} disabled={busy}>Reset</button>}>
          <div className="insp-chips">
            <button className="insp-chip" onClick={() => kenBurns(1, 1.25)}>Zoom in</button>
            <button className="insp-chip" onClick={() => kenBurns(1.25, 1)}>Zoom out</button>
            <button className="insp-chip" onClick={() => patchTransform({ scale: 1.2, scale_end: null })}>Punch in</button>
            <button className="insp-chip"
              onClick={() => patchTransform({ pos_x: -0.4, pos_x_end: 0.4, scale: Math.max(1.15, transform.scale) })}>
              Pan L→R
            </button>
          </div>

          <Slider label="Zoom" value={transform.scale} min={0.2} max={3} step={0.01}
            onChange={(v) => patchTransform({ scale: v })} format={(v) => `${v.toFixed(2)}×`}
            hint="1.0 fills the frame. Below 1 shrinks with black bars." />
          <Row label="Animate zoom" hint="Ken Burns: the clip moves from the zoom above to the one below.">
            <input type="checkbox" checked={animatedZoom}
              onChange={(e) => patchTransform({ scale_end: e.target.checked ? transform.scale * 1.2 : null })} />
            <span className="text-xs text-muted">{animatedZoom ? 'on' : 'off'}</span>
          </Row>
          {animatedZoom && (
            <Slider label="Zoom to" value={transform.scale_end ?? transform.scale} min={0.2} max={3}
              onChange={(v) => patchTransform({ scale_end: v })} format={(v) => `${v.toFixed(2)}×`} />
          )}

          <Slider label="Pan X" value={transform.pos_x} min={-1} max={1}
            onChange={(v) => patchTransform({ pos_x: v })} />
          <Slider label="Pan Y" value={transform.pos_y} min={-1} max={1}
            onChange={(v) => patchTransform({ pos_y: v })} />
          <Row label="Animate pan">
            <input type="checkbox" checked={animatedPan}
              onChange={(e) => patchTransform({
                pos_x_end: e.target.checked ? transform.pos_x + 0.3 : null,
                pos_y_end: e.target.checked ? transform.pos_y : null,
              })} />
            <span className="text-xs text-muted">{animatedPan ? 'on' : 'off'}</span>
          </Row>
          {animatedPan && (
            <>
              <Slider label="Pan X to" value={transform.pos_x_end ?? transform.pos_x} min={-1} max={1}
                onChange={(v) => patchTransform({ pos_x_end: v })} />
              <Slider label="Pan Y to" value={transform.pos_y_end ?? transform.pos_y} min={-1} max={1}
                onChange={(v) => patchTransform({ pos_y_end: v })} />
            </>
          )}

          <div className="insp-subhead">Crop</div>
          <Slider label="Left" value={transform.crop_left} min={0} max={0.45}
            onChange={(v) => patchTransform({ crop_left: v })} format={(v) => `${Math.round(v * 100)}%`} />
          <Slider label="Right" value={transform.crop_right} min={0} max={0.45}
            onChange={(v) => patchTransform({ crop_right: v })} format={(v) => `${Math.round(v * 100)}%`} />
          <Slider label="Top" value={transform.crop_top} min={0} max={0.45}
            onChange={(v) => patchTransform({ crop_top: v })} format={(v) => `${Math.round(v * 100)}%`} />
          <Slider label="Bottom" value={transform.crop_bottom} min={0} max={0.45}
            onChange={(v) => patchTransform({ crop_bottom: v })} format={(v) => `${Math.round(v * 100)}%`} />

          <Slider label="Rotation" value={transform.rotation} min={-180} max={180} step={1}
            onChange={(v) => patchTransform({ rotation: Math.round(v) })} format={(v) => `${Math.round(v)}°`} />
          <Row label="Flip" hint="Mirrors the picture. The crop stays on the side of the original frame you chose.">
            <button className={`insp-chip ${transform.flip_h ? 'active' : ''}`}
              onClick={() => patchTransform({ flip_h: !transform.flip_h })}
              title="Mirror left to right">⇋ Horizontal</button>
            <button className={`insp-chip ${transform.flip_v ? 'active' : ''}`}
              onClick={() => patchTransform({ flip_v: !transform.flip_v })}
              title="Mirror top to bottom">⇵ Vertical</button>
          </Row>
          {item && item.track !== 'V1' && (
            <Slider label={isAdjustment ? 'Strength' : 'Opacity'} value={transform.opacity} min={0} max={1}
              onChange={(v) => patchTransform({ opacity: v })} format={(v) => `${Math.round(v * 100)}%`}
              hint={isAdjustment ? 'How much of this layer’s treatment reaches the picture.' : undefined} />
          )}
        </Section>
      )}

      {!isText && (
        <Section title="Colour correction"
          action={<button className="btn btn-xs" disabled={busy}
            onClick={() => { setDraftColor(null); projectId && run(() => setColor(projectId, selectedClipId, {}, null, true)); }}>
            Reset</button>}>
          {presets && <PresetChips items={presets.color} active={color.preset} onPick={applyColorPreset} />}

          {/* Correct first: get the picture right before any creative decision
              is made against it. The filter chain applies these in this order. */}
          <div className="insp-group-title">Correct</div>
          <Slider label="Exposure" value={color.exposure} min={-3} max={3} step={0.05}
            onChange={(v) => patchColor({ exposure: v })}
            format={(v) => `${v > 0 ? '+' : ''}${v.toFixed(2)} EV`}
            hint="In stops, the unit the camera works in. +1 is twice the light." />
          <Slider label="Contrast" value={color.contrast} min={0} max={2.5}
            onChange={(v) => patchColor({ contrast: v })} />
          <Slider label="Brightness" value={color.brightness} min={-0.5} max={0.5}
            onChange={(v) => patchColor({ brightness: v })} />
          <Slider label="Gamma" value={color.gamma} min={0.3} max={3}
            onChange={(v) => patchColor({ gamma: v })} />
          <Slider label="Temperature" value={color.temperature} min={-1} max={1}
            onChange={(v) => patchColor({ temperature: v })}
            format={(v) => (v === 0 ? 'neutral' : v > 0 ? `warm ${Math.round(v * 100)}` : `cool ${Math.round(-v * 100)}`)} />
          <Slider label="Tint" value={color.tint} min={-1} max={1}
            onChange={(v) => patchColor({ tint: v })}
            format={(v) => (v === 0 ? 'neutral' : v > 0 ? `magenta ${Math.round(v * 100)}` : `green ${Math.round(-v * 100)}`)}
            hint="The green–magenta axis. This is what fixes fluorescent light; temperature cannot." />
          <Slider label="Shadows" value={color.shadows} min={-1} max={1}
            onChange={(v) => patchColor({ shadows: v })}
            hint="Lifts or crushes the low end only, leaving midtones alone." />
          <Slider label="Highlights" value={color.highlights} min={-1} max={1}
            onChange={(v) => patchColor({ highlights: v })}
            hint="Recovers a blown sky without flattening the face." />

          {/* Then grade. */}
          <div className="insp-group-title">Grade</div>
          <Slider label="Saturation" value={color.saturation} min={0} max={3}
            onChange={(v) => patchColor({ saturation: v })} />
          <Slider label="Vibrance" value={color.vibrance} min={-2} max={2}
            onChange={(v) => patchColor({ vibrance: v })}
            hint="Saturates only the muted colours, so skin tones survive it." />
          <Slider label="Hue" value={color.hue} min={-180} max={180} step={1}
            onChange={(v) => patchColor({ hue: Math.round(v) })} format={(v) => `${Math.round(v)}°`} />
          <Wheel label="Lift (shadows)" value={color.lift}
            onChange={(v) => patchColor({ lift: { ...color.lift, ...v } })}
            hint="Colour cast in the darkest part of the picture." />
          <Wheel label="Midtones" value={color.midtones}
            onChange={(v) => patchColor({ midtones: { ...color.midtones, ...v } })}
            hint="Where skin lives — the range that reads as the picture's colour." />
          <Wheel label="Gain (highlights)" value={color.gain}
            onChange={(v) => patchColor({ gain: { ...color.gain, ...v } })}
            hint="Colour cast in the brightest part of the picture." />

          {/* Finally the look. */}
          <div className="insp-group-title">Look</div>
          <Row label="LUT" hint="A .cube 3D LUT, applied after everything above.">
            <span className="insp-value" style={{ flex: 1, textAlign: 'left', overflow: 'hidden',
              textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {color.lut_file ? color.lut_file.split(/[\\/]/).pop() : 'none'}
            </span>
            <button className="btn btn-xs" onClick={async () => {
              const picked = await pickFile([{ name: '3D LUT', extensions: ['cube'] }]);
              if (picked) patchColor({ lut_file: picked });
            }}>Load…</button>
            {color.lut_file && (
              <button className="btn btn-xs" onClick={() => patchColor({ lut_file: null })}>✕</button>
            )}
          </Row>
          <Slider label="Denoise" value={color.denoise} min={0} max={1}
            onChange={(v) => patchColor({ denoise: v })} format={(v) => `${Math.round(v * 100)}%`}
            hint="Runs before the grade, so contrast does not amplify the grain." />
          <Slider label="Sharpen" value={color.sharpen} min={0} max={2}
            onChange={(v) => patchColor({ sharpen: v })} />
          <Slider label="Vignette" value={color.vignette} min={0} max={1}
            onChange={(v) => patchColor({ vignette: v })} format={(v) => `${Math.round(v * 100)}%`} />
          <Slider label="Fade in" value={color.fade_in} min={0} max={3}
            onChange={(v) => patchColor({ fade_in: v })} format={(v) => `${v.toFixed(2)}s`} />
          <Slider label="Fade out" value={color.fade_out} min={0} max={3}
            onChange={(v) => patchColor({ fade_out: v })} format={(v) => `${v.toFixed(2)}s`} />
        </Section>
      )}

      {canKey && (
        <Section title="Chroma key" defaultOpen={keyed}
          action={keyed ? (
            <button className="btn btn-xs" disabled={busy}
              onClick={() => { setDraftChroma(null); projectId && run(() => setChroma(projectId, selectedClipId!, {}, true)); }}>
              Remove</button>
          ) : undefined}>
          {!keyed ? (
            <>
              <div className="text-xs text-muted" style={{ marginBottom: 6 }}>
                Drop out a green or blue screen so the layers below show through.
              </div>
              <div className="insp-chips">
                {KEY_PRESETS.map((p) => (
                  <button key={p.id} className="insp-chip" disabled={busy}
                    onClick={() => { setDraftChroma(null); patchChroma({ enabled: true, color: p.color }); }}>
                    {p.label}
                  </button>
                ))}
              </div>
            </>
          ) : (
            <>
              <ColorField label="Key colour" value={chroma.color} allowAlpha={false}
                onChange={(v) => patchChroma({ color: v })} />
              <Row label="Keying" hint="Chroma keys on hue, which tolerates a screen that is lit unevenly. Colour keys on RGB distance — right for a flat graphic, wrong for a screen.">
                <select value={chroma.key_type}
                  onChange={(e) => patchChroma({ key_type: e.target.value as 'chroma' | 'color' })}>
                  <option value="chroma">Chroma (lit screen)</option>
                  <option value="color">Colour (flat graphic)</option>
                </select>
              </Row>
              <Slider label="Similarity" value={chroma.similarity} min={0.01} max={1} step={0.005}
                onChange={(v) => patchChroma({ similarity: v })}
                hint="How far from the key colour still counts as background. Raise it until the screen is gone, then stop — past that it starts eating the subject." />
              <Slider label="Blend" value={chroma.blend} min={0} max={1} step={0.005}
                onChange={(v) => patchChroma({ blend: v })}
                hint="Softness of the key in colour space. A little keeps hair and motion blur." />

              <div className="insp-group-title">Edges</div>
              <Slider label="Choke" value={chroma.choke} min={0} max={1} step={0.05}
                onChange={(v) => patchChroma({ choke: v })} format={(v) => `${Math.round(v * 3)} px`}
                hint="Shrinks the matte by whole pixels. This is what removes the bright fringe that no similarity setting reaches." />
              <Slider label="Feather" value={chroma.feather} min={0} max={1} step={0.05}
                onChange={(v) => patchChroma({ feather: v })} format={(v) => `${Math.round(v * 100)}%`}
                hint="Softens the matte edge so the composite is not a hard staircase." />

              <div className="insp-group-title">Spill</div>
              <Slider label="Suppress" value={chroma.spill} min={0} max={1} step={0.05}
                onChange={(v) => patchChroma({ spill: v })} format={(v) => `${Math.round(v * 100)}%`}
                hint="Removes screen colour reflected onto the subject. That is inside them, not at the edge, so no amount of keying will do it." />
              <Slider label="Reach" value={chroma.spill_expand} min={0} max={1} step={0.05}
                onChange={(v) => patchChroma({ spill_expand: v })} format={(v) => `${Math.round(v * 100)}%`} />

              <label className="insp-row" title="White keeps, black drops. A fringe invisible against a dark background is glaring against a light one, so this is the only reliable way to judge a key.">
                <span className="insp-label">Show matte</span>
                <div className="insp-control">
                  <input type="checkbox" checked={chroma.show_matte}
                    onChange={(e) => patchChroma({ show_matte: e.target.checked })} />
                  <span className="insp-value">{chroma.show_matte ? 'matte' : 'picture'}</span>
                </div>
              </label>
            </>
          )}
        </Section>
      )}

      <Section title={item && !isText && !isAdjustment ? 'Transition into this clip' : 'Transitions'}
        defaultOpen={false}>
        <div className="text-xs text-muted" style={{ marginBottom: 6 }}>
          {item && !isText
            ? 'Applies where this clip joins the one before it.'
            : 'Applied at every cut in the programme. Select a clip to override one junction.'}
        </div>
        {presets && (
          <PresetChips
            items={presets.transition}
            active={transitionActive}
            onPick={(id) => projectId && run(() =>
              setTransition(projectId, item && !isText && !isAdjustment ? selectedClipId : null, { preset: id }))}
          />
        )}
        {activeTransition && (
          <Slider label="Length" value={activeTransition.duration ?? 0.5} min={0.1} max={2.0}
            onChange={(v) => projectId && schedule(() => setTransition(
              projectId, item && !isText && !isAdjustment ? selectedClipId : null,
              { updates: { duration: v } }))}
            format={(v) => `${v.toFixed(2)}s`} />
        )}
        <Row label="More">
          <select
            value={activeTransition?.type ?? ''}
            onChange={(e) => projectId && run(() => setTransition(
              projectId, item && !isText && !isAdjustment ? selectedClipId : null,
              { updates: { type: e.target.value, duration: activeTransition?.duration ?? 0.5 } }))}>
            <option value="">— pick a transition —</option>
            {(presets?.transition_catalogue ?? []).map((group) => (
              <optgroup key={group.group} label={group.group}>
                {group.transitions.map((name) => <option key={name} value={name}>{name}</option>)}
              </optgroup>
            ))}
          </select>
        </Row>
      </Section>

      <Section title="Effects & atmosphere" defaultOpen={isAdjustment}>
        <div className="text-xs text-muted" style={{ marginBottom: 6 }}>
          {isAdjustment
            ? `Generated over every layer below ${item.track}, for as long as this clip runs.`
            : 'Generated over the finished picture — no stock footage needed.'}
        </div>
        {presets && (
          <PresetChips items={presets.effect} onPick={(id) =>
            projectId && run(() => addEffect(projectId, id, {}, effectOwner))} />
        )}

        {effects.length === 0 ? (
          <div className="text-xs text-muted">No effects yet — pick one above.</div>
        ) : effects.map((effect, index) => (
          <div key={index} className="insp-effect">
            <div className="insp-effect-head">
              <span className="insp-effect-name">{effect.type.replace('_', ' ')}</span>
              <label className="text-xs text-muted" style={{ display: 'flex', gap: 4 }}>
                <input type="checkbox" checked={effect.enabled}
                  onChange={(e) => projectId && run(() =>
                    updateEffect(projectId, index, { enabled: e.target.checked }, effectOwner))} />
                on
              </label>
              <button className="btn btn-xs btn-danger" title="Remove"
                onClick={() => projectId && run(() => removeEffect(projectId, index, effectOwner))}>✕</button>
            </div>
            <Slider label="Amount" value={effect.intensity} min={0} max={1}
              onChange={(v) => projectId && schedule(() =>
                updateEffect(projectId, index, { intensity: v }, effectOwner))}
              format={(v) => `${Math.round(v * 100)}%`} />
            <Slider label="Speed" value={effect.speed} min={0.2} max={3}
              onChange={(v) => projectId && schedule(() =>
                updateEffect(projectId, index, { speed: v }, effectOwner))}
              format={(v) => `${v.toFixed(1)}×`} />
            {(effect.type === 'sunlight' || effect.type === 'light_leak') && (
              <ColorField label="Tint" value={toHex(effect.color || '#ffcc88')}
                allowAlpha={false}
                onChange={(v) => projectId && schedule(() => updateEffect(
                  projectId, index, { color: `0x${v.replace('#', '').toUpperCase()}FF` }, effectOwner))} />
            )}
          </div>
        ))}
      </Section>

      <Section title="Cinematic framing" defaultOpen={false}>
        <div className="text-xs text-muted" style={{ marginBottom: 6 }}>
          Mattes black bars on without changing the export resolution.
        </div>
        {presets && (
          <PresetChips
            items={presets.aspect}
            active={aspectActive}
            onPick={(id) => {
              const chosen = presets.aspect.find((a) => a.id === id);
              const ratio = (chosen as any)?.ratio ?? null;
              if (projectId) run(() => setAspectBars(projectId, ratio));
            }}
          />
        )}
      </Section>

      <Section title="Match a reference video" defaultOpen={false}>
        <div className="text-xs text-muted" style={{ marginBottom: 6 }}>
          Measures how a video was edited — its pace, movement, look and caption
          placement — and applies that to your footage. It cannot recover the original
          effects; a finished video only carries the result, not how it was made.
        </div>

        <div className="insp-actions">
          <button className="btn btn-sm btn-primary" disabled={busy} onClick={handleAnalyzeReference}>
            {busy ? 'Analysing…' : '＋ Analyse a video'}
          </button>
        </div>

        {styles.length > 0 && (
          <>
            <Row label="Style">
              <select value={styleId} onChange={(e) => { setStyleId(e.target.value); setStyleReport(null); }}>
                {styles.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
              </select>
              <button className="btn btn-xs btn-danger" title="Delete this style"
                disabled={busy || !styleId}
                onClick={async () => {
                  await deleteStyleProfile(styleId).catch(() => {});
                  setStyleId('');
                  setStyleReport(null);
                  refreshStyles();
                }}>✕</button>
            </Row>

            {selectedStyle && <StyleSummary profile={selectedStyle} />}

            <div className="insp-subhead">Apply</div>
            <div className="insp-chips">
              {([
                ['look', 'Colour look'],
                ['motion', 'Pace & zooms'],
                ['captions', 'Caption placement'],
                ['transitions', 'Fades'],
              ] as [keyof StyleApplyParts, string][]).map(([key, label]) => (
                <button key={key}
                  className={`insp-chip ${styleParts[key] ? 'active' : ''}`}
                  onClick={() => setStyleParts((p) => ({ ...p, [key]: !p[key] }))}>
                  {styleParts[key] ? '✓ ' : ''}{label}
                </button>
              ))}
            </div>

            <div className="insp-actions">
              <button className="btn btn-sm btn-primary" disabled={busy || !projectId || !styleId}
                onClick={handleApplyStyle}>
                Apply to this project
              </button>
            </div>

            {styleReport && (
              <div className="insp-style-report">
                {styleReport.map((line) => (
                  <div key={line} className="text-xs">{line}</div>
                ))}
              </div>
            )}
          </>
        )}
      </Section>

      <Section title="Captions" defaultOpen={!item}>
        <Row label="Style">
          <select value={captionPreset} onChange={(e) => setCaptionPreset(e.target.value)}>
            {(presets?.caption ?? []).map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
        </Row>
        <div className="insp-actions">
          <button className="btn btn-sm btn-primary" disabled={busy || !projectId}
            onClick={() => projectId && run(() => generateCaptions(projectId, captionPreset))}>
            Generate captions
          </button>
          <button className="btn btn-sm" disabled={busy || !projectId}
            onClick={() => projectId && run(() => clearCaptions(projectId))}>Clear</button>
        </div>
        <div className="text-xs text-muted">
          Built from the words still enabled in the transcript, so captions follow your cuts.
        </div>
      </Section>

      <Section title="Intro" defaultOpen={false}>
        <Row label="Preset">
          <select value={introPreset} onChange={(e) => setIntroPreset(e.target.value)}>
            {(presets?.intro ?? []).map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
          </select>
        </Row>
        <Row label="Title">
          <input type="text" value={introTitle} onChange={(e) => setIntroTitle(e.target.value)} />
        </Row>
        <Row label="Subtitle">
          <input type="text" value={introSubtitle} onChange={(e) => setIntroSubtitle(e.target.value)} />
        </Row>
        <div className="insp-actions">
          <button className="btn btn-sm btn-primary" disabled={busy || !projectId}
            onClick={() => projectId && run(() => applyIntro(projectId, introPreset, introTitle, introSubtitle))}>
            Apply intro
          </button>
          <button className="btn btn-sm" disabled={busy || !projectId}
            onClick={() => projectId && run(() => clearIntro(projectId))}>Remove</button>
        </div>
      </Section>

      <Section title="Add text" defaultOpen={false}>
        <div className="insp-actions">
          <button className="btn btn-sm btn-primary" disabled={busy} onClick={handleAddText}>
            ＋ Text at playhead
          </button>
        </div>
        <div className="text-xs text-muted">
          Lands on a text track at the playhead. Select it to restyle, or drag its edges to retime.
        </div>
      </Section>
    </div>
  );
}
