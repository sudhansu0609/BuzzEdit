import { useState, useEffect, useCallback } from 'react';
import { useProjectStore } from '../hooks/store';
import {
  getComfyUIStatus, triggerThumbnail, triggerCaptions,
  startPresentationPass, getPresentationStatus, getProject, getTimeline,
  listGenerationWorkflows, updateAppSettings, WorkflowInfo,
  getShotPlan, ShotPrompt, setProjectScript, getProjectScript, ScriptStatus,
  listStyleProfiles, StyleProfile, COMFY_BASE, hostPort,
} from '../hooks/api';

/** The dressing layers the pass adds on top of the cut (presentation/models.py
    PresentationSettings). Each is one switch sent as-is to the backend. */
const DRESSING_OPTIONS: [string, string, string][] = [
  ['music', 'Music bed + ambience', 'A bed from data/music/<genre>/ (or a synthesised drone for horror), ducked under the voice; wind/room tone under it'],
  ['sfx', 'Sound effects', 'Whooshes on cutaways, pops on text, stingers/thunder/risers at the story\'s hits'],
  ['cards', 'Text cards', 'Chapter titles, stat call-outs, location/date, character, source, definition and quote cards read from what the speaker says'],
  ['maps', 'Maps', 'A map cutaway for each place named (offline Natural Earth data)'],
  ['moods', 'Moods + hits', 'Per-topic grade shifts and pushes; flash/shake/thunder at the key moment in horror and true crime'],
  ['structure', 'Story structure', 'Acts weigh the B-roll density; the cold open plays the most gripping line first; chapters and listing are written'],
  ['grade', 'Genre grade + transitions', 'A master grade for the genre and a transition at each chapter boundary'],
  ['verify', 'Verify the result', 'Text off the face, cards not stacked, bed ducked, loudness on target'],
];

/** Video genres the backend can style prompts for (presentation/genre.py).
    '' means "detect it from the transcript". */
const GENRE_OPTIONS: [string, string][] = [
  ['', 'Auto-detect from transcript'],
  ['horror', 'Horror'],
  ['true_crime', 'True crime'],
  ['comedy', 'Comedy'],
  ['gaming', 'Gaming'],
  ['tech', 'Tech'],
  ['science_education', 'Science / Education'],
  ['finance', 'Finance / Business'],
  ['motivational', 'Motivational'],
  ['health_fitness', 'Health / Fitness'],
  ['cooking', 'Cooking / Food'],
  ['travel', 'Travel'],
  ['devotional', 'Devotional'],
  ['news', 'News'],
  ['vlog', 'Vlog / Lifestyle'],
  ['general', 'General (no styling)'],
];

/** The generation jobs a workflow can be assigned to. */
const ROLE_LABELS: [string, string, string][] = [
  ['broll_image', 'B-roll images', 'Stills generated for each topic — e.g. Z-Image Turbo'],
  ['broll_video', 'B-roll video', 'Moving shots. Leave unset to use stills everywhere'],
  ['thumbnail', 'Thumbnail', 'The YouTube thumbnail'],
  ['graphic', 'Graphics', 'Transparent overlays (experimental)'],
];

export default function AgentPanel() {
  const { project, updateProject, setProcessing, setProgress, setError } = useProjectStore();
  const [comfyStatus, setComfyStatus] = useState<{ connected: boolean; status: string; reason?: string }>({ connected: false, status: 'checking' });
  const [genBroll, setGenBroll] = useState(true);
  const [genZoom, setGenZoom] = useState(true);
  const [genThumb, setGenThumb] = useState(true);
  const [burnCaps, setBurnCaps] = useState(true);
  const [customTitle, setCustomTitle] = useState('');
  // How much of the video B-roll should cover (%). The backend derives every
  // gap and budget from this one knob; it hard-caps at 80% server-side too.
  const [coverage, setCoverage] = useState(75);
  const [zoomDepth, setZoomDepth] = useState(10);
  // The video's category. Styles every generated prompt (B-roll + thumbnail)
  // to match — horror imagery for a horror video. '' lets the backend detect it.
  const [genre, setGenre] = useState('');
  const [dressing, setDressing] = useState<Record<string, boolean>>(
    () => Object.fromEntries(DRESSING_OPTIONS.map(([key]) => [key, true])));
  const [voicePreset, setVoicePreset] = useState('clean');
  const [pip, setPip] = useState('auto');
  const [bilingual, setBilingual] = useState('auto');
  const [styleProfile, setStyleProfile] = useState('');
  const [styleProfiles, setStyleProfiles] = useState<StyleProfile[]>([]);
  useEffect(() => {
    listStyleProfiles().then((res) => setStyleProfiles(res.profiles || [])).catch(() => {});
  }, []);
  const [scriptText, setScriptText] = useState('');
  const [scriptStatus, setScriptStatus] = useState<ScriptStatus | null>(null);
  const [scriptBusy, setScriptBusy] = useState(false);
  const [lastReport, setLastReport] = useState<any>(null);
  // The "Prompts used" tab: the text that produced each generated shot.
  const [showPrompts, setShowPrompts] = useState(false);
  const [prompts, setPrompts] = useState<ShotPrompt[] | null>(null);
  const [promptsError, setPromptsError] = useState<string | null>(null);
  const [workflows, setWorkflows] = useState<WorkflowInfo[]>([]);
  const [selected, setSelected] = useState<Record<string, string | null>>({});
  const [settingsKeys, setSettingsKeys] = useState<Record<string, string>>({});

  const loadWorkflows = useCallback(async () => {
    try {
      const res = await listGenerationWorkflows();
      setWorkflows(res.workflows || []);
      setSelected(res.selected || {});
      setSettingsKeys(res.settings_keys || {});
    } catch (err) {
      console.error('Could not list generation workflows:', err);
    }
  }, []);

  useEffect(() => { loadWorkflows(); }, [loadWorkflows]);

  const loadPrompts = useCallback(async () => {
    if (!project) return;
    setPromptsError(null);
    try {
      const res = await getShotPlan(project.id);
      setPrompts(res.prompts || []);
    } catch (err: any) {
      setPrompts([]);
      setPromptsError('No prompts yet — run an AI edit with B-roll first.');
    }
  }, [project]);

  const togglePrompts = useCallback(() => {
    setShowPrompts((open) => {
      const next = !open;
      if (next) loadPrompts();
      return next;
    });
  }, [loadPrompts]);

  const pickWorkflow = useCallback(async (role: string, file: string) => {
    const key = settingsKeys[role];
    if (!key) return;
    setSelected((s) => ({ ...s, [role]: file || null }));
    try {
      await updateAppSettings({ [key]: file || null });
    } catch (err: any) {
      setError(err.message);
    }
  }, [settingsKeys, setError]);

  const checkComfy = useCallback(async () => {
    try {
      const res = await getComfyUIStatus();
      setComfyStatus({ connected: res.connected, status: res.status, reason: (res as any).reason });
    } catch {
      setComfyStatus({ connected: false, status: 'offline' });
    }
  }, []);

  useEffect(() => {
    checkComfy();
    const interval = setInterval(checkComfy, 10000);
    return () => clearInterval(interval);
  }, [checkComfy]);

  // Run the presentation pass as a polled job and reflect the new timeline
  // (B-roll on the overlay track, zoom transforms) back into the editor.
  const runPass = useCallback(async (settings: Record<string, any>, startMsg: string) => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, startMsg);
    setLastReport(null);
    try {
      // `grade` covers the master grade and the chapter transitions; `structure`
      // covers acts, the cold open and the listing; `cards` and `maps` share
      // the entity pass. Everything else is one switch each.
      const dressingSettings = {
        music: dressing.music, ambience: dressing.music,
        sfx: dressing.sfx,
        cards: dressing.cards,
        maps: dressing.maps,
        moods: dressing.moods,
        structure: dressing.structure, cold_open: dressing.structure, metadata: dressing.structure,
        grade: dressing.grade ? 'auto' : 'off', topic_transitions: dressing.grade,
        verify: dressing.verify,
        voice_preset: voicePreset,
        pip,
        captions_bilingual: bilingual,
        style_profile: styleProfile || null,
      };
      const start = await startPresentationPass(project.id, {
        target_coverage: coverage / 100,
        zoom_depth: zoomDepth / 100,
        genre: genre || null,
        ...dressingSettings,
        ...settings,
      });
      const jobId = start?.job_id;
      let result: any = null;
      if (jobId) {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          await new Promise((r) => setTimeout(r, 1200));
          const job: any = await getPresentationStatus(jobId);
          setProgress(job?.progress ?? 0, job?.message || 'Working…');
          if (job?.status === 'completed') { result = job.result; break; }
          if (job?.status === 'failed') { throw new Error(job?.error || 'Presentation pass failed'); }
        }
      }
      // Pull the updated project + timeline so the new B-roll and zooms appear.
      const [proj, tl] = await Promise.all([
        getProject(project.id).catch(() => null),
        getTimeline(project.id).catch(() => null),
      ]);
      if (proj) updateProject(proj);
      if (tl) updateProject({ timeline: tl });
      setLastReport(result);
      const placed = result?.broll_placed;
      const zooms = (result?.zooms_segment ?? 0) + (result?.zooms_windowed ?? 0);
      setProgress(1, placed != null
        ? `Done — ${placed} B-roll clip${placed === 1 ? '' : 's'}, ${zooms} zoom${zooms === 1 ? '' : 's'}`
        : 'Presentation pass complete');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }, [project, updateProject, setProcessing, setProgress, setError, coverage, zoomDepth, genre,
      dressing, voicePreset, pip, bilingual, styleProfile]);

  // The speaker's script: shown when the project has one, aligned on demand.
  useEffect(() => {
    if (!project) { setScriptStatus(null); setScriptText(''); return; }
    getProjectScript(project.id)
      .then((status) => { setScriptStatus(status); if (status.text) setScriptText(status.text); })
      .catch(() => setScriptStatus(null));
  }, [project?.id]);

  const handleAlignScript = async () => {
    if (!project || !scriptText.trim()) return;
    setScriptBusy(true);
    try {
      const status = await setProjectScript(project.id, scriptText);
      setScriptStatus(status);
      // The words' spelling changed; pull the timeline so the transcript panel shows it.
      const tl = await getTimeline(project.id);
      updateProject({ timeline: tl } as any);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setScriptBusy(false);
    }
  };

  const handleFullAgentEdit = () => {
    // The backend refuses (503) a pass that wants pictures nothing can make;
    // offer the degraded run here instead of surfacing that as an error.
    let broll = genBroll, thumbnail = genThumb;
    if (!comfyStatus.connected && (broll || thumbnail)) {
      if (!window.confirm('ComfyUI is offline — run without B-roll and thumbnail?\n'
        + 'Captions and zooms will still be added.')) return;
      broll = false; thumbnail = false;
    }
    runPass(
      { broll, face_zoom: genZoom, captions: burnCaps, thumbnail, popups: true, render: true },
      'Running full AI edit (B-roll + zoom)…');
  };

  const handleBRollOnly = () => runPass(
    { broll: true, face_zoom: false, captions: false, thumbnail: false, popups: false, render: true },
    'Generating AI B-roll…');

  const handleZoomOnly = () => runPass(
    { broll: false, face_zoom: true, captions: false, thumbnail: false, popups: false, render: true },
    'Adding auto zoom & punch-ins…');

  const handleThumbnailOnly = async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Generating Thumbnail...');
    try {
      await triggerThumbnail(project.id, customTitle || undefined);
      setProgress(1, 'Thumbnail generation completed');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleCaptionsOnly = async () => {
    if (!project) return;
    setProcessing(true);
    setProgress(0, 'Burning styled captions...');
    try {
      await triggerCaptions(project.id);
      setProgress(1, 'Caption burn-in completed');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="agent-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px', color: '#fff' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px', background: '#1e1e2e', borderRadius: '8px' }}>
        <span>ComfyUI Bridge:</span>
        <span style={{
          padding: '4px 10px',
          borderRadius: '12px',
          fontSize: '12px',
          fontWeight: 'bold',
          background: comfyStatus.connected ? '#10b981' : '#ef4444',
          color: '#fff'
        }}>
          {comfyStatus.connected
            ? `ONLINE (${COMFY_BASE ? hostPort(COMFY_BASE) : 'ComfyUI'})`
            : 'OFFLINE (Fallback Active)'}
        </span>
      </div>
      {!comfyStatus.connected && comfyStatus.reason && (
        <div style={{ fontSize: '12px', color: '#f38ba8', padding: '2px 10px 6px', lineHeight: 1.4 }}>
          {comfyStatus.reason} Without it, B-roll images cannot be generated — the
          pass will still add captions, zooms and topic pop-ups.
        </div>
      )}

      {/* Which of the user's own ComfyUI workflows runs for each job. Files are
          discovered from workflows/ and their prompt/size/seed inputs detected,
          so dropping a new one in is all that is needed to use it. */}
      <div style={{ background: '#181825', padding: '14px', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h4 style={{ margin: 0, color: '#a6adc8' }}>Generation Workflows</h4>
          <button className="btn btn-sm" onClick={loadWorkflows} title="Rescan the workflows folder">↻</button>
        </div>
        {workflows.length === 0 ? (
          <span style={{ fontSize: '12px', color: '#a6adc8' }}>
            No workflows found. Save one from ComfyUI with <b>Export (API)</b> into the
            app's <code>workflows/</code> folder.
          </span>
        ) : ROLE_LABELS.map(([role, label, hint]) => {
          const usable = workflows.filter(w => w.valid && w.roles_ok?.[role]);
          return (
            <label key={role} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: 12 }}
              title={hint}>
              <span style={{ minWidth: 100, color: '#bac2de' }}>{label}</span>
              <select
                value={selected[role] || ''}
                onChange={(e) => pickWorkflow(role, e.target.value)}
                style={{ flex: 1, padding: '6px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}
              >
                <option value="">— none —</option>
                {usable.map(w => (
                  <option key={w.file} value={w.file}>
                    {w.file} ({w.output_kind})
                  </option>
                ))}
                {/* A file chosen for a role it does not obviously suit stays
                    selectable: the detection is a good guess, not a verdict. */}
                {selected[role] && !usable.some(w => w.file === selected[role]) && (
                  <option value={selected[role]!}>{selected[role]} (unverified)</option>
                )}
              </select>
            </label>
          );
        })}
        {workflows.some(w => !w.valid) && (
          <span style={{ fontSize: '11px', color: '#f59e0b' }}>
            {workflows.filter(w => !w.valid).map(w => `${w.file}: ${w.error}`).join(' · ')}
          </span>
        )}
      </div>

      <div style={{ background: '#181825', padding: '14px', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <h4 style={{ margin: 0, color: '#a6adc8' }}>AI Agent Settings</h4>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input type="checkbox" checked={genBroll} onChange={(e) => setGenBroll(e.target.checked)} />
          Generate AI B-Roll Overlays (on V3, above the video)
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input type="checkbox" checked={genZoom} onChange={(e) => setGenZoom(e.target.checked)} />
          Auto Zoom / Punch-ins (Ken Burns on the speaker)
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input type="checkbox" checked={genThumb} onChange={(e) => setGenThumb(e.target.checked)} />
          Generate YouTube Thumbnail
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input type="checkbox" checked={burnCaps} onChange={(e) => setBurnCaps(e.target.checked)} />
          Burn-in Styled Captions
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: 12 }}
          title="Styles every generated image and the thumbnail to the kind of video this is — a horror video gets horror imagery. Auto-detect reads it from the transcript.">
          <span style={{ minWidth: 100, color: '#bac2de' }}>Video genre</span>
          <select value={genre} onChange={(e) => setGenre(e.target.value)}
            style={{ flex: 1, padding: '6px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}>
            {GENRE_OPTIONS.map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: 12 }}
          title="How much of the video is covered by generated B-roll. Gaps and budgets are derived from this one number.">
          <span style={{ minWidth: 100, color: '#bac2de' }}>B-roll coverage</span>
          <input type="range" min={0} max={80} step={5} value={coverage}
            onChange={(e) => setCoverage(Number(e.target.value))} style={{ flex: 1 }} />
          <span style={{ minWidth: 34, textAlign: 'right' }}>{coverage}%</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: 12 }}
          title="How far the punch-ins push toward the speaker.">
          <span style={{ minWidth: 100, color: '#bac2de' }}>Zoom depth</span>
          <input type="range" min={4} max={40} step={2} value={zoomDepth}
            onChange={(e) => setZoomDepth(Number(e.target.value))} style={{ flex: 1 }} />
          <span style={{ minWidth: 34, textAlign: 'right' }}>{zoomDepth}%</span>
        </label>
        <div style={{ borderTop: '1px solid #313244', paddingTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span style={{ fontSize: 12, color: '#a6adc8' }}>Dressing (added on top of the cut, all editable afterwards)</span>
          {DRESSING_OPTIONS.map(([key, label, hint]) => (
            <label key={key} title={hint}
              style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: 12 }}>
              <input type="checkbox" checked={dressing[key]}
                onChange={(e) => setDressing({ ...dressing, [key]: e.target.checked })} />
              {label}
            </label>
          ))}
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: 12 }}
            title="Denoise, de-ess and compress the voice, then normalise the mix to -14 LUFS (YouTube's level).">
            <span style={{ minWidth: 100, color: '#bac2de' }}>Voice</span>
            <select value={voicePreset} onChange={(e) => setVoicePreset(e.target.value)}
              style={{ flex: 1, padding: '6px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}>
              <option value="clean">Clean (denoise + light compression)</option>
              <option value="podcast">Podcast (fuller compression)</option>
              <option value="horror_intimate">Horror intimate (close, compressed)</option>
              <option value="light">Light touch</option>
              <option value="off">Off (leave the audio alone)</option>
            </select>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: 12 }}
            title="The speaker in the top-right corner over cutaways longer than 4 s. Auto: explainer genres only (horror keeps its pictures alone).">
            <span style={{ minWidth: 100, color: '#bac2de' }}>Speaker corner</span>
            <select value={pip} onChange={(e) => setPip(e.target.value)}
              style={{ flex: 1, padding: '6px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}>
              <option value="auto">Auto (explainers)</option>
              <option value="on">Always</option>
              <option value="off">Never</option>
            </select>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: 12 }}
            title="A smaller English line under each caption. Auto: Hindi and other Indic projects when a model can translate.">
            <span style={{ minWidth: 100, color: '#bac2de' }}>English line</span>
            <select value={bilingual} onChange={(e) => setBilingual(e.target.value)}
              style={{ flex: 1, padding: '6px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}>
              <option value="auto">Auto (Indic projects)</option>
              <option value="on">Always</option>
              <option value="off">Never</option>
            </select>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: 12 }}
            title="Edit like this channel: a reference video analysed in the Style panel. Its look and transitions win over the genre's; its motion only when auto zoom is off.">
            <span style={{ minWidth: 100, color: '#bac2de' }}>Style profile</span>
            <select value={styleProfile} onChange={(e) => setStyleProfile(e.target.value)}
              style={{ flex: 1, padding: '6px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}>
              <option value="">None (genre look)</option>
              {styleProfiles.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </label>
        </div>
        <div style={{ borderTop: '1px solid #313244', paddingTop: '10px' }}>
          <label style={{ fontSize: '12px', color: '#bac2de', display: 'block', marginBottom: '4px' }}
            title="Paste the script you read from. Its spelling replaces Whisper's in the captions, its paragraphs become the chapters, and bracketed directions become beats: [map: Jaipur] [sfx: thunder] [broll: an old haveli at night] [text: 40% of students] [stat: 25%] [chapter: The Knock] [quote: … | who] [mood: tense] [title: …]. A line starting with # is a chapter heading.">
            Script (optional)
            {scriptStatus?.status === 'aligned' && (
              <span style={{ color: '#a6e3a1', marginLeft: 8 }}>
                aligned {scriptStatus.aligned_words}/{scriptStatus.token_count} words · {scriptStatus.spelling_fixed} spellings fixed · {scriptStatus.directives?.length ?? 0} directions
              </span>
            )}
          </label>
          <textarea value={scriptText} onChange={(e) => setScriptText(e.target.value)}
            placeholder={'# The Haveli\nRaat ke do baje the... [map: Jaipur]\n\nAchanak darwaza khula. [sfx: thunder]'}
            rows={5}
            style={{ width: '100%', padding: '8px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px', fontFamily: 'inherit', fontSize: 12, resize: 'vertical' }} />
          <button className="btn btn-sm" onClick={handleAlignScript} disabled={scriptBusy || !scriptText.trim()}
            style={{ marginTop: 4 }}>
            {scriptBusy ? 'Aligning…' : 'Align script to transcript'}
          </button>
        </div>
        <div style={{ marginTop: '8px' }}>
          <label style={{ fontSize: '12px', color: '#bac2de', display: 'block', marginBottom: '4px' }}>Thumbnail Custom Title:</label>
          <input
            type="text"
            placeholder="e.g. EPIC VIDEO EDIT!"
            value={customTitle}
            onChange={(e) => setCustomTitle(e.target.value)}
            style={{ width: '100%', padding: '8px', background: '#313244', border: '1px solid #45475a', color: '#fff', borderRadius: '4px' }}
          />
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <button
          className="btn btn-primary"
          style={{ width: '100%', padding: '12px', fontWeight: 'bold', fontSize: '14px' }}
          onClick={handleFullAgentEdit}
        >
          🚀 Run Full AI Edit (B-roll + Zoom + Captions)
        </button>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <button className="btn btn-secondary" onClick={handleBRollOnly}
            disabled={!comfyStatus.connected}
            title={comfyStatus.connected ? undefined : 'ComfyUI is offline — B-roll needs it'}
            style={comfyStatus.connected ? undefined : { opacity: 0.5, cursor: 'not-allowed' }}>
            🎬 B-Roll Only
          </button>
          <button className="btn btn-secondary" onClick={handleZoomOnly}>🔍 Auto Zoom Only</button>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <button className="btn btn-secondary" onClick={handleThumbnailOnly}>🖼️ Thumbnail</button>
          <button className="btn btn-secondary" onClick={handleCaptionsOnly}>💬 Burn Captions</button>
        </div>
        <span style={{ fontSize: 11, color: '#a6adc8' }}>
          Runs on the current edit. B-roll needs ComfyUI online and a workflow set above.
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button className={`btn btn-sm ${showPrompts ? 'btn-primary' : ''}`} onClick={togglePrompts}>
          🖼️ {showPrompts ? 'Hide' : 'Prompts used'}
        </button>
        {showPrompts && (
          <button className="btn btn-sm" onClick={loadPrompts} title="Reload the prompts from the last run">↻</button>
        )}
        {showPrompts && prompts && !promptsError && (
          <span style={{ fontSize: 12, color: '#a6adc8' }}>{prompts.length} shot{prompts.length === 1 ? '' : 's'}</span>
        )}
      </div>
      {showPrompts && <PromptsPanel prompts={prompts} error={promptsError} />}

      {lastReport && <PassReport report={lastReport} />}
    </div>
  );
}

/** The text that produced each generated shot, from the last B-roll run. */
function PromptsPanel({ prompts, error }: { prompts: ShotPrompt[] | null; error: string | null }) {
  const fmt = (s?: number) => (s == null ? '' : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`);
  if (error) {
    return <div style={{ background: '#181825', padding: '12px', borderRadius: 8, fontSize: 12, color: '#a6adc8' }}>{error}</div>;
  }
  if (!prompts) {
    return <div style={{ background: '#181825', padding: '12px', borderRadius: 8, fontSize: 12, color: '#a6adc8' }}>Loading prompts…</div>;
  }
  if (prompts.length === 0) {
    return <div style={{ background: '#181825', padding: '12px', borderRadius: 8, fontSize: 12, color: '#a6adc8' }}>No prompts recorded for this project yet.</div>;
  }
  return (
    <div style={{ background: '#181825', padding: '12px', borderRadius: 8, display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 340, overflowY: 'auto' }}>
      {prompts.map((p, i) => {
        const text = p.image_prompt || p.video_prompt || p.popup_text || '(no prompt)';
        return (
          <div key={p.id || i} style={{ borderLeft: '3px solid #585b70', padding: '2px 0 2px 10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 11, color: '#a6adc8' }}>
              <span style={{ fontWeight: 600 }}>{i + 1}. {p.topic || p.kind}</span>
              <span>{fmt(p.start_s)}{p.end_s != null ? `–${fmt(p.end_s)}` : ''} · {p.kind?.replace('broll_', '')}</span>
            </div>
            <div style={{ fontSize: 12, color: '#cdd6f4', marginTop: 2, whiteSpace: 'pre-wrap' }}>{text}</div>
          </div>
        );
      })}
    </div>
  );
}

/** What the last pass actually did — planned vs placed vs rejected vs failed.
    A single "N clips" line hid every silent drop; this is where "75% coverage
    quietly became 20%" becomes visible instead. */
function PassReport({ report }: { report: any }) {
  const failed = report.assets_failed?.length ?? 0;
  const rejected = report.beats_placement_rejected?.length ?? 0;
  const target = report.coverage_target ?? 0;
  const achieved = report.coverage_achieved ?? 0;
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  // Trouble worth a red banner: images failing to generate, or coverage landing
  // at less than half of what was asked for.
  const bad = failed > 0 || (target > 0 && achieved < 0.5 * target);
  const rows: [string, string][] = [
    ['Video genre', report.genre || 'general'],
    ['Beats planned', String(report.beats_planned ?? 0)],
    ['B-roll placed', String(report.broll_placed ?? 0)],
    ['Placement rejected', String(rejected)],
    ['Assets generated / cached / failed',
      `${report.assets_generated ?? 0} / ${report.assets_cached ?? 0} / ${failed}`],
    ['Coverage', target > 0 ? `${pct(achieved)} of ${pct(target)} target` : pct(achieved)],
    ['Zooms (segment + windowed)',
      `${(report.zooms_segment ?? 0) + (report.zooms_windowed ?? 0)}`
      + (report.zooms_suppressed_by_broll ? ` (${report.zooms_suppressed_by_broll} under B-roll)` : '')],
    ['Pop-ups', String(report.popups_placed ?? 0)],
    ['Captions', String(report.captions ?? 0)],
    ['Cards', Object.entries(report.cards_placed || {}).map(([k, v]) => `${k.replace(/_/g, ' ')} ${v}`).join(', ') || 'none'],
    ['Maps', String(report.maps_placed ?? 0)],
    ['Sound', [report.music_used ? `bed ${report.music_used}` : 'no bed',
      `${report.sfx_placed ?? 0} effects`,
      report.ambience_used ? `ambience ${report.ambience_used}` : null,
      report.voice_preset ? `voice ${report.voice_preset}` : null].filter(Boolean).join(' · ')],
    ['Look', [report.grade_applied ? `grade ${report.grade_applied}` : null,
      report.atmosphere_applied ? `atmosphere ${report.atmosphere_applied}` : null,
      `${report.topic_transitions ?? 0} chapter transitions`].filter(Boolean).join(' · ')],
    ['Story', [(report.acts || []).length ? `acts ${report.acts.join(' → ')}` : null,
      (report.moods || []).length ? `moods ${report.moods.join(', ')}` : null,
      report.cold_open ? `cold open "${String(report.cold_open.text || '').slice(0, 40)}" (${report.cold_open.seconds}s)` : 'no cold open',
      `${report.mood_hits ?? 0} hits`].filter(Boolean).join(' · ')],
    ['Script', report.script_aligned_words
      ? `${report.script_aligned_words} words aligned, ${report.script_spelling_fixed} spellings fixed, ${report.script_directives} directions`
      : 'none'],
  ];
  const failedChecks = (report.verification || []).filter((c: any) => !c.ok);
  return (
    <div style={{ background: '#181825', padding: '14px', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <h4 style={{ margin: 0, color: '#a6adc8' }}>Last Pass</h4>
      {bad && (
        <div style={{ background: '#7f1d1d', color: '#fecaca', padding: '8px 10px', borderRadius: '6px', fontSize: 12 }}>
          {failed > 0 && <div>{failed} asset{failed === 1 ? '' : 's'} failed to generate
            {report.comfyui_online === false ? ' — ComfyUI was offline' : ''}.</div>}
          {target > 0 && achieved < 0.5 * target && (
            <div>Coverage reached only {pct(achieved)} of the {pct(target)} target.</div>
          )}
        </div>
      )}
      <table style={{ fontSize: 12, borderSpacing: 0 }}>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <td style={{ color: '#bac2de', padding: '2px 12px 2px 0' }}>{label}</td>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rejected > 0 && (
        <details style={{ fontSize: 11, color: '#a6adc8' }}>
          <summary>Why beats were rejected</summary>
          {report.beats_placement_rejected.map((r: any, i: number) => (
            <div key={i}>· {r.topic || r.beat_id || r.beat}: {r.reason}</div>
          ))}
        </details>
      )}
      {failed > 0 && (
        <details style={{ fontSize: 11, color: '#a6adc8' }}>
          <summary>Why assets failed</summary>
          {report.assets_failed.map((f: any, i: number) => (
            <div key={i}>· {f.beat_id}: {f.reason}</div>
          ))}
        </details>
      )}
      {failedChecks.length > 0 && (
        <div style={{ background: '#78350f', color: '#fde68a', padding: '8px 10px', borderRadius: '6px', fontSize: 12 }}>
          {failedChecks.map((c: any) => (
            <div key={c.name}>⚠ {c.name.replace(/_/g, ' ')}: {c.detail}</div>
          ))}
        </div>
      )}
      {(report.verification?.length ?? 0) > 0 && failedChecks.length === 0 && (
        <span style={{ fontSize: 11, color: '#a6e3a1' }}>
          All {report.verification.length} checks passed
          {(() => { const l = report.verification.find((c: any) => c.name === 'loudness'); return l?.value != null ? ` · ${l.value} LUFS` : ''; })()}
        </span>
      )}
      {(report.degraded?.length ?? 0) > 0 && (
        <span style={{ fontSize: 11, color: '#f59e0b' }}>
          Degraded: {report.degraded.join(', ')}
        </span>
      )}
    </div>
  );
}
