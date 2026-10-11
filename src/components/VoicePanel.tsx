/**
 * Voice panel -- SkinWalker inside BuzzEdit (VOICE_CHANGER_PLAN.md §2.7, brief S2b). Four tabs:
 *
 *   - Convert: one whole file (the selected clip's, a media-pool item, or a microphone
 *     recording uploaded into the pool) into a voice or a preset; the result is a new media-pool
 *     item that plays here and downloads as WAV or MP3.
 *   - Batch: several media-pool files into one voice + preset, with progress per file.
 *   - Regions: SkinWalker's selective conversion on the timeline -- `voice_regions` in programme
 *     seconds, each with its voice, preset, params and FX; Apply pending (one version), Undo.
 *   - Voices: the reference-voice library (record, upload, rename, delete, consent note) and
 *     the seed-vc worker's status with Stop.
 *
 * The library, the presets and the effect catalogue are loaded once here and shared. Every tab
 * stays mounted while another is shown (a job, a recording or a half-filled form survives a tab
 * switch). The Timeline's range menu and the Media pool's "Convert voice…" reach the panel
 * through `openVoicePanel()` (`hooks/voiceApi.ts`); a request waits until the panel takes it,
 * since a hidden panel is not mounted. A missing route is a note, never a crash.
 */

import { useCallback, useEffect, useState } from 'react';
import { useProjectStore } from '../hooks/store';
import { EffectDef, errorText, getEffects, routeMissing } from '../hooks/audioApi';
import {
  PresetsResponse, VoiceEntry, VoiceTab, getVoicePresets, listVoices, takeVoicePanelRequest,
  voicePanelRequests,
} from '../hooks/voiceApi';
import ConvertTab from './voice/ConvertTab';
import BatchTab from './voice/BatchTab';
import RegionsTab from './voice/RegionsTab';
import VoicesTab from './voice/VoicesTab';
import '../styles/audio.css';
import '../styles/voice.css';

const TABS: Array<{ id: VoiceTab; label: string; hint: string }> = [
  { id: 'convert', label: 'Convert', hint: 'One whole file into a voice: the selected clip, a media-pool item or a microphone recording.' },
  { id: 'batch', label: 'Batch', hint: 'Several media-pool files into one voice, each into a new media-pool item.' },
  { id: 'regions', label: 'Regions', hint: 'Parts of the programme in other voices, applied as one version; unmarked audio stays yours.' },
  { id: 'voices', label: 'Voices', hint: 'The reference voices (shared by every project) and the conversion engine.' },
];

/** Requests are numbered so a repeated one (the same media item twice) still reaches its tab. */
let requestSeq = 0;

function NoProject() {
  return <div className="fx-empty">Open a project to convert voices. The Voices tab works without one.</div>;
}

export default function VoicePanel() {
  const projectId = useProjectStore((s) => s.project?.id ?? null);
  const [tab, setTab] = useState<VoiceTab>('convert');

  // -- shared data: the library, the presets, the effect catalogue -----------------------------
  const [voices, setVoices] = useState<VoiceEntry[]>([]);
  const [voicesLoaded, setVoicesLoaded] = useState(false);
  const [voicesMissing, setVoicesMissing] = useState(false);
  const [voicesNote, setVoicesNote] = useState<string | null>(null);
  const [presets, setPresets] = useState<PresetsResponse | null>(null);
  const [presetsNote, setPresetsNote] = useState<string | null>(null);
  const [defs, setDefs] = useState<Record<string, EffectDef>>({});

  const loadVoices = useCallback(async () => {
    try {
      const res = await listVoices();
      setVoices(res.voices || []);
      setVoicesMissing(false);
      setVoicesNote(null);
    } catch (err) {
      if (routeMissing(err)) setVoicesMissing(true);
      else setVoicesNote(`Could not load the voices: ${errorText(err)}`);
    } finally {
      setVoicesLoaded(true);
    }
  }, []);

  const loadPresets = useCallback(async () => {
    try {
      setPresets(await getVoicePresets());
      setPresetsNote(null);
    } catch (err) {
      // A missing route is said once, by the voices banner; anything else here.
      setPresetsNote(routeMissing(err) ? null : `Could not load the presets: ${errorText(err)}`);
    }
  }, []);

  useEffect(() => {
    void loadVoices();
    void loadPresets();
    getEffects()
      .then((catalogue) => {
        const out: Record<string, EffectDef> = {};
        for (const def of catalogue.effects || []) out[def.id] = def;
        setDefs(out);
      })
      .catch(() => { /* the FX are named by their ids then */ });
  }, [loadVoices, loadPresets]);

  /** The library changed (Voices tab): the voices and the presets' voice_in_library. */
  const onLibraryChanged = useCallback(async () => {
    await Promise.all([loadVoices(), loadPresets()]);
  }, [loadVoices, loadPresets]);

  // -- requests from the timeline and the media pool -------------------------------------------
  // Each request belongs to the project open when it was made: a tab remounted for another
  // project must not replay it.
  const [convertRequest, setConvertRequest] = useState<{ mediaId: string; seq: number; projectId: string | null } | null>(null);
  const [regionsEpoch, setRegionsEpoch] = useState(0);
  const [regionsNote, setRegionsNote] = useState<{ note: string; seq: number; projectId: string | null } | null>(null);
  const [panelNote, setPanelNote] = useState<string | null>(null);

  useEffect(() => {
    const take = () => {
      const request = takeVoicePanelRequest();
      if (!request) return;
      requestSeq += 1;
      const seq = requestSeq;
      const owner = useProjectStore.getState().project?.id ?? null;
      if (request.tab) setTab(request.tab);
      if (request.mediaId) setConvertRequest({ mediaId: request.mediaId, seq, projectId: owner });
      if (request.reloadRegions) setRegionsEpoch((n) => n + 1);
      if (request.note) {
        if (!request.tab || request.tab === 'regions') setRegionsNote({ note: request.note, seq, projectId: owner });
        else setPanelNote(request.note);
      }
    };
    take();
    return voicePanelRequests.subscribe(take);
  }, []);

  const current = TABS.find((t) => t.id === tab) ?? TABS[0];

  return (
    <div className="voice-panel" tabIndex={0}>
      <div className="fx-scope voice-tabs">
        {TABS.map((t) => (
          <button key={t.id} type="button" className={`insp-chip${tab === t.id ? ' active' : ''}`} title={t.hint}
            onClick={() => { setTab(t.id); setPanelNote(null); }}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="voice-tab-hint">{current.hint}</div>

      {voicesMissing && (
        <div className="audio-note">
          The voice routes are not available on this backend yet (it predates them; restart BuzzEdit to load them).
          The panel works again once they answer.
        </div>
      )}
      {presetsNote && <div className="audio-note">{presetsNote}</div>}
      {panelNote && <div className="audio-note">{panelNote}</div>}

      <div hidden={tab !== 'convert'}>
        {projectId ? (
          <ConvertTab key={projectId} projectId={projectId} voices={voices} presets={presets} defs={defs}
            request={convertRequest && convertRequest.projectId === projectId ? convertRequest : null} />
        ) : <NoProject />}
      </div>
      <div hidden={tab !== 'batch'}>
        {projectId ? (
          <BatchTab key={projectId} projectId={projectId} voices={voices} presets={presets} defs={defs} />
        ) : <NoProject />}
      </div>
      <div hidden={tab !== 'regions'}>
        {projectId ? (
          <RegionsTab key={projectId} projectId={projectId} voices={voices} presets={presets} defs={defs}
            reloadEpoch={regionsEpoch} incoming={regionsNote && regionsNote.projectId === projectId ? regionsNote : null} />
        ) : <NoProject />}
      </div>
      <div hidden={tab !== 'voices'}>
        <VoicesTab voices={voices} loaded={voicesLoaded} missing={voicesMissing} loadNote={voicesNote}
          presets={presets} active={tab === 'voices'} onChanged={onLibraryChanged} />
      </div>
    </div>
  );
}
