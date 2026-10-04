"""The AI editor as the auto-edit planner (AUTO_CUT_EDITOR_PLAN.md Phase 5).

`plan_auto_edit` calls this when the project's `planner` is "editor" (the default). It returns
words annotated exactly the way the classic planner (fumble_engine.refine_disfluencies) does,
`disfluency` / `enabled` / `reason` per word plus the report on the first word, so the
timeline build, the cut audits and every entry point downstream are unchanged.

What it does:
1. The audio passes the classic planner already runs (fumble_engine.plan_audio_cuts): word
   timings repaired against the speech map, and the speech map and loudness envelope the
   timeline needs to place cuts. Not its filler-sound detector (see plan_with_editor_sync).
2. The utterance table (asr.utterances): every stretch of speech with what the audio showed.
3. The editor (asr.editor) reads the whole table AGREEMENT_RUNS times in parallel. A line is
   removed only when every read removed it; a trim only when every read made it. Measured in
   §12: that took Life3Baje ep1 from 8.2 s of good speech wrongly removed (one read) to 1.3 s,
   where the full second-opinion pipeline cost four times as much and did no better.
4. Lines the reads disagreed on are kept (§10 item 3: doubt resolves to keeping) and listed in
   the report's review list, the "decided under doubt" report the Cuts view shows.

Raises EditorUnavailable when the editor cannot run (no proxy key, proxy down, the proxy served
another model, fewer than two reads answered); plan_auto_edit then runs the classic planner and
says so in the report.
"""

import asyncio
import concurrent.futures as cf
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("editor_planner")

AGREEMENT_RUNS = 3
MIN_RUNS = 2                 # fewer reads than this answering is no agreement at all
REVIEW_CONTEXT = 2           # utterances either side stored with a review item


class EditorUnavailable(RuntimeError):
    pass


def _style_for(settings: Dict[str, Any], exclude_source: str = "") -> str:
    from presentation.genre import editing_notes_for
    from .style import stored_style
    return stored_style(channel=str(settings.get("glossary") or settings.get("channel") or ""),
                        exclude_source=exclude_source,
                        notes=editing_notes_for(settings.get("genre")))


def _placeholder(start: float, end: float, reason: str) -> Dict[str, Any]:
    """Speech no word covers that the editor removed, as a disabled word, so the timeline
    treats it as a deliberate cut (never bridged back) and the Cuts view can show it."""
    return {"word": "[…]", "word_native": "[…]", "hinglish": "[…]", "start": round(start, 3),
            "end": round(end, 3), "probability": 0.0, "disfluency": True, "candidate": False,
            "reason": reason, "untranscribed": True}


def _agree(passes, utterances) -> Tuple[Dict[str, str], Dict[int, str], Dict[str, int]]:
    """(unanimously removed uid -> reason, unanimous trims word index -> reason,
    disputed uid -> how many reads removed it)."""
    from .double_check import _removed_uids, _trimmed
    removed_sets = [_removed_uids(d, utterances) for d in passes]
    union = set().union(*[set(r) for r in removed_sets])
    votes = {uid: sum(uid in r for r in removed_sets) for uid in union}
    unanimous = {uid: next(r[uid] for r in removed_sets if uid in r)
                 for uid, n in votes.items() if n == len(passes)}
    disputed = {uid: n for uid, n in votes.items() if n < len(passes)}
    trims = _trimmed(passes[0], utterances)
    for d in passes[1:]:
        other = _trimmed(d, utterances)
        trims = {i: r for i, r in trims.items() if i in other}
    return unanimous, trims, disputed


def _reasons(answers: Sequence[str]) -> Dict[str, str]:
    """uid -> the editor's own 'why' (first read that gave one), for tooltips and the review."""
    from .editor import first_json
    out: Dict[str, str] = {}
    for text in answers:
        data = first_json(text or "") or {}
        for g in data.get("groups") or []:
            members = [m for m in (g.get("utterances") or []) if m != g.get("keep")]
            for uid in members:
                out.setdefault(uid, f"{g.get('why') or 'an earlier attempt'} (kept {g.get('keep')})")
        for d in data.get("drops") or []:
            for uid in d.get("utterances") or []:
                out.setdefault(uid, str(d.get("why") or d.get("kind") or ""))
    return out


def _review_items(utterances, disputed: Dict[str, int], runs: int, why: Dict[str, str]) -> List[Dict[str, Any]]:
    from .utterances import table
    index = {u.uid: n for n, u in enumerate(utterances)}
    items = []
    for uid in sorted(disputed, key=lambda u: index[u]):
        n = index[uid]
        u = utterances[n]
        twin = next((other for other, _score in u.similar), None)
        window = utterances[max(0, n - REVIEW_CONTEXT):n + REVIEW_CONTEXT + 1]
        items.append({
            "take": uid, "start": round(u.start, 2), "end": round(u.end, 2), "text": u.text,
            "decision": "kept", "votes": f"{disputed[uid]} of {runs} reads removed it",
            "why": why.get(uid, ""), "twin": twin,
            "twin_start": round(utterances[index[twin]].start, 2) if twin in index else None,
            "context": table(window), "kind": "doubt", "answer": None,
        })
    return items


def _run_reads(utterances, words, client, style: str, runs: int):
    from . import editor
    passes, answers, metas, errors = [], [], [], []

    def one(_):
        return editor.edit(utterances, words, client, style)

    with cf.ThreadPoolExecutor(max_workers=runs) as pool:
        futures = [pool.submit(one, n) for n in range(runs)]
        for f in futures:
            try:
                decisions, meta = f.result()
            except Exception as e:                 # one read failing is not the edit failing
                logger.warning("Editor read failed (%s: %s); agreement over the reads that answered.",
                               type(e).__name__, e)
                errors.append(str(e))
                continue
            passes.append(decisions)
            answers.append(meta.pop("answer", ""))
            metas.append(meta)
    return passes, answers, metas, errors


def plan_with_editor_sync(words: List[Dict[str, Any]], audio_path: Optional[str],
                          settings: Optional[Dict[str, Any]] = None, client=None,
                          exclude_source: str = "", runs: int = AGREEMENT_RUNS) -> List[Dict[str, Any]]:
    """The whole editor plan; blocking (model calls and audio analysis). See module doc."""
    from faster_whisper.audio import decode_audio
    from . import utterances as utt
    from .editor import DEFAULT_EFFORT, DEFAULT_MODEL, EditorClient, proxy_config
    from .fumble_engine import EditReport, plan_audio_cuts
    from .gap_recovery import speech_regions

    settings = settings or {}
    started = time.time()
    if client is None:
        proxy = proxy_config(settings)
        if not proxy["api_key"]:
            raise EditorUnavailable("no key for the Claude proxy (set the auto_cut_editor app "
                                    "setting or CLAUDE_API_KEY)")
        client = EditorClient(base_url=proxy["base_url"], api_key=proxy["api_key"])
    if not audio_path:
        raise EditorUnavailable("no audio to read the recording from")

    report = EditReport(words_total=len(words))
    # No filler-sound detection: it labels any speech the transcript missed a filler and cuts
    # it, and on Life3Baje ep1 that alone removed 34.3 s the creator kept (264 "fillers"). The
    # editor sees that speech as "[speech, no transcript]" lines and judges it in context.
    words = plan_audio_cuts(words, audio_path, report, detect_fillers=False)
    if not report.used_audio or not words:
        raise EditorUnavailable("the audio could not be analysed")

    # The editor reads the transcript; the filler sounds the audio pass added stay cut on
    # their own and stay out of its table (they were never in the recordings it learnt from).
    lexical = [i for i, w in enumerate(words) if not w.get("detected")]
    view = [words[i] for i in lexical]
    audio = decode_audio(audio_path, sampling_rate=16000)
    regions = speech_regions(audio)
    utterances = utt.build(view, regions, audio=audio)
    if not utterances:
        raise EditorUnavailable("nothing to edit")

    style = _style_for(settings, exclude_source)
    passes, answers, metas, errors = _run_reads(utterances, view, client, style, runs)
    if len(passes) < min(MIN_RUNS, runs):
        raise EditorUnavailable("; ".join(dict.fromkeys(errors)) or "the editor did not answer")

    removed, trims, disputed = _agree(passes, utterances)
    why = _reasons(answers)
    by_id = {u.uid: u for u in utterances}

    for u in utterances:
        for i in u.word_indices:
            words[lexical[i]]["take"] = u.uid
    for uid, reason in removed.items():
        u = by_id[uid]
        for i in u.word_indices:
            w = words[lexical[i]]
            w.update(disfluency=True, candidate=False, reason=reason, note=why.get(uid) or None)
    for i, reason in trims.items():
        words[lexical[i]].update(disfluency=True, candidate=False, reason=reason)
    for uid in disputed:
        for i in by_id[uid].word_indices:
            words[lexical[i]]["candidate"] = True
            words[lexical[i]]["note"] = f"kept under doubt: {why.get(uid) or 'the reads disagreed'}"

    # Speech no word covers, inside utterances the editor removed: cut it too.
    extra = []
    for uid, reason in removed.items():
        u = by_id[uid]
        if u.untranscribed_s <= 0:
            continue
        covered = sorted((float(view[i]["start"]), float(view[i]["end"])) for i in u.word_indices)
        for a, b in regions:
            a, b = max(a, u.start), min(b, u.end)
            if b - a < 0.15:
                continue
            gaps, t = [], a
            for s, e in covered:
                if s > t:
                    gaps.append((t, min(s, b)))
                t = max(t, e)
            if t < b:
                gaps.append((t, b))
            for s, e in gaps:
                if e - s >= 0.15:
                    p = _placeholder(s, e, reason)
                    p["take"], p["note"] = uid, why.get(uid) or None
                    extra.append(p)
    if extra:
        words = sorted(words + extra, key=lambda w: float(w.get("start", 0.0)))

    reasons: Dict[str, int] = {}
    for w in words:
        w["enabled"] = not w.get("disfluency", False)
        if w.get("disfluency"):
            reasons[w.get("reason") or "unspecified"] = reasons.get(w.get("reason") or "unspecified", 0) + 1
    report.words_total = len(words)
    report.words_cut = sum(1 for w in words if not w["enabled"])
    report.reasons = reasons
    report.used_llm = report.used_fluency = True

    tokens_in = sum((m.get("usage") or {}).get("prompt_tokens", 0) for m in metas)
    tokens_out = sum((m.get("usage") or {}).get("completion_tokens", 0) for m in metas)
    out = report.as_dict()
    out["planner"] = "editor"
    out["editor"] = {
        "model": metas[0].get("model") if metas else DEFAULT_MODEL, "effort": DEFAULT_EFFORT,
        "reads": len(passes), "reads_failed": len(errors), "utterances": len(utterances),
        "removed_takes": len(removed), "trims": len(trims), "kept_under_doubt": len(disputed),
        "untranscribed_cut_s": round(sum(float(p["end"]) - float(p["start"]) for p in extra), 1),
        "refused": sum(len(d.refused) for d in passes), "style_examples": style.count("What the creator did"),
        "tokens_in": tokens_in, "tokens_out": tokens_out, "seconds": round(time.time() - started, 1),
    }
    out["review"] = _review_items(utterances, disputed, len(passes), why)
    words[0]["_edit_report"] = out
    logger.info("Editor plan: %s", {k: v for k, v in out["editor"].items()})
    return words


async def plan_with_editor(words: List[Dict[str, Any]], audio_path: Optional[str],
                           settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(plan_with_editor_sync, [dict(w) for w in words], audio_path, settings)
