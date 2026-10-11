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
   removed when MORE THAN HALF of the reads removed it (2 of 3; 2 of 2 when one read failed);
   a trim likewise. Until 2026-10-08 only a unanimous removal counted: on Raat3Baje ep2 the
   creator restarted the whole video at U120 and two reads removed the abandoned first attempt
   (U18-U118, 4 minutes 46 seconds of the cut), one did not, and every line of it shipped --
   the intro and the story opening twice. The creator's rule since: decide and cut, no
   "decided under doubt" list to answer.
4. Candidates (AUTO_CUT_V3_PLAN.md §2.2-3): a minority removal (1 of 3), and every attempt the
   retake sweep (asr.retake_sweep) finds that is not the latest of its group, get one focused
   question (double_check.verify_candidates). A candidate goes only on a "remove"; one not
   removed stays, noted on the word for the tooltip.
5. The deterministic stutter rule (asr.disfluency: a word said back to back, a lexical filler
   sound) runs over the kept words as well. The editor was the only thing cutting stutters, and
   it cuts one only when enough reads name the identical words, so 85 "jo jo" / "thi thi" /
   "bahut" x10 survived the same cut. Then the phrase-repeat rule (asr.repeats): a phrase said
   twice back to back loses its earlier copy.
6. The chunked final read (double_check.final_read_chunked) of the programme as it plays, then
   both rules once more as a guard. The report counts the phrase repeats still playing
   (repeats_remaining) and the restarts of a broken-off line that survived
   (false_starts_remaining).

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


def _uncovered_inside(a: float, b: float, covered: Sequence[Tuple[float, float]],
                      regions: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """The stretches of at least 0.15 s inside [a, b] that a speech region holds and none of
    `covered` (sorted (start, end) word spans) does: speech the transcript has no word for."""
    out: List[Tuple[float, float]] = []
    for ra, rb in regions:
        ra, rb = max(ra, a), min(rb, b)
        if rb - ra < 0.15:
            continue
        gaps, t = [], ra
        for s, e in covered:
            if s > t:
                gaps.append((t, min(s, rb)))
            t = max(t, e)
        if t < rb:
            gaps.append((t, rb))
        out.extend((s, e) for s, e in gaps if e - s >= 0.15)
    return out


def _agree(passes, utterances) -> Tuple[Dict[str, str], Dict[int, str], Dict[str, Dict[str, Any]]]:
    """(uid removed by a majority of the reads -> reason, word index trimmed by a majority ->
    reason, uid removed by a minority (kept unless the verifier removes it) -> {"votes": how many
    reads removed it, "reason": the first such read's reason})."""
    from .double_check import _removed_uids, _trimmed
    removed_sets = [_removed_uids(d, utterances) for d in passes]
    needed = len(passes) // 2 + 1                      # 2 of 3, 2 of 2, 3 of 4
    union = set().union(*[set(r) for r in removed_sets])
    votes = {uid: sum(uid in r for r in removed_sets) for uid in union}
    removed = {uid: next(r[uid] for r in removed_sets if uid in r)
               for uid, n in votes.items() if n >= needed}
    minority = {uid: {"votes": n, "reason": next(r[uid] for r in removed_sets if uid in r)}
                for uid, n in votes.items() if n < needed}
    trim_sets = [_trimmed(d, utterances) for d in passes]
    trim_votes: Dict[int, int] = {}
    for t in trim_sets:
        for i in t:
            trim_votes[i] = trim_votes.get(i, 0) + 1
    trims = {i: next(t[i] for t in trim_sets if i in t) for i, n in trim_votes.items() if n >= needed}
    return removed, trims, minority


def _false_starts(utterances, removed: Dict[str, str], view: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Kept lines that break off ("cut off mid-sound", "ends on a connective (unfinished)") and
    whose next kept line opens the same way: a restart of the same line that survived. A kept
    line is one not removed with a word that plays."""
    marks = ("cut off mid-sound", "ends on a connective (unfinished)")
    kept = [u for u in utterances if u.uid not in removed
            and any(view[i].get("enabled", not view[i].get("disfluency")) for i in u.word_indices)]
    out = []
    for u, v in zip(kept, kept[1:]):
        if any(mark in note for note in u.notes for mark in marks) and (
                u.uid in v.same_opening or v.uid in u.same_opening):
            out.append({"take": u.uid, "start": round(float(u.start), 2), "text": u.text[:80]})
    return out


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


def _run_reads(utterances, words, client, style: str, runs: int, speaker_style: str = ""):
    from . import editor
    passes, answers, metas, errors = [], [], [], []

    def one(_):
        return editor.edit(utterances, words, client, style, speaker_style=speaker_style)

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
        client = EditorClient(base_url=proxy["base_url"], api_key=proxy["api_key"],
                              model=proxy["model"], effort=proxy["effort"])
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
    script_text = settings.get("script_text")
    utterances = utt.build(view, regions, audio=audio,
                           script_text=utt.spoken_script(script_text) if script_text else None)
    if not utterances:
        raise EditorUnavailable("nothing to edit")

    style = _style_for(settings, exclude_source)
    speaker_style = str(settings.get("speaker_style") or "")[:1500]
    passes, answers, metas, errors = _run_reads(utterances, view, client, style, runs,
                                                speaker_style=speaker_style)
    if len(passes) < min(MIN_RUNS, runs):
        raise EditorUnavailable("; ".join(dict.fromkeys(errors)) or "the editor did not answer")

    removed, trims, minority = _agree(passes, utterances)
    why = _reasons(answers)
    stutters = 0
    steps_failed: List[str] = []

    # A restart from the top is decided by the greeting, not by the reads
    # (asr/restart.py): everything before the last early greeting goes.
    from .restart import restart_from_top
    restarted = restart_from_top(utterances, max((float(u.end) for u in utterances), default=0.0), view)
    for uid, note in restarted.items():
        if uid not in removed:
            removed[uid] = "retake"
            why.setdefault(uid, note)
        minority.pop(uid, None)
    by_id = {u.uid: u for u in utterances}
    order = {u.uid: n for n, u in enumerate(utterances)}

    # Candidates (AUTO_CUT_V3_PLAN.md §2.2-3): a removal only a minority of the reads made, and
    # every attempt the retake sweep finds that is not the latest of its group. One focused
    # question decides them. The reads take the whole proxy pool, so these calls come after.
    from .double_check import Report as CallLog, final_read_chunked, verify_candidates
    calls = CallLog()
    candidates: Dict[str, Dict[str, Any]] = {uid: {"reason": m["reason"], "why": why.get(uid, "")}
                                             for uid, m in minority.items() if uid not in removed}
    groups: List[List[str]] = []
    swept: Dict[str, int] = {}
    try:
        from .retake_sweep import sweep
        groups, swept = sweep(utterances, client)
        if swept.get("failed"):
            steps_failed.append("sweep")
        for group in groups:
            alive = sorted((uid for uid in group if uid not in removed), key=lambda uid: order[uid])
            for uid in alive[:-1]:
                candidates.setdefault(uid, {"reason": "retake", "twin": alive[-1],
                                            "why": f"sweep: an attempt at the same content as {alive[-1]}"})
    except Exception as e:                     # the sweep failing is not the edit failing
        logger.warning("Retake sweep failed (%s: %s); the edit goes on without it.", type(e).__name__, e)
        steps_failed.append("sweep")

    verdicts: Dict[str, Dict[str, Any]] = {}
    if candidates:
        try:
            verdicts = verify_candidates(utterances, candidates, removed, client, report=calls)
        except Exception as e:                 # unverified, every candidate stays, as before
            logger.warning("Verifying %d candidates failed (%s: %s); all kept.", len(candidates),
                           type(e).__name__, e)
            steps_failed.append("verify")
    kept_notes: Dict[str, str] = {}
    leaning: List[Tuple[str, Optional[str]]] = []       # (removed by the verifier, its kept twin)
    for uid in sorted(verdicts, key=lambda uid: order.get(uid, 0)):
        verdict, said = verdicts[uid]["verdict"], verdicts[uid]["why"]
        covered_by = verdicts[uid]["covered_by"]
        if uid in candidates and verdict == "remove":
            removed[uid] = candidates[uid]["reason"]
            why[uid] = said or why.get(uid, "")
            leaning.append((uid, covered_by))
        elif uid in candidates:
            kept_notes[uid] = "kept after verification" + (f" -- {said}" if said else "")
        elif verdict == "remove" and uid not in removed:
            # a twin that was not in question goes only for a line that is kept
            if covered_by and covered_by != uid and covered_by not in removed:
                removed[uid] = "retake"
                why[uid] = said or f"another attempt at {covered_by}"
                leaning.append((uid, covered_by))
    for uid, covered_by in leaning:            # both attempts removed: the later one stays
        if covered_by and uid in removed and covered_by in removed:
            earlier, later = sorted((uid, covered_by), key=lambda u: order[u])
            del removed[later]
            kept_notes[later] = f"kept after verification -- the later attempt; {earlier} is cut"
    verified_removed = sum(1 for uid, _ in leaning if uid in removed)
    verified_kept = sum(1 for uid in candidates if uid in verdicts and uid not in removed)

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
    for uid in list(minority) + [uid for uid in kept_notes if uid not in minority]:
        if uid in removed:
            continue
        note = kept_notes.get(uid) or (f"kept: {minority[uid]['votes']} of {len(passes)} reads removed it"
                                       + (f" -- {why[uid]}" if why.get(uid) else ""))
        for i in by_id[uid].word_indices:
            words[lexical[i]]["note"] = note

    # The rule-based stutter pass over what the editor kept: a word said back to back (the
    # earlier copy goes), a lexical filler sound. Safe without a model; the editor alone
    # left 85 of them in one cut.
    from .disfluency import analyze_disfluencies

    def stutter_rule() -> int:
        kept_view = [i for i in lexical if not words[i].get("disfluency")]
        flagged_n = 0
        for i, flagged in zip(kept_view, analyze_disfluencies([words[i] for i in kept_view])):
            if flagged.get("disfluency") and flagged.get("reason") in ("stutter", "filler"):
                words[i].update(disfluency=True, candidate=False, reason=flagged["reason"])
                flagged_n += 1
        return flagged_n

    stutters += stutter_rule()

    # A phrase said twice back to back (asr/repeats.py, V3 §2.4): the earlier copy goes. In a
    # storytelling genre a clean, paused repeat may be said for effect, so the verifier decides
    # those, in one call.
    from presentation.genre import editing_notes_for
    from . import repeats
    storytelling = bool(editing_notes_for(settings.get("genre")))

    def kept_words() -> List[Dict[str, Any]]:
        kept = [words[i] for i in lexical if not words[i].get("disfluency")]
        for w in kept:
            w["enabled"] = True                # what the repeat scan reads; every word is set below
        return kept

    phrase_repeats = 0
    pending: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    for _ in range(3):
        kept = kept_words()
        cut, found = repeats.cut_phrase_repeats(kept, storytelling)
        phrase_repeats += cut
        pending = [(rep, [kept[j] for j in rep["first"]]) for rep in found]
        if not cut:
            break
    if storytelling and pending:
        asked = {f"R{n}": {"reason": "repeat", "why": "", "uid": first[0].get("take"), "text": rep["text"]}
                 for n, (rep, first) in enumerate(pending, 1) if first[0].get("take") in by_id}
        said_r: Dict[str, Dict[str, Any]] = {}
        try:
            if asked:
                said_r = verify_candidates(utterances, asked, removed, client, report=calls)
        except Exception as e:                 # unverified, the repeats stay
            logger.warning("Verifying %d phrase repeats failed (%s: %s); all kept.", len(asked),
                           type(e).__name__, e)
            steps_failed.append("repeat_verify")
        for n, (rep, first) in enumerate(pending, 1):
            if (said_r.get(f"R{n}") or {}).get("verdict") == "remove":
                for w in first:
                    w.update(disfluency=True, candidate=False, enabled=False, reason="repeat")
                phrase_repeats += 1

    # The final read (V3 §2.8): the programme as it plays, in overlapping chunks. A removal
    # needs a kept line that carries its content; the restart rule's lines never come back.
    read: Dict[str, Any] = {"remove": {}, "trims": {}, "restore": set(), "calls": 0, "tokens_in": 0,
                            "tokens_out": 0, "failed": 0}
    try:
        read = final_read_chunked(utterances, removed, view, client, protected=set(restarted))
        if read.get("failed"):
            steps_failed.append("final_read")
    except Exception as e:                     # the edit stands without it
        logger.warning("The final read failed (%s: %s); the edit stands without it.", type(e).__name__, e)
        steps_failed.append("final_read")
    final_removed = final_restored = final_trims = 0
    for uid, (covered_by, said) in read["remove"].items():
        if uid in removed:
            continue
        removed[uid] = "final_read"
        why[uid] = said or f"said again in {covered_by}"
        for i in by_id[uid].word_indices:
            words[lexical[i]].update(disfluency=True, candidate=False, reason="final_read", note=why[uid])
        final_removed += 1
    for uid in read["restore"]:
        if uid not in removed or uid in restarted:
            continue
        del removed[uid]
        for i in by_id[uid].word_indices:
            words[lexical[i]].update(disfluency=False, reason=None, note="restored by the final read")
        final_restored += 1
    for i in read["trims"]:
        if not words[lexical[i]].get("disfluency"):
            words[lexical[i]].update(disfluency=True, candidate=False, reason="stutter")
            final_trims += 1
    # The rules once more over what now plays, as a guard.
    stutters += stutter_rule()
    cut, _ = repeats.cut_phrase_repeats(kept_words(), storytelling)
    phrase_repeats += cut

    # Speech no word covers, inside utterances the editor removed: cut it too.
    extra = []
    for uid, reason in removed.items():
        u = by_id[uid]
        if u.untranscribed_s <= 0:
            continue
        covered = sorted((float(view[i]["start"]), float(view[i]["end"])) for i in u.word_indices)
        for s, e in _uncovered_inside(u.start, u.end, covered, regions):
            p = _placeholder(s, e, reason)
            p["take"], p["note"] = uid, why.get(uid) or None
            extra.append(p)
    # And inside a trim whose words are cut: the stumble the transcript only half heard
    # ("बारिश के बाद [speech, no transcript] मिट्टी") goes with the words around it.
    trim_ranges: Dict[Tuple[str, int, int], str] = {}
    for d in passes:
        for uid, first, last in d.trim_ranges:
            trim_ranges.setdefault((uid, first, last), d.removed_words.get(first) or "stutter")
    for (uid, first, last), reason in trim_ranges.items():
        u = by_id.get(uid)
        span = [i for i in u.word_indices if first <= i <= last] if u else []
        if not span or not all(words[lexical[i]].get("disfluency") for i in span):
            continue
        # what is already cut in this take counts as covered, so overlapping trims cut a hole once
        covered = sorted([(float(view[i]["start"]), float(view[i]["end"])) for i in u.word_indices]
                         + [(float(p["start"]), float(p["end"])) for p in extra if p.get("take") == uid])
        a = max(u.start, float(view[first]["start"]) - 0.3)
        b = min(u.end, float(view[last]["end"]) + 0.3)
        for s, e in _uncovered_inside(a, b, covered, regions):
            p = _placeholder(s, e, reason)
            p["take"] = uid
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

    every_call = metas + calls.calls             # the reads, the verify calls; sweep and final read below
    tokens_in = (sum((m.get("usage") or {}).get("prompt_tokens", 0) for m in every_call)
                 + swept.get("tokens_in", 0) + read.get("tokens_in", 0))
    tokens_out = (sum((m.get("usage") or {}).get("completion_tokens", 0) for m in every_call)
                  + swept.get("tokens_out", 0) + read.get("tokens_out", 0))
    remaining = repeats.repeats_remaining(words)
    false_starts = _false_starts(utterances, removed, view)
    out = report.as_dict()
    out["planner"] = "editor"
    out["editor"] = {
        "model": (metas[0].get("model") if metas else None) or getattr(client, "model", None) or DEFAULT_MODEL,
        "effort": getattr(client, "effort", None) or DEFAULT_EFFORT,
        "reads": len(passes), "reads_failed": len(errors), "utterances": len(utterances),
        "removed_takes": len(removed), "trims": len(trims),
        "kept_by_minority": sum(1 for uid in minority if uid not in removed),
        "stutters": stutters, "restart_cut": len(restarted),
        "untranscribed_cut_s": round(sum(float(p["end"]) - float(p["start"]) for p in extra), 1),
        "refused": sum(len(d.refused) for d in passes), "style_examples": style.count("What the creator did"),
        # The refusal notes themselves (what the guard would not apply, and why), for the bench.
        "refused_notes": [note for d in passes for note in d.refused][:300],
        "candidates": len(candidates), "verified_removed": verified_removed, "verified_kept": verified_kept,
        "sweep_groups": len(groups), "sweep_calls": swept.get("calls", 0),
        "phrase_repeats_cut": phrase_repeats, "final_read_removed": final_removed,
        "final_read_restored": final_restored, "final_read_trims": final_trims,
        "steps_failed": list(dict.fromkeys(steps_failed)),
        "repeats_remaining": remaining["count"], "repeats_examples": remaining["examples"],
        "false_starts_remaining": len(false_starts), "false_starts_examples": false_starts[:10],
        "tokens_in": tokens_in, "tokens_out": tokens_out, "seconds": round(time.time() - started, 1),
    }
    out["review"] = []        # no "decided under doubt" list: the majority decided (2026-10-08)
    words[0]["_edit_report"] = out
    logger.info("Editor plan: %s", {k: v for k, v in out["editor"].items()})
    return words


async def plan_with_editor(words: List[Dict[str, Any]], audio_path: Optional[str],
                           settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(plan_with_editor_sync, [dict(w) for w in words], audio_path, settings)
