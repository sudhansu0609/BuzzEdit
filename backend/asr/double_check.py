"""Double-check the editor's cut before anything ships (AUTO_CUT_EDITOR_PLAN.md §4.5).

One editing pass is one opinion. The measured failure of the first passes was not missing
fumbles but removing content the creator kept (a story's setup, an aside about the channel), so
every check here leans towards keeping:

1. **Agreement.** The editor reads the recording several times; a line is removed outright only
   when every pass removed it.
2. **Second opinion.** The lines the passes disagreed on go back to the editor in one focused
   call, each with its neighbours. Still unclear: between attempts the later one stays (the
   creator's rule), anything else stays.
3. **Content-loss check.** Each removed line is set beside what the viewer will still hear; if it
   said something nowhere else in the kept programme (a name, a number, a beat of the story), it
   comes back.
4. **Final read.** The kept programme, top to bottom, as the viewer hears it: a repeat or broken
   sentence that survived is removed, and a missing link is restored from what was cut.

Every call is one request for the whole recording, not one per line: each request through the
proxy carries about 4k tokens of the CLI's own prompt.
"""

import concurrent.futures as cf
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import editor
from .editor import Decisions, EditorClient

logger = logging.getLogger("double_check")

CONTEXT_LINES = 3

SECOND_OPINION = """You are the video editor for a Hindi/English (Hinglish) talking-head recording. Several
reads of the recording disagreed about the lines marked >> below. For each marked line decide
whether the viewer should hear it.

Remove a marked line only if it is an earlier attempt at something said again nearby, an abandoned
or false start, recording chatter, or filler. If two attempts are equally good, keep the LATER one.
If you are unsure, keep it. Spellings come from speech recognition and are often wrong.

Answer with one JSON object and nothing else: {"U12": "remove", "U14": "keep", ...} with every
marked line."""

CONTENT_CHECK = """You are checking a video edit for lost content. Below are lines the editor REMOVED, each
with the kept lines around it. For each removed line, decide whether removing it loses something the
viewer will not hear anywhere in the kept programme: a fact, a name, a number, a step of the story,
a joke or an emotional beat. If a kept line says the same thing (another take of it, even reworded),
nothing is lost. Spellings come from speech recognition and are often wrong.

Answer with one JSON object and nothing else: {"U12": "covered", "U14": "lost: <what is lost>", ...}
with every removed line."""

FINAL_READ = """You are doing the last read of a video edit: the lines below are exactly what the viewer
will hear, in order (removed lines are listed separately at the end). Find what still does not
play as one fluent take:
- a sentence said twice in a row, or an idea repeated because two takes both survived;
- a broken or abandoned sentence;
- a jump where something needed to understand the next line was cut.

Fix it only with: "remove" (kept line ids, or a trim of exact words) and "restore" (removed line ids).
Never remove content said nowhere else. Spellings come from speech recognition and are often wrong.

Answer with one JSON object and nothing else:
{"remove": ["U12"], "trims": [{"utterance": "U9", "remove": "exact words"}], "restore": ["U30"]}"""


@dataclass
class Report:
    runs: int = 0
    unanimous: int = 0
    disputed: int = 0
    second_opinion_removed: int = 0
    restored_by_content_check: int = 0
    final_read_removed: int = 0
    final_read_restored: int = 0
    calls: List[Dict[str, Any]] = field(default_factory=list)


def _removed_uids(d: Decisions, utterances: Sequence[Any]) -> Dict[str, str]:
    out = {}
    for u in utterances:
        reasons = [d.removed_words[i] for i in u.word_indices if i in d.removed_words]
        if u.word_indices and len(reasons) == len(u.word_indices):
            out[u.uid] = reasons[0]
        elif not u.word_indices and any(abs(s - u.start) < 1e-6 for s, _e, _r in d.removed_spans):
            out[u.uid] = next(r for s, _e, r in d.removed_spans if abs(s - u.start) < 1e-6)
    return out


def _trimmed(d: Decisions, utterances: Sequence[Any]) -> Dict[int, str]:
    """Word removals that are trims (part of an utterance), keyed by word index."""
    out = {}
    for u in utterances:
        hit = [i for i in u.word_indices if i in d.removed_words]
        if hit and len(hit) < len(u.word_indices):
            out.update({i: d.removed_words[i] for i in hit})
    return out


def _context(utterances: Sequence[Any], marked: Set[str], removed: Set[str]) -> str:
    from .utterances import table
    index = {u.uid: n for n, u in enumerate(utterances)}
    keep_lines: Set[int] = set()
    for uid in marked:
        n = index[uid]
        keep_lines.update(range(max(0, n - CONTEXT_LINES), min(len(utterances), n + CONTEXT_LINES + 1)))
    lines, last = [], None
    for n in sorted(keep_lines):
        if last is not None and n != last + 1:
            lines.append("...")
        u = utterances[n]
        prefix = ">> " if u.uid in marked else ("(cut) " if u.uid in removed else "   ")
        lines.append(prefix + table([u]))
        last = n
    return "\n".join(lines)


def _ask_json(client: EditorClient, system: str, user: str, report: Report, step: str) -> Dict[str, Any]:
    text, meta = client.ask(system, user)
    meta = {k: v for k, v in meta.items() if k != "answer"}
    meta["step"] = step
    report.calls.append(meta)
    return editor.first_json(text) or {}


def check(utterances: Sequence[Any], words: Sequence[Dict[str, Any]], client: EditorClient,
          style: str = "", runs: int = 3, final_read: bool = True) -> Tuple[Decisions, Report]:
    """The full §4.5 pipeline around the editor. Returns the decisions that ship and a report."""
    report = Report(runs=runs)
    by_id = {u.uid: u for u in utterances}

    def one_pass(_):
        decisions, meta = editor.edit(utterances, words, client, style)
        report.calls.append({k: v for k, v in meta.items() if k != "answer"} | {"step": "edit"})
        return decisions

    with cf.ThreadPoolExecutor(max_workers=min(runs, 3)) as pool:
        passes = list(pool.map(one_pass, range(runs)))

    # 1. Agreement, per utterance; trims only when every pass made the same trim.
    removed_sets = [_removed_uids(d, utterances) for d in passes]
    union = set().union(*[set(r) for r in removed_sets])
    unanimous = {uid for uid in union if all(uid in r for r in removed_sets)}
    disputed = union - unanimous
    trims = _trimmed(passes[0], utterances)
    for d in passes[1:]:
        other = _trimmed(d, utterances)
        trims = {i: r for i, r in trims.items() if i in other}
    report.unanimous, report.disputed = len(unanimous), len(disputed)
    removed: Dict[str, str] = {uid: removed_sets[0].get(uid, "retake") for uid in unanimous}

    # 2. Second opinion on the disputed lines.
    if disputed:
        verdicts = _ask_json(client, SECOND_OPINION, _context(utterances, disputed, set(removed)),
                             report, "second_opinion")
        for uid in disputed:
            if str(verdicts.get(uid, "keep")).lower().startswith("remove"):
                removed[uid] = next((r[uid] for r in removed_sets if uid in r), "retake")
                report.second_opinion_removed += 1

    # 3. Content-loss check on everything removed.
    if removed:
        verdicts = _ask_json(client, CONTENT_CHECK, _context(utterances, set(removed), set(removed)),
                             report, "content_check")
        for uid, verdict in verdicts.items():
            if uid in removed and str(verdict).lower().startswith("lost"):
                del removed[uid]
                report.restored_by_content_check += 1

    # 4. Final read of the programme as it will play.
    if final_read:
        from .utterances import table
        kept = [u for u in utterances if u.uid not in removed]
        cut = [u for u in utterances if u.uid in removed]
        user = ("The programme, in order:\n" + table(kept)
                + "\n\nRemoved lines (for restoring only):\n" + (table(cut) if cut else "(none)"))
        answer = _ask_json(client, FINAL_READ, user, report, "final_read")
        for uid in answer.get("restore") or []:
            if uid in removed:
                del removed[uid]
                report.final_read_restored += 1
        for uid in answer.get("remove") or []:
            if uid in by_id and uid not in removed:
                removed[uid] = "final_read"
                report.final_read_removed += 1
        extra = editor.parse({"trims": answer.get("trims") or []}, utterances, words)
        trims.update(extra.removed_words)

    out = Decisions(groups=sum(d.groups for d in passes) // max(1, runs),
                    drops=sum(d.drops for d in passes) // max(1, runs))
    for uid, reason in removed.items():
        u = by_id[uid]
        for i in u.word_indices:
            out.removed_words[i] = reason
        if not u.word_indices:
            out.removed_spans.append((u.start, u.end, reason))
    for i, reason in trims.items():
        out.removed_words.setdefault(i, reason)
    out.trims = len(trims)
    return out, report
