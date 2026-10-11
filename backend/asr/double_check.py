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

VERIFY_SYSTEM = """Several reads of this Hindi/English talking-head recording disagreed about the lines marked
>>, or a sweep found them to be attempts at the same content as another line. Lines marked (kept)
will be heard by the viewer; lines marked (cut) will not. For each marked line decide: 'remove' only
if (a) a kept line says the same content (another attempt at it, even reworded) — name it in
covered_by — or (b) the line breaks off or is filler, chatter or a false start that carries no
content of its own; otherwise 'keep'. Prefer the LATER complete attempt. If the marked line is the
complete attempt and its kept twin is the broken one, answer 'keep' for the marked line and 'remove'
for the twin with covered_by = the marked line. Spellings come from speech recognition.

Answer with one JSON object and nothing else: {"U12": {"verdict": "remove", "covered_by": "U14",
"why": "..."}, "U14": {"verdict": "keep", "why": "..."}}"""

FINAL_READ_CHUNKED = """You are doing the last read of a video edit. The kept lines below are exactly what the viewer
will hear, in order; the removed lines nearby are listed for restoring only. Find what still does
not play as one fluent take: a line said twice (two attempts both kept), a sentence that breaks off
and is not continued, a stutter or doubled phrase inside a line, a jump where something needed to
understand the next line was cut. Fix only with: remove (a kept line whose content another KEPT line
carries — name it in covered_by), trims (a word or a short phrase said twice back to back inside one kept line: copy the FIRST copy exactly; never use a trim to shorten or tidy a sentence), restore
(removed line ids). Never remove content said nowhere else. Spellings come from speech recognition.

Answer with one JSON object and nothing else: {"remove": [{"utterance": "U12", "covered_by": "U14",
"why": "..."}], "trims": [{"utterance": "U9", "remove": "exact words", "why": "..."}], "restore": ["U30"]}"""


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


def _context(utterances: Sequence[Any], marked: Set[str], removed: Set[str], shown: Set[str] = frozenset(),
             states: bool = False, labels: Optional[Dict[str, str]] = None) -> str:
    """The marked lines (">>") with CONTEXT_LINES either side, "(cut)" on removed lines. `shown`
    lines get the same window without the mark; with `states` a kept line reads "(kept)";
    `labels` appends text to a line by uid."""
    from .utterances import table
    index = {u.uid: n for n, u in enumerate(utterances)}
    keep_lines: Set[int] = set()
    for uid in set(marked) | set(shown):
        n = index[uid]
        keep_lines.update(range(max(0, n - CONTEXT_LINES), min(len(utterances), n + CONTEXT_LINES + 1)))
    lines, last = [], None
    for n in sorted(keep_lines):
        if last is not None and n != last + 1:
            lines.append("...")
        u = utterances[n]
        prefix = ">> " if u.uid in marked else ("(cut) " if u.uid in removed else
                                                ("(kept) " if states else "   "))
        line = prefix + table([u])
        if labels and labels.get(u.uid):
            line += "  << " + labels[u.uid]
        lines.append(line)
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


VERIFY_BATCH = 40


def _verdict(value: Any, known: Set[str]) -> Optional[Dict[str, Any]]:
    """{"verdict": "remove"|"keep", "covered_by": uid or None, "why"} from one answer entry."""
    if isinstance(value, dict):
        said, covered_by, why = value.get("verdict"), value.get("covered_by"), value.get("why")
    else:
        said, covered_by, why = value, None, ""
    said = str(said or "").strip().lower()
    verdict = "remove" if said.startswith("remove") else ("keep" if said.startswith("keep") else None)
    if verdict is None:
        return None
    return {"verdict": verdict, "covered_by": covered_by if isinstance(covered_by, str) and covered_by in known
            else None, "why": str(why or "")}


def verify_candidates(utterances: Sequence[Any], candidates: Dict[str, Dict[str, Any]], removed: Dict[str, str],
                      client: EditorClient, report: Optional[Report] = None,
                      batch: int = VERIFY_BATCH) -> Dict[str, Dict[str, Any]]:
    """The focused question on removals the reads did not agree on (AUTO_CUT_V3_PLAN.md §2.2-3).

    `candidates`: uid -> {"reason", "why"} and optionally "twin" (the line a sweep found it to be
    another attempt at). Each is shown marked, with its `.similar` twins (and its "twin") and
    CONTEXT_LINES either side, every other line as (kept) or (cut). A candidate may instead be a
    phrase on a line, keyed "R<n>" with "uid" (the line) and "text" (the phrase said twice back to
    back); its verdict comes back under its own key. One call per `batch` candidates. Returns
    key -> {"verdict", "covered_by", "why"} for the candidates and shown twins the answer named;
    what to apply is decided by the caller. A failing call raises.
    """
    report = report if report is not None else Report()
    index = {u.uid: n for n, u in enumerate(utterances)}
    by_id = {u.uid: u for u in utterances}
    keys = sorted((k for k, c in candidates.items() if ((c or {}).get("uid") or k) in index),
                  key=lambda k: (index[(candidates[k] or {}).get("uid") or k], k))
    out: Dict[str, Dict[str, Any]] = {}
    for at in range(0, len(keys), max(1, int(batch))):
        marked: Set[str] = set()
        shown: Set[str] = set()
        named: Set[str] = set()
        labels: Dict[str, List[str]] = {}
        for k in keys[at:at + max(1, int(batch))]:
            c = candidates[k] or {}
            line = c.get("uid") or k
            marked.add(line)
            named.add(k)
            if c.get("text") is not None:                 # a phrase on the line, not the line
                labels.setdefault(line, []).append(f"{k}: \"{c['text']}\" is said twice back to back here")
                continue
            twins = [s[0] for s in (by_id[line].similar or [])] + ([c["twin"]] if c.get("twin") else [])
            for twin in twins:
                if twin in index and twin != line:
                    shown.add(twin)
                    named.add(twin)
        text = _context(utterances, marked, set(removed), shown=shown - marked, states=True,
                        labels={uid: "; ".join(v) for uid, v in labels.items()})
        if labels:
            text = ("Each R id names a phrase said twice back to back on the line it is written after: "
                    "answer for each R id 'remove' when the first copy is a stumble the second one "
                    "carries, 'keep' when it is said twice on purpose.\n\n" + text)
        answer = _ask_json(client, VERIFY_SYSTEM, text, report, "verify")
        for key, value in answer.items():
            if key in named:
                verdict = _verdict(value, set(index))
                if verdict:
                    out[key] = verdict
    return out


def final_read_chunked(utterances: Sequence[Any], removed: Dict[str, str], words: Sequence[Dict[str, Any]],
                       client: EditorClient, chunk: int = 80, overlap: int = 20, workers: int = 3,
                       protected: Set[str] = frozenset()) -> Dict[str, Any]:
    """The last read of the programme as it plays, in overlapping chunks (AUTO_CUT_V3_PLAN.md §2.8).

    Each chunk shows its kept lines in order and then the removed lines between its first and
    last kept line. A `remove` stands only when its covered_by is a kept line that nothing in this
    read removes (and it then stays); a `restore` only for a removed line outside `protected`;
    trims are validated by `editor.parse`. Restore wins over remove for the same uid. A chunk
    whose call raises is counted in "failed" and skipped. Returns {"remove": uid -> (covered_by,
    why), "trims": word index -> reason, "restore": set of uids, "calls", "tokens_in",
    "tokens_out", "failed"}.
    """
    from .retake_sweep import chunks
    from .utterances import table
    order = {u.uid: n for n, u in enumerate(utterances)}
    kept = [u for u in utterances if u.uid not in removed]
    kept_ids = {u.uid for u in kept}
    result: Dict[str, Any] = {"remove": {}, "trims": {}, "restore": set(), "calls": 0, "tokens_in": 0,
                              "tokens_out": 0, "failed": 0}
    parts = chunks(kept, chunk, overlap)
    if not parts:
        return result

    def request(part: Sequence[Any]) -> str:
        lo, hi = order[part[0].uid], order[part[-1].uid]
        cut = [u for u in utterances if u.uid in removed and lo <= order[u.uid] <= hi]
        return ("The programme, in order:\n" + table(part)
                + "\n\nRemoved lines (for restoring only):\n" + (table(cut) if cut else "(none)"))

    answers: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(parts)))) as pool:
        futures = [pool.submit(client.ask, FINAL_READ_CHUNKED, request(part)) for part in parts]
        for f in futures:
            result["calls"] += 1
            try:
                text, meta = f.result()
            except Exception as e:                 # one chunk failing is not the read failing
                logger.warning("Final read: a chunk got no answer (%s: %s); skipped.", type(e).__name__, e)
                result["failed"] += 1
                continue
            usage = (meta or {}).get("usage") or {}
            result["tokens_in"] += int(usage.get("prompt_tokens") or 0)
            result["tokens_out"] += int(usage.get("completion_tokens") or 0)
            answers.append(editor.first_json(text) or {})

    restore: Set[str] = set()
    proposals: Dict[str, Tuple[Any, str]] = {}
    trims: List[Dict[str, Any]] = []
    for answer in answers:
        for item in answer.get("restore") or []:
            uid = item.get("utterance") if isinstance(item, dict) else item
            if isinstance(uid, str) and uid in removed and uid not in protected:
                restore.add(uid)
        for item in answer.get("remove") or []:
            if isinstance(item, dict) and item.get("utterance") in kept_ids:
                proposals.setdefault(item["utterance"], (item.get("covered_by"), str(item.get("why") or "")))
        trims.extend(t for t in answer.get("trims") or [] if isinstance(t, dict))
    removes: Dict[str, Tuple[str, str]] = {}
    covers: Set[str] = set()                       # lines a removal leans on: they stay
    for uid in sorted(proposals, key=lambda u: order[u]):
        covered_by, why = proposals[uid]
        if uid in restore or uid in covers:
            continue
        if not isinstance(covered_by, str) or covered_by not in kept_ids or covered_by == uid \
                or covered_by in removes:
            continue
        removes[uid] = (covered_by, why)
        covers.add(covered_by)
    result["remove"], result["restore"] = removes, restore
    if trims:
        result["trims"] = dict(editor.parse({"trims": trims}, utterances, words).removed_words)
    return result
