"""The retake sweep (AUTO_CUT_V3_PLAN.md §2, step 3).

The three reads decide what goes, and a retake that only one of them saw, or none, survives
them. This is a narrow second look whose only job is to list the sets of utterances that are
attempts at the same content (a line said again, a restart, a repeated passage), over chunks of
about 80 utterances with 25 of overlap. Each chunk also carries the lines elsewhere in the
recording that its own lines are "similar to" or share an opening with, so a retake a chunk
boundary separates is still seen.

It decides nothing. The planner turns each group into candidates (every attempt not already
removed except the latest) and the verifier (`double_check.verify_candidates`) decides each one.
"""

import concurrent.futures as cf
import logging
from typing import Any, Dict, List, Sequence, Tuple

from . import editor

logger = logging.getLogger("retake_sweep")

SWEEP_SYSTEM = """You are listing retakes in a Hindi/English talking-head recording. List every set of
utterances that are attempts at the same content (a line said again, a restart, a repeated
passage), anywhere in this excerpt. Do not decide what to keep. Spellings come from speech
recognition and are often wrong; read past them. Deliberate repetition for effect is not a retake.

Answer with one JSON object and nothing else: {"groups": [{"utterances": ["U3","U5"], "why": "..."}]}"""


def chunks(utterances: Sequence[Any], size: int = 80, overlap: int = 25) -> List[List[Any]]:
    """Consecutive runs of `size` utterances, each starting `size - overlap` after the last, so
    neighbouring chunks share `overlap` lines and every utterance is in at least one."""
    items = list(utterances)
    if not items:
        return []
    size = max(1, int(size))
    step = max(1, size - max(0, int(overlap)))
    out: List[List[Any]] = []
    start = 0
    while True:
        out.append(items[start:start + size])
        if start + size >= len(items):
            return out
        start += step


def referenced(chunk: Sequence[Any], utterances: Sequence[Any]) -> List[Any]:
    """The utterances outside `chunk` that a line in it points at (`similar`, `same_opening`),
    in recording order."""
    inside = {u.uid for u in chunk}
    wanted = set()
    for u in chunk:
        wanted.update(s[0] for s in (u.similar or []))
        wanted.update(u.same_opening or [])
    return [u for u in utterances if u.uid in wanted and u.uid not in inside]


def _request(chunk: Sequence[Any], utterances: Sequence[Any]) -> str:
    from .utterances import table
    text = "The excerpt:\n" + table(chunk)
    others = referenced(chunk, utterances)
    if others:
        text += "\n\nReferenced lines from elsewhere in the recording:\n" + table(others)
    return text


def _groups(answer: str, known: Dict[str, int]) -> List[List[str]]:
    """The answer's groups with unknown ids dropped; a group needs two known members."""
    data = editor.first_json(answer or "") or {}
    out: List[List[str]] = []
    for g in data.get("groups") or []:
        if not isinstance(g, dict):
            continue
        members = list(dict.fromkeys(uid for uid in (g.get("utterances") or [])
                                     if isinstance(uid, str) and uid in known))
        if len(members) >= 2:
            out.append(members)
    return out


def _merge(groups: Sequence[Sequence[str]], order: Dict[str, int]) -> List[List[str]]:
    """Groups that share a member are one group (union-find), each in recording order."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in groups:
        for uid in g[1:]:
            a, b = find(g[0]), find(uid)
            if a != b:
                parent[b] = a
    merged: Dict[str, List[str]] = {}
    for uid in list(parent):
        merged.setdefault(find(uid), []).append(uid)
    out = [sorted(members, key=lambda uid: order[uid]) for members in merged.values()]
    return sorted(out, key=lambda members: order[members[0]])


def sweep(utterances: Sequence[Any], client: Any, workers: int = 3
          ) -> Tuple[List[List[str]], Dict[str, int]]:
    """(groups of uids that are attempts at the same content, merged across chunks and in
    recording order; {"calls", "tokens_in", "tokens_out", "failed"}). A chunk whose call raises
    is counted in `failed` and skipped."""
    utterances = list(utterances)
    order = {u.uid: n for n, u in enumerate(utterances)}
    meta = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "failed": 0}
    parts = chunks(utterances)
    if not parts:
        return [], meta

    def one(chunk):
        return client.ask(SWEEP_SYSTEM, _request(chunk, utterances))

    found: List[List[str]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(parts)))) as pool:
        futures = [pool.submit(one, chunk) for chunk in parts]
        for f in futures:
            meta["calls"] += 1
            try:
                text, call = f.result()
            except Exception as e:                 # one chunk failing is not the sweep failing
                logger.warning("Retake sweep: a chunk got no answer (%s: %s); skipped.",
                               type(e).__name__, e)
                meta["failed"] += 1
                continue
            usage = (call or {}).get("usage") or {}
            meta["tokens_in"] += int(usage.get("prompt_tokens") or 0)
            meta["tokens_out"] += int(usage.get("completion_tokens") or 0)
            found.extend(_groups(text, order))
    return _merge(found, order), meta
