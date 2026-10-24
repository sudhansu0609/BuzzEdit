"""Turn answer keys (the creator's own cuts) into training pairs for the cut editor.

    ./.venv/Scripts/python.exe backend/tools/autocut_dataset.py raat3baje_ep3 [more names] [--all]
        [--window 50] [--overlap 12] [--test raat3baje_ep2] [--no-notes]

Step 2 of AUTO_CUT_FINETUNE_PLAN.md. For each name this takes the answer key
`data/eval/<name>.json` (made by tools/autocut_truth.py from the raw take and the
creator's final video, or by tools/autocut_mark.py from his corrected cut list),
transcribes the raw take exactly as production does (cached in
`data/eval/cache/<name>_words.json`), builds the utterance table the editor
reads (asr.utterances), and maps the key's removed spans onto it:

- an utterance whose speech lies mostly (>= DROP_SHARE) inside removed spans is
  removed; if it resembles a later kept utterance (the table's own "similar to"
  pointers, >= RETAKE_SIM) it joins that utterance's retake **group** with the
  kept one as "keep", otherwise it is a **drop** (false_start when its notes say
  it broke off, repeat when it resembles an earlier kept line, else abandoned);
- a removed stretch inside an otherwise kept utterance (TRIM_SHARE .. DROP_SHARE)
  is a **trim**, named by the exact contiguous words the table shows;
- `editorial` spans (whole passages he dropped for pacing, which no transcript
  reveals) are not decisions: the model is not taught to drop good content.

The table is cut into windows of `--window` utterances overlapping by
`--overlap`, and each window becomes one pair: the editor's SYSTEM prompt, the
request the product builds (asr.editor.request) over that window's table, and
the decisions JSON restricted to that window, in the shape asr.editor.parse
accepts. A group whose "keep" falls outside the window becomes drops of kind
"repeat" ("said again later"). Output: `data/finetune/pairs/<name>.jsonl`, the
key copied to `data/finetune/keys/`, the name appended to
`data/finetune/splits/train.txt` -- or `test.txt` for the held-out take
(`--test`, default raat3baje_ep2, never trained on). `--all` does every key in
data/eval. A summary per take is printed and kept in `pairs/<name>.stats.json`.

Everything above `# --- pipeline ---` is pure (no audio, no model) and is what
tests/test_autocut_dataset.py exercises with synthetic words and spans.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "tools"))

DROP_SHARE = 0.80      # share of an utterance's speech inside removed spans -> the utterance goes
TRIM_SHARE = 0.15      # above this (and below DROP_SHARE) -> a trim of the removed words
WORD_SHARE = 0.50      # a word is "removed" when this much of it lies in a removed span
RETAKE_SIM = 0.45      # "similar to" score at which a removed line is a retake of a kept one
MIN_TRIM_WORDS = 1
DEFAULT_WINDOW = 50
DEFAULT_OVERLAP = 12
DEFAULT_TEST = "raat3baje_ep2"
BROKE_OFF = ("cut off", "ends on", "restart", "unfinished", "level pitch")


# --- mapping (pure) -----------------------------------------------------------------

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def removed_share(start: float, end: float, spans: Sequence[Tuple[float, float]]) -> float:
    """Share of [start, end] covered by the removed spans (0..1)."""
    length = max(1e-6, end - start)
    covered = sum(_overlap(start, end, s, e) for s, e in spans)
    return min(1.0, covered / length)


def removed_spans_of(key: Dict[str, Any]) -> List[Tuple[float, float]]:
    """The key's removed spans that are decisions (not `editorial`), merged and sorted."""
    raw = []
    for s in key.get("spans") or []:
        if str(s.get("kind") or "") == "editorial":
            continue
        try:
            a, b = float(s["start"]), float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if b > a:
            raw.append((a, b))
    raw.sort()
    merged: List[Tuple[float, float]] = []
    for a, b in raw:
        if merged and a <= merged[-1][1] + 0.01:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _word_text(w: Dict[str, Any]) -> str:
    return str(w.get("word_native") or w.get("word") or "").strip()


def removed_word_runs(u: Any, words: Sequence[Dict[str, Any]], spans: Sequence[Tuple[float, float]]) -> List[List[int]]:
    """Contiguous runs of the utterance's words that lie in removed spans."""
    runs: List[List[int]] = []
    for idx in u.word_indices:
        w = words[idx]
        if removed_share(float(w["start"]), float(w["end"]), spans) >= WORD_SHARE:
            if runs and runs[-1][-1] == idx - 1:
                runs[-1].append(idx)
            else:
                runs.append([idx])
    return runs


def classify(utterances: Sequence[Any], words: Sequence[Dict[str, Any]],
             spans: Sequence[Tuple[float, float]]) -> Dict[str, Dict[str, Any]]:
    """Per utterance id: {"state": removed|trim|kept, "share": float, "trim_words": [idx..]}."""
    out: Dict[str, Dict[str, Any]] = {}
    for u in utterances:
        share = removed_share(u.start, u.end, spans)
        if share >= DROP_SHARE:
            out[u.uid] = {"state": "removed", "share": share}
            continue
        if share > TRIM_SHARE and u.word_indices:
            runs = removed_word_runs(u, words, spans)
            runs = [r for r in runs if len(r) >= MIN_TRIM_WORDS]
            if runs:
                longest = max(runs, key=len)
                out[u.uid] = {"state": "trim", "share": share, "trim_words": longest}
                continue
        out[u.uid] = {"state": "kept", "share": share}
    return out


def _broke_off(u: Any) -> bool:
    text = " ".join(u.notes).lower()
    return any(k in text for k in BROKE_OFF)


def decisions_for(utterances: Sequence[Any], words: Sequence[Dict[str, Any]],
                  states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The decisions JSON (the editor's answer format) for these utterances."""
    by_uid = {u.uid: u for u in utterances}
    order = {u.uid: n for n, u in enumerate(utterances)}
    groups: List[Dict[str, Any]] = []
    drops: List[Dict[str, Any]] = []
    trims: List[Dict[str, Any]] = []
    grouped: Dict[str, List[str]] = {}     # keep uid -> removed members
    for u in utterances:
        st = states.get(u.uid, {}).get("state")
        if st == "removed":
            # A retake of a later kept line? The table's own pointers decide.
            later_kept = [(uid, score) for uid, score in (u.similar or [])
                          if uid in by_uid and states.get(uid, {}).get("state") != "removed"
                          and order[uid] > order[u.uid] and score >= RETAKE_SIM]
            if not later_kept and u.same_opening:
                later_kept = [(uid, RETAKE_SIM) for uid in u.same_opening
                              if uid in by_uid and states.get(uid, {}).get("state") != "removed" and order[uid] > order[u.uid]]
            if later_kept:
                keep = max(later_kept, key=lambda p: (p[1], -order[p[0]]))[0]
                grouped.setdefault(keep, []).append(u.uid)
                continue
            earlier_kept = any(uid in by_uid and order[uid] < order[u.uid] and states.get(uid, {}).get("state") != "removed"
                               and score >= RETAKE_SIM for uid, score in (u.similar or []))
            kind = "false_start" if _broke_off(u) else "repeat" if earlier_kept else "abandoned"
            why = {"false_start": "breaks off; the next attempt carries it",
                   "repeat": "said already in a kept line",
                   "abandoned": "not carried anywhere; the creator cut it"}[kind]
            drops.append({"utterances": [u.uid], "kind": kind, "why": why, "confidence": 0.9})
        elif st == "trim":
            idxs = states[u.uid]["trim_words"]
            remove = " ".join(_word_text(words[i]) for i in idxs).strip()
            if remove:
                trims.append({"utterance": u.uid, "remove": remove, "why": "the creator cut these words", "confidence": 0.85})
    for keep, members in grouped.items():
        groups.append({"utterances": sorted(members + [keep], key=lambda x: order[x]), "keep": keep,
                       "why": "earlier attempts at the kept line", "confidence": 0.9})
    groups.sort(key=lambda g: order[g["utterances"][0]])
    return {"groups": groups, "drops": drops, "trims": trims}


def windows(count: int, size: int, overlap: int) -> List[Tuple[int, int]]:
    """[start, end) index ranges of `size` overlapping by `overlap` over `count` items."""
    if count <= 0:
        return []
    size = max(8, size)
    step = max(1, size - max(0, min(overlap, size - 1)))
    out: List[Tuple[int, int]] = []
    start = 0
    while True:
        end = min(count, start + size)
        out.append((start, end))
        if end >= count:
            break
        start += step
    # A last window much smaller than the rest is folded into the one before it.
    if len(out) > 1 and out[-1][1] - out[-1][0] < size // 3:
        out[-2] = (out[-2][0], out[-1][1])
        out.pop()
    return out


def window_decisions(full: Dict[str, Any], shown: Sequence[str]) -> Dict[str, Any]:
    """`full` restricted to the shown utterance ids; a group whose keep is not
    shown becomes repeat-drops of its shown members."""
    shown_set = set(shown)
    groups, drops, trims = [], [], []
    for g in full["groups"]:
        members = [u for u in g["utterances"] if u in shown_set]
        if not members:
            continue
        if g["keep"] in shown_set and len(members) >= 2:
            groups.append({**g, "utterances": members})
        else:
            for u in members:
                if u != g["keep"]:
                    drops.append({"utterances": [u], "kind": "repeat", "why": "said again later, in a kept line",
                                  "confidence": 0.85})
    for d in full["drops"]:
        if all(u in shown_set for u in d["utterances"]):
            drops.append(d)
    for t in full["trims"]:
        if t["utterance"] in shown_set:
            trims.append(t)
    order = {uid: n for n, uid in enumerate(shown)}
    groups.sort(key=lambda g: order[g["utterances"][0]])
    drops.sort(key=lambda d: order[d["utterances"][0]])
    trims.sort(key=lambda t: order[t["utterance"]])
    return {"groups": groups, "drops": drops, "trims": trims}


def make_pairs(name: str, utterances: Sequence[Any], words: Sequence[Dict[str, Any]], key: Dict[str, Any],
               window: int = DEFAULT_WINDOW, overlap: int = DEFAULT_OVERLAP, style: str = "",
               speaker_style: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from asr import editor, utterances as utt_mod

    spans = removed_spans_of(key)
    states = classify(utterances, words, spans)
    full = decisions_for(utterances, words, states)
    pairs: List[Dict[str, Any]] = []
    for n, (a, b) in enumerate(windows(len(utterances), window, overlap)):
        part = list(utterances[a:b])
        shown = [u.uid for u in part]
        decisions = window_decisions(full, shown)
        pairs.append({
            "take": name, "window": n, "source": "creator", "weight": 1.0,
            "utterances": [shown[0], shown[-1]],
            "system": editor.SYSTEM,
            "input": editor.request(utt_mod.table(part), style, speaker_style),
            "output": json.dumps(decisions, ensure_ascii=False),
        })
    counts = {"removed": 0, "trim": 0, "kept": 0}
    for st in states.values():
        counts[st["state"]] += 1
    removed_s = sum(b - a for a, b in spans)
    covered = sum(u.end - u.start for u in utterances if states[u.uid]["state"] == "removed")
    stats = {
        "take": name, "utterances": len(utterances), "windows": len(pairs), **counts,
        "groups": len(full["groups"]), "drops": len(full["drops"]), "trims": len(full["trims"]),
        "removed_s_in_key": round(removed_s, 1), "removed_s_as_utterances": round(covered, 1),
        "editorial_spans": sum(1 for s in key.get("spans") or [] if s.get("kind") == "editorial"),
        "made_at": datetime.now().isoformat(timespec="seconds"),
    }
    return pairs, stats


# --- pipeline -----------------------------------------------------------------------

def _finetune_dir() -> Path:
    from config import DATA_DIR
    return DATA_DIR / "finetune"


def _eval_dir() -> Path:
    from config import DATA_DIR
    return DATA_DIR / "eval"


def _add_to_split(name: str, split: str) -> None:
    path = _finetune_dir() / "splits" / f"{split}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    other = _finetune_dir() / "splits" / ("train.txt" if split == "test" else "test.txt")
    if other.exists():
        others = [l.strip() for l in other.read_text(encoding="utf-8").splitlines() if l.strip() != name]
        other.write_text("\n".join(others) + ("\n" if others else ""), encoding="utf-8")
    if name not in names:
        names.append(name)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


async def build_take(name: str, args) -> Dict[str, Any]:
    from tools.editor_bench import prepare

    prepared = await prepare(name, SimpleNamespace(script=getattr(args, "script", ""), no_notes=args.no_notes,
                                                   no_hints=False))
    pairs, stats = make_pairs(name, prepared["utterances"], prepared["words"], prepared["key"],
                              window=args.window, overlap=args.overlap)
    out_dir = _finetune_dir() / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out_dir / f"{name}.stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    keys_dir = _finetune_dir() / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_eval_dir() / f"{name}.json", keys_dir / f"{name}.json")
    split = "test" if name == args.test else "train"
    _add_to_split(name, split)
    stats["split"] = split
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("names", nargs="*", help="answer keys in data/eval (without .json)")
    p.add_argument("--all", action="store_true", help="every key in data/eval")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    p.add_argument("--test", default=DEFAULT_TEST, help="the held-out take (never in train.txt)")
    p.add_argument("--no-notes", action="store_true", help="leave the audio notes out of the table")
    p.add_argument("--script", default="", help="written script for script tags (one take only)")
    args = p.parse_args(argv)
    names = list(args.names)
    if args.all:
        names += [q.stem for q in sorted(_eval_dir().glob("*.json")) if q.stem not in names]
    if not names:
        p.error("give at least one key name or --all")
    total = 0
    for name in names:
        try:
            stats = asyncio.run(build_take(name, args))
        except Exception as exc:  # noqa: BLE001 - one bad key must not stop the rest
            print(f"{name}: FAILED {type(exc).__name__}: {exc}")
            continue
        total += stats["windows"]
        print(f"{name} [{stats['split']}]: {stats['utterances']} utterances -> {stats['windows']} pairs | "
              f"{stats['groups']} groups, {stats['drops']} drops, {stats['trims']} trims | "
              f"removed {stats['removed_s_as_utterances']} s of {stats['removed_s_in_key']} s in the key"
              + (f" | {stats['editorial_spans']} editorial spans left out" if stats["editorial_spans"] else ""))
    train = _finetune_dir() / "splits" / "train.txt"
    n_train = len([l for l in train.read_text(encoding="utf-8").splitlines() if l.strip()]) if train.exists() else 0
    print(f"\n{total} pairs written this run; {n_train} take(s) in train.txt. "
          f"Minimum to train: 12-15 takes (150-250 pairs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
