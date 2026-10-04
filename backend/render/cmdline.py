"""Keep FFmpeg command lines under the Windows limit.

CreateProcess refuses a command line over 32,767 characters with
"[WinError 206] The filename or extension is too long". The filter graph
already goes through -filter_complex_script, but every compiled input is still
a `-ss .. -t .. -threads 1 -i <absolute path>` group on the command line, and a
14-minute Raat3Baje episode compiles to ~370 of them (~44K characters). The
chunked render's audio pass handed all of them to FFmpeg although its graph
reads a handful, and the single-graph fallback died the same way.

Two layers, applied to every render spawn:

- `prune_inputs` drops the inputs a graph never references and renumbers the
  `[n:v]` / `[n:a]` references, so a pass only opens what it reads.
- `fit_command` is the guarantee: when a command is still too long, every
  input path is hard-linked (symlinked, or as a last resort copied) into a
  short temporary directory as `<n><ext>` and FFmpeg runs from there, which
  brings each input down to ~45 characters (~700 inputs fit).
"""
import itertools
import logging
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from config import TEMP_DIR

logger = logging.getLogger(__name__)

# The CreateProcess limit is 32,767; leave room for quoting differences.
SAFE_CMDLINE_CHARS = 30000

_INPUT_REF = re.compile(r"\[(\d+):")


def cmdline_length(cmd: Sequence[str]) -> int:
    return len(subprocess.list2cmdline([str(c) for c in cmd]))


def _input_groups(inputs: Sequence[str]) -> Tuple[List[List[str]], List[str]]:
    """Split compiled input args into one option group per `-i path`."""
    groups: List[List[str]] = []
    current: List[str] = []
    expect_path = False
    for token in inputs:
        current.append(token)
        if expect_path:
            groups.append(current)
            current = []
            expect_path = False
        elif token == "-i":
            expect_path = True
    return groups, current


def prune_inputs(inputs: Sequence[str], graph: str, *maps: str
                 ) -> Tuple[List[str], str, List[str]]:
    """Drop the inputs neither `graph` nor the `-map` labels reference.

    Returns (inputs, graph, maps) with the input references renumbered. With
    no graph, or a reference it cannot account for, everything is returned
    unchanged -- pruning is an optimisation, never a reason to fail.
    """
    groups, rest = _input_groups(inputs)
    if not graph or not groups:
        return list(inputs), graph, list(maps)
    used = {int(n) for n in _INPUT_REF.findall(graph)}
    for m in maps:
        found = re.match(r"\[?(\d+):", m or "")
        if found:
            used.add(int(found.group(1)))
    if not used or max(used) >= len(groups) or len(used) == len(groups):
        return list(inputs), graph, list(maps)
    renumber = {old: new for new, old in enumerate(sorted(used))}

    def swap(match: "re.Match[str]") -> str:
        return f"[{renumber[int(match.group(1))]}:"

    new_inputs = list(itertools.chain.from_iterable(groups[i] for i in sorted(used))) + rest
    new_graph = _INPUT_REF.sub(swap, graph)
    new_maps = [re.sub(r"^(\[?)(\d+):", lambda m: f"{m.group(1)}{renumber[int(m.group(2))]}:", m)
                if re.match(r"^\[?\d+:", m or "") else m for m in maps]
    return new_inputs, new_graph, new_maps


def _alias(source: str, target: Path) -> bool:
    for make in (os.link, os.symlink):
        try:
            make(source, target)
            return True
        except OSError:
            continue
    try:
        shutil.copyfile(source, target)
        return True
    except OSError:
        return False


@contextmanager
def fit_command(cmd: Sequence[str]) -> Iterator[Tuple[List[str], Optional[str]]]:
    """Yield (cmd, cwd) that CreateProcess will accept.

    Unchanged (cwd None) when the command already fits. Otherwise the `-i`
    paths are replaced by short aliases in a temporary directory that FFmpeg
    runs from; the directory is removed afterwards.
    """
    cmd = [str(c) for c in cmd]
    if cmdline_length(cmd) <= SAFE_CMDLINE_CHARS:
        yield cmd, None
        return
    os.makedirs(TEMP_DIR, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="in_", dir=str(TEMP_DIR)))
    try:
        fitted = list(cmd)
        # FFmpeg will run from `work`: anchor the other relative paths first.
        for i in range(1, len(fitted)):
            if fitted[i - 1] == "-filter_complex_script" and not os.path.isabs(fitted[i]):
                fitted[i] = os.path.abspath(fitted[i])
        if (fitted[-1] not in ("-",) and ":" not in fitted[-1][2:]
                and not os.path.isabs(fitted[-1])):
            fitted[-1] = os.path.abspath(fitted[-1])
        aliases = {}
        for i in range(len(fitted) - 1):
            if fitted[i] != "-i":
                continue
            source = fitted[i + 1]
            if source in aliases:
                fitted[i + 1] = aliases[source]
                continue
            if not os.path.isfile(source):
                continue
            name = f"{len(aliases)}{Path(source).suffix.lower()}"
            if _alias(os.path.abspath(source), work / name):
                aliases[source] = name
                fitted[i + 1] = name
        before, after = cmdline_length(cmd), cmdline_length(fitted)
        logger.info("FFmpeg command line %d chars; %d inputs aliased under %s -> %d chars",
                    before, len(aliases), work, after)
        if after > SAFE_CMDLINE_CHARS:
            raise RuntimeError(
                f"FFmpeg command line is {after} characters even with short input names "
                f"({len(aliases)} inputs); Windows allows 32767.")
        yield fitted, str(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
