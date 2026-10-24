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

Short names were not enough for the whole-programme audio pass: it opens the
recording once per kept segment (`-ss A -t D -threads 1 -i 3.mp4`), and a
22-minute Raat3Baje cut has 685 of them -- 30,968 characters of options alone
(674aefab, 2026-10-21, after 5.5 h of B-roll). So, as the last step, every
`-ss/-t` input the graph reads as audio only is moved into the graph script as
`amovie=..:seek_point=..,atrim=start=A:duration=D` (sample-identical to the
input-side seek, measured), where its length no longer counts.
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


_GLOBAL_FLAGS = {"-y", "-n", "-hide_banner", "-nostdin"}
_CUT_OPTIONS = {"-ss", "-t", "-threads"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _start_time(path: str, cwd: Path) -> Optional[float]:
    """The container's start_time, which `-ss` is relative to."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0", path],
            cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        value = (out.stdout or "").strip()
        return float(value) if value and value != "N/A" else 0.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _move_audio_cuts_into_graph(cmd: List[str], work: Path) -> Optional[List[str]]:
    """Replace `-ss A -t D [-threads 1] -i <alias>` inputs that the graph reads
    only as `[n:a]` with `amovie` sources inside the graph script.

    Returns the new command, or None when nothing could be moved (no graph
    script, no such inputs) -- the caller then reports the overflow as before.
    """
    try:
        script_at = cmd.index("-filter_complex_script") + 1
    except ValueError:
        return None
    try:
        graph = Path(cmd[script_at]).read_text(encoding="utf-8")
    except OSError:
        return None

    # One (first token, options, path) per input, in FFmpeg's input order.
    inputs: List[Tuple[int, List[str], str]] = []
    start = 1
    while start < len(cmd) and cmd[start] in _GLOBAL_FLAGS:
        start += 1
    i = start
    while i < len(cmd):
        if cmd[i] == "-i" and i + 1 < len(cmd):
            inputs.append((start, cmd[start:i], cmd[i + 1]))
            i += 2
            start = i
        else:
            i += 1
    if not inputs:
        return None

    refs = {}
    for n, spec in re.findall(r"\[(\d+):([^\]]*)\]", graph):
        refs.setdefault(int(n), set()).add(spec)
    mapped = set()
    for k in range(len(cmd) - 1):
        if cmd[k] == "-map":
            found = re.match(r"^\[?(\d+):", cmd[k + 1])
            if found:
                mapped.add(int(found.group(1)))

    starts = {}
    statements: List[str] = []
    moved = set()
    for n, (_first, options, path) in enumerate(inputs):
        if refs.get(n) != {"a"} or n in mapped or not _SAFE_NAME.match(path):
            continue
        if len(options) % 2 or not options:
            continue
        opts = dict(zip(options[::2], options[1::2]))
        if set(opts) - _CUT_OPTIONS or "-ss" not in opts:
            continue
        try:
            at = float(opts["-ss"])
            length = float(opts["-t"]) if "-t" in opts else None
        except ValueError:
            continue
        if path not in starts:
            starts[path] = _start_time(path, work)
        if starts[path] is None:
            continue
        at_abs = at + starts[path]
        trim = f"atrim=start={at_abs:.6f}" + (f":duration={length:.6f}" if length is not None else "")
        source = (f"amovie=filename={path}:seek_point={max(0.0, at - 1.0):.3f},"
                  f"{trim},asetpts=PTS-STARTPTS")
        # A filter output feeds one input, so each reference gets its own source.
        count = [0]

        def label(_match: "re.Match[str]", n: int = n, source: str = source) -> str:
            name = f"[mv{n}_{count[0]}]"
            statements.append(f"{source}{name}")
            count[0] += 1
            return name

        graph = re.sub(rf"\[{n}:a\]", label, graph)
        moved.add(n)
    if not moved:
        return None

    keep = [n for n in range(len(inputs)) if n not in moved]
    renumber = {old: new for new, old in enumerate(keep)}
    graph = re.sub(r"\[(\d+):", lambda m: f"[{renumber[int(m.group(1))]}:", graph)
    graph = ";\n".join(statements + [graph])

    out = cmd[:inputs[0][0]]
    for n in keep:
        first, options, path = inputs[n]
        out += options + ["-i", path]
    tail = cmd[inputs[-1][0] + len(inputs[-1][1]) + 2:]
    for k in range(len(tail) - 1):
        if tail[k] == "-map":
            tail[k + 1] = re.sub(r"^(\[?)(\d+):",
                                 lambda m: f"{m.group(1)}{renumber[int(m.group(2))]}:", tail[k + 1])
    script = work / "graph_moved.txt"
    script.write_text(graph, encoding="utf-8")
    for k in range(len(tail) - 1):
        if tail[k] == "-filter_complex_script":
            tail[k + 1] = str(script)
    logger.info("Moved %d audio cut(s) of %d input(s) into the graph script",
                len(moved), len(inputs))
    return out + tail


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
            moved = _move_audio_cuts_into_graph(fitted, work)
            if moved:
                fitted = moved
                after = cmdline_length(fitted)
                logger.info("FFmpeg command line %d chars after moving audio cuts into the graph",
                            after)
        if after > SAFE_CMDLINE_CHARS:
            raise RuntimeError(
                f"FFmpeg command line is {after} characters even with short input names "
                f"({len(aliases)} inputs); Windows allows 32767.")
        yield fitted, str(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
