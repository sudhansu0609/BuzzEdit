"""render/cmdline.py: renders must never die on the Windows command-line limit."""
import os

import render.cmdline as cmdline
from render.cmdline import cmdline_length, fit_command, prune_inputs


def test_prune_drops_unreferenced_inputs_and_renumbers():
    inputs = ["-i", "a.mp4", "-ss", "1.000", "-threads", "1", "-i", "b.png", "-i", "c.wav"]
    graph = "[2:a]volume=1[a0];[0:a][a0]amix=inputs=2[aout]"
    new_inputs, new_graph, maps = prune_inputs(inputs, graph, "[aout]")
    assert new_inputs == ["-i", "a.mp4", "-i", "c.wav"]
    assert new_graph == "[1:a]volume=1[a0];[0:a][a0]amix=inputs=2[aout]"
    assert maps == ["[aout]"]


def test_prune_keeps_inputs_named_only_by_map():
    inputs = ["-i", "a.mp4", "-i", "b.mp4"]
    new_inputs, graph, maps = prune_inputs(inputs, "[1:v]null[v]", "[v]", "0:a")
    assert new_inputs == inputs and graph == "[1:v]null[v]" and maps == ["[v]", "0:a"]


def test_prune_leaves_everything_without_a_graph():
    inputs = ["-i", "a.mp4", "-i", "b.mp4"]
    assert prune_inputs(inputs, "")[0] == inputs


def test_fit_command_aliases_long_input_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(cmdline, "TEMP_DIR", tmp_path / "temp")
    long_dir = tmp_path / ("x" * 120)
    long_dir.mkdir()
    cmd = ["ffmpeg"]
    for n in range(400):  # ~400 * 170 chars: well past 32,767
        source = long_dir / f"still_{n:04d}_{'y' * 20}.png"
        source.write_bytes(b"png")
        cmd += ["-ss", "1.000", "-t", "2.000", "-threads", "1", "-i", str(source)]
    cmd += ["-filter_complex_script", str(tmp_path / "graph.txt"), "out.mp4"]
    assert cmdline_length(cmd) > 32767

    with fit_command(cmd) as (fitted, cwd):
        assert cwd is not None
        assert cmdline_length(fitted) <= cmdline.SAFE_CMDLINE_CHARS
        assert fitted[fitted.index("-i") + 1] == "0.png"
        assert os.path.isfile(os.path.join(cwd, "399.png"))
        assert os.path.isabs(fitted[-1])
    assert not os.path.exists(cwd)


def test_fit_command_leaves_short_commands_alone():
    cmd = ["ffmpeg", "-i", "a.mp4", "out.mp4"]
    with fit_command(cmd) as (fitted, cwd):
        assert fitted == cmd and cwd is None
