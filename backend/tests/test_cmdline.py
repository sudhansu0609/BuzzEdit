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


def test_fit_command_moves_repeated_audio_cuts_into_the_graph(tmp_path, monkeypatch):
    # 674aefab: the recording opened once per kept segment overflowed even
    # with short names; the audio-only cuts must move into the graph script.
    monkeypatch.setattr(cmdline, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(cmdline, "_start_time", lambda path, cwd: 0.0)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"")
    sfx = tmp_path / "hit.wav"
    sfx.write_bytes(b"")
    cuts = 900
    cmd = ["ffmpeg", "-y", "-hide_banner", "-threads", "1", "-i", str(sfx)]
    for k in range(cuts):
        cmd += ["-ss", f"{k * 2.5:.3f}", "-t", "2.000", "-threads", "1", "-i", str(source)]
    statements = [f"[{k + 1}:a]atrim=duration=2.000,asetpts=PTS-STARTPTS[s{k}]" for k in range(cuts)]
    labels = "".join(f"[s{k}]" for k in range(cuts))
    graph = ";".join(statements + [f"{labels}concat=n={cuts}:v=0:a=1[speech]",
                                   "[0:a][speech]amix=inputs=2[mix]"])
    script = tmp_path / "graph.txt"
    script.write_text(graph, encoding="utf-8")
    cmd += ["-filter_complex_script", str(script), "-map", "[mix]", str(tmp_path / "out.wav")]

    with fit_command(cmd) as (fitted, cwd):
        assert cmdline_length(fitted) <= cmdline.SAFE_CMDLINE_CHARS
        assert fitted.count("-i") == 1  # only the SFX stays an input
        moved = fitted[fitted.index("-filter_complex_script") + 1]
        text = open(moved, encoding="utf-8").read()
        assert text.count("amovie=filename=") == cuts
        assert "atrim=start=5.000000:duration=2.000000" in text
        assert "[0:a][speech]amix" in text and "[1:a]" not in text
    assert script.read_text(encoding="utf-8") == graph  # resume compares this file


def test_fit_command_keeps_cuts_the_picture_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(cmdline, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(cmdline, "_start_time", lambda path, cwd: 0.0)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"")
    cmd = ["ffmpeg", "-y"]
    for k in range(900):
        cmd += ["-ss", f"{k:.3f}", "-t", "1.000", "-threads", "1", "-i", str(source)]
    script = tmp_path / "graph.txt"
    script.write_text(";".join(f"[{k}:v]null[v{k}]" for k in range(900)), encoding="utf-8")
    cmd += ["-filter_complex_script", str(script), "-map", "[v0]", "out.mp4"]
    try:
        with fit_command(cmd):
            raise AssertionError("a video-read cut cannot become an amovie source")
    except RuntimeError as e:
        assert "Windows allows 32767" in str(e)
