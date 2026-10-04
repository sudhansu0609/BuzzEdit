"""Pre-rendered Ken Burns moves (render/prerender.py) look the same as the
in-graph ones, and the rendered timeline copy points at the clips."""

import asyncio
import re
import subprocess

import pytest

from render.prerender import prerender_animated_stills
from timeline import build_timeline_from_transcript, clip_ops
from timeline.schema import SourceFile


def _setup(tmp_path):
    from config import FFMPEG_BIN
    source, still = tmp_path / "src.mp4", tmp_path / "still.png"
    ok = subprocess.run([FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi", "-i",
                         "testsrc2=size=640x360:rate=25:duration=6", "-f", "lavfi", "-i",
                         "sine=frequency=440:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                         "-c:a", "aac", "-shortest", str(source)]).returncode == 0
    ok = ok and subprocess.run([FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi", "-i",
                                "mandelbrot=size=1280x720", "-frames:v", "1", str(still)]).returncode == 0
    if not ok:
        pytest.skip("ffmpeg unavailable")
    words = [{"word": f"w{i}", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(15)]
    timeline = build_timeline_from_transcript(str(source), 6.0, words, fps_num=25, fps_den=1,
                                              width=640, height=360, pause_padding_seconds=0.0)
    timeline.sources["still"] = SourceFile(id="still", path=str(still), duration_seconds=3.0, width=1280,
                                           height=720, fps_num=25, fps_den=1, has_audio=False, kind="image")
    item = clip_ops.add_media_item(timeline, "still", "V3", 25, 0, 75, origin="broll")
    clip_ops.set_transform(timeline, item.id, {"scale": 1.0, "scale_end": 1.15,
                                               "pos_x": -0.1, "pos_x_end": 0.2})
    return timeline, item


def test_the_rendered_copy_points_at_a_clip_and_the_original_is_untouched(tmp_path, monkeypatch):
    import render.prerender as P
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path / "kb")
    timeline, item = _setup(tmp_path)
    copy = prerender_animated_stills(timeline)
    moved = next(i for i in copy.items if i.id == item.id)
    assert moved.transform is None and copy.sources[moved.source_id].kind == "video"
    assert timeline.items[-1].transform.scale_end == pytest.approx(1.15), "the edit itself is not changed"


def test_prerendered_moves_match_the_in_graph_render(tmp_path, monkeypatch):
    import render.prerender as P
    from config import FFMPEG_BIN
    from render.runner import render_timeline_async
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path / "kb")
    timeline, _ = _setup(tmp_path)
    graph, pre = tmp_path / "graph.mp4", tmp_path / "pre.mp4"
    asyncio.run(render_timeline_async(timeline, str(graph), prefer_nvenc=False, prerender=False))
    asyncio.run(render_timeline_async(timeline, str(pre), prefer_nvenc=False, prerender=True))
    out = subprocess.run([FFMPEG_BIN, "-v", "info", "-i", str(graph), "-i", str(pre),
                          "-lavfi", "[0:v]trim=1:4,setpts=PTS-STARTPTS[a];[1:v]trim=1:4,setpts=PTS-STARTPTS[b];[a][b]psnr",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    psnr = float(re.search(r"average:([\d.]+|inf)", out).group(1).replace("inf", "99"))
    assert psnr > 38.0, f"pre-rendered move differs visibly (PSNR {psnr:.1f} dB)"
