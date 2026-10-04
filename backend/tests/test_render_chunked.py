"""Chunk planning never cuts through anything animated, and a chunk's timeline
is the programme's picture for that stretch, starting at frame 0."""
from render.chunked import plan_chunks, sub_timeline
from render.runner import _render_canvas
from timeline.schema import SourceFile, Timeline, TimelineItem, Transform

FPS = 24


def _tl(extra=()):
    total = 300 * FPS
    sources = {
        "main": SourceFile(id="main", path="main.mp4", duration_seconds=300, width=3840,
                           height=2160, fps_num=24, fps_den=1, has_audio=True, kind="video"),
        "img": SourceFile(id="img", path="b.png", duration_seconds=5, width=1024, height=576,
                          fps_num=24, fps_den=1, has_audio=False, kind="image"),
    }
    items = [
        TimelineItem(id="v1", track="V1", source_id="main", source_start_frame=0,
                     source_end_frame=total, timeline_start_frame=0, timeline_end_frame=total),
        TimelineItem(id="a1", track="A1", source_id="main", source_start_frame=0,
                     source_end_frame=total, timeline_start_frame=0, timeline_end_frame=total),
        *extra,
    ]
    return Timeline(fps_num=24, fps_den=1, width=3840, height=2160, duration_frames=total,
                    sources=sources, items=items)


def test_chunks_cover_the_programme_without_gaps():
    tl = _tl()
    chunks = plan_chunks(tl)
    assert chunks[0][0] == 0 and chunks[-1][1] == tl.duration_frames
    assert all(a < b for a, b in chunks)
    assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))
    assert len(chunks) >= 5


def test_no_boundary_inside_an_animated_overlay():
    zoom = TimelineItem(id="kb", track="V3", source_id="img", source_start_frame=0,
                        source_end_frame=20 * FPS, timeline_start_frame=35 * FPS,
                        timeline_end_frame=55 * FPS,
                        transform=Transform(scale=1.0, scale_end=1.2))
    chunks = plan_chunks(_tl([zoom]))
    for a, _b in chunks[1:]:
        assert not (35 * FPS < a < 55 * FPS)


def test_a_chunk_timeline_starts_at_zero_and_keeps_only_its_items():
    still = TimelineItem(id="s", track="V3", source_id="img", source_start_frame=0,
                         source_end_frame=4 * FPS, timeline_start_frame=100 * FPS,
                         timeline_end_frame=104 * FPS)
    tl = _tl([still])
    part = sub_timeline(tl, 90 * FPS, 130 * FPS)
    ids = {i.id: i for i in part.items}
    assert set(ids) == {"v1", "s"}           # audio dropped, still kept
    assert ids["v1"].timeline_start_frame == 0
    assert ids["v1"].source_start_frame == 90 * FPS
    assert ids["v1"].duration_frames == 40 * FPS
    assert ids["s"].timeline_start_frame == 10 * FPS
    assert part.duration_frames == 40 * FPS and set(part.sources) == {"main", "img"}
    assert part.audio_master is None


def test_compose_at_the_delivery_size_only_when_smaller():
    assert _render_canvas((3840, 2160), (1920, 1080)) == (1920, 1080)
    assert _render_canvas((1920, 1080), (1920, 1080)) is None
    assert _render_canvas((1280, 720), (1920, 1080)) is None
    assert _render_canvas((2160, 3840), (1920, 1080)) == (606, 1080)
    assert _render_canvas((3840, 2160), None) is None


def test_parallel_chunks_follow_resolution_and_free_ram():
    from render.chunked import parallel_chunks
    assert parallel_chunks((1920, 1080), free_gb=24) == 4
    assert parallel_chunks((3840, 2160), free_gb=24) == 2
    assert parallel_chunks((3840, 2160), free_gb=12) == 1
    assert parallel_chunks((1920, 1080), free_gb=4) == 1


def test_text_is_scaled_to_a_4k_canvas_and_left_alone_at_1080p():
    from render.runner import _scale_text
    from timeline.schema import TextClip, TextStyle
    cap = TimelineItem(id="t", track="TC", kind="text", timeline_start_frame=0,
                       timeline_end_frame=24, text=TextClip(content="hi",
                       style=TextStyle(font_size=46, box_padding=16, stroke_width=0)))
    tl = _tl([cap])
    scaled = {i.id: i for i in _scale_text(tl, (3840, 2160)).items}["t"].text.style
    assert (scaled.font_size, scaled.box_padding, scaled.stroke_width) == (92, 32, 0)
    assert _scale_text(tl, (1920, 1080)) is tl
