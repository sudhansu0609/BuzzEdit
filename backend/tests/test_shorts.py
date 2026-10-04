"""Vertical variants and Shorts: reframing, slicing and picking."""

import pytest

from backend.presentation import shorts
from backend.presentation.models import Topic
from backend.presentation.program import build_program
from backend.timeline import build_timeline_from_transcript, clip_ops
from backend.timeline.schema import SourceFile


def _timeline(width=1920, height=1080, seconds=90.0):
    words = [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4}
             for i in range(int(seconds * 2) - 1)]
    return build_timeline_from_transcript(
        "C:/media/talk.mp4", seconds, words, fps_num=30, fps_den=1, width=width, height=height,
        speech_regions=[(0.0, seconds)], pause_padding_seconds=0.0)


def test_cover_scale_and_face_position():
    assert shorts.cover_scale(1920, 1080, 1080, 1920) == pytest.approx(1920 / 607.5, rel=1e-3)
    assert shorts.cover_scale(1080, 1920, 1080, 1920) == 1.0
    assert shorts.face_pos_x(None) == 0.0
    # The zoom stage's pulled-in anchor is undone; the result stays in range.
    assert shorts.face_pos_x((0.3, 0.0)) == pytest.approx(0.5)
    assert shorts.face_pos_x((-0.9, 0.0)) == -1.0


def test_a_landscape_timeline_is_reframed_around_the_face():
    timeline = _timeline()
    program = build_program(timeline)
    timeline.sources["b"] = SourceFile(id="b", path="C:/b.png", duration_seconds=3, width=1920,
                                       height=1080, has_audio=False, kind="image")
    broll = clip_ops.add_media_item(timeline, "b", "V3", 60, 0, 90, origin="broll")
    clip_ops.set_transform(timeline, broll.id, {"scale": 1.0, "scale_end": 1.1})
    first = program.segments[0].item_id
    vertical = shorts.vertical_variant(timeline, program, {first: (0.3, -0.1)})
    assert (vertical.width, vertical.height) == (1080, 1920)
    v1 = next(i for i in vertical.items if i.id == first)
    assert v1.transform.scale == pytest.approx(shorts.cover_scale(1920, 1080, 1080, 1920))
    assert v1.transform.pos_x == pytest.approx(0.5)
    cut = next(i for i in vertical.items if i.origin == "broll")
    assert cut.transform.scale_end == pytest.approx(1.1 * shorts.cover_scale(1920, 1080, 1080, 1920))
    # The original is untouched, and a portrait source is left alone.
    assert timeline.width == 1920
    portrait = _timeline(1080, 1920)
    same = shorts.vertical_variant(portrait, build_program(portrait))
    assert next(i for i in same.items if i.track == "V1").transform is None


def test_slicing_keeps_every_overlapping_layer_and_rebases_time():
    timeline = _timeline()
    timeline.sources["m"] = SourceFile(id="m", path="C:/bed.wav", duration_seconds=5,
                                       has_audio=True, kind="audio", width=0, height=0)
    bed = clip_ops.add_media_item(timeline, "m", "A2", 0, 0, timeline.duration_frames, origin="music")
    bed.loop = True
    card = clip_ops.add_text_item(timeline, "HELLO THERE", 290, 60, track="TC",
                                  style={"animation": "karaoke"})
    card.text.words = [{"text": "HELLO", "start_s": 0.0, "end_s": 0.9},
                       {"text": "THERE", "start_s": 1.0, "end_s": 1.9}]
    clip_ops.add_text_item(timeline, "later", 2000, 30, track="T1")

    clip = shorts.slice_timeline(timeline, 10.0, 20.0)
    assert clip.duration_frames == 300
    assert not any(i.text and i.text.content == "later" for i in clip.items)
    sliced_bed = next(i for i in clip.items if i.origin == "music")
    assert sliced_bed.timeline_start_frame == 0 and sliced_bed.timeline_end_frame == 300
    assert sliced_bed.source_end_frame - sliced_bed.source_start_frame == 300
    sliced_card = next(i for i in clip.items if i.kind == "text" and i.track == "TC")
    assert sliced_card.timeline_start_frame == 0 and sliced_card.timeline_end_frame == 50
    # The card started 10 frames before the slice: its first word is mostly gone.
    assert [w["text"] for w in sliced_card.text.words] == ["HELLO", "THERE"]
    assert sliced_card.text.words[1]["start_s"] == pytest.approx(1.0 - 10 / 30, abs=0.01)
    v1 = [i for i in clip.items if i.track == "V1"]
    assert v1 and v1[0].timeline_start_frame == 0
    assert sum(i.duration_frames for i in v1) == 300


def test_shorts_are_the_best_topics_clipped_to_the_limit_and_never_the_cta():
    timeline = _timeline(seconds=200.0)
    program = build_program(timeline)
    topics = [Topic(start_s=0, end_s=30, topic="a", priority=0.4),
              Topic(start_s=30, end_s=110, topic="long", priority=0.9),
              Topic(start_s=110, end_s=118, topic="short", priority=1.0),
              Topic(start_s=118, end_s=160, topic="b", priority=0.7),
              Topic(start_s=160, end_s=200, topic="outro", priority=0.95, act="cta")]
    picked = shorts.pick_shorts(topics, program, 2)
    assert [t.topic for _, _, t in picked] == ["long", "b"]
    start, end, _ = picked[0]
    assert end - start <= shorts.SHORT_MAX_S + 0.5
    assert shorts.pick_shorts(topics, program, 0) == []


def test_a_long_topic_keeps_its_conclusion_and_ends_on_a_sentence():
    from presentation.models import ProgramWord, Program as _Program
    # 90 s topic, one word per second, a sentence ending every 10th word.
    words = [ProgramWord(text=f"w{i}." if i % 10 == 9 else f"w{i}", tl_start_s=float(i),
                         tl_end_s=i + 0.8, source_start_frame=i * 30) for i in range(90)]
    program = _Program(words=words, duration_s=95.0, fps=30.0)
    topics = [Topic(start_s=0, end_s=88.5, topic="long", priority=0.9)]
    (start, end, _), = shorts.pick_shorts(topics, program, 1)
    assert end - start <= shorts.SHORT_MAX_S + 0.5
    # Runs on past the topic edge to finish the last sentence (word 89 ends at 89.8).
    assert end == pytest.approx(89.8 + 0.3, abs=0.01)
    # Trimmed from the front, at a sentence start (a multiple of 10 s).
    assert round(start + 0.1) % 10 == 0
