"""Horror zooms creep: the move leaves the clip's prompt and comes back as a slow timeline zoom."""

import pytest

from presentation import camera
from presentation.models import Beat


@pytest.mark.parametrize("prompt,move,kept", [
    ("Slow zoom in on her face in the dark corridor, dim light", "in", "Her face in the dark corridor"),
    ("A slow push toward the door, flickering bulb", "in", "The door, flickering bulb"),
    ("pull back from the empty bed, cold blue light", "out", "the empty bed, cold blue light"),
    ("dolly in to the mirror", "in", "the mirror"),
    ("Gentle push in on the photograph, dust in the air", "in", "The photograph, dust in the air"),
])
def test_the_move_is_found_and_only_the_move_is_removed(prompt, move, kept):
    assert camera.camera_move(prompt) == move
    steady = camera.steady_prompt(prompt)
    assert steady.startswith(kept) and steady.endswith(camera.STEADY)


@pytest.mark.parametrize("prompt", ["a candle on the table, static", "a pushcart in the old market",
                                    "", None])
def test_no_move_no_change(prompt):
    assert camera.camera_move(prompt) is None


def _beat(kind="broll_video", prompt="Slow zoom in on the well, fog"):
    return Beat(start_s=0.0, end_s=3.0, kind=kind, video_prompt=prompt, image_prompt=prompt)


def test_only_horror_video_beats_are_steadied():
    horror = [_beat(), _beat(kind="broll_image"), _beat(prompt="the well at night")]
    assert camera.steady_for_genre(horror, "horror") == 1
    assert horror[0].camera_move == "in" and "zoom" not in horror[0].video_prompt.lower()
    assert camera.NO_MOVE_NEGATIVE in horror[0].negative_prompt
    assert horror[1].camera_move is None and horror[2].camera_move is None
    vlog = [_beat()]
    assert camera.steady_for_genre(vlog, "vlog") == 0 and vlog[0].camera_move is None
    assert camera.steady_for_genre(horror, "horror") == 0          # never twice


def test_the_timeline_zoom_is_slow_and_follows_the_direction():
    assert camera.gradual_depth(1.0) == camera.GRADUAL_MIN
    assert camera.gradual_depth(3.0) == pytest.approx(0.06)
    assert camera.gradual_depth(20.0) == camera.GRADUAL_MAX
    from timeline import build_timeline_from_transcript
    from timeline import clip_ops
    from presentation.placement import _apply_gradual_zoom
    tl = build_timeline_from_transcript("src.mp4", 10.0, [{"word": "a", "start": 0.0, "end": 9.0}])
    sid = next(iter(tl.sources))
    for direction, (start, end) in (("in", (1.0, 1.06)), ("out", (1.06, 1.0))):
        item = clip_ops.add_media_item(tl, sid, "V2", 0, 0, 90)
        _apply_gradual_zoom(tl, item.id, direction, 3.0)
        t = next(i for i in tl.items if i.id == item.id).transform
        assert (t.scale, t.scale_end) == pytest.approx((start, end))
