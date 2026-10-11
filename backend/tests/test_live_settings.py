"""presentation/live.py -- the mid-production change: the file a running
pass re-reads, the clip quota, the coverage trim and the clip re-balance."""
from pathlib import Path

from presentation import live
from presentation.models import Beat, PresentationSettings


def _beat(i: int, kind: str = "broll_image", priority: float = 0.5, start: float = 0.0,
          length: float = 5.0, origin: str = "script") -> Beat:
    return Beat(id=f"b{i:02d}", start_s=start, end_s=start + length, topic=f"t{i}", kind=kind,
                priority=priority, image_prompt="a forest at night", planned_duration_s=length,
                origin=origin)


def test_file_round_trip_keeps_only_live_keys(tmp_path: Path):
    project_dir = tmp_path / "abc123"
    assert live.read(project_dir) == {}
    stored = live.write(project_dir, {"video_broll_share": 0.35, "target_coverage": 0.85,
                                      "genre": "horror", "broll": False})
    assert stored == {"video_broll_share": 0.35, "target_coverage": 0.85}
    assert live.read(project_dir) == stored
    assert live.path_for(project_dir).name == "abc123_live_settings.json"
    live.write(project_dir, {})
    assert live.read(project_dir) == {}
    assert not live.path_for(project_dir).exists()


def test_apply_validates_like_settings():
    base = PresentationSettings(target_coverage=0.9, video_broll_share=0.7)
    out = live.apply(base, {"target_coverage": 0.85, "video_broll_share": 0.353, "broll": False})
    assert (out.target_coverage, out.video_broll_share, out.broll) == (0.85, 0.353, True)
    assert live.apply(base, {}) is base


def test_clip_quota_is_a_share_of_broll_seconds():
    beats = [_beat(i, length=5.0) for i in range(10)]
    assert live.clip_quota_seconds(beats, 0.3) == 15.0
    assert live.clip_quota_seconds(beats, 1.5) == 50.0


def test_trim_drops_lowest_priority_cutaways_to_the_new_coverage():
    beats = [_beat(i, priority=i / 10, start=i * 10.0, length=5.0) for i in range(10)]
    settings = PresentationSettings(target_coverage=0.25)  # 25 s of a 100 s programme
    kept, dropped = live.trim_to_coverage(beats, settings, duration_s=100.0)
    assert [b.id for b in kept] == ["b05", "b06", "b07", "b08", "b09"]
    assert len(dropped) == 5 and all(d["reason"].startswith("over the coverage") for d in dropped)
    # Already within budget: untouched.
    same, none = live.trim_to_coverage(kept, PresentationSettings(target_coverage=0.5), 100.0)
    assert same == kept and none == []


def test_rebalance_keeps_made_clips_then_priority():
    beats = [_beat(i, kind="broll_video", priority=i / 10, start=i * 10.0) for i in range(10)]
    settings = PresentationSettings(video_broll_share=0.3)  # 15 s of 50 s -> 3 clips
    out = live.rebalance_clips(beats, settings, made_clips={"b01", "b02"})
    clips = [b.id for b in out if b.kind == "broll_video"]
    assert clips == ["b01", "b02", "b09"]
    assert all(b.kind == "broll_image" for b in out if b.id not in clips)
    # A raised share promotes stills, deriving a motion prompt.
    stills = [_beat(i, priority=i / 10, start=i * 10.0) for i in range(4)]
    up = live.rebalance_clips(stills, PresentationSettings(video_broll_share=0.5), set())
    promoted = [b for b in up if b.kind == "broll_video"]
    assert [b.id for b in promoted] == ["b02", "b03"] and all(b.video_prompt for b in promoted)
