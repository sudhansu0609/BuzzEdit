"""OWN_FOOTAGE_PLAN.md OF3: `own=` clips on a `[video:]` direction."""

from backend.presentation import assets as assets_stage
from backend.presentation import script as script_stage
from backend.presentation.live import rebalance_clips
from backend.presentation.models import Beat, PresentationSettings, Program
from backend.presentation.shotplan import balance_video_share

WIN_PATH = r"C:\Users\me\My Clips\night walk 01.mp4"


def test_own_option_survives_spaces_and_the_drive_colon():
    program = Program(duration_s=100.0, fps=30.0)
    arg = f"his lane at night | own={WIN_PATH} | in=3.5 | seconds=6 | natsound=0.4"
    ctx = script_stage.ScriptContext(
        directives=[script_stage.ScriptDirective(kind="video", arg=arg, tl_at_s=2.0)])
    beat = script_stage.directive_beats(ctx, program, PresentationSettings())["beats"][0]
    assert beat.own_path == WIN_PATH
    assert beat.own_in_s == 3.5 and beat.natsound == 0.4
    assert beat.planned_duration_s == 6.0 and beat.video_prompt == "his lane at night"


def test_own_asset_picks_the_segment_and_clamps(tmp_path, monkeypatch):
    clip = tmp_path / "walk.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(assets_stage, "_probe_own", lambda p: (1920, 1080, 10.0, True))

    def beat(bid, in_s, secs):
        return Beat(id=bid, start_s=0.0, end_s=secs, kind="broll_video", planned_duration_s=secs,
                    own_path=str(clip), own_in_s=in_s)

    held, short, gone = beat("a", 4.0, 6.3), beat("b", 4.0, 8.0), beat("c", 9.9, 4.0)
    made, failed = assets_stage.resolve_own_assets([held, short, gone], tmp_path / "proj")
    by_id = {a.beat_id: a for a in made}
    assert by_id["a"].duration_s == 6.3 and by_id["a"].own_in_s == 4.0  # holds <= 0.5 s
    assert by_id["b"].duration_s == 6.0 and short.planned_duration_s == 6.0  # shortened
    assert by_id["a"].source == "own" and by_id["a"].natsound == 0.15
    assert [f["beat_id"] for f in failed] == ["c"]
    missing = beat("d", 0.0, 4.0).model_copy(update={"own_path": str(tmp_path / "nope.mp4")})
    assert assets_stage.resolve_own_assets([missing], tmp_path / "proj")[0] == []


def test_balancing_passes_leave_own_beats_alone():
    own = Beat(id="o", start_s=0.0, end_s=6.0, kind="broll_video", planned_duration_s=6.0,
               origin="llm", own_path=WIN_PATH)
    still = Beat(id="s", start_s=10.0, end_s=16.0, kind="broll_image", image_prompt="a road",
                 planned_duration_s=6.0, origin="llm", priority=0.9)
    settings = PresentationSettings(video_broll_share=0.0)
    assert [b.kind for b in rebalance_clips([own, still], settings)] == ["broll_video", "broll_image"]
    settings = PresentationSettings(video_broll_share=0.9)
    assert balance_video_share([own, still], settings)[0].kind == "broll_video"
