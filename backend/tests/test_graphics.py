"""Designed graphics (presentation/graphics.py) and the directives that call them.

`GOLDEN` is the contract with BuzzcafStudio: these are exactly the strings its
effect catalog writes (BuzzcafStudio/backend/tests/test_effect_catalog.py::
GOLDEN). Each must parse into the beat that draws it.
"""

import asyncio
import subprocess
from pathlib import Path

import pytest

from presentation import graphics as G
from presentation import script as S
from presentation.models import Asset, Beat, PresentationSettings
from timeline import build_timeline_from_transcript, clip_ops

GOLDEN = {
    "broll_image": "[broll: boot prints in fresh snow at dawn, pine forest | look=blue_duotone]",
    "statement_card": "[statement: If you want *more money*, you have to spend *more hours* on it]",
    "canvas_card": "[canvas: teenager gaming late at night, monitor glow | Me at age 14]",
    "pill_labels": "[pills: Financially free; Retire her parents; Pay off the mortgage]",
    "numbered_point": "[point: 1 | Super friendly for beginners]",
    "name_title": "[name: James Clear | Author of Atomic Habits]",
    "source_quote": "[source_quote: opindia.com | He went undercover as ==Iftikhar Bhatt== to reach "
                    "the militants | a quiet village street in Kashmir at night]",
    "document": "[document: web | www.thesentinel.com | The Hidden Story of ==Operation Rakshak== | "
                "Operations were led by the Army's ==Para Special Forces== units]",
    "timeline": "[timeline: 1975, 1980, *1985*, 1990 | 60 KG | weight gained in four weeks]",
    "transition": "[fx: whiteflash dur=0.5]",
}


class _Program:
    duration_s = 600.0


def _parse(directive: str):
    import re
    match = re.match(r"\[(\w+):\s*(.*)\]$", directive)
    ctx = S.ScriptContext(directives=[S.ScriptDirective(kind=match.group(1), arg=match.group(2),
                                                        tl_at_s=10.0)])
    return S.directive_beats(ctx, _Program(), PresentationSettings())


def test_every_golden_directive_is_in_the_vocabulary():
    from asr.script_align import DIRECTIVES
    import re
    for directive in GOLDEN.values():
        assert re.match(r"\[(\w+):", directive).group(1) in DIRECTIVES


@pytest.mark.parametrize("kind", [k for k in GOLDEN if k != "transition"])
def test_every_golden_directive_parses_into_its_beat(kind):
    beats = _parse(GOLDEN[kind])["beats"]
    assert [b.kind for b in beats] == [kind]


def test_the_golden_fields_land_where_the_renderers_read_them():
    beat = lambda k: _parse(GOLDEN[k])["beats"][0]
    assert beat("broll_image").data["look"] == "blue_duotone"
    assert beat("broll_image").image_prompt == "boot prints in fresh snow at dawn, pine forest"
    assert beat("statement_card").data["with_speaker"] is True
    assert beat("canvas_card").data["caption"] == "Me at age 14"
    assert beat("pill_labels").data["items"] == ["Financially free", "Retire her parents",
                                                 "Pay off the mortgage"]
    assert beat("numbered_point").data["number"] == "1"
    assert beat("name_title").subtext == "Author of Atomic Habits"
    quote = beat("source_quote")
    assert quote.data["source"] == "opindia.com" and "==Iftikhar Bhatt==" in quote.text
    assert quote.image_prompt == "a quiet village street in Kashmir at night"
    doc = beat("document")
    assert doc.data == {"format": "web", "site": "www.thesentinel.com"}
    assert doc.text.startswith("The Hidden Story") and "==Para Special Forces==" in doc.subtext
    timeline = beat("timeline")
    assert timeline.data == {"ticks": ["1975", "1980", "1985", "1990"], "active": 2}
    assert timeline.text == "60 KG" and timeline.subtext == "weight gained in four weeks"


def test_the_transition_is_a_whiteflash_fx():
    fx = _parse(GOLDEN["transition"])["fx"]
    assert fx[0]["effect"] == "whiteflash" and fx[0]["dur"] == pytest.approx(0.5)


# --- drawing ------------------------------------------------------------------------------

def test_marks_split_into_highlighted_words_and_punctuation_stays_attached():
    assert G._tokens("spend *more money*, now") == [
        ("spend", False), ("more", True), ("money,", True), ("now", False)]
    assert G.plain("a ==b c== d") == "a b c d"


@pytest.mark.parametrize("render", [
    lambda: G.render_statement_card("Spend *more hours* on it", 640, 360),
    lambda: G.render_source_quote("BBC", "The dam ==failed overnight==", 640, 360),
    lambda: G.render_document("web", "Headline ==here==", "Body ==text== here", 640, 360, "bbc.com"),
    lambda: G.render_document("book", "Chapter", "Body ==text==", 640, 360, "Book (Author)"),
    lambda: G.render_timeline(["1990", "2000"], 1, "60 KG", "caption", 640, 360),
])
def test_full_frame_cards_render_at_canvas_size(render):
    assert render().size == (640, 360)


def test_overlays_are_transparent_pngs():
    pill = G.render_pill("Financially free", 1080)
    point = G.render_numbered_point("2", "a long point that must wrap onto two lines at most", 1080)
    assert pill.mode == "RGBA" and point.mode == "RGBA"
    assert pill.getpixel((0, 0))[3] == 0
    assert point.width < 1080 * 1.4, "long text wraps rather than running off the frame"


def test_every_look_writes_a_graded_copy(tmp_path):
    from PIL import Image
    src = tmp_path / "pic.png"
    Image.new("RGB", (64, 36), (120, 160, 90)).save(src)
    for look in G.LOOKS:
        out = G.apply_look(str(src), look)
        assert out != str(src) and Path(out).exists()
    assert G.apply_look(str(src), "vaporwave") == str(src)


def test_canvas_cards_are_framed_and_looks_applied_after_generation(tmp_path):
    from PIL import Image
    pic = tmp_path / "gen.png"
    Image.new("RGB", (160, 90), (200, 40, 40)).save(pic)
    beat = Beat(id="c1", kind="canvas_card", start_s=1, end_s=4, image_prompt="x",
                data={"caption": "Me", "look": "bw"})
    asset = Asset(beat_id="c1", kind="image", path=str(pic), width=160, height=90)
    G.post_process_generated([asset], [beat], 640, 360)
    assert Path(asset.path).name.startswith("canvas_") and (asset.width, asset.height) == (640, 360)


# --- placement ----------------------------------------------------------------------------

def _timeline(tmp_path, seconds=12.0, colour="white"):
    from config import FFMPEG_BIN
    source = tmp_path / "src.mp4"
    made = subprocess.run(
        [FFMPEG_BIN, "-y", "-v", "error", "-f", "lavfi", "-i",
         f"color=c={colour}:size=320x180:rate=25:duration={seconds}", "-f", "lavfi", "-i",
         f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(source)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if made.returncode != 0:
        pytest.skip("ffmpeg unavailable")
    words = [{"word": f"w{i}", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(int(seconds / 0.4))]
    return build_timeline_from_transcript(str(source), seconds, words, fps_num=25, fps_den=1,
                                          width=320, height=180, pause_padding_seconds=0.0)


def test_overlays_stay_inside_the_frame_and_off_cutaways(tmp_path):
    from presentation.program import build_program
    timeline = _timeline(tmp_path)
    program = build_program(timeline)
    beats = [
        Beat(id="p", kind="pill_labels", start_s=1.0, end_s=4.0, text="a",
             data={"items": ["One label", "Two label", "Three label"]}),
        Beat(id="n", kind="numbered_point", start_s=5.0, end_s=7.0, text="a long point here",
             data={"number": "3"}),
        Beat(id="x", kind="numbered_point", start_s=9.0, end_s=10.5, text="under a cutaway",
             data={"number": "4"}),
        Beat(id="t", kind="name_title", start_s=11.0, end_s=12.0, text="James Clear", subtext="Author"),
    ]
    counts = G.place_overlays(timeline, beats, program, tmp_path, busy=[(8.9, 11.0)])
    assert counts == {"pill_labels": 1, "numbered_point": 1, "name_title": 1}
    pills = [i for i in timeline.items if (i.label or "").startswith("pill:")]
    assert len(pills) == 3
    assert pills[0].timeline_start_frame < pills[1].timeline_start_frame, "they pop in one by one"
    for item in [i for i in timeline.items if i.origin == G.OVERLAY_ORIGIN and i.kind == "media"]:
        source = timeline.sources[item.source_id]
        half = item.transform.scale * timeline.width / 2
        centre = timeline.width / 2 + item.transform.pos_x * timeline.width / 2
        assert centre - half >= 0 and centre + half <= timeline.width


def test_a_layout_gets_the_speaker_in_its_slot_and_no_ken_burns(tmp_path):
    from presentation import placement
    from presentation.program import build_program
    timeline = _timeline(tmp_path)
    program = build_program(timeline)
    beat = Beat(id="s1", kind="statement_card", start_s=2.0, end_s=5.5, planned_duration_s=3.0,
                text="Spend *more hours*", data={"with_speaker": True})
    drawn, _ = G.render_graphic_assets([beat], tmp_path, 320, 180)
    assert placement.place_broll(timeline, [beat], drawn, program, PresentationSettings(), 0) == 1
    cut = next(i for i in timeline.items if (i.label or "").startswith("layout:"))
    assert cut.transform is None or cut.transform.scale_end is None, "a layout is not zoomed"
    assert G.place_layout_speakers(timeline, [beat], {}) == 1
    speakers = [i for i in timeline.items if i.origin == G.LAYOUT_ORIGIN]
    assert speakers and all(s.transform.crop_left + s.transform.crop_right > 0.3 for s in speakers)


def test_a_designed_edit_renders_for_real(tmp_path):
    """Layout, live speaker slot and alpha overlays through the real compiler."""
    from presentation import placement
    from presentation.program import build_program
    from render.runner import render_timeline_async
    timeline = _timeline(tmp_path, seconds=8.0, colour="blue")
    program = build_program(timeline)
    beats = [Beat(id="s1", kind="statement_card", start_s=1.0, end_s=4.0, planned_duration_s=2.5,
                  text="Spend *more hours*", data={"with_speaker": True}),
             Beat(id="p1", kind="pill_labels", start_s=5.0, end_s=7.5, text="a",
                  data={"items": ["One", "Two"]})]
    drawn, _ = G.render_graphic_assets(beats, tmp_path, 320, 180)
    placement.place_broll(timeline, beats, drawn, program, PresentationSettings(), 0)
    G.place_layout_speakers(timeline, beats, {})
    G.place_overlays(timeline, beats, program, tmp_path, placement.broll_windows(timeline))
    out = tmp_path / "out.mp4"
    asyncio.run(render_timeline_async(timeline, str(out), prefer_nvenc=False))
    assert out.exists() and out.stat().st_size > 0


def test_a_whiteflash_renders_and_brightens_the_window(tmp_path):
    from config import FFMPEG_BIN
    from render.runner import render_timeline_async
    from timeline.schema import AtmosphereEffect
    timeline = _timeline(tmp_path, seconds=4.0, colour="gray")
    layer = clip_ops.add_adjustment_item(timeline, 25, 25)          # 1.0s .. 2.0s
    layer.atmosphere = [AtmosphereEffect(type="whiteflash", intensity=1.0, enabled=True)]
    out = tmp_path / "flash.mp4"
    asyncio.run(render_timeline_async(timeline, str(out), prefer_nvenc=False))

    def luma(at: float) -> float:
        raw = subprocess.run([FFMPEG_BIN, "-v", "error", "-ss", str(at), "-i", str(out), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE).stdout
        return sum(raw) / max(1, len(raw))

    assert luma(1.5) > luma(3.0) + 40, "the middle of the flash is near white"


def test_timed_picture_directions_run_exactly_their_length():
    """BuzzcafStudio times story shots itself (`seconds=`) so clips fill an exact share
    of the video; without it a script shot ended 4-5 s after its word."""
    still = _parse("[broll: a dark corridor | look=noir | seconds=6.5]")["beats"][0]
    clip = _parse("[video: fog over a road, slow push | seconds=7.25]")["beats"][0]
    assert (still.planned_duration_s, round(still.end_s - still.start_s, 3)) == (6.5, 6.5)
    assert still.data == {"look": "noir", "hold": True}
    assert clip.kind == "broll_video" and clip.planned_duration_s == 7.25
    assert clip.video_prompt == "fog over a road, slow push" == clip.image_prompt
    # Over the clip cap it is clamped; without `seconds=` nothing changes.
    assert _parse("[video: a door | seconds=20]")["beats"][0].planned_duration_s == 8.0
    plain = _parse("[video: a door | look=noir]")["beats"][0]
    assert plain.planned_duration_s is None and plain.video_prompt == "a door | look=noir"


def test_timed_shots_sit_back_to_back_without_a_sliver_of_speaker():
    from presentation.placement import _window_for

    class _P:
        duration_s = 600.0
        words = []

    def beat(start, seconds):
        return Beat(kind="broll_video", start_s=start, end_s=start + seconds, priority=1.0,
                    origin="script", planned_duration_s=seconds, data={"hold": True},
                    video_prompt="x", image_prompt="x")

    video = lambda s: Asset(beat_id="b", kind="video", path="x.mp4", duration_s=s + 0.3)
    settings = PresentationSettings(density="busy")
    # 0.1 s after the previous shot ends: pulled back to close the gap.
    start, dur, _ = _window_for(beat(20.1, 6.0), video(6.0), _P(), [(14.0, 20.0)], settings)
    assert (start, round(dur, 3)) == (20.0, 6.1)
    # 0.1 s inside it (rounding): trimmed to start where it ends.
    start, dur, _ = _window_for(beat(19.9, 6.0), video(6.0), _P(), [(14.0, 20.0)], settings)
    assert (start, round(dur, 3)) == (20.0, 5.9)
    # Truly on top of it: refused, never stacked.
    assert _window_for(beat(17.0, 6.0), video(6.0), _P(), [(14.0, 20.0)], settings)[1] == 0.0
