"""presentation/art_style.py (2026-10-21): a drawn art style changes the look
stamped on every prompt, keeps a real face off the drawings, draws the
thumbnail scene in the medium and restyles the text cards."""
import asyncio
from pathlib import Path

import pytest
from PIL import Image

from presentation import art_style as A
from presentation import cards
from presentation import genre as G
from presentation.models import PresentationSettings


def test_styles_normalise_and_the_default_is_real():
    assert A.ART_STYLES == ("real", "explainer", "stick_figure", "cartoon", "painterly",
                            "illustrated", "stick_world", "mixed", "mixed_quirky")
    assert A.normalise("whiteboard") == "stick_figure" and A.normalise(None) == "real"
    assert A.uses_face_swap("real") and not A.uses_face_swap("cartoon")
    assert PresentationSettings().art_style == "real"


def test_mixed_is_real_based_but_cards_take_the_explainer_look():
    assert A.normalise("mix") == "mixed" and A.normalise("combo") == "mixed"
    assert A.normalise("hybrid") == "mixed" and A.normalise("mixed_media") == "mixed"
    assert A.uses_face_swap("mixed")
    assert A.card_style("mixed") == A.card_style("explainer")
    assert A.is_photoreal_base("mixed") and A.is_photoreal_base("real")
    assert not A.is_photoreal_base("stick_figure")
    assert not A.is_real("mixed"), "is_real stays true only for \"real\""


def test_mixed_quirky_is_mostly_drawn_but_still_face_swaps():
    assert A.normalise("quirky") == "mixed_quirky"
    assert A.normalise("quirky_doc") == "mixed_quirky"
    assert A.normalise("quirky_documentary") == "mixed_quirky"
    assert A.uses_face_swap("mixed_quirky"), "a per-shot art=real still gets the face swap"
    assert A.card_style("mixed_quirky") == A.card_style("explainer")
    assert not A.is_photoreal_base("mixed_quirky"), "mostly drawn: no grain, no photographic look"
    assert not A.is_real("mixed_quirky")


def test_a_drawn_style_replaces_the_genre_look_with_its_medium_and_mood():
    real = G.apply_look("a man in a dark corridor", "horror")
    assert "low-key lighting" in real
    stick = G.apply_look("a man in a dark corridor", "horror", None, "stick_figure")
    assert "low-key lighting" not in stick and "black marker" in stick and "eerie" in stick
    assert G.apply_look(stick, "horror", None, "stick_figure") == stick, "stamped once"
    assert G.apply_look("x", "general", None, "real") == G.apply_look("x", "general")
    negative = G.apply_negative("text, watermark", "horror", None, "explainer")
    assert "photorealistic" in negative and "cartoon" not in negative
    block = G.genre_block("horror", "cartoon")
    assert "ART STYLE" in block and "cel shading" in block and "low-key lighting" not in block
    assert "ART STYLE" not in G.genre_block("horror")


def test_cards_take_the_medium_s_fonts():
    assert cards._style_for("chapter_title", "horror")["font_family"] == "Georgia"
    assert cards._style_for("chapter_title", "horror", "stick_figure")["font_family"] == "Segoe Print"
    assert cards._style_for("stat_callout", "finance", "explainer")["font_family"] == "Poppins"
    assert cards._style_for("stat_callout", "finance", "real")["color"] == "#8CFF9E"


def test_the_thumbnail_scene_is_drawn_in_the_medium(tmp_path, monkeypatch):
    from agents.thumbnail_agent import ThumbnailAgent
    from models import Project

    scene_out = tmp_path / "scene.png"
    Image.new("RGB", (64, 64), (20, 60, 110)).save(scene_out)
    recording = tmp_path / "take.mp4"
    recording.write_bytes(b"")

    class Queue:
        async def submit_and_wait(self, graph, timeout=0):
            self.graph = graph
            return [str(scene_out)]

    queue = Queue()
    monkeypatch.setattr("agents.thumbnail_agent.run_ffmpeg", lambda *a, **k: None)
    project = Project(id="p1", name="take.mp4", source_video=str(recording), settings={})
    agent = ThumbnailAgent(queue_manager=queue)
    asyncio.run(agent.generate_thumbnail(project, tmp_path, title="OKENE", genre="documentary",
                                         scene="a man in an air pocket inside a sunken ship",
                                         words="60 घंटे", art_style="stick_figure"))
    prompt = next(str(v) for n in queue.graph.values() if isinstance(n, dict)
                  for v in (n.get("inputs") or {}).values() if "air pocket" in str(v))
    assert "black marker" in prompt and "Cinematic film still" not in prompt and "no text" in prompt


def test_painterly_is_a_drawn_style_with_its_own_cards():
    assert A.normalise("gouache") == "painterly" and A.normalise("painting") == "painterly"
    assert not A.uses_face_swap("painterly")
    assert A.card_style("painterly")["font_family"] == "Georgia"
    painted = G.apply_look("a lamp in a far window", "horror", None, "painterly")
    assert A.look_for("painterly") in painted and "low-key lighting" not in painted


def test_illustrated_is_a_drawn_style_never_photoreal():
    assert not A.uses_face_swap("illustrated")
    illustrated = G.apply_look("a lamp in a far window", "horror", None, "illustrated")
    assert A.look_for("illustrated") in illustrated and "low-key lighting" not in illustrated
    assert "photorealistic" in A.negative_for("illustrated") and "anime" in A.negative_for("illustrated")
