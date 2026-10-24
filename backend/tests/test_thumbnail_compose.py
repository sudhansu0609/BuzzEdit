"""presentation/thumbnail_compose.py + agents/thumbnail_agent.py: the thumbnail
is composed like the creator's references (2026-10-21) -- scene, his picture
behind a torn edge, red dripping Devanagari words, a corner label."""
import asyncio
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from presentation import thumbnail_compose as tc


def _picture(path: Path, color, size=(1920, 1080)):
    Image.new("RGB", size, color).save(path)
    return path


def test_compose_puts_the_creator_right_the_words_left_and_a_label_in_the_corner(tmp_path):
    scene = _picture(tmp_path / "scene.png", (20, 60, 110))        # a blue night
    creator = _picture(tmp_path / "me.png", (200, 160, 120))        # a warm face-coloured frame
    out = tc.compose(scene, creator, "जंगल का रेंजर", tmp_path / "thumb.jpg", label="HORROR STORY", genre="horror")
    assert out.exists() and (tmp_path / "thumb_1280.jpg").exists()
    img = np.asarray(Image.open(out).convert("RGB")).astype(int)
    assert img.shape == (1080, 1920, 3)
    # Right edge: the creator's warm picture; left edge: the blue scene.
    right = img[300:800, 1800:1900].mean(axis=(0, 1))
    left = img[300:800, 20:120].mean(axis=(0, 1))
    assert right[0] > right[2] + 30, f"creator panel expected on the right, got {right}"
    assert left[2] > left[0] + 30, f"scene expected on the left, got {left}"
    # The words: strongly red pixels in the left-centre band.
    band = img[250:900, 100:1150]
    red = (band[:, :, 0] > 150) & (band[:, :, 1] < 90) & (band[:, :, 2] < 90)
    assert red.mean() > 0.02, "no red title ink found"
    # The label: bright (white) pixels in the top-left corner.
    corner = img[20:120, 30:420]
    assert ((corner.min(axis=2)) > 200).mean() > 0.01, "no white label found"
    small = Image.open(tmp_path / "thumb_1280.jpg")
    assert small.size == (1280, 720)


def test_compose_survives_a_missing_scene_or_creator(tmp_path):
    creator = _picture(tmp_path / "me.png", (180, 140, 100))
    out = tc.compose(None, creator, "कमरा 209", tmp_path / "a.jpg", label="", genre="horror")
    assert out.exists()
    scene = _picture(tmp_path / "scene.png", (10, 10, 10))
    out = tc.compose(scene, None, "KHALI KURSI", tmp_path / "b.jpg", label="TRUE CRIME", genre="documentary")
    assert out.exists()
    out = tc.compose(None, None, "", tmp_path / "c.jpg")
    assert out.exists()


def test_the_base_picture_is_found_beside_the_recording_or_in_the_channel_folder(tmp_path):
    channel = tmp_path / "Raat3Baje"
    episode = channel / "ep3"
    episode.mkdir(parents=True)
    recording = episode / "take.mp4"
    recording.write_bytes(b"")
    assert tc.find_base_picture(str(recording), {}) is None
    _picture(channel / "thumbnail_base.png", (1, 2, 3), size=(64, 64))
    assert tc.find_base_picture(str(recording), {}) == channel / "thumbnail_base.png"
    _picture(episode / "thumbnail_base.jpg", (1, 2, 3), size=(64, 64))
    assert tc.find_base_picture(str(recording), {}) == episode / "thumbnail_base.jpg"
    explicit = _picture(tmp_path / "me.png", (1, 2, 3), size=(64, 64))
    assert tc.find_base_picture(str(recording), {"thumbnail_base_image": str(explicit)}) == explicit


def test_titles_wrap_into_two_balanced_lines():
    assert tc._wrap_title("जंगल का रेंजर") == ["जंगल का", "रेंजर"]
    assert tc._wrap_title("कमरा 209") == ["कमरा 209"]
    assert tc._wrap_title("दादी की खाली कुर्सी") == ["दादी की", "खाली कुर्सी"]


def test_the_concept_pass_reads_the_words_and_the_label():
    from presentation import assets as assets_stage
    from presentation.program import build_program
    from tests.test_presentation import _timeline

    program = build_program(_timeline(count=200))

    async def ask(system, user):
        assert "THUMB:" in system and "Devanagari" in system and "LABEL:" in system
        return ("TITLE: DADI KI KHALI KURSI\nTHUMB: दादी की खाली कुर्सी वाली रात\n"
                "SCENE: an empty cane chair with a plate of food, night, a dim lantern\nLABEL: horror story")

    concept = asyncio.run(assets_stage.generate_thumbnail_concept(program, ask))
    assert concept["title"] == "DADI KI KHALI KURSI"
    assert concept["thumb"] == "दादी की खाली कुर्सी"          # capped at four words
    assert concept["scene"].startswith("an empty cane chair")
    assert concept["label"] == "HORROR STORY"

    async def none_label(system, user):
        return "TITLE: X\nTHUMB: कमरा 209\nSCENE: a door\nLABEL: NONE"

    concept = asyncio.run(assets_stage.generate_thumbnail_concept(program, none_label))
    assert "label" not in concept and concept["thumb"] == "कमरा 209"


def test_the_agent_composes_from_the_scene_the_base_picture_and_the_words(tmp_path, monkeypatch):
    from agents.thumbnail_agent import ThumbnailAgent
    from models import Project

    scene_out = _picture(tmp_path / "scene_out.png", (20, 60, 110))
    base = _picture(tmp_path / "thumbnail_base.png", (200, 160, 120))
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
    out = asyncio.run(agent.generate_thumbnail(project, tmp_path, title="DADI KI KURSI", genre="horror",
                                               scene="an empty chair at night", words="खाली कुर्सी"))
    assert Path(out).exists() and (tmp_path / "thumbnail_scene.png").exists()
    img = np.asarray(Image.open(out).convert("RGB")).astype(int)
    assert img[300:800, 1800:1900].mean(axis=(0, 1))[0] > 150, "the base picture fills the right panel"
    prompt = next(str(v) for n in queue.graph.values() if isinstance(n, dict)
                  for v in (n.get("inputs") or {}).values() if "empty chair" in str(v))
    assert "camera" not in prompt.lower() and "no text" in prompt
