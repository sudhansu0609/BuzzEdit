import asyncio

from presentation import assets
from presentation.models import Program, ProgramWord, Topic


def _program(seconds: int = 200) -> Program:
    words = [ProgramWord(text=f"word{i}", tl_start_s=float(i), tl_end_s=i + 0.8,
                         source_start_frame=i * 30) for i in range(seconds)]
    return Program(words=words, duration_s=float(seconds), fps=30.0)


def test_concept_reads_the_whole_video_and_parses_title_and_scene():
    seen = {}

    async def ask(system, user):
        seen["user"] = user
        return "**TITLE:** START SMALL TODAY\nSCENE: A hand drawing the first line in an empty notebook, warm light"

    concept = asyncio.run(assets.generate_thumbnail_concept(
        _program(), ask, [Topic(start_s=0, end_s=50, topic="The big trap")]))
    assert concept == {"title": "START SMALL TODAY",
                       "scene": "A hand drawing the first line in an empty notebook, warm light"}
    # Opening, middle and ending of the transcript, plus the topics.
    for part in ("Opening:", "Middle:", "Ending:", "word199", "Topics: The big trap"):
        assert part in seen["user"]


def test_a_bare_title_answer_still_works():
    async def ask(system, user):
        return "PERFECTION IS A TRAP"

    assert asyncio.run(assets.generate_thumbnail_title(_program(60), ask)) == "PERFECTION IS A TRAP"
