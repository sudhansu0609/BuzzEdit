import pytest

from backend.render.compiler import FilterGraphCompiler, flatten_items
from backend.render.text import build_drawtext
from backend.timeline import build_timeline_from_transcript, clip_ops
from backend.timeline.authoring import (
    apply_intro, generate_captions, remove_captions, remove_intro, source_to_timeline_frame,
)
from backend.timeline.schema import SourceFile, TextClip, TextStyle, Timeline, TimelineItem


WORDS = [
    {"word": "Hello", "start": 0.0, "end": 0.9},
    {"word": "um", "start": 0.9, "end": 1.2, "disfluency": True},
    {"word": "there", "start": 1.2, "end": 1.8},
    {"word": "friends", "start": 1.8, "end": 2.4},
    {"word": "welcome", "start": 4.0, "end": 4.6},
    {"word": "back", "start": 4.6, "end": 5.0},
]


def _timeline():
    return build_timeline_from_transcript("C:/media/main.mp4", 10.0, WORDS,
                                          width=1920, height=1080,
                                          pause_padding_seconds=0.0)


def _with_manual_clips():
    tl = _timeline()
    tl.sources["src_b"] = SourceFile(id="src_b", path="C:/media/broll.mp4",
                                     duration_seconds=20.0, kind="video")
    a = clip_ops.add_media_item(tl, "src_b", "V2", 30, 0, 60)
    b = clip_ops.add_media_item(tl, "src_b", "V3", 90, 0, 60)
    return tl, a, b


# --- text clips -----------------------------------------------------------

def test_add_text_lands_on_a_text_track(tmp_path):
    tl = _timeline()
    item = clip_ops.add_text_item(tl, "Hello world", 30, 60, preset="title_bold")
    assert item.track == "T1"
    assert item.kind == "text"
    assert item.source_id is None
    assert item.text.style.bold is True          # from the preset
    assert item.text.style.font_size == 96


def test_text_cannot_be_placed_on_a_video_track():
    tl = _timeline()
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.add_text_item(tl, "nope", 0, 30, track="V2")


def test_changing_preset_replaces_the_look_not_merges_it():
    tl = _timeline()
    item = clip_ops.add_text_item(tl, "Hi", 0, 30, preset="youtube_pop")
    assert item.text.style.stroke_width == 8
    clip_ops.set_text(tl, item.id, preset="minimal")
    assert item.text.style.stroke_width == 0     # minimal has no outline
    assert item.text.content == "Hi"             # copy is preserved


def test_style_overrides_survive_later_edits():
    tl = _timeline()
    item = clip_ops.add_text_item(tl, "Hi", 0, 30, preset="minimal")
    clip_ops.set_text(tl, item.id, style={"font_size": 123})
    clip_ops.set_text(tl, item.id, content="Bye")
    assert item.text.style.font_size == 123
    assert item.text.content == "Bye"


def test_drawtext_always_pins_a_fontfile(tmp_path):
    """A drawtext with no fontfile falls back to the fontconfig default, which on
    a Windows ffmpeg build with no config aborts the whole render (exit 139). The
    filter must always carry an explicit fontfile — or draw nothing — never a
    bare default."""
    clip = TextClip(content="Caption line", style=TextStyle())
    built = build_drawtext(clip, 1.0, 4.0, tmp_path)
    # Either a real font is pinned, or the clip is skipped — never a fontless
    # drawtext that would crash the render.
    assert built is None or "fontfile='" in built


def test_a_chosen_caption_preset_owns_its_band_over_a_profile_position():
    """Picking a preset must place captions in that preset's band; a style
    profile's measured position (colour etc. still apply) must not drag them off
    it. The callers strip position from the stored overrides before generating."""
    tl = _timeline()
    stored = {"pos_y": 0.1, "color": "#ff0000"}          # a reference-video profile
    overrides = {k: v for k, v in stored.items() if k not in ("pos_x", "pos_y")}
    caps = generate_captions(tl, preset="youtube_shorts", style_overrides=overrides)
    bands = {round(c.text.style.pos_y, 3) for c in caps}
    assert bands and 0.1 not in bands                     # not forced to the profile
    assert all(c.text.style.color == "#ff0000" for c in caps)   # colour still carried


def test_drawtext_writes_a_sidecar_and_quotes_its_expressions(tmp_path):
    clip = TextClip(content="It's 3:30 — 100% ready", style=TextStyle(animation="fade"))
    built = build_drawtext(clip, 1.0, 4.0, tmp_path)
    assert "textfile='" in built
    # Awkward characters go in the sidecar, never into the filter string.
    assert "It's" not in built
    assert "x='" in built and "y='" in built and "alpha='" in built
    assert "enable='between(t,1,4)'" in built
    written = list(tmp_path.glob("text_*.txt"))
    assert len(written) == 1
    # The percent sign is escaped for drawtext's expansion: a bare "%" is a
    # parse error that draws nothing at all.
    assert written[0].read_text(encoding="utf-8") == "It's 3:30 — 100\% ready"


def test_identical_text_reuses_one_sidecar(tmp_path):
    clip = TextClip(content="same")
    build_drawtext(clip, 0.0, 1.0, tmp_path)
    build_drawtext(clip, 2.0, 3.0, tmp_path)
    assert len(list(tmp_path.glob("text_*.txt"))) == 1


def test_blank_text_draws_nothing(tmp_path):
    assert build_drawtext(TextClip(content="   "), 0.0, 1.0, tmp_path) is None


def test_text_trim_does_not_consult_a_source():
    tl = _timeline()
    item = clip_ops.add_text_item(tl, "Hi", 30, 60)
    clip_ops.trim_item(tl, item.id, "start", 45)
    clip_ops.trim_item(tl, item.id, "end", 100)
    assert (item.timeline_start_frame, item.timeline_end_frame) == (45, 100)


# --- detach audio ---------------------------------------------------------

def test_detach_audio_creates_the_matching_audio_clip():
    tl, clip, _ = _with_manual_clips()
    detached = clip_ops.detach_audio(tl, clip.id)
    assert detached.track == "A2"
    assert detached.source_id == clip.source_id
    assert detached.timeline_start_frame == clip.timeline_start_frame
    assert detached.source_start_frame == clip.source_start_frame
    assert clip.mute is True


def test_detaching_twice_is_refused():
    tl, clip, _ = _with_manual_clips()
    clip_ops.detach_audio(tl, clip.id)
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.detach_audio(tl, clip.id)


def test_cannot_detach_audio_from_a_still():
    tl = _timeline()
    tl.sources["src_img"] = SourceFile(id="src_img", path="C:/media/a.png",
                                       duration_seconds=5, kind="image", has_audio=False)
    still = clip_ops.add_media_item(tl, "src_img", "V2", 0, 0, 60)
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.detach_audio(tl, still.id)


def test_detached_audio_reaches_the_render():
    tl, clip, _ = _with_manual_clips()
    clip_ops.detach_audio(tl, clip.id)
    _inputs, fc, _v, alabel = FilterGraphCompiler(tl).compile()
    assert alabel == "[mix_a]"
    assert "adelay=1000:all=1" in fc          # clip starts at frame 30 @ 30fps


# --- compound clips -------------------------------------------------------

def test_compound_groups_clips_and_stores_them_relative():
    tl, a, b = _with_manual_clips()
    group = clip_ops.make_compound(tl, [a.id, b.id])
    assert group.kind == "compound"
    assert group.timeline_start_frame == 30 and group.timeline_end_frame == 150
    assert [c.timeline_start_frame for c in group.children] == [0, 60]
    assert clip_ops.get_item(tl, a.id) is None


def test_compound_needs_more_than_one_clip():
    tl, a, _ = _with_manual_clips()
    with pytest.raises(clip_ops.ClipOpError):
        clip_ops.make_compound(tl, [a.id])


def test_moving_a_compound_moves_everything_inside_it():
    tl, a, b = _with_manual_clips()
    group = clip_ops.make_compound(tl, [a.id, b.id])
    clip_ops.move_item(tl, group.id, 300)
    flat = flatten_items(tl.items)
    starts = sorted(i.timeline_start_frame for i in flat if i.source_id == "src_b")
    assert starts == [300, 360]


def test_trimming_a_compound_trims_its_contents():
    tl, a, b = _with_manual_clips()
    group = clip_ops.make_compound(tl, [a.id, b.id])
    clip_ops.trim_item(tl, group.id, "end", 120)      # cut the second child in half
    flat = [i for i in flatten_items(tl.items) if i.source_id == "src_b"]
    assert len(flat) == 2
    tail = max(flat, key=lambda i: i.timeline_start_frame)
    assert tail.timeline_end_frame == 120
    assert tail.source_end_frame == 30                # source trimmed to match


def test_trimming_a_compound_start_holds_its_contents_in_place():
    tl, a, b = _with_manual_clips()
    group = clip_ops.make_compound(tl, [a.id, b.id])
    clip_ops.trim_item(tl, group.id, "start", 60)
    flat = [i for i in flatten_items(tl.items) if i.source_id == "src_b"]
    # The second child never moved; the first was cut down to the new start.
    assert sorted(i.timeline_start_frame for i in flat) == [60, 90]


def test_splitting_a_compound_rebases_the_right_half():
    tl, a, b = _with_manual_clips()
    group = clip_ops.make_compound(tl, [a.id, b.id])
    clip_ops.split_item(tl, group.id, 90)
    flat = [i for i in flatten_items(tl.items) if i.source_id == "src_b"]
    assert sorted(i.timeline_start_frame for i in flat) == [30, 90]


def test_uncompound_restores_independent_clips():
    tl, a, b = _with_manual_clips()
    group = clip_ops.make_compound(tl, [a.id, b.id])
    restored = clip_ops.break_compound(tl, group.id)
    assert len(restored) == 2
    assert sorted(i.timeline_start_frame for i in restored) == [30, 90]
    assert all(i.kind == "media" for i in restored)
    assert clip_ops.get_item(tl, group.id) is None


def test_compound_children_inherit_the_group_transform():
    tl, a, b = _with_manual_clips()
    group = clip_ops.make_compound(tl, [a.id, b.id])
    clip_ops.set_transform(tl, group.id, {"scale": 0.5})
    flat = [i for i in flatten_items(tl.items) if i.source_id == "src_b"]
    assert all(i.transform is not None and i.transform.scale == 0.5 for i in flat)


# --- captions -------------------------------------------------------------

def test_captions_follow_the_cut_not_the_raw_timestamps():
    tl = _timeline()
    segments = [(i.source_start_frame, i.source_end_frame, i.timeline_start_frame)
                for i in tl.items if i.track == "V1"]
    # "welcome" starts at 4.0s in the source but the removed filler and the gap
    # pull it earlier on the timeline.
    raw = int(round(4.0 * 30))
    mapped = source_to_timeline_frame(sorted(segments, key=lambda s: s[2]), raw)
    assert mapped is not None and mapped < raw


def test_generate_captions_splits_on_the_preset_word_count():
    tl = _timeline()
    cards = generate_captions(tl, "youtube_shorts")     # 3 words per card, uppercase
    assert cards
    assert all(c.origin == "caption" and c.track == "TC" for c in cards)
    assert all(len(c.text.content.split()) <= 3 for c in cards)
    assert cards[0].text.content == cards[0].text.content.upper()
    assert "UM" not in " ".join(c.text.content for c in cards)   # disfluency was cut


def test_captions_never_overlap():
    tl = _timeline()
    cards = sorted(generate_captions(tl, "podcast"), key=lambda c: c.timeline_start_frame)
    for earlier, later in zip(cards, cards[1:]):
        assert earlier.timeline_end_frame <= later.timeline_start_frame


def test_regenerating_captions_replaces_the_previous_batch():
    tl = _timeline()
    generate_captions(tl, "classic")
    first = len([i for i in tl.items if i.origin == "caption"])
    generate_captions(tl, "youtube_shorts")
    second = [i for i in tl.items if i.origin == "caption"]
    assert first and len(second) != 0
    assert all(len(i.text.content.split()) <= 3 for i in second)


def test_clearing_captions_leaves_manual_text_alone():
    tl = _timeline()
    manual = clip_ops.add_text_item(tl, "keep me", 0, 30)
    generate_captions(tl, "classic")
    remove_captions(tl)
    assert clip_ops.get_item(tl, manual.id) is not None
    assert not [i for i in tl.items if i.origin == "caption"]


# --- intros ---------------------------------------------------------------

def test_intro_pushes_the_program_back_and_adds_text():
    tl = _timeline()
    apply_intro(tl, "title_card", title="My Show", subtitle="Episode 1")
    assert tl.program_offset_frames == 90            # 3.0s @ 30fps
    v1 = [i for i in tl.items if i.track == "V1"]
    assert min(i.timeline_start_frame for i in v1) == 90
    beats = [i for i in tl.items if i.origin == "intro"]
    assert {b.text.content for b in beats} == {"My Show", "Episode 1"}


def test_intro_card_is_concatenated_in_front_of_the_program():
    tl = _timeline()
    apply_intro(tl, "title_card", title="Hi")
    _inputs, fc, _v, _a = FilterGraphCompiler(tl).compile()
    assert "color=c=black:s=1920x1080" in fc
    assert "[pad_v]" in fc and "[pad_a]" in fc
    assert "anullsrc=r=48000:cl=stereo,atrim=duration=3.000" in fc


def test_overlay_style_intro_shifts_nothing():
    tl = _timeline()
    apply_intro(tl, "overlay_hook", title="Wait for it")
    assert tl.program_offset_frames == 0
    assert [i for i in tl.items if i.origin == "intro"]


def test_beats_without_content_are_skipped():
    tl = _timeline()
    apply_intro(tl, "title_card", title="Only a title")   # no subtitle supplied
    assert len([i for i in tl.items if i.origin == "intro"]) == 1


def test_removing_an_intro_pulls_the_program_back():
    tl = _timeline()
    before = min(i.timeline_start_frame for i in tl.items if i.track == "V1")
    apply_intro(tl, "title_card", title="Hi")
    remove_intro(tl)
    assert tl.program_offset_frames == 0
    assert min(i.timeline_start_frame for i in tl.items if i.track == "V1") == before
    assert not [i for i in tl.items if i.origin == "intro"]


def test_intro_keeps_overlays_in_sync_with_the_program():
    tl, clip, _ = _with_manual_clips()
    start = clip.timeline_start_frame
    apply_intro(tl, "title_card", title="Hi")
    assert clip_ops.get_item(tl, clip.id).timeline_start_frame == start + 90


def test_reapplying_an_intro_does_not_stack_the_offset():
    tl = _timeline()
    apply_intro(tl, "title_card", title="One")
    apply_intro(tl, "title_card", title="Two")
    assert tl.program_offset_frames == 90


# --- effects survive a transcript rebuild ---------------------------------

def test_clip_effects_survive_a_word_toggle():
    tl = _timeline()
    v1 = [i for i in tl.items if i.track == "V1"]
    clip_ops.set_transform(tl, v1[0].id, {"scale": 1.4})
    clip_ops.set_color(tl, v1[0].id, {}, preset="moody")

    # Toggling a word rebuilds V1/A1 from scratch.
    word = next(w for w in tl.words if w.text == "friends")
    from backend.timeline.ops import toggle_word
    toggle_word(tl, word.id, False, list(tl.sources)[0])

    rebuilt = [i for i in tl.items if i.track == "V1"]
    first = min(rebuilt, key=lambda i: i.timeline_start_frame)
    assert first.transform is not None and first.transform.scale == 1.4
    assert first.color is not None and first.color.preset == "moody"


def test_master_grade_is_applied_after_the_overlays():
    tl, _clip, _ = _with_manual_clips()
    clip_ops.set_master_color(tl, {}, preset="vibrant")
    _inputs, fc, vlabel, _a = FilterGraphCompiler(tl).compile()
    assert vlabel == "[graded_v]"
    assert fc.index("overlay=") < fc.index("[graded_v]")


def test_text_is_burned_after_the_master_grade(tmp_path):
    tl = _timeline()
    clip_ops.set_master_color(tl, {}, preset="vibrant")
    clip_ops.add_text_item(tl, "On top", 0, 30)
    _inputs, fc, vlabel, _a = FilterGraphCompiler(tl, assets_dir=tmp_path).compile()
    assert vlabel == "[text_v]"
    assert fc.index("[graded_v]") < fc.index("drawtext=")


# --- captions that match the voice -----------------------------------------

def test_native_script_captions_use_the_spoken_words_and_a_shaping_font():
    """The romanizer writes "lie" for "लिए"; to a viewer that caption does not
    match the voice, where the native spelling matches it exactly. Native mode
    also swaps in a font that can shape Devanagari — the preset's Latin face
    would draw boxes."""
    words = [
        {"word": "dosto", "word_native": "दोस्तों", "start": 0.0, "end": 0.5},
        {"word": "aapke", "word_native": "आपके", "start": 0.5, "end": 1.0},
        {"word": "lie", "word_native": "लिए", "start": 1.0, "end": 1.5},
    ]
    tl = build_timeline_from_transcript("C:/media/main.mp4", 5.0, words,
                                        pause_padding_seconds=0.0)
    caps = generate_captions(tl, preset="youtube_shorts", script="native")
    assert caps
    text = " ".join(c.text.content for c in caps)
    assert "लिए" in text and "lie" not in text.lower()
    assert all(c.text.style.font_family == "Nirmala UI" for c in caps)


def test_romanized_captions_are_unchanged_by_default():
    words = [
        {"word": "dosto", "word_native": "दोस्तों", "start": 0.0, "end": 0.5},
        {"word": "aapke", "word_native": "आपके", "start": 0.5, "end": 1.0},
    ]
    tl = build_timeline_from_transcript("C:/media/main.mp4", 5.0, words,
                                        pause_padding_seconds=0.0)
    caps = generate_captions(tl, preset="youtube_shorts")
    assert "DOSTO" in caps[0].text.content


def test_caption_script_resolution():
    from backend.timeline.authoring import caption_script_for
    assert caption_script_for({"language": "hi"}) == "native"
    assert caption_script_for({"language": "en"}) == "romanized"
    assert caption_script_for({"language": "hi",
                               "caption_script": "romanized"}) == "romanized"
    assert caption_script_for({"language": "en",
                               "caption_script": "native"}) == "native"
    assert caption_script_for(None) == "romanized"


def test_a_caption_card_never_spans_a_cut():
    """A cut ripples its two sides together on the timeline, so by gap alone
    the words on either side land in ONE card — the tail of one sentence glued
    to the head of the next while the voice audibly jumps. Cards must break at
    every V1 join."""
    words = [
        {"word": "one", "start": 0.0, "end": 0.4},
        {"word": "two", "start": 0.4, "end": 0.8},
        # a removed stretch long enough to force a real cut
        {"word": "bad", "start": 0.8, "end": 2.8, "disfluency": True},
        {"word": "three", "start": 2.8, "end": 3.2},
        {"word": "four", "start": 3.2, "end": 3.6},
    ]
    tl = build_timeline_from_transcript("C:/media/main.mp4", 5.0, words,
                                        pause_padding_seconds=0.0)
    v1 = [i for i in tl.items if i.track == "V1"]
    assert len(v1) == 2, "the fixture needs a real cut"
    caps = generate_captions(tl, preset="classic")   # 8 words/card would merge
    contents = [c.text.content for c in caps]
    assert contents == ["one two", "three four"]
