import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import TEMP_DIR
from render import transitions
from render.atmosphere import build_aspect_bars, build_effect
from timeline.schema import Timeline, TimelineItem, Transform, frame_to_time
from render.effects import (
    build_canvas_transform,
    build_chroma_key_chain,
    build_color_chain,
    build_fit_to_canvas,
    build_opacity_filters,
    build_overlay_transform,
)
from render.text import build_drawtext
from render.ass import build_ass, build_ass_filter, needs_ass, write_ass_asset
from render.audio import build_ducking, build_loudness_chain, build_voice_chain

# Below this many V1 segments the single shared decode is cheaper than opening
# an input per segment; above it the fan-out dominates (see `compile`).
_SEEK_INPUTS_MIN_SEGMENTS = 12
# An overlay clip whose source time runs this far AHEAD of its slot on the
# timeline is decoded from its own seeked input instead of the shared stream.
_OVERLAY_SEEK_LAG_SECONDS = 1.0

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# GPU (NVDEC) decoding: at most this many inputs, and only ones big enough for
# the CPU decode to matter (a 4K recording, not a 512px B-roll clip).
_HW_DECODERS_MAX = 6
_HW_DECODE_MIN_EDGE = 1280


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


def _track_num(track: str) -> int:
    digits = "".join(ch for ch in track if ch.isdigit())
    return int(digits) if digits else 0


def flatten_items(items: List[TimelineItem]) -> List[TimelineItem]:
    """Expand compound clips into their children with absolute timeline frames.

    Children are stored relative to their parent so the group slides as one unit;
    rendering needs them flat. A child is also clipped to the parent's window, so
    trimming a compound really does trim what plays, and it inherits the parent's
    transform/colour/mute unless it carries its own.
    """
    flat: List[TimelineItem] = []
    for item in items:
        if not item.enabled:
            continue
        if item.kind != "compound":
            flat.append(item)
            continue

        flat_for_this_group = False
        for child in item.children:
            if not child.enabled:
                continue
            abs_start = item.timeline_start_frame + child.timeline_start_frame
            abs_end = item.timeline_start_frame + child.timeline_end_frame
            # Drop children that the parent's trim pushed entirely out of view.
            if abs_end <= item.timeline_start_frame or abs_start >= item.timeline_end_frame:
                continue
            head = max(0, item.timeline_start_frame - abs_start)
            tail = max(0, abs_end - item.timeline_end_frame)

            resolved = child.model_copy(deep=True)
            resolved.timeline_start_frame = abs_start + head
            resolved.timeline_end_frame = abs_end - tail
            resolved.source_start_frame = child.source_start_frame + head
            resolved.source_end_frame = child.source_end_frame - tail
            if resolved.transform is None:
                resolved.transform = item.transform
            if resolved.color is None:
                resolved.color = item.color
            if item.mute:
                resolved.mute = True
            resolved.volume = resolved.volume * item.volume
            # A transition on the group belongs to the group's *leading edge* —
            # the one junction the timeline actually shows — and to no interior
            # cut, which is why this cannot be inherited like the grade above.
            # Without it a transition set on a compound was stored, drawn on the
            # lane, and then dropped here: the flattened children carried only
            # their own, and the parent stopped existing.
            if not flat_for_this_group and resolved.transition is None:
                resolved.transition = item.transition
            flat_for_this_group = True
            flat.append(resolved)
    return flat


class FilterGraphCompiler:
    def __init__(self, timeline: Timeline, assets_dir: Optional[Path] = None,
                 canvas: Optional[Tuple[int, int]] = None, hw_decode: bool = False):
        """`canvas` (w, h) composes the programme at that size instead of the
        primary source's: a 4K recording delivered at 1080p is composed at
        1080p, a quarter of the pixels in every overlay, zoom and grade.
        `hw_decode` decodes the larger video inputs on the GPU (NVDEC)."""
        self.timeline = timeline
        self.hw_decode = hw_decode
        self.fps_num = timeline.fps_num
        self.fps_den = timeline.fps_den
        self.fps = self.fps_num / self.fps_den
        # Text clips are rendered from sidecar files; keep them beside the other
        # scratch artefacts unless a caller (tests) points somewhere else.
        self.assets_dir = Path(assets_dir) if assets_dir else Path(TEMP_DIR) / "text"
        self.canvas_w, self.canvas_h = canvas or self._canvas_size()

    # -- helpers ---------------------------------------------------------

    def _canvas_size(self) -> Tuple[int, int]:
        """Program resolution: the primary V1 source, falling back to the timeline."""
        for item in self.timeline.items:
            if item.track == "V1" and item.source_id:
                src = self.timeline.sources.get(item.source_id)
                if src and src.width and src.height:
                    return int(src.width), int(src.height)
        return int(self.timeline.width or 1920), int(self.timeline.height or 1080)

    def _sec(self, frame: int) -> float:
        return frame_to_time(frame, self.fps_num, self.fps_den)

    def _track_hidden(self, track: str) -> bool:
        state = self.timeline.tracks.get(track)
        return bool(state and state.hidden)

    def _track_muted(self, track: str) -> bool:
        state = self.timeline.tracks.get(track)
        return bool(state and (state.muted or state.hidden))

    def _has_audio(self, item: TimelineItem) -> bool:
        src = self.timeline.sources.get(item.source_id or "")
        return bool(src and src.has_audio and src.kind != "image")

    def _is_still(self, item: TimelineItem) -> bool:
        src = self.timeline.sources.get(item.source_id or "")
        if src is None:
            return False
        return src.kind == "image" or _is_image(src.path)

    # -- compile ---------------------------------------------------------

    def compile(self, loudness_measured: Optional[Dict[str, float]] = None
               ) -> Tuple[List[str], str, str, str]:
        """
        Compiles the EDL Timeline into FFmpeg command components:
        Returns:
            - inputs: List[str] input arguments (e.g. ["-i", "path1.mp4", "-i", "broll.jpg"])
            - filter_complex_str: The compiled FFmpeg filter_complex graph
            - video_map_label: Label of final output video stream (e.g. "[final_v]")
            - audio_map_label: Label of final output audio stream (e.g. "[final_a]")

        `loudness_measured` is the first pass's loudnorm stats (see
        render.audio.measure_loudness_stats). None (the default) compiles the
        loudness stage as plain single-pass loudnorm, exactly as before —
        two-pass is something a caller opts into by measuring first.
        """
        input_args: List[str] = []
        source_index_map: Dict[str, int] = {}
        input_count = 0
        hw_budget = [_HW_DECODERS_MAX if self.hw_decode else 0]

        def hw_args(src) -> List[str]:
            """`-hwaccel cuda` for a large video input while the budget lasts.
            Each NVDEC session holds its own VRAM, so an edit with 150 seeked
            segment inputs must not open 150 of them; the rest decode on the CPU
            as before. Without an output format ffmpeg downloads the frames
            itself, so the graph is unchanged, and it falls back to software
            decoding if CUDA cannot start."""
            if hw_budget[0] <= 0 or src is None or src.kind == "image" or _is_image(src.path):
                return []
            if max(int(src.width or 0), int(src.height or 0)) < _HW_DECODE_MIN_EDGE:
                return []
            hw_budget[0] -= 1
            return ["-hwaccel", "cuda"]

        # The primary programme sources first, so they get the GPU decoders.
        ordered = sorted(self.timeline.sources.items(),
                         key=lambda kv: -max(int(kv[1].width or 0), int(kv[1].height or 0)))
        index_of = {src_id: n for n, (src_id, _) in enumerate(self.timeline.sources.items())}
        hw_for = {src_id: hw_args(src) for src_id, src in ordered}

        def thread_args(src, hw: List[str]) -> List[str]:
            """One decoder thread for every small input. Left to itself each
            input starts a pool sized to the machine (28 here) with its own
            frame buffers: a 230-input render ran 4,195 threads, for stills,
            short clips and sound effects that decode in microseconds."""
            if hw:
                return []
            big = max(int(src.width or 0), int(src.height or 0)) >= _HW_DECODE_MIN_EDGE
            is_video = src.kind != "image" and not _is_image(src.path) and src.kind != "audio"
            return [] if (big and is_video) else ["-threads", "1"]

        for src_id, src in self.timeline.sources.items():
            source_index_map[src_id] = index_of[src_id]
            input_args.extend([*hw_for[src_id], *thread_args(src, hw_for[src_id]), "-i", src.path])
            input_count += 1

        # Segment inputs. With the whole cut trimmed out of ONE decoded stream,
        # every decoded frame fans out to every segment's trim branch, and the
        # ones whose turn has not come yet hold their frames -- measured on a
        # 150-segment edit, the V1 chain ran at 0.28x realtime where its first
        # 10 segments alone ran at 8.6x: the cost grows with the number of cuts,
        # not the length. Past a handful of segments each one gets its own
        # input, seeked to its start (-ss before -i is frame-accurate: FFmpeg
        # decodes from the keyframe and drops up to the mark), so it decodes
        # only its own few seconds. -threads 1 keeps 150 decoders from each
        # allocating a full frame-thread pool.
        segment_inputs: Dict[Tuple[str, int, int], int] = {}

        def segment_input(item: TimelineItem) -> Optional[int]:
            nonlocal input_count
            if len(v1_items_all) < _SEEK_INPUTS_MIN_SEGMENTS:
                return None
            src = self.timeline.sources.get(item.source_id)
            if src is None or not src.path:
                return None
            key = (item.source_id, item.source_start_frame, item.source_end_frame)
            if key not in segment_inputs:
                start = self._sec(item.source_start_frame)
                span = self._sec(item.source_end_frame) - start
                input_args.extend([*hw_args(src), "-ss", f"{start:.3f}", "-t", f"{span + 0.1:.3f}",
                                   "-threads", "1", "-i", src.path])
                segment_inputs[key] = input_count
                input_count += 1
            return segment_inputs[key]

        # Overlay clips cut from a video source AHEAD of where they play. The
        # cold open is the one that matters: a hook from 39s into the programme
        # shown at t=0. Trimmed out of the same decoded stream as V1, that
        # overlay cannot emit its first frame until the decoder reaches 39s,
        # and every frame the V1 chain produced meanwhile (upscaled for the
        # zoompan, ~16 MB each) waits in the graph's queues: hundreds of MB per
        # second of lag, gigabytes within a minute, no output frame at all --
        # until Sentinel freezes and then kills ffmpeg (exit code 1, no message).
        # A seeked input decodes just the clip, so the graph flows from frame 0.
        overlay_seek_inputs: Dict[Tuple[str, int, int], int] = {}

        def overlay_seek_input(item: TimelineItem) -> Optional[int]:
            nonlocal input_count
            src = self.timeline.sources.get(item.source_id or "")
            if src is None or not src.path or self._is_still(item):
                return None
            lag = self._sec(item.source_start_frame) - self._sec(item.timeline_start_frame)
            # The mirror image of the cold-open case: a clip whose slot comes
            # LATER than its own time (every B-roll clip -- source time 0,
            # shown minutes in). On the shared stream its packets are stamped
            # from 0, ffmpeg reads the lowest timestamp first, so the whole
            # clip was decoded at once and held until its slot: a real 8-minute
            # render peaked at 14.7 GB. `-itsoffset` stamps the input with its
            # slot, and ffmpeg then reads it only when the programme gets there.
            if abs(lag) <= _OVERLAY_SEEK_LAG_SECONDS:
                return None
            key = (item.source_id, item.source_start_frame, item.source_end_frame,
                   item.timeline_start_frame)
            if key not in overlay_seek_inputs:
                start = self._sec(item.source_start_frame)
                span = self._sec(item.source_end_frame) - start
                offset = self._sec(item.timeline_start_frame)
                input_args.extend([*hw_args(src), "-itsoffset", f"{offset:.3f}", "-ss", f"{start:.3f}",
                                   "-t", f"{span + 0.1:.3f}", "-threads", "1", "-i", src.path])
                overlay_seek_inputs[key] = input_count
                input_count += 1
            return overlay_seek_inputs[key]

        items = flatten_items(self.timeline.items)

        media = [i for i in items if i.kind == "media"]
        text_items = sorted(
            [i for i in items if i.kind == "text" and i.text
             and not self._track_hidden(i.track)],
            key=lambda i: (_track_num(i.track), i.timeline_start_frame),
        )

        # V1 and A1 are concatenated in the order given, so they are put in time
        # order first: a hand-split clip used to append both halves to the end
        # of the list, and the render then played them after everything else.
        by_start = lambda i: i.timeline_start_frame
        v1_items = ([] if self._track_hidden("V1")
                    else sorted((i for i in media if i.track == "V1"), key=by_start))
        # An item whose source carries no audio stream cannot be trimmed with
        # [n:a] — ffmpeg rejects the whole graph with "matches no streams". The
        # transcript builder creates A1 items unconditionally, so silent footage
        # would otherwise make the project unrenderable.
        a1_items = ([] if self._track_muted("A1")
                    else sorted((i for i in media if i.track == "A1" and self._has_audio(i)),
                                key=by_start))
        # Manual overlay video tracks (V2, V3, ...) — composited on top of the V1 program,
        # ordered bottom-to-top by track number, then by timeline position.
        overlay_items = sorted(
            [i for i in media
             if i.track.upper().startswith("V") and i.track != "V1"
             and i.track.upper() != "CAP" and not self._track_hidden(i.track)],
            key=lambda i: (_track_num(i.track), i.timeline_start_frame),
        )
        # Adjustment layers — no picture of their own; they treat whatever is
        # underneath them. Composited in the same pass as the overlays so the
        # stacking order decides what each one reaches.
        adjustment_items = [
            i for i in items
            if i.kind == "adjustment" and i.track.upper().startswith("V")
            and not self._track_hidden(i.track)
        ]
        # Manual mix audio tracks (A2, A3, ...) — mixed with the A1 program audio.
        extra_audio_items = sorted(
            [i for i in media
             if i.track.upper().startswith("A") and i.track != "A1"
             and not i.mute and self._has_audio(i) and not self._track_muted(i.track)],
            key=lambda i: (_track_num(i.track), i.timeline_start_frame),
        )

        if not v1_items:
            raise ValueError("Timeline has no enabled V1 video items to render.")
        v1_items_all = v1_items

        filters: List[str] = []

        # 1. Build V1 trims
        # Concat demands identical geometry across its inputs, so the moment any
        # V1 clip carries a transform — or the cut mixes sources of different
        # sizes — every clip has to be normalised to the canvas.
        v1_dims = {
            (self.timeline.sources[i.source_id].width, self.timeline.sources[i.source_id].height)
            for i in v1_items if i.source_id and i.source_id in self.timeline.sources
        }
        pad_frames = max(0, self.timeline.program_offset_frames)
        pad_sec = self._sec(pad_frames)
        normalize_v1 = len(v1_dims) > 1 or pad_frames > 0 or any(
            i.transform and not i.transform.is_identity() for i in v1_items
        ) or any(  # composed at a size other than the footage's (a 4K take on a 1080p canvas)
            (int(w or 0), int(h or 0)) != (self.canvas_w, self.canvas_h) for w, h in v1_dims
        )

        v1_labels: List[str] = []
        for idx, item in enumerate(v1_items):
            src_idx = source_index_map[item.source_id]
            start_sec = self._sec(item.source_start_frame)
            end_sec = self._sec(item.source_end_frame)
            out_label = f"[v1_{idx}]"
            seg_idx = segment_input(item)
            if seg_idx is not None:
                src_idx = seg_idx
                chain = [f"trim=duration={end_sec - start_sec:.3f}", "setpts=PTS-STARTPTS"]
            else:
                chain = [f"trim=start={start_sec:.3f}:end={end_sec:.3f}", "setpts=PTS-STARTPTS"]
            geometry = self._geometry_for(item, normalize_v1)
            # Every V1 clip is either concatenated or cross-faded with its
            # neighbours, and both demand identical frame rate, SAR and pixel
            # format — otherwise `xfade` aborts ("Failed to configure output pad").
            # Frame rate is the subtle one. A segment zoom is a `zoompan`, whose
            # `:fps=` only *labels* the output rate — it does NOT resample. Fed
            # 25fps footage on a 30fps timeline it emits one frame per input frame
            # and calls it 30fps, so a 2.0s segment becomes 1.667s of picture while
            # its audio stays 2.0s: the video slides ahead of the cut and every
            # zoomed segment ends in the wrong place. It also throws off the zoom
            # itself, whose `on/span` progress assumes timeline-rate frames. So the
            # `fps` conversion has to run BEFORE the zoompan, normalising the input
            # rate; a plain clip can take it anywhere, so it goes here too.
            chain.append(f"fps={self.fps:.5f}")
            chain.extend(geometry)
            chain.extend(build_color_chain(item.color, end_sec - start_sec))
            chain.append("setsar=1")
            chain.append("format=yuv420p")
            filters.append(f"[{src_idx}:v]" + ",".join(chain) + out_label)
            v1_labels.append(out_label)

        # 1b. Intro card: a generated colour/silence segment concatenated in
        #     front of the program. Built from lavfi source filters rather than
        #     extra -i inputs so the input index map stays untouched.
        if pad_frames > 0:
            filters.append(
                f"color=c={self.timeline.program_offset_color}:"
                f"s={self.canvas_w}x{self.canvas_h}:r={self.fps:.5f}:d={pad_sec:.3f},"
                f"format=yuv420p,setsar=1[pad_v]"
            )
            v1_labels.insert(0, "[pad_v]")

        # 2. Join the V1 clips. Hard cuts concatenate; anything with a transition
        #    has to overlap instead, which is what xfade does.
        durations = [self._sec(i.duration_frames) for i in v1_items]
        if pad_frames > 0:
            durations.insert(0, pad_sec)
        junctions = self._junction_transitions(v1_items, pad_frames > 0)
        base_v_label, program_seconds = self._join_video(
            v1_labels, durations, junctions, filters)

        # 3. Build A1 trims
        a1_labels: List[str] = []
        for idx, item in enumerate(a1_items):
            if item.mute:
                continue
            src_idx = source_index_map[item.source_id]
            start_sec = self._sec(item.source_start_frame)
            end_sec = self._sec(item.source_end_frame)
            out_label = f"[a1_{idx}]"
            # Same fan-out as V1; an A1 item usually spans exactly its V1
            # partner's range, so it shares that segment's seeked input.
            seg_idx = segment_input(item)
            if seg_idx is not None:
                src_idx = seg_idx
                chain = [f"atrim=duration={end_sec - start_sec:.3f}", "asetpts=PTS-STARTPTS"]
            else:
                chain = [
                    f"atrim=start={start_sec:.3f}:end={end_sec:.3f}",
                    "asetpts=PTS-STARTPTS",
                ]
            if item.volume != 1.0:
                chain.append(f"volume={item.volume:.3f}")
            # Declick every join: a hard concat at an arbitrary sample almost
            # never lands on a zero crossing, and the discontinuity is an audible
            # tick at every cut. 8ms edge fades are far below the threshold of
            # sounding like a fade but remove the click entirely.
            segment_sec = end_sec - start_sec
            if segment_sec > 0.05:
                chain.append("afade=t=in:st=0:d=0.008:curve=tri")
                chain.append(f"afade=t=out:st={segment_sec - 0.008:.3f}:d=0.008:curve=tri")
            filters.append(f"[{src_idx}:a]" + ",".join(chain) + out_label)
            a1_labels.append(out_label)

        if pad_frames > 0 and a1_labels:
            filters.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={pad_sec:.3f},"
                f"asetpts=PTS-STARTPTS[pad_a]"
            )
            a1_labels.insert(0, "[pad_a]")

        # 4. Join the A1 clips with a plain concat. Video transitions no longer
        #    shorten the programme (each junction is padded with cloned frames
        #    before the xfade), so the audio must NOT crossfade: an acrossfade
        #    both shortened it out of step with the picture and smeared 0.12s of
        #    neighbouring speech over every join. The declick is already handled
        #    by each segment's 8ms edge fades.
        if not a1_labels:
            base_a_label = "[anull]"
            filters.append(f"anullsrc=r=48000:cl=stereo{base_a_label}")
        else:
            audio_durations = [self._sec(i.duration_frames) for i in a1_items if not i.mute]
            if pad_frames > 0:
                audio_durations.insert(0, pad_sec)
            base_a_label = self._join_audio(a1_labels, audio_durations, {}, filters)

        # 4b. Program-wide geometry (applied before overlays so B-roll is not
        #     dragged along by a master reframe).
        current_v_label = base_v_label
        master_t = self.timeline.master_transform
        if master_t and not master_t.is_identity():
            chain = build_canvas_transform(
                master_t, self.canvas_w, self.canvas_h,
                max(1, self.timeline.duration_frames), self.fps,
            )
            out_label = "[master_v]"
            filters.append(f"{current_v_label}" + ",".join(chain) + out_label)
            current_v_label = out_label

        # The programme's real length *after* transitions have overlapped it —
        # timeline.duration_frames does not account for that shortening. A little
        # headroom keeps a generated layer from running out early.
        total_seconds = max(1.0, program_seconds) + 1.0

        # 5. Composite overlay video tracks (B-roll + manual clips) onto the V1
        #    program, with adjustment layers taking effect at their own height in
        #    the stack: an adjustment on V3 treats V1 and V2 but not V4.
        stack = sorted(
            overlay_items + adjustment_items,
            key=lambda i: (_track_num(i.track), i.timeline_start_frame),
        )
        for idx, item in enumerate(stack):
            if item.kind == "adjustment":
                current_v_label = self._apply_adjustment(
                    item, idx, current_v_label, filters, total_seconds)
                continue

            overlay_input, x_expr, y_expr = self._build_overlay_input(
                item, idx, source_index_map, filters, overlay_seek_input
            )
            tl_start_sec = self._sec(item.timeline_start_frame)
            tl_end_sec = self._sec(item.timeline_end_frame)
            out_label = f"[vov_{idx}]"
            # x/y are quoted: an animated pan expression contains commas, which
            # the filtergraph parser would read as a filter separator.
            filters.append(
                f"{current_v_label}{overlay_input}overlay=x='{x_expr}':y='{y_expr}':"
                f"enable='between(t,{tl_start_sec:.3f},{tl_end_sec:.3f})'{out_label}"
            )
            current_v_label = out_label

        # 5b. Voice treatment on the programme audio, before anything is mixed
        #     in: a denoiser or compressor running over the music would pump
        #     with the speech, and a de-esser has no business touching a cymbal.
        master_a = self.timeline.audio_master
        voice_chain = build_voice_chain(master_a)
        if voice_chain and base_a_label != "[anull]":
            out_label = "[voice_a]"
            filters.append(f"{base_a_label}" + ",".join(voice_chain) + out_label)
            base_a_label = out_label

        # 6. Mix extra audio tracks (A2, A3, ...) with the A1 program audio.
        #    Items marked `duck` are pushed down under speech by a sidechain
        #    compressor keyed off the programme voice, so the music follows the
        #    speech exactly; the voice is split once per ducked item because a
        #    filter output can feed only one consumer.
        final_a_label = base_a_label
        if extra_audio_items:
            ducked = [i for i in extra_audio_items if i.duck > 0.0]
            sidechain_labels: List[str] = []
            if ducked and base_a_label != "[anull]":
                main_label = "[a1_main]"
                sidechain_labels = [f"[a1_sc_{n}]" for n in range(len(ducked))]
                filters.append(
                    f"{base_a_label}asplit={len(ducked) + 1}{main_label}"
                    f"{''.join(sidechain_labels)}")
                base_a_label = main_label
            mix_inputs = [base_a_label]
            sidechain_index = 0
            for idx, item in enumerate(extra_audio_items):
                src_idx = source_index_map[item.source_id]
                src = self.timeline.sources[item.source_id]
                src_start_sec = self._sec(item.source_start_frame)
                src_end_sec = self._sec(item.source_end_frame)
                tl_start_sec = self._sec(item.timeline_start_frame)
                span_sec = self._sec(item.duration_frames)
                delay_ms = int(tl_start_sec * 1000)
                aout = f"[amix_{idx}]"
                chain: List[str] = []
                if item.loop and src.duration_seconds > 0:
                    # Repeat the whole file enough times to cover the span, then
                    # trim; aloop counts samples, and the sample rate of the
                    # source is unknown here, so it is resampled first.
                    loops = int(span_sec // src.duration_seconds) + 1
                    samples = int(round(src.duration_seconds * 48000))
                    chain.append("aresample=48000")
                    chain.append(f"aloop=loop={loops}:size={max(1, samples)}")
                    chain.append(f"atrim=start={src_start_sec:.3f}:end={src_start_sec + span_sec:.3f}")
                else:
                    chain.append(f"atrim=start={src_start_sec:.3f}:end={src_end_sec:.3f}")
                chain.append("asetpts=PTS-STARTPTS")
                if item.volume != 1.0:
                    chain.append(f"volume={item.volume:.3f}")
                if item.audio_fade_in > 0:
                    chain.append(f"afade=t=in:st=0:d={min(item.audio_fade_in, span_sec):.3f}")
                if item.audio_fade_out > 0:
                    fade_out = min(item.audio_fade_out, span_sec)
                    chain.append(
                        f"afade=t=out:st={max(0.0, span_sec - fade_out):.3f}:d={fade_out:.3f}")
                chain.append(f"adelay={delay_ms}:all=1")
                if item.duck > 0.0 and sidechain_labels:
                    pre_label = f"[aduck_in_{idx}]"
                    filters.append(f"[{src_idx}:a]" + ",".join(chain) + pre_label)
                    filters.append(
                        f"{pre_label}{sidechain_labels[sidechain_index]}"
                        + build_ducking(item.duck) + aout)
                    sidechain_index += 1
                else:
                    filters.append(f"[{src_idx}:a]" + ",".join(chain) + aout)
                mix_inputs.append(aout)
            mixed_label = "[mix_a]"
            filters.append(
                f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0:"
                f"dropout_transition=0{mixed_label}"
            )
            final_a_label = mixed_label

        # 6b. Loudness target on the finished mix — what the platform measures.
        loudness_chain = build_loudness_chain(master_a, measured=loudness_measured)
        if loudness_chain and final_a_label != "[anull]":
            out_label = "[master_a]"
            filters.append(f"{final_a_label}" + ",".join(loudness_chain) + out_label)
            final_a_label = out_label

        # 7. Program-wide colour, after the overlays so B-roll grades with the
        #    program, but before text so captions stay the colour they were set.
        master_c = self.timeline.master_color
        if master_c and not master_c.is_identity():
            total_sec = self._sec(self.timeline.duration_frames)
            chain = build_color_chain(master_c, total_sec)
            if chain:
                out_label = "[graded_v]"
                filters.append(f"{current_v_label}" + ",".join(chain) + out_label)
                current_v_label = out_label

        # 7b. Generated atmosphere — rain, light leaks, grain. After the grade so
        #     it is not itself graded, before the text so captions stay readable
        #     through it.
        for index, effect in enumerate(self.timeline.effects):
            if not effect.enabled:
                continue
            out_label = f"[atmo_{index}]"
            statements = build_effect(
                effect, current_v_label, out_label,
                self.canvas_w, self.canvas_h, self.fps, total_seconds, index,
            )
            if statements:
                filters.extend(statements)
                current_v_label = out_label

        # 7c. Cinematic letterbox, matted on last so nothing draws over the bars.
        if self.timeline.aspect_bars:
            out_label = "[bars_v]"
            bars = build_aspect_bars(
                float(self.timeline.aspect_bars), self.canvas_w, self.canvas_h,
                current_v_label, out_label,
            )
            if bars:
                filters.append(bars)
                current_v_label = out_label

        # 8. Burn text and captions onto the finished picture. Animated clips
        #    (karaoke, typewriter, scale-in, glitch, …) go through libass in one
        #    `ass` filter; the plain ones stay on drawtext.
        animated = [i for i in text_items if needs_ass(i.text)]
        if animated:
            script = build_ass(
                [(i.text, self._sec(i.timeline_start_frame), self._sec(i.timeline_end_frame))
                 for i in animated],
                self.canvas_w, self.canvas_h,
            )
            ass_path = write_ass_asset(script, self.assets_dir)
            out_label = "[ass_v]"
            filters.append(f"{current_v_label}{build_ass_filter(ass_path)}{out_label}")
            current_v_label = out_label
        text_filters: List[str] = []
        for item in text_items:
            if needs_ass(item.text):
                continue
            built = build_drawtext(
                item.text,
                self._sec(item.timeline_start_frame),
                self._sec(item.timeline_end_frame),
                self.assets_dir,
            )
            if built:
                text_filters.append(built)
        if text_filters:
            out_label = "[text_v]"
            filters.append(f"{current_v_label}" + ",".join(text_filters) + out_label)
            current_v_label = out_label

        final_v_label = current_v_label

        filter_complex_str = ";".join(filters)
        return input_args, filter_complex_str, final_v_label, final_a_label

    # -- per-item chains -------------------------------------------------

    def _junction_transitions(self, v1_items: List[TimelineItem],
                              has_pad: bool) -> Dict[int, Any]:
        """Transition for each junction, keyed by the index of the clip it leads into.

        A clip's own transition wins; otherwise the programme default applies. The
        first clip has nothing before it, so it never takes one.
        """
        default = self.timeline.default_transition
        junctions: Dict[int, Any] = {}
        offset = 1 if has_pad else 0
        for index, item in enumerate(v1_items):
            position = index + offset
            if position == 0:
                continue                      # nothing precedes the first clip
            chosen = item.transition or default
            if chosen and chosen.duration > 0 and transitions.is_valid(chosen.type):
                junctions[position] = chosen
        return junctions

    def _join_video(self, labels: List[str], durations: List[float],
                    junctions: Dict[int, Any], filters: List[str]) -> Tuple[str, float]:
        """Concatenate runs of hard cuts, xfade across the rest.

        Returns (label, programme_seconds). Transitions are NON-shortening: a
        plain xfade consumes its overlap from the clips and shortens the
        programme by its own duration — and since the audio side concatenates at
        full length, every dissolve pushed picture 0.12s ahead of sound,
        cumulatively. Instead, each side of the junction is extended by half the
        transition (`tpad` cloning the edge frame) and the xfade runs over the
        cloned material, so the programme keeps exactly the length the audio
        has. At the default 0.12s that is two frozen frames per side — invisible
        — and A/V sync holds by construction, whatever the junction count.
        """
        if len(labels) == 1:
            return labels[0], durations[0] if durations else 0.0

        # Split into runs that can be concatenated in one go.
        groups: List[List[int]] = [[0]]
        for index in range(1, len(labels)):
            if index in junctions:
                groups.append([index])
            else:
                groups[-1].append(index)

        group_labels: List[str] = []
        group_durations: List[float] = []
        for number, group in enumerate(groups):
            members = [labels[i] for i in group]
            total = sum(durations[i] for i in group)
            if len(members) == 1:
                group_labels.append(members[0])
            else:
                out = f"[vgrp_{number}]"
                filters.append(f"{''.join(members)}concat=n={len(members)}:v=1:a=0{out}")
                group_labels.append(out)
            group_durations.append(total)

        if len(group_labels) == 1:
            return group_labels[0], group_durations[0]

        current = group_labels[0]
        elapsed = group_durations[0]
        for number in range(1, len(group_labels)):
            transition = junctions[groups[number][0]]
            duration = transitions.clamp_duration(
                transition.duration, elapsed, group_durations[number])
            out = f"[vx_{number}]"
            if duration <= transitions.MIN_DURATION:
                filters.append(f"{current}{group_labels[number]}concat=n=2:v=1:a=0{out}")
            else:
                half = duration / 2.0
                left = f"[vxa_{number}]"
                right = f"[vxb_{number}]"
                filters.append(
                    f"{current}tpad=stop_mode=clone:stop_duration={half:.3f}{left}")
                filters.append(
                    f"{group_labels[number]}tpad=start_mode=clone:"
                    f"start_duration={half:.3f}{right}")
                filters.append(transitions.build_video(
                    left, right, out, transition.type, duration, elapsed + half))
            elapsed += group_durations[number]
            current = out
        return current, elapsed

    def _join_audio(self, labels: List[str], durations: List[float],
                    junctions: Dict[int, Any], filters: List[str]) -> str:
        """Mirror of _join_video for sound: crossfade at the same junctions."""
        if len(labels) == 1:
            return labels[0]

        groups: List[List[int]] = [[0]]
        for index in range(1, len(labels)):
            if index in junctions:
                groups.append([index])
            else:
                groups[-1].append(index)

        group_labels: List[str] = []
        group_durations: List[float] = []
        for number, group in enumerate(groups):
            members = [labels[i] for i in group]
            total = sum(durations[i] for i in group) if durations else 0.0
            if len(members) == 1:
                group_labels.append(members[0])
            else:
                out = f"[agrp_{number}]"
                filters.append(f"{''.join(members)}concat=n={len(members)}:v=0:a=1{out}")
                group_labels.append(out)
            group_durations.append(total)

        if len(group_labels) == 1:
            return group_labels[0]

        current = group_labels[0]
        elapsed = group_durations[0]
        for number in range(1, len(group_labels)):
            transition = junctions[groups[number][0]]
            duration = transitions.clamp_duration(
                transition.duration, elapsed, group_durations[number])
            out = f"[ax_{number}]"
            if duration <= transitions.MIN_DURATION:
                filters.append(f"{current}{group_labels[number]}concat=n=2:v=0:a=1{out}")
                elapsed += group_durations[number]
            else:
                filters.append(transitions.build_audio(
                    current, group_labels[number], out, duration))
                elapsed += group_durations[number] - duration
            current = out
        return current

    def _geometry_for(self, item: TimelineItem, normalize: bool) -> List[str]:
        """Geometry filters for a V1 clip: its transform, or a plain canvas fit."""
        if item.transform and not item.transform.is_identity():
            return build_canvas_transform(
                item.transform, self.canvas_w, self.canvas_h,
                max(1, item.duration_frames), self.fps,
            )
        if normalize:
            return build_fit_to_canvas(self.canvas_w, self.canvas_h)
        return []

    def _apply_adjustment(self, item: TimelineItem, index: int, input_label: str,
                          filters: List[str], total_seconds: float) -> str:
        """Apply an adjustment clip to everything composited so far.

        The picture is split in two: one copy goes through the clip's transform,
        grade and atmosphere, and is then overlaid back onto the untouched copy
        with an `enable` window. Windowing the treatment this way rather than
        hanging `enable=` off each filter keeps one mechanism for all three — not
        every filter supports the timeline feature, and `zoompan` and the blends
        behind the atmosphere effects certainly do not.

        Returns the label to carry on with — the input label unchanged when the
        clip adjusts nothing.
        """
        transform = item.transform
        start_frame = item.timeline_start_frame
        chain: List[str] = []

        # The work copy is trimmed to this clip's own window before anything
        # runs on it. Untrimmed, every adjustment processed the WHOLE programme
        # -- a 0.8s punch-in ran zoompan at 5760x3240 over all ~10k frames and
        # overlay's `enable` threw 99% of it away; 27 of them made the final
        # render crawl at <0.1x realtime. zoompan restarts `on` (and its PTS)
        # at the window, hence frame_offset=0 and the re-anchor below, which
        # puts timestamps back on the programme clock the effects expect.
        win_start = self._sec(start_frame)
        win_end = self._sec(item.timeline_end_frame)
        if transform is not None and not self._is_opacity_only(transform):
            chain.extend(build_canvas_transform(
                transform, self.canvas_w, self.canvas_h,
                max(1, item.duration_frames), self.fps, frame_offset=0,
            ))
        chain.extend(build_color_chain(item.color, self._sec(item.duration_frames)))
        effects = [e for e in item.atmosphere if e.enabled]
        fade = build_opacity_filters(transform)

        if not chain and not effects:
            return input_label

        base_label = f"[adjbase_{index}]"
        work_label = f"[adjsrc_{index}]"
        filters.append(f"{input_label}split{base_label}{work_label}")

        out_label = f"[adjfx_{index}]"
        filters.append(
            f"{work_label}trim=start={win_start:.3f}:end={win_end:.3f},"
            + "".join(f"{c}," for c in chain)
            + f"setpts=PTS-STARTPTS+{win_start:.3f}/TB{out_label}"
        )
        work_label = out_label

        for order, effect in enumerate(effects):
            out_label = f"[adjatmo_{index}_{order}]"
            statements = build_effect(
                effect, work_label, out_label, self.canvas_w, self.canvas_h,
                self.fps, total_seconds, f"{index}_{order}",
                assets_dir=self.assets_dir,
            )
            if statements:
                filters.extend(statements)
                work_label = out_label

        if fade:
            # Partial strength: the treated copy is composited back at alpha, so
            # the untouched picture shows through it.
            out_label = f"[adjmix_{index}]"
            filters.append(f"{work_label}" + ",".join(fade) + out_label)
            work_label = out_label

        if start_frame > 0:
            # The trimmed copy has no frames before its window, and overlay's
            # framesync will not let a base frame through until the treated
            # copy has produced its first one. With the first window at 109s
            # that held every base frame before it in RAM (~8 GB at 1080p),
            # FFmpeg never wrote a frame, and the memory guardian froze and
            # then killed it (exit code 1, nothing on stderr). Padding the copy
            # with black from t=0 keeps the two streams in step; the overlay is
            # disabled outside the window, so the pad is never composited.
            # Re-anchoring to 0 first makes tpad land the real frames exactly
            # at win_start (start_frame pad frames at the timeline rate).
            out_label = f"[adjpad_{index}]"
            filters.append(
                f"{work_label}setpts=PTS-STARTPTS,"
                f"tpad=start_mode=add:start_duration={win_start:.3f}:color=black{out_label}"
            )
            work_label = out_label

        out_label = f"[adj_{index}]"
        # eof_action=pass: the trimmed copy ends with its window, and the rest
        # of the programme must flow through untouched rather than stall on it.
        filters.append(
            f"{base_label}{work_label}overlay=x=0:y=0:eof_action=pass:"
            f"enable='between(t,{win_start:.3f},{win_end:.3f})'{out_label}"
        )
        return out_label

    @staticmethod
    def _is_opacity_only(transform: Transform) -> bool:
        """True when the only thing set is opacity — nothing to do geometrically."""
        return transform.model_copy(update={"opacity": 1.0}).is_identity()

    def _build_overlay_input(
        self,
        item: TimelineItem,
        idx: int,
        source_index_map: Dict[str, int],
        filters: List[str],
        seek_input: Optional[Callable[[TimelineItem], Optional[int]]] = None,
    ) -> Tuple[str, str, str]:
        """Prepare an overlay clip's stream and return (label, x_expr, y_expr).

        `seek_input` may hand a video clip its own seeked input index (see
        `overlay_seek_input` in `compile`); None keeps the shared stream.
        """
        src_idx = source_index_map[item.source_id]
        tl_start_sec = self._sec(item.timeline_start_frame)
        tl_end_sec = self._sec(item.timeline_end_frame)
        duration_frames = max(1, item.duration_frames)

        if self._is_still(item):
            # A still decodes to a single frame, which overlay simply holds for the
            # whole `enable` window — so a static still needs no trim and no PTS
            # shift at all.
            chain, x_expr, y_expr = build_overlay_transform(
                item.transform, self.canvas_w, self.canvas_h, duration_frames,
                self.fps, tl_start_sec, tl_end_sec,
            )
            animated_zoom = any(f.startswith("zoompan=") for f in chain)
            if animated_zoom:
                # A Ken Burns move needs real frames to move across, so tell
                # zoompan to expand the single input frame into the clip's span.
                chain = self._retime_still(chain, duration_frames)
            # Key before grading: the grade moves the very colour the key hunts for.
            chain.extend(build_chroma_key_chain(item.chroma))
            chain.extend(build_color_chain(item.color, tl_end_sec - tl_start_sec))
            if animated_zoom:
                # Those generated frames start at t=0, so without this shift the
                # move would play out before the clip is ever on screen and only
                # its final frame would be visible.
                chain.append(f"setpts=PTS-STARTPTS+{tl_start_sec:.3f}/TB")
            label = f"[vclip_{idx}]"
            filters.append(f"[{src_idx}:v]" + ",".join(chain) + label)
            return label, x_expr, y_expr

        src_start_sec = self._sec(item.source_start_frame)
        src_end_sec = self._sec(item.source_end_frame)
        seek_idx = seek_input(item) if seek_input is not None else None
        if seek_idx is not None:
            src_idx = seek_idx
            chain = [
                f"trim=duration={src_end_sec - src_start_sec:.3f}",
                "setpts=PTS-STARTPTS",
            ]
        else:
            chain = [
                f"trim=start={src_start_sec:.3f}:end={src_end_sec:.3f}",
                "setpts=PTS-STARTPTS",
            ]
        chain_geo, x_expr, y_expr = build_overlay_transform(
            item.transform, self.canvas_w, self.canvas_h, duration_frames,
            self.fps, tl_start_sec, tl_end_sec,
        )
        chain.extend(chain_geo)
        chain.extend(build_chroma_key_chain(item.chroma))
        chain.extend(build_color_chain(item.color, src_end_sec - src_start_sec))
        # Shift into place last, so every filter above saw clip-local time.
        chain.append(f"setpts=PTS-STARTPTS+{tl_start_sec:.3f}/TB")
        label = f"[vclip_{idx}]"
        filters.append(f"[{src_idx}:v]" + ",".join(chain) + label)
        return label, x_expr, y_expr

    @staticmethod
    def _retime_still(chain: List[str], duration_frames: int) -> List[str]:
        """Make a still's zoompan emit `duration_frames` frames instead of one."""
        return [f.replace(":d=1:", f":d={duration_frames}:") if f.startswith("zoompan=") else f
                for f in chain]
