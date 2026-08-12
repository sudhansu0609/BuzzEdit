from typing import Dict, List, Tuple
from ..timeline.schema import Timeline, frame_to_time

class FilterGraphCompiler:
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.fps_num = timeline.fps_num
        self.fps_den = timeline.fps_den
        self.fps = self.fps_num / self.fps_den

    def compile(self) -> Tuple[List[str], str, str, str]:
        """
        Compiles the EDL Timeline into FFmpeg command components:
        Returns:
            - inputs: List[str] input arguments (e.g. ["-i", "path1.mp4", "-i", "broll.jpg"])
            - filter_complex_str: The compiled FFmpeg filter_complex graph
            - video_map_label: Label of final output video stream (e.g. "[final_v]")
            - audio_map_label: Label of final output audio stream (e.g. "[final_a]")
        """
        # Collect source inputs and assign index
        input_args: List[str] = []
        source_index_map: Dict[str, int] = {}

        for src_id, src in self.timeline.sources.items():
            source_index_map[src_id] = len(input_args) // 2
            input_args.extend(["-i", src.path])

        v1_items = [item for item in self.timeline.items if item.enabled and item.track == "V1"]
        a1_items = [item for item in self.timeline.items if item.enabled and item.track == "A1"]
        v2_items = [item for item in self.timeline.items if item.enabled and item.track == "V2"]

        if not v1_items:
            raise ValueError("Timeline has no enabled V1 video items to render.")

        filters: List[str] = []

        # 1. Build V1 trims
        v1_labels: List[str] = []
        for idx, item in enumerate(v1_items):
            src_idx = source_index_map[item.source_id]
            start_sec = frame_to_time(item.source_start_frame, self.fps_num, self.fps_den)
            end_sec = frame_to_time(item.source_end_frame, self.fps_num, self.fps_den)
            out_label = f"[v1_{idx}]"
            filters.append(
                f"[{src_idx}:v]trim=start={start_sec:.3f}:end={end_sec:.3f},setpts=PTS-STARTPTS{out_label}"
            )
            v1_labels.append(out_label)

        # 2. Concat V1 clips
        if len(v1_labels) == 1:
            base_v_label = v1_labels[0]
        else:
            concat_inputs = "".join(v1_labels)
            base_v_label = "[base_v]"
            filters.append(
                f"{concat_inputs}concat=n={len(v1_labels)}:v=1:a=0{base_v_label}"
            )

        # 3. Build A1 trims
        a1_labels: List[str] = []
        for idx, item in enumerate(a1_items):
            src_idx = source_index_map[item.source_id]
            start_sec = frame_to_time(item.source_start_frame, self.fps_num, self.fps_den)
            end_sec = frame_to_time(item.source_end_frame, self.fps_num, self.fps_den)
            out_label = f"[a1_{idx}]"
            filters.append(
                f"[{src_idx}:a]atrim=start={start_sec:.3f}:end={end_sec:.3f},asetpts=PTS-STARTPTS{out_label}"
            )
            a1_labels.append(out_label)

        # 4. Concat A1 clips
        if not a1_labels:
            base_a_label = "[anull]"
            filters.append(f"anullsrc=r=44100:cl=stereo{base_a_label}")
        elif len(a1_labels) == 1:
            base_a_label = a1_labels[0]
        else:
            concat_inputs = "".join(a1_labels)
            base_a_label = "[base_a]"
            filters.append(
                f"{concat_inputs}concat=n={len(a1_labels)}:v=0:a=1{base_a_label}"
            )

        # 5. Apply V2 (B-roll overlays)
        current_v_label = base_v_label
        for idx, broll in enumerate(v2_items):
            src_idx = source_index_map[broll.source_id]
            tl_start_sec = frame_to_time(broll.timeline_start_frame, self.fps_num, self.fps_den)
            tl_end_sec = frame_to_time(broll.timeline_end_frame, self.fps_num, self.fps_den)
            out_label = f"[v2_ov_{idx}]"

            # Overlay filter with enable condition
            overlay_str = (
                f"{current_v_label}[{src_idx}:v]overlay=x=0:y=0:"
                f"enable='between(t,{tl_start_sec:.3f},{tl_end_sec:.3f})'{out_label}"
            )
            filters.append(overlay_str)
            current_v_label = out_label

        final_v_label = current_v_label
        final_a_label = base_a_label

        filter_complex_str = ";".join(filters)
        return input_args, filter_complex_str, final_v_label, final_a_label
