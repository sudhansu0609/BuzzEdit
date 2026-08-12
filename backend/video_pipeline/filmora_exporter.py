import logging
from pathlib import Path
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

logger = logging.getLogger(__name__)


def export_filmora_xml(
    source_video_path: str,
    good_segments: List[Tuple[float, float]],
    output_xml_path: str,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    project_name: str = "Buzzcaf_Edit",
) -> str:
    Path(output_xml_path).parent.mkdir(parents=True, exist_ok=True)
    timebase = str(fps)

    x_x = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(x_x, "sequence", id="Sequence-1")
    ET.SubElement(sequence, "name").text = project_name
    ET.SubElement(sequence, "duration").text = str(int(sum(end - start for start, end in good_segments) * fps))

    rate = ET.SubElement(sequence, "rate")
    ET.SubElement(rate, "timebase").text = timebase
    ET.SubElement(rate, "ntsc").text = "FALSE"

    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    
    # Format settings
    format_elem = ET.SubElement(video, "format")
    samplecharacteristics = ET.SubElement(format_elem, "samplecharacteristics")
    ET.SubElement(samplecharacteristics, "width").text = str(width)
    ET.SubElement(samplecharacteristics, "height").text = str(height)

    v_track = ET.SubElement(video, "track")
    
    audio = ET.SubElement(media, "audio")
    a_track1 = ET.SubElement(audio, "track")

    curr_timeline_frame = 0

    for idx, (start, end) in enumerate(good_segments, 1):
        duration_sec = end - start
        if duration_sec < 0.1:
            continue

        in_frame = int(start * fps)
        out_frame = int(end * fps)
        duration_frames = out_frame - in_frame
        timeline_start = curr_timeline_frame
        timeline_end = curr_timeline_frame + duration_frames

        # Video clip item
        v_item = ET.SubElement(v_track, "clipitem", id=f"clipitem-v-{idx}")
        ET.SubElement(v_item, "name").text = Path(source_video_path).name
        ET.SubElement(v_item, "duration").text = str(duration_frames)

        v_rate = ET.SubElement(v_item, "rate")
        ET.SubElement(v_rate, "timebase").text = timebase
        ET.SubElement(v_rate, "ntsc").text = "FALSE"

        ET.SubElement(v_item, "start").text = str(timeline_start)
        ET.SubElement(v_item, "end").text = str(timeline_end)
        ET.SubElement(v_item, "in").text = str(in_frame)
        ET.SubElement(v_item, "out").text = str(out_frame)

        # File reference
        file_elem = ET.SubElement(v_item, "file", id=f"file-1")
        ET.SubElement(file_elem, "name").text = Path(source_video_path).name
        ET.SubElement(file_elem, "pathurl").text = Path(source_video_path).as_uri()

        # Audio clip item
        a_item = ET.SubElement(a_track1, "clipitem", id=f"clipitem-a-{idx}")
        ET.SubElement(a_item, "name").text = Path(source_video_path).name
        ET.SubElement(a_item, "duration").text = str(duration_frames)

        a_rate = ET.SubElement(a_item, "rate")
        ET.SubElement(a_rate, "timebase").text = timebase
        ET.SubElement(a_rate, "ntsc").text = "FALSE"

        ET.SubElement(a_item, "start").text = str(timeline_start)
        ET.SubElement(a_item, "end").text = str(timeline_end)
        ET.SubElement(a_item, "in").text = str(in_frame)
        ET.SubElement(a_item, "out").text = str(out_frame)
        ET.SubElement(a_item, "file", id=f"file-1")

        curr_timeline_frame += duration_frames

    # Format pretty XML string
    xml_str = minidom.parseString(ET.tostring(x_x)).toprettyxml(indent="  ")
    Path(output_xml_path).write_text(xml_str, encoding="utf-8")
    
    logger.info(f"Filmora XML timeline exported successfully: {output_xml_path}")
    return output_xml_path
