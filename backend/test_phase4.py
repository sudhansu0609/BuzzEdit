import os
import sys
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from video_pipeline.filmora_exporter import export_filmora_xml
from video_pipeline.filters import apply_color_grading, convert_to_vertical_shorts
from utils.ffmpeg_utils import run_ffmpeg, get_video_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_phase4")

def run_test():
    print("--- STARTING PHASE 4 ADVANCED FEATURES TEST ---")
    
    test_dir = backend_dir / "data" / "test_tmp_phase4"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_video = str(test_dir / "sample_input.mp4")
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=duration=4:size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-c:v", "libx264", "-c:a", "aac", "-y", test_video
    ])
    
    # 1. Test Filmora XML Export
    print("[1/3] Testing Filmora XML Export...")
    xml_output = str(test_dir / "test_timeline.xml")
    good_segments = [(0.5, 2.0), (2.5, 3.8)]
    
    export_filmora_xml(test_video, good_segments, xml_output, project_name="Test_Filmora_Export")
    assert Path(xml_output).exists(), "XML output file should exist"
    
    # Verify XML format validity
    tree = ET.parse(xml_output)
    root = tree.getroot()
    assert root.tag == "xmeml", f"Expected root xmeml, got {root.tag}"
    print(f"Filmora XML Export verified! File size: {os.path.getsize(xml_output)} bytes")
    
    # 2. Test Color Grading Filter
    print("[2/3] Testing Color Grading Filter...")
    graded_output = str(test_dir / "sample_graded.mp4")
    apply_color_grading(test_video, graded_output, preset="cinematic")
    assert Path(graded_output).exists(), "Graded output file should exist"
    print("Color grading filter verified!")
    
    # 3. Test Vertical 9:16 Shorts Cropping
    print("[3/3] Testing 9:16 Vertical Shorts Crop...")
    shorts_output = str(test_dir / "sample_shorts_9x16.mp4")
    convert_to_vertical_shorts(test_video, shorts_output)
    info = get_video_info(shorts_output)
    print(f"Shorts Output Video Info: {info['width']}x{info['height']}")
    assert info['width'] == 1080 and info['height'] == 1920, f"Expected 1080x1920, got {info['width']}x{info['height']}"
    
    print("[4/4] PHASE 4 VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    run_test()
