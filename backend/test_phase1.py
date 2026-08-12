import os
import sys
import logging
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from utils.ffmpeg_utils import run_ffmpeg, get_video_info, get_video_duration, extract_audio, cut_clip, concat_with_transitions
from audio_pipeline.silence_detector import detect_silence_segments
from audio_pipeline.fumble_detector import detect_fumbles
from models import TranscriptSegment, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_phase1")

def run_test():
    print("--- STARTING PHASE 1 VERIFICATION TEST ---")
    
    test_dir = backend_dir / "data" / "test_tmp"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create synthetic 6-second video with audio tone using FFmpeg
    test_video = str(test_dir / "synthetic_test.mp4")
    print(f"[1/5] Generating synthetic test video: {test_video}")
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=duration=6:size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-c:a", "aac", "-y", test_video
    ])
    
    # Check video info
    info = get_video_info(test_video)
    print(f"Video Info: duration={info['duration']}s, res={info['width']}x{info['height']}, fps={info['fps']}")
    assert info['duration'] >= 5.9, "Synthetic video duration check failed"
    
    # 2. Extract Audio
    audio_path = extract_audio(test_video)
    assert Path(audio_path).exists(), "Audio extraction failed"
    print(f"[2/5] Extracted audio successfully: {audio_path}")
    
    # 3. Test silence & fumble detection logic with mock segments
    mock_segments = [
        TranscriptSegment(
            id=0, start=0.5, end=2.0, text="Hello and welcome um to our video today.",
            words=[
                {"word": "Hello", "start": 0.5, "end": 0.8},
                {"word": "and", "start": 0.8, "end": 1.0},
                {"word": "welcome", "start": 1.0, "end": 1.3},
                {"word": "um", "start": 1.4, "end": 1.6},
                {"word": "to", "start": 1.6, "end": 1.7},
                {"word": "our", "start": 1.7, "end": 1.8},
                {"word": "video", "start": 1.8, "end": 2.0},
            ]
        ),
        TranscriptSegment(
            id=1, start=4.0, end=5.5, text="Today we are building something awesome.",
            words=[
                {"word": "Today", "start": 4.0, "end": 4.3},
                {"word": "we", "start": 4.3, "end": 4.5},
            ]
        )
    ]
    
    silences = detect_silence_segments(test_video, mock_segments, min_duration=1.0)
    print(f"[3/5] Detected silence gaps: {silences}")
    assert len(silences) >= 1, "Expected at least 1 silence gap between 2.0s and 4.0s"
    
    fumbles = detect_fumbles(mock_segments)
    print(f"[3/5] Detected fumble filler words: {fumbles}")
    assert len(fumbles) >= 1, "Expected at least 1 fumble segment for word 'um'"
    assert fumbles[0]['start'] == 1.4, f"Expected fumble start 1.4, got {fumbles[0]['start']}"
    
    # 4. Test Clip Cutting & Transition Concatenation
    clip1 = str(test_dir / "clip1.mp4")
    clip2 = str(test_dir / "clip2.mp4")
    output_final = str(test_dir / "output_test_xfade.mp4")
    
    cut_clip(test_video, clip1, 0.5, 2.5) # 2s clip
    cut_clip(test_video, clip2, 3.5, 5.5) # 2s clip
    
    assert Path(clip1).exists() and Path(clip2).exists(), "Cut clip generation failed"
    
    print("[4/5] Testing concat_with_transitions...")
    concat_with_transitions([clip1, clip2], output_final, transition="xfade", transition_duration=0.5)
    assert Path(output_final).exists(), "Transition output video file missing"
    
    out_duration = get_video_duration(output_final)
    print(f"Concatenated transition video duration: {out_duration}s (Expected ~3.5s)")
    assert 3.0 <= out_duration <= 4.0, f"Unexpected output duration {out_duration}"
    
    print("[5/5] PHASE 1 VERIFICATION SUCCESSFUL!")

if __name__ == "__main__":
    run_test()
