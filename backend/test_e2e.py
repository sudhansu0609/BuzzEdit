import os
import sys
import json
import logging
import urllib.request
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from utils.ffmpeg_utils import run_ffmpeg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_e2e")

# Where the backend actually is, not where it would like to be. The ledger
# first, then an identity scan of the preferred range — GUARDIAN_PLAN.md
# section 11 rule 5. `BUZZEDIT_URL` overrides everything, for a backend on
# another machine or in a container.
import buzzcaf_ports
from config import APP_NAME, PORT_SPAN, PREFERRED_PORT

BASE_URL = buzzcaf_ports.discover(
    APP_NAME, PREFERRED_PORT, "/api/health", PORT_SPAN,
    os.environ.get("BUZZEDIT_URL"),
) or f"http://127.0.0.1:{PREFERRED_PORT}"

def run_e2e_test():
    print("--- STARTING END-TO-END SYSTEM INTEGRATION TEST ---")
    
    test_dir = backend_dir / "data" / "test_tmp_e2e"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create a 5-second test video
    test_video = str(test_dir / "e2e_input.mp4")
    print(f"[1/7] Creating test video: {test_video}")
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=duration=5:size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-c:a", "aac", "-y", test_video
    ])
    
    # 2. Upload video file via multipart form
    print("[2/7] Testing Video Upload Endpoint...")
    import requests
    with open(test_video, "rb") as f:
        resp = requests.post(f"{BASE_URL}/api/projects/upload", files={"file": f})
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    project_data = resp.json()
    project_id = project_data["id"]
    print(f"Project created: ID={project_id}, Name={project_data['name']}")
    
    # 3. Transcribe & Analyze
    print("[3/7] Testing Transcription & Analysis Endpoints...")
    t_resp = requests.post(f"{BASE_URL}/api/transcription/transcribe", json={"project_id": project_id})
    assert t_resp.status_code == 200, f"Transcribe failed: {t_resp.text}"
    
    a_resp = requests.post(f"{BASE_URL}/api/analysis/analyze", json={"project_id": project_id, "remove_silence": True})
    assert a_resp.status_code == 200, f"Analyze failed: {a_resp.text}"
    print("Transcription & Analysis completed!")
    
    # 4. Render Clean Cut
    print("[4/7] Testing Render Engine Endpoint...")
    r_resp = requests.post(f"{BASE_URL}/api/rendering/render", json={"project_id": project_id})
    assert r_resp.status_code == 200, f"Render failed: {r_resp.text}"
    print(f"Render Complete! Output: {r_resp.json()['output_path']}")
    
    # 5. Export Filmora XML
    print("[5/7] Testing Filmora XML Exporter Endpoint...")
    f_resp = requests.post(f"{BASE_URL}/api/advanced/export_filmora", json={"project_id": project_id})
    assert f_resp.status_code == 200, f"Filmora export failed: {f_resp.text}"
    xml_path = f_resp.json()["xml_path"]
    assert Path(xml_path).exists(), "Filmora XML file missing"
    print(f"Filmora XML Exporter Verified: {xml_path}")
    
    # 6. Apply Color Grading
    print("[6/7] Testing Color Grading Endpoint...")
    cg_resp = requests.post(f"{BASE_URL}/api/advanced/color_grade", json={"project_id": project_id, "preset": "cinematic"})
    assert cg_resp.status_code == 200, f"Color grade failed: {cg_resp.text}"
    print(f"Color Grade Verified: {cg_resp.json()['output_video']}")
    
    # 7. Convert to 9:16 Shorts
    print("[7/7] Testing 9:16 Shorts Converter Endpoint...")
    s_resp = requests.post(f"{BASE_URL}/api/advanced/shorts", json={"project_id": project_id})
    assert s_resp.status_code == 200, f"Shorts convert failed: {s_resp.text}"
    print(f"9:16 Shorts Converter Verified: {s_resp.json()['output_video']}")
    
    print("--- ALL END-TO-END INTEGRATION TESTS PASSED 100%! ---")

if __name__ == "__main__":
    run_e2e_test()
