"""Download the Natural Earth data the map cutaways draw from (once, ~48 MB).

Public-domain GeoJSON from the natural-earth-vector repository:
countries (50m), states/provinces (10m) and populated places (10m).
Usage: python tools/fetch_geo.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from presentation.maps import COUNTRIES_FILE, PLACES_FILE, STATES_FILE  # noqa: E402

BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"

for target in (COUNTRIES_FILE, STATES_FILE, PLACES_FILE):
    if target.exists() and target.stat().st_size > 100_000:
        print(f"have {target.name}")
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {target.name} …")
    with urllib.request.urlopen(BASE + target.name, timeout=300) as resp:
        target.write_bytes(resp.read())
    print(f"  {target.stat().st_size // 1024} KB")
