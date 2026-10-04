"""
Talk to BuzzcafAI, the Studio whose agents write scripts and the structured
per-scene `visual_plan.json` this app renders B-roll from.

This is the BuzzEdit -> BuzzcafAI side of the bridge (the other half lives in
BuzzcafAI's own `backend/integrations/buzzedit_client.py`). It is used by
`routes/agents.py::plan_visuals` — a record-first flow, where a recording was
imported and transcribed here first, and BuzzcafAI is asked after the fact to
plan the visuals and hand back a directive-annotated script.

Discovery goes through the shared port ledger (`buzzcaf_ports.discover`,
already vendored into this app at `backend/buzzcaf_ports.py`), never a
hardcoded port: BuzzcafAI and this app's own backend both default to 8099.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import buzzcaf_ports  # backend_dir is already on sys.path (see main.py)
import requests


class BuzzcafUnavailable(Exception):
    """BuzzcafAI could not be found, or did not answer."""


def base_url() -> Optional[str]:
    env_url = os.environ.get("BUZZCAF_URL")
    return buzzcaf_ports.discover("buzzcaf", 8099, health_path="/health", env_url=env_url)


def plan_visuals(transcript: str, brand: str, timeout: tuple = (5, 120)) -> Dict[str, Any]:
    """`POST /api/produce/plan {brand, transcript}` on BuzzcafAI.

    Returns `{annotated_script, settings, visual_plan, skipped_beats}`.
    Raises `BuzzcafUnavailable` if BuzzcafAI cannot be found or does not
    answer -- callers should turn that into a clear HTTP error rather than a
    raw traceback.
    """
    url = base_url()
    if not url:
        raise BuzzcafUnavailable("BuzzcafAI is not reachable (not running, or not discoverable on the shared port ledger).")
    try:
        resp = requests.post(f"{url}/api/produce/plan", json={"brand": brand, "transcript": transcript}, timeout=timeout)
    except requests.RequestException as exc:
        raise BuzzcafUnavailable(f"BuzzcafAI did not answer: {exc}") from exc
    resp.raise_for_status()
    return resp.json()
