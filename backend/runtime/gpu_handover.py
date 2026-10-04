"""Clearing the card (and enough RAM) before a heavy ComfyUI phase.

The gpu_broker hands out leases but never evicts anyone but ComfyUI, and it
only reads free VRAM to log it. That was not enough for B-roll video: on the
reference 16 GB card a real run started its Wan 2.2 clips with ~5.5 GB free
(an LM Studio model squatting beside ComfyUI), the 14B sampler spilled into
shared system memory at ~470 s/step, and three clips each burned the full
15-minute timeout before the pass gave up on video. The stills that ran in the
same squeeze took ~30 s each instead of ~5.

`prepare_for_phase` is the explicit hand-over: eject every LM Studio model
(through the `lms` CLI, unconditionally — the HTTP listing can point at a
different server than the one actually holding the card), drop the in-process
ASR and aligner weights, ask ComfyUI to unload whatever the previous phase
left resident, then measure what is actually free and say whether the phase
can run. Never raises.

**Policy, for every model this app loads** (image, video, SFX, music,
thumbnail, ASR): a generation phase starts with `prepare_for_phase`, so the
model it needs has the card to itself — LM Studio and Ollama models ejected,
in-process weights dropped, ComfyUI's previous model unloaded — and a task
ends with `offload_all`, so nothing stays resident once the work is done.
Within a phase the model stays loaded between items (reloading Wan per clip
would cost more than it saves).
"""

import asyncio
import ctypes
import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger("gpu_handover")


def available_ram_mb() -> Optional[float]:
    """Physical RAM available right now, in MB, or None when unknown."""
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / (1024.0 * 1024.0)
            return None
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception as e:
        logger.debug("Could not read available RAM: %s", e)
    return None


def free_vram_mb() -> float:
    from runtime.gpu_broker import gpu_broker
    return gpu_broker.get_free_vram_mb()


def lm_studio_has_models(base_url: str = "http://127.0.0.1:1234") -> bool:
    """Whether LM Studio has any model resident right now (never raises)."""
    try:
        import requests
        data = requests.get(f"{base_url}/api/v0/models", timeout=2).json().get("data") or []
        return any(m.get("state") == "loaded" for m in data)
    except Exception:
        return False


async def eject_lm_studio() -> bool:
    """`lms unload --all`, whatever the HTTP API claims is loaded."""
    try:
        from llm.lm_launcher import _run, find_lms_cli, forget_resolution
    except Exception:
        return False
    lms = find_lms_cli()
    if not lms:
        return False
    rc, out, err = await _run([lms, "unload", "--all"], timeout=30)
    forget_resolution()
    if rc == 0:
        logger.info("GPU hand-over: LM Studio models ejected.")
        return True
    logger.info("GPU hand-over: lms unload --all rc=%s: %s", rc, (err or out or "").strip()[:160])
    return False


async def eject_ollama(base_url: str = "http://127.0.0.1:11434") -> bool:
    """Unload every model Ollama has resident (`keep_alive: 0`); it reloads on
    its next request. False when Ollama is not running or held nothing."""
    def _run() -> bool:
        try:
            import requests
            resident = requests.get(f"{base_url}/api/ps", timeout=2).json().get("models") or []
        except Exception:
            return False
        ejected = False
        for model in resident:
            name = model.get("name") or model.get("model")
            if not name:
                continue
            try:
                requests.post(f"{base_url}/api/generate",
                              json={"model": name, "keep_alive": 0}, timeout=15)
                ejected = True
            except Exception as e:
                logger.debug("Ollama unload of %s failed: %s", name, e)
        if ejected:
            logger.info("GPU hand-over: Ollama models ejected.")
        return ejected
    return await asyncio.to_thread(_run)


async def offload_all(reason: str = "task finished") -> None:
    """End of a task: unload every model this machine holds for us — LM
    Studio, Ollama, the in-process ASR/aligner, and ComfyUI's resident
    checkpoints. Never raises; each step is best-effort."""
    for step in (eject_lm_studio, eject_ollama):
        try:
            await step()
        except Exception as e:
            logger.debug("Offload step %s skipped: %s", step.__name__, e)
    release_in_process_models()
    await free_comfyui()
    logger.info("GPU offload (%s): all models released.", reason)


def release_in_process_models() -> None:
    """Whisper and the MMS aligner, if this process still holds them."""
    try:
        from asr.auto_edit import release_asr_gpu
        release_asr_gpu()
    except Exception as e:
        logger.debug("ASR release skipped: %s", e)
    try:
        from asr.forced_align import release_model
        release_model()
    except Exception as e:
        logger.debug("Aligner release skipped: %s", e)


async def free_comfyui() -> bool:
    """Ask an idle ComfyUI to unload its models (the previous phase's)."""
    try:
        from runtime.gpu_broker import gpu_broker
        return await gpu_broker.release_comfyui_vram()
    except Exception as e:
        logger.debug("ComfyUI free skipped: %s", e)
        return False


async def prepare_for_phase(name: str, need_vram_mb: float, need_ram_mb: float,
                            settle_s: float = 3.0) -> Dict[str, Any]:
    """Clear the card for `name`; report whether it now fits.

    Returns {"phase", "ready", "free_vram_mb", "available_ram_mb",
    "need_vram_mb", "need_ram_mb", "reason"}. Two rounds: if the first
    hand-over leaves too little VRAM (a model still unloading), wait and try
    once more before calling it.
    """
    result: Dict[str, Any] = {"phase": name, "need_vram_mb": need_vram_mb,
                              "need_ram_mb": need_ram_mb, "ready": False, "reason": ""}
    for attempt in range(2):
        await eject_lm_studio()
        await eject_ollama()
        release_in_process_models()
        await free_comfyui()
        await asyncio.sleep(settle_s * (attempt + 1))
        vram, ram = free_vram_mb(), available_ram_mb()
        result.update(free_vram_mb=round(vram), available_ram_mb=round(ram) if ram is not None else None)
        short = []
        if vram < need_vram_mb:
            short.append(f"{vram / 1024:.1f} GB VRAM free, needs {need_vram_mb / 1024:.1f} GB")
        if ram is not None and ram < need_ram_mb:
            short.append(f"{ram / 1024:.1f} GB RAM free, needs {need_ram_mb / 1024:.1f} GB")
        if not short:
            result["ready"], result["reason"] = True, ""
            break
        result["reason"] = "; ".join(short)
    logger.info("GPU hand-over for %s: %s (VRAM free %s MB, RAM available %s MB)", name,
                "ready" if result["ready"] else f"NOT ready -- {result['reason']}",
                result.get("free_vram_mb"), result.get("available_ram_mb"))
    return result

