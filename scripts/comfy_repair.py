"""Check and repair the Python environment ComfyUI runs in.

BuzzEdit's Electron launches ComfyUI with `<COMFY_DATA>/.venv/Scripts/python.exe`.
On 2026-10-20 that folder had vanished (cause unknown) and every production since
2026-10-12 had rendered with no pictures or clips. This script is what Electron
runs before starting ComfyUI and whenever ComfyUI keeps dying, so a missing or
broken environment is rebuilt instead of failing in silence.

    python comfy_repair.py --check     # exit 0 healthy, 3 broken (nothing changed)
    python comfy_repair.py             # repair if broken; exit 0 healthy, 1 failed

Repair, in order of cost:
  1. the environment exists and only some packages are missing: install those;
  2. otherwise rebuild it in `.venv.rebuild` (Python 3.12, the pinned CUDA 12.8
     PyTorch, ComfyUI's requirements, then every custom node's requirements),
     verify it, and only then swap it in -- a failed rebuild never removes a
     working environment.

Versions come from the newest ComfyUI Desktop snapshot when there is one, so the
rebuilt environment matches the last known-good one. State is written to
`<COMFY_DATA>/.venv_repair.json`; a lock file keeps two repairs from running.
Standard library only, so any Python 3 can run it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HOME = Path.home()
DEFAULT_DATA = os.environ.get("COMFY_DATA", str(HOME / "Documents" / "ComfyUI"))
DEFAULT_CODE = os.environ.get("COMFY_CODE", str(HOME / "ComfyUI-Installs" / "ComfyUI" / "ComfyUI"))
PYTHON_VERSION = "3.12.11"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_PINS = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128", "torchaudio": "2.10.0+cu128"}

# Modules a production cannot do without (import name -> pip name). Pictures and
# clips need torch + ComfyUI's own deps; the clip workflow needs gguf; the
# main-character face swap needs insightface + onnxruntime + cv2.
ESSENTIAL_MODULES: Dict[str, str] = {
    "torch": "torch", "torchvision": "torchvision", "torchaudio": "torchaudio",
    "aiohttp": "aiohttp", "yaml": "pyyaml", "safetensors": "safetensors", "av": "av",
    "einops": "einops", "transformers": "transformers", "PIL": "pillow", "scipy": "scipy",
    "psutil": "psutil", "sqlalchemy": "sqlalchemy", "alembic": "alembic",
    "comfy_kitchen": "comfy-kitchen", "comfy_aimdo": "comfy-aimdo",
    "comfyui_frontend_package": "comfyui-frontend-package",
    "torchsde": "torchsde", "kornia": "kornia", "spandrel": "spandrel",
    "gguf": "gguf", "cv2": "opencv-python", "insightface": "insightface",
    "onnxruntime": "onnxruntime", "imageio_ffmpeg": "imageio-ffmpeg",
}
# Custom nodes BuzzEdit's workflows use: a rebuild that cannot import these is
# reported as degraded (still swapped in -- pictures beat no pictures).
ESSENTIAL_NODES = ["ComfyUI-GGUF", "ComfyUI-ReActor", "comfyui-videohelpersuite"]
# Installed first among the custom nodes so a later node's requirements cannot
# leave them broken.
NODE_ORDER_FIRST = ESSENTIAL_NODES + ["comfyui-kjnodes"]


class Repair:
    def __init__(self, data: Path, code: Path, uv: Optional[str], log_file: Optional[Path]):
        self.data, self.code = data, code
        self.venv = data / ".venv"
        self.rebuild = data / ".venv.rebuild"
        self.state_file = data / ".venv_repair.json"
        self.lock_file = data / ".venv_repair.lock"
        self.uv = uv or find_uv()
        self.log_file = log_file
        self.state: Dict = {}

    # ---------------------------------------------------------------- output
    def log(self, msg: str) -> None:
        line = f"[comfy-repair {_dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass

    def save_state(self, **kw) -> None:
        self.state.update(kw, updated_at=_dt.datetime.now().isoformat(timespec="seconds"))
        try:
            self.state_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ---------------------------------------------------------------- checks
    @staticmethod
    def python_of(venv: Path) -> Path:
        return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def missing_modules(self, venv: Path) -> Tuple[bool, List[str], str]:
        """(python runs, missing essential modules, note)."""
        py = self.python_of(venv)
        if not py.exists():
            return False, list(ESSENTIAL_MODULES), f"{py} does not exist"
        probe = (
            "import importlib, json, sys\n"
            f"mods = {list(ESSENTIAL_MODULES)!r}\n"
            "bad = []\n"
            "for m in mods:\n"
            "    try:\n"
            "        importlib.import_module(m)\n"
            "    except Exception as e:\n"
            "        bad.append(m)\n"
            "cuda = None\n"
            "try:\n"
            "    import torch; cuda = bool(torch.cuda.is_available()); ver = torch.__version__\n"
            "except Exception:\n"
            "    ver = None\n"
            "print(json.dumps({'bad': bad, 'cuda': cuda, 'torch': ver}))\n"
        )
        try:
            out = subprocess.run([str(py), "-c", probe], capture_output=True, text=True,
                                 timeout=600, creationflags=_no_window())
        except Exception as exc:  # the interpreter itself is broken
            return False, list(ESSENTIAL_MODULES), f"python would not run: {exc}"
        try:
            res = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:
            return False, list(ESSENTIAL_MODULES), f"probe failed: {(out.stderr or '')[-400:]}"
        bad = list(res["bad"])
        if "torch" not in bad and res.get("cuda") is False:
            bad.append("torch")  # a CPU-only torch is as good as none here
            return True, bad, f"torch {res.get('torch')} has no CUDA"
        return True, bad, f"torch {res.get('torch')}, cuda={res.get('cuda')}"

    def node_import_failures(self, venv: Path) -> Tuple[bool, List[str], str]:
        """Start ComfyUI with --quick-test-for-ci and read which custom nodes failed."""
        py = self.python_of(venv)
        port = 8199
        cmd = [str(py), "main.py", "--base-directory", str(self.data), "--quick-test-for-ci",
               "--listen", "127.0.0.1", "--port", str(port)]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        try:
            out = subprocess.run(cmd, cwd=str(self.code), capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=900, env=env,
                                 creationflags=_no_window())
        except subprocess.TimeoutExpired:
            return False, [], "ComfyUI quick test timed out"
        text = (out.stdout or "") + (out.stderr or "")
        failed = sorted(set(re.findall(r"\(IMPORT FAILED\): .*?[\\/]custom_nodes[\\/]([^\\/\r\n]+)", text))
                        | set(Path(m).name for m in re.findall(r"IMPORT FAILED: (.+)", text)))
        ok = out.returncode == 0
        note = "" if ok else (text[-800:] or f"exit {out.returncode}")
        return ok, failed, note

    # ----------------------------------------------------------------- pins
    def snapshot_pins(self) -> Dict[str, str]:
        """name -> version from the newest ComfyUI Desktop snapshot (torch excluded)."""
        pats = [str(self.code.parent / ".launcher" / "snapshots" / "*.json"),
                str(self.data / ".snapshots" / "*.json")]
        files = sorted((f for p in pats for f in glob.glob(p)), key=os.path.getmtime, reverse=True)
        for f in files:
            try:
                d = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception:
                continue
            pk = d.get("pipPackages")
            if isinstance(pk, list):
                pk = {x.get("name"): x.get("version") for x in pk if isinstance(x, dict)}
            if isinstance(pk, dict) and len(pk) > 20:
                self.log(f"version pins from {Path(f).name} ({len(pk)} packages)")
                return {k.lower(): v for k, v in pk.items()
                        if k and v and k.lower() not in TORCH_PINS and "+" not in str(v)}
        return {}

    def constraints_file(self, pins: Dict[str, str]) -> Path:
        lines = [f"{k}=={v}" for k, v in TORCH_PINS.items()]
        lines += [f"{k}=={v}" for k, v in sorted(pins.items())]
        path = Path(tempfile.gettempdir()) / "comfy_repair_constraints.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def torch_only_constraints(self) -> Path:
        path = Path(tempfile.gettempdir()) / "comfy_repair_torch.txt"
        path.write_text("\n".join(f"{k}=={v}" for k, v in TORCH_PINS.items()) + "\n", encoding="utf-8")
        return path

    # ------------------------------------------------------------- installing
    def uv_pip(self, venv: Path, args: List[str], what: str, timeout: int = 3600) -> bool:
        cmd = [self.uv, "pip", "install", "--python", str(self.python_of(venv)),
               "--index-strategy", "unsafe-best-match", *args]
        self.log(f"installing {what}")
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=timeout, creationflags=_no_window())
        except subprocess.TimeoutExpired:
            self.log(f"  {what}: timed out")
            return False
        if out.returncode != 0:
            tail = " | ".join((out.stderr or out.stdout or "").strip().splitlines()[-6:])
            self.log(f"  {what}: failed ({tail[:600]})")
            return False
        return True

    def install_with_fallback(self, venv: Path, args: List[str], what: str,
                              pinned: Path, loose: Path) -> bool:
        """Install under the snapshot's pins; if they cannot be met, under torch's alone."""
        if self.uv_pip(venv, [*args, "-c", str(pinned)], what):
            return True
        return self.uv_pip(venv, [*args, "-c", str(loose)], f"{what} (against installed versions)")

    def freeze(self, venv: Path) -> Optional[Path]:
        """The environment's current packages as a constraints file."""
        out = subprocess.run([self.uv, "pip", "freeze", "--python", str(self.python_of(venv))],
                             capture_output=True, text=True, encoding="utf-8", errors="replace",
                             creationflags=_no_window())
        lines = [l for l in out.stdout.splitlines() if "==" in l and not l.startswith("-e")]
        if out.returncode != 0 or not lines:
            return None
        path = Path(tempfile.gettempdir()) / "comfy_repair_frozen.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def node_dirs(self) -> List[Path]:
        root = self.data / "custom_nodes"
        nodes = [p for p in root.iterdir() if p.is_dir() and (p / "requirements.txt").exists()]
        order = {n.lower(): i for i, n in enumerate(NODE_ORDER_FIRST)}
        return sorted(nodes, key=lambda p: (order.get(p.name.lower(), 99), p.name.lower()))

    def install_everything(self, venv: Path) -> List[str]:
        """Torch, ComfyUI, then custom nodes. Returns the requirement sets that failed."""
        failed: List[str] = []
        pins = self.snapshot_pins()
        pinned, loose = self.constraints_file(pins), self.torch_only_constraints()
        torch_args = [f"{k}=={v}" for k, v in TORCH_PINS.items()]
        if not self.uv_pip(venv, [*torch_args, "--index-url", TORCH_INDEX], "PyTorch (CUDA 12.8)"):
            raise RuntimeError("PyTorch could not be installed")
        req = self.code / "requirements.txt"
        if not self.install_with_fallback(venv, ["-r", str(req)], "ComfyUI requirements", pinned, loose):
            raise RuntimeError("ComfyUI's own requirements could not be installed")
        # From here on nothing may change what ComfyUI itself runs on: a custom
        # node whose snapshot pins do not resolve is retried against the
        # packages already installed (frozen), never with them unpinned -- an
        # unpinned retry let faster-whisper drag `av` below ComfyUI's minimum.
        loose = self.freeze(venv) or loose
        for node in self.node_dirs():
            if not self.install_with_fallback(venv, ["-r", str(node / "requirements.txt")],
                                              f"custom node {node.name}", pinned, loose):
                failed.append(node.name)
        # Face swap: insightface/onnxruntime come from ReActor's install.py, not
        # its requirements.txt.
        extra = [f"{n}=={pins[n]}" if n in pins else n for n in ("insightface", "onnxruntime")]
        if not self.install_with_fallback(venv, extra, "face swap runtime (insightface, onnxruntime)",
                                          pinned, loose):
            failed.append("insightface/onnxruntime")
        # Anything still missing from the essential list, by name.
        _, missing, _ = self.missing_modules(venv)
        if missing:
            self.install_with_fallback(venv, [ESSENTIAL_MODULES[m] for m in missing],
                                       "missing essentials " + ", ".join(missing), pinned, loose)
        return failed

    # ---------------------------------------------------------------- flows
    def check(self) -> Tuple[bool, List[str], str]:
        runs, missing, note = self.missing_modules(self.venv)
        return runs and not missing, missing, note

    def targeted(self, missing: List[str]) -> bool:
        pins = self.snapshot_pins()
        pinned = self.constraints_file(pins)
        loose = self.freeze(self.venv) or self.torch_only_constraints()
        if "torch" in missing or "torchvision" in missing or "torchaudio" in missing:
            ok = self.uv_pip(self.venv, [f"{k}=={v}" for k, v in TORCH_PINS.items()]
                             + ["--index-url", TORCH_INDEX, "--reinstall"], "PyTorch (CUDA 12.8)")
            if not ok:
                return False
        rest = [ESSENTIAL_MODULES[m] for m in missing if m not in TORCH_PINS]
        if rest:
            self.install_with_fallback(self.venv, rest, "missing packages " + ", ".join(rest), pinned, loose)
        ok, still, note = self.check()
        self.log(f"after targeted install: {'healthy' if ok else 'still missing ' + ', '.join(still)} ({note})")
        return ok

    def full_rebuild(self) -> bool:
        if self.rebuild.exists():
            shutil.rmtree(self.rebuild, ignore_errors=True)
        self.log(f"building a new environment in {self.rebuild}")
        out = subprocess.run([self.uv, "venv", "--python", PYTHON_VERSION, "--seed", str(self.rebuild)],
                             capture_output=True, text=True, creationflags=_no_window())
        if out.returncode != 0:
            out = subprocess.run([self.uv, "venv", "--python", "3.12", "--seed", str(self.rebuild)],
                                 capture_output=True, text=True, creationflags=_no_window())
        if out.returncode != 0:
            raise RuntimeError(f"uv venv failed: {out.stderr[-400:]}")
        failed_sets = self.install_everything(self.rebuild)
        runs, missing, note = self.missing_modules(self.rebuild)
        if not runs or any(m in missing for m in ("torch", "aiohttp", "safetensors", "comfy_kitchen")):
            raise RuntimeError(f"the new environment is not usable: missing {missing} ({note})")
        ok, failed_nodes, qnote = self.node_import_failures(self.rebuild)
        if not ok:
            raise RuntimeError(f"ComfyUI did not start in the new environment: {qnote}")
        essential_failed = [n for n in failed_nodes if n in ESSENTIAL_NODES]
        self.save_state(failed_requirement_sets=failed_sets, failed_nodes=failed_nodes,
                        essential_nodes_failed=essential_failed, missing_modules=missing)
        # Swap: keep a broken old one aside rather than deleting it.
        if self.venv.exists():
            aside = self.data / f".venv.broken-{_dt.datetime.now():%Y%m%d-%H%M%S}"
            self.log(f"moving the old environment aside to {aside.name}")
            os.replace(self.venv, aside)
        os.replace(self.rebuild, self.venv)
        self.log("new environment in place"
                 + (f"; custom nodes that failed to import: {', '.join(failed_nodes)}" if failed_nodes else ""))
        return not essential_failed

    def run(self) -> int:
        if not self.uv:
            self.log("uv not found; cannot repair (install uv or set COMFY_UV)")
            self.save_state(state="failed", error="uv not found")
            return 1
        if not acquire_lock(self.lock_file):
            self.log("another repair is running; waiting for it")
            while self.lock_file.exists() and lock_alive(self.lock_file):
                time.sleep(10)
            ok, missing, note = self.check()
            return 0 if ok else 1
        try:
            self.state = {"state": "checking", "started_at": _dt.datetime.now().isoformat(timespec="seconds")}
            self.save_state()
            ok, missing, note = self.check()
            self.log(f"check: {'healthy' if ok else 'broken'} ({note}"
                     + (f"; missing {', '.join(missing)}" if missing and len(missing) < len(ESSENTIAL_MODULES) else "") + ")")
            if ok:
                self.save_state(state="healthy", note=note)
                return 0
            runs = self.python_of(self.venv).exists() and len(missing) < len(ESSENTIAL_MODULES)
            if runs:
                self.save_state(state="repairing", mode="targeted", missing_modules=missing)
                if self.targeted(missing):
                    self.save_state(state="healthy", mode="targeted")
                    return 0
            self.save_state(state="repairing", mode="rebuild")
            essential_ok = self.full_rebuild()
            self.save_state(state="healthy" if essential_ok else "degraded", mode="rebuild",
                            finished_at=_dt.datetime.now().isoformat(timespec="seconds"))
            return 0
        except Exception as exc:
            self.log(f"repair failed: {exc}")
            self.save_state(state="failed", error=str(exc)[:1000],
                            finished_at=_dt.datetime.now().isoformat(timespec="seconds"))
            return 1
        finally:
            try:
                self.lock_file.unlink()
            except OSError:
                pass


def find_uv() -> Optional[str]:
    cands = [os.environ.get("COMFY_UV"), shutil.which("uv"),
             str(HOME / ".local" / "bin" / "uv.exe"),
             str(HOME / "AppData" / "Local" / "Programs" / "ComfyUI" / "Comfy Desktop" / "resources"
                 / "bootstrap-python" / "uv.exe")]
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def acquire_lock(path: Path) -> bool:
    if path.exists() and lock_alive(path):
        return False
    try:
        path.write_text(json.dumps({"pid": os.getpid(), "at": time.time()}), encoding="utf-8")
        return True
    except OSError:
        return False


def lock_alive(path: Path) -> bool:
    """A lock older than three hours, or whose process is gone, is stale."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if time.time() - float(d.get("at", 0)) > 3 * 3600:
        return False
    pid = int(d.get("pid", 0))
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                             creationflags=_no_window())
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _no_window() -> int:
    return 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--code", default=DEFAULT_CODE)
    ap.add_argument("--uv", default=None)
    ap.add_argument("--log", default=None, help="also append progress lines to this file")
    ap.add_argument("--check", action="store_true", help="only check; exit 3 when broken")
    ap.add_argument("--rebuild", action="store_true", help="rebuild even when the check passes")
    a = ap.parse_args()
    r = Repair(Path(a.data), Path(a.code), a.uv, Path(a.log) if a.log else None)
    if a.check:
        ok, missing, note = r.check()
        print(json.dumps({"healthy": ok, "missing": missing, "note": note}))
        return 0 if ok else 3
    if a.rebuild:
        if not acquire_lock(r.lock_file):
            r.log("another repair is running")
            return 1
        try:
            r.state = {"state": "repairing", "mode": "rebuild"}
            ok = r.full_rebuild()
            r.save_state(state="healthy" if ok else "degraded")
            return 0
        except Exception as exc:
            r.log(f"repair failed: {exc}")
            r.save_state(state="failed", error=str(exc)[:1000])
            return 1
        finally:
            try:
                r.lock_file.unlink()
            except OSError:
                pass
    return r.run()


if __name__ == "__main__":
    sys.exit(main())
