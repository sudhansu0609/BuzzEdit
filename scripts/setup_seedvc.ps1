# BuzzEdit seed-vc voice engine - one-time setup on a fresh machine (Windows PowerShell)
# Ported from skinWalker\scripts\setup.ps1 (VOICE_CHANGER_PLAN.md, brief S0).
# Builds engines\seedvc\: clones seed-vc (GPLv3) into engines\seedvc\seed-vc and creates
# engines\seedvc\.venv (Python 3.11) with the CUDA 12.8 PyTorch and seed-vc's requirements.
# BuzzEdit runs engines\seedvc\cli.py with that Python as a separate process (backend\audio\seedvc_env.py).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup_seedvc.ps1
$ErrorActionPreference = "Stop"
$root = (Get-Item $PSScriptRoot).Parent.FullName
$engine = Join-Path $root "engines\seedvc"
$seedvc = Join-Path $engine "seed-vc"
$venv = Join-Path $engine ".venv"
Write-Host "== BuzzEdit seed-vc setup ==  engine: $engine"

# 1. Require Python 3.11 (torch / seed-vc need 3.10-3.12; 3.13 and 3.14 are too new)
& py -3.11 --version *> $null
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 not found. Install it, then re-run. (Check with: py -0p)" }

# 2. Clone the seed-vc engine (GPLv3 third-party project) if missing
if (-not (Test-Path (Join-Path $seedvc "seed_vc_wrapper.py"))) {
    Write-Host "Cloning seed-vc into engines\seedvc\seed-vc ..."
    New-Item -ItemType Directory -Force -Path $engine | Out-Null
    git clone --depth 1 https://github.com/Plachtaa/seed-vc.git $seedvc
    if ($LASTEXITCODE -ne 0) { throw "git clone of seed-vc failed" }
}

# 3. Create the virtual environment
$py = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment (engines\seedvc\.venv) with Python 3.11 ..."
    & py -3.11 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Creating the virtual environment failed" }
}

# 4. Upgrade packaging tools. Pin setuptools<82: the CUDA 12.8 torch wheels
#    require it, and a newer setuptools breaks `import torch`.
& $py -m pip install --upgrade pip wheel "setuptools<82"
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

# 5. PyTorch for CUDA 12.8 - REQUIRED for RTX 50-series / Blackwell (sm_120).
#    Do NOT let seed-vc's pinned torch==2.4.0 install; it has no Blackwell kernels.
Write-Host "Installing CUDA 12.8 PyTorch (Blackwell-compatible) ..."
& $py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "Installing PyTorch failed" }

# 6. seed-vc's own requirements, without its torch lines (the nightly cu126 index lines and the
#    torch==2.4.0 pins): torch is already installed above.
$requirements = Join-Path $env:TEMP "seedvc-requirements-no-torch.txt"
Get-Content (Join-Path $seedvc "requirements.txt") |
    Where-Object { $_ -notmatch '^\s*(torch|torchvision|torchaudio)\b' } |
    Set-Content -Encoding ascii $requirements
& $py -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Installing seed-vc's requirements failed" }

# 7. Verify the GPU is visible to PyTorch
& $py -c "import torch; print('CUDA available:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'))"

Write-Host ""
Write-Host "Setup complete. Download the models once (a few GB from HuggingFace, offline after that):"
Write-Host "  & `"$py`" `"$engine\cli.py`" warm"
Write-Host "BuzzEdit finds engines\seedvc by itself; the seedvc_dir / seedvc_python settings override it."
