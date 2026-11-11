# BuzzEdit's environment, (re)built in place. Run by "BuzzEdit Setup.exe", or by hand:
#   powershell -ExecutionPolicy Bypass -File installer\setup_env.ps1
# Every step skips what is already in place, so re-running it repairs a half-finished or
# cleaned-up install (e.g. after a disk cleaner removed .venv / node_modules).
. (Join-Path $PSScriptRoot "setup_common.ps1")
$root = $script:AppRoot
Set-Steps 8  # the last step is the self-test

Step "Checking the tools BuzzEdit needs (Python 3.11, Node.js, git, ffmpeg)"
$null = Find-Python "3.11"
$null = Require-Command "npm.cmd" "Install Node.js (LTS) from nodejs.org, then run Setup again."
$null = Require-Command "git" "Install Git for Windows, then run Setup again (seed-vc is cloned with it)."
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-SetupLog "WARNING: ffmpeg is not on PATH. Install it (winget install Gyan.FFmpeg.Shared) before rendering."
}

Step "Backend environment (.venv: Python 3.11, CUDA 12.8 PyTorch, backend\requirements.txt)"
$py = Ensure-Venv (Join-Path $root ".venv") "3.11"
# torch is not in requirements.txt on purpose: the CUDA build comes from PyTorch's own index.
Ensure-CudaTorch $py @("torch", "torchaudio")
# torchvision drives the DeepLabV3 matte (presentation/matte.py); it must match the installed torch.
if (-not (Test-PyImport $py "torchvision")) {
    $torchVer = (& $py -c "import torch; print(torch.__version__.split('+')[0])").Trim()
    Pip-Install $py @("torch==$torchVer", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128") "Installing torchvision"
}
Pip-Requirements $py (Join-Path $root "backend\requirements.txt")

Step "Desktop shell (npm packages and the UI bundle)"
Ensure-NodeModules $root
# Electron's binary is a separate download (node_modules\electron\install.js); a node_modules that
# survived without it (or a skipped npm ci) still cannot start the app.
if (-not (Test-Path (Join-Path $root "node_modules\electron\dist\electron.exe"))) {
    Invoke-Native "Downloading Electron" (Require-Command "node" "Install Node.js (LTS) from nodejs.org.") @((Join-Path $root "node_modules\electron\install.js")) $root
}
if (-not (Test-Path (Join-Path $root "dist\index.html"))) {
    Invoke-Native "Building the UI bundle" (Require-Command "npm.cmd" "") @("run", "build") $root
}

Step "Narration engine (engines\tts\.venv: Chatterbox Multilingual, Kokoro, faster-whisper)"
$tts = Ensure-Venv (Join-Path $root "engines\tts\.venv") "3.11"
if (-not ((Test-PyImport $tts "chatterbox.mtl_tts") -and (Test-PyImport $tts "kokoro") -and (Test-PyImport $tts "faster_whisper"))) {
    Pip-Install $tts @("chatterbox-tts", "kokoro", "faster-whisper", "indic-transliteration", "librosa", "soundfile") "Installing the narration engine"
}
# Chatterbox's watermarker (resemble-perth) imports pkg_resources, which setuptools 81+ dropped;
# without it the watermarker is None and every narration fails "'NoneType' object is not callable".
if (-not (Test-PyImport $tts "pkg_resources")) {
    Pip-Install $tts @("setuptools<81") "Restoring pkg_resources for the narration watermarker"
}
# Chatterbox pins torch 2.6, whose wheels cannot drive the Blackwell 5060 Ti; 2.7.1+cu128 runs it
# (measured 2026-10-23: 13 s of speech in 17 s on the GPU). Put the CUDA build back over the pin.
Ensure-CudaTorch $tts @("torch", "torchaudio") "2.7.1"
if (-not (Test-Native $tts @("-c", "import torch, sys; sys.exit(0 if torch.version.cuda else 1)"))) {
    Ensure-CudaTorch $tts @("torch", "torchaudio") "2.7.1" -Force
}

Step "Voice conversion engine (engines\seedvc: seed-vc, own Python 3.11)"
$svc = Join-Path $root "engines\seedvc\.venv\Scripts\python.exe"
$wrapper = Join-Path $root "engines\seedvc\seed-vc\seed_vc_wrapper.py"
if ((Test-Path $svc) -and (Test-Path $wrapper) -and (Test-PyImport $svc "torch")) {
    Write-SetupLog "seed-vc present"
} else {
    Invoke-Native "Setting up seed-vc" "powershell.exe" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $root "scripts\setup_seedvc.ps1"))
}

Step "ComfyUI (pictures, clips, music and sounds)"
$comfyPy = Join-Path $env:USERPROFILE "Documents\ComfyUI\.venv\Scripts\python.exe"
if (Test-Path $comfyPy) { Write-SetupLog "ComfyUI Desktop environment present: $comfyPy" }
else { Write-SetupLog "WARNING: ComfyUI Desktop's environment is missing ($comfyPy). BuzzEdit repairs it on start (scripts\comfy_repair.py); see workflows\README.md for the models and custom nodes." }

Step "Motion graphics engine (motion\: Remotion, MOTION_GRAPHICS_PLAN.md)"
# A self-contained Node project, never the Electron app's own node_modules --
# `npm install` only, never `npm ci` (Ensure-NodeModules above runs `npm ci`
# when a lockfile exists, which is why this is its own small step rather than
# a call to it). `npm ls` first, same as Ensure-NodeModules: a node_modules
# that already resolves is left alone.
$motionDir = Join-Path $root "motion"
$npmExe = Require-Command "npm.cmd" "Install Node.js (LTS) from nodejs.org, then run Setup again."
$motionModules = Join-Path $motionDir "node_modules"
if ((Test-Path $motionModules) -and (Test-Native $npmExe @("ls", "--omit=dev", "--depth=0", "--prefix", $motionDir))) {
    Write-SetupLog "motion/node_modules present and complete: $motionModules"
} else {
    Invoke-Native "npm install (motion)" $npmExe @("install", "--no-audit", "--no-fund") $motionDir
}

Step "Self-test: does every part actually work (installer\selftest.py)"
# Files being there is not enough (2026-10-26: after a rebuild Whisper could not load cuBLAS, fell
# back to the CPU and hung a production for 9 h). A setup that does not pass is not finished.
$selftest = Join-Path $root "installer\selftest.py"
if (-not (Test-Native $py @($selftest))) {
    Write-SetupLog "Self-test failed; installing CUDA 12 / cuDNN 9 runtime wheels for Whisper and testing again."
    Pip-Install $py @("nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*", "nvidia-cuda-runtime-cu12") "Installing Whisper's CUDA libraries"
    if (-not (Test-Native $py @($selftest))) {
        $report = Get-Content (Join-Path $root "installer\selftest.json") -Raw -ErrorAction SilentlyContinue
        Fail "BuzzEdit's self-test still fails after repair. Failing checks: $report"
    }
}
Invoke-Native "Self-test" $py @($selftest) $root
Done "BuzzEdit's environment is ready and passes its self-test. Models (Whisper, Chatterbox, seed-vc) download on first use."
