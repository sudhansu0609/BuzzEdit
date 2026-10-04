# Generation Benchmark -- current vs upgraded clip and still workflows.
# Plan: docs\GENERATION_UPGRADE_PLAN.md. Start it with RUN_GENERATION_BENCHMARK.bat.
#
# Run only when Buzzcaf Studio and BuzzEdit are closed: it installs one package
# into ComfyUI's environment and starts its own ComfyUI. It changes nothing
# BuzzEdit uses; the results go to BuzzEdit\data\bench\<date>.

$ErrorActionPreference = 'Stop'
$BE        = 'B:\youtubeProjects\Buzzcaf_Media\BuzzEdit'
$ComfyData = 'C:\Users\singh\Documents\ComfyUI'
$ComfyPy   = Join-Path $ComfyData '.venv\Scripts\python.exe'
$Staged    = Join-Path $ComfyData 'custom_nodes_staged\ComfyUI-FlashVSR_Ultra_Fast'
$NodeDst   = Join-Path $ComfyData 'custom_nodes\ComfyUI-FlashVSR_Ultra_Fast'
$Out       = Join-Path $BE ('data\bench\' + (Get-Date -Format 'yyyy-MM-dd_HHmm'))
New-Item -ItemType Directory -Force $Out | Out-Null
Start-Transcript -Path (Join-Path $Out 'run.log') | Out-Null

function Step($text) { Write-Host ''; Write-Host "== $text" -ForegroundColor Cyan }

# 1. Nothing else may be using ComfyUI or the GPU.
Step '1/6 Checking that Buzzcaf Studio, BuzzEdit and ComfyUI are closed'
$busy = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'desktop_app\.py|BuzzEdit\\node_modules\\electron|ComfyUI\\\.venv.+main\.py|BuzzEdit\\\.venv.+main\.py' }
if ($busy) {
    Write-Host 'Still running -- close Buzzcaf Studio (and let BuzzEdit finish), then start this again:' -ForegroundColor Yellow
    $busy | ForEach-Object { Write-Host ("  pid {0}: {1}" -f $_.ProcessId, $_.Name) }
    Stop-Transcript | Out-Null; exit 1
}

# 2. The files downloaded on 2026-10-03 must be there.
Step '2/6 Checking the downloaded models'
$need = @(
  'B:\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable\ComfyUI\models\unet\wan2.2_i2v_high_noise_14B_Q5_K_M.gguf',
  'B:\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable\ComfyUI\models\unet\wan2.2_i2v_low_noise_14B_Q5_K_M.gguf',
  'B:\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable\ComfyUI\models\loras\wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors',
  'B:\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable\ComfyUI\models\loras\wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors',
  (Join-Path $ComfyData 'models\frame_interpolation\rife_v4.26.safetensors'),
  (Join-Path $ComfyData 'models\FlashVSR-v1.1\diffusion_pytorch_model_streaming_dmd.safetensors'))
$missing = $need | Where-Object { -not (Test-Path $_) }
if ($missing) { Write-Host 'Missing:' -ForegroundColor Red; $missing | ForEach-Object { "  $_" }; Stop-Transcript | Out-Null; exit 1 }
Write-Host 'All present.'

# 3. The FlashVSR node's one missing dependency, pinned to this torch (2.10 -> Triton 3.6).
Step '3/6 Installing triton-windows 3.6 into ComfyUI (torch is left as it is)'
$torchBefore = & $ComfyPy -c "import torch; print(torch.__version__)"
& $ComfyPy -m pip install --disable-pip-version-check --no-warn-script-location 'triton-windows>=3.6,<3.7' einops
if ($LASTEXITCODE -ne 0) { Write-Host 'pip failed; nothing else was changed.' -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }
$torchAfter = & $ComfyPy -c "import torch, triton; print(torch.__version__)"
& $ComfyPy -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.cuda.is_available())"
if ($torchBefore -ne $torchAfter) { Write-Host "WARNING: torch changed ($torchBefore -> $torchAfter)" -ForegroundColor Red }

# 4. The FlashVSR node goes live (ComfyUI loads nodes only at start).
Step '4/6 Enabling the FlashVSR node'
if (-not (Test-Path $NodeDst)) { Move-Item $Staged $NodeDst; Write-Host "Moved to $NodeDst" } else { Write-Host 'Already enabled.' }

# 5. GPU model policy: nothing else holds the card.
Step '5/6 Unloading LM Studio models'
$lms = Join-Path $env:USERPROFILE '.lmstudio\bin\lms.exe'
if (Test-Path $lms) { try { & $lms unload --all } catch { Write-Host "lms: $_" } }

# 6. The benchmark itself (~45 min). It starts and stops its own ComfyUI.
Step '6/6 Running the benchmark (about 45 minutes)'
& (Join-Path $BE '.venv\Scripts\python.exe') (Join-Path $BE 'backend\tools\bench_generation.py') `
    --out $Out --prompts (Join-Path $BE 'backend\tools\bench_prompts.json') --n 5
$code = $LASTEXITCODE
if ($code -eq 0) { Write-Host "Done. Results: $Out (summary.md, compare_*.png)" -ForegroundColor Green }
else { Write-Host "The benchmark stopped with code $code -- see $Out\run.log and comfyui.log" -ForegroundColor Red }
if (Test-Path $lms) { try { & $lms unload --all | Out-Null } catch {} }
Stop-Transcript | Out-Null
Start-Process explorer.exe $Out
exit $code
