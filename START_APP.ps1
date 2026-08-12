# Buzzcaf Editor Desktop-Only Launcher
$comfyDir = "B:\ComfyUI_windows_portable_nvidia_cu118_or_cpu\ComfyUI_windows_portable"
$workspaceDir = $PSScriptRoot

# 1. Build dist bundle if missing
if (-not (Test-Path "$workspaceDir\dist\index.html")) {
    Write-Host "Building desktop bundle..." -ForegroundColor Yellow
    Set-Location $workspaceDir
    npm run build
}

# 2. Start ComfyUI in background
Write-Host "Starting AI Engine (ComfyUI GPU)..." -ForegroundColor Cyan
Start-Process -FilePath "$comfyDir\python_embeded\python.exe" -ArgumentList "-s ComfyUI\main.py --windows-standalone-build" -WorkingDirectory $comfyDir -WindowStyle Minimized

# 3. Start Desktop Electron Application
Write-Host "Launching Buzzcaf Editor Desktop App..." -ForegroundColor Green
$env:NODE_ENV = "production"
Set-Location $workspaceDir
npx electron .
