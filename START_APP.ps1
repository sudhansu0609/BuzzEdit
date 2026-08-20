# BuzzEdit Desktop-Only Launcher
# ComfyUI Desktop install. The old portable build was cu118 against everything
# else here being cu12, which is what threw "the procedure entry point could not
# be located in the dynamic link library".
$comfyCode = "C:\Users\singh\ComfyUI-Installs\ComfyUI\ComfyUI"
$comfyData = "C:\Users\singh\Documents\ComfyUI"
$comfyPython = "C:\Users\singh\Documents\ComfyUI\.venv\Scripts\python.exe"
$workspaceDir = $PSScriptRoot

# 1. Build dist bundle if missing OR stale. Building only-if-missing meant a
# source fix never reached the app until someone remembered `npm run build` —
# the window kept loading the old bundle and the fix looked like it did nothing.
$distIndex = "$workspaceDir\dist\index.html"
$needBuild = -not (Test-Path $distIndex)
if (-not $needBuild) {
    $distTime = (Get-Item $distIndex).LastWriteTime
    $newest = Get-ChildItem "$workspaceDir\src", "$workspaceDir\index.html" -Recurse -File |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newest -and $newest.LastWriteTime -gt $distTime) { $needBuild = $true }
}
if ($needBuild) {
    Write-Host "Building desktop bundle (source changed)..." -ForegroundColor Yellow
    Set-Location $workspaceDir
    npm run build
}

# 2. Start ComfyUI in background
Write-Host "Starting AI Engine (ComfyUI GPU)..." -ForegroundColor Cyan
# PYTHONIOENCODING: ComfyUI logs emoji and the cp1252 console codepage makes the
# logging call itself throw, killing the process.
$env:PYTHONIOENCODING = "utf-8"
Start-Process -FilePath $comfyPython -ArgumentList "main.py --base-directory `"$comfyData`"" -WorkingDirectory $comfyCode -WindowStyle Minimized

# 3. Start Desktop Electron Application
Write-Host "Launching BuzzEdit Desktop App..." -ForegroundColor Green
$env:NODE_ENV = "production"
Set-Location $workspaceDir
npx electron .
