# BuzzEdit desktop launcher.
#
# Runs with no console of its own (START_APP.vbs starts it hidden), so nothing
# here may depend on a visible window: progress goes to logs\launcher.log and a
# hard failure is surfaced with a message box instead of a printed error.
#
# ComfyUI is NOT started here any more. electron\main.cjs owns both the Python
# backend and ComfyUI: it spawns them with windowsHide so neither pops a console
# window, and kills them on quit so nothing is left holding the GPU.

$workspaceDir = $PSScriptRoot
$logDir = Join-Path $workspaceDir "logs"
$logFile = Join-Path $logDir "launcher.log"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

function Write-Log($message) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $message
    Add-Content -Path $logFile -Value $line -Encoding utf8
    Write-Host $message
}

function Show-Failure($message) {
    Write-Log "FAILED: $message"
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "$message`n`nSee $logFile", "BuzzEdit", "OK", "Error") | Out-Null
}

Write-Log "=== launch ==="
Set-Location $workspaceDir

# 1. Build the React bundle if missing OR stale. Building only-if-missing meant a
# source fix never reached the app until someone remembered `npm run build` --
# the window kept loading the old bundle and the fix looked like it did nothing.
$distIndex = Join-Path $workspaceDir "dist\index.html"
$needBuild = -not (Test-Path $distIndex)
if (-not $needBuild) {
    $distTime = (Get-Item $distIndex).LastWriteTime
    $newest = Get-ChildItem (Join-Path $workspaceDir "src"), (Join-Path $workspaceDir "index.html") -Recurse -File |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newest -and $newest.LastWriteTime -gt $distTime) { $needBuild = $true }
}

if ($needBuild) {
    Write-Log "Building desktop bundle (source changed)..."
    # npm.cmd, not npm: the bare name resolves to the shell shim, which needs a
    # console this process does not have.
    $build = Start-Process -FilePath "npm.cmd" -ArgumentList "run build" `
        -WorkingDirectory $workspaceDir -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$logFile.build.out" -RedirectStandardError "$logFile.build.err"
    if ($build.ExitCode -ne 0) {
        Get-Content "$logFile.build.err" -ErrorAction SilentlyContinue | ForEach-Object { Write-Log "build: $_" }
        if (-not (Test-Path $distIndex)) {
            Show-Failure "The UI bundle failed to build and there is no previous build to fall back on."
            exit 1
        }
        Write-Log "Build failed; falling back to the existing bundle."
    } else {
        Write-Log "Build complete."
    }
}

# 2. Start Electron. electron.exe directly, not `npx electron .` -- npx runs
# through node and cmd shims, each of which is a console process, and that chain
# is what put a stray cmd window next to the UI.
$electron = Join-Path $workspaceDir "node_modules\electron\dist\electron.exe"
if (-not (Test-Path $electron)) {
    Show-Failure "Electron is not installed. Run ``npm install`` in $workspaceDir."
    exit 1
}

Write-Log "Launching BuzzEdit..."
$env:NODE_ENV = "production"
Start-Process -FilePath $electron -ArgumentList "." -WorkingDirectory $workspaceDir
Write-Log "Electron started; launcher exiting."
