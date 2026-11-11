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

# 0. Repair the environment when part of it is missing (2026-10-24: a disk cleaner removed every
# .venv and node_modules, and BuzzEdit only said "Electron is not installed"). The setup wizard
# runs installer\setup_env.ps1 in its own window, so the creator sees what is being installed; a
# setup already running (the Studio's auto-repair) is waited for by setup_env.ps1's own lock.
$required = @(".venv\Scripts\python.exe", "node_modules\electron\dist\electron.exe",
              "engines\tts\.venv\Scripts\python.exe", "engines\seedvc\.venv\Scripts\python.exe")
$missing = @($required | Where-Object { -not (Test-Path (Join-Path $workspaceDir $_)) })
if ($missing.Count -gt 0) {
    Write-Log ("Environment incomplete (missing: {0}); running setup." -f ($missing -join ", "))
    $wizard = Join-Path $workspaceDir "installer\BuzzEdit Setup.exe"
    if (Test-Path $wizard) {
        Start-Process -FilePath $wizard -ArgumentList "/quiet" -Wait
    } else {
        Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -Wait -ArgumentList `
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$(Join-Path $workspaceDir 'installer\setup_env.ps1')`""
    }
    $missing = @($required | Where-Object { -not (Test-Path (Join-Path $workspaceDir $_)) })
    if ($missing.Count -gt 0) {
        Show-Failure ("BuzzEdit's environment could not be repaired (still missing: {0}). See logs\setup.log, or run installer\BuzzEdit Setup.exe." -f ($missing -join ", "))
        exit 1
    }
    Write-Log "Environment repaired."
}

# 0b. The environment must WORK, not only exist (2026-10-26: a rebuilt .venv had every file but
# Whisper could not load cuBLAS, so a production hung for 9 h). installer\selftest.py takes ~10 s;
# a pass newer than the last package change (site-packages' timestamp) and under a week old is
# trusted, so most launches skip it. A failure runs the repair, which installs what is missing.
$selftest = Join-Path $workspaceDir "installer\selftest.py"
$selftestJson = Join-Path $workspaceDir "installer\selftest.json"
$python = Join-Path $workspaceDir ".venv\Scripts\python.exe"
if (Test-Path $selftest) {
    $fresh = $false
    if (Test-Path $selftestJson) {
        try {
            $last = Get-Content $selftestJson -Raw | ConvertFrom-Json
            $passedAt = (Get-Item $selftestJson).LastWriteTime
            $packagesAt = (Get-Item (Join-Path $workspaceDir ".venv\Lib\site-packages")).LastWriteTime
            $fresh = $last.ok -and ($passedAt -gt $packagesAt) -and ($passedAt -gt (Get-Date).AddDays(-7))
        } catch { $fresh = $false }
    }
    if (-not $fresh) {
        Write-Log "Running the environment self-test..."
        $t = Start-Process -FilePath $python -ArgumentList "`"$selftest`"" -WorkingDirectory $workspaceDir -WindowStyle Hidden -Wait -PassThru `
            -RedirectStandardOutput "$logFile.selftest.out" -RedirectStandardError "$logFile.selftest.err"
        Get-Content "$logFile.selftest.out" -ErrorAction SilentlyContinue | ForEach-Object { Write-Log "selftest: $_" }
        if ($t.ExitCode -ne 0) {
            Write-Log "Self-test failed; repairing the environment."
            Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -Wait -ArgumentList `
                "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$(Join-Path $workspaceDir 'installer\setup_env.ps1')`""
            $t = Start-Process -FilePath $python -ArgumentList "`"$selftest`"" -WorkingDirectory $workspaceDir -WindowStyle Hidden -Wait -PassThru `
                -RedirectStandardOutput "$logFile.selftest.out" -RedirectStandardError "$logFile.selftest.err"
            Get-Content "$logFile.selftest.out" -ErrorAction SilentlyContinue | ForEach-Object { Write-Log "selftest: $_" }
            if ($t.ExitCode -ne 0) {
                $failed = (Get-Content "$logFile.selftest.out" -ErrorAction SilentlyContinue | Where-Object { $_ -like "*FAIL*" }) -join "`n"
                Show-Failure "BuzzEdit's environment still fails its self-test after an automatic repair:`n$failed`n`nSee logs\setup.log."
                exit 1
            }
            Write-Log "Environment repaired; self-test passes."
        }
    }
}

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
