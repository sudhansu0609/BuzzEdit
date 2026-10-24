# BuzzEdit "installer": puts a BuzzEdit shortcut on the Desktop and in the
# Start Menu (Buzzcaf Media > BuzzEdit), pointing at START_APP.vbs in this
# checkout. Nothing is copied: the app runs from here, next to the Studio.
#
# One click on the shortcut opens one window and nothing else: START_APP.vbs
# starts Electron hidden-console, and if a BuzzEdit backend is already running
# (the Studio keeps a headless one alive) the new window adopts it instead of
# starting a second backend (electron/port-plan.cjs resolveBackend).
#
# Run:  powershell -ExecutionPolicy Bypass -File scripts\install_shortcuts.ps1
#       (add -Remove to take the shortcuts away again)
param([switch]$Remove)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vbs = Join-Path $root "START_APP.vbs"
$icon = Join-Path $root "build\icon.ico"
if (-not (Test-Path $vbs)) { throw "START_APP.vbs not found in $root" }

$desktop = [Environment]::GetFolderPath("Desktop")
$startMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "Buzzcaf Media"
$targets = @((Join-Path $desktop "BuzzEdit.lnk"), (Join-Path $startMenu "BuzzEdit.lnk"))

if ($Remove) {
    foreach ($lnk in $targets) { if (Test-Path $lnk) { Remove-Item $lnk; "removed $lnk" } }
    exit 0
}

$shell = New-Object -ComObject WScript.Shell
foreach ($lnk in $targets) {
    $dir = Split-Path $lnk
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    $s = $shell.CreateShortcut($lnk)
    $s.TargetPath = "$env:WINDIR\System32\wscript.exe"
    $s.Arguments = "`"$vbs`""
    $s.WorkingDirectory = $root
    $s.Description = "BuzzEdit - AI video editor"
    if (Test-Path $icon) { $s.IconLocation = "$icon,0" }
    $s.Save()
    "created $lnk"
}
