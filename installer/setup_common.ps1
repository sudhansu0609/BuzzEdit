# Helpers dot-sourced by an app's installer\setup_env.ps1 (scripts/app_installer, see README.md).
# Every helper is idempotent: a step whose result is already in place is skipped, so a setup
# that stopped half-way is finished by running it again. Output goes to the console (the setup
# wizard shows it) and to logs\setup.log under the app's root.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$script:AppRoot = Split-Path -Parent $PSScriptRoot
$script:SetupLog = Join-Path $script:AppRoot "logs\setup.log"
New-Item -ItemType Directory -Force -Path (Split-Path $script:SetupLog) | Out-Null
$script:StepNo = 0
$script:StepTotal = 1
# Logging uses Write-Host so helpers that return a value (Ensure-Venv) return only that value;
# powershell.exe writes host output to stdout, which is what the wizard reads.

# A log line that cannot be written (a reader holding the file for a moment) is dropped, never fatal.
function Add-LogLine([string]$Text) {
    try { Add-Content -Path $script:SetupLog -Value $Text -Encoding utf8 -ErrorAction Stop } catch { }
}

function Write-SetupLog([string]$Message) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-LogLine $line
    Write-Host $Message
}

function Set-Steps([int]$Total) { $script:StepTotal = [Math]::Max(1, $Total); $script:StepNo = 0 }

function Step([string]$Text) {
    $script:StepNo++
    # The wizard turns this line into its progress bar and status text.
    Write-Host ("##STEP {0}/{1} {2}" -f $script:StepNo, $script:StepTotal, $Text)
    Add-LogLine ("{0}  == {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Text)
}

function Fail([string]$Message) {
    Write-SetupLog "FAILED: $Message"
    [Console]::Error.WriteLine("FAILED: $Message")
    Release-SetupLock
    exit 1
}

# One setup per app at a time. installer\setup.running holds the PID of the powershell running it;
# the wizard, the app's own launcher and the Studio's auto-repair all read it. A second setup waits
# for the first (up to an hour) and then runs, which is quick: every step finds its work done.
# A lock whose PID is gone (a killed setup) is stale and simply taken over.
$script:LockFile = Join-Path $PSScriptRoot "setup.running"

function Release-SetupLock {
    try {
        if ((Test-Path $script:LockFile) -and ((Get-Content $script:LockFile -Raw).Trim() -eq "$PID")) {
            Remove-Item $script:LockFile -Force
        }
    } catch { }
}

if (Test-Path $script:LockFile) {
    $other = 0
    [int]::TryParse(((Get-Content $script:LockFile -Raw) -as [string]).Trim(), [ref]$other) | Out-Null
    if ($other -ne $PID -and $other -gt 0 -and (Get-Process -Id $other -ErrorAction SilentlyContinue)) {
        Write-SetupLog "Another setup of this app is running (PID $other); waiting for it to finish."
        $deadline = (Get-Date).AddHours(1)
        while ((Get-Process -Id $other -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 10 }
        if (Get-Process -Id $other -ErrorAction SilentlyContinue) { Fail "the other setup (PID $other) is still running after an hour" }
    }
}
Set-Content -Path $script:LockFile -Value $PID -Encoding ascii

# Runs a native command, streams its output into the log, fails the setup on a non-zero exit.
function Invoke-Native([string]$What, [string]$Exe, [string[]]$Arguments, [string]$WorkDir = $script:AppRoot) {
    Write-SetupLog ("> {0} {1}" -f $Exe, ($Arguments -join " "))
    Push-Location $WorkDir
    try {
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        & $Exe @Arguments 2>&1 | ForEach-Object {
            $s = "$_"; if ($s.Trim()) { Add-LogLine "    $s"; Write-Host "    $s" }
        }
        $code = $LASTEXITCODE
        $ErrorActionPreference = $old
    } finally { Pop-Location }
    if ($code -ne 0) { Fail "$What failed (exit code $code). See $script:SetupLog" }
}

function Require-Command([string]$Name, [string]$Hint) {
    $c = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $c) { Fail "$Name was not found. $Hint" }
    return $c.Source
}

# Runs a probe (a native command whose only answer is its exit code) and returns $true on exit 0.
# Windows PowerShell 5.1 turns a native command's stderr into a terminating NativeCommandError under
# $ErrorActionPreference = "Stop" -- even with 2>$null -- so a probe such as `python -c "import torch"`
# on a venv without torch would end the whole setup silently. Probes therefore run under "Continue".
function Test-Native([string]$Exe, [string[]]$Arguments) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $Exe @Arguments 2>&1 | Out-Null; return ($LASTEXITCODE -eq 0) }
    catch { return $false }
    finally { $ErrorActionPreference = $old }
}

# The python.exe of a given version through the py launcher, e.g. Find-Python "3.11".
function Find-Python([string]$Version) {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        $exe = (& py "-$Version" -c "import sys; print(sys.executable)" 2>$null)
        $code = $LASTEXITCODE; $ErrorActionPreference = $old
        if ($code -eq 0 -and $exe) { return "$exe".Trim() }
    }
    Fail "Python $Version was not found (py -$Version). Install it from python.org, then run Setup again."
}

# uv (astral.sh) makes venvs and installs several times faster than pip; used when it is on PATH.
$script:Uv = (Get-Command uv -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source)

# Creates <Dir> as a venv with Python <Version> unless it already holds a working python.exe.
# The venv always gets pip too (uv venv --seed), so a tool's own setup script can use python -m pip.
function Ensure-Venv([string]$Dir, [string]$Version) {
    $exe = Join-Path $Dir "Scripts\python.exe"
    if (Test-Path $exe) {
        if (Test-Native $exe @("-c", "import sys")) { Write-SetupLog "venv present: $Dir"; return $exe }
        Write-SetupLog "venv at $Dir is broken; recreating it"
        Remove-Item -Recurse -Force $Dir
    }
    $base = Find-Python $Version
    if ($script:Uv) {
        Invoke-Native "Creating the Python $Version environment" $script:Uv @("venv", "--seed", "--python", $base, $Dir)
    } else {
        Invoke-Native "Creating the Python $Version environment" $base @("-m", "venv", $Dir)
        Invoke-Native "Upgrading pip" $exe @("-m", "pip", "install", "--upgrade", "pip", "wheel")
    }
    return $exe
}

# pip install into the venv of <Python> (uv pip when available).
function Pip-Install([string]$Python, [string[]]$Arguments, [string]$What = "Installing packages") {
    if ($script:Uv) { Invoke-Native $What $script:Uv (@("pip", "install", "--python", $Python) + $Arguments) }
    else { Invoke-Native $What $Python (@("-m", "pip", "install") + $Arguments) }
}

function Test-PyImport([string]$Python, [string]$Module) {
    return (Test-Native $Python @("-c", "import $Module"))
}

# Installs the CUDA 12.8 build of torch (the 5060 Ti is Blackwell: cu128 or newer) unless the venv
# already has a CUDA torch (of version <Pin>, when one is given).
function Ensure-CudaTorch([string]$Python, [string[]]$Packages = @("torch", "torchaudio"), [string]$Pin = "", [switch]$Force) {
    $check = if ($Pin) { "import torch, sys; sys.exit(0 if torch.version.cuda and torch.__version__.startswith('$Pin') else 1)" }
             else { "import torch, sys; sys.exit(0 if torch.version.cuda else 1)" }
    if (-not $Force) {
        if (Test-Native $Python @("-c", $check)) { Write-SetupLog "CUDA torch present"; return }
    }
    # @(...) keeps a one-package list an array: a lone value would turn the + below into string concatenation.
    $pkgs = @(if ($Pin) { $Packages | ForEach-Object { "$_==$Pin" } } else { $Packages })
    $extra = @(if ($Force) { "--reinstall", "--no-deps" })
    if (-not $script:Uv -and $Force) { $extra = @("--force-reinstall", "--no-deps") }
    Pip-Install $Python ($pkgs + $extra + @("--index-url", "https://download.pytorch.org/whl/cu128")) "Installing CUDA PyTorch"
}

function Pip-Requirements([string]$Python, [string]$Requirements) {
    Pip-Install $Python @("-r", $Requirements) "Installing $([IO.Path]::GetFileName($Requirements))"
}

# npm ci when a lockfile exists, npm install otherwise; skipped when node_modules is newer than the lockfile.
function Ensure-NodeModules([string]$Dir = $script:AppRoot) {
    $npm = Require-Command "npm.cmd" "Install Node.js (LTS) from nodejs.org, then run Setup again."
    $nm = Join-Path $Dir "node_modules"
    $lock = Join-Path $Dir "package-lock.json"
    # Whether the packages RESOLVE decides, not timestamps: a folder's date says nothing about its
    # contents, and a needless `npm ci` deletes node_modules first -- with the app running, Windows
    # refuses half the deletes (EPERM) and the app is left without packages (2026-10-26, claudeAPI).
    if ((Test-Path $nm) -and (Test-Native $npm @("ls", "--omit=dev", "--depth=0", "--prefix", $Dir))) {
        Write-SetupLog "node_modules present and complete: $nm"; return
    }
    if (Test-Path $lock) { Invoke-Native "npm ci" $npm @("ci", "--no-audit", "--no-fund") $Dir }
    else { Invoke-Native "npm install" $npm @("install", "--no-audit", "--no-fund") $Dir }
}

function Done([string]$Message = "Environment ready.") { Write-SetupLog $Message; Release-SetupLock; exit 0 }
