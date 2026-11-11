const { app, BrowserWindow, Tray, Menu, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const plan = require('./port-plan.cjs');
const { looksLikeBrokenEnv, diedAtStartup } = require('./comfy-crash.cjs');

// Dexter drives this window through Windows UI Automation. Chromium only
// publishes the page to it when a screen reader is around, so ask for it
// outright: every labelled control is then visible to Dexter by name.
app.commandLine.appendSwitch('force-renderer-accessibility');

let mainWindow = null;
let tray = null;
let backendProcess = null;
let comfyProcess = null;

// ─────────────────────────── preferred ports ───────────────────────────
//
// GUARDIAN_PLAN.md section 11: a port is a *wish*, never a fact. Not one is
// written down in this file. Everything below — the renderer, the backend,
// ComfyUI, the health poll, the ledger entry — uses the port that was actually
// taken, which is discovered or picked at launch and may be anywhere in
// preferred … preferred+span.
//
// The three preferred numbers (BUZZEDIT_PORT, COMFYUI_PORT, BUZZEDIT_VITE_PORT)
// and the "where is it / where shall we put it" decisions live in
// electron/port-plan.cjs, which imports no Electron and so can be unit-tested
// without launching the app — and, with it, ComfyUI and the GPU.
const { HOST, getJson, preferredVitePort, resolveBackend, resolveComfy } = plan;
// The shared cross-language helper (scripts/buzzcaf-ports.mjs), lazily imported.
const ports = plan.ports;

// Filled in during startup, then treated as read-only.
let backendPort = null;
let comfyPort = null;
let apiUrl = null;
let comfyUrl = null;
// True when *we* spawned the backend, i.e. when we own its ledger entry. An
// already-running backend was published by whoever started it.
let ownsLedgerEntry = false;

// ─────────────────────── backend supervisor state ───────────────────────
//
// The backend used to exit and stay dead: the 'close' handler only logged it,
// so a crash — an unhandled exception, an OOM, a GPU driver reset mid-render —
// left Electron alive with no backend, and the Studio's Music page waited on
// it forever (2026-10-20). See the 'close' handler below and restartBackend().
const BACKEND_RESTART_DELAYS_MS = [3000, 10000, 30000];
const BACKEND_RESTART_STEADY_MS = 60000;
const BACKEND_STABLE_MS = 10 * 60 * 1000; // up this long resets the backoff
let backendRestartTimer = null;
let backendRestartAttempts = 0;
let backendStartedAt = 0; // 0 while down; set once waitForBackend() succeeds

// ComfyUI (the "AI Engine") used to be launched by START_APP.bat with
// `start /min`, which left a console window of its own in the taskbar and
// outlived the app. It is owned here instead: spawned hidden, and killed on
// quit — but only when we were the ones who started it, so a ComfyUI the user
// is running themselves is left alone.
//
// This is the Desktop install (cu12). The old portable build was cu118 against
// everything else here being cu12, which is what threw "the procedure entry
// point could not be located in the dynamic link library".
const COMFY_CODE = process.env.COMFY_CODE || 'C:/Users/singh/ComfyUI-Installs/ComfyUI/ComfyUI';
const COMFY_DATA = process.env.COMFY_DATA || 'C:/Users/singh/Documents/ComfyUI';
const COMFY_PYTHON = process.env.COMFY_PYTHON || path.join(COMFY_DATA, '.venv', 'Scripts', 'python.exe');

// Started for a background render (Buzzcaf Studio sets BUZZEDIT_HEADLESS=1 when
// it auto-launches us), the window is created but never shown: the tray icon's
// "Show" -- or a click on it -- brings it up if anyone wants to look.
const HEADLESS = process.env.BUZZEDIT_HEADLESS === '1' || process.argv.includes('--headless');

// Backend and ComfyUI output used to go only to this process's console, which
// nobody sees under the silent launcher -- a render that died left no trace of
// FFmpeg's stderr anywhere. Append it to logs\backend.log as well.
const backendLogPath = path.join(__dirname, '../logs/backend.log');
let backendLogStream = null;
function logLine(prefix, data) {
  const text = `${prefix}${data}`;
  if (prefix.startsWith('Backend err') || prefix.startsWith('ComfyUI err')) console.error(text);
  else console.log(text);
  try {
    if (!backendLogStream) {
      fs.mkdirSync(path.dirname(backendLogPath), { recursive: true });
      backendLogStream = fs.createWriteStream(backendLogPath, { flags: 'a' });
    }
    const stamp = new Date().toISOString();
    backendLogStream.write(`${stamp} ${text}${text.endsWith('\n') ? '' : '\n'}`);
  } catch (err) {
    // Logging must never take the app down.
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    show: !HEADLESS,
    // Low enough for a half-screen snap on a 1080p display (960px) — with the
    // old 1000px floor Windows let the window run past the screen edge and its
    // right side was cut off. The layout fits itself down to this size.
    minWidth: 640,
    minHeight: 480,
    frame: false,
    backgroundColor: '#0f0f0f',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.cjs'),
      // How the renderer learns which ports were actually taken. It cannot read
      // the ledger (no fs) and it cannot derive the API base from
      // window.location.origin either, because in production it is loaded from
      // file://. Passing the URLs as switches makes them available to the
      // preload *synchronously*, before the first module of the bundle runs —
      // an async IPC round-trip would leave the very first fetch with nothing.
      additionalArguments: [
        `--buzzedit-api-url=${apiUrl}`,
        `--buzzedit-comfy-url=${comfyUrl}`,
      ],
    },
  });

  const isDev = process.env.NODE_ENV === 'development';
  if (isDev) {
    // The dev server's port is picked by scripts/dev.mjs and handed down here;
    // the preferred value is only a fallback for `electron .` run by hand.
    mainWindow.loadURL(process.env.BUZZEDIT_DEV_URL || `http://${HOST}:${preferredVitePort()}`);
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function createTray() {
  const iconPath = path.join(__dirname, '../build/icon.png');
  if (fs.existsSync(iconPath)) {
    tray = new Tray(iconPath);
  } else {
    tray = new Tray(createFallbackIcon());
  }

  const contextMenu = Menu.buildFromTemplate([
    { label: 'Show', click: () => mainWindow?.show() },
    { label: 'Hide', click: () => mainWindow?.hide() },
    { type: 'separator' },
    { label: 'Quit', click: () => app.quit() },
  ]);

  tray.setToolTip('BuzzEdit');
  tray.setContextMenu(contextMenu);
  tray.on('click', () => {
    if (mainWindow.isVisible()) {
      mainWindow.hide();
    } else {
      mainWindow.show();
    }
  });
}

function createFallbackIcon() {
  const { nativeImage } = require('electron');
  const img = nativeImage.createEmpty();
  return img;
}

// Force-kills a process and its children. `taskkill /T` walks the process tree,
// which `child.kill()` does not do on Windows (grandchildren would leak). It is
// only ever pointed at a child *we* spawned — never at whoever holds a port.
function killProcessTree(pid) {
  return new Promise((resolve) => {
    if (!pid) return resolve();
    if (process.platform === 'win32') {
      const { execFile } = require('child_process');
      execFile('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true }, () => resolve());
    } else {
      try { process.kill(pid, 'SIGKILL'); } catch { /* already gone */ }
      resolve();
    }
  });
}

// ─────────────────────── ComfyUI environment repair ───────────────────────
//
// scripts/comfy_repair.py checks (and, when broken, fixes) the Python
// environment ComfyUI runs in -- see its own docstring for why. It already
// checks first and only repairs when that check fails, so every call here
// just runs it and waits; "nothing to do" and "fixed it" both exit 0.
//
// Single-flight: a missing COMFY_PYTHON at startup, a crash whose stderr
// tail looks like a broken environment, and three crashes inside the restart
// window can all ask for a repair around the same time. They share one
// promise so only one repair process ever runs, and the watchdog/restart
// paths check the same flag so they never start ComfyUI, or a second repair,
// while one is in flight.
let comfyRepairPromise = null;
let comfyLastRepairAttempt = 0; // Date.now() of the last attempt (0 = never)
// Exit timestamps, independent of whether a restart was actually scheduled --
// used only to notice "crashed 3 times in a row" so ensureComfyEnv() runs
// before trying a 4th time. Declared here, ahead of startComfyUI's use of it.
let comfyExitTimes = [];
const COMFY_REPAIR_RETRY_MS = 60 * 60 * 1000; // a failed repair: at most once/hour
// How long a crash still counts as "at startup" for looksLikeBrokenEnv() below
// (electron/comfy-crash.cjs) — a broken environment dies right away, not
// twenty minutes into a render, and that module also drops the custom-node /
// optional-backend noise a healthy start itself prints before the pattern runs.
const COMFY_STARTUP_CRASH_MS = 180 * 1000;
const COMFY_REPAIR_SCRIPT = path.join(__dirname, '../scripts/comfy_repair.py');
const COMFY_REPAIR_STATE_PATH = path.join(COMFY_DATA, '.venv_repair.json');
const COMFY_STATUS_PATH = path.join(__dirname, '../data/comfy_status.json');

function readComfyRepairState() {
  try {
    return JSON.parse(fs.readFileSync(COMFY_REPAIR_STATE_PATH, 'utf-8'));
  } catch (err) {
    return null;
  }
}

function writeJsonAtomic(filePath, payload) {
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    const tmp = `${filePath}.tmp-${process.pid}-${Date.now()}`;
    fs.writeFileSync(tmp, JSON.stringify(payload, null, 2), 'utf-8');
    fs.renameSync(tmp, filePath);
  } catch (err) {
    console.warn('Could not write ComfyUI status file:', err);
  }
}

// data/comfy_status.json: our own account of ComfyUI, for the backend's
// health routes to read (it has no window into this process). Written
// atomically (tmp + rename) every time the state actually changes.
let comfyStatusState = null;
let comfyStatusSince = null;

function setComfyStatus(state, detail) {
  const now = new Date().toISOString();
  if (state !== comfyStatusState) {
    comfyStatusState = state;
    comfyStatusSince = now;
  }
  writeJsonAtomic(COMFY_STATUS_PATH, {
    state: comfyStatusState,
    since: comfyStatusSince,
    detail: detail || '',
    repair: readComfyRepairState(),
    updated_at: now,
  });
}

// Spawns comfy_repair.py hidden, streaming every line it prints through
// logLine('ComfyUI repair: ', ...) so a repair is visible in logs/backend.log
// as it happens, not just at the end.
function spawnComfyRepair(pythonPath, extraArgs) {
  return new Promise((resolve) => {
    const args = [COMFY_REPAIR_SCRIPT, '--data', COMFY_DATA, '--code', COMFY_CODE,
      '--log', backendLogPath, ...extraArgs];
    let child;
    try {
      child = spawn(pythonPath, args, {
        cwd: path.join(__dirname, '..'),
        env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });
    } catch (err) {
      resolve({ code: 1, lastLine: '', error: String(err) });
      return;
    }
    let lastLine = '';
    const onData = (d) => {
      const text = d.toString();
      logLine('ComfyUI repair: ', text);
      const lines = text.split(/\r?\n/).filter(Boolean);
      if (lines.length) lastLine = lines[lines.length - 1];
    };
    child.stdout?.on('data', onData);
    child.stderr?.on('data', onData);
    child.on('close', (code) => resolve({ code: code == null ? 1 : code, lastLine }));
    child.on('error', (err) => resolve({ code: 1, lastLine: '', error: String(err) }));
  });
}

// Runs the repair script (which checks first, and only truly repairs when
// broken) and returns whether the environment is healthy afterwards. Never
// throws. `reason` is only for the log line.
function ensureComfyEnv(reason) {
  if (comfyRepairPromise) return comfyRepairPromise;
  const now = Date.now();
  if (comfyLastRepairAttempt && now - comfyLastRepairAttempt < COMFY_REPAIR_RETRY_MS) {
    logLine('ComfyUI err: ', `repair skipped -- last attempt ${Math.round((now - comfyLastRepairAttempt) / 60000)} min ago, retried at most once/hour (${reason})`);
    return Promise.resolve(false);
  }
  comfyRepairPromise = (async () => {
    comfyLastRepairAttempt = Date.now();
    logLine('ComfyUI err: ', `environment repair starting (${reason})`);
    setComfyStatus('repairing', reason);
    let pythonPath;
    try {
      pythonPath = await findPython();
    } catch (err) {
      logLine('ComfyUI err: ', `repair could not run, no backend python: ${err}`);
      setComfyStatus('repair_failed', String(err).slice(0, 500));
      return false;
    }
    const result = await spawnComfyRepair(pythonPath, []);
    if (result.code === 0) {
      logLine('ComfyUI: ', 'environment repair finished: healthy');
      return true;
    }
    const state = readComfyRepairState();
    const errMsg = (state && state.error) || result.lastLine || result.error || `exit ${result.code}`;
    logLine('ComfyUI err: ', `environment repair failed: ${errMsg}`);
    setComfyStatus('repair_failed', String(errMsg).slice(0, 500));
    return false;
  })();
  return comfyRepairPromise.finally(() => {
    comfyRepairPromise = null;
  });
}

// A one-off, read-only `--check` run at app start, purely to get the state of
// the environment into the log before anything goes wrong with it. It never
// repairs -- a broken environment found here is left for the crash path
// (ensureComfyEnv, below) to fix.
async function backgroundComfyCheck() {
  if (!fs.existsSync(COMFY_PYTHON)) return; // startComfyUI's own repair path handles this
  try {
    const pythonPath = await findPython();
    const result = await spawnComfyRepair(pythonPath, ['--check']);
    if (result.code === 0) {
      logLine('ComfyUI: ', 'startup environment check: healthy');
    } else {
      logLine('ComfyUI err: ', `startup environment check: broken (${result.lastLine || result.error || `exit ${result.code}`}); will repair on the next crash`);
    }
  } catch (err) {
    logLine('ComfyUI err: ', `startup environment check could not run: ${err}`);
  }
}

// ───────────────────────────── the children ─────────────────────────────

async function startComfyUI(adopt) {
  if (adopt) return;
  // A repair already in flight will start ComfyUI itself once it resolves;
  // starting it here too would race two copies onto the same port.
  if (comfyRepairPromise) return;
  if (!fs.existsSync(COMFY_PYTHON)) {
    logLine('ComfyUI err: ', `python not found at ${COMFY_PYTHON}; repairing the environment`);
    setComfyStatus('down', 'python not found');
    const healthy = await ensureComfyEnv('python not found at startup');
    if (quitting || !healthy) return;
    if (!fs.existsSync(COMFY_PYTHON)) {
      logLine('ComfyUI err: ', 'repair finished but python is still missing; not starting ComfyUI');
      return;
    }
  }

  setComfyStatus('starting', 'launching ComfyUI');
  // A fresh process gets a full silence window of its own. Without this, a
  // start that follows a long repair (30+ min) inherits the old "last answered"
  // time and the watchdog kills it 30 s later, before it has loaded.
  comfyLastAnswer = Date.now();
  comfySpawnedAt = Date.now();
  // Reset per process: only this spawn's own stderr should decide whether its
  // own exit looks like a broken environment.
  const stderrTail = [];
  comfyProcess = spawn(
    COMFY_PYTHON,
    // ComfyUI's own main.py takes --listen/--port; this is how the port we
    // picked is handed down to the sub-service (rule 7).
    // Memory (creator rule 2026-10-26: the PC must not lag or hang): ComfyUI
    // drops its cache once free RAM falls under 6 GB (default ~3 GB), and pins
    // no RAM -- 13 GB of pinned memory Windows cannot page out filled the
    // commit limit and crashed ComfyUI and the Studio window that night.
    ['main.py', '--base-directory', COMFY_DATA, '--listen', HOST, '--port', String(comfyPort),
      '--cache-ram', '6', '--disable-pinned-memory'],
    {
      cwd: COMFY_CODE,
      // PYTHONIOENCODING: ComfyUI logs emoji, and on the cp1252 codepage the
      // logging call itself throws UnicodeEncodeError and takes the process down.
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    },
  );
  const child = comfyProcess;
  child.stdout?.on('data', (d) => logLine('ComfyUI: ', d));
  child.stderr?.on('data', (d) => {
    logLine('ComfyUI err: ', d);
    const lines = d.toString().split(/\r?\n/).filter(Boolean);
    stderrTail.push(...lines);
    if (stderrTail.length > 60) stderrTail.splice(0, stderrTail.length - 60);
  });
  child.on('close', (code, signal) => {
    logLine('ComfyUI: ', `exited with code ${code}${signal ? ` (signal ${signal})` : ''}`);
    if (comfyProcess === child) comfyProcess = null;
    // Nothing stops ComfyUI on purpose except quitting the app, so any other
    // exit is a crash (or a sleep/wake that took the GPU context with it). A
    // production waiting on it would otherwise sit out a timeout per asset.
    if (quitting) return;
    setComfyStatus('down', `exited with code ${code}${signal ? ` (signal ${signal})` : ''}`);
    const now = Date.now();
    comfyExitTimes = comfyExitTimes.filter((t) => now - t < COMFY_RESTART_WINDOW_MS);
    comfyExitTimes.push(now);
    const envLooksBroken = diedAtStartup(comfySpawnedAt, now, COMFY_STARTUP_CRASH_MS)
      && looksLikeBrokenEnv(stderrTail.join('\n'));
    const crashedThrice = comfyExitTimes.length >= 3;
    if (envLooksBroken || crashedThrice) {
      const reason = envLooksBroken
        ? 'stderr matched a known environment failure'
        : 'crashed 3 times within the restart window';
      ensureComfyEnv(reason).finally(() => {
        if (!quitting) scheduleComfyRestart();
      });
      return;
    }
    scheduleComfyRestart();
  });
}

// Restarts after an unexpected exit: a short pause (the port and the GPU need
// a moment), then again with a longer one, giving up after a few attempts in a
// short window so a ComfyUI that cannot start does not loop forever.
const COMFY_RESTART_DELAYS_MS = [5000, 15000, 60000];
const COMFY_RESTART_WINDOW_MS = 10 * 60 * 1000;
let comfyRestarts = [];
let comfyRestartTimer = null;

function scheduleComfyRestart() {
  // A repair in flight already restarts ComfyUI itself once it resolves.
  if (comfyRepairPromise || comfyRestartTimer || comfyProcess) return;
  const now = Date.now();
  comfyRestarts = comfyRestarts.filter((t) => now - t < COMFY_RESTART_WINDOW_MS);
  if (comfyRestarts.length >= COMFY_RESTART_DELAYS_MS.length) {
    logLine('ComfyUI err: ', `crashed ${comfyRestarts.length} times in 10 minutes -- not restarting again; restart BuzzEdit`);
    return;
  }
  const delay = COMFY_RESTART_DELAYS_MS[comfyRestarts.length];
  comfyRestarts.push(now);
  logLine('ComfyUI: ', `restarting in ${delay / 1000}s (attempt ${comfyRestarts.length})`);
  comfyRestartTimer = setTimeout(() => {
    comfyRestartTimer = null;
    if (quitting || comfyProcess || comfyRepairPromise) return;
    startComfyUI(false).catch((err) => logLine('ComfyUI err: ', `restart failed: ${err}`));
  }, delay);
}

// A ComfyUI that is alive but mute is as bad as one that crashed -- the crash
// handler above never fires for it, and a production waited an hour per asset
// on it (2026-10-20). So its /system_stats is probed every COMFY_WATCH_MS; one
// we own that has not answered for COMFY_SILENT_MS is stopped, and the 'close'
// handler restarts it. One we did not start (adopted) that goes mute that long
// is replaced by our own. And a ComfyUI that stayed down because it crashed
// three times in ten minutes is started again once that window has passed,
// instead of staying down until someone restarts BuzzEdit.
const COMFY_WATCH_MS = 30 * 1000;
const COMFY_SILENT_MS = 10 * 60 * 1000;
let comfyAdopted = false;
let comfyLastAnswer = Date.now();
let comfySpawnedAt = 0; // set by startComfyUI(); read by the 'close' handler above
let comfyGiveUpNoted = false;

async function comfyWatchTick() {
  // While a repair is in flight ComfyUI is expected to be down; let
  // ensureComfyEnv() finish and start it back up rather than racing it.
  if (quitting || !comfyPort || comfyRepairPromise) return;
  const stats = await getJson(`${comfyUrl}/system_stats`, 10000);
  const now = Date.now();
  if (stats && stats.system) {
    comfyLastAnswer = now;
    comfyGiveUpNoted = false;
    if (comfyStatusState !== 'running') setComfyStatus('running', 'answered /system_stats');
    return;
  }
  const silentMs = now - comfyLastAnswer;
  if (comfyProcess) {
    if (silentMs < COMFY_SILENT_MS) return;
    logLine('ComfyUI err: ', `has not answered for ${Math.round(silentMs / 60000)} min; stopping it so it restarts`);
    comfyLastAnswer = now; // the restarted one gets a full window of its own
    try {
      comfyProcess.kill();
    } catch (err) {
      logLine('ComfyUI err: ', `could not stop it: ${err}`);
    }
    return; // the 'close' handler schedules the restart
  }
  if (comfyRestartTimer) return;
  if (comfyAdopted) {
    if (silentMs < COMFY_SILENT_MS) return;
    logLine('ComfyUI: ', `the ComfyUI we adopted on ${comfyPort} has not answered for ${Math.round(silentMs / 60000)} min; starting our own`);
    comfyAdopted = false;
    comfyLastAnswer = now;
    startComfyUI(false).catch((err) => logLine('ComfyUI err: ', `start failed: ${err}`));
    return;
  }
  const recent = comfyRestarts.filter((t) => now - t < COMFY_RESTART_WINDOW_MS).length;
  if (recent >= COMFY_RESTART_DELAYS_MS.length) {
    if (!comfyGiveUpNoted) {
      comfyGiveUpNoted = true;
      logLine('ComfyUI err: ', 'still down after repeated crashes; trying again once ten minutes have passed');
    }
    return;
  }
  comfyGiveUpNoted = false;
  logLine('ComfyUI: ', 'is down; starting it again');
  scheduleComfyRestart();
}

function startComfyWatchdog(adopt) {
  comfyAdopted = Boolean(adopt);
  comfyLastAnswer = Date.now();
  const watch = setInterval(() => {
    comfyWatchTick().catch((err) => logLine('ComfyUI err: ', `watchdog: ${err}`));
  }, COMFY_WATCH_MS);
  watch.unref();
}

async function startBackend(adopt) {
  if (adopt) return;

  // Note what is *not* here any more: the old launcher looked up whoever held
  // 8099 and killed it (`freePort`/`findPidsOnPort`). Rule 2 forbids that — a
  // busy port is stepped over, never evicted. `resolveBackend()` has already
  // found a port nobody holds.

  const backendPath = path.join(__dirname, '../backend');
  const mainPy = path.join(backendPath, 'main.py');

  if (!fs.existsSync(mainPy)) {
    console.error('Backend not found:', mainPy);
    return;
  }

  try {
    const pythonPath = await findPython();
    const child = spawn(pythonPath, ['main.py', '--port', String(backendPort)], {
      cwd: backendPath,
      // PYTHONIOENCODING: the backend logs non-ASCII (Devanagari captions, emoji
      // from third-party libs) and the default cp1252 codepage makes the logging
      // call itself throw, taking the server down.
      //
      // COMFYUI_URL: the sub-service port travels down by env, so the backend's
      // bridge talks to the ComfyUI we actually started rather than the default.
      //
      // BUZZEDIT_PORT: the health route reports the port it was told to take, so
      // the identity payload matches reality even before the socket is up.
      //
      // BUZZEDIT_SUPERVISED: this supervisor restarts the backend on any exit,
      // so runtime/code_reload.py may exit an idle backend onto changed code.
      env: {
        ...process.env,
        PYTHONPATH: backendPath,
        PYTHONIOENCODING: 'utf-8',
        COMFYUI_URL: comfyUrl,
        BUZZEDIT_PORT: String(backendPort),
        BUZZEDIT_SUPERVISED: '1',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
      // The app launches without a console of its own, so python.exe — a console
      // subsystem binary — would allocate a visible one. The user sees a black
      // cmd window next to the UI. stdout/stderr stay piped here regardless.
      windowsHide: true,
    });
    backendProcess = child;

    child.stdout?.on('data', (data) => {
      logLine('Backend: ', data);
    });

    child.stderr?.on('data', (data) => {
      logLine('Backend err: ', data);
    });

    // The supervisor: only an actual exit of *this* child ever restarts
    // anything — never a slow or failed health check, which waitForBackend()
    // below handles on its own and which a render in progress must never pay
    // for with a kill.
    child.on('close', (code, signal) => {
      if (backendProcess === child) backendProcess = null;
      if (quitting) return;
      onBackendExit(code, signal).catch((err) => logLine('Backend err: ', `exit handling failed: ${err}`));
    });

    await waitForBackend();
    console.log(`Backend started successfully on ${apiUrl}`);
    ownsLedgerEntry = true;
    backendStartedAt = Date.now();
  } catch (err) {
    console.error('Failed to start backend:', err);
  }
}

/** The backend's own 'close' handler, once we know we are not quitting. */
async function onBackendExit(code, signal) {
  logLine('Backend err: ', `backend exited with code ${code}${signal ? ` (signal ${signal})` : ''}; restarting`);
  // Still us, but now a pid with nothing listening behind it: left in the
  // ledger it strands every app that discovers 'buzzedit' the way the
  // Studio's Music page did on 2026-10-20, because main.py's own atexit
  // never got to run — it was the one just killed. We are still alive, so we
  // take it back ourselves rather than leaving it for a `finally` that will
  // never fire -- and we do it before scheduling the restart, so nothing can
  // discover the dead entry in between.
  try {
    await withdrawPorts();
  } catch (err) {
    logLine('Backend err: ', `could not withdraw the stale ledger entry: ${err}`);
  }
  if (backendStartedAt && Date.now() - backendStartedAt >= BACKEND_STABLE_MS) {
    backendRestartAttempts = 0; // it earned a fresh run at the short delays
  }
  backendStartedAt = 0;
  scheduleBackendRestart();
}

function scheduleBackendRestart() {
  if (quitting || backendRestartTimer || backendProcess) return;
  const delay = backendRestartAttempts < BACKEND_RESTART_DELAYS_MS.length
    ? BACKEND_RESTART_DELAYS_MS[backendRestartAttempts]
    : BACKEND_RESTART_STEADY_MS;
  backendRestartAttempts += 1;
  logLine('Backend: ', `restarting in ${delay / 1000}s (attempt ${backendRestartAttempts})`);
  backendRestartTimer = setTimeout(() => {
    backendRestartTimer = null;
    if (quitting || backendProcess) return;
    restartBackend().catch((err) => logLine('Backend err: ', `restart failed: ${err}`));
  }, delay);
}

/**
 * Brings the backend back after an unexpected exit — never gives up.
 *
 * `resolveBackend()` is the same helper `app.whenReady()` used for the very
 * first launch: it tries BuzzEdit's preferred port first (where the backend
 * almost always still is, since it was free when we parked there) and only
 * steps to another free one in the span if something else has taken it in
 * the meantime — rule 2, never evict. The ComfyUI URL we hand the restarted
 * backend is untouched; only the backend's own port can move.
 */
async function restartBackend() {
  const oldPort = backendPort;
  const backend = await resolveBackend();
  backendPort = backend.port;
  apiUrl = `http://${HOST}:${backendPort}`;
  const changedPort = backendPort !== oldPort;
  if (changedPort) {
    logLine('Backend: ', `port ${oldPort} is held by something else now; restarting on ${backendPort} instead`);
  }
  await startBackend(backend.adopt);
  await publishPorts();
  if (changedPort) followNewBackendUrl();
}

/**
 * Makes the UI follow a backend that landed on a different port.
 *
 * The renderer only ever learns the API base once, from `additionalArguments`
 * at window construction (see `createWindow()`) — there is no live channel to
 * tell an already-open page its base changed. Recreating the window is the
 * same trick `createWindow()` already uses at first launch, just run again.
 */
function followNewBackendUrl() {
  const old = mainWindow;
  if (!old) {
    createWindow();
    return;
  }
  // createWindow() already has an `on('closed', ...)` that clears `mainWindow`
  // — registered first, so it runs before this one — so waiting for the old
  // window's own 'closed' event (guaranteed even for a forced destroy()) is
  // what keeps a late event from wiping out the *new* window instead.
  old.once('closed', () => createWindow());
  try {
    old.destroy();
  } catch (err) {
    logLine('Backend err: ', `could not close the window for the new port: ${err}`);
  }
}

async function findPython() {
  if (process.env.PYTHON_EXECUTABLE && fs.existsSync(process.env.PYTHON_EXECUTABLE)) {
    return process.env.PYTHON_EXECUTABLE;
  }
  const venvPy = path.join(__dirname, '../.venv/Scripts/python.exe');
  if (fs.existsSync(venvPy)) {
    return venvPy;
  }
  const venvPyUnix = path.join(__dirname, '../.venv/bin/python');
  if (fs.existsSync(venvPyUnix)) {
    return venvPyUnix;
  }
  throw new Error('Project Python virtual environment (.venv) not found. Run `uv venv --python 3.11 .venv` and `uv pip install -r backend/requirements.txt`.');
}

/**
 * Wait until the health route answers *and* names itself.
 *
 * Rule 4: a 200 is not enough. Something else could have taken the port we
 * picked in the gap between our bind test and the backend's own bind, and
 * publishing that to the ledger would send every other app to the wrong door.
 */
async function waitForBackend(retries = 20) {
  for (let i = 0; i < retries; i++) {
    const body = await getJson(`${apiUrl}/api/health`, 2000);
    if (body && body.app === 'buzzedit') return body;
    if (body) console.warn(`Something answered ${apiUrl}/api/health but calls itself "${body.app}".`);
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error(`Backend failed to start on ${apiUrl}`);
}

// ───────────────────────────── the ledger ─────────────────────────────

async function publishPorts() {
  if (!ownsLedgerEntry) return;
  try {
    const { publish } = await ports();
    await publish('buzzedit', backendPort, `${apiUrl}/api/health`, { comfyui: comfyPort });
    console.log(`Published buzzedit -> ${backendPort} (comfyui ${comfyPort}).`);
  } catch (err) {
    // A ledger we cannot write is a discovery problem, not a startup problem.
    console.warn('Could not publish to the port ledger:', err);
  }
}

async function withdrawPorts() {
  if (!ownsLedgerEntry) return;
  ownsLedgerEntry = false;
  try {
    const { withdraw } = await ports();
    await withdraw('buzzedit');
  } catch (err) {
    console.warn('Could not withdraw from the port ledger:', err);
  }
}

app.whenReady().then(async () => {
  // Both ports are settled before the window exists, because the renderer is
  // handed them as command-line switches at construction time.
  const comfy = await resolveComfy();
  comfyPort = comfy.port;
  comfyUrl = `http://${HOST}:${comfyPort}`;
  // The bridge inside the backend (and anything else we spawn) uses the real one.
  process.env.COMFYUI_URL = comfyUrl;

  const backend = await resolveBackend();
  backendPort = backend.port;
  apiUrl = `http://${HOST}:${backendPort}`;
  console.log(`BuzzEdit ports: backend ${backendPort}, comfyui ${comfyPort}.`);

  // The window comes up first. Waiting on the backend here meant up to 20s of
  // nothing on screen after launch, which read as "the app didn't start" — the
  // renderer polls /api/health and can show its own connecting state.
  createWindow();
  createTray();

  startBackend(backend.adopt).then(publishPorts);
  startComfyUI(comfy.adopt).catch((err) => logLine('ComfyUI err: ', `start failed: ${err}`));
  startComfyWatchdog(comfy.adopt);
  // Read-only, non-blocking: just gets the environment's state into the log
  // before anything goes wrong with it. A broken environment found here is
  // left for the crash path to actually repair.
  backgroundComfyCheck().catch((err) => logLine('ComfyUI err: ', `startup check failed: ${err}`));

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

function stopBackend() {
  const pending = [];
  if (comfyRestartTimer) {
    clearTimeout(comfyRestartTimer);
    comfyRestartTimer = null;
  }
  if (backendRestartTimer) {
    clearTimeout(backendRestartTimer);
    backendRestartTimer = null;
  }
  if (backendProcess && backendProcess.pid) {
    // Kill the whole tree — the python launcher may have spawned children
    // (e.g. uvicorn workers) that would otherwise keep the port held.
    pending.push(killProcessTree(backendProcess.pid));
    backendProcess = null;
  }
  if (comfyProcess && comfyProcess.pid) {
    pending.push(killProcessTree(comfyProcess.pid));
    comfyProcess = null;
  }
  return Promise.all(pending);
}

let quitting = false;

// Fires on every quit path (including force-quit via the tray), so the backend
// is torn down and the ledger entry removed even when window-all-closed does
// not run. The quit is deferred exactly once: withdrawing is a file write under
// a lock, which cannot finish inside a synchronous handler.
app.on('before-quit', (event) => {
  if (quitting) return;
  quitting = true;
  event.preventDefault();
  Promise.resolve()
    .then(withdrawPorts)
    .then(stopBackend)
    .catch((err) => console.warn('Teardown problem:', err))
    .finally(() => app.quit());
});

// Started headless by the desktop Studio (BUZZCAF_STUDIO_PID): live only as
// long as it does. The Studio stops us when it closes normally, but a crash, a
// killed process or a closed console runs no cleanup, and this BuzzEdit would
// then hold Whisper, the aligner and ComfyUI's models on the GPU with nobody
// left to use them. `process.kill(pid, 0)` only tests that the pid exists (it
// never signals on Windows); EPERM means it exists and is someone else's.
//
// STUDIO_DIRECTOR_PLAN.md enhancement 4 (2026-10-24): quitting the instant the
// pid is gone killed a reel/narration mid-job, since any Studio exit (crash,
// close, the reaper) took the headless BuzzEdit it started down with it. Now:
// while the backend reports work (`GET /api/reload/busy`), the watchdog only
// polls and waits (hard cap `QUIT_HARD_CAP_MS`); a *new* Studio can adopt this
// BuzzEdit instead of spawning a second one (`POST /api/supervisor/parent`,
// routes/supervisor.py) -- `currentParentPid()` reads that pid once set,
// cancelling the pending quit.
const STUDIO_PID = Number(process.env.BUZZCAF_STUDIO_PID || 0);
const STUDIO_POLL_MS = 3000;
const BUSY_POLL_MS = 30000;
const QUIT_HARD_CAP_MS = 6 * 60 * 60 * 1000;

function studioAlive(pid) {
  if (!(pid > 0)) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err.code === 'EPERM';
  }
}

// The pid to watch: a Studio that adopted this BuzzEdit (`/api/supervisor/parent`)
// instead of starting a new one, or the env var it was started with.
async function currentParentPid() {
  if (!apiUrl) return STUDIO_PID;
  const body = await getJson(`${apiUrl}/api/supervisor/parent`, 2000);
  if (body && Number.isInteger(body.pid) && body.pid > 0) return body.pid;
  return STUDIO_PID;
}

// Whether the backend itself says it must not be restarted/quit right now.
// Unreachable/unsure counts as busy -- never quit blind on a bad poll.
async function backendBusy() {
  if (!apiUrl) return true;
  const body = await getJson(`${apiUrl}/api/reload/busy`, 5000);
  if (!body) return true;
  return Boolean(body.busy);
}

if (HEADLESS && STUDIO_PID > 0) {
  let quitPendingSince = 0;
  let lastBusyCheckAt = 0;
  let busyNow = true;
  const watch = setInterval(async () => {
    if (quitting) return;
    const parentPid = await currentParentPid();
    if (studioAlive(parentPid)) {
      if (quitPendingSince) {
        console.log(`Buzzcaf Studio (pid ${parentPid}) is back (or a new one adopted this BuzzEdit); cancelling the quit.`);
      }
      quitPendingSince = 0;
      return;
    }
    if (!quitPendingSince) {
      quitPendingSince = Date.now();
      lastBusyCheckAt = 0;
      console.log(`Buzzcaf Studio (pid ${parentPid}) is gone; waiting for BuzzEdit to be idle before quitting `
        + `(checked every ${BUSY_POLL_MS / 1000}s, hard cap ${QUIT_HARD_CAP_MS / 3600000}h).`);
    }
    if (Date.now() - quitPendingSince > QUIT_HARD_CAP_MS) {
      clearInterval(watch);
      console.log('Buzzcaf Studio has been gone for the full hard cap; quitting the headless BuzzEdit regardless of work.');
      app.quit();
      return;
    }
    const now = Date.now();
    if (now - lastBusyCheckAt >= BUSY_POLL_MS) {
      lastBusyCheckAt = now;
      busyNow = await backendBusy();
    }
    if (busyNow) return;
    clearInterval(watch);
    console.log(`Buzzcaf Studio (pid ${parentPid}) is gone and BuzzEdit is idle; quitting the headless BuzzEdit it started.`);
    app.quit();
  }, STUDIO_POLL_MS);
  watch.unref();
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

ipcMain.handle('dialog:openFile', async (_, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result;
});

ipcMain.handle('dialog:saveFile', async (_, options) => {
  const result = await dialog.showSaveDialog(mainWindow, options);
  return result;
});

ipcMain.handle('app:windowMinimize', () => {
  mainWindow?.minimize();
});

ipcMain.handle('app:windowMaximize', () => {
  if (mainWindow?.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow?.maximize();
  }
});

ipcMain.handle('app:windowClose', () => {
  mainWindow?.close();
});

ipcMain.handle('app:getApiUrl', () => apiUrl);

ipcMain.handle('app:getPorts', () => ({
  apiUrl, comfyUrl, backendPort, comfyPort,
}));

ipcMain.handle('shell:openPath', async (_, filePath) => {
  if (!filePath) return 'no path';
  // shell.openPath returns '' on success, or an error string.
  return await shell.openPath(filePath);
});

ipcMain.handle('shell:showItemInFolder', (_, filePath) => {
  if (filePath) shell.showItemInFolder(filePath);
});
