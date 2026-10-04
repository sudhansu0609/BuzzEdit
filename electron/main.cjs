const { app, BrowserWindow, Tray, Menu, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const plan = require('./port-plan.cjs');

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

// ───────────────────────────── the children ─────────────────────────────

async function startComfyUI(adopt) {
  if (adopt) return;
  if (!fs.existsSync(COMFY_PYTHON)) {
    console.warn('ComfyUI python not found, skipping AI engine:', COMFY_PYTHON);
    return;
  }

  comfyProcess = spawn(
    COMFY_PYTHON,
    // ComfyUI's own main.py takes --listen/--port; this is how the port we
    // picked is handed down to the sub-service (rule 7).
    ['main.py', '--base-directory', COMFY_DATA, '--listen', HOST, '--port', String(comfyPort)],
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
  child.stderr?.on('data', (d) => logLine('ComfyUI err: ', d));
  child.on('close', (code, signal) => {
    logLine('ComfyUI: ', `exited with code ${code}${signal ? ` (signal ${signal})` : ''}`);
    if (comfyProcess === child) comfyProcess = null;
    // Nothing stops ComfyUI on purpose except quitting the app, so any other
    // exit is a crash (or a sleep/wake that took the GPU context with it). A
    // production waiting on it would otherwise sit out a timeout per asset.
    if (!quitting) scheduleComfyRestart();
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
  if (comfyRestartTimer || comfyProcess) return;
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
    if (quitting || comfyProcess) return;
    startComfyUI(false).catch((err) => logLine('ComfyUI err: ', `restart failed: ${err}`));
  }, delay);
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
    backendProcess = spawn(pythonPath, ['main.py', '--port', String(backendPort)], {
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
      env: {
        ...process.env,
        PYTHONPATH: backendPath,
        PYTHONIOENCODING: 'utf-8',
        COMFYUI_URL: comfyUrl,
        BUZZEDIT_PORT: String(backendPort),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
      // The app launches without a console of its own, so python.exe — a console
      // subsystem binary — would allocate a visible one. The user sees a black
      // cmd window next to the UI. stdout/stderr stay piped here regardless.
      windowsHide: true,
    });

    backendProcess.stdout?.on('data', (data) => {
      logLine('Backend: ', data);
    });

    backendProcess.stderr?.on('data', (data) => {
      logLine('Backend err: ', data);
    });

    backendProcess.on('close', (code) => {
      console.log(`Backend exited with code ${code}`);
    });

    await waitForBackend();
    console.log(`Backend started successfully on ${apiUrl}`);
    ownsLedgerEntry = true;
  } catch (err) {
    console.error('Failed to start backend:', err);
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
  startComfyUI(comfy.adopt);

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
const STUDIO_PID = Number(process.env.BUZZCAF_STUDIO_PID || 0);
const STUDIO_POLL_MS = 3000;

function studioAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err.code === 'EPERM';
  }
}

if (HEADLESS && STUDIO_PID > 0) {
  const watch = setInterval(() => {
    if (quitting || studioAlive(STUDIO_PID)) return;
    clearInterval(watch);
    console.log(`Buzzcaf Studio (pid ${STUDIO_PID}) is gone; quitting the headless BuzzEdit it started.`);
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
