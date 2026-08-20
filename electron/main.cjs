const { app, BrowserWindow, Tray, Menu, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow = null;
let tray = null;
let backendProcess = null;

const BACKEND_PORT = 8099;
// 127.0.0.1, never "localhost". The backend binds IPv4 only (main.py), while on
// Windows "localhost" resolves to ::1 first — so every health poll from here was
// refused before it ever reached the server. The symptoms were "Backend failed
// to start" printed over a backend that had in fact started, and
// `isBackendRunning()` always answering false, which made each launch spawn a
// *second* backend to fight the first for the port. Chromium hides the same
// problem in the renderer because it retries over IPv4 by itself.
const API_URL = `http://127.0.0.1:${BACKEND_PORT}`;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    frame: false,
    backgroundColor: '#0f0f0f',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.cjs'),
    },
  });

  const isDev = process.env.NODE_ENV === 'development';
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
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

async function isBackendRunning() {
  const http = require('http');
  return new Promise((resolve) => {
    const req = http.get(`${API_URL}/api/health`, (res) => {
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(1000, () => { req.destroy(); resolve(false); });
  });
}

// Returns the PIDs of processes listening on the given TCP port (Windows).
function findPidsOnPort(port) {
  return new Promise((resolve) => {
    const { execFile } = require('child_process');
    // `netstat -ano` lists all connections with the owning PID in the last
    // column; filter to LISTENING sockets on our port.
    execFile('netstat', ['-ano', '-p', 'TCP'], { windowsHide: true }, (err, stdout) => {
      if (err || !stdout) return resolve([]);
      const pids = new Set();
      for (const line of stdout.split(/\r?\n/)) {
        const parts = line.trim().split(/\s+/);
        // e.g. ["TCP", "127.0.0.1:8099", "0.0.0.0:0", "LISTENING", "47568"]
        if (parts.length >= 5 && parts[3] === 'LISTENING') {
          const local = parts[1];
          if (local.endsWith(`:${port}`)) {
            const pid = parseInt(parts[4], 10);
            if (Number.isInteger(pid) && pid > 0) pids.add(pid);
          }
        }
      }
      resolve([...pids]);
    });
  });
}

// Kills any orphaned process still listening on the port and waits for it to
// be released, so a freshly spawned backend can bind without Errno 10048.
async function freePort(port) {
  const pids = await findPidsOnPort(port);
  if (pids.length === 0) return;

  console.warn(`Port ${port} held by stale process(es) [${pids.join(', ')}]; terminating.`);
  for (const pid of pids) {
    // Skip our own process just in case, then force-kill the whole tree.
    if (pid === process.pid) continue;
    await killProcessTree(pid);
  }

  // Wait up to ~5s for the socket to actually be released.
  for (let i = 0; i < 10; i++) {
    const remaining = await findPidsOnPort(port);
    if (remaining.length === 0) return;
    await new Promise((r) => setTimeout(r, 500));
  }
  console.warn(`Port ${port} still occupied after cleanup attempt.`);
}

// Force-kills a process and its children. `taskkill /T` walks the process tree,
// which `child.kill()` does not do on Windows (grandchildren would leak).
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

async function startBackend() {
  const alreadyRunning = await isBackendRunning();
  if (alreadyRunning) {
    console.log('Backend server is already running on port', BACKEND_PORT);
    return;
  }

  // The port may be held by a stale backend from a previous session that
  // crashed or was force-quit. It won't answer /api/health as expected, so
  // isBackendRunning() returns false, but spawning a new backend would fail
  // to bind (Errno 10048). Clear any such orphan before spawning.
  await freePort(BACKEND_PORT);

  const backendPath = path.join(__dirname, '../backend');
  const mainPy = path.join(backendPath, 'main.py');


  if (!fs.existsSync(mainPy)) {
    console.error('Backend not found:', mainPy);
    return;
  }

  try {
    const pythonPath = await findPython();
    backendProcess = spawn(pythonPath, ['main.py'], {
      cwd: backendPath,
      env: { ...process.env, PYTHONPATH: backendPath },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    backendProcess.stdout?.on('data', (data) => {
      console.log(`Backend: ${data}`);
    });

    backendProcess.stderr?.on('data', (data) => {
      console.error(`Backend err: ${data}`);
    });

    backendProcess.on('close', (code) => {
      console.log(`Backend exited with code ${code}`);
    });

    await waitForBackend();
    console.log('Backend started successfully');
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

async function waitForBackend(retries = 20) {
  const http = require('http');
  for (let i = 0; i < retries; i++) {
    try {
      await new Promise((resolve, reject) => {
        const req = http.get(`${API_URL}/api/health`, (res) => {
          if (res.statusCode === 200) resolve();
          else reject(new Error(`Status ${res.statusCode}`));
        });
        req.on('error', reject);
        req.setTimeout(2000, () => { req.destroy(); reject(new Error('Timeout')); });
      });
      return;
    } catch {
      await new Promise(r => setTimeout(r, 1000));
    }
  }
  throw new Error('Backend failed to start');
}

app.whenReady().then(async () => {
  await startBackend();
  createWindow();
  createTray();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

function stopBackend() {
  if (backendProcess && backendProcess.pid) {
    // Kill the whole tree — the python launcher may have spawned children
    // (e.g. uvicorn workers) that would otherwise keep port 8099 held.
    killProcessTree(backendProcess.pid);
    backendProcess = null;
  }
}

app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Fires on every quit path (including force-quit via tray/menu), ensuring the
// backend is torn down even when window-all-closed doesn't run.
app.on('before-quit', stopBackend);

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

ipcMain.handle('app:getApiUrl', () => {
  return API_URL;
});

ipcMain.handle('shell:openPath', async (_, filePath) => {
  if (!filePath) return 'no path';
  // shell.openPath returns '' on success, or an error string.
  return await shell.openPath(filePath);
});

ipcMain.handle('shell:showItemInFolder', (_, filePath) => {
  if (filePath) shell.showItemInFolder(filePath);
});
