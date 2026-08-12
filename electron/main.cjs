const { app, BrowserWindow, Tray, Menu, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow = null;
let tray = null;
let backendProcess = null;

const BACKEND_PORT = 8099;
const API_URL = `http://localhost:${BACKEND_PORT}`;

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

  tray.setToolTip('Buzzcaf Editor');
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

async function startBackend() {
  const alreadyRunning = await isBackendRunning();
  if (alreadyRunning) {
    console.log('Backend server is already running on port', BACKEND_PORT);
    return;
  }

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
  const portablePy = 'B:\\ComfyUI_windows_portable_nvidia_cu118_or_cpu\\ComfyUI_windows_portable\\python_embeded\\python.exe';
  if (fs.existsSync(portablePy)) {
    return portablePy;
  }
  const candidates = ['python', 'python3', 'py', 'python.exe'];
  for (const cmd of candidates) {
    try {
      const { execSync } = require('child_process');
      execSync(`${cmd} --version`, { stdio: 'pipe' });
      return cmd;
    } catch {}
  }
  return 'python';
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

app.on('window-all-closed', () => {
  if (backendProcess) {
    backendProcess.kill();
  }
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

ipcMain.handle('app:getApiUrl', () => {
  return API_URL;
});
