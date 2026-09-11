const { contextBridge, ipcRenderer } = require('electron');

// The ports the main process actually took, handed over as command-line
// switches (see `additionalArguments` in main.cjs). They have to be readable
// *synchronously*: `src/hooks/api.ts` resolves its API base at module scope, so
// an async `getApiUrl()` round-trip would leave the very first fetch pointing
// nowhere. GUARDIAN_PLAN.md section 11 rule 6 — a desktop shell opens the port
// the backend reports, never the one it asked for.
function switchValue(name) {
  const prefix = `--${name}=`;
  const found = process.argv.find((arg) => arg.startsWith(prefix));
  return found ? found.slice(prefix.length) : null;
}

contextBridge.exposeInMainWorld('electronAPI', {
  apiUrl: switchValue('buzzedit-api-url'),
  comfyUrl: switchValue('buzzedit-comfy-url'),
  dialogOpenFile: (options) => ipcRenderer.invoke('dialog:openFile', options),
  dialogSaveFile: (options) => ipcRenderer.invoke('dialog:saveFile', options),
  windowMinimize: () => ipcRenderer.invoke('app:windowMinimize'),
  windowMaximize: () => ipcRenderer.invoke('app:windowMaximize'),
  windowClose: () => ipcRenderer.invoke('app:windowClose'),
  getApiUrl: () => ipcRenderer.invoke('app:getApiUrl'),
  getPorts: () => ipcRenderer.invoke('app:getPorts'),
  openPath: (filePath) => ipcRenderer.invoke('shell:openPath', filePath),
  showItemInFolder: (filePath) => ipcRenderer.invoke('shell:showItemInFolder', filePath),
});
