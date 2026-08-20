const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  dialogOpenFile: (options) => ipcRenderer.invoke('dialog:openFile', options),
  dialogSaveFile: (options) => ipcRenderer.invoke('dialog:saveFile', options),
  windowMinimize: () => ipcRenderer.invoke('app:windowMinimize'),
  windowMaximize: () => ipcRenderer.invoke('app:windowMaximize'),
  windowClose: () => ipcRenderer.invoke('app:windowClose'),
  getApiUrl: () => ipcRenderer.invoke('app:getApiUrl'),
  openPath: (filePath) => ipcRenderer.invoke('shell:openPath', filePath),
  showItemInFolder: (filePath) => ipcRenderer.invoke('shell:showItemInFolder', filePath),
});
