const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("desktop", {
  minimize: () => ipcRenderer.invoke("win:minimize"),
  maximize: () => ipcRenderer.invoke("win:maximize"),
  close: () => ipcRenderer.invoke("win:close"),
  setTheme: (t) => ipcRenderer.send("win:theme", t),
  openExternal: (url) => ipcRenderer.invoke("win:open", url),
  pickDir: (title) => ipcRenderer.invoke("dialog:dir", title || "选择目录"),
  pickFile: (title, filters) => ipcRenderer.invoke("dialog:file", title || "选择文件", filters),
  /* Electron 32+ 起 File.path 被移除，取真实路径必须走 webUtils */
  pathForFile: (file) => {
    try { return webUtils.getPathForFile(file) || ""; } catch (e) { return ""; }
  },
  /* 截图（主题切换的圆形揭幕用） */
  snapshot: () => ipcRenderer.invoke("win:shot"),
  isDesktop: true
});
