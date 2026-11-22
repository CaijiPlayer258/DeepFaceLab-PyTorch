const { contextBridge, ipcRenderer, webUtils } = require("electron");

/* 拖放的真实路径。
   webUtils.getPathForFile 必须拿到**渲染进程里那个 File 本身**：
   页面把 File 通过 contextBridge 传进来时已经被序列化过，webUtils 拿到的是拷贝，
   返回空字符串 —— 这正是「拖进去提示无法读取路径」的原因。
   所以在 preload（同一个渲染进程）的捕获阶段先记下来，页面只来取结果。 */
let lastDropPath = "";
window.addEventListener("drop", function (e) {
  try {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    lastDropPath = f ? (webUtils.getPathForFile(f) || "") : "";
  } catch (err) { lastDropPath = ""; }
}, true);

contextBridge.exposeInMainWorld("desktop", {
  minimize: () => ipcRenderer.invoke("win:minimize"),
  maximize: () => ipcRenderer.invoke("win:maximize"),
  close: () => ipcRenderer.invoke("win:close"),
  setTheme: (t) => ipcRenderer.send("win:theme", t),
  openExternal: (url) => ipcRenderer.invoke("win:open", url),
  pickDir: (title) => ipcRenderer.invoke("dialog:dir", title || "选择目录"),
  pickFile: (title, filters) => ipcRenderer.invoke("dialog:file", title || "选择文件", filters),
  /* 最近一次拖放进来的路径（preload 捕获阶段就取好，见文件头） */
  lastDropPath: () => lastDropPath,
  /* Electron 32+ 起 File.path 被移除，取真实路径必须走 webUtils（仅对同进程 File 有效） */
  pathForFile: (file) => {
    try { return webUtils.getPathForFile(file) || ""; } catch (e) { return ""; }
  },
  /* 截图（主题切换的圆形揭幕用） */
  snapshot: () => ipcRenderer.invoke("win:shot"),
  isDesktop: true
});
