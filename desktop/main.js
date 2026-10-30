/* DeepFaceLab 4.1 — Electron 外壳。
 *
 * 它只做三件事：拉起 Python 后端、等端口、开一个窗口指向它。
 * 界面全部在 app/web，换壳（Tauri / 浏览器直开）都不用动界面代码。
 */
const { app, BrowserWindow, ipcMain, shell, dialog } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const ROOT = path.resolve(__dirname, "..");
const PY_CANDIDATES = [
  path.join(ROOT, "runtime", "server", "Scripts", "pythonw.exe"),
  path.join(ROOT, "runtime", "server", "Scripts", "python.exe")
];
const PY = PY_CANDIDATES.find((p) => fs.existsSync(p)) || PY_CANDIDATES[1];
const SERVE = path.join(ROOT, "app", "serve.py");

let win = null;
let pyProc = null;

function startBackend() {
  return new Promise((resolve, reject) => {
    const logFile = path.join(ROOT, "_electron_backend.log");
    const out = fs.openSync(logFile, "w");
    pyProc = spawn(PY, [SERVE], {
      cwd: path.join(ROOT, "app"),
      env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
      stdio: ["ignore", out, out],
      windowsHide: true
    });

    let buf = "";
    const timer = setTimeout(() => reject(new Error("backend timeout")), 25000);

    const tail = () => {
      try {
        return fs.readFileSync(logFile, "utf8");
      } catch (e) {
        return "";
      }
    };

    pyProc.on("exit", (code) => {
      clearTimeout(timer);
      if (code !== 0) reject(new Error("backend exited " + code + "\n" + tail()));
    });

    const poll = setInterval(() => {
      const text = tail();
      const m = text.match(/PORT=(\d+)/);
      if (m) {
        clearInterval(poll);
        clearTimeout(timer);
        resolve(Number(m[1]));
      } else if (text.includes("Traceback")) {
        clearInterval(poll);
        clearTimeout(timer);
        reject(new Error(text));
      }
    }, 200);
  });
}

async function createWindow(port) {
  win = new BrowserWindow({
    width: 1180,
    height: 760,
    minWidth: 1040,
    minHeight: 680,
    frame: false,
    icon: path.join(__dirname, "app.ico"),
    backgroundColor: "#ffffff",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  /* ready-to-show 在渲染进程首次绘制后才会触发；页面若卡住窗口就永远不显示。
     用「加载完成」和「兜底 3 秒」双保险。 */
  let shown = false;
  const showOnce = () => {
    if (shown) return;
    shown = true;
    try { win.show(); } catch (e) {}
  };
  win.once("ready-to-show", showOnce);
  win.webContents.once("did-finish-load", showOnce);
  setTimeout(showOnce, 3000);
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
  await win.loadURL("http://127.0.0.1:" + port + "/");
}

const THEME_BG = { dark: "#12121b", light: "#ffffff" };
ipcMain.on("win:theme", (_e, t) => {
  try { if (win) win.setBackgroundColor(THEME_BG[t] || "#ffffff"); } catch (err) {}
});

/* 主题切换用的真·位图快照：capturePage 出来的 PNG 喂给 mask-image 绝对可靠，
   不会有任何图层逃出裁剪（DOM 克隆那套在 Chromium 上做不到这件事）*/
ipcMain.handle("win:shot", async () => {
  if (!win) return "";
  try {
    const img = await win.webContents.capturePage();
    return img.toDataURL();
  } catch (e) {
    return "";
  }
});

ipcMain.handle("win:open", (_e, url) => {
  try {
    if (typeof url === "string" && /^https?:\/\//i.test(url)) shell.openExternal(url);
  } catch (err) {}
});

ipcMain.handle("win:minimize", () => win && win.minimize());
ipcMain.handle("win:maximize", () => {
  if (!win) return;
  win.isMaximized() ? win.unmaximize() : win.maximize();
});
ipcMain.handle("win:close", () => win && win.close());

app.setAppUserModelId("com.caijiplayer.deepfacelab");

ipcMain.handle("dialog:file", async (e, title, filters) => {
  const r = await dialog.showOpenDialog(win, {
    title: title || "选择文件",
    properties: ["openFile"],
    filters: filters || [
      { name: "视频", extensions: ["mp4", "mov", "mkv", "avi", "webm", "m4v", "flv", "wmv"] },
      { name: "全部", extensions: ["*"] },
    ],
  });
  return r.canceled ? "" : (r.filePaths[0] || "");
});

ipcMain.handle("dialog:dir", async (e, title) => {
  const win = BrowserWindow.fromWebContents(e.sender);
  const r = await dialog.showOpenDialog(win, {
    title: title || "选择目录",
    properties: ["openDirectory", "createDirectory"]
  });
  return r.canceled ? "" : (r.filePaths[0] || "");
});

app.whenReady().then(async () => {
  try {
    const port = await startBackend();
    await createWindow(port);
  } catch (e) {
    const { dialog } = require("electron");
    dialog.showErrorBox("DeepFaceLab 启动失败", String(e.message || e));
    app.quit();
  }
});

function killBackend() {
  if (pyProc && !pyProc.killed) {
    try {
      pyProc.kill();
    } catch (e) {}
    pyProc = null;
  }
}

app.on("window-all-closed", () => {
  killBackend();
  app.quit();
});
app.on("before-quit", killBackend);
app.on("will-quit", killBackend);
