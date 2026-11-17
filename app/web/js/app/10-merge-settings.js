  /* ============================================================
     合成视图 · 检测框 / 每张脸的模型分配 / 参数实时预览 / 加进人脸库
     ------------------------------------------------------------
     以前「检测」和「合成」是两个页面级视图，互相切换：看框的时候看不到换脸
     结果，看结果时又没有框。现在只有一个画面（换脸结果），框只是叠加层，
     由「框」开关控制显隐；每个框右上角外侧是这张脸用哪个 dfm 的 chip，
     右下角外侧是「＋ 加进人脸数据库」。

     后端契约（后端并行改，字段对不上时用旧字段兜底）：
       POST /api/merge/frame   { frame|n, video|t, settings, models, model }
       POST /api/merge/merged  { frame, settings, models }
       settings = { ct_mode, blend, mask_mode, mask_erode, mask_blur }
       models   = [{ name, faces: [脸序号] }]        ← 沿用导出协议
       POST /api/merge/faces/add     { project?, kps68, box, dfm, frame, source }
                                        ← 点「＋」走这条：已知是哪张脸，68 点仿射
                                          直接落盘，不重检测、不要求有项目
       POST /api/merge/faces/extract { project, frame, dfm }   ← 整帧提取（老接口）
     ============================================================ */
  var MG_COLOR_CARD = ["#e8590c", "#1c7ed6", "#2f9e44", "#a61e4d",
                       "#7048e8", "#0c8599", "#d6336c", "#f08c00"];
  var MG_COLORS = {};          /* dfm 名 → 颜色；同一模型任何一帧都是这个色 */
  var MG_COLORS_KEY = "dfl-merge-model-colors";
  var MG_FACE_MODEL = {};      /* 脸序号 → dfm 名；**用户手动指定**的（会随请求发回去） */
  /* 脸序号 → {dfm, group, groupName, similarity, matched}：**后端按 embedding 自动判定**
     的结果（每次合成回包里的 faces）。和 MG_FACE_MODEL 分开放是有意的 ——
     这份不能当输入发回去，否则第一次判错就会被当成"用户指定"永远错下去。 */
  var MG_FACE_AUTO = {};
  var MG_FACE_NOSWAP = {};     /* 脸序号 → true：这张脸「不换脸」，请求里走 no_swap_faces */
  /* 脸序号 → true：这张脸被「点框」切成了原图（框线虚线 + 框角小标签）。
     和 MG_FACE_NOSWAP 走同一个契约键 no_swap_faces，区别只是它不改模型分配：
     用户只是想在这一帧上对比「换没换」，不是把这张脸从合成里摘出去。
     记住是**用户偏好**：切帧/切素材都不重置（存 localStorage）。 */
  var MG_FACE_RAWVIEW = {};
  var MG_FACE_RAWVIEW_KEY = "dfl-merge-face-raw";
  var MG_FACE_BUSY = {};       /* 脸序号 → true：这一张正在重算（画细进度环） */
  var MG_RECTS = [];           /* 脸序号 → 画布内矩形（chip / ＋ 的定位依据） */
  var MG_FACE_RAW = [];        /* 脸序号 → 原图坐标里的 box（提取人脸时回传） */
  var MG_CMP = { seq: 0, ctl: null, wait: null, lastMs: 0 };

  (function loadColors() {
    try {
      var raw = window.localStorage.getItem(MG_COLORS_KEY);
      if (!raw) return;
      var o = JSON.parse(raw);
      Object.keys(o || {}).forEach(function (k) {
        if (typeof o[k] === "string" && o[k]) MG_COLORS[k] = o[k];
      });
    } catch (e) { /* 无痕模式 / 关掉存储：内存里随机一份照样能用 */ }
  })();
  function mgSaveColors() {
    try { window.localStorage.setItem(MG_COLORS_KEY, JSON.stringify(MG_COLORS)); } catch (e) {}
  }

  /* 点框切「这张脸的原图」的偏好：按脸序号记，切帧/重开页面都还在 */
  (function loadRawView() {
    try {
      var raw = window.localStorage.getItem(MG_FACE_RAWVIEW_KEY);
      if (!raw) return;
      var o = JSON.parse(raw);
      Object.keys(o || {}).forEach(function (k) {
        var i = parseInt(k, 10);
        if (!isNaN(i) && o[k]) MG_FACE_RAWVIEW[i] = true;
      });
    } catch (e) { /* 无痕模式：内存里照样能用，只是不跨刷新 */ }
  })();
  function mgSaveRawView() {
    try {
      window.localStorage.setItem(MG_FACE_RAWVIEW_KEY, JSON.stringify(MG_FACE_RAWVIEW));
    } catch (e) {}
  }

  /* 固定色卡里随机抽一个；抽过就不再变（= 同一模型跨帧同色） */
  function mgModelColor(name) {
    var n = String(name || "");
    if (!n) return "";
    if (MG_COLORS[n]) return MG_COLORS[n];
    var used = {};
    Object.keys(MG_COLORS).forEach(function (k) { used[MG_COLORS[k]] = 1; });
    var free = MG_COLOR_CARD.filter(function (c) { return !used[c]; });
    var pool = free.length ? free : MG_COLOR_CARD;
    var c = pool[Math.floor(Math.random() * pool.length)];
    MG_COLORS[n] = c;
    mgSaveColors();
    return c;
  }
  window.mgModelColor = mgModelColor;

  /* 未分配的脸 = 灰色描边。以前取 --ink-3（深色主题 #6d6d78）：压在近黑的帧上
     实测对比只有 1.0~1.3:1 —— 框其实画了，肉眼和截图都看不出来。
     中间那版是 rgba(255,255,255,.55)：亮度够了，但**半透明白**压在脸上就是用户报的
     「框上有一层白蒙版」。现在用等亮度的**不透明**浅灰（#9aa0ae，压在 #1a1a1a 上仍然
     6:1），亮背景那一头由 mvDrawBoxes 里的 2px 深色外描边兜底（近白帧实测 4.4~5.0:1）。
     规矩：描边只准用不透明色 —— 框里一个像素都不许被半透明层糊到。 */
  function mgUnassignedColor() {
    return "#9aa0ae";
  }
  window.mgUnassignedColor = mgUnassignedColor;

  /* 后端自动判定的模型名（没有就是空串） */
  function mgFaceAuto(i) {
    var a = MG_FACE_AUTO[i];
    return (a && a.dfm) ? String(a.dfm) : "";
  }
  window.mgFaceAuto = mgFaceAuto;

  /* 这张脸现在用哪个 dfm：手动指定 > 后端自动判定 > 空（灰框） */
  function mgFaceModel(i) {
    if (Object.prototype.hasOwnProperty.call(MG_FACE_MODEL, i)) return String(MG_FACE_MODEL[i] || "");
    return mgFaceAuto(i);
  }
  window.mgFaceModel = mgFaceModel;

  /* 左栏「当前选中的模型」= 这一帧真正在用的那个 dfm（单选时整帧都归它）。
     0 个选中 = 空串。顺序就是列表顺序（多选时第一个是 mgCmpBody 发出去的 model）。
     **不**退回 MergePages.settings().model：那个是参数页的下拉值，用户在第 0 个选中时
     看到的应该是「还没选」，不是一个他以为自己没选过的模型。 */
  function mgFaceSelNames() {
    try { return mgSelNames() || []; } catch (e) { return []; }
  }
  window.mgFaceSelNames = mgFaceSelNames;

  /* chip 上那个模型名到底是谁（只用于上色 / 文案，不参与请求体）：
     用户手动指定 > 后端自动判定(faces[]) > 左栏选中的第一个。都没有 = 空串（灰框）。 */
  function mgFaceChipModel(i) {
    var nm = mgFaceModel(i);
    if (nm) return nm;
    if (mgFaceIsNoSwap(i)) return "";
    var a = MG_FACE_AUTO[i];
    if (a && a.matched === false) return "";
    var sel = mgFaceSelNames();
    return sel.length ? String(sel[0] || "") : "";
  }
  window.mgFaceChipModel = mgFaceChipModel;

  /* chip 上的字：手动指定 > 后端自动判定的 dfm > 不换脸 > 没匹配上 > 左栏选中的模型 > 选择模型。
     「没匹配」= 后端算完了但谁的组都不像（< 阈值），这张脸会被保持原像素。
     关键的一条：**后端没回 faces[] 不等于没有模型**。单模型预览/合成回包只带
     model / dfm（那才是这一帧真正生效的 dfm），chip 以前只认 faces[]，于是一个模型
     换得好好的画面，框上却写着「选择模型」（用户报的就是这个）。现在单选用左栏
     选中的那个名字，多选用「第一个 + 剩余个数」。 */
  function mgFaceChipText(i) {
    var nm = mgFaceModel(i);
    if (nm) return nm;
    if (mgFaceIsNoSwap(i)) return "不换脸";
    var a = MG_FACE_AUTO[i];
    if (a && a.matched === false) return "没匹配";
    var sel = mgFaceSelNames();
    if (sel.length === 1) return String(sel[0] || "");
    if (sel.length > 1) return String(sel[0] || "") + " +" + (sel.length - 1);
    return "选择模型";
  }
  window.mgFaceChipText = mgFaceChipText;

  /* 「不换脸」= 契约里的 no_swap_faces：后端会把这张脸原样留在帧里（连一个像素都
     不动），而不是"没分配 → 兜给第 0 个模型"。序号就是检测顺序，从 0 起。 */
  function mgFaceIsNoSwap(i) { return !!MG_FACE_NOSWAP[i]; }
  window.mgFaceIsNoSwap = mgFaceIsNoSwap;

  /* 这张脸现在显示的是「原图」（点框切过来的） */
  function mgFaceIsRawView(i) { return !!MG_FACE_RAWVIEW[i]; }
  window.mgFaceIsRawView = mgFaceIsRawView;

  /* 请求体的 no_swap_faces = 「不换脸」的脸 ∪ 「点成原图」的脸。
     两件事在契约上是同一件：这张脸保持原像素。 */
  function mgFaceNoSwapIds() {
    var out = {}, k;
    [MG_FACE_NOSWAP, MG_FACE_RAWVIEW].forEach(function (m) {
      Object.keys(m || {}).forEach(function (key) {
        var i = parseInt(key, 10);
        if (!isNaN(i) && m[key]) out[i] = true;
      });
    });
    return Object.keys(out).map(function (x) { return parseInt(x, 10); })
      .filter(function (i) { return !isNaN(i); })
      .sort(function (x, y) { return x - y; });
  }
  window.mgFaceNoSwapIds = mgFaceNoSwapIds;
  /* 换素材/换帧不影响这里（和 MG_FACE_MODEL 一样是"这一轮合成"的设置） */

  /* 检测框始终显示：它是「点框切这张脸的合成/原图」的唯一入口，界面上没有开关。
     函数留着是因为调用点一大片（都在问"要不要画框"），现在答案恒为"要"。 */
  function mgBoxesOn() { return true; }
  window.mgBoxesOn = mgBoxesOn;

  /* 视图切换（合成 / 调试）已经删掉：合成视图就是唯一视图。
     右上角那个 #mgViewSwitch、MG_VIEW 状态、按视图预热与缓存，全都不要了 ——
     现在合成只有一条路（mgCmpNow），每一发回包只贴当前这一帧。
     「调试」下沉到**每张脸**：框上的「调试」按钮打开那张脸的六宫格
     （见 mgFaceDebug* / /api/merge/face-debug），舞台不动、底图不动。 */


  /* 一次请求的底稿：settings / models / 帧号 / 不换脸的脸，全在这儿。
     合成视图只有一种，所以这里就是**完整**的请求体（六宫格那条接口拿它去掉
     idx 再补一个 face 就是自己的 body，参数永远和画面一致）。 */
  function mgCmpBody(frame) {
    frame = Math.max(0, parseInt(frame, 10) || 0);
    var mv = window.__mv || {};
    var v = mgVid();
    var fps = mv.fps || 30;
    var t = (v && mgVideoMode()) ? (Math.round(v.currentTime * 1000) / 1000) : (frame / fps);
    var ns = mgFaceNoSwapIds();
    return {
      frame: frame, idx: frame, n: frame,
      video: mgVidPath(), t: t, w: 900,
      /* 人脸库跟着当前项目走（空串 = 默认库 <工作区>/merge） */
      project: window.__mergeInputDir || "",
      /* 点过「不换脸」/ 点成原图的那几张脸：后端按这个列表原样留下 */
      no_swap_faces: ns,
      /* 有「不换脸」的脸就要无损（png）：那几张脸要和原帧逐像素相同 */
      fmt: (ns.length ? "png" : "jpg"),
      model: mgFaceSingleModel(),
      models: mgFaceModelsPayload(),
      settings: mgCmpSettings()
    };
  }
  window.mgCmpBody = mgCmpBody;

  /* ---------- 点框 = 只切这一张脸的「合成 / 原图」 ----------
     契约上就是 no_swap_faces：把这张脸的序号加进/移出请求体，后端把它原样留在
     帧里（其它脸照常换）。视觉反馈分两半：
       · 立刻：这条框线变虚线 + 框角出现「原图」小标签（不等网络）；
       · 随后：整帧重算回来，这张脸是原像素、其它脸还是换过的。
     状态按脸序号记在 localStorage 里（用户偏好），切帧不重置。 */
  function mgToggleFaceRaw(i, opt) {
    i = parseInt(i, 10);
    if (isNaN(i) || i < 0) return false;
    if (MG_FACE_RAWVIEW[i]) delete MG_FACE_RAWVIEW[i];
    else MG_FACE_RAWVIEW[i] = true;
    mgSaveRawView();
    mgSyncFaceCards();
    if (window.mvRedrawBoxes) window.mvRedrawBoxes();   /* 实线/虚线立刻换 */
    if (opt && opt.silent) return mgFaceIsRawView(i);
    var p = (window.mgCmpNow ? window.mgCmpNow() : null);
    if (p && p.catch) p.catch(function () {});
    return mgFaceIsRawView(i);
  }
  window.mgToggleFaceRaw = mgToggleFaceRaw;

  /* 画布坐标里的点命中哪个框；没有就 -1 */
  function mgFaceHitAt(x, y) {
    var stage = document.getElementById("mgStage");
    if (!stage || !MG_RECTS.length) return -1;
    var cv = stage.querySelector("canvas.mgovl");
    if (!cv) return -1;
    var r = cv.getBoundingClientRect();
    if (!r.width || !r.height) return -1;
    var px = (x - r.left) * (cv.width / r.width);
    var py = (y - r.top) * (cv.height / r.height);
    for (var i = 0; i < MG_RECTS.length; i++) {
      var b = MG_RECTS[i];
      if (!b || !(b.w > 0) || !(b.h > 0)) continue;
      if (px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h) return i;
    }
    return -1;
  }
  window.mgFaceHitAt = mgFaceHitAt;

  /* 绑一次：按下→抬起位移 < 4px 才算「点框」（拖拽/选择一律不算）。
     点在 chip / ＋ 上的不算 —— 它们是框外的按钮，有自己的事要干。 */
  var MG_PICK = { id: null, x: 0, y: 0, t: 0 };
  function mgBindFacePick() {
    var stage = document.getElementById("mgStage");
    if (!stage || window.__mgFacePickBound) return;
    window.__mgFacePickBound = 1;
    stage.setAttribute("data-face-pick", "1");
    var PICK_SLOP = 4;          /* 点击判定阈值（像素）：位移 >= 4 就是拖拽 */

    /* 绑在 document 上而不是舞台上：合成图 / 帧图 / <video> 三层的 pointer-events
       各不相同，绑在舞台上时点画面未必落到舞台自己身上（实测无头 Edge 里
       elementFromPoint 命中的是舞台里的图，不是舞台）。命中判定自己做，
       只有落在某个框里才切，点别处一律不动。 */
    /* 点在其它控件上不算：左右栏、时间轴、顶部的「人脸数据库」胶囊、toast…
       它们和舞台在同一个坐标系里，光靠"点落在框的矩形里"会误判。 */
    var PICK_SKIP = ".mgfc, button, a, input, select, label, #toastroot, " +
                    ".mgleft, .mgright, .mtl, .mgdbentry, .mgviews, .mgfgrid";

    document.addEventListener("pointerdown", function (e) {
      if (e.button != null && e.button !== 0) { MG_PICK.id = null; return; }
      var t = e.target;
      if (t && t.closest && t.closest(PICK_SKIP)) { MG_PICK.id = null; return; }
      MG_PICK.id = (e.pointerId == null ? 1 : e.pointerId);
      MG_PICK.x = e.clientX;
      MG_PICK.y = e.clientY;
      MG_PICK.t = Date.now();
    });

    document.addEventListener("pointerup", function (e) {
      var id = (e.pointerId == null ? 1 : e.pointerId);
      if (MG_PICK.id !== id) return;
      MG_PICK.id = null;
      var dx = (e.clientX || 0) - MG_PICK.x;
      var dy = (e.clientY || 0) - MG_PICK.y;
      if (Math.sqrt(dx * dx + dy * dy) >= PICK_SLOP) return;   /* 拖拽，不算点击 */
      if (Date.now() - MG_PICK.t > 1500) return;               /* 长按，不算点击 */
      var i = mgFaceHitAt(e.clientX, e.clientY);
      if (i < 0) return;
      e.preventDefault();
      mgToggleFaceRaw(i);
    });

    document.addEventListener("pointercancel", function () { MG_PICK.id = null; });
  }
  window.mgBindFacePick = mgBindFacePick;
  /* 测试/自动化用：每个框在页面坐标里的位置 + 现在是不是「原图」状态。
     只读，不改任何行为。 */
  window.mgFaceRects = function () {
    var stage = document.getElementById("mgStage");
    var cv = stage && stage.querySelector("canvas.mgovl");
    if (!cv) return [];
    var r = cv.getBoundingClientRect();
    if (!r.width || !cv.width) return [];
    var sx = r.width / cv.width, sy = r.height / cv.height;
    var out = [];
    MG_RECTS.forEach(function (b, i) {
      if (!b || !(b.w > 0) || !(b.h > 0)) return;
      out.push({ i: i,
                 x: r.left + b.x * sx, y: r.top + b.y * sy,
                 w: b.w * sx, h: b.h * sy,
                 cx: r.left + (b.x + b.w / 2) * sx,
                 cy: r.top + (b.y + b.h / 2) * sy,
                 raw: mgFaceIsRawView(i) });
    });
    return out;
  };
  function mgSetRects(rects, raw) {
    MG_RECTS = rects || [];
    MG_FACE_RAW = raw || [];
    mgRenderFaces();
  }

  /* ---------- 单脸调试：六宫格（3 列 × 2 行） ----------
     用户点某张脸框上的「调试」→ 舞台换成这张脸的特写六宫格：
       第 1 行  (1) mask(dfm-pred)      (2) mask(dst)    (3) merged(mask(dfm-pred), mask(dst))
       第 2 行  (4) dfm-pred            (5) dst-original (6) merged
     图从后端 GET/POST /api/merge/face-debug 来（**六张独立图**，前端自己排版，
     所以点任意一格就能放大）。后端复用预览 worker 里那一份 MergeMaskedFace，
     参数和画面用的是同一份 mgCmpBody（去掉 idx 再补 face）。
     六宫格是**盖在舞台上的一层**：底下的合成图、描边 canvas、chip 一个都不动，
     所以点「返回」= 把这层收起来，画面和进来之前逐像素一致。 */
  var MG_FDBG_CELLS = [
    ["mask-prd", "mask(dfm-pred)"],
    ["mask-dst", "mask(dst)"],
    ["mask-both", "merged(mask(dfm-pred), mask(dst))"],
    ["dfm-pred", "dfm-pred"],
    ["dst-original", "dst-original"],
    ["merged", "merged"]
  ];
  var MG_FDBG = { face: -1, seq: 0, data: null, zoom: "", busy: false, err: "" };

  function mgFdbgEl() { return document.getElementById("mgFaceGrid"); }

  /* 这张脸的六宫格请求体：和画面上那一发**同一套参数**，
     只把 idx（老接口里是帧号）摘掉 —— 这个接口的 face 才是「第几张脸」。 */
  function mgFdbgBody(i) {
    var frame = mgCmpFrameIdx();
    var base = window.mgCmpBody ? (window.mgCmpBody(frame) || {}) : {};
    var body = {};
    Object.keys(base).forEach(function (k) { body[k] = base[k]; });
    delete body.idx;
    body.frame = frame;
    body.n = frame;
    body.face = parseInt(i, 10) || 0;
    body.panel_size = 256;
    return body;
  }
  window.mgFdbgBody = mgFdbgBody;

  function mgFdbgSkeleton(i, frame) {
    var el = mgFdbgEl();
    if (!el) return null;
    el.innerHTML = "";
    var hd = document.createElement("div");
    hd.className = "mgfg-hd";
    var back = document.createElement("button");
    back.type = "button";
    back.id = "mgFaceGridBack";
    back.className = "mgfg-back";
    back.textContent = "← 返回";
    back.title = "回到合成视图（Esc）";
    var tt = document.createElement("span");
    tt.className = "mgfg-tt";
    tt.textContent = "";
    var sub = document.createElement("span");
    sub.className = "mgfg-sub";
    sub.id = "mgfgSub";
    sub.textContent = "正在算第 " + frame + " 帧…";
    hd.appendChild(back);
    hd.appendChild(tt);
    hd.appendChild(sub);
    el.appendChild(hd);

    var body = document.createElement("div");
    body.className = "mgfg-body";
    body.id = "mgfgBody";
    MG_FDBG_CELLS.forEach(function (pair, k) {
      var cell = document.createElement("figure");
      cell.className = "mgfg-cell";
      cell.setAttribute("data-k", pair[0]);
      cell.setAttribute("data-no", String(k + 1));
      cell.innerHTML =
        '<figcaption><span class="mgfg-nm">' + pair[1] + '</span></figcaption>' +
        '<div class="mgfg-im"><img alt="' + pair[1] + '"></div>';
      body.appendChild(cell);
    });
    el.appendChild(body);
    return el;
  }

  function mgFdbgFill(d) {
    var el = mgFdbgEl();
    if (!el) return;
    var by = {};
    ((d && d.panels) || []).forEach(function (p) { if (p && p.key) by[p.key] = p; });
    Array.prototype.forEach.call(el.querySelectorAll(".mgfg-cell"), function (cell) {
      var k = cell.getAttribute("data-k");
      var p = by[k] || null;
      var img = cell.querySelector("img");
      if (p && p.png) {
        img.src = p.png;
        cell.classList.add("has");
        cell.setAttribute("data-w", String(p.w || 0));
        cell.setAttribute("data-h", String(p.h || 0));
      } else {
        img.removeAttribute("src");
        cell.classList.remove("has");
      }
    });
    var sub = document.getElementById("mgfgSub");
    if (sub) {
      var st = (d && d.stats) || {};
      var bits = [];
      if (d && d.face && d.face.model) bits.push("模型 " + d.face.model);
      if (st.mask_prd != null) bits.push("遮罩覆盖 " + Math.round(Number(st.mask_prd) * 100) + "%");
      if (d && d.frozen) bits.push("冻结参数");
      if (d && d.ms != null) bits.push(Number(d.ms) + "ms");
      if (d && d.cached) bits.push("缓存");
      sub.textContent = "";
    }
  }
  window.mgFdbgFill = mgFdbgFill;

  function mgFaceDebugOpen(i) {
    i = parseInt(i, 10);
    if (isNaN(i) || i < 0) return null;
    var el = mgFdbgEl();
    if (!el) return null;
    var frame = mgCmpFrameIdx();
    MG_FDBG.face = i;
    MG_FDBG.zoom = "";
    MG_FDBG.data = null;
    MG_FDBG.err = "";
    mgFdbgSkeleton(i, frame);
    el.hidden = false;
    el.setAttribute("aria-hidden", "false");
    el.classList.remove("zoom");
    var st = document.getElementById("mgStage");
    if (st) st.setAttribute("data-face-debug", String(i));
    mgFaceDebugBind();
    var seq = ++MG_FDBG.seq;
    MG_FDBG.busy = true;
    fetch("/api/merge/face-debug", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(mgFdbgBody(i))
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok || !d || d.ok === false) {
          throw new Error((d && (d.detail || d.error)) || ("HTTP " + r.status));
        }
        return d;
      }, function () { throw new Error("HTTP " + r.status); });
    }).then(function (d) {
      if (seq !== MG_FDBG.seq) return null;      /* 已经关了/换了另一张脸：丢掉 */
      MG_FDBG.busy = false;
      MG_FDBG.data = d;
      mgFdbgFill(d);
      return d;
    }).catch(function (e) {
      if (seq !== MG_FDBG.seq) return null;
      MG_FDBG.busy = false;
      MG_FDBG.err = String((e && e.message) || e);
      var sub = document.getElementById("mgfgSub");
      if (sub) sub.textContent = "六宫格取图失败：" + MG_FDBG.err.slice(0, 90);
      if (window.mgHint) window.mgHint("");
      if (window.toast) {
        try { toast("六宫格取图失败", MG_FDBG.err.slice(0, 90), "warn", 5000); } catch (x) {}
      }
      return null;
    });
    return el;
  }
  window.mgFaceDebugOpen = mgFaceDebugOpen;

  function mgFaceDebugClose() {
    MG_FDBG.seq++;                 /* 在飞的那一发回来也不要了 */
    MG_FDBG.face = -1;
    MG_FDBG.data = null;
    MG_FDBG.busy = false;
    MG_FDBG.zoom = "";
    var el = mgFdbgEl();
    if (el) {
      el.hidden = true;
      el.setAttribute("aria-hidden", "true");
      el.classList.remove("zoom", "on");
      el.innerHTML = "";
    }
    var st = document.getElementById("mgStage");
    if (st) st.removeAttribute("data-face-debug");
    return true;
  }
  window.mgFaceDebugClose = mgFaceDebugClose;

  /* 点格子放大 / 返回 / Esc —— 只绑一次（网格节点会重建，事件挂在容器上） */
  function mgFaceDebugBind() {
    var el = mgFdbgEl();
    if (!el || el.dataset.bound) return;
    el.dataset.bound = "1";
    el.addEventListener("click", function (e) {
      var t = e.target;
      if (t && t.closest && t.closest(".mgfg-back")) {
        e.preventDefault();
        mgFaceDebugClose();
        return;
      }
      var cell = t && t.closest ? t.closest(".mgfg-cell") : null;
      if (!cell || !el.contains(cell)) return;
      var k = String(cell.getAttribute("data-k") || "");
      if (!cell.classList.contains("has")) return;
      /* 再点一次同一格 = 缩回去 */
      if (MG_FDBG.zoom === k) {
        MG_FDBG.zoom = "";
        el.classList.remove("zoom");
        Array.prototype.forEach.call(el.querySelectorAll(".mgfg-cell.on"), function (c) {
          c.classList.remove("on");
        });
        return;
      }
      MG_FDBG.zoom = k;
      el.classList.add("zoom");
      Array.prototype.forEach.call(el.querySelectorAll(".mgfg-cell"), function (c) {
        c.classList.toggle("on", c === cell);
      });
    });
    if (!window.__mgFdbgKeyBound) {
      window.__mgFdbgKeyBound = 1;
      document.addEventListener("keydown", function (e) {
        if (e.key !== "Escape") return;
        var box = mgFdbgEl();
        if (!box || box.hidden) return;
        e.preventDefault();
        e.stopPropagation();
        if (MG_FDBG.zoom) {          /* 先退出放大，再退六宫格 */
          MG_FDBG.zoom = "";
          box.classList.remove("zoom");
          Array.prototype.forEach.call(box.querySelectorAll(".mgfg-cell.on"), function (c) {
            c.classList.remove("on");
          });
          return;
        }
        mgFaceDebugClose();
      });
    }
  }
  window.mgFaceDebugBind = mgFaceDebugBind;

  /* 测试/自动化用（只读）：六宫格现在开着没、哪张脸、六格都拿到图没有 */
  window.mgFaceDebugState = function () {
    var el = mgFdbgEl();
    if (!el) return { present: false };
    var cells = Array.prototype.map.call(el.querySelectorAll(".mgfg-cell"), function (c) {
      var im = c.querySelector("img");
      var r = c.getBoundingClientRect();
      return { k: c.getAttribute("data-k"), no: c.getAttribute("data-no"),
               nm: String((c.querySelector(".mgfg-nm") || {}).textContent || ""),
               has: c.classList.contains("has"),
               src: im ? String(im.getAttribute("src") || "").slice(0, 32) : "",
               nw: im ? im.naturalWidth : 0, nh: im ? im.naturalHeight : 0,
               w: Math.round(r.width), h: Math.round(r.height) };
    });
    return { present: true, open: !el.hidden, face: MG_FDBG.face,
             zoom: MG_FDBG.zoom, busy: MG_FDBG.busy, err: MG_FDBG.err,
             rect: (function () { var r = el.getBoundingClientRect();
                                 return [r.x, r.y, r.width, r.height]; })(),
             cells: cells };
  };


  /* 换素材了：脸序号是「这一帧的检测顺序」，把上一份素材的分配带过去毫无意义
     （3 张脸的项目点了 face 2 不换脸，再开一个 1 张脸的视频，那张脸就永远不换）。 */
  function mgClearFaceAssign() {
    MG_FACE_MODEL = {};
    MG_FACE_NOSWAP = {};
    MG_FACE_AUTO = {};
    mgResetFaces();
    /* 换了素材 → 刚才那份「能不能拿到这一帧的图」的缓存作废 */
    if (window.mgSourceForget) window.mgSourceForget();
  }
  window.mgClearFaceAssign = mgClearFaceAssign;

  function mgResetFaces() {
    MG_RECTS = [];
    MG_FACE_RAW = [];
    var layer = document.getElementById("mgFaceLayer");
    if (layer) layer.innerHTML = "";
    /* 脸都没了：还开着的六宫格指的就是不存在的脸，收掉 */
    if (window.mgFaceDebugClose) window.mgFaceDebugClose();
  }
  window.mgResetFaces = mgResetFaces;

  /* ---------- chip 层：每张脸一个小卡片 + 一个 ＋，全部是 DOM ----------
     关键：**脸没变就不重建**。
     以前这里无条件 layer.innerHTML = "" 再重建。而这张图每次合成回包都会重画
     （mgShowOverlay → mgApplyView → mvRedrawBoxes，加上 p.onload 再补一发、检测回包再来一发，
     一次换脸 = 3 次重画），重建会把**打开着的下拉菜单**连同里面的 .ddopt 一起扔掉：
     用户的 mousedown 落在旧节点上、mouseup/click 落在新节点或外层上 —— 浏览器把 click
     派发给两者共同的祖先，选项上的监听器一次都没跑，于是 MG_FACE_MODEL 没写、
     chip 还是「选择模型」、重开下拉也没有高亮（用户报的就是这个）。
     现在：脸序号列表一致 → 只原地更新位置/文字/颜色/忙碌环（下拉不受影响）；真的换了脸数才重建。 */
  function mgRenderFaces() {
    var layer = document.getElementById("mgFaceLayer");
    if (!layer) return;
    if (!mgBoxesOn()) { layer.innerHTML = ""; return; }
    var want = [], i, r;
    for (i = 0; i < MG_RECTS.length; i++) {
      r = MG_RECTS[i];
      if (!r || !(r.w > 0) || !(r.h > 0)) continue;
      want.push(i);
    }
    var cards = Array.prototype.slice.call(layer.querySelectorAll(".mgfc"));
    var same = (cards.length === want.length);
    for (i = 0; same && i < cards.length; i++) {
      if (parseInt(cards[i].getAttribute("data-face"), 10) !== want[i]) same = false;
    }
    if (same) { mgSyncFaceCards(); return; }   /* ← 原地更新，别炸掉打开着的下拉 */
    layer.innerHTML = "";
    for (i = 0; i < want.length; i++) layer.appendChild(mgMakeFaceCard(want[i], MG_RECTS[want[i]]));
    mgSyncFaceCards();
  }
  window.mgRenderFaces = mgRenderFaces;

  function mgCloseChipMenus(except) {
    Array.prototype.forEach.call(document.querySelectorAll("#mgFaceLayer .mgfdd.open"), function (d) {
      if (d !== except) d.classList.remove("open");
    });
  }

  function mgMakeFaceCard(i, r) {
    /* 颜色跟着 chip 上显示的那个名字走：单选/多选时那一帧的全部脸确实都在用
       左栏的 dfm，chip 不该是灰的（以前只有 MG_FACE_MODEL / faces[] 才算"已分配"）。 */
    var mname = mgFaceChipModel(i);
    var color = mname ? mgModelColor(mname) : "";

    /* 零尺寸锚点落在「框的右上角」：left 要加宽度，否则 anchor 停在左边线，
       chip 会跑到框外左侧、压到左栏上去（点都点不到） */
    var card = document.createElement("div");
    card.className = "mgfc" + (MG_FACE_BUSY[i] ? " busy" : "");
    card.setAttribute("data-face", String(i));
    /* 框角小标签「原图」：这张脸被点成原图时才露出来（别的框一个像素都不变）。
       pointer-events:none —— 它盖在画面上，点它要继续算「点框」。 */
    var vtag = document.createElement("i");
    vtag.className = "mgvtag";
    vtag.textContent = "原图";
    vtag.hidden = !mgFaceIsRawView(i);
    card.style.left = (r.x + r.w) + "px";
    card.style.top = r.y + "px";

    /* 右上角外侧：模型 chip（.dd / .ddbtn / .ddopt 是项目里的主题化下拉） */
    var dd = document.createElement("div");
    dd.className = "dd mgfdd";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ddbtn mgfb" + (color ? " has" : "");
    btn.title = mname ? ("这张脸用 " + mname + "（点击换）") : "给这张脸选一个模型";
    if (color) { btn.style.borderColor = color; btn.style.color = color; }
    var txt = document.createElement("span");
    txt.className = "ddtxt";
    txt.textContent = mgFaceChipText(i);
    btn.appendChild(txt);

    var menu = document.createElement("div");
    menu.className = "ddmenu mgfmenu";
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      var open = dd.classList.contains("open");
      mgCloseChipMenus(null);
      if (open) { dd.classList.remove("open"); return; }
      menu.innerHTML = "";
      /* 选项 = 左栏已选中的 dfm + 「不换脸」 */
      var names = mgSelNames();
      var opts = names.concat([""]);
      if (!names.length) {
        var none = document.createElement("div");
        none.className = "ddopt off";
        none.textContent = "左栏还没选模型";
        menu.appendChild(none);
      }
      opts.forEach(function (nm) {
        var it = document.createElement("div");
        /* 「不换脸」那一项的高亮看的是 MG_FACE_NOSWAP，不是"有没有分配" */
        it.className = "ddopt" + ((nm ? ((mgFaceModel(i) || "") === nm) : mgFaceIsNoSwap(i)) ? " on" : "");
        it.textContent = nm || "不换脸";
        if (nm) it.style.color = mgModelColor(nm);
        it.addEventListener("click", function (ev) {
          ev.stopPropagation();
          dd.classList.remove("open");
          mgAssignFace(i, nm);
        });
        menu.appendChild(it);
      });
      dd.classList.add("open");
    });
    dd.appendChild(btn);
    dd.appendChild(menu);

    /* 重算中的细进度环（只圈这一张脸，不整屏 loading） */
    var ring = document.createElement("i");
    ring.className = "mgfring";
    ring.hidden = !MG_FACE_BUSY[i];
    if (!ring.hidden && ring.animate) {
      try {
        ring.animate([{ transform: "rotate(0deg)" }, { transform: "rotate(360deg)" }],
                     { duration: 720, iterations: Infinity, easing: "linear" });
      } catch (e) { /* 没有 WAAPI 就留一个静态环 */ }
    }
    dd.appendChild(ring);

    /* 右下角外侧，和 ＋ 并排：这张脸的六宫格调试（3 列 × 2 行特写）。
       位置靠右对齐、在 ＋ 左边一格 —— 两个按钮同高同圆角，视觉上是一对。 */
    var dbg = document.createElement("button");
    dbg.type = "button";
    dbg.className = "mgfdbg";
    dbg.textContent = "调试";
    dbg.title = "看这张脸的六宫格（遮罩 / 预测脸 / 原脸 / 合成）";
    dbg.setAttribute("data-tip", "看这张脸的六宫格");
    dbg.setAttribute("aria-label", "看这张脸的六宫格");
    dbg.style.top = Math.round(r.h + 3) + "px";
    dbg.addEventListener("click", function (e) {
      e.stopPropagation();
      mgFaceDebugOpen(i);
    });

    /* 最右下角外侧：加进人脸数据库 */
    var plus = document.createElement("button");
    plus.type = "button";
    plus.className = "mgplus";
    plus.textContent = "+";
    plus.title = "把当前人脸添加到人脸数据库";
    plus.setAttribute("data-tip", "把当前人脸添加到人脸数据库");
    plus.setAttribute("aria-label", "把当前人脸添加到人脸数据库");
    plus.style.top = Math.round(r.h + 3) + "px";
    plus.addEventListener("click", function (e) {
      e.stopPropagation();
      mgAddFaceToDb(i, plus);
    });

    card.appendChild(vtag);
    card.appendChild(dd);
    card.appendChild(dbg);
    card.appendChild(plus);
    return card;
  }

  /* 只改已存在卡片的外观（换色 / 换字 / 进度环），不重建 DOM：
     重建会把正在 hover 的下拉关掉，也会让动画闪。 */
  function mgSyncFaceCards() {
    var layer = document.getElementById("mgFaceLayer");
    if (!layer) return;
    Array.prototype.forEach.call(layer.querySelectorAll(".mgfc"), function (card) {
      var i = parseInt(card.getAttribute("data-face"), 10);
      /* 卡片不再重建，位置就得在这里跟着当前帧的框走（切帧/换图后 chip 要贴回框的右上角） */
      var rc = MG_RECTS[i];
      if (rc && rc.w > 0 && rc.h > 0) {
        card.style.left = (rc.x + rc.w) + "px";
        card.style.top = rc.y + "px";
        var pl = card.querySelector(".mgplus");
        if (pl) pl.style.top = Math.round(rc.h + 3) + "px";
        /* 「调试」和 ＋ 同一行（并排），高度跟着框走 */
        var dg = card.querySelector(".mgfdbg");
        if (dg) dg.style.top = Math.round(rc.h + 3) + "px";
      }
      var nm = mgFaceChipModel(i);
      var color = nm ? mgModelColor(nm) : "";
      var btn = card.querySelector(".mgfb");
      var tx = card.querySelector(".ddtxt");
      if (tx) tx.textContent = mgFaceChipText(i);
      if (btn) {
        btn.classList.toggle("has", !!color);
        btn.style.borderColor = color || "";
        btn.style.color = color || "";
        btn.title = nm ? ("这张脸用 " + nm + "（点击换）") : "给这张脸选一个模型";
      }
      var ring = card.querySelector(".mgfring");
      if (ring) ring.hidden = !MG_FACE_BUSY[i];
      card.classList.toggle("busy", !!MG_FACE_BUSY[i]);
      var vt = card.querySelector(".mgvtag");
      if (vt) vt.hidden = !mgFaceIsRawView(i);
    });
  }

  /* ---------- 每张脸分到哪个模型 → models payload ---------- */
  function mgFaceModelsPayload() {
    var byName = {}, order = [];
    Object.keys(MG_FACE_MODEL).forEach(function (k) {
      var i = parseInt(k, 10), nm = String(MG_FACE_MODEL[k] || "");
      if (!nm || isNaN(i)) return;
      if (!byName[nm]) { byName[nm] = []; order.push(nm); }
      byName[nm].push(i);
    });
    /* 一张都没手动分配 → 发空数组：**让后端按人脸 embedding 自动决定每张脸用哪个
       dfm**（/api/merge/frame、/api/merge/merged 的自动分配）。
       以前这里退回「左栏选中 = 全部脸」，那等于用同一个模型换所有人 ——
       和「组里的人脸 = 那个组的 dfm」的语义正好相反。 */
    if (!order.length) return [];
    return order.map(function (nm) {
      return { name: nm, faces: byName[nm].sort(function (a, b) { return a - b; }) };
    });
  }
  window.mgFaceModelsPayload = mgFaceModelsPayload;

  function mgFaceSingleModel() {
    var keys = Object.keys(MG_FACE_MODEL);
    if (keys.length) {
      var k = keys.sort(function (a, b) { return Number(a) - Number(b); })[0];
      return String(MG_FACE_MODEL[k] || "");
    }
    try { return mgPreviewModel(); } catch (e) { return ""; }
  }
  window.mgFaceSingleModel = mgFaceSingleModel;

  /* ---------- 右栏参数 → 请求里的 settings ---------- */
  var MG_MASK_INT = { "prd": 1, "dst": 2, "prd×dst": 3 };   /* 位标志：1=prd 2=dst 3=both */
  function mgSegVal(key, dflt) {
    var b = document.querySelector('#p-merge .seg[data-k="' + key + '"] b.on');
    if (!b) return dflt;
    var dv = b.getAttribute("data-v");
    return (dv == null) ? String(b.textContent || "").trim() : String(dv);
  }
  function mgNumVal(id, dflt) {
    var el = document.getElementById(id);
    var v = parseFloat(el ? el.value : "");
    return isFinite(v) ? v : dflt;
  }
  /* 融合方式只有这 4 个（和「模式」下拉的选项、后端白名单三处必须一致）。 */
  var MG_MERGE_BLENDS = ["overlay", "hist-match", "seamless", "seamless-hist-match"];

  /* 人脸超分：滑块 0..100。0=关；1..50 用 GPEN-256（α=lv/50）；51..100 用 GPEN-512（α=(lv-50)/50）。
     模型本身没有强度参数，连续旋钮就是「SR 与 原图 的混合比例」。 */
  function mgSrLevel() {
    var el = document.getElementById("mgSr");
    var v = el ? Number(el.value) : 0;
    if (!isFinite(v)) v = 0;
    return Math.max(0, Math.min(100, Math.round(v)));
  }
  function mgSrText(lv) {
    if (lv <= 0) return "off";
    if (lv <= 50) return "256 · α" + (lv / 50).toFixed(2);
    return "512 · α" + ((lv - 50) / 50).toFixed(2);
  }
  function mgSrPaint() {
    var lv = mgSrLevel();
    var out = document.getElementById("mgSrVal");
    if (out) out.textContent = mgSrText(lv);
    var sr = document.getElementById("mgSr");
    if (sr) {
      sr.classList.toggle("is-off", lv <= 0);
      sr.title = lv <= 0 ? "关（不做超分）"
        : (lv <= 50 ? "GPEN-256 · 混合 " + (lv / 50).toFixed(2) + "（单次约 10~20ms）"
                    : "GPEN-512 · 混合 " + ((lv - 50) / 50).toFixed(2) + "（单次约 60~73ms，慢 7 倍）");
    }
  }
  window.mgSrLevel = mgSrLevel;
  window.mgSrPaint = mgSrPaint;
  (function bindSr() {
    var sr = document.getElementById("mgSr");
    if (!sr || sr.dataset.srBound) return;
    sr.dataset.srBound = "1";
    sr.addEventListener("input", mgSrPaint);
    sr.addEventListener("change", mgSrPaint);
    mgSrPaint();
  })();

  function mgCmpSettings() {
    var ct = String((document.getElementById("mgColor") || {}).value || "mt");
    var mm = mgSegVal("mask", "prd×dst") || "prd×dst";
    var maskInt = (MG_MASK_INT[mm] == null) ? 3 : MG_MASK_INT[mm];
    var erode = mgNumVal("mgErode", 0);
    var blur = mgNumVal("mgBlur", 4);
    /* 融合方式的真源 = 「模式」下拉（4 个）。右边那个两档分段（叠加/无缝）
       只是它的快捷方式，两边由 mgSyncBlendSeg 保持同一个值。
       以前 blend 只读分段、mode 读下拉，而后端以 blend 为准 —— 下拉里选
       hist-match 画面一点不动（用户报的「选了没反应」就是这个）。 */
    var modeSel = String((document.getElementById("mgMode") || {}).value || "");
    if (MG_MERGE_BLENDS.indexOf(modeSel) < 0) modeSel = "";
    var blend = modeSel || mgSegVal("blend", "seamless");
    if (MG_MERGE_BLENDS.indexOf(blend) < 0) blend = "seamless";
    return {
      /* —— 新契约 —— */
      ct_mode: ct,
      blend: blend,
      mask_mode: maskInt,
    /* 遮罩来源单独发：后端只认这个字段（mask_mode 整数分不清 XSeg 与 XSegLite） */
    mask_source: mgSegVal("seg", "XSegLite"),
      mask_erode: erode,
      mask_blur: blur,
      /* —— 旧字段兜底：后端还是老版本时认这一组 —— */
      mode: modeSel || blend,
      color_transfer: ct,
      mask_source: mgSegVal("seg", "XSegLite"),
      mask_style: mm,
      erode_mask: erode,
      blur_mask: blur,
      /* 人脸超分滑块 0..100（0 = 关；<=50 用 256，>50 用 512）—— 后端 knobs 读这个键 */
      sr: mgSrLevel(),
      detector: mgDetectorName(),
      face_type: String((document.getElementById("mgFaceType") || {}).value || "whole_face")
    };
  }
  window.mgCmpSettings = mgCmpSettings;

  /* 「模式」下拉 ↔ 右边的「融合方式」两档分段：同一个值，两边一起亮。
     overlay → 叠加，其它三种（hist-match / seamless / seamless-hist-match）→ 无缝，
     因为分段只有这两档 —— 它就是个快捷方式，真值永远在下拉里。 */
  function mgSyncBlendSeg(val) {
    var v = String(val || "");
    if (MG_MERGE_BLENDS.indexOf(v) < 0) return;
    var seg = document.querySelector('#p-merge .seg[data-k="blend"]');
    /* 融合方式分段已按用户要求删掉（与「模式」下拉重复），有元素才同步，没有就什么都不做。 */
    if (!seg) return;
    var want = (v === "overlay") ? "overlay" : "seamless";
    Array.prototype.forEach.call(seg.querySelectorAll("b"), function (b) {
      b.classList.toggle("on", (b.getAttribute("data-v") || "") === want);
    });
  }
  window.mgSyncBlendSeg = mgSyncBlendSeg;

  function mgCmpFrameIdx() {
    if (window.MergeTimeline && window.MergeTimeline.getFrame) {
      var n = window.MergeTimeline.getFrame();
      if (typeof n === "number" && isFinite(n)) return Math.max(0, Math.round(n));
    }
    var f = document.getElementById("mtlFrame");
    return Math.max(0, parseInt(f ? f.value : 0, 10) || 0);
  }
  window.mgCmpFrameIdx = mgCmpFrameIdx;

  /* 画面右下角的小字「正在合成…」：不遮画面、不吃点击 */
  function mgCmpNote(on, text) {
    var el = document.getElementById("mgCmpNote");
    if (!el) return;
    if (!on) { el.hidden = true; return; }
    el.textContent = text || "正在合成…";
    el.hidden = false;
  }
  window.mgCmpNote = mgCmpNote;

  function mgCmpBusyAll(ids) {
    MG_FACE_BUSY = {};
    (ids || []).forEach(function (i) { MG_FACE_BUSY[i] = true; });
    mgSyncFaceCards();
  }
  function mgCmpBusyClear() {
    MG_FACE_BUSY = {};
    mgSyncFaceCards();
  }

  /* ---------- 重算当前帧 ----------
     主路：POST /api/merge/frame（settings + models 分配）
     兜底：POST 不认（404/405/501）→ 老 GET /api/merge/frame/<n>?w=900
           再不行（视频模式）→ 老 POST /api/merge/preview
     每次新请求都 ++seq 并 abort 上一发：旧结果永远盖不到新画面上。 */
  function mgCmpNow(opt) {
    opt = opt || {};
    var mv = window.__mv || {};
    var frame = (opt.frame == null) ? mgCmpFrameIdx() : Math.max(0, parseInt(opt.frame, 10) || 0);
    if (!mv.path && !mgVidPath()) return Promise.resolve(null);

    var seq = ++MG_CMP.seq;
    if (MG_CMP.ctl) { try { MG_CMP.ctl.abort(); } catch (e) {} MG_CMP.ctl = null; }
    /* 请求体全在 mgCmpBody 里：合成只有这一种出图方式（没有视图字段可加了） */
    var body = mgCmpBody(frame);
    var ctrl = (typeof AbortController === "function") ? new AbortController() : null;
    MG_CMP.ctl = ctrl;
    mgCmpNote(true);
    var t0 = (window.performance && performance.now) ? performance.now() : Date.now();

    return fetch("/api/merge/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl ? ctrl.signal : undefined
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok || !d) {
          var err = new Error((d && d.detail) || ("HTTP " + r.status));
          err.status = r.status;
          /* 结构化错误（后端自己的 {ok:false,error}）和「路由不存在」要分得开：
             前者是参数/模型真错了，必须说出来；后者才轮到老接口兜底。 */
          err.api = !!(d && typeof d.ok === "boolean");
          throw err;
        }
        return d;
      }, function () {
        var err = new Error("HTTP " + r.status);
        err.status = r.status;
        err.api = false;                 /* 响应体不是 JSON：多半是老后端/没这个路由 */
        throw err;
      });
    }).then(function (d) {
      if (seq !== MG_CMP.seq) return null;            /* 过期响应：丢掉 */
      MG_CMP.ctl = null;
      MG_CMP.lastMs = ((window.performance && performance.now) ? performance.now() : Date.now()) - t0;
      mgCmpBusyClear();
      mgCmpNote(false);
      var png = d.png || d.image || d.data || d.frame_png || "";
      /* 过期响应上面已经丢了：到这儿就一定是当前这一帧，直接贴 */
      if (png) { window.mgShowOverlay(png); mgHint(""); }
      /* 合成图到了：它只换底图，框一个都不动（框是 detect 那条线画的） */
      if (window.mgTimingMark) window.mgTimingMark("t_frame");
      /* 每张脸被判给了哪个组 → chip 上直接显示（用户手动改则覆盖） */
      mgApplyFacePlan(d);
      return d;
    }).catch(function (e) {
      if (seq !== MG_CMP.seq) return null;
      if (e && e.name === "AbortError") return null;   /* 被新请求取代，不算失败 */
      MG_CMP.ctl = null;
      mgCmpBusyClear();
      mgCmpNote(false);
      var stt = e && e.status;
      if (stt === 404 || stt === 405 || stt === 501 || stt === 400 || stt === 422) {
        /* 后端自己给的 4xx（{ok:false,error}）= 参数/模型/检测器真错了。
           以前这里一并退回老 GET，老 GET 不吃 settings/model —— 结果是"点了个
           没导出 .dfm 的模型 / 传了个不存在的检测器"，画面悄悄换成**默认参数**那一张，
           错误被吞掉。现在只有「路由不存在」（响应体不是我们的结构化 JSON）才兜底。 */
        if (!(e && e.api)) return mgCmpLegacy(frame);
        mgCmpNote(false);
        mgHint(String(e.message || "合成失败").slice(0, 80));
        return null;
      }
      mgCmpNote(false);
      mgHint(String((e && e.message) || "合成失败").slice(0, 80));
      return null;
    });
  }
  window.mgCmpNow = mgCmpNow;

  function mgCmpLegacy(frame) {
    var seq = MG_CMP.seq;                    /* 不 ++：还是同一发 */
    mgCmpNote(true, "正在合成…（旧接口）");
    return fetch("/api/merge/frame/" + frame + "?w=900").then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (d) {
      if (seq !== MG_CMP.seq) return null;
      mgCmpNote(false);
      if (d && d.png) { window.mgShowOverlay(d.png); mgHint(""); }
      return d;
    }).catch(function () {
      if (seq !== MG_CMP.seq) return null;
      mgCmpNote(false);
      /* 逐帧接口也没有：视频模式退回老的预览接口（它自己也带 settings/models） */
      if (mgVideoMode() && window.mgPreviewNow) {
        return window.mgPreviewNow({ force: true, fromFallback: true });
      }
      return null;
    });
  }

  /* 参数改一下 → 150ms 防抖 → 重算一击。连拖滑块 10 次只发最后一次。 */
  function mgCmpSoon() {
    if (MG_CMP.wait) clearTimeout(MG_CMP.wait);
    MG_CMP.wait = setTimeout(function () {
      MG_CMP.wait = null;
      mgCmpNow();
    }, 150);
  }
  window.mgCmpSoon = mgCmpSoon;

  /* 「原图」那一态：后端 /api/merge/frame 默认出的是合成结果，
     原片要显式 raw=1（契约里 raw=1 或 compose=0 就是不合成）。
     老后端不认 raw → 404/400 时退回 mvFrame 那条路，画面不至于空着。 */
  window.mgRawFrame = function (n) {
    n = Math.max(0, parseInt(n, 10) || 0);
    var seq = ++MV_SEQ.orig;
    return fetch("/api/merge/frame/" + n + "?w=900&raw=1&fmt=png").then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(function (d) {
      if (seq !== MV_SEQ.orig) return null;
      if (d && d.png) window.mvStage(d.png);
      return d;
    }).catch(function () { return window.mvFrame(n); });
  };

  /* ---------- 底图跟帧：项目 / 帧目录模式下一帧一发原图 ----------
     底图是 img.mgfrm（原片），合成结果贴在上面的 #mgPreview。以前底图只在本帧
     「无人脸」时才换（mgNoFaceFrame → mgRawFrame）：有人脸就只贴合成层，于是
       · 合成慢一拍（项目模式 300~500ms）时画面钉在上一帧；
       · 合成失败 / 没选模型 / 被取消时，画面**永远**停在上一帧 —— 看起来就是
         「点时间轴不刷新」；
       · 时间轴播放时更彻底：播放中本来就不发合成，整屏冻在第一帧。
     现在底图和合成解耦：帧一变先换底图（只解码 + 缩放 + 编码，不检测不换脸），
     合成到了再盖上去。单发在飞时只记「最新目标」，回来补最后一帧 —— 拖时间轴
     连发几十帧也只补最后一张，不会排队排到画面走远。 */
  var MG_BASE = { want: -1, busy: false };
  function mgBaseFrame(n) {
    if (mgVidPath()) return;                  /* 视频模式底图是 <video> 自己 */
    MG_BASE.want = Math.max(0, parseInt(n, 10) || 0);
    if (MG_BASE.busy) return;                 /* 在飞：记下最新目标，回来再补 */
    mgBasePump();
  }
  function mgBasePump() {
    if (MG_BASE.want < 0) return;
    var n = MG_BASE.want;
    MG_BASE.want = -1;
    MG_BASE.busy = true;
    function done() {
      MG_BASE.busy = false;
      if (MG_BASE.want >= 0) mgBasePump();
    }
    var p = null;
    try { p = window.mgRawFrame(n); } catch (e) { p = null; }
    if (p && p.then) p.then(done, done); else done();
  }
  window.mgBaseFrame = mgBaseFrame;

  /* ---------- 帧变化编排：清框 → 检测（画框）→ 合成（只换底图） ----------
     一条帧变化只从这里走。三件事，不是两次等待：
       ① 同步清框（canvas 描边 + chip 层），这一代检测一并作废；
       ② 立刻发 /api/merge/detect —— 回来就画框 + chip，t_draw 紧跟 t_detect；
       ③ 检测结论「有人脸」才发合成（项目模式 /api/merge/frame，视频模式走预览），
          合成回包只替换底图，框一个都不碰。
     为什么合成要等检测这一拍：本帧没人脸时合成毫无意义（后端也只是回原图），
     而且先发 frame 会把它（内部含检测 + dfm + 遮罩 + 编码，~400ms）排在
     自己那一发检测前面 —— 看起来就像「检测很慢」，其实是被合成堵住了。
     检测本身是 3~20ms 的活，等它一拍不亏。
     兜底：检测冷启动（首次加载 onnx 要几秒）时不能让画面钉住 —— MG_DET_GATE
     到点就照发合成；那次检测晚回来照样把框画上（帧没换的话）。 */
  /* 等检测结论的兜底：第一档 800ms，如果那一发检测**还在飞**就再等一档，
     最多等到 3000ms。只在检测彻底挂住（或冷启动加载 onnx 要好几秒）时才放行合成 ——
     后端一忙，检测 700ms 才回来是常事，那种情况必须等它，不然又白跑一轮 dfm。 */
  var MG_DET_GATE = 800;
  var MG_DET_GATE_MAX = 3000;
  var MG_CHAIN = { seq: 0, frame: -1, gate: null, gateOpen: true, gateFrom: 0 };

  function mgComposeFrame(n) {
    var videoMode = !!(window.mgVideoMode && window.mgVideoMode() &&
                       !(window.__mgVideo || {}).streamDown);
    if (videoMode) {
      /* 视频模式：底图是 <video> 自己，合成结果只是贴上去的那一张 overlay */
      if (window.mgPreviewSoon) window.mgPreviewSoon();
      return null;
    }
    return window.mgCmpNow ? window.mgCmpNow({ frame: n }) : null;
  }

  /* 本帧无人脸：不发合成，底图换成原图，框 / chip 都不画 */
  function mgNoFaceFrame(n) {
    MG_FRAME_T.noFace = true;
    if (window.mgShowOverlay) window.mgShowOverlay("");     /* 撤掉上一帧留下的合成层 */
    if (window.mgCmpNote) window.mgCmpNote(false);
    if (window.mgHint) window.mgHint("本帧无人脸");
    var videoMode = !!(window.mgVideoMode && window.mgVideoMode());
    if (!videoMode && window.mgRawFrame) {
      /* 轻量取帧：GET /api/merge/frame/<n>?raw=1 —— 只解码 + 编码，不检测不换脸。
         它是「原图」，不是合成请求（raw=1 那条分支在 _frame_payload 最前面就返回了）。 */
      var p = null;
      try { p = window.mgRawFrame(n); } catch (e) { p = null; }
      /* 原图落地会把舞台中间的小字收起来（mvStage 里 hint.display=none），
         所以状态行等它回来再补一次 —— 帧/无人脸结论没变才补。 */
      if (p && p.then) {
        p.then(function () {
          if (MG_CHAIN.frame === n && MG_FRAME_T.noFace && window.mgHint) window.mgHint("本帧无人脸");
        }, function () {});
      }
    }
    return true;
  }
  window.mgNoFaceFrame = mgNoFaceFrame;

  function mgFrameChange(n, opt) {
    opt = opt || {};
    var mv = window.__mv || {};
    n = Math.max(0, parseInt(n, 10) || 0);
    if (!mv.path && !mgVidPath()) return null;
    /* 换帧了：开着的那张脸的六宫格是上一帧的，直接收掉（别让人对着旧证据调参） */
    if (window.mgFaceDebugClose) window.mgFaceDebugClose();
    /* 同一帧的连环召唤（进页面 init 归零 + open 补帧会连着叫三次）并成一发：
       上一发还在飞（gate 关着）就不用再清一次框、再打一发检测 */
    if (!opt.force && MG_CHAIN.frame === n && !MG_CHAIN.gateOpen) return null;
    var seq = ++MG_CHAIN.seq;
    MG_CHAIN.frame = n;
    MG_CHAIN.gateOpen = false;               /* 合成这道门先关上，等检测结论 */
    if (MG_CHAIN.gate) { clearTimeout(MG_CHAIN.gate); MG_CHAIN.gate = null; }

    /* ① 帧要变了：先把框清干净（同步，不等任何回包） */
    window.mgInvalidateBoxes(opt.why || "frame", n);

    /* ①-b 底图立刻跟到这一帧。合成还在路上（或根本不会来：无人脸 / 没模型 /
       失败 / 播放中）时，画面也必须先动 —— 否则时间轴点了像没点。 */
    if (!mgVidPath()) {
      mgHideOverlay();                       /* 上一帧的合成图不能盖着新底图 */
      mgBaseFrame(n);
    }

    /* 播放中：只清框 + 一发「在飞就跳过」的检测，绝不发合成。
       一帧一发 4 秒的合成会把 worker 队列塞死（原来项目模式播放就是这样），
       画面反而更卡；而检测结果几乎必然过期（见 mvDetect 的 outdated 分支），
       所以这一拍宁可没有框 —— 停下来那一发才是准的。 */
    var playing = !!((window.MergeTimeline && window.MergeTimeline.isPlaying &&
                      window.MergeTimeline.isPlaying()) ||
                     (window.__mgVideo && window.__mgVideo.playing));
    if (playing) {
      MG_CHAIN.gateOpen = true;              /* 没有待发的合成，门开着 */
      if (!MG_DET.inflight) {
        try { window.mvDetect(n); } catch (e) {}
      }
      return null;
    }

    /* 视频模式：先让 <video> 走到这一帧（画面本体）；seeked 里的预览被上面的门挡住 */
    if (window.mgVideoMode && window.mgVideoMode() && !(window.__mgVideo || {}).streamDown) {
      try { window.mgVideoSeek(n); } catch (e) {}
    }

    var fired = false;
    function openGate() {
      MG_CHAIN.gateOpen = true;
      if (MG_CHAIN.gate) { clearTimeout(MG_CHAIN.gate); MG_CHAIN.gate = null; }
    }
    function composeOnce() {
      if (fired || seq !== MG_CHAIN.seq) return;
      fired = true;
      openGate();
      mgComposeFrame(n);
    }

    /* ② 检测先发（3~20ms 的活）：回来就画框，然后才轮到合成 */
    var det = null;
    if (!opt.skipDetect) {
      try { det = window.mvDetect(n); } catch (e) { det = null; }
    }
    function afterDetect(res) {
      if (seq !== MG_CHAIN.seq) return;               /* 已经换到别的帧了：这一发作废 */
      if (res && res.ok && !(res.boxes || []).length) {
        fired = true;
        openGate();
        mgNoFaceFrame(n);                             /* ③-a 没人脸：不合成，回原图 */
        return;
      }
      composeOnce();                                  /* ③-b 有人脸：发合成 */
    }
    if (det && det.then) det.then(afterDetect, function () { afterDetect(null); });
    else afterDetect(null);
    MG_CHAIN.gateFrom = mgNowT();
    armGate();
    return det;

    function armGate() {
      MG_CHAIN.gate = setTimeout(function () {
        MG_CHAIN.gate = null;
        if (seq !== MG_CHAIN.seq) return;
        var waiting = !!(MG_DET.inflight && MG_DET.inflight === det);
        if (waiting && (mgNowT() - MG_CHAIN.gateFrom) < MG_DET_GATE_MAX) {
          armGate();                      /* 检测还在飞：再等一档 */
          return;
        }
        composeOnce();                    /* 检测挂了/太慢：照发，画面不能一直钉着 */
      }, MG_DET_GATE);
    }
  }
  window.mgFrameChange = mgFrameChange;

  /* 进页面 / 切回合成页：把当前帧补齐。同一帧已经画过框就不清（重进页面不该闪一下） */
  function mgCmpSync(opt) {
    opt = opt || {};
    var mv = window.__mv || {};
    if (!mv.path && !mgVidPath()) return;
    var n = mgCmpFrameIdx();
    if (MG_CHAIN.frame === n && MG_FRAME_T.tDraw) {
      MG_CHAIN.gateOpen = true;
      mgComposeFrame(n);
      return;
    }
    mgFrameChange(n, { why: "sync", skipDetect: !!opt.skipDetect });
  }
  window.mgCmpSync = mgCmpSync;

  /* ---------- 后端自动判定的结果 → 画面上每张脸的 chip ---------- */
  /* 合成回包里的 faces 是「这一帧每张脸被判给了哪个组/dfm + 相似度」的**权威答案**
     （auto=true 表示来自人脸库的 embedding 对比，false 表示用户手动指定过）。
     整份替换而不是合并：这一帧有几张脸、各自像谁，答案就在这一次响应里；
     上一帧的残留必须清掉，否则换人之后 chip 还挂着上一个模型。 */
  function mgApplyFacePlan(d) {
    var list = (d && d.faces) || [];
    if (!list.length) {
      /* 这一帧没有 faces = 走了单模型老路径（库里还没有组 / 老后端）。
         上一帧自动判出来的 chip 必须清掉，否则换人之后还挂着上个模型。 */
      if (d && d.ok && Object.keys(MG_FACE_AUTO).length) {
        MG_FACE_AUTO = {};
        mgSyncFaceCards();
        if (window.mvRedrawBoxes) window.mvRedrawBoxes();
      }
      return;
    }
    var next = {};
    list.forEach(function (f) {
      if (!f) return;
      var raw = (f.idx != null) ? f.idx : f.face;
      var i = parseInt(raw, 10);
      if (isNaN(i) || i < 0) return;
      var sim = (f.similarity == null) ? null : Number(f.similarity);
      next[i] = {
        dfm: String(f.dfm || f.model || ""),
        group: String(f.group || ""),
        groupName: String(f.group_name || ""),
        similarity: (sim == null || isNaN(sim)) ? null : sim,
        matched: f.matched !== false,
        auto: f.auto !== false
      };
    });
    MG_FACE_AUTO = next;
    mgSyncFaceCards();
    if (window.mvRedrawBoxes) window.mvRedrawBoxes();   /* 框色跟着判出来的模型走 */
  }
  window.mgApplyFacePlan = mgApplyFacePlan;

  /* ---------- 分配模型 → 立刻重算 ---------- */
  function mgAssignFace(i, name) {
    name = String(name || "");
    /* 手动指定 = 覆盖自动判定：把这一张从 MG_FACE_AUTO 里摘掉，
       否则下一次回包之前 chip 还会显示自动判出来的那个名字。 */
    delete MG_FACE_AUTO[i];
    if (name) { MG_FACE_MODEL[i] = name; delete MG_FACE_NOSWAP[i]; }
    else { delete MG_FACE_MODEL[i]; MG_FACE_NOSWAP[i] = true; }   /* 「不换脸」→ no_swap_faces */
    MG_FACE_BUSY[i] = true;
    /* 先用 canvas 换色 + 卡片换字（不等网络），再发重算 */
    window.mvRedrawBoxes();
    mgSyncFaceCards();
    var p = mgCmpNow();
    if (p && p.catch) p.catch(function () {});
  }
  window.mgAssignFace = mgAssignFace;

  /* ---------- 「＋」：把这张脸加进人脸数据库 ----------
     不再走 /api/merge/faces/extract（那个要 project、要整帧重检测，视频模式下
     没项目就直接 400）。现在点「＋」= 「我已经知道是哪张脸了，把这块存进库」：
     前端把这一帧的 68 点（/api/merge/detect 回包里就带着，和 extract 同一条
     2D106 链路）连同 frame/dfm 一起 POST /api/merge/faces/add，后端用**同一条
     68 点仿射**裁成该 dfm 的输入尺寸落盘。所以同一帧同一张脸，走 extract 和
     走 add 存下来的 PNG 逐像素相同。
     project 可有可无：没有就用默认库 <工作区>/merge，后端自己建。 */
  var MG_ADD = {};            /* 脸序号 → {busy, queue}：连点排队，一次只飞一发 */
  var MG_TOAST_SEEN = {};     /* 同一条提示 2 秒内只弹一次（错误不刷屏） */

  function mgToastOnce(key, title, sub, type, ms) {
    var now = Date.now();
    if (now - (MG_TOAST_SEEN[key] || 0) < 2000) return null;
    MG_TOAST_SEEN[key] = now;
    return toast(title, sub, type, ms);
  }
  window.mgToastOnce = mgToastOnce;

  /* 同一次操作里只允许出现**一条**错误：2 秒内的第二、第三条直接丢掉。
     以前「缺少 68 点」和「还没有素材」各用各的 key，两条会一起弹，还互相矛盾
     （画面里明明有视频）。现在「点＋」和「导出」的前置失败都走这里。 */
  var MG_ERR_AT = 0;
  function mgToastErr(key, title, sub, type, ms) {
    var now = Date.now();
    if (now - MG_ERR_AT < 2000) return null;
    if (now - (MG_TOAST_SEEN[key] || 0) < 2000) return null;
    MG_ERR_AT = now;
    MG_TOAST_SEEN[key] = now;
    /* toast() 是拼 innerHTML 的：服务端消息里带路径，尖括号别让它变成标签 */
    return toast(title, String(sub || "").replace(/[<>]/g, ""), type || "warn", ms || 5000);
  }
  window.mgToastErr = mgToastErr;

  /* 「现在到底能不能拿到这一帧的图」—— 问后端，不问变量。
     判定权在 /api/merge/faces/source：项目模式看项目帧、视频模式看合成器打开着
     的素材，两样都没有才算「没有素材」。以前只看 window.__mergeInputDir ——
     视频模式下它永远是空串。导出前的前置检查（mgNeedProjectDir）只认这个口径。
     结果缓存 1.2 秒（连点、重试不重复发请求）；
     老后端没这个接口就退回本地判断，绝不因为问不到就把用户拦下来。 */
  var MG_SRC = { at: 0, data: null, inflight: null };
  function mgSourceForget() {
    MG_SRC.at = 0; MG_SRC.data = null; MG_SRC.inflight = null;
  }
  window.mgSourceForget = mgSourceForget;

  function mgSrcHint(info) {
    info = info || {};
    var why = String(info.why || "");
    if (why) return why;                 /* 后端已经说了「当前是哪种模式、缺哪一个」 */
    if (info.mode === "project") return "当前是项目模式：项目里这一帧取不到图";
    if (info.mode === "video") return "当前是视频模式：还没打开素材 —— 先在左栏导入一段视频";
    return "当前既没有打开视频、也没有项目目录 —— 先导入一段视频或打开 DFL 项目";
  }

  function mgSourceInfo() {
    var now = Date.now();
    if (MG_SRC.data && now - MG_SRC.at < 1200) return Promise.resolve(MG_SRC.data);
    if (MG_SRC.inflight) return MG_SRC.inflight;
    var project = window.__mergeInputDir || "";
    var frame = mgCmpFrameIdx();
    MG_SRC.inflight = fetch("/api/merge/faces/source?project=" + encodeURIComponent(project) +
                            "&frame=" + frame)
      .then(function (r) { return r.json(); })
      .then(function (d) {
        return {
          ok: !!(d && d.ok !== false),
          available: !!(d && d.available),
          mode: String((d && d.mode) || ""),
          project: String((d && d.project) || ""),
          default_project: String((d && d.default_project) || ""),
          material: String((d && d.material) || ""),
          why: String((d && d.why) || ""),
          frame: frame
        };
      })
      .catch(function () {
        var video = mgVidPath();
        return { ok: false, available: !!(video || project),
                 mode: video ? "video" : (project ? "project" : "none"),
                 project: project, default_project: "", material: video,
                 why: "", frame: frame };
      })
      .then(function (info) {
        MG_SRC.data = info; MG_SRC.at = Date.now(); MG_SRC.inflight = null;
        return info;
      });
    return MG_SRC.inflight;
  }
  window.mgSourceInfo = mgSourceInfo;

  /* 导出（整片 / 标准项目）还要一个真正的项目目录：视频模式得先抽帧 + 对齐。 */
  function mgNeedProjectDir() {
    return mgSourceInfo().then(function (info) {
      var dir = String(info.project || window.__mergeInputDir || "");
      if (dir) return { info: info, dir: dir };
      if (!info.available) {
        mgToastErr("need-source", "还没有素材", mgSrcHint(info), "warn", 5000);
        return null;
      }
      mgToastErr("need-project", "当前是视频模式：还没有能导出的项目",
                 "视频还没抽帧 + 对齐成项目（先在左栏导入并等它准备完），导出要项目目录",
                 "warn", 6000);
      return null;
    });
  }
  window.mgNeedProjectDir = mgNeedProjectDir;


  function mgPlusBtn(i) {
    var layer = document.getElementById("mgFaceLayer");
    if (!layer) return null;
    return layer.querySelector('.mgfc[data-face="' + i + '"] .mgplus');
  }

  /* 连点 10 次要真的加 10 张（用户的意图），所以按钮不 disabled ——
     只是排队：请求在飞的时候点，就把这几次记进 queue，落地后接着发下一发。 */
  function mgAddFaceToDb(i, btn) {
    i = parseInt(i, 10);
    if (isNaN(i)) return;
    var q = MG_ADD[i] || (MG_ADD[i] = { busy: false, queue: 0 });
    q.queue += 1;
    var b0 = btn || mgPlusBtn(i);
    if (b0) b0.classList.add("busy");
    if (q.busy) return;
    mgAddFaceNext(i);
  }
  window.mgAddFaceToDb = mgAddFaceToDb;

  /* 后端给的业务失败（200 + ok:false）：一条提示、说清原因，绝不叠加第二条。 */
  function mgAddFaceFail(d) {
    var why = String((d && d.reason) || "");
    var detail = String((d && d.detail) || "").slice(0, 140);
    if (why === "no_landmarks68") {
      mgToastErr("add-no68", "这张脸没算出 68 点",
                 detail || "后端跑不出 2D106 的 68 点 —— 换一帧或重新检测再点", "warn", 5000);
    } else if (why === "no_face") {
      mgToastErr("add-noface", "这一帧没检测到脸", detail || "换一帧再点", "warn", 5000);
    } else if (why === "detect_failed") {
      mgToastErr("add-detfail", "服务端检测失败", detail || "检测器没跑起来", "error", 5000);
    } else if (why === "no_group_match") {
      /* 这张脸谁的组都不像（相似度都 < 阈值）→ 后端没落库，等用户手动指定 */
      mgToastErr("add-nogroup", "这张脸和现有分组都不像",
                 "请手动拖到某个组：先在框上的 chip 里给它选一个模型，再点＋",
                 "warn", 6000);
    } else {
      mgToastErr("add-" + why, "没有加到人脸库",
                 detail || why || "后端没有给出原因", "warn", 5000);
    }
  }

  function mgAddFaceNext(i) {
    var q = MG_ADD[i];
    if (!q) return;
    var btn = mgPlusBtn(i);
    if (q.queue <= 0) {
      if (btn) btn.classList.remove("busy");
      return;
    }
    q.queue -= 1;
    q.busy = true;
    if (btn) btn.classList.add("busy");

    var rec = MG_FACE_RAW[i] || null;
    var box = rec && rec.box ? rec.box.slice() : null;
    var kps68 = rec && rec.kps68 ? rec.kps68 : null;
    var has68 = !!(kps68 && kps68.length === 68);
    var frame = mgCmpFrameIdx();
    var video = mgVidPath() || "";

    function done() {
      q.busy = false;
      var b = mgPlusBtn(i);
      if (b && q.queue <= 0) b.classList.remove("busy");
      if (q.queue > 0) mgAddFaceNext(i);
    }

    /* 只把我们**真的知道**的东西发出去：
         有 kps68 → 直接带（快路径，前端算好的 68 点原样落库）；
         没有 / 不是 68 个 → 就不带，让后端从这一帧的图上自己跑
         「检测 → 2D106 → 68 点仿射裁剪」。
       绝不因为手里没有 kps68 就在这里弹错返回 —— 老后端（/api/merge/detect
       不回 kps68）时那样等于＋什么都没干，而用户看到的是「缺少 68 点」和
       「还没有素材」两条自相矛盾的提示（画面里明明有视频）。 */
    /* 落进哪一组**不由前端决定**（「提取到 <dfm>」下拉已经删掉）：
       后端会算这张脸的 embedding，和每个组的均值向量比余弦，像谁就归谁
       （响应里的 group / group_name / similarity / matched）。
       唯一带过去的偏好是「这张脸在框上的 chip 里被手动指定过模型」——
       那只是**匹配不上时的兜底**，不是目标组。 */
    var payload = {
      /* 有项目用项目库；没有留空 → 后端用默认库 <工作区>/merge 自己建 */
      project: window.__mergeInputDir || "",
      frame: frame,
      /* 同一帧只留一张脸：idx = 这张脸在这一帧里是第几张（多脸帧各存各的） */
      idx: i,
      detector: mgDetectorName() || "",
      source: { video: video, frame: frame }
    };
    var chip = String(MG_FACE_MODEL[i] || "");
    if (chip) payload.dfm = chip;
    if (has68) payload.kps68 = kps68;
    if (box) payload.box = box;

    fetch("/api/merge/faces/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (r) {
      return r.json().then(function (d) {
        /* no_face / no_landmarks68 / detect_failed / no_group_match 这类是业务结果
           （200 + ok:false），当成正常返回交给下面按 reason 说话，
           别报成「失败 HTTP 200」。 */
        if (d && d.ok === false && d.reason) return d;
        if (r.ok && d && d.ok !== false) return d;
        var err = new Error((d && (d.detail || d.error)) || ("HTTP " + r.status));
        err.status = r.status;
        throw err;
      }, function () {
        var err = new Error("HTTP " + r.status);
        err.status = r.status;
        throw err;
      });
    }).then(function (d) {
      done();
      if (d && d.ok === false) {
        mgAddFaceFail(d);
        return;
      }
      var faces = (d && d.faces) || [];
      var hit = faces[0] || null;
      var fid = (hit && hit.id) ? String(hit.id) : "";
      /* landmarks_from=server 说明这 68 点是后端自己补算的（前端手里没有） */
      var how = (d && d.landmarks_from === "server") ? " · 服务端补算 68 点" : "";
      /* 落进哪一组是后端按 embedding 判的：group_name 组名、similarity 余弦相似度。
         auto=true = 「和组均值比出来最像的那个」；chip 兜底时 source="chip"。 */
      var where = String((d && d.group_name) || "");
      var sim = (d && d.similarity != null) ? Number(d.similarity) : null;
      var why = (sim == null || isNaN(sim)) ? "" : (" · 相似度 " + sim.toFixed(2));
      var by = (d && d.auto) ? "（自动判定）"
             : ((d && d.source === "chip") ? "（按 chip 的模型）" : "");
      /* 后端按 (video, frame, idx) 去重：已有就回 replaced=true —— 那是**更新**，
         不能再报「已加入」（否则连点 5 次会看到 5 条「已加入」却只有 1 张脸）。 */
      if (d && d.replaced) {
        toast("已更新该帧的人脸", fid + (where ? " · " + where : "") + by + why + how,
              "success");
      } else {
        toast("已加入人脸数据库", (where ? fid + " · " + where : fid) + by + why +
              (d && d.created ? " · 新建了库" : "") + how, "success");
        /* 角标只数「新增」：更新没有让库变大，不该涨 */
        mgDbBadgeBump(1);
      }
      /* chip 立刻显示「这张脸被判给了谁」，不等下一次整帧重算 */
      if (where && d && d.auto) {
        MG_FACE_AUTO[i] = { dfm: String(d.group_dfm || ""), group: String(d.group || ""),
                            groupName: where, similarity: sim, matched: true, auto: true };
        mgSyncFaceCards();
        if (window.mvRedrawBoxes) window.mvRedrawBoxes();
      }
      mgFlyToDb(i, box);
      /* 人脸库开着就地刷新，别让用户以为没加进去 */
      try { if (window.FaceDB && window.FaceDB.refresh && window.FaceDB.isOpen()) window.FaceDB.refresh(); } catch (e) {}
    }).catch(function (e) {
      done();
      /* 拿不到图时后端会把「当前是视频模式/项目模式、缺的是哪一个」写进 detail */
      mgToastErr("add-err", "添加失败", String((e && e.message) || e).slice(0, 140), "error");
    });
  }
  window.mgAddFaceNext = mgAddFaceNext;

  function mgDbBadgeBump(delta) {
    var el = document.getElementById("mgDbBadge");
    if (!el) return 0;
    var n = Math.max(0, (parseInt(el.textContent, 10) || 0) + (delta == null ? 1 : delta));
    el.textContent = String(n);
    el.hidden = n === 0;
    return n;
  }
  window.mgDbBadgeBump = mgDbBadgeBump;

  /* 从当前合成帧里裁这一块当飞行缩略图（纯 dataURL，画布不会跨域污染） */
  function mgCropThumb(i) {
    try {
      var img = document.getElementById("mgPreview");
      if (!img || !img.naturalWidth || img.style.display === "none") return "";
      var r = MG_RECTS[i];
      if (!r) return "";
      var stage = document.getElementById("mgStage");
      var sr = stage.getBoundingClientRect();
      var ix = img.getBoundingClientRect();
      /* 和 object-fit: contain 用同一套映射：屏幕坐标 → 图片像素 */
      var s = Math.min(ix.width / img.naturalWidth, ix.height / img.naturalHeight);
      var ox = ix.left + (ix.width - img.naturalWidth * s) / 2;
      var oy = ix.top + (ix.height - img.naturalHeight * s) / 2;
      var sx = ((sr.left + 6 + r.x) - ox) / s;
      var sy = ((sr.top + 6 + r.y) - oy) / s;
      var sw = r.w / s, sh = r.h / s;
      var S = 44;
      var cv = document.createElement("canvas");
      cv.width = S; cv.height = S;
      var ctx = cv.getContext("2d");
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, S, S);
      return cv.toDataURL("image/png");
    } catch (e) { return ""; }
  }

  /* 飞入动画：WAAPI，不引库。缩略图从框的位置抛物线飞向顶部「人脸数据库」入口。 */
  function mgFlyToDb(i, box) {
    var entry = document.getElementById("mgDbEntry");
    var stage = document.getElementById("mgStage");
    if (!entry || !stage) return null;
    var sr = stage.getBoundingClientRect();
    var r = MG_RECTS[i] || { x: 0, y: 0, w: 40, h: 40 };
    var S = 44;
    var from = {
      left: sr.left + 6 + r.x + Math.max(0, (r.w - S) / 2),
      top: sr.top + 6 + r.y + Math.max(0, (r.h - S) / 2)
    };
    var to = entry.getBoundingClientRect();
    var fly = document.createElement("div");
    fly.className = "mgfly";
    fly.id = "mgFlyThumb";
    var src = mgCropThumb(i);
    if (src) {
      var im = document.createElement("img");
      im.src = src;
      im.alt = "";
      fly.appendChild(im);
    } else {
      fly.classList.add("ph");
    }
    fly.style.left = Math.round(from.left) + "px";
    fly.style.top = Math.round(from.top) + "px";
    document.body.appendChild(fly);

    var dx = (to.left + to.width / 2) - (from.left + S / 2);
    var dy = (to.top + to.height / 2) - (from.top + S / 2);
    if (fly.animate) {
      var anim = fly.animate([
        { transform: "translate(0,0) scale(1)", opacity: 1, offset: 0 },
        { transform: "translate(" + (dx * 0.62) + "px," + (dy * 0.62 - 34) + "px) scale(.66)", opacity: 1, offset: 0.6 },
        { transform: "translate(" + dx + "px," + dy + "px) scale(.1)", opacity: 0, offset: 1 }
      ], { duration: 640, easing: "cubic-bezier(.4,0,.2,1)" });
      anim.onfinish = function () { if (fly.parentNode) fly.parentNode.removeChild(fly); };
      /* 落点回弹：告诉眼睛「加到这个入口了」 */
      if (entry.animate) {
        try {
          entry.animate([{ transform: "scale(1)" }, { transform: "scale(1.1)" }, { transform: "scale(1)" }],
                        { duration: 300, delay: 560, easing: "ease-out" });
        } catch (e) {}
      }
    } else {
      setTimeout(function () { if (fly.parentNode) fly.parentNode.removeChild(fly); }, 640);
    }
    return fly;
  }
  window.mgFlyToDb = mgFlyToDb;

  /* ---------- 事件绑定（合成器主视图） ---------- */
  function mgBindCmp() {
    var page = document.getElementById("p-merge");
    if (!page) return;

    /* 点框切这张脸的合成/原图：舞台上的指针判定（阈值 4px），只绑一次 */
    if (window.mgBindFacePick) window.mgBindFacePick();
    /* 六宫格上的「返回」/ 点格放大：只绑一次 */
    if (window.mgFaceDebugBind) window.mgFaceDebugBind();

    /* 「模式」下拉（4 个融合方式）和右边那个两档分段共用同一个值：动哪个，
       另一个跟着高亮。以前 blend 只读分段、下拉写 mode，后端以 blend 为准 ——
       下拉里选 hist-match 画面一动不动。 */
    var modeEl = document.getElementById("mgMode");
    if (modeEl && !modeEl.dataset.blendBound) {
      modeEl.dataset.blendBound = "1";
      modeEl.addEventListener("change", function () {
        mgSyncBlendSeg(modeEl.value);
        /* 融合方式动了 → 合成后端那两档的可用性要跟着重判（seamless 时 gpu 选不了） */
        if (window.mgGateCmpBackend) window.mgGateCmpBackend();
      });
      mgSyncBlendSeg(modeEl.value);
      if (window.mgGateCmpBackend) window.mgGateCmpBackend();
    }
    var bseg = page.querySelector('.seg[data-k="blend"]');
    if (bseg && !bseg.dataset.blendBound) {
      bseg.dataset.blendBound = "1";
      bseg.addEventListener("click", function (e) {
        var b = e.target && e.target.closest ? e.target.closest("b") : null;
        if (!b || !bseg.contains(b)) return;
        var v = b.getAttribute("data-v") || "";
        var sel = document.getElementById("mgMode");
        if (sel && MG_MERGE_BLENDS.indexOf(v) >= 0) sel.value = v;
      });
    }

    /* 参数实时预览：右栏任何控件动一下就防抖重算 */
    var right = page.querySelector(".mgright");
    if (right && !right.dataset.cmpBound) {
      right.dataset.cmpBound = "1";
      right.addEventListener("change", mgCmpSoon);
      right.addEventListener("input", mgCmpSoon);
      right.addEventListener("click", function (e) {
        var b = e.target && e.target.closest ? e.target.closest(".seg b") : null;
        if (!b) return;
        var seg = b.closest(".seg[data-k]");
        if (!seg) return;
        var k = seg.getAttribute("data-k");
        if (k === "blend" || k === "mask" || k === "seg" || k === "detect") mgCmpSoon();
      });
    }

    /* 点页面空白处收起 chip 下拉 */
    if (!page.dataset.cmpDocBound) {
      page.dataset.cmpDocBound = "1";
      document.addEventListener("click", function () { mgCloseChipMenus(null); });
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") mgCloseChipMenus(null);
      });
    }

    /* 顶部「人脸数据库」入口：**不再**在这里判「有没有素材」。
       project 空着是合法的（视频模式）—— 后端把空串解析成默认库 <工作区>/merge，
       真打不开时由 face_db.js 把后端的 msg 弹出来。以前这里只判 window.__mergeInputDir：
       视频模式下它永远是空串，于是画面里明明有视频，点「人脸数据库」也弹
       「还没有素材」（「＋」那边再弹一条就自相矛盾了）。 */
    var entry = document.getElementById("mgDbEntry");
    if (entry && !entry.dataset.bound) {
      entry.dataset.bound = "1";
      entry.addEventListener("click", function () {
        if (!window.FaceDB || !window.FaceDB.open) {
          toast("人脸数据库没有加载", "face_db.js 版本不匹配，刷新页面再试", "error");
          return;
        }
        window.FaceDB.open({
          project: window.__mergeInputDir || "",   /* 空 = 后端落到默认库 */
          mode: "export",
          allowExtract: true,
          onCancel: function () { if (window.mgStep) window.mgStep("main"); }
        });
      });
    }


    /* 色域迁移存着老值（lut / idt / sot-m …）时回落到 mt，别让下拉空着 */
    var color = document.getElementById("mgColor");
    if (color) {
      var ok = Array.prototype.some.call(color.options, function (o) { return o.value === color.value; });
      if (!ok) {
        color.value = "mt";
        try { color.dispatchEvent(new Event("change", { bubbles: true })); } catch (e) {}
      }
    }

    /* 模式下拉只剩 4 个**融合方式**：本地存着老值（debug / original / raw-*）时
       回落到 seamless，别让下拉空着、也别把视图名当融合方式发给后端（会 400）。 */
    var modeSel = document.getElementById("mgMode");
    if (modeSel) {
      var okm = Array.prototype.some.call(modeSel.options, function (o) { return o.value === modeSel.value; });
      if (!okm) {
        modeSel.value = "seamless";
        try { modeSel.dispatchEvent(new Event("change", { bubbles: true })); } catch (e) {}
      }
    }
  }
  window.mgBindCmp = mgBindCmp;

  /* 遮罩绘制：把 MaskProcessor 的 WebUI 直接嵌进来 */
  function mkOpenEditor() {
    var inp = document.getElementById("mkDir");
    var frame = document.getElementById("mkFrame");
    var empty = document.getElementById("mkEmpty");
    var st = document.getElementById("mkState");
    if (!inp || !frame) return;
    var dir = String(inp.value || "").trim();
    if (st) { st.classList.remove("on"); st.innerHTML = "<i></i>连接中…"; }
    fetch("/api/project/open", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: dir })
    }).then(function (r) {
      if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || "打开失败"); });
      return r.json();
    }).then(function (d) {
      frame.src = "/mask-editor/index.html?dir=" + encodeURIComponent(d.path);
      if (empty) empty.classList.add("hide");
      if (st) { st.classList.add("on"); st.innerHTML = "<i></i>" + d.count + " 张"; }
      toast("已打开", d.path + " · " + d.count + " 张", "success");
    }).catch(function (e) {
      if (st) { st.classList.remove("on"); st.innerHTML = "<i></i>未连接"; }
      toast("打开失败", String((e && e.message) || e), "error");
    });
  }
  window.mkOpenEditor = mkOpenEditor;

  function bindMaskEditor() {
    var btn = document.getElementById("mkOpen");
    if (btn && !btn.dataset.bound) {
      btn.dataset.bound = "1";
      btn.addEventListener("click", mkOpenEditor);
    }
    var inp = document.getElementById("mkDir");
    if (inp && !inp.dataset.bound) {
      inp.dataset.bound = "1";
      inp.addEventListener("keydown", function (e) { if (e.key === "Enter") mkOpenEditor(); });
    }
  }
  window.bindMaskEditor = bindMaskEditor;

  function bindMergeUI() {
    var page = $("p-merge");
    if (!page) return;

    /* 时间轴：每次进合成页都重绘一次（元素可能被重建），幂等。
       是「重进页面」还是「换素材」交给 mgTimelineSync 的 key=(path|总帧数) 判断：
       相同 → keep（保住播放头和人脸轨），不同 → 整条重建。
       没素材时 total 就是 0 —— 以前用 300 兜底，画出来的是假帧。 */
    setTimeout(function () {
      window.mvList();
      var mv = window.__mv || {};
      mgTimelineSync(mv.path ? (mv.frames || 0) : 0, mv.fps || 30);
      /* 视频模式：已有的视频要接回上次停下的位置（window.__mv.t） */
      if (window.mgResumeVideo) window.mgResumeVideo();
      /* 进合成页时若已经有视频/素材：用户多半马上会暂停看效果，先热身。
         每次进页面都判一次；重复请求由 __mgWarm 的 10 秒去重挡住 */
      /* 选模型即加载：不等打开视频。模型选好了、视频还没选，也应该在后台把 DFM 装进显存
         （多选时每个选中模型都热一遍，__mgWarm 自带 10 秒去重） */
      window.mgWarmSelected();
      /* 内容不足一屏时把栈拉满，刻度尺才不会只铺一半 */
      var st = document.querySelector(".mtl-stack");
      var sc = document.querySelector(".mtl-scroll");
      if (st && sc) st.style.width = Math.max(st.scrollWidth, sc.clientWidth) + "px";
      /* 合成视图：chip / 参数实时预览的绑定（幂等），有素材就把框和画面都补齐 */
      if (window.mgBindCmp) window.mgBindCmp();
      if (window.mgApplyView) window.mgApplyView();
      if (window.mgCmpSync) window.mgCmpSync();
      /* 舞台空态：干净工作区进来 = 中央一句话 + 两个入口；已有素材 = 收起 */
      if (window.mgStageEmpty) window.mgStageEmpty();
    }, 60);

    if (page.dataset.bound) return;
    page.dataset.bound = "1";
    /* 视频模式：播放键 / 空格交给 <video> 的时钟（幂等，重复进入不会重复绑） */
    if (window.mgBindTransport) window.mgBindTransport();
    page.querySelectorAll(".seg").forEach(function (seg) {
      seg.addEventListener("click", function (e) {
        var b = e.target.closest("b");
        if (!b || !seg.contains(b)) return;
        Array.prototype.forEach.call(seg.children, function (x) { x.classList.remove("on"); });
        b.classList.add("on");
      });
    });
    /* 「合成 / 原图」不再有分段控件，也不再是整帧视图：
       画面永远是「合成图压在原片上」，点某张脸的框只把那张脸切回原像素
       （= 请求体里的 no_swap_faces，见 mgToggleFaceRaw / mgBindFacePick）。 */

    /* 导出流程：工作台（预览/剪辑）→ 人脸库（分组）→ 导出设置 → 导出中 */
    var MG_STEPS = ["main", "facedb", "settings"];
    function mgStep(name) {
      var step = document.getElementById("mgStep");
      var back = document.getElementById("mgBack");
      page.querySelectorAll(".mgview").forEach(function (v) {
        v.classList.toggle("in", v.dataset.view === name);
      });
      if (step) {
        var label = { facedb: "人脸库", settings: "导出设置",
                      project: "导出项目", progress: "导出进度" }[name] || "";
        step.textContent = label;
        step.hidden = !label;
      }
      if (back) back.hidden = (name === "main");
      /* 面包屑跟着走：合成工作室 › 人脸数据库 › 导出设置 */
      var trail = [{ t: "合成工作室", step: "main" }];
      if (name === "facedb") trail.push({ t: "人脸数据库", step: "facedb" });
      if (name === "settings") trail = [
        { t: "合成工作室", step: "main" },
        { t: "导出设置", step: "settings" }
      ];
      if (name === "progress") trail = [
        { t: "合成工作室", step: "main" },
        { t: "导出设置", step: "settings" },
        { t: "导出进度", step: "progress" }
      ];
      if (name === "project") trail = [
        { t: "合成工作室", step: "main" },
        { t: "导出项目", step: "project" }
      ];
      CRUMBS.merge = trail;
      renderCrumbs("merge");
      window.__mergeStep = name;
      if (window.SUB_MEMORY) window.SUB_MEMORY["merge"] = null;   /* 合成器用 mgStep，不走 go */
      /* 一律交给 MergePages.show：name==="main" 时它会清 .mgp-on，
      否则回主视图后主容器还被 .mgp-on 的 CSS 藏着 */
      if (window.MergePages) window.MergePages.show(name);
      document.querySelector("#p-merge .mgbody") &&
        (document.querySelector("#p-merge .mgbody").style.display = (name === "main") ? "" : "none");
    }
    window.mgStep = mgStep;

    var backBtn2 = document.getElementById("mgBack");
    if (backBtn2 && !backBtn2.dataset.bound) {
      backBtn2.dataset.bound = "1";
      backBtn2.addEventListener("click", function () {
        var cur = (page.querySelector(".mgview.in") || {}).dataset;
        var at = cur ? cur.view : "main";
        /* 人脸库这一页归 face_db.js 管：必须走它的 requestClose()（=「返回会取消
           本次导出？」那条确认），直接 mgStep 会把 state.open 留在 true ——
           导出流程里等于绕过确认，浏览模式又说不上「已经关掉」。 */
        if (at === "facedb" && window.FaceDB && window.FaceDB.requestClose) {
          window.FaceDB.requestClose();
          return;
        }
        var idx = MG_STEPS.indexOf(at);
        mgStep(idx <= 1 ? "main" : "facedb");
      });
    }

    /* 两条导出都先经过人脸库：整理完点「继续导出」才回调真正发请求。
       face_db.js 自己不发 /export；它只把控制权交回这里，参数组装仍然只有下面
       mgRunExport / mgpjStart 这一份，避免两套导出契约。 */
    function mgOpenFaceDb(mode, proceed, project) {
      project = String(project || window.__mergeInputDir || "");
      if (!project) {
        mgToastErr("open-db-noproj", "还没有项目目录",
                   "当前是视频模式：后端没给能放人脸库的目录（先导入一段视频或打开 DFL 项目）",
                   "warn", 5000);
        return;
      }
      if (!window.FaceDB || !window.FaceDB.open) {
        toast("人脸库没有加载", "face_db.js 版本不匹配，刷新页面再试", "error");
        return;
      }
      window.FaceDB.open({
        project: project,
        mode: mode,
        allowExtract: true,
        onContinue: proceed,
        onCancel: function () { if (window.mgStep) window.mgStep("main"); }
      });
    }

    /* ---------- 导出：先设置页，再开跑 ----------
       以前的「导出」是**直接开跑**：点下去就 mgRunExport，参数全是默认值，想改只能
       先手动摸到导出设置页去（还得知道有那一页）。现在入口只做两件事：
       ① 校验「有没有选模型」（0 个 = 按钮本来就是禁用的，这里再兜一层）；
       ② 把这一页的只读回显填好，切到导出设置页。**一个请求都不发。**
       真正 POST 阶段请求的是下面 #mgStartExport 那一下。 */
    function mgDefOutPath() {
      /* 只认「输出路径」框里用户手填的值。这里不能读 MergePages.settings().output_path：
         那条现在会把「输出目录」里改过的值也当 output_path 返回，默认值就会跟着自己跑
         （改一版 → 默认值变成上一版 → 再改又叠一层）。 */
      var outEl = document.getElementById("mgOutPath");
      var manual = String((outEl && outEl.value) || "").trim();
      if (manual) return manual;
      var dir = String(window.__mergeInputDir || "");
      if (dir) {
        var base = dir.split(/[\\/]/).filter(Boolean).pop() || "project";
        return dir.replace(/[\\/]+$/, "") + "\\" + base + "_merged.mp4";
      }
      var video = mgVidPath();
      if (video) return video.replace(/\.[^./\\]+$/, "") + "_merged.mp4";
      return "（默认：源视频名 _merged.mp4，与源视频同目录）";
    }

    /* 打开/刷新导出设置页：只读回显 + 当前参数，不动任何请求 */
    function mgFillExportSettings() {
      var names = mgSelNames();
      var box = document.getElementById("mgSelModels");
      if (box) {
        box.innerHTML = "";
        if (!names.length) {
          var em = document.createElement("span");
          em.className = "mgexp-models-empty";
          em.textContent = "还没选模型";
          box.appendChild(em);
        } else {
          names.forEach(function (n) {
            var chip = document.createElement("span");
            chip.className = "mgexp-chip";
            chip.textContent = n;
            box.appendChild(chip);
          });
        }
      }
      var out = document.getElementById("mgDefOut");
      if (out) {
        out.value = mgDefOutPath();
        /* 每次进导出设置页都把「用户改过没有」清掉：这一格刚被重算成默认值，
           继续把它当用户填的目录发出去，就会悄悄改掉上一次的输出位置。 */
        window.__mgDefOutEdited = false;
      }
      var total = 0;
      try {
        total = parseInt(String((window.__mv || {}).frames || 0), 10) || 0;
      } catch (e) { total = 0; }
      if (!total) {
        var tf = document.getElementById("mtlTotal");
        total = parseInt(String((tf && tf.textContent) || "0"), 10) || 0;
      }
      /* 右边那颗「全片」按钮本身就是「恢复整段」，这行字只补「一共多少帧」；
         总帧数还不知道（视频还没探测完）就干脆不显示，别写成「全片 全片」。 */
      var tot = document.getElementById("mgFrameTotal");
      if (tot) {
        if (total > 0) { tot.textContent = "全片 · " + total + " 帧"; tot.hidden = false; }
        else { tot.textContent = ""; tot.hidden = true; }
      }
      var to = document.getElementById("mgFrameTo");
      if (to) to.placeholder = total > 0 ? String(Math.max(0, total - 1)) : "末帧";
      var cnt = document.getElementById("mgSettingsCount");
      if (cnt) {
        cnt.textContent = names.length
          ? (names.length + " 个模型 · 全片")
          : "没有选中模型";
      }
      if (window.MergePages && window.MergePages.syncStatus) window.MergePages.syncStatus();
      mgSyncExportGate();
    }
    window.mgFillExportSettings = mgFillExportSettings;

    function mgOpenExportSettings() {
      if (!mgSyncExportGate()) {
      mgDflModeSync();   /* 素材可能刚换过：进设置页前刷一次「沿用 DFL」的可用性 */
        mgToastErr("export-nosel", "先选一个模型",
                   "左栏「模型」里点一个（可多选），导出要用它换脸", "warn", 4000);
        return false;
      }
      mgFillExportSettings();
      mgStep("settings");
      return true;
    }
    window.mgOpenExportSettings = mgOpenExportSettings;

    /* ---------------- 检测策略：决定「导出」走哪条流水线 ----------------
       · 总是重检   从源视频重新检测（融合流水线）
       · 沿用 DFL   用同名文件夹里现成的 aligned/（DFL 项目模式，不重新检测）
       能不能选「沿用 DFL」由后端一处判（/api/merge/dfl-mode → _dfl_mode_state）：
       同名文件夹 + aligned/ + 至少一张帧图。别在前端自己猜路径。 */
    var MGPJ_UNAVAILABLE = "";
    function mgDflSeg() {
      return document.querySelector('#p-merge .seg[data-k="detect"]');
    }
    function mgDflModeSync(done) {
      var seg = mgDflSeg();
      if (!seg) { if (done) done(null); return; }
      var bs = Array.prototype.slice.call(seg.querySelectorAll("b"));
      var reuse = bs[1];
      if (!reuse) { if (done) done(null); return; }
      var q = [];
      var vp = (typeof mgVidPath === "function") ? String(mgVidPath() || "") : "";
      if (vp) q.push("video=" + encodeURIComponent(vp));
      if (window.__mergeInputDir) q.push("path=" + encodeURIComponent(window.__mergeInputDir));
      fetch("/api/merge/dfl-mode" + (q.length ? ("?" + q.join("&")) : ""))
        .then(function (r) { return r.json(); })
        .then(function (d) {
          var st = (d && d.data) || d || {};
          MGPJ_UNAVAILABLE = st.ok ? "" : String(st.why || "不可用");
          reuse.classList.toggle("dis", !st.ok);
          reuse.setAttribute("aria-disabled", st.ok ? "false" : "true");
          reuse.title = st.ok
            ? ("DFL 项目模式：用 " + st.dir + " 里现成的 aligned/，不重新检测")
            : ("不能选：「" + MGPJ_UNAVAILABLE + "」。要这一档请先抽帧 + 人脸对齐，或改用「总是重检」");
          if (!st.ok && reuse.classList.contains("on")) {
            reuse.classList.remove("on");
            if (bs[0]) bs[0].classList.add("on");
          }
          if (done) done(st);
        })
        .catch(function () { if (done) done(null); });
    }
    window.mgDflModeSync = mgDflModeSync;

    (function bindDflSeg() {
      var seg = mgDflSeg();
      if (!seg || seg.dataset.dflBound) return;
      seg.dataset.dflBound = "1";
      /* 捕获阶段拦住「点不可用的那一格」：分段的高亮是通用逻辑加的，
         这里必须在它之前把事件吃掉，否则会出现"亮着但用不了"。 */
      seg.addEventListener("click", function (e) {
        var b = e.target && e.target.closest ? e.target.closest("b") : null;
        if (b && b.classList.contains("dis")) {
          e.stopPropagation();
          e.preventDefault();
          mgToastErr("pj-unavailable", "不能选「沿用 DFL」", MGPJ_UNAVAILABLE || "还没有可用的 DFL 项目", "warn", 6000);
          return;
        }
        mgDflModeSync();
      }, true);
    })();
    /* 首屏进来也同步一次（素材可能在上次会话里就打开了） */
    setTimeout(function () { try { mgDflModeSync(); } catch (e) {} }, 1200);

    var exBtn = $("mgExport");
    if (exBtn && !exBtn.dataset.bound) {
      exBtn.dataset.bound = "1";
      exBtn.addEventListener("click", function () {
        /* 只切页；不解析项目、不请求后端 —— 断言「点导出 → 0 个 /api/merge/stage」靠这一条 */
        mgOpenExportSettings();
      });
    }
