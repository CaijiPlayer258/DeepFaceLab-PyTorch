/* 界面文案的中英映射。按"整句"匹配，不改 DOM 结构。
 * 只翻译含中文的文本节点；切回中文时从 orig 里恢复原值。
 */
(function () {
  "use strict";
  var DICT = {
  "主页": "Home",
  "提取器": "Extractor",
  "数据处理": "Processor",
  "训练器": "Trainer",
  "合成器": "Merger",
  "工具": "Tools",
  "设置": "Settings",
  "关于": "About",
  "上；导航选中、焦点环、输入框、卡片全部走中性灰阶。": "— nav selection, focus rings, inputs and cards stay on the neutral ramp.",
  "进度条": "progress bars",
  "（开始提取 / 开始训练 / 继续训练）和": "(Start extraction / Start training / Resume) and",
  "主按钮": "primary buttons ",
  "主题色只出现在": "Theme color only appears on",
  "天蓝": "Sky",
  "暗紫灰": "Violet muted",
  "淡紫蓝": "Lavender",
  "紫（深）": "Violet deep",
  "紫（亮）": "Violet",
  "绿": "Green",
  "红": "Red",
  "琥珀黄": "Amber",
  "8 色盘点缀 · 单个控件里不超过 2 个 · 语义固定（悬停看用途）": "8-color accent set · max 2 per control · fixed meaning (hover for usage)",
  "辅助色": "Accents",
  "只用于标签和图标，不做大面积背景、不进正文（浅 / 深）": "Tags and icons only — no large fills, never in body text (light / dark)",
  "状态色": "Status",
  "同一套 token 的深色取值": "Same tokens, dark values",
  "中性 · 深色": "Neutral · dark",
  "表面层次与文字，界面主体全靠这一组拉开": "Surfaces and text — the whole UI's depth comes from this set",
  "中性 · 浅色": "Neutral · light",
  "9 档色阶（OKLCH 生成，50 = 种子）· 仅用于主按钮与进度条": "9 tones (OKLCH, 50 = seed) · primary buttons and progress bars only",
  "品牌色": "Brand",
  "Linear 色系 · 种子 #5D6CCF · 规范见 UI设计规范.md §1": "Linear palette · seed #5D6CCF · spec in UI设计规范.md §1",
  "色板": "Palette",
  "提取器完工：检测器 RetinaFace_10g / TinyMog 双档、动态分辨率对齐、越界填黑、帧跳跃语义修正（1 = 逐帧）、四格实时统计": "Extractor shipped: dual detectors (RetinaFace_10g / TinyMog), dynamic-resolution alignment, black padding, frame-skip fixed (1 = every frame), four live stats",
  "界面布局与交互设计完成：参数页四段式（标签 · 说明 · 控件）、吸顶搜索、吸底操作条、下拉框主题化、按架构过滤冻结层、设备下拉直显 GPU 型号": "UI layout & interaction done: 4-part param rows (label · hint · control), sticky search, sticky action bar, themed dropdowns, arch-filtered freeze layers, GPU names in the device picker",
  "按完成时间倒序": "Newest first",
  "里程碑": "Milestones",
  "关于界面": "Interface build",
  "深色模式": "Dark mode",
  "浅色模式": "Light mode",
  "切换深浅色": "Toggle theme",
  "折叠侧边栏": "Collapse sidebar",
  "展开侧边栏": "Expand sidebar",
  "最小化": "Minimize",
  "最大化": "Maximize",
  "关闭": "Close",
  "开始一个新工程": "Start a new project",
  "界面重写": "UI rewrite",
  "天前": "days ago",
  "查看全部": "View all",
  "页面改为侧边栏仪表盘": "Sidebar dashboard layout",
  "数据处理改成三级渐进": "Processing uses progressive disclosure",
  "模块 → 模式 → 参数": "Modules → Modes → Parameters",
  "提取器只剩 5 个参数": "Extractor trimmed to 5 parameters",
  "检测器固定为": "Detector fixed to",
  "所有 WebUI 端口集中到设置页": "All WebUI ports now live in Settings",
  "改完立即生效": "Takes effect immediately",
  "提取": "Extract",
  "训练": "Train",
  "合成": "Merge",
  "从视频或图片目录里裁出对齐后的人脸": "Crop aligned faces from videos or image folders",
  "让模型学会你给的那张脸": "Teach the model the face you gave it",
  "把训练好的脸贴回原视频": "Paste the trained face back onto the video",
  "继续上次": "Continue last",
  "继续训练": "Resume training",
  "最近的数据集": "Recent datasets",
  "张": "images",
  "人脸提取": "Face extraction",
  "路径": "Paths",
  "输入": "Input",
  "输出": "Output",
  "选择目录": "Choose folder",
  "参数": "Parameters",
  "质量档": "Quality",
  "高质量": "High",
  "快速": "Fast",
  "检测角度": "Rotation",
  "帧跳跃": "Frame skip",
  "开始提取": "Start extraction",
  "正在处理 · 视频与图片混合目录": "Processing · mixed video/image folder",
  "检测帧速度": "Detection FPS",
  "输出脸速度": "Faces / sec",
  "文件进度": "Files",
  "张/秒": "faces/s",
  "就绪": "Ready",
  "当前数据集": "Current dataset",
  "换数据集": "Change dataset",
  "打开一个文件夹以继续": "Open a folder to continue",
  "选择已对齐的人脸集目录": "Choose an aligned faceset folder",
  "打开数据集": "Open dataset",
  "最近": "Recent",
  "过滤器": "Filter",
  "排序器": "Sorter",
  "角度可视化": "Angle view",
  "遮罩应用": "Apply mask",
  "模糊 · 坏图 · 重复": "Blur · Broken · Duplicates",
  "清晰度 · 角度 · 时间": "Sharpness · Angle · Time",
  "偏航 · 俯仰": "Yaw · Pitch",
  "推理 · 平滑": "Inference · Smooth",
  "模糊筛选": "Blur filter",
  "重复图": "Duplicates",
  "合并子文件夹": "Merge subfolders",
  "错脸": "Wrong face",
  "坏图": "Broken",
  "清晰度": "Sharpness",
  "角度": "Angle",
  "时间": "Time",
  "散点图": "Scatter",
  "直方图": "Histogram",
  "XSeg 推理": "XSeg inference",
  "遮罩平滑": "Mask smoothing",
  "强度档位": "Strength",
  "处理方式": "Action",
  "移到 _trash": "Move to _trash",
  "直接删除": "Delete",
  "训练分辨率": "Train resolution",
  "分箱数": "Bins",
  "模型": "Model",
  "高": "High",
  "中": "Medium",
  "低": "Low",
  "平滑强度": "Smoothing",
  "阈值": "Threshold",
  "分段数": "Segments",
  "分组数": "Groups",
  "目标分辨率": "Target resolution",
  "判定阈值": "Detect threshold",
  "相似阈值": "Similarity threshold",
  "预计筛掉 96 张，剩 246 张": "About 96 will be filtered, 246 left",
  "按清晰度重命名 342 张": "Rename 342 files by sharpness",
  "在浏览器里打开可视化": "Open the visualization in your browser",
  "为 342 张生成遮罩": "Generate masks for 342 images",
  "开始处理": "Start",
  "全部": "All",
  "训练中": "Training",
  "已暂停": "Paused",
  "刚新建": "Just created",
  "未开始": "Not started",
  "搜索模型名": "Search models",
  "新建模型": "New model",
  "开始训练": "Start training",
  "删除": "Delete",
  "合成服务未启动": "Merge service stopped",
  "合成服务运行中": "Merge service running",
  "启动后在浏览器里打开工作台": "The workspace opens in your browser",
  "模型在网页里再选": "Models are picked inside the web workspace",
  "启动合成器": "Start merger",
  "停止服务": "Stop service",
  "模型导出": "Model export",
  "已导出 DFM": "DFM exported",
  "未导出": "Not exported",
  "导出中 62%": "Exporting 62%",
  "导出": "Export",
  "重新导出": "Re-export",
  "已在浏览器打开": "opened in your browser",
  "遮罩绘制": "Mask editor",
  "手工修 XSeg 遮罩": "Hand-fix XSeg masks",
  "改完直接写回数据集": "Writes straight back to the dataset",
  "端口": "Port",
  "打开遮罩绘制": "Open mask editor",
  "B 站下载": "Bilibili downloader",
  "下 B 站视频当素材": "Download Bilibili videos as source material",
  "登录后才能拿高码率": "Sign in to get high bitrate",
  "已登录": "Signed in",
  "打开下载器": "Open downloader",
  "色域迁移": "Color conversion",
  "杜比视界 / HDR 先转成普通色域再做提取": "Convert Dolby Vision / HDR to SDR before extraction",
  "不再内置 HDR 提取": "HDR extraction is no longer built in",
  "基于 FFmpeg · 输出到同目录": "Powered by FFmpeg · writes next to the source",
  "选择视频": "Choose video",
  "全局设置": "Settings",
  "WebUI 端口": "WebUI ports",
  "服务": "Service",
  "状态": "Status",
  "可用": "Available",
  "未启动": "Stopped",
  "端口被占用时自动顺延到下一个空闲端口": "Falls back to the next free port when occupied",
  "这里显示实际生效的号": "The number shown is the one actually in use",
  "B 站账号": "Bilibili account",
  "有效期至": "valid until",
  "清除 Cookie 并重新登录": "Clear cookie and sign in again",
  "GitHub 代理": "GitHub proxy",
  "自动更新走这个代理": "Updates go through this proxy",
  "留空则直连 —— 国内直连通常连不上": "Leave empty to connect directly",
  "建议用": "Recommended: paste the",
  "的系统代理并填这里": "system proxy address here",
  "语言": "Language",
  "中文": "中文",
  "English": "English",
  "版本": "Version",
  "界面": "UI",
  "推理": "Inference",
  "协议": "License",
  "作者": "Author",
  "自研控件": "custom widgets",
  "提取完成": "Extraction finished",
  "个文件": "files",
  "后端未连接": "Backend not connected",
  "请通过 app/engine/main.py 启动": "Launch through app/engine/main.py",
  "合成服务已启动": "Merge service started",
  "合成服务已停止": "Merge service stopped",
  "正在提取": "Extracting…"
};

  var orig = new WeakMap();
  var attrOrig = new WeakMap();
  var lang = "zh";

  function hasCJK(s) { return /[\u4e00-\u9fa5]/.test(s); }

  /* 统一查表：字典 → 去空白索引 → 正则。
     属性（title / placeholder / value）和文本节点必须走同一套，
     否则「帧 0 ~ 8」这类动态 tooltip 永远翻不到。 */
  function lookup(raw) {
    var one = String(raw).trim().replace(/\s+/g, " ");
    var hit = DICT[one] || SPACELESS[one.replace(/\s+/g, "")];
    if (hit) return hit;
    for (var pi = 0; pi < PATTERNS.length; pi++) {
      if (PATTERNS[pi][0].test(one)) return one.replace(PATTERNS[pi][0], PATTERNS[pi][1]);
    }
    return null;
  }

  /* 动态文案：数字会变，只能按模式替换 */
  var PATTERNS = [
    [/^(\d[\d,]*)\s*个模型 · 最新 (\S+) · iter ([\d,]+)$/, "$1 models · latest $2 · iter $3"],
    [/^(\d[\d,]*)\s*个已导出 DFM$/, "$1 DFM exported"],
    [/^停在 iter ([\d,]+)，上次训练于 (\d+) 小时前。$/, "Stopped at iter $1, last trained $2h ago."],
    [/^停在 iter ([\d,]+)，上次训练于 (\d+) 天前。$/, "Stopped at iter $1, last trained $2d ago."],
    [/^停在 iter ([\d,]+)，上次训练于 (\d+) 分钟前。$/, "Stopped at iter $1, last trained $2m ago."],
    [/^正在配置 (.+)，确认后开始训练。$/, "Configuring $1 — confirm to start training."],
    [/^当前 iter ([\d,]+) · (.+)$/, "iter $1 · $2"],
    [/^当前 (\d+) 条$/, "$1 messages"],
    [/^张$/, "images"],
    [/^(\d[\d,]*)\s*张$/, "$1 images"],
    [/^(\d[\d,]*)\s*个$/, "$1 items"],
    [/^(\d+)\s*秒前$/, "$1s ago"],
    [/^(\d+)\s*分钟前$/, "$1m ago"],
    [/^(\d+)\s*小时前$/, "$1h ago"],
    [/^(\d+)\s*天前$/, "$1d ago"],
    [/^(\d+)\s*\/\s*(\d+)\s*帧$/, "$1 / $2 frames"],
    [/^(\d+)\s*张人脸$/, "$1 faces"],
    [/^已建库\s*(\d+)\s*张人脸$/, "$1 faces built"],
    [/^(\d+)\s*个模型 · 最新 (\S+) · iter ([\d,]+)$/, "$1 models · latest $2 · iter $3"],
    [/^(\d+)\s*个已导出 DFM$/, "$1 DFM exported"],
    [/^已跑 (\S+)$/, "Running $1"],
    [/^约 (\d+)\s*小时剩余$/, "~$1 h left"],
    [/^显存 ([\d.]+) \/ (\d+) GB$/, "VRAM $1 / $2 GB"],
    [/^剩 (\d+) GB · 模型 ([\d.]+) GB$/, "$1 GB free · model $2 GB"],
    [/^(\d+)\s*分钟前$/, "$1m ago"],
    [/^(\d+)\s*分钟后$/, "in $1m"],
    [/^(\d+)\s*\/\s*(\d+)\s*份$/, "$1 / $2"],
    [/^bs (\d+) · 显示 (\d+) 行$/, "bs $1 · $2 rows"],
    [/^(\d+)\s*条$/, "$1 messages"],
    [/^当前 (\d+)\s*条$/, "$1 messages"],
    [/^界面构建 (.+) · 客户端 (.+)$/, "Build $1 · client $2"],
    [/^(.+) · 客户端 (.+)$/, "$1 · client $2"],
    [/^已启用 · 端口 (\d+)$/, "Enabled · port $1"],
    [/^剪切 帧 (\d+) ~ (\d+)$/, "Cut frames $1 ~ $2"],
    [/^人脸 (\d+)（占位）$/, "Face $1 (placeholder)"],
    [/^角度 ([\d,]+)$/, "Angles $1"],
    [/^帧 (\d+) ~ (\d+)$/, "Frames $1 ~ $2"],
    [/^人脸 帧 (\d+) ~ (\d+)$/, "Face frames $1 ~ $2"],
    [/^个模型 · 最新 (.+) · iter ([\d,]+)$/, "models · latest $1 · iter $2"],
    [/^个已导出 DFM$/, "exported DFM"],
    [/^个已导出 DFM · 最新 (.+)$/, "exported DFM · latest $1"],
    [/^(\d+)\s*张 · 散点图$/, "$1 points · scatter"],
    [/^(\d[\d,]*)\s*张$/, "$1 images"],
    [/^晚上好，(.+)$/, "Good evening, $1"],
    [/^早上好，(.+)$/, "Good morning, $1"],
    [/^下午好，(.+)$/, "Good afternoon, $1"],
    [/^停在 iter ([\d,]+)，上次训练于 (.+?)前。$/, "Stopped at iter $1, last trained $2 ago."],
    [/^继续训练 (.+)$/, "Resume $1"],
    [/^已跑 (.+)$/, "Running $1"],
    [/^剩余 (.+)$/, "$1 left"],
    [/^当前 (\d+)\s*条$/, "$1 messages"]
  ];

  /* 去空白的二级索引：跨行写的文案归一化后会在「、」「：」后面留一个空格，
     单空格 key 匹配不上。用去空白版本再查一次，命中率就上来了。 */
  var SPACELESS = {};
  Object.keys(DICT).forEach(function (k) {
    var sk = k.replace(/\s+/g, "");
    if (sk && !SPACELESS[sk]) SPACELESS[sk] = DICT[k];
  });

  /* 数字 + 量词（"342 张" / "0 个" / "12 帧"）是拼出来的，字典永远对不上。
     这类按"数字后面的量词"整体替换，一条规则覆盖全站的计数文案。 */
  var UNITS = { "张": "images", "个": "items", "帧": "frames", "秒": "s",
                "分钟": "min", "小时": "h", "万": "0k" };
  function transUnits(t) {
    return t.replace(/(\d[\d,.]*)\s*(张|个|帧|秒|分钟|小时)/g, function (_, n, u) {
      return n + " " + (UNITS[u] || u);
    });
  }

  function translateText(lang, rootEl) {
    var walker = document.createTreeWalker(rootEl || document.body, NodeFilter.SHOW_TEXT, null, false);
    var n;
    while ((n = walker.nextNode())) {
      var raw = n.nodeValue;
      if (raw == null) continue;
      /* 语言切换器这类控件必须原样显示（"中文" 永远是 "中文"），
         带 data-i18n-skip 或位于 #wLang 内的节点不参与翻译 */
      var host = n.parentElement;
      if (host && host.closest && host.closest("#wLang,[data-i18n-skip]")) continue;
      var stored = orig.get(n);

      if (lang !== "en") {
        /* 还原时必须用"有没有存过原文"判断，不能看当前文本含不含中文 ——
           翻译成英文之后节点里已经没有汉字了，用 hasCJK 过滤会永远跳过它，
           于是切回中文时什么都没恢复（这正是"能切英文、切不回中文"的原因）。*/
        if (stored !== undefined && stored !== raw) n.nodeValue = stored;
        continue;
      }

      if (!hasCJK(raw)) continue;
      if (stored === undefined) orig.set(n, raw);
      /* 多行写的说明在文本节点里带 \n 和缩进，直接按 raw.trim() 匹配永远对不上 ——
         先把连续空白压成一个空格再查。 */
      var hit = lookup(raw);
      var one = raw.trim().replace(/\s+/g, " ");
      if (hit) {
        var lead = raw.match(/^\s*/)[0];
        var trail = raw.match(/\s*$/)[0];
        n.nodeValue = lead + hit + trail;
      }
    }
  }

  function translateAttrs(lang) {
    /* input 的 value 不算文本节点 —— 路径框里的占位文案就是 value，必须单独处理 */
    document.querySelectorAll("input[value]").forEach(function (el) {
      var cur = el.value;
      if (!cur) return;
      if (!el.dataset.origValue && hasCJK(cur)) el.dataset.origValue = cur;
      if (lang === "en") {
        var hit = lookup(cur);
        if (hit) el.value = hit;
      } else if (el.dataset.origValue) {
        el.value = el.dataset.origValue;
      }
    });

    document.querySelectorAll("[title],[placeholder]").forEach(function (el) {
      var rec = attrOrig.get(el) || {};
      ["title", "placeholder"].forEach(function (a) {
        var cur = el.getAttribute(a);
        if (cur == null) return;
        if (rec[a] == null) rec[a] = cur;
        if (lang === "en") {
          var hit = lookup(cur);
          if (hit) el.setAttribute(a, hit);
        } else {
          el.setAttribute(a, rec[a]);
        }
      });
      attrOrig.set(el, rec);
    });
  }

  function apply(lang2) {
    lang = (lang2 === "en") ? "en" : "zh";
    document.documentElement.setAttribute("lang", lang === "en" ? "en" : "zh-CN");
    translateText(lang);
    translateAttrs(lang);
    if (lang === "en") watchDynamic();
    try { window.localStorage.setItem("dfl-lang", lang); } catch (e) {}
    /* 遮罩编辑器是 iframe，独立文档，只能靠消息把语言推过去 */
    var mk = document.getElementById("mkFrame");
    if (mk) { try { mk.contentWindow.postMessage({ dflLang: lang }, "*"); } catch (e) {} }
    return lang;
  }

  Object.assign(DICT, {
    "把一张脸，训练成能换到任何视频里的模型。": "Train one face into a model you can swap into any video.",
    "简洁的交互界面": "A clean interface",
    "所有流程都在一个窗口里完成，不用记命令。": "The whole pipeline lives in one window, no commands to memorise.",
    "更快的训练": "Faster training",
    "预览与 loss 实时流出，不用等一轮跑完。": "Live preview and loss, without waiting for a full run.",
    "更自然的合成": "More natural merging",
    "遮罩、色域、角度逐帧校正，边缘不糊。": "Mask, colour and angle corrected per frame.",
    "更好的数据处理": "Better data handling",
    "张数、对齐质量、角度分布一目了然。": "Face counts, alignment quality and angle spread at a glance.",
    "最快的人脸提取": "Fastest face extraction",
    "RetinaFace 与 TinyMog 双检测器，批量对齐。": "RetinaFace and TinyMog extractors, batched alignment.",
    "更多实用工具": "More useful tools",
    "遮罩绘制、色域迁移、角度分析、B 站下载，都在同一个侧边栏。": "Mask painting, colour transfer, angle analysis and Bilibili download, all in one sidebar.",
    "二维码": "QR code",
    "用 B 站 App 扫码登录": "Sign in with the Bilibili app",
    "登录后可以下载高码率素材，主页也会记住你的名字。": "Sign in to fetch high-bitrate sources and have the home page remember you.",
    "模拟扫码登录": "Simulate scan",
    "跳过，先看看": "Skip for now"
  });


  /* 全站补译（2026-10-22）：把界面上所有残留中文补齐 */
  Object.assign(DICT,   {
    "退出": "Sign out",
    "累计迭代": "Total iterations",
    "已对齐人脸": "Aligned faces",
    "导出 DFM": "Exported DFM",
    "数据集": "Datasets",
    "打开": "Open",
    "管理": "Manage",
    "最近活动": "Recent activity",
    "帧": "frame",
    "正在处理": "Processing",
    "准备中…": "Preparing…",
    "未扫描": "Not scanned",
    "模式": "Mode",
    "按清晰度阈值挑出糊片": "Drop blurry frames by a sharpness threshold",
    "等待扫描": "Waiting for scan",
    "越清晰排越前": "Sharper first",
    "用训练好的遮罩模型推理": "Run the trained mask model",
    "最近 2500 iter": "last 2500 iter",
    "按 UP 主 UID 拉取视频列表，勾选后批量下载成训练素材。": "Fetch a creator's video list by UID, tick the ones you want and batch-download them as source material.",
    "用户搜索": "Creator lookup",
    "未查询": "Not queried",
    "UP 主 UID": "Creator UID",
    "获取视频列表": "Fetch video list",
    "下载设置": "Download settings",
    "Cookie 有效至 2027-03": "Cookie valid until 2027-03",
    "保存路径": "Save to",
    "画质": "Quality",
    "最高": "Best",
    "下载线程": "Threads",
    "视频列表": "Videos",
    "已选": "Selected",
    "下载选中": "Download selected",
    "标题": "Title",
    "BV 号": "BV id",
    "合作": "Collab",
    "暂无视频数据，请先搜索 UID": "No videos yet — look up a UID first",
    "下载中": "Downloading",
    "空闲": "Idle",
    "暂停": "Pause",
    "终止": "Stop",
    "B 站下载的实时队列，离开这页也会继续跑。": "Live Bilibili download queue — keeps running if you leave.",
    "队列": "Queue",
    "当前没有任务": "No tasks right now",
    "杜比视界 / HDR 先转成普通色域再做提取。4.1 不再内置 HDR 提取。": "Convert Dolby Vision / HDR to a standard gamut before extraction. 4.1 no longer bundles HDR extraction.",
    "源视频": "Source video",
    "与源文件同目录，后缀": "Same folder as the source, suffix",
    "编码": "Codec",
    "开始转换": "Start conversion",
    "基于 FFmpeg": "Powered by FFmpeg",
    "把对齐后人脸的偏航 / 俯仰画成图，看数据集的角度分布是否均衡。": "Plot yaw / pitch of aligned faces to see whether the angle distribution is balanced.",
    "图表": "Chart",
    "格子图": "Grid",
    "服务端口": "Service ports",
    "分布": "Distribution",
    "偏航 yaw": "Yaw",
    "俯仰": "Pitch",
    "调试": "Debug",
    "3D 人脸编辑器": "3D face editor",
    "拖动旋转 · 点选特征点": "Drag to orbit · click a landmark",
    "特征点": "Landmarks",
    "保存": "Save",
    "重置": "Reset",
    "导出 JSON": "Export JSON",
    "通知调试": "Toast debug",
    "内容": "Content",
    "右下角气泡": "Bottom-right toast",
    "副标题": "Subtitle",
    "持续（毫秒）": "Duration (ms)",
    "类型": "Type",
    "成功": "Success",
    "信息": "Info",
    "警告": "Warning",
    "错误": "Error",
    "触发": "Fire",
    "发一条": "Send one",
    "连发 3 条": "Send 3",
    "连发 5 条": "Send 5",
    "超长文案": "Long text",
    "全部清空": "Clear all",
    "参数设置": "Parameters",
    "训练前确认参数。带 · 的是本次会话才会生效的项。": "Confirm before training. Items marked · only apply to this session.",
    "恢复默认": "Restore defaults",
    "正式训练": "Training",
    "隐藏": "Hide",
    "立即保存": "Save now",
    "停止": "Stop",
    "最近 100 均": "last 100 avg",
    "利用率": "Utilisation",
    "磁盘": "Disk",
    "学习率": "Learning rate",
    "上次保存": "Last save",
    "下次自动": "Next auto",
    "备份": "Backups",
    "预览": "Preview",
    "视频": "Video",
    "还没有导入视频": "No video imported",
    "＋ 导入视频": "＋ Import video",
    "原图": "Source",
    "检测": "Detect",
    "选择视频后显示": "Shown after a video is chosen",
    "高精度 · RetinaFace": "Accurate · RetinaFace",
  "高性能 · TinyMog": "Fast · TinyMog",
  "检测器": "Detector",
    "人脸类型": "Face type",
    "检测策略": "Detect policy",
    "总是重检": "Always re-detect",
    "沿用 DFL": "Reuse DFL",
    "遮罩来源": "Mask from",
    "遮罩模式": "Mask mode",
    "遮罩腐蚀": "Mask erode",
    "遮罩羽化": "Mask blur",
    "人脸": "Faces",
    "剪切": "Cuts",
    "人脸库": "Face DB",
    "已建库": "Built",
    "张人脸": "faces",
    "计算人脸特征": "Compute embeddings",
    "下一步": "Next",
    "还没有建立人脸库": "No face database yet",
    "导出设置": "Export settings",
    "输出 · 编码 · 性能": "Output · Codec · Performance",
    "开始导出": "Start export",
    "格式": "Format",
    "质量": "Quality",
    "输出路径": "Output path",
    "编码器": "Encoder",
    "解码加速": "HW decode",
    "开": "On",
    "关": "Off",
    "性能": "Performance",
    "并行进程": "Workers",
    "0 = 自动": "0 = auto",
    "正在导出": "Exporting",
    "人脸检测": "Face detection",
    "封装": "Mux",
    "载入模型": "Loading model",
    "逐帧合成": "Frame merge",
    "完成": "Done",
    "夜深了": "It's late",
    "不限": "unlimited",
    "遮罩": "Mask",
    "合并": "Merged",
    "刷新": "Refresh",
    "连接中…": "Connecting…",
    "还没有预览图 —— 训练跑起来后点「刷新」": "No preview yet — refresh once training is running",
    "中午好": "Good noon",
    "GPU 利用率": "GPU usage",
    "CPU": "CPU",
    "内存": "RAM",
    "取消导出": "Cancel export",
    "隐私密码": "Privacy password",
    "密码": "Password",
    "隐藏模式会用它解锁。只存在本机浏览器里。": "Unlocks hidden mode. Stored only in this browser.",
    "Cookie 有效期至 2027-03": "Cookie valid until 2027-03",
    "4.0 是技术样机，4.1 是产品。": "4.0 was a tech demo. 4.1 is a product.",
    "后端": "Backend",
    "工作区": "Workspace",
    "正在获取…": "Fetching…",
    "刷新二维码": "Refresh QR code",
    "确认": "OK",
    "取消": "Cancel",
    "确定": "Confirm",
    "这是一段私密内容 · 只有你能打开": "This is private · only you can open it",
    "解锁": "Unlock"
  });


  /* 参数页补译 */
  Object.assign(DICT, {
  "模型选择": "Models",
  "参数设置": "Parameters",
  "训练界面": "Training",
  "遮罩绘制": "Mask editor",
  "B 站下载": "Bilibili",
  "色域迁移": "Colour transfer",
  "角度可视化": "Angle view",
  "关于": "About",
  "调试": "Debug",
  "下载中": "Downloading",
  "合成工作室": "Merge studio",
  "人脸数据库": "Face database",
  "导出设置": "Export settings",
  "导出进度": "Export progress",
  "模型架构参数": "Architecture",
  "数据集参数": "Datasets",
  "训练参数": "Training",
  "优化器参数": "Optimizer",
  "GAN 训练参数": "GAN",
  "补丁参数": "Patch",
  "数据增强参数": "Augmentation",
  "冻结层参数": "Frozen layers",
  "src 人脸集": "src faceset",
  "dst 人脸集": "dst faceset",
  "批次大小": "Batch size",
  "目标迭代数": "Target iterations",
  "自动保存间隔": "Autosave interval",
  "训练精度": "Precision",
  "梯度检查点": "Gradient checkpointing",
  "梯度裁剪": "Gradient clipping",
  "优化器放显存": "Optimizer on GPU",
  "快速生成器": "Fast generator",
  "写入预览历史": "Keep preview history",
  "预训练": "Pretrain",
  "预训练单解码器": "Pretrain single decoder",
  "崩溃阈值": "Crash threshold",
  "Inter code 分布日志": "Inter-code stats log",
  "最大备份数": "Max backups",
  "优化器": "Optimizer",
  "学习率": "Learning rate",
  "学习率策略": "LR policy",
  "余弦周期": "Cosine cycles",
  "学习率总步数": "LR total steps",
  "平台衰减系数": "Plateau factor",
  "平台耐心": "Plateau patience",
  "平台最小比例": "Plateau min ratio",
  "LR Dropout": "LR dropout",
  "PIA 周期": "PIA cycle",
  "GAN 强度": "GAN power",
  "GAN 块大小": "GAN patch size",
  "GAN 维度": "GAN dims",
  "真脸强度": "True face power",
  "面部风格强度": "Face style power",
  "背景风格强度": "Background style power",
  "旋转范围": "Rotation range",
  "缩放范围": "Scale range",
  "偏移范围": "Shift range",
  "随机扭曲": "Random warp",
  "随机 HSV 强度": "Random HSV power",
  "颜色迁移": "Colour transfer",
  "随机遮挡": "Random occluder",
  "随机噪点": "Random noise",
  "偏色算法": "Colour degrade",
  "偏色强度": "Colour power",
  "边缘增强": "Edge enhance",
  "眼睛缩放": "Eyes scale",
  "冻结编码器": "Freeze encoder",
  "冻结遮罩解码器": "Freeze mask decoder",
  "冻结 inter_AB": "Freeze inter_AB",
  "冻结 inter_B": "Freeze inter_B",
  "遮罩训练": "Masked training",
  "嘴眼优先": "Eyes/mouth priority",
  "备份间隔": "Backup interval",
  "VGG感知": "VGG perceptual",
  "嘴巴范围": "Mouth range",
  "SRC翻转": "SRC flip",
  "DST翻转": "DST flip",
  "均匀偏航": "Uniform yaw",
  "冻结中间": "Freeze inter",
  "冻结输出": "Freeze out",
  "中文": "Chinese",
  "创建后不可修改": "Fixed after creation",
  "模型名称": "Model name",
  "分辨率": "Resolution",
  "正在配置": "Configuring",
  "确认后开始训练。": "— confirm to start training.",
  "训练前确认参数。带 · 的是本次会话才会生效的项。": "Confirm before training. Items marked · apply to this session only."
});


  /* 参数说明补译 */
  Object.assign(DICT, {
  "每次训练的样本数量": "Samples per step",
  "0 表示无限训练": "0 means train indefinitely",
  "4.0 无此项，仅为 UI 占位": "Not in 4.0 — UI placeholder only",
  "省显存，略慢": "Saves VRAM, slightly slower",
  "防止梯度爆炸": "Prevents exploding gradients",
  "关掉可省显存": "Turn off to save VRAM",
  "预览图存到 _history": "Write previews into _history",
  "只练通用特征": "Train generic features only",
  "预训练时只用一个解码器": "Use a single decoder while pretraining",
  "loss 超过它视为崩溃，不覆盖备份": "Treat loss above this as a crash and keep the backup",
  "0 = 关闭，每 N 迭代输出一次": "0 = off, otherwise log every N iterations",
  "0 = 关闭": "0 = off",
  "1.0（关闭）": "1.0 (off)",
  "连续 N 轮不降才衰减": "Decay after N epochs without improvement",
  "0 = 关闭 GAN": "0 = GAN off",
  "CodeDiscriminator 真脸损失权重": "True-face loss weight for CodeDiscriminator",
  "风格迁移中面部的权重": "Face weight in style transfer",
  "风格迁移中背景的权重": "Background weight in style transfer",
  "训练参数 · 补充": "Training · extras",
  "4.0 training_config_page 同名项": "Same key as 4.0's training_config_page",
  "启用后只在遮罩区域内计算损失": "Compute loss inside the mask only",
  "对输入的 mask 外周做高斯模糊": "Gaussian blur on the mask border",
  "优先训练嘴部和眼部区域": "Prioritise mouth and eye regions",
  "每 N 次迭代自动保存一次，0 = 关闭": "Autosave every N iterations, 0 = off",
  "补丁参数 · 补充": "Patch · extras",
  "VGG16 特征 L1 损失，0~100，50 约等于与重建损失等权": "VGG16 feature L1 loss, 0–100; 50 ≈ equal weight with reconstruction",
  "数据增强 · 补充": "Augmentation · extras",
  "配合嘴眼优先，1.0 = 原始范围，0.25~2.0": "Works with eyes/mouth priority; 1.0 = original range, 0.25–2.0",
  "src 图像随机水平翻转": "Randomly flip src images horizontally",
  "dst 图像随机水平翻转": "Randomly flip dst images horizontally",
  "侧脸少的 faceset 用，让 yaw 分布均匀": "For facesets with few profiles — evens out the yaw distribution",
  "冻结层 · 补充": "Frozen layers · extras",
  "按模型架构生效": "Applies per architecture",
  "SAEHD / DFSingle / PIA 的 DF 架构 inter": "inter for the DF architecture in SAEHD / DFSingle / PIA",
  "冻结 decoder_dst，仅 DF 架构有这一支": "Freeze decoder_dst — only the DF architecture has it",
  "冻结源": "Freeze src",
  "仅 AMP 的源中间层": "src inter layer of AMP only",
  "冻结目标": "Freeze dst",
  "仅 AMP 的目标中间层": "dst inter layer of AMP only",
  "冻结瓶颈": "Freeze bottleneck",
  "仅 DeepFakeLarge 的 Bottleneck": "Bottleneck of DeepFakeLarge only",
  "冻结A遮罩": "Freeze A mask",
  "仅 DeepFakeLarge 的 src 解码器遮罩分支": "src decoder mask branch of DeepFakeLarge only",
  "冻结B遮罩": "Freeze B mask",
  "仅 DeepFakeLarge 的 dst 解码器遮罩分支": "dst decoder mask branch of DeepFakeLarge only",
  "冻结共享": "Freeze shared",
  "仅 LIAELarge 的共享瓶颈 Bottleneck_AB": "Shared Bottleneck_AB of LIAELarge only",
  "冻结瓶颈B": "Freeze bottleneck B",
  "仅 LIAELarge 的目标瓶颈 Bottleneck_B": "dst Bottleneck_B of LIAELarge only",
  "PIA · 补充": "PIA · extras",
  "仅 PIA 生效": "PIA only",
  "预训练集": "Pretrain set",
  "设备": "Device",
  "4.0 对应 --force-gpu-idxs / --cpu-only": "Maps to 4.0's --force-gpu-idxs / --cpu-only",
  "auto = 自动选卡；cpu = 纯 CPU 训练；0,1 = 指定多张卡（逗号分隔）": "auto = pick a GPU; cpu = CPU only; 0,1 = specific GPUs (comma separated)",
  "需要 select": "Needs a select",
  "预览调色": "Preview grading",
  "只影响「合并预览」第 5 列的显示，不影响训练和真正的合成": "Only affects column 5 of the merge preview — training and final merging are untouched",
  "自动更新走这个代理。留空则直连 —— 国内直连通常连不上，建议用 Steam++ 的系统代理并填这里。": "Auto-update goes through this proxy. Leave empty to connect directly — usually unreachable from mainland China; use the Steam++ system proxy and fill it in here."
});

  Object.assign(DICT, {
    "还没有选择文件": "No file selected",
    "还没有导入视频": "No video imported",
    "选择视频": "Choose video",
    "路径": "Path",
    "基于 FFmpeg": "Powered by FFmpeg",
    "与源文件同目录，后缀": "Same folder as the source, suffix",
    "开始转换": "Start conversion",
    "源视频": "Source video",
    "输出": "Output",
    "编码": "Codec"
  });

  /* ==========================================================================
     4.1 新增界面（新建模型面板 / 模型结构只读卡 / 续训卡 / 参数搜索）的文案。
     这些块是 JS 动态生成的，静态 html 里搜不到，只能在这里补；配合各渲染函数
     末尾的 I18N.refresh() 生效。
     ========================================================================== */
  Object.assign(DICT, {
    "搜索模型名…": "Search models…",
    "搜索参数（例如 遮罩 / batch / 学习率 / 翻转）": "Search parameters (mask / batch / lr / flip)",

    "新建模型 · 结构": "New model · Structure",
    "定完结构，点下面的「创建模型」就会开始训练": "Set the structure, then hit Create model to start training",
    "名字": "Name",
    "模型目录名，建完就固定": "Model folder name — fixed once created",
    "架构": "Architecture",
    "DF：双解码器　LIAE：双 inter。Lite 版换掉 5×5 稠密卷积":
      "DF: two decoders　LIAE: two inters. Lite swaps the 5×5 dense convs",
    "子分支": "Sub-branch",
    "u = 像素归一化　d = 可学习上采样　t = 编码器多一次下采样":
      "u = pixel norm　d = learnable upsample　t = one more encoder downsample",
    "算子": "Kernel",
    "Lite 架构的卷积实现；dw5 最快且保住 5×5 感受野":
      "Conv implementation for Lite; dw5 is fastest and keeps a 5×5 receptive field",
    "分辨率": "Resolution",
    "对齐后的人脸边长，越大越吃显存": "Aligned face edge length — larger costs more VRAM",
    "编码器宽度": "Encoder width",
    "ae_dims：瓶颈层宽度，决定“能装下多少长相”":
      "ae_dims: bottleneck width — how much identity it can hold",
    "编码器维度": "Encoder dims",
    "e_dims：encoder 内部通道": "e_dims: encoder channels",
    "解码器维度": "Decoder dims",
    "d_dims：decoder 内部通道": "d_dims: decoder channels",
    "遮罩维度": "Mask dims",
    "mask：解码器里遮罩分支的通道数": "mask: channels in the decoder's mask branch",
    "脸型": "Face type",
    "训练时按哪种脸对齐裁切，改等于重做数据集":
      "Which face crop the set was aligned to — changing it means redoing the data",
    "退出新建": "Exit",
    "创建模型": "Create model",
    "启动中…": "Starting…",

    "模型结构": "Model structure",
    "创建后不可修改，要换结构请新建模型":
      "Fixed after creation — create a new model to change it",
    "已训练": "Trained",
    "上次训练": "last trained",
    "还没有预览图（跑一会儿就有了）": "no preview yet (it appears once training runs)",

    "半脸（只要脸，不要额头下巴）": "Half face (no forehead or chin)",
    "中全脸": "Mid-full face",
    "全脸（默认）": "Full face (default)",
    "整头（含额头和下巴，推荐）": "Whole face (forehead + chin, recommended)",
    "头肩（要脖子和肩膀）": "Head & shoulders",

    "original - 原版（5×5 稠密，与 DF 完全一致）": "original - 5×5 dense (identical to DF)",
    "dw5 - DW5×5+PW1×1 / Rep3×3（默认，最快）": "dw5 - DW5×5+PW1×1 / Rep3×3 (default, fastest)",
    "dw5+stride - 额外压缩瓶颈空间投影（参数最少）":
      "dw5+stride - compressed bottleneck projection (fewest params)",
    "3x3 - 稠密 3×3 / Rep3×3（可折叠，感受野 3×3）":
      "3x3 - dense 3×3 / Rep3×3 (foldable, 3×3 receptive field)",
    "叠加 - DW5×5+identity 可折叠（感受野 5×5）":
      "stacked - DW5×5+identity foldable (5×5 receptive field)",
    "多尺度 - DW5×5+DW3×3 双分支可折叠": "multi-scale - DW5×5+DW3×3 dual branch, foldable"
  });

  PATTERNS.push(
    [/^按你这张卡的可用显存，建议 (\d+) 分辨率 \/ batch (\d+)（训练显存约 ([\d.]+|-) GB）。显卡 (.+)$/,
      "This card: $1 / batch $2 (~$3 GB VRAM). GPU $4"],
    [/^按你这张卡的可用显存，建议 (\d+) 分辨率 \/ batch (\d+)（训练显存约 ([\d.]+|-) GB）$/,
      "This card: $1 / batch $2 (~$3 GB VRAM)"],
    [/^([\d,]+) iter · (\S+) · 权重 (\S+)$/, "$1 iter · $2 · weights $3"],
    [/^([\d,]+) iter · (\S+)$/, "$1 iter · $2"]
  );

  Object.assign(DICT, {
    "一键体检": "One-shot checkup",
    "强度": "Strength",
    "严格": "Strict",
    "标准": "Normal",
    "宽松": "Loose",
    "建议清理": "Suggested to remove",
    "清理中…": "Cleaning…",
    "扫描中…": "Scanning…",
    "总数": "Total",
    "模糊": "Blur",
    "纯色 / 空图": "Flat / empty",
    "曝光异常": "Exposure",
    "打不开": "Unreadable",
    "重复阈值": "Duplicate threshold",
    "清晰度阈值": "Sharpness threshold"
  });

  Object.assign(DICT, {
    "IQA 百分位（学习型）": "IQA percentile (learned)",
    "最差 5 张（分数 · 三维）": "Worst 5 (score · axes)"
  });

  Object.assign(DICT, {
    "姿态分布（按张数）": "Pose distribution (by count)",
    "左大": "far left", "左小": "slight left", "正面": "front",
    "右小": "slight right", "右大": "far right",
    "低头": "chin down", "平视": "level", "抬头": "chin up"
  });

  Object.assign(DICT, {
    "模型轨": "Score track",
    "关": "Off",
    "额外 5%": "+5%",
    "额外 10%": "+10%",
    "模型轨（规则之外额外）": "Score track (extra)",
    "稀缺角度救回": "Rescued by pose floor"
  });

  Object.assign(DICT, {
    "引擎编译": "Engine build",
    "全部编译": "Build all",
    "编译日志": "Build log",
    "已编译": "Ready",
    "已过期": "Stale",
    "未编译": "Not built",
    "无模型": "No model",
    "无 GPU": "No GPU",
    "编译": "Build",
    "（还没有编译任务）": "(no build yet)"
  });

  /* 全站 i18n 审计补的一批：设置页、合成器、提取器统计、若干 tooltip */
  Object.assign(DICT, {
    "4.0 值 jpg（Cache Format，JPEG 100%）": "4.0 used jpg (Cache Format, JPEG 100%)",
    "4.0 值 png（PNG 帧缓存）": "4.0 used png (PNG frame cache)",
    "4.0 固定 -q:v 2，无此参数": "4.0 hard-coded -q:v 2 — no such option",
    "留空 = 源视频名_result.mp4": "Empty = <source>_result.mp4",
    "选择输出目录": "Choose output folder",
    "NVENC 硬件编码（4.0 默认）": "NVENC hardware encoding (4.0 default)",
    "x264 软件编码": "x264 software encoding",
    "hwaccel='auto'，无 GPU 时自动回退软解": "hwaccel='auto' — falls back to software without a GPU",
    "hwaccel=''（关闭硬解）": "hwaccel='' (hardware decode off)",
    "num_workers · 0 = 自动 · 范围 0 ~ 16": "num_workers · 0 = auto · range 0 ~ 16",
    "请输入哔哩哔哩 UID（纯数字）…": "Enter a Bilibili UID (digits only)…",
    "342 个文件 · 用时 1 分 12 秒": "342 files · took 1m 12s",
    "构建": "Build",
    "帧速度": "Frame rate",
    "人脸速度": "Face rate",
    "已处理帧": "Frames done",
    "帧/秒": "fps",
    "素材": "Media",
    "还没有导入素材": "No media imported yet",
    "＋ 打开 DFL 项目": "＋ Open DFL project",
    "外观": "Appearance",
    "跟随系统": "System",
    "浅色": "Light",
    "深色": "Dark",
    "远程访问": "Remote access",
    "地址": "Address",
    "默认跟随系统；选过之后按你的来，下次启动也记住。": "Follows the system by default; once you pick one, it is remembered on next launch.",
    "开启后整个界面监听 0.0.0.0，同网段的设备（或部署在服务器上时）用浏览器直接打开。只在自己信任的网络里开。": "When on, the UI listens on 0.0.0.0 — devices on the same network (or any browser, if hosted) can open it. Only enable this on a network you trust.",
    "未设置时无法解锁隐藏模式": "Without it the hidden mode stays locked",
    "原始大小": "Original size",
    "引擎大小": "Engine size",
    "当前 0 条": "0 messages",
    "选择文件": "Choose file",
    "全选 / 取消全选": "Select all / none",
    "点一下换数据集": "Click to switch dataset",
    "点我": "Click me",
    "少一帧": "One frame less",
    "多一帧": "One frame more",
    "每 N 帧取一张（24fps 下 N 超过 120 等同于 0.2 fps 下限）": "Take one frame every N (above 120 at 24 fps it hits the 0.2 fps floor)",
    "按住左右拖动调值": "Drag left/right to change",
    "导出为 .dfm": "Export as .dfm",
    "范围 −100 ~ 100": "Range −100 ~ 100",
    "范围 0 ~ 总数-1": "Range 0 ~ count-1",
    "上一帧": "Previous frame",
    "播放 / 暂停": "Play / pause",
    "下一帧": "Next frame",
    "缩小": "Zoom out",
    "放大": "Zoom in",
    "警告 / 待处理 / 编译中": "Warning / pending / compiling",
    "错误 / 断开 / 停止": "Error / disconnected / stopped",
    "在线 / 成功 / 正常": "Online / success / normal",
    "渐变起点 / 强调": "Gradient start / emphasis",
    "渐变终点": "Gradient end",
    "滑块选中 / 高亮": "Slider active / highlight",
    "未激活的强调底": "Inactive emphasis fill",
    "链接 / 信息提示": "Link / info",
    "早上好": "Good morning",
    "上午好": "Good morning",
    "下午好": "Good afternoon",
    "晚上好": "Good evening"
  });

  /* 运行期文案 batch 6：保存提示、远程访问、杂项 */
  Object.assign(DICT, {
    "已保存 · ": "Saved · ",
    " 项即时生效": " applied immediately",
    "要重启训练才生效：": "Needs a training restart:",
    "全部立即生效": "All applied immediately",
    "已保存，但没推给训练进程": "Saved, but not pushed to the training process",
    "参数和文件里的一致": "Parameters match the file",
    "下次启动训练时生效": "Takes effect the next time training starts",
    "拿不到模型列表": "Cannot get the model list",
    "点开始训练没反应": "Clicking Start does nothing",
    "已应用 · 立即生效": "Applied · takes effect immediately",
    "改动 ": "Changed ",
    " 项，": " items, ",
    " 项要重启才生效": " need a restart",
    "全部即时生效": "All take effect immediately",
    "参数已保存": "Parameters saved",
    " · 改动 ": " · changed ",
    " 项": " items",
    "训练已启动": "Training started",
    " · 4.0 进程已拉起": " · the 4.0 process is up",
    "训练未启动": "Training did not start",
    "已启用 · 端口 ": "Enabled · port ",
    "设置失败": "Setting failed",
    "远程访问已开启": "Remote access enabled",
    "远程访问已关闭": "Remote access disabled",
    "浏览器打开 ": "Open in browser: ",
    "只在本机可用了": "now only reachable locally",
    "并发": "Concurrency",
    "并发 ": "Concurrency ",
    "硬解": "Hardware decode",
    "软解": "Software decode",
    "计算中…": "Calculating…",
    "自动选择": "Automatic",
    "未选择": "Not selected",
    "全脸": "Full face",
    " 进程": " processes",
    "自动进程": "Auto processes",
    "就绪 · ": "ready · ",
    "已提交导出 · ": "Export submitted · ",
    "帧 ": "frame ",
    "人脸 帧 ": "face frame ",
    "剪切 帧 ": "cut frame ",
    "角度 ": "angle ",
    "数据集参数 · 补充": "Dataset parameters · extras",
    "多身份人脸集目录，留空 = 用 dst 人脸集兜底": "Multi-identity face set folder; empty means fall back to the dst set",
    "有没有 train-extra 包装层": "Whether there is a train-extra wrapper layer",
    "用什么设备训练": "Which device trains the model",
    "按你这张卡的可用显存，建议 ": "Based on this card's available VRAM, recommended ",
    " 分辨率 / batch ": " resolution / batch ",
    "（训练显存约 ": "(training VRAM about "
  });

  /* 运行期文案 batch 5：训练页（参数说明、状态、显存、预览） */
  Object.assign(DICT, {
    "目标批次": "Target batch",
    "想要的等效批次；= 批次大小 × 累积次数。0 = 关闭累积": "Desired effective batch = batch size × accumulation steps. 0 disables accumulation",
    "1.0 = 关闭": "1.0 = off",
    "0.0~0.3 · 随机调整输入的 HSV：让模型从输入就学习更丰富的色彩，解出的人脸颜色也更丰富": "0.0–0.3 · randomises the input HSV: the model learns richer colour and decodes more varied skin tones",
    "把 src 的颜色映射到 dst 上：通过改变输入数据，让模型学习调色后的 src 图像，生成更贴合 dst 色彩的人脸": "Maps src colour onto dst: the model learns from colour-shifted src images and produces faces closer to dst tones",
    "随机遮挡输入图像的一部分、重建目标不变：让模型在信息缺失（被挡住）的情况下也能解码出人脸": "Randomly occludes part of the input while the target stays the same: the model learns to decode faces from incomplete data",
    "给输入图像随机加噪声、重建目标不变：让模型在噪点严重时也能解码出清晰的人脸": "Adds random noise to the input while the target stays the same: the model stays clean under noise",
    "对输入图像随机偏色、重建目标不变：让模型无论输入什么颜色的人脸，都只解码出正常颜色的人脸": "Randomly shifts input colour while the target stays the same: the model always decodes normal skin tones",
    "应用参数": "Apply parameters",
    "热": "live",
    "训练中改这一项会立即生效": "Changes to this take effect immediately during training",
    "混乱": "messy",
    "编辑框但打不了字": "the input box cannot be typed into",
    "文件选择器还没接": "the file picker is not wired yet",
    "开关拨不动": "the toggle does not move",
    "正在配置 ": "Configuring ",
    "，确认后开始训练。": " — confirm to start training.",
    "训练前确认参数。": "Check the parameters before training.",
    "当前 iter ": "Current iter ",
    " · 已读入 ": " · loaded ",
    " 项真实参数": " real parameters",
    "请求失败": "Request failed",
    "操作失败": "Action failed",
    "已继续": "Resumed",
    "保存中…": "Saving…",
    "已请求保存": "Save requested",
    "停止训练？": "Stop training?",
    "会先让训练进程保存再退出，可能要等十几秒。": "The training process saves first, then exits — this can take a dozen seconds.",
    "停止训练": "Stop training",
    "停止中…": "Stopping…",
    "训练已停止": "Training stopped",
    "连接中": "Connecting",
    "已停止": "Stopped",
    "正在停止": "Stopping",
    "加载中": "Loading",
    "2M 参数": "2M params",
    " / 显卡 ": " / GPU ",
    "启动中": "Starting",
    "显存不足": "Out of VRAM",
    "显存不足，训练已停止": "Out of VRAM — training stopped",
    "测速中…": "Measuring speed…",
    "未在训练": "Not training",
    "最近 ": "Last ",
    " iter（实时）": " iter (live)",
    "已跑 ": "Elapsed ",
    "约 ": "about ",
    " 剩余": " left",
    "显存 ": "VRAM ",
    "模型 ": "Model ",
    "刚刚": "just now",
    " iter（约）": " iter (approx.)",
    " 份": " copies",
    "4.0 进程正在起来，模型加载要一会儿": "The 4.0 process is starting; loading the model takes a moment",
    "正在停止…": "Stopping…",
    "等它保存完再退出": "waiting for it to save before exiting",
    "正在保存…": "Saving…",
    "下一个迭代后落盘": "written after the next iteration",
    "加载模型…": "Loading model…",
    "已等 ": "waited ",
    "s 前": "s ago",
    "m 前": "m ago",
    "等新图…": "waiting for a new frame…",
    "预览还没准备好": "Preview is not ready yet",
    "4.0 通常几十秒后才有第一张": "the first frame usually arrives within tens of seconds",
    "没拿到新图": "No new frame",
    "训练那边还没出下一张，稍后再点一次": "training has not produced the next frame yet — try again shortly",
    "通知": "Notifications",
    "当前 ": "Current ",
    " 条": " items",
    "隐藏模式密码已更新": "Hidden-mode password updated",
    "密码已清空": "Password cleared",
    "还没选中模型": "No model selected",
    "没有改动": "No changes",
    "参数还是原来的值": "parameters are unchanged"
  });

  /* 运行期文案 batch 4：拼接片段的翻译（前后空格必须保留，否则英文会粘成一串） */
  Object.assign(DICT, {
    "共 ": "total ",
    " 线程 · ": " threads · ",
    " 张/秒": " images/s",
    " · 剩 ": " · left ",
    " 秒": " s",
    " 张，命中 ": " images, matched ",
    " 张": " images",
    "已重排 ": "Reordered ",
    "已按「": "Re-sorted by “",
    "」重排 ": "” — ",
    "身份": "Identity",
    "重复": "Duplicates",
    "已移走 ": "Moved ",
    " 张失败": " images failed",
    "并入 ": "merged ",
    " 张 · 重名改名 ": " images · renamed ",
    " 张 · 删除空目录 ": " images · empty dirs removed ",
    " 个": " dirs",
    "合并 ": "Merged ",
    " 张，清理 ": " images, cleaned ",
    " 个空目录": " empty dirs",
    "已读取 ": "Read ",
    " 张 · 命中 ": " images · matched ",
    " 张没有元数据的脸，用 2D106 重新检测了关键点": " faces without metadata had 2D106 landmarks re-detected",
    " 张缺有效 landmarks": " images lack valid landmarks",
    " · 例：": " · e.g. ",
    "（都在 _by_id/ 下）": "(all under _by_id/)",
    " 张已分到 _by_id/ 下的 ": " images sorted into ",
    " 个目录": " folders under _by_id/",
    "分成 ": "Split into ",
    " 组，移走 ": " groups, moved ",
    " 主体 ": "Main ",
    " 张 · 其他身份 ": " images · other identities ",
    " 张已移出 · ": " images moved out · ",
    " 簇": " clusters",
    " · 噪声 ": " · noise ",
    " 张，没有命中 · ": " images, no match · ",
    "扫描完成 · 命中 ": "Scan finished · matched ",
    " 张，正在移走…": " images, moving…",
    " 小时 ": " h ",
    " 张人脸": " faces",
    " 个文件": " files",
    "正在提取人脸特征并聚类…（这一步要跑引擎进程，通常 5~10 秒）": "Extracting and clustering face features… (runs the engine process, usually 5–10 s)",
    /* 素材 / 抽帧 进度 */
    "http://127.0.0.1:8001 · 已在浏览器打开": "http://127.0.0.1:8001 · opened in your browser",
    " 帧 · ": " frames · ",
    " 帧": " frames",
    " 张 · ": " images · ",
    " 张带 landmark": " images with landmarks",
    " 张对齐脸": " aligned faces",
    "每隔 ": "Every ",
    " 帧检测一次，稍等…": " frame(s) is analysed, hold on…",
    " 个采样点 · ": " sample points · ",
    " 处有脸 · 用时 ": " with a face · took ",
    /* B 站下载 */
    "获取中…": "Fetching…",
    "视频 ": "Video ",
    "第 ": "Batch ",
    " 批 · 已取 ": " · fetched ",
    "获取完成": "Fetch finished",
    " · 共 ": " · total ",
    " 个视频 · ": " videos · ",
    " 批": " batches",
    "下载完成": "Download finished",
    "全部 ": "All ",
    " 个视频已下载": " videos downloaded",
    "继续": "Continue",
    "终止下载": "Stop downloading",
    "会清空当前队列，已下载的文件保留。": "This clears the queue; already downloaded files are kept.",
    "UID 不合法": "Invalid UID",
    "请输入纯数字的 UP 主 UID": "Enter the uploader UID as digits only",
    "开始下载": "Start download",
    "已选 ": "Selected ",
    /* 任务列表 / 模型卡 */
    " 个在排队": " queued",
    " 个进行中 / 共 ": " running / ",
    " 个已完成": " done",
    "个模型": "models",
    "个已导出 DFM": "exported DFMs",
    ".dfm 已生成": " .dfm created",
    "会移除 ": "Removes ",
    " 的全部权重与配置，无法恢复。": " — all weights and config, cannot be undone."
  });

  /* 运行期文案 batch 3/3：B 站登录、首页卡片、任务列表、时间单位 */
  Object.assign(DICT, {
    "还没有设置密码，去「设置 → 隐私密码」里设一个": "No password set yet — go to Settings → Privacy password",
    "密码不对": "Wrong password",
    "用户": "User",
    "正在向 B 站申请二维码…": "Requesting a QR code from Bilibili…",
    "生成失败": "Generation failed",
    "点这里重试": "Click here to retry",
    "无法获取二维码：": "Cannot get a QR code: ",
    "登录成功，欢迎 ": "Signed in, welcome ",
    "登录成功": "Signed in",
    "欢迎，": "Welcome, ",
    "已过期 · 点这里刷新": "Expired · click here to refresh",
    "二维码已过期": "The QR code expired",
    "等待扫码…": "Waiting for scan…",
    "退出登录？": "Sign out?",
    "会清除本机的 B 站授权，下次使用需要重新扫码。训练和数据都不受影响。": "This clears the local Bilibili authorisation; you will need to scan again next time. Training and data are unaffected.",
    "已退出登录": "Signed out",
    "B 站授权已清除": "Bilibili authorisation cleared",
    " 分钟前": " min ago",
    " 小时前": " h ago",
    " 天前": " d ago",
    "先导入一份人脸集": "Import a face set first",
    "src 与 dst 都还是空的。提取器可以从视频或图片里裁出对齐后的人脸。": "src and dst are both empty. The extractor cuts aligned faces out of videos or images.",
    "打开提取器": "Open the extractor",
    "创建第一个模型": "Create your first model",
    "数据集已就绪（src ": "Dataset ready (src ",
    "），接下来建模型开始训练。": ") — create a model and start training.",
    "继续训练 ": "Resume training ",
    "停在 iter ": "stopped at iter ",
    "，上次训练于 ": ", last trained ",
    "未知": "Unknown",
    "还没有模型": "No models yet",
    "个模型": "models",
    "个已导出 DFM": "exported DFMs",
    "个进行中 / 共 ": " running / ",
    "个已完成": " done",
    "任务": "Jobs",
    "还有 ": "Another ",
    "个在排队": "queued",
    "已完成 ": "Done ",
    "下载选中（": "Download selected (",
    "是": "Yes",
    "否": "No",
    "转换": "Convert",
    "导出": "Export",
    "删除": "Delete",
    "参数": "Parameters",
    "模型": "Models",
    "切换语言": "Switch language",
    "无预览": "No preview",
    "暂无 loss 记录": "No loss recorded yet",
    "正在加载显卡信息…": "Loading GPU info…",
    "正在读取工作区…": "Reading workspace…",
    "个文件": "files",
    "张人脸": "faces",
    "剩 ": "left ",
    "分": "min",
    "正在提取…": "Extracting…",
    "正在中断…": "Stopping…"
  });

  /* 运行期文案 batch 2/3：抽帧 / 合成 / 导出 / 合成器 */
  Object.assign(DICT, {
    "抽帧": "Frame extraction",
    "人脸对齐": "Face alignment",
    "处理完成": "Finished",
    "返回": "Back",
    "已离开运行页": "Left the run page",
    "扫描/移动会在后台继续完成": "scanning and moving continue in the background",
    "正在中断…": "Stopping…",
    "正在提取…": "Extracting…",
    "缺少输入": "Missing input",
    "先在「输入」里填一个视频文件，或把文件拖进框里": "Put a video file in “Input”, or drop it into the box",
    "开始抽帧": "Start frame extraction",
    "启动后在浏览器里打开工作台，模型在网页里再选": "Open the workbench in your browser after start; pick the model there",
    "端口 8001": "Port 8001",
    "已经在训练": "Already training",
    "导出中…": "Exporting…",
    " · 要加载模型，几分钟": " · loads the model, takes a few minutes",
    "导出完成": "Export finished",
    "导出失败": "Export failed",
    "删除模型": "Delete model",
    "删除失败": "Delete failed",
    "已删除": "Deleted",
    "好": "OK",
    "取不到成片": "No rendered video found",
    "成片预览": "Rendered preview",
    "还没有成片 —— 先导出一次": "Nothing rendered yet — export once first",
    "打开视频": "Open video",
    "打开失败": "Open failed",
    "未知错误": "Unknown error",
    "已打开项目": "Project opened",
    "已打开": "Opened",
    "打开项目": "Open project",
    "网页版拿不到本地路径": "The web build cannot access local paths",
    "选择 DFL 项目目录（data_dst）": "Choose a DFL project folder (data_dst)",
    "读不了这个目录": "Cannot read this folder",
    "素材已就绪": "Sources ready",
    "正在准备": "Preparing",
    "等这一段处理完再来": "Wait until this pass finishes",
    "抽帧…": "Extracting frames…",
    "准备素材": "Prepare sources",
    " · 抽帧 + 对齐": " · frames + alignment",
    "准备失败：": "Preparation failed: ",
    "准备失败": "Preparation failed",
    "先打开视频": "Open a video first",
    "分析需要先选一个视频": "Analysis needs a video selected first",
    "开始分析": "Start analysis",
    "分析失败": "Analysis failed",
    "分析完成": "Analysis finished",
    "检测失败": "Detection failed",
    "还没有素材": "No sources yet",
    "先导入一段视频（会自动抽帧 + 对齐）": "Import a video first (frames and alignment are automatic)",
    "剩余 ": "Remaining ",
    "正在取消": "Cancelling",
    "取消失败": "Cancel failed",
    "等它退出，几秒": "waiting for it to exit, a few seconds",
    "没有素材": "No sources",
    "先在工作台导入一段视频 —— 会自动抽帧 + 对齐": "Import a video in the workbench first — frames and alignment are automatic",
    "正在合成": "Merging",
    "上一次还没结束": "The previous run has not finished",
    "已开始合成": "Merge started",
    "合成未启动": "Merge did not start",
    "导入视频": "Import video",
    "网页版拿不到本地路径，请把视频放进工作区": "The web build cannot access local paths — put the video in the workspace",
    "选择视频所在目录": "Choose the folder containing the video",
    "这个目录里没有视频": "No video in this folder",
    "没有找到视频": "No video found",
    "导出中": "Exporting",
    " → .dfm（要加载模型，可能几分钟）": " → .dfm (loads the model, may take a few minutes)",
    " 已生成 .dfm": " .dfm created",
    "看控制台日志": "Check the console log",
    "合成中…": "Merging…",
    "合成完成": "Merge finished",
    "合成失败": "Merge failed",
    "输出 ": "Output ",
    "不存在": "does not exist",
    "数据集已更换": "Dataset switched",
    "路径已填入": "Path filled in",
    "网页版拿不到本地路径，请直接填写或拖入文件夹": "The web build cannot access local paths — type it or drop a folder",
    "选择数据集目录": "Choose a dataset folder",
    "无法读取路径": "Cannot read the path",
    "请把文件夹从资源管理器拖进来": "Drag the folder in from Explorer",
    "已重置": "Reset",
    "回到内置坐标": "Back to built-in coordinates",
    "服务端拒绝": "Rejected by the server",
    "已保存": "Saved",
    "隐藏模式的脸已更新": "Hidden-mode faces updated",
    "保存失败": "Save failed",
    "已复制": "Copied",
    "68 点坐标在剪贴板里": "68-point coordinates are on the clipboard"
  });

  /* 运行期文案 batch 1/3：toast、确认框、空状态（都是 JS 拼出来、只有操作时才出现的） */
  Object.assign(DICT, {
    "点了没反应": "Nothing happened",
    "这一节的后端还没接": "This section is not wired to the backend yet",
    "应用遮罩中…": "Applying mask…",
    "正在应用遮罩": "Applying mask",
    "启动失败": "Failed to start",
    "应用遮罩已开始": "Mask application started",
    "应用遮罩失败": "Mask application failed",
    "运行中…": "Running…",
    "等待开始": "Waiting to start",
    "应用中…": "Applying…",
    "应用遮罩完成": "Mask application finished",
    "已完成": "Done",
    "开始扫描": "Scan started",
    " · 正在统计…": " · counting…",
    "正在扫描清晰度": "Scanning sharpness",
    "正在查找重复图": "Looking for duplicates",
    "正在检测错脸": "Detecting wrong faces",
    "正在聚类人脸身份": "Clustering identities",
    "扫描完成": "Scan finished",
    "扫描失败": "Scan failed",
    "开始排序": "Sort started",
    " · 正在处理…": " · working…",
    "正在排序": "Sorting",
    "排序完成": "Sort finished",
    "排序失败": "Sort failed",
    "正在统计清晰度": "Measuring sharpness",
    "正在计算哈希": "Computing hashes",
    "特征提取中": "Extracting features",
    "正在聚类": "Clustering",
    "按特征分组…": "Grouping by feature…",
    "计算清晰度…": "Computing sharpness…",
    "计算感知哈希…": "Computing perceptual hashes…",
    "提取 ArcFace 特征…": "Extracting ArcFace features…",
    "失败：": "Failed: ",
    "处理失败": "Processing failed",
    "移动到 _trash…": "Moving to _trash…",
    "张已移入 _trash": "images moved to _trash",
    "部分未移走": "some could not be moved",
    "在 _trash 里": "in _trash",
    "移动失败": "Move failed",
    "正在合并子文件夹": "Merging subfolders",
    "把子目录里的图搬回根目录…": "Moving images from subfolders back to root…",
    "张已并回根目录": "images merged into root",
    "合并完成": "Merge finished",
    "合并失败": "Merge failed",
    "正在统计…": "Counting…",
    " · 扫描中": " · scanning",
    "正在读取 68 点并分类…": "Reading 68-point landmarks and classifying…",
    "正在统计数据集…": "Counting the dataset…",
    "已现场补点": "landmarks re-detected",
    "部分图片未参与错脸判定": "Some images were skipped in wrong-face detection",
    "已按身份分文件夹": "Sorted into per-identity folders",
    "没有可分的组": "No groups to split",
    "身份聚类完成": "Identity clustering finished",
    "没有需要处理的图片": "No images needed processing",
    "扫描完成 · 没有命中": "Scan finished · nothing matched",
    "选择数据集": "Choose a dataset",
    "文件选择器还没接 · 也可以直接把文件夹拖到这个框里": "File picker not wired yet · you can also drop a folder into this box",
    "模糊 · 身份 · 重复 · 错脸 · 合并": "Blur · Identity · Duplicates · Wrong face · Merge",
    "ArcFace 特征向量聚类，同一组≈同一个人（顺带清掉打不开的文件）": "ArcFace embedding clustering — same group, same person (unreadable files removed too)",
    "感知哈希(phash)检测高度相似，剔除重复帧（顺带清掉打不开的文件）": "pHash finds near-identical frames and drops duplicates (unreadable files removed too)",
    "68 点几何分类判错检（需要数据集用 2D106 真点重提过）": "Flags bad detections by 68-point geometry (needs 2D106 landmarks in the dataset)",
    "原序": "Original order",
    "感知哈希": "Perceptual hash",
    "越清晰排越前（blur）": "Sharpest first (blur)",
    "按偏航角从小到大（pose/yaw）": "By yaw, smallest first",
    "恢复原始文件顺序（reset）": "Restore original file order",
    "按感知哈希相近度排序（phash）": "Sort by perceptual-hash similarity",
    "按直方图相似度排序（histogram）": "Sort by histogram similarity",
    "邻域半径": "Neighbour radius",
    "瞬间": "Instant",
    "卡片已经白了、字还是旧的浅灰": "the card turns white while the text is still in the old light grey",
    "点太快把画面卡成一整块遮罩": "clicking too fast turns the preview into one solid mask",
    "这次没动画": "no animation this time"
  });

  /* 全盘补译（扫 index.html 文本节点 + app.js 渲染字面量得出的缺口） */
  Object.assign(DICT, {
    "退出登录": "Sign out",
    "你好": "Hello",
    "正在读取工作区…": "Reading workspace…",
    "TinyMog 速度优先 · Retina 质量优先": "TinyMog favours speed · Retina favours quality",
    "源视频旋转角度": "Source rotation",
    "每 N 帧取一张": "One frame every N",
    "正在提取人脸": "Extracting faces",
    "中断提取": "Stop extraction",
    "已处理": "Processed",
    "速度": "Speed",
    "已用": "Elapsed",
    "秒": "s",
    "剩余": "Remaining",
    "中断": "Stop",
    "模型转换": "Model conversion",
    "失败": "Failed",
    "加载模块": "Load module",
    "转换模型": "Convert model",
    "尚未加载数据": "No data loaded",
    "训练失败": "Training failed",
    "剩余 —": "Remaining —",
    "未登录": "Not signed in",
    "登录后可下载 1080P / 会员清晰度": "Sign in to download 1080P / member quality",
    "去登录": "Sign in",
    "未启用": "Not enabled",
    /* 数据处理页的模式行与参数行（都是 JS 渲染的，之前完全没翻） */
    "模式": "Mode",
    "邻居半径": "Neighbour radius",
    "最小样本": "Min samples",
    "相似度阈值": "Similarity threshold",
    "判定阈值": "Decision threshold",
    "最差 %（0=按阈值）": "Worst % (0 = by threshold)",
    "遮罩类型": "Mask type",
    "遮罩尺寸": "Mask size",
    "开始处理": "Start",
    "还没有导出过 DFM": "No DFM exported yet",
    "停止处理": "Stop",
    "正在处理…": "Working…"
  });

  /* 数据处理页：模式名与说明（原来只有中文） */
  Object.assign(DICT, {
    "人脸身份": "Identity",
    "应用遮罩": "Apply mask",
    "按特征向量聚类，同一组副本相同一张脸（顺带清掉打不开的文件）": "Cluster faces by ArcFace embedding — same group, same person (unreadable files removed too)",
    "高频/低频能量比 E(b0)/E(b2) 判清晰度，按分位淘汰最差一批（顺带清掉打不开的文件）": "Blur score from the high/low frequency energy ratio E(b0)/E(b2); worst percentile is dropped (unreadable files too)",
    "感知哈希(pHash)检测高度相似，剔除重复帧（顺带清掉打不开的文件）": "pHash detects near-identical frames and drops the duplicates (unreadable files too)",
    "68 点几何分类判错检（需要数据集里 2D106 真点重提过）": "Flags bad detections using 68-point geometry (requires 2D106 landmarks in the dataset)",
    "把所有子目录（含多层）里的图片搬回根目录，清掉空出来的文件夹；跳过 _trash 和隐藏目录": "Flatten every subfolder (any depth) into the root and remove the emptied folders; skips _trash and hidden dirs"
  });

  /* 登录页扫码卡的文案 */
  Object.assign(DICT, {
    "使用": "Scan with",
    "APP 扫码授权登录": "the Bilibili app",
    "用手机 B 站 App 扫码，扫完在手机上确认。": "Scan the code with the Bilibili app on your phone, then confirm there.",
    "等待扫码": "Waiting for scan",
    "已扫码，请在手机上确认": "Scanned — confirm on your phone"
  });

  /* ---------- 动态渲染的补翻 ----------
     问题：i18n 只在切换语言那一刻扫一遍 DOM。之后由 JS 渲染出来的内容
     （数据处理页的模式行、下拉选项、列表项……）都是新建节点，永远保持中文。
     解决：语言是 en 时挂一个 MutationObserver，凡是有新节点进 DOM，
     就在那棵子树里补跑一遍同样的翻译。只监听 childList，
     改文本本身不会触发，不会自激。 */
  function translateNode(node) {
    if (!node) return;
    if (node.nodeType === 3) {
      var raw = node.nodeValue;
      if (raw == null || !hasCJK(raw)) return;
      var hostEl = node.parentElement;
      if (hostEl && hostEl.closest && hostEl.closest("#wLang,[data-i18n-skip]")) return;
      if (orig.get(node) === undefined) orig.set(node, raw);
      var hit = lookup(raw);
      if (hit) {
        var lead = raw.match(/^\s*/)[0];
        var trail = raw.match(/\s*$/)[0];
        node.nodeValue = lead + hit + trail;
      }
      return;
    }
    if (node.nodeType !== 1) return;
    if (node.closest && node.closest("#wLang,[data-i18n-skip]")) return;
    translateText(lang, node);
    translateAttrs(lang);
  }

  var mo = null;
  function watchDynamic() {
    if (mo || typeof MutationObserver !== "function" || !document.body) return;
    mo = new MutationObserver(function (muts) {
      if (lang !== "en") return;
      for (var i = 0; i < muts.length; i++) {
        var added = muts[i].addedNodes;
        for (var j = 0; j < added.length; j++) translateNode(added[j]);
      }
    });
    mo.observe(document.body, { childList: true, subtree: true });
  }

  window.I18N = {
    apply: apply,
    refresh: function () { apply(lang); },
    get: function () { return lang; },
    dict: DICT
  };

  function boot() {
    var saved = "zh";
    try { saved = window.localStorage.getItem("dfl-lang") || "zh"; } catch (e) {}
    apply(saved);
  }
  /* 脚本在 body 末尾，DOMContentLoaded 可能已经过去了，两种时机都要兜住 */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
