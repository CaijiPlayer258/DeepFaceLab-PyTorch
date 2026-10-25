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
  "请通过 app/main.py 启动": "Launch through app/main.py",
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

  function translateText(lang) {
    var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    var n;
    while ((n = walker.nextNode())) {
      var raw = n.nodeValue;
      if (raw == null) continue;
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
