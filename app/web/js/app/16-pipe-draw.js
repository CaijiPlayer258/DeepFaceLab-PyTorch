    /* ============================================================
       六阶段状态机 + 流水线实况（SVG）
       ------------------------------------------------------------
       这一页有两条后端的路，但只有一张图：
         · fused（快速导出，默认）：六段**并行**，只有帧进度；
         · 分段链（frames→faces→pred→mask→merge→encode）：串行，一段一段来。
       图本身（6 列 lane + 贝塞尔曲线 + 光点）的数据**全部**来自 status：
         status.stages[key] = {ticks, inflight, lanes[], ms?, batch?}
       前端一条都不写死 —— lane 有几条就是图上几个方块（合成池 8 线程 = t0..t7，
       编码 6 线程 = e0..e5），ticks 只增、光点按**增量**放，inflight 就是列间那句
       列标题的「N 帧」。没有这份数据时（老后端 / 还没开单）列是占位的「—」，不瞎猜。
       点亮：某条 lane 的列 tick 了 → 那条 lane 瞬间到峰值、1.15s 暗回常态。
       卡住：某列 >600ms 没 tick 且还没完成 → 整列琥珀；恢复自动复原。
       ============================================================ */
    var MGP_STAGES = ["frames", "faces", "pred", "mask", "merge", "encode"];
    /* 分段链的阶段名 → status.stages 的 key（两套名字，只在这张表里对一次） */
    var MG_KEY = { frames: "frames", faces: "det", pred: "dfm",
                   mask: "xseg", merge: "compose", encode: "encode" };

    var MG_PIPE = (function () {
      var NS = "http://www.w3.org/2000/svg";
      var KEYS = ["frames", "det", "dfm", "xseg", "compose", "encode"];
      /* 最后一列不叫「导出」了：这一段真的只是把合成好的帧**编码/缓存**出来，
         真正写出成片是后面 ffmpeg 那一趟（也计时到这一段）。用户看到的「导出」两个字
         出现在这一列会以为进度已经到写视频了，所以叫「缓存帧」。后端/日志里那个
         内部名 encode 不动。 */
      var NAME = { frames: "切帧", det: "切脸", dfm: "换脸",
                   xseg: "遮罩", compose: "合成", encode: "缓存帧" };
      /* viewBox 尺寸定死（和 UI 设计稿同一比例）：进度卡铺满内容区（≈1200~1400px），
         所以 1200×340 渲染出来接近 1:1～1.15:1，9.5px 的 lane 名字不会缩成 6px。
         高度不变 → 换 lane 数也不会让页面跳动。 */
      var W = 1200, H = 340, PADL = 56, TOPY = 58, BOXW = 74, GAP = 7;
      /* 右侧留 84px 给「成片」那一格（水位杯 + 标题）。它**不是第七列**：没有 lane、
         没有 t-stage，只画一杯水 —— 六列仍然是六列（验收按 text.t-stage 数）。 */
      var RGT = 0, OUTW = 56;   /* 成片杯现在吃满一格宽度 */
      /* 光点密度由**两个物理上限**决定，不再按「每次轮询硬切 10 个」：
           · PULSE_PER_S：每秒最多放多少个（按 POLL_MS 折算成每次轮询的预算）；
           · MAX_PULSE  ：同屏最多几个在飞（≈ 40 帧/秒 × 8 张脸 × 0.6s 飞行时间的极端值）。
         常见档（20~30 帧/秒、每帧 1~4 张脸）一条都不丢；只有极端帧率/人脸数才按比例稀释。
         以前 TICK_MAX=10 把「20 帧/秒」也一起砍了（500ms × 20fps = 每段正好 10 个），
         而且 10 个还在同一瞬间一次性放出来 —— 肉眼就是「每半秒跳一波」。 */
      var MAX_PULSE = 200;     /* 同屏在飞上限（丢最旧的） */
      var PULSE_PER_S = 240;   /* 每秒最多放几个光点 */
      var POLL_MS = 250;       /* 和轮询同一个窗口：一拍的光点铺在这 250ms 里 */
      var LIT_MS = 1250;       /* 1.15s 动画 + 余量，之后摘 .lit 回稳态 */
      var STALL_MS = 600;      /* 某列多久没 tick 算卡住 */

      var geo = {};            /* key -> {cx, off, lanes:[{y,rect,txt}], count, queue, col} */
      var edgesIn = {};        /* "si:ai" -> {d, el}：源 lane → 腰点 */
      var edgesOut = {};       /* "si:bi" -> {d, el}：腰点 → 目标 lane（和 edgesIn 共用腰点） */
      /* "in:out:q" / "out:out" -> d：编码列 → 成片的 n+1 条。**以前这个变量没声明**，
         只要 geo.out 先于这段 IIFE 被建出来（下面那版就是把杯子提前了），
         第一句 edges[...] 就是 ReferenceError → build() 直接抛出去，
         svg 一个子节点都没挂上、drawBypass() 永远走不到。 */
      var edges = {};
      var waists = {};         /* si -> {x, y}：这一对相邻列的腰点（进出共用） */
      var pulses = [];         /* 活跃光点 {c, path, L, t0, dur} */
      var pool = [];           /* 复用起来的 <circle>：不每个 tick 重建 DOM */
      var flashes = [];        /* 待点亮 [{k, i}]（合到下一帧一起做，减少强制重排） */
      var raf = 0;
      var sig = "";            /* lanes 的签名：变了才重建结构 */
      var prev = {};           /* key -> 上次的 ticks */
      var prevNf;              /* 上次的 det.no_face_ticks（旁路光点按它的增量放） */
      var lastAt = {};         /* key -> ticks 上次变化的时刻 */
      var stallNow = {};       /* key -> 现在卡没卡 */
      var state = {};          /* key -> "" / on / done / err（列标题颜色） */
      var seq = 0;             /* 光点序号：多 lane 时按它轮转 */
      var lastFlashed = [];    /* 最近一拍真正点亮过的 lane（验收拍照时按它重挂） */
      var lastSpawn = 0;       /* 最近一拍实际放出去几个光点（验收要看「按预算稀释」有没有生效） */
      var lastBypass = 0;      /* 其中走旁路的几个（无脸帧） */
      var layers = null;
      /* 旁路几何的「版本号」：build() 每重建一次结构就 +1（lane 数/列宽/杯子位置全变了）。
         drawBypass() 画完把当时的版本记下来，setBypass() 一比就知道要不要重画 ——
         以前它无条件只改读数，旁路因此会冻在**旧坐标**上。 */
      var bypassGen = 0, bypassDrawnGen = -1;
      /* 截图专用：把光点的时钟按住（>0 = 定格在这个时刻），只给验收拍照用。
         光点只有 470–750ms 的命，一张 headless 截图往返就可能比它长 —— 拍照时
         用 mgpPipeFreeze() 把它们钉在曲线上，运行时代码路径完全不看这个值。 */
      var freezeAt = 0;

      function nowMs() {
        return (window.performance && window.performance.now)
          ? window.performance.now() : Date.now();
      }
      function reduceMotion() {
        try {
          return !!(window.matchMedia &&
                    window.matchMedia("(prefers-reduced-motion: reduce)").matches);
        } catch (e) { return false; }
      }
      function mk(tag, attrs) {
        var e = document.createElementNS(NS, tag), k;
        for (k in (attrs || {})) {
          if (Object.prototype.hasOwnProperty.call(attrs, k)) e.setAttribute(k, attrs[k]);
        }
        return e;
      }
      /* 相邻两列之间只有一个**腰点**：x = 两列中线，y = 两列所有 lane 的 y 的中点。
         源侧：每条源 lane → 腰点（一条曲线）；目标侧：腰点 → 每条目标 lane（一条曲线）。
         于是「每个源 lane × 每个目标 lane」的交叉（合成 8 × 导出 6 = 48 条）变成
         8 + 6 = 14 条，进出共用同一个点 —— 视觉上是「先收束到一点、再散开」。
         单段仍是三次贝塞尔；两个控制点分别与两端同 y，所以**两端切线都水平**，
         两段在腰点处方向也连续，拼起来是一整条平滑的流（不是折线、也不打结）。 */
      function arc(a, b) {
        var dx = b.x - a.x;
        return "M" + a.x + "," + a.y +
          " C" + (a.x + dx * 0.55) + "," + a.y +
          " " + (b.x - dx * 0.55) + "," + b.y +
          " " + b.x + "," + b.y;
      }
      function lanesOf(st) {
        var out = {}, i, k, row;
        for (i = 0; i < KEYS.length; i++) {
          k = KEYS[i];
          row = (st && st[k]) || {};
          out[k] = (row.lanes && row.lanes.length) ? row.lanes.slice(0) : [];
        }
        return out;
      }
      function signature(lanes) {
        var parts = [], i;
        for (i = 0; i < KEYS.length; i++) parts.push((lanes[KEYS[i]] || []).join(","));
        return parts.join("|");
      }
      /* status.stages 的「实况」形状：必须有 lanes 或 ticks。
         分段链的 /stage/status 也带一个 stages（那是每段的元数据），别混。 */
      function isLive(st) {
        if (!st || !st.frames) return false;
        return !!(st.frames.lanes || typeof st.frames.ticks === "number");
      }

      function build(lanes) {
        var svg = $("mgPipeSvg");
        if (!svg) return false;
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        layers = { defs: mk("defs", {}), edge: mk("g", { "class": "edges" }),
                   bypass: mk("g", { "class": "bypass" }), box: mk("g", {}),
                   lbl: mk("g", {}), pulse: mk("g", { "class": "pulses" }) };
        geo = {}; edgesIn = {}; edgesOut = {}; edges = {}; waists = {};
        pulses = []; pool = [];
        bypassD = "";
        bLabel = null;
        var colW = (W - PADL * 2) / KEYS.length;   /* 七等分：6 个阶段 + 成片各占一格（成片不再被挤在最右边） */
        var space = H - TOPY - 46;
        var i, j, k;
        for (i = 0; i < KEYS.length; i++) {
          k = KEYS[i];
          var cx = PADL + i * colW;
          var ln = (lanes[k] || []).slice(0);
          var off = !ln.length;                 /* 这一单不用这一段：占位，不假装有线程 */
          if (off) ln = ["—"];
          var n = ln.length;
          var laneH = Math.min(24, Math.max(14, space / n - GAP));
          var y0 = TOPY + (space - (n * laneH + (n - 1) * GAP)) / 2;
          var col = mk("g", { "class": "col" });
          col.setAttribute("data-k", k);
          /* ★ 重建时把这一列**当前**的状态贴回去：state[k] 还记着 on/done，可这些 g.col
             是新建的 —— 不回填就成了「图上全灰、其实这一单还在跑」那种假死
             （分段导出往上传 stages 时 lane 签名一变就重建，列标题的颜色就丢过一次；
             验收 f 那段就是拿「六列有没有推进」当断言，一眼就看到了）。 */
          if (state[k]) col.classList.add(state[k]);
          var t1 = mk("text", { x: cx, y: 24, "text-anchor": "middle", "class": "t-stage" });
          t1.textContent = NAME[k] || k;
          col.appendChild(t1);
          var t2 = mk("text", { x: cx, y: 40, "text-anchor": "middle", "class": "t-count" });
          t2.textContent = "— 帧";
          col.appendChild(t2);
          layers.lbl.appendChild(col);
          geo[k] = { cx: cx, off: off, lanes: [], count: t2, col: col };
          /* 水位只填「缓存帧」这一列 —— 别的列是线程/队列，没有「攒了多少帧」这回事 */
          var cup = (k === "encode" && !off);
          for (j = 0; j < n; j++) {
            var y = y0 + j * (laneH + GAP);
            var rect = mk("rect", { x: cx - BOXW / 2, y: y, width: BOXW, height: laneH,
                                    rx: 7, "class": off ? "lane off" : "lane" });
            rect.setAttribute("data-lane", k + "#" + j);   /* key#序号：排障/拍照直接定位 */
            layers.box.appendChild(rect);
            var water = null;
            if (cup) {
              /* 水面：一块和方块同样大小的 rect，用 clip-path 拿到圆角，再 scaleY(level)。
                 夹在方块和它的名字之间（层级正好：水盖住底色、不盖住 t-lane 字）。 */
              var cid = "mgpW" + i + "_" + j;
              var cp = mk("clipPath", { id: cid });
              cp.appendChild(mk("rect", { x: cx - BOXW / 2, y: y, width: BOXW,
                                          height: laneH, rx: 7 }));
              layers.defs.appendChild(cp);
              /* rx=0：圆角由 clipPath 给。给水面自己 rx 的话，scaleY 会把圆角压扁，
                 顶边变成圆弧 —— 看着像「一小块色块」而不是「一条水平水面线」。 */
              water = mk("rect", { x: cx - BOXW / 2, y: y, width: BOXW, height: laneH,
                                   "class": "water", "data-level": "0" });
              /* clip-path 必须挂在**不被缩放**的外层 g 上：直接挂在水面上时，
                 clip 用的是元素自身的用户坐标系，会跟着 scaleY 一起被压扁 ——
                 结果是「一颗越来越小的药丸」，而不是「一条水平的水面线」。 */
              var wg = mk("g", { "clip-path": "url(#" + cid + ")" });
              wg.appendChild(water);
              layers.box.appendChild(wg);
            }
            var tx = mk("text", { x: cx, y: y + laneH / 2 + 3.2, "text-anchor": "middle",
                                  "class": "t-lane" });
            tx.textContent = ln[j];
            layers.box.appendChild(tx);
            geo[k].lanes.push({ y: y + laneH / 2, y0: y, h: laneH,
                                rect: rect, txt: tx, water: water });
          }
        }
        /* 六列之后的「成片」：一只窄水杯（缓存里的水倒进这里）。它不进 KEYS、没有 lane，
           所以 text.t-stage 仍然只有 6 个、rect.lane 仍然只数真线程。 */
        var ocx = PADL + KEYS.length * colW;
        var ocid = "mgpWout";
        var ocp = mk("clipPath", { id: ocid });
        ocp.appendChild(mk("rect", { x: ocx - OUTW / 2, y: TOPY, width: OUTW,
                                     height: space, rx: 7 }));
        layers.defs.appendChild(ocp);
        /* 壳子用 .outcup 而不是 .lane：rect.lane 只数真线程（验收按它数 lane），
           给它挂 lane 会让「lane 方块数 == status 里的线程数」这条断言多出一个 */
        layers.box.appendChild(mk("rect", { x: ocx - OUTW / 2, y: TOPY, width: OUTW,
                                            height: space, rx: 7, "class": "outcup" }));
        /* 杯子里的**两层**水（以前只有一层：蓝面直接画 encode，于是「编码一开始蓝面就突然满」）：
             蓝面 .water.out = 已合成 = cache + out = compose / 总帧   ← 帧已经从合成那列出来了
             绿面 .encflow  = 已编码 = out           = encode / 总帧   ← 帧已经变成视频了
           绿面画在蓝面**之上**、半透明（让蓝透出来）：两层之间的那一段蓝就是「合成好了、
           还没编码」的帧。编码阶段绿面从下往上追蓝面 —— 看起来就是「水被倒进视频里」。
           绿面用 .encflow 而不是再加一块 .water：验收按 rect.water 的条数断言「水位只填
           两处（缓存列 n 条 + 成片 1 条）」，多一块 .water 会把它数成 8 条。 */
        var owater = mk("rect", { x: ocx - OUTW / 2, y: TOPY, width: OUTW, height: space,
                                  "class": "water out", "data-level": "0" });
        var oenc = mk("rect", { x: ocx - OUTW / 2, y: TOPY, width: OUTW, height: space,
                                "class": "encflow", "data-level": "0" });
        var owg = mk("g", { "clip-path": "url(#" + ocid + ")" });
        owg.appendChild(owater);
        owg.appendChild(oenc);
        layers.box.appendChild(owg);

        var otxt = mk("text", { x: ocx, y: 24, "text-anchor": "middle", "class": "t-out" });
        otxt.textContent = "成片";
        layers.lbl.appendChild(otxt);
        /* top/span = 杯子的杯口与杯高：setWater 用它们算水面线的 translateY */
        geo.out = { cx: ocx, water: owater, enc: oenc,  };
  /* 编码列 → 成片：与其它列同构（每条 lane 收束到腰点，再一条进杯）。
     成片不是一个 lane 列，所以单独画：n 条入腰 + 1 条腰→杯，都走 arc()。 */
  (function () {
    var E = geo.encode, O = geo.out;
    if (!E || !O || E.off || !E.lanes.length) return;
    var ys = [], q;
    for (q = 0; q < E.lanes.length; q++) ys.push(E.lanes[q].y);
    var wy = (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2;
    var x1 = E.cx + BOXW / 2;
    var x2 = O.cx - OUTW / 2;
    var wx = (x1 + x2) / 2;
    E.waist = { x: wx, y: wy };
    for (q = 0; q < E.lanes.length; q++) {
      var din2 = arc({ x: x1, y: E.lanes[q].y }, { x: wx, y: wy }, (x1 + wx) / 2);
      var pin2 = mk("path", { d: din2, fill: "none", stroke: "var(--line-strong)",
        "stroke-width": 1, opacity: ".75" });
      layers.edge.appendChild(pin2);
      edges["in:out:" + q] = din2;
    }
    var dcup = arc({ x: wx, y: wy }, { x: x2, y: wy }, (wx + x2) / 2);
    var pcup = mk("path", { d: dcup, fill: "none", stroke: "var(--line-strong)",
      "stroke-width": 1, opacity: ".75" });
    layers.edge.appendChild(pcup);
    edges["out:out"] = dcup;
    /* 旁路要「并进主路」：把**腰点**和**这段进杯曲线**都挂到 geo.out 上，
       让 bypassGeom() 复用同一份数字（同一个腰点、同一条三次贝塞尔）。
       各算一条「差不多」的曲线在杯口碰一下，差 0.1px 就会被看成两条线，
       读者不会觉得那是「汇入主流」——所以尾段必须是这一个 d 的子串。 */
    O.waist = { x: wx, y: wy };
    O.tailD = dcup;
    O.tailC = String(dcup).replace(/^\s*M[^C]*/, "");   /* 去掉开头的 M，只剩那段 C */
  })();

        /* 连线：源侧 lane → 腰点、腰点 → 目标侧 lane（相邻两列共用同一个腰点）。
           空列（off）不连线，光点会跳过它。 */
        for (i = 0; i < KEYS.length - 1; i++) {
          var A = geo[KEYS[i]], B = geo[KEYS[i + 1]];
          if (A.off || B.off) continue;
          var ys = [];
          for (j = 0; j < A.lanes.length; j++) ys.push(A.lanes[j].y);
          for (j = 0; j < B.lanes.length; j++) ys.push(B.lanes[j].y);
          var waist = { x: (A.cx + B.cx) / 2,
                        y: (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2 };
          waists[i] = waist;
          var sx = A.cx + BOXW / 2, tx = B.cx - BOXW / 2;
          for (var ai = 0; ai < A.lanes.length; ai++) {
            var din = arc({ x: sx, y: A.lanes[ai].y }, waist);
            var pin = mk("path", { d: din, fill: "none", stroke: "var(--line-strong)",
                                   "stroke-width": 1, opacity: ".75" });
            layers.edge.appendChild(pin);
            edgesIn[i + ":" + ai] = { d: din, el: pin };
          }
          for (var bi = 0; bi < B.lanes.length; bi++) {
            var dout = arc(waist, { x: tx, y: B.lanes[bi].y });
            var pout = mk("path", { d: dout, fill: "none", stroke: "var(--line-strong)",
                                    "stroke-width": 1, opacity: ".75" });
            layers.edge.appendChild(pout);
            edgesOut[i + ":" + bi] = { d: dout, el: pout };
          }
        }
        svg.appendChild(layers.defs);
        svg.appendChild(layers.edge);
        svg.appendChild(layers.bypass);  /* 旁路在连线之上、方块之下（不挡文字） */
        svg.appendChild(layers.box);
        svg.appendChild(layers.lbl);
        svg.appendChild(layers.pulse);   /* 光点在最上层 */
        sig = signature(lanes);
        lenCache = {};                   /* 连线/旁路的 d 都变了：量过的长度作废 */
        /* 结构重建后把上一拍的读数补回去：水位/旁路是「数据」，不是「结构」 */
        setWater(waterLv.cache, waterLv.out);
        /* 注：没有 edges["out:out"] 就没有「腰点 → 成片」那条实线，也就没有可并的
           汇合点 —— 那种情况下 bypassGeom() 直接返回 null（宁可不画，也不自己猜落点）。 */
        /* 结构重建 = 坐标全变：先升版本，再按新坐标重画（setBypass 靠这个版本号判定） */
        bypassGen += 1;
        drawBypass();
        return true;
      }
