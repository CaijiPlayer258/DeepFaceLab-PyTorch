      /* ============================================================
         无脸旁路：切脸之后直接汇入「缓存帧」，绕开 换脸 / 遮罩 / 合成
         ------------------------------------------------------------
         事实：一张脸都没检出的帧不需要 dfm/xseg，合成也是原帧直通，最后只走编码。
         所以图上从 det 右侧拉一条**淡线**沉到所有 lane 下面、再升到编码列 —— 线上写
         「无脸 N 帧」。不画的话，「切脸 1439 / 换脸 1204」这 235 帧看着像丢了
         （用户就是这么理解的，还自己在截图上补了一条红线）。
         只有这一单真有 no_face_ticks>0 时才出现；0 就没有这条线，也不留空节点。
         ============================================================ */
      var bypassN = 0, bypassD = "", bLabel = null;
      function bypassGeom() {
        var A = geo.det, B = geo.encode, O = geo.out;
        if (!A || !B || A.off || B.off || !A.lanes.length || !B.lanes.length) return null;
        /* 没有 O.waist / O.tailC 就不画：那条「腰点 → 成片」的实线是汇合点，
           它不存在时旁路就无处可并（以前这里会退到 B.cx 上另算一个落点 ——
           于是「旁路」和主路各走各的，这正是用户说「看着和主路没关系」的由来）。 */
        if (!O || !O.waist || !O.tailC) return null;
        var a0 = A.lanes[0];
        var x1 = A.cx + BOXW / 2, y1 = a0.y;
        /* 汇合点 = 主路腰点：x = (缓存帧右缘 + 杯左缘) / 2，y = 编码 lane 的中线。
           末段直接拼 O.tailC（主路 out:out 去掉 M 的那一段）——**同一份坐标**，
           所以虚线的尾巴就压在进杯实线上，看起来是「汇进去、一起倒进成片」，
           而不是两条线在杯口各画各的。 */
        var wx = O.waist.x, wy = O.waist.y;
        /* 沉到**所有列**的 lane 底下（再往下 20px）：只看首尾两列不够 —— 中间那列
           合成有 8 条 lane，比切脸/编码都深，旁路会从它最后两条中间穿过去。 */
        var bot = 0, i2, g2, l2;
        for (i2 = 0; i2 < KEYS.length; i2++) {
          g2 = geo[KEYS[i2]];
          if (!g2 || g2.off) continue;
          for (l2 = 0; l2 < g2.lanes.length; l2++) {
            bot = Math.max(bot, g2.lanes[l2].y0 + g2.lanes[l2].h);
          }
        }
        var yb = Math.min(H - 12, (bot || (a0.y0 + a0.h)) + 20);
        /* 「竖起来」的那一段要留在**缓存帧列右缘之外的空档**里（xr → wx 这一段，
           999.7~1116 之间）。以前是从 mx 一路斜拉到腰点，曲线中段正好从 e4/e5
           两个方块背后穿过去 —— 方块盖在旁路上面，只看得到几个角上的虚线段，
           读者根本读不出「升上来、汇进去」这件事。 */
        var xr = B.cx + BOXW / 2 + 12;       /* 刚出缓存帧列：竖起来的起点 */
        var mx = (x1 + xr) / 2;              /* 谷底水平段的中点（标签挂这儿） */
        var r = function (v) { return Math.round(v * 10) / 10; };
        var rel = function (a, b) { return arc(a, b).replace(/^M[^C]*/, ""); };
        return { /* ① det 右缘 →（沉到谷底）→ 空档里竖直升起 → 腰点
                    每段两端切线都水平，和主路同一个 arc() 约定，接上去不打折 */
                 d: "M" + r(x1) + "," + r(y1) +
                    rel({ x: x1, y: y1 }, { x: xr, y: yb }) +
                    rel({ x: xr, y: yb }, { x: wx, y: wy }) +
                    /* ② 腰点 → 杯左缘：主路那一段的原文（含控制点），一字不改地接上 */
                    O.tailC,
                 label: { x: mx, y: yb - 6 },
                 merge: { x: wx, y: wy },                     /* 汇合点 = 主路腰点 */
                 end: { x: O.cx - OUTW / 2, y: wy } };        /* 终点 = 杯左缘 */
      }
      function drawBypass() {
        if (!layers) return;
        /* 「我已经按这一版结构算过了」——即便下面因为 bypassN=0 提前返回也要记，
           否则 setBypass 会以为没画过、每一拍都重建 DOM。 */
        bypassDrawnGen = bypassGen;
        var g = layers.bypass;
        while (g.firstChild) g.removeChild(g.firstChild);
        bypassD = "";
        bLabel = null;
        if (!bypassN) return;
        var bg = bypassGeom();
        if (!bg) return;
        bypassD = bg.d;
        var p = mk("path", { d: bg.d });        /* 颜色/虚线/透明度全在 CSS 里 */
        p.setAttribute("data-bypass", "no-face");
        g.appendChild(p);
        /* 「无脸 N 帧」这句**不画在图上**了（用户：流水线本身已经说明了这件事）。
           但信息不丢：挂成虚线的 <title>，鼠标停在旁路线上才显示帧数。 */
        bLabel = mk("title", {});
        bLabel.textContent = "无脸 " + bypassN + " 帧（这些帧直接进缓存，不经过换脸/遮罩/合成）";
        p.appendChild(bLabel);
      }
      function setBypass(n) {
        n = Math.max(0, Math.floor(Number(n) || 0));
        var had = bypassN > 0;
        var shown = !!(layers && layers.bypass && layers.bypass.firstChild);
        bypassN = n;
        /* 已经画着**而且结构没重建过**，才只改读数（250ms 一拍，不重建 DOM）。
           以前这里是无条件 return —— 只要先画过一次，之后 build() 哪怕把列宽、
           杯子位置全换了，旁路的 d 也冻在旧坐标上（用户改了两版都「没被画出来」，
           另一半原因就是它）。bypassDrawnGen !== bypassGen 就说明坐标已经作废。 */
        if (n > 0 && shown && bypassDrawnGen === bypassGen) {
          if (bLabel && bLabel.textContent.indexOf("无脸 " + n + " 帧") !== 0) {
            bLabel.textContent = "无脸 " + n + " 帧（这些帧直接进缓存，不经过换脸/遮罩/合成）";
          }
          return;
        }
        if (n === 0 && !shown && !had) return;   /* 本来就没有：不碰 DOM */
        drawBypass();
      }

      function setCount(k, ticks) {
        var g = geo[k];
        if (!g || !g.count) return;
        var txt = (ticks === null || ticks === undefined) ? "— 帧" : (ticks + " 帧");
        if (g.count.textContent !== txt) g.count.textContent = txt;
      }
      function markStall(k, on) {
        on = !!on;
        if (stallNow[k] === on) return;
        stallNow[k] = on;
        var g = geo[k];
        if (!g) return;
        for (var i = 0; i < g.lanes.length; i++) {
          if (on) g.lanes[i].rect.classList.add("stall");
          else g.lanes[i].rect.classList.remove("stall");
        }
      }
      function setStates(m) {
        var changed = false, i, k;
        for (i = 0; i < KEYS.length; i++) {
          k = KEYS[i];
          var st = (m && m[k]) || "";
          if (state[k] === st) continue;
          state[k] = st;
          changed = true;
          var g = geo[k];
          if (g && g.col) {
            g.col.classList.remove("on", "done", "err");
            if (st) g.col.classList.add(st);
          }
        }
        return changed;
      }
      /* 点亮也铺在同一个窗口里：at = 这一次交接「应该」发生的时刻（不会早于现在）。
         同一帧里同一条 lane 只闪一次（5 帧叠在 1/60 秒里看不出差别，但会白跑 4 次强制重排）。 */
      function flashOne(k, i, at) {
        flashes.push({ k: k, i: i, at: at || 0 });
      }
      function recycle(p) {
        p.c.setAttribute("opacity", "0");
        /* 坐标也清掉：池里的圆点不该带着上一次飞行的残坐标（探针/截图都可能看见它） */
        p.c.removeAttribute("cx");
        p.c.removeAttribute("cy");
        pool.push(p.c);
        /* 光点的临时 path 用完就删（它只是给 getPointAtLength 当尺子，
           不进 DOM 的可见层，也不留垃圾） */
        if (p.path && p.path.parentNode) p.path.parentNode.removeChild(p.path);
        p.path = null;
      }
      /* d = 源 lane→腰点 那段 + 腰点→目标 lane 那段（第二段以 "M 腰点" 开头，
         位置正好落在腰点上，所以拼起来的路径是**连续**的，光点从源 lane 一路走到目标 lane）。 */
      var lastSeen = {};      /* 这一拍要亮哪几条 lane（去重，给拍照用） */
      function rememberFlash(k, i) {
        var id = k + "#" + i;
        if (lastSeen[id]) return;
        lastSeen[id] = 1;
        lastFlashed.push({ k: k, i: i });
        if (lastFlashed.length > 64) lastFlashed.shift();
      }
      var lenCache = {};      /* 同一条「源→腰→目标」路径的长度只量一次 */
      function addPulse(key, d, startAt) {
        if (!layers) return;
        if (pulses.length >= MAX_PULSE) recycle(pulses.shift());   /* 超上限丢最旧的 */
        var path = mk("path", { d: d, fill: "none", stroke: "none" });
        layers.pulse.appendChild(path);
        var L = lenCache[key];
        if (L === undefined) {
          /* getTotalLength() 会逼一次 SVG 布局；同一条路径反复量就是纯浪费（240 个/秒时很致命） */
          L = path.getTotalLength();
          lenCache[key] = L;
        }
        var c = pool.pop() || mk("circle", { r: 2.4, fill: "var(--ok)" });
        if (!c.parentNode) layers.pulse.appendChild(c);
        c.setAttribute("opacity", "0");          /* 还没到起飞时刻：先藏着 */
        pulses.push({ c: c, path: path, L: L, t0: startAt,
                      dur: 470 + Math.random() * 280 });
      }
      /* 一次 tick：点亮发起的那条 lane，再沿曲线放一个光点过去。
         多 lane 的阶段（合成/编码）拿不到「是哪一条 lane 干的」—— status 里不带这个，
         每帧都带一份会让 500ms 的那次轮询白白变大。所以按 tick 序号**轮转**，
         保证光点始终落在**真实存在的 lane** 上（不是编出来的位置）。 */
      function spawn(k, n, sched) {
        var si = KEYS.indexOf(k);
        if (si < 0) return;
        if (si === KEYS.length - 1) {
          for (var z = 0; z < n; z++) {
            seq += 1;
            var li = seq % Math.max(1, geo[k].lanes.length);
            flashOne(k, li, sched());
            rememberFlash(k, li);
          }
          return;
        }
        var to = si + 1;
        while (to < KEYS.length && geo[KEYS[to]] && geo[KEYS[to]].off) to += 1;
        var A = geo[k];
        var B = (to < KEYS.length) ? geo[KEYS[to]] : null;
        for (var j = 0; j < n; j++) {
          seq += 1;
          var ai = A.lanes.length ? (seq % A.lanes.length) : 0;
          var at = sched();          /* 这一发「应该」发生的时刻（铺在轮询窗口里） */
          flashOne(k, ai, at);
          rememberFlash(k, ai);
          if (!B || reduceMotion()) continue;
          var bi = B.lanes.length ? ((seq * 7 + 3) % B.lanes.length) : 0;
          /* 出边要挂在**去的那一列左边那个**腰点上：to 跳过空列时 to-1 != si，
             以前一律写 si 会取到别的腰点（只在有空列时才错，且会取到 undefined 而不画）。 */
          var ein = edgesIn[si + ":" + ai], eout = edgesOut[(to - 1) + ":" + bi];
          if (ein && eout) addPulse(si + ":" + ai + ":" + bi, ein.d + " " + eout.d, at);
        }
      }
      function kick() {
        if (raf) return;
        raf = window.requestAnimationFrame(tick);
      }
      function tick(t) {
        raf = 0;
        if (freezeAt > 0) t = freezeAt;      /* 截图定格（见 freezeAt 的说明） */
        /* ① 点亮：摘类 → 强制重排 → 挂类（动画才会重放；衰减途中再来一次会重新点亮，
              不同 lane 各挂各的，互不打断）。合到一帧里做，减少强制重排次数。 */
        /* 点亮是铺在窗口里的：只处理 at <= t 的那些，没到点的留到下一帧（不丢） */
        var keep = [];
        while (flashes.length) {
          var f = flashes.shift();
          if (f.at && f.at > t) { keep.push(f); continue; }
          var g = geo[f.k];
          var lane = g && g.lanes[f.i];
          if (!lane) continue;
          if (t - (lane.lastLit || -1e9) < 16) continue;   /* 同一帧里同一条 lane 只闪一次 */
          lane.lastLit = t;
          lane.rect.classList.remove("lit");
          void lane.rect.getBoundingClientRect();
          lane.rect.classList.add("lit");
          /* 每条 lane 自己记「这是第几次亮」：1.25s 后只允许**自己那一次**摘类。
             没有这个守卫时，同一 lane 在 1.25s 内又亮一次，前一次的定时器会把
             后一次的 .lit 提前摘掉（动画被掐断、图上忽然全灭）—— 而「叠加」正是
             用户要的效果：衰减中再来一次应该重新点亮，不是提前熄掉。 */
          lane.gen = (lane.gen || 0) + 1;
          /* 注意把 lane 本身也捕进来：只捕 rect 的话闭包里那个 lane 是外层循环变量，
             循环结束后指向最后一条 lane —— 守卫会比错对象，个别 lane 就一直亮着。 */
          (function (ln, rect, mine) {
            window.setTimeout(function () {
              if (mine !== ln.gen) return;        /* 已经被更新的一次点亮接管 */
              rect.classList.remove("lit");
            }, LIT_MS);
          })(lane, lane.rect, lane.gen);
        }
        if (keep.length) flashes = keep.concat(flashes);
        /* ② 光点：沿曲线 getPointAtLength 走；只写 cx/cy/opacity，不读布局。 */
        for (var i = pulses.length - 1; i >= 0; i--) {
          var p = pulses[i];
          var kk = (t - p.t0) / p.dur;
          if (kk < 0) {                     /* 还没到起飞时刻：藏在源 lane 边上 */
            if (p.c.getAttribute("opacity") !== "0") p.c.setAttribute("opacity", "0");
            continue;
          }
          kk = Math.min(1, kk);
          var pt = p.path.getPointAtLength(p.L * kk);
          p.c.setAttribute("cx", pt.x.toFixed(1));
          p.c.setAttribute("cy", pt.y.toFixed(1));
          p.c.setAttribute("opacity", (0.9 * (1 - kk * 0.5)).toFixed(2));
          if (kk >= 1) { recycle(p); pulses.splice(i, 1); }
        }
        /* 定格时只画这一帧就停（不然 pulses 永远不结束 = 白烧 rAF） */
        if (!freezeAt && (pulses.length || flashes.length)) kick();
      }

      /* 喂一拍 status：列标题计数、点亮、光点、卡住，全在这里
         （inflight 后端照给，只是界面不再显示「在飞 N」） */
      function paint(d) {
        d = d || {};
        var right = $("mgPipeRight");
        var rate = Number((d.vid || {}).frame_fps || d.frame_fps || 0);
        if (right) right.textContent = rate > 0 ? (rate.toFixed(1) + " 帧/秒") : "— 帧/秒";
        /* 分相位流水线（多模型）：六列图是并发视图，表达不了「趟是串行的」。
           后端发 pipeline.shape=passes 就改画相位图，两条路各画各的真实形状。 */
        if (d.pipeline && d.pipeline.shape === "passes" && window.mgPipePhases) {
          if (window.mgPipePhases(d)) {
            var sub0 = $("mgPipeSub");
            if (sub0) sub0.textContent = "分相位 · 趟是串行的";
            return;
          }
        }
        var live = isLive(d.stages) ? d.stages
                 : (isLive(d.stages_live) ? d.stages_live : null);
        if (!live) {
          /* 还没开单 / 后端还没起流水线（模型加载那几秒）：先把六列摆出来当占位，
             别让这块是个空白框；真 lanes 一到，签名变了会自动重建。 */
          if (!layers) build(lanesOf(null));
          for (var z = 0; z < KEYS.length; z++) setCount(KEYS[z], null);
          return;
        }
        var lanes = lanesOf(live);
        if (signature(lanes) !== sig) build(lanes);

        /* ---- 水位 + 无脸旁路：两个读数都只用 status 里的真数字 ----
           水位 = 这一单攒了多少帧（缓存列）/ 已经变成成片的多少帧（成片杯）；
           旁路 = 这一单有多少帧根本没有脸（det.no_face_ticks，后端低成本计数）。
           收工那一拍：缓存清零、成片满格（cache 目录删了/帧都进了成片）。 */
        var framesTotal = Number(((d.vid || {}).total) || d.frames_total || d.total || 0);
        var composed = Math.max(0, Number((live.compose || {}).ticks) || 0);
        var coded = Math.max(0, Number((live.encode || {}).ticks) || 0);
        var cacheLv = framesTotal > 0 ? ((composed - coded) / framesTotal) : 0;
        var outLv = framesTotal > 0 ? (coded / framesTotal) : 0;
        var finished = !d.running &&
                       (d.ok === true || d.phase === "done" || d.state === "done");
        if (finished) { cacheLv = 0; outLv = 1; }
        setWater(cacheLv, outLv);
        var nf = Math.max(0, Number((live.det || {}).no_face_ticks) || 0);
        var dnf = (prevNf === undefined) ? 0 : Math.max(0, nf - prevNf);
        prevNf = nf;
        setBypass(nf);

        var t = nowMs();
        var running = !!d.running;
        var wants = [], total = 0, i, k, row, ticks, delta, n;
        for (i = 0; i < KEYS.length; i++) {
          k = KEYS[i]; row = live[k] || {};
          ticks = Math.max(0, Number(row.ticks) || 0);
          setCount(k, ticks);
          delta = (prev[k] === undefined) ? 0 : (ticks - prev[k]);
          if (delta > 0) lastAt[k] = t;
          else if (lastAt[k] === undefined) lastAt[k] = t;  /* 第一次看到：从这儿算「多久没动」 */
          prev[k] = ticks;
          wants.push(delta > 0 ? delta : 0);
          if (delta > 0) total += delta;
          /* 卡住 = 还在跑 + 这一段没完成 + 已经 >600ms 没 tick（tick 一恢复就自动复原） */
          markStall(k, running && state[k] !== "done" &&
                       (t - (lastAt[k] || t)) > STALL_MS);
        }
        /* 这一拍放多少个光点：按 PULSE_PER_S 折算的预算**按比例**稀释（各段比例不变），
           再把这个总数的每一发均匀铺在 POLL_MS 窗口里 —— 20 帧/秒时就是每 250/5 = 50ms
           一发，肉眼是连续流，而不是「每半秒跳一波」。 */
        var budget = Math.max(4, Math.round(PULSE_PER_S * POLL_MS / 1000));
        var scale = total > budget ? (budget / total) : 1;
        var plan = [], wantSpawn = 0;
        for (i = 0; i < KEYS.length; i++) {
          n = Math.max(0, Math.round(wants[i] * scale));
          plan.push(n);
          wantSpawn += n;
        }
        lastSpawn = wantSpawn;
        /* 无脸帧也真的走旁路：按 no_face_ticks 的增量在旁路线上放几个点（同一份 status
           的真实数字）。预算被主路占满时就不放 —— 旁路是配角，不能把主路的点挤掉。 */
        var bSpawn = 0;
        if (dnf > 0 && bypassD && !reduceMotion()) {
          bSpawn = Math.max(0, Math.min(budget - wantSpawn,
                                        Math.max(1, Math.round(dnf * scale))));
        }
        lastBypass = bSpawn;
        if (wantSpawn + bSpawn > 0) {
          var slotMs = POLL_MS / (wantSpawn + bSpawn);
          var slot = 0;
          var sched = function () { slot += 1; return t + slot * slotMs; };
          lastSeen = {};
          for (i = 0; i < KEYS.length; i++) {
            if (plan[i] > 0) spawn(KEYS[i], plan[i], sched);
          }
          for (i = 0; i < bSpawn; i++) addPulse("bypass", bypassD, sched());
        }
        kick();
      }

      function reset() {
        var i;
        freezeAt = 0;
        lastFlashed = [];
        lastSeen = {};
        lastSpawn = 0;
        lastBypass = 0;
        prevNf = undefined;      /* 换一单：旁路光点不补放历史增量 */
        setWater(0, 0);
        setBypass(0);
        kick();                       /* 上一张照片可能把时钟按住了：重开后要重新起 ticker */
        for (i = 0; i < pulses.length; i++) recycle(pulses[i]);
        pulses = [];
        flashes = [];
        prev = {};
        lastAt = {};
        seq = 0;
        for (i = 0; i < KEYS.length; i++) {
          stallNow[KEYS[i]] = false;
          markStall(KEYS[i], false);
          setCount(KEYS[i], null);
          /* 换一单 / 重连：把上一单残留的 .lit 也摘掉（否则它要等各自的 1.25s
             定时器才消失，图上会有一片「没人点也亮着」的方块） */
          var g = geo[KEYS[i]];
          if (g) {
            for (var j = 0; j < g.lanes.length; j++) g.lanes[j].rect.classList.remove("lit", "stall");
          }
        }
        setStates({});
        var right = $("mgPipeRight");
        if (right) right.textContent = "— 帧/秒";
      }

      function debug() {
        var out = { keys: KEYS, sig: sig, pulses: pulses.length, pool: pool.length,
                    stalled: {}, ticks: {}, state: {}, lanes: {},
                    edges: 0, edges_in: 0, edges_out: 0, waists: {},
                    last_flashed: lastFlashed, last_spawn: lastSpawn,
                    last_bypass: lastBypass,
                    /* cache/out 是**口径源**（差值 / encode），blue/green 是成片杯真正画出来的两层 */
                    water: { cache: waterLv.cache, out: waterLv.out,
                             blue: waterLv.blue, green: waterLv.green },
                    bypass: bypassN, bypass_drawn: !!bypassD,
                    /* 旁路原文 + 两个关键点：验收/排障直接读，不用再从像素里猜。
                       merge = 与主路汇合的腰点；end = 杯左缘（贴着进杯实线走完最后一段）；
                       out_tail_c = 主路 out:out 去掉 M 的那一段 —— bypass_d 必须**以它结尾**，
                       否则就还是「两条线各画各的」。 */
                    bypass_d: bypassD,
                    bypass_merge: (bypassD && geo.out && geo.out.waist)
                      ? { x: geo.out.waist.x, y: geo.out.waist.y } : null,
                    bypass_end: (bypassD && geo.out && geo.out.waist)
                      ? { x: geo.out.cx - OUTW / 2, y: geo.out.waist.y } : null,
                    out_tail_c: (geo.out && geo.out.tailC) || "",
                    bypass_gen: bypassGen, bypass_drawn_gen: bypassDrawnGen,
                    max_pulse: MAX_PULSE, pulse_per_s: PULSE_PER_S, poll_ms: POLL_MS,
                    built: !!layers };
        var i, k;
        for (i = 0; i < KEYS.length; i++) {
          k = KEYS[i];
          out.stalled[k] = !!stallNow[k];
          out.ticks[k] = (prev[k] === undefined ? null : prev[k]);
          out.state[k] = state[k] || "";
          out.lanes[k] = geo[k] ? geo[k].lanes.length : 0;
        }
        for (i in edgesIn) { if (Object.prototype.hasOwnProperty.call(edgesIn, i)) out.edges_in += 1; }
        for (i in edgesOut) { if (Object.prototype.hasOwnProperty.call(edgesOut, i)) out.edges_out += 1; }
        out.edges = out.edges_in + out.edges_out;      /* 画出来的连线总条数 */
        for (i in waists) {
          if (Object.prototype.hasOwnProperty.call(waists, i)) {
            out.waists[i] = { x: Math.round(waists[i].x * 100) / 100,
                              y: Math.round(waists[i].y * 100) / 100 };
          }
        }
        return out;
      }

      function freeze(at) {
        freezeAt = (Number(at) > 0) ? Number(at) : 0;
        if (!freezeAt) kick();
      }

      return { paint: paint, reset: reset, setStates: setStates, debug: debug,
               freeze: freeze, KEYS: KEYS, isLive: isLive };
    })();

    /* 列标题的状态（和分段链那套 mgpPaint 保持同一个入参口径） */
    function mgpWant(opts) {
      if (opts.state === "err") return "err";
      if (opts.state === "done") return "done";
      if (opts.state === "idle") return "";
      return "on";
    }
    /* 画到第 upto 步：前面全部 done，当前步 on / done / err / idle */
    function mgpPaint(upto, opts) {
      opts = opts || {};
      var st = mgpWant(opts);
      var m = {}, k;
      for (k = 0; k < MGP_STAGES.length; k++) {
        m[MG_KEY[MGP_STAGES[k]]] = (k < upto) ? "done" : (k === upto ? st : "");
      }
      MG_PIPE.setStates(m);
    }
    /* 融合流水线：六段并行，六个列标题同一个状态 */
    function mgpFusedSteps(st) {
      var v = (st === "done" ? "done" : (st === "err" ? "err" : (st === "idle" ? "" : "on")));
      var m = {}, k;
      for (k = 0; k < MGP_STAGES.length; k++) m[MG_KEY[MGP_STAGES[k]]] = v;
      MG_PIPE.setStates(m);
    }
    /* 整条链跑完：老接口保留（调用点在这页有好几处），只在图上写一句结论。
       不再「六个点一起发光」—— 发光那一套随竖排六步一起去掉了。 */
    function mgpChainDone() {
      var sub = $("mgPipeSub");
      if (sub) sub.textContent = "六段全部完成";
    }
    function mgpRejectShow(d, onForce) {
      d = d || {};
      var box = $("mgpResult");
      /* 底部那排出口：终态才出现（文本在标题下、按钮在页面底部） */
      var actsBox = $("mgpResultActs"); if (actsBox) actsBox.hidden = false;
      var txt = $("mgpResultText");
      var cb = $("mgpCancel");
      if (cb) cb.hidden = true;
      ["mgpStepsBox", "mgpFoot"].forEach(function (id) {
        var el = $(id);
        if (el) el.hidden = true;
      });
      var ttl = $("mgpHdTitle");
      /* 强制尝试撞 OOM 那一档是「跑过才失败」，别再说「未启动」—— 用户会以为没动过 */
      if (ttl) ttl.textContent = d.vram_oom ? "确实不够（已回滚）" : "导出未启动";
      if (txt) {
        var line = mgVramLine(d);
        var tip = mgVramAdvice(d);
        txt.textContent = tip ? (line + " · " + tip) : line;
      }
      /* 「仍然尝试（可能 OOM）」：只有后端说 can_force 且有重试入口时才亮出来。
         用户口径：明说不理解为什么被拒，那他更想**试**一次 —— 拒绝面板得给这条路。 */
      var fb = $("mgpForceTry");
      if (fb) {
        var can = (typeof onForce === "function") && d.can_force !== false;
        fb.hidden = !can;
        /* mgpRejectHide / mgpResultShow 会把 display:none 写成**内联**样式；这里必须清掉，
           否则「被拒 → 离开 → 再被拒」之后那颗按钮再也回不来（内联样式压过类规则）。
           以前它能一直挂着，靠的正是「.btn 的 display 压过了 [hidden]」这个 bug —— 修好
           [hidden] 之后，这条显式的清场就成了必须的。 */
        fb.style.display = can ? "" : "none";
        fb.disabled = false;
        fb.title = "按最小档真跑一次；真 OOM 会自动回滚到这一页，不会把后端搞坏";
      }
      if (box) { box.hidden = false; box.classList.add("is-rej"); }
      return box;
    }
    /* 回到「正常进度页」的样子（用户改完设置再点一次时先把拒绝态收起来） */
    function mgpRejectHide() {
      var box = $("mgpResult");
      if (box) box.classList.remove("is-rej");
      var fb0 = $("mgpForceTry");
      if (fb0) { fb0.hidden = true; fb0.style.display = "none"; }   /* 离开被拒态就收起「仍然尝试」 */
      var ttl = $("mgpHdTitle");
      if (ttl) ttl.textContent = "正在导出";
      ["mgpStepsBox", "mgpFoot"].forEach(function (id) {
        var el = $(id);
        if (el) el.hidden = false;
      });
    }
    window.mgpRejectShow = mgpRejectShow;
    window.mgpRejectHide = mgpRejectHide;

    /* 显存数字 -> 「需要 2.44 GB，可用 0.60 GB」（后端 400 里的 need_mb / avail_mb） */
    function mgGb(mb) {
      var n = Number(mb);
      if (!isFinite(n) || n < 0) return "";
      return (n / 1024).toFixed(2) + " GB";
    }
    /* 一行话：优先用后端的 summary（它把「预留」和「本程序 / 其它程序 / 空闲」
       都摊在同一行里 —— 老文案只给 need/avail 两个数，用户当然算不出那 1.37GB 去哪了）。
       后端没给（老版本后端）就按同一口径在本地拼一遍，字段都来自 400 的顶层。 */
    function mgVramLine(d) {
      d = d || {};
      if (d.vram_oom) {
        return String(d.vram_oom.summary || d.vram_oom.msg || "强制尝试失败：确实不够，已回滚")
          .split("\n")[0];
      }
      if (d.summary) return String(d.summary).split("\n")[0];
      var need = mgGb(d.need_mb), avail = mgGb(d.avail_mb);
      if (!need || !avail) {
        return String(d.error || d.detail || "显存不足").split("\n")[0];
      }
      var bits = ["需要 " + need];
      if (d.free_mb != null) {
        bits.push("空闲 " + mgGb(d.free_mb) +
                  " 里预留 " + (d.reserve_mb != null ? mgGb(d.reserve_mb) : "?") +
                  " 给驱动/碎片 = 实际可用 " + avail);
      } else {
        bits.push("实际可用 " + avail);
      }
      if (d.own_mb != null || d.other_mb != null) {
        bits.push("本程序 " + mgGb(d.own_mb) +
                  " / 其它程序 " + (d.other_mb == null ? "?" : mgGb(d.other_mb)) +
                  " / 空闲 " + (d.free_mb == null ? "?" : mgGb(d.free_mb)));
      }
      return "显存不足：" + bits.join(" · ");
    }
    function mgVramAdvice(d) {
      var s = (d && d.suggest) || [];
      return String(s[0] || "").trim();
    }
    /* 同一条拒绝 4 秒内只弹一次（连点导出不叠 toast） */
    var MG_VRAM_TOAST = { at: 0, text: "" };
    function mgVramToast(d) {
      var line = mgVramLine(d);
      var now = Date.now();
      if (line === MG_VRAM_TOAST.text && (now - MG_VRAM_TOAST.at) < 4000) return null;
      MG_VRAM_TOAST.at = now;
      MG_VRAM_TOAST.text = line;
      return toast(line, mgVramAdvice(d) || "关掉占 GPU 的程序后重试", "error", 7000);
    }
    /* 被拒的统一收口：**一次**提示 + 拒绝态 + 复位三个忙闲位（还能再点一次）。
       被拒也要进进度页：用户口径是「只显示一行错误 + 返回导出设置」——
       停在设置页虽然也能再点，但就没有那个明确的出口和结论了。 */
    function mgVramReject(d, onForce) {
      d = d || {};
      window.__mergeExporting = false;
      window.__mgStageRunning = false;
      window.__mgFusedRunning = false;
      window.__mgVramChecking = false;
      var cb = $("mgpCancel");
      if (cb) { cb.disabled = false; cb.textContent = "取消导出"; }
      if (window.mgStep) window.mgStep("progress");
      /* onForce：这颗「仍然尝试（可能 OOM）」点下去干什么（见 mgVramPreflight）。
         不给就不亮 —— 预览这种没有重试入口的地方不会画出一颗死按钮。 */
      if (d.vram_oom) {
        /* 强制尝试撞 OOM：**不是**「未启动」，是「跑过、确实不够、已回滚」。
           文案里必须带上后端的 summary（需要在 / 可用 / 差多少），否则用户以为白点了一次。 */
        if (window.mgpRejectShow) window.mgpRejectShow(d, onForce);
        toast("确实不够，已回滚", String(d.summary || d.msg || "").slice(0, 200), "error", 9000);
        return false;
      }
      mgpRejectShow(d, onForce);
      mgVramToast(d);
      return false;
    }
    window.mgVramReject = mgVramReject;

    /* 「仍然尝试（可能 OOM）」的点击口：重试回调由 mgVramPreflight 挂在 window 上 */
    var mgForceBtn = $("mgpForceTry");
    if (mgForceBtn && !mgForceBtn.dataset.bound) {
      mgForceBtn.dataset.bound = "1";
      mgForceBtn.addEventListener("click", function () {
        var fn = window.__mgForceRetry;
        if (typeof fn !== "function") return;
        mgForceBtn.disabled = true;
        var old = mgForceBtn.textContent;
        mgForceBtn.textContent = "正在尝试…";
        try { fn(); } catch (e) { /* 重试自己挂了：留在被拒态，别把界面弄坏 */ }
        setTimeout(function () { mgForceBtn.disabled = false; mgForceBtn.textContent = old; }, 2500);
      });
    }

    /* 起 job 之前的显存预检：POST 同一份 body + dry_run=1。
       后端只算账（_vram_plan：不加载模型、不建 job、不开线程）：
         · 200 → 照常开跑；
         · 400 + need_mb/avail_mb/suggest → 这里就收口（**一次** toast + 拒绝态）；
         · 别的失败 → 不挡路，交给真请求自己报（免得两套错误文案）。
       回调 cb(proceed)：proceed=false 表示已经处理完（别开跑）。 */
    function mgVramPreflight(body, cb) {
      var probe = {};
      for (var k in body) {
        if (Object.prototype.hasOwnProperty.call(body, k)) probe[k] = body[k];
      }
      probe.dry_run = 1;
      var fired = false;                      /* 回调只许走一次：重复进 mgFusedStart 会发两次 POST */
      fetch("/api/merge/export", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(probe)
      }).then(function (r) {
        return r.json().then(function (d) { return { status: r.status, ok: r.ok, d: d || {} }; });
      }).then(function (res) {
        if (fired) return;
        var d = res.d || {};
        if (res.ok && d.ok !== false) { fired = true; cb(true, d); return; }
        if (res.status === 400 && d.need_mb != null) {
          fired = true;
          /* 强制尝试：把 force=1 写进**调用方那一份 body**（同一个对象引用），
             于是真请求（mgFusedStart / mgStageStart）也会带着它 —— 否则预检放行、
             真请求又被 400，用户看到的是「点了按钮还是不行」。 */
          var forceRetry = function () {
            body.force = 1;
            window.__mgVramChecking = true;
            mgVramPreflight(body, function (proceed2, d2) {
              window.__mgVramChecking = false;
              cb(proceed2, d2);
            });
          };
          window.__mgForceRetry = forceRetry;
          mgVramReject(d, forceRetry);        /* 一次 toast + 拒绝态 + 复位忙闲位 */
          cb(false, d);
          return;
        }
        fired = true;
        cb(true, d);                          /* 别的错误交给真请求自己报 */
      }).catch(function () {                  /* 预检自己挂了：别把用户挡在门外 */
        if (fired) return;
        fired = true;
        cb(true, {});
      });
    }
    window.mgVramPreflight = mgVramPreflight;
