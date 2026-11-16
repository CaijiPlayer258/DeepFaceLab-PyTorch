      /* ============================================================
         水位（进度）：只填「缓存帧」那一列 + 右侧「成片」那一格
         ------------------------------------------------------------
         两个数都来自 status.stages，前端不编：
           cache = (compose.ticks − encode.ticks) / 总帧数   ← 缓存里**现存**的帧
           out   = encode.ticks / 总帧数                     ← 已经编码出来的帧
         为什么是差值：帧从合成那一列流进缓存、再被编码那一列取走。合成 72 而编码 40
         就是「缓存里还压着 32 帧」—— 这就是缓存水位该显示的东西（用户口径：
         用 encode 与 compose 的差值推「已倒出多少」，不要新造后端字段）。
         快速导出（fused）是边合成边编码，这个差值就只有队列里那几十帧 —— 按真实比例，
         不做「先攒满再倒」的假动画。
         成片那杯是**两层**（就是用户要的「绿追蓝」）：
           blue  = cache + out = compose / 总帧   ← 蓝面：已经合成出来的帧
           green = out         = encode / 总帧    ← 绿面：已经编码成视频的帧（半透明，在蓝面之上）
         绿面永远 ≤ 蓝面；差值在收尾那一拍可能瞬时为负，setWater 里夹一刀。
         收工那一拍 compose = encode = 总帧 → 两层都到 1、两块面重合。
         ============================================================ */
      var waterLv = { cache: 0, out: 0, blue: 0, green: 0 };
      function wkey(v) { return Math.round(v * 1000) / 1000; }
      function setWater(cache, out) {
        cache = Math.max(0, Math.min(1, Number(cache) || 0));
        out = Math.max(0, Math.min(1, Number(out) || 0));
        waterLv.cache = cache;
        waterLv.out = out;
        var g = geo.encode;
        if (g && g.lanes.length) {
          var first = g.lanes[0], last = g.lanes[g.lanes.length - 1];
          var top = first.y0, bot = last.y0 + last.h;
          var span = Math.max(1, bot - top);
          /* 整列当一个杯子：水位线 y = 杯底 − level×杯高；每条 lane 只显示
             「自己这一段里被淹到的部分」→ 视觉上是一杯水往上漫，而不是六条一样高的条。 */
          var levelY = bot - cache * span;
          for (var i = 0; i < g.lanes.length; i++) {
            var ln = g.lanes[i];
            if (!ln.water) continue;
            var k = Math.max(0, Math.min(1, (ln.y0 + ln.h - levelY) / ln.h));
            ln.water.setAttribute("data-level", String(wkey(k)));
            ln.water.style.transform = "scaleY(" + wkey(k) + ")";
          }
        }
        /* 成片杯的两层水：蓝面 = 已合成（cache + out = compose/总帧）、绿面 = 已编码（out）。
           以前蓝面直接画 out —— 于是「编码一开始蓝面就突然满」，看不出水在被倒进视频。
           绿面夹在蓝面以内：encode 追不过 compose（差值瞬时为负时也不倒挂）。 */
        var blue = Math.max(0, Math.min(1, cache + out));
        var green = Math.min(blue, Math.max(0, Math.min(1, out)));
        waterLv.blue = blue;
        waterLv.green = green;
        var O = geo.out;
        if (O && O.water) {
          O.water.setAttribute("data-level", String(wkey(blue)));
          O.water.style.transform = "scaleY(" + wkey(blue) + ")";
        }
        if (O && O.enc) {
          O.enc.setAttribute("data-level", String(wkey(green)));
          O.enc.style.transform = "scaleY(" + wkey(green) + ")";
        }
        /* 标题区那颗小杯跟成片**同一个数**（蓝面 = 已合成）：它的 aria-label 写的是「已完成帧」 */
        var cup = $("mgPipeCupFill");
        if (cup) {
          cup.setAttribute("data-level", String(wkey(blue)));
          cup.style.transform = "scaleY(" + wkey(blue) + ")";
        }
      }
