# -*- coding: utf-8 -*-
"""应用遮罩：给 aligned 目录里的每张 DFLJPG 补 XSeg 遮罩（提取之后的单独一步）。

  _mask_state / _mask_ai_stub   遮罩编辑器（routes_mask）用的状态与 501 占位
  _mask_worker                  批量补遮罩的线程体（进度写 JOB）

提取本身不跑 XSeg：DFL 的流程也是 extract → mask 两步。函数体 / 注释逐字节照抄 app.py。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from .._common.jobs import JOB
from ..backend_extract.detectors import _get_xseg


# ============================================================
# 遮罩编辑器（照搬 MaskProcessor/ui 的 WebUI；那份源码已归档到
# app/docs/legacy-reference/MaskProcessor/ui，4.1 的实现在 app/web/mask-editor/）
# 只实现不依赖模型的三个：开项目 / 读图 / 存遮罩
# 需要 AI 的功能（SAM / BiSeNet / XSeg）返回 501，界面上会提示
# ============================================================
_mask_state = {"dir": None, "files": [], "index": 0}






def _mask_ai_stub():
    raise HTTPException(501, "这个功能要加载 AI 模型（SAM / BiSeNet / XSeg），当前只接了手绘与保存")

def _mask_worker(folder: str, workers: int = 8, model: str | None = None, out_size: int = 256):
    """给一个 aligned 目录里的所有 DFLJPG 补 XSeg 遮罩（提取之后的单独一步）。

    提取本身不跑 XSeg：DFL 的流程也是 extract → mask 两步，
    而且遮罩模型是 3.5 ms/张，放在提取热路径上纯属浪费。
    """
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    from .._align import read_dfljpg_metas, rewrite_dfljpg_meta, encode_xseg_mask

    xseg = _get_xseg(model)
    root = Path(folder)
    files = sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))
    total = len(files)
    with JOB.lock:
        JOB.running = True
        JOB.msg = "应用遮罩：共 %d 张" % total
    if total == 0:
        with JOB.lock:
            JOB.running = False
            JOB.pct = 100
            JOB.msg = "目录里没有 DFLJPG：%s" % str(root)
        return 0
    JOB.begin(2, total)
    if xseg is None:
        with JOB.lock:
            JOB.running = False
            JOB.msg = "找不到 XSeg 模型（workspace/model/XSegLite）"
        return 0

    done = {"n": 0, "ok": 0, "moved": 0}

    def _one(fp):
        try:
            if not JOB.running:
                return
            head, legacy = read_dfljpg_metas(fp)
            if head is None and legacy is None:
                return                                  # 不是 DFLJPG（裸 jpg/png）
            if (head or {}).get("xseg_mask") is not None:
                return                                  # 头部已带遮罩：引擎读得到
            meta = dict(legacy or {})
            meta.update(head or {})                     # 头部优先，legacy 只补缺键
            if meta.get("xseg_mask") is not None:
                # 遮罩在 EOI 前那份 legacy 里（引擎看不到）：搬到头部，不重算
                rewrite_dfljpg_meta(fp, meta)
                with JOB.lock:
                    done["moved"] += 1
            else:
                img = _imread_any(fp)
                if img is None:
                    return
                enc = encode_xseg_mask(xseg.mask(img, out_size=int(out_size)))
                if enc:
                    meta["xseg_mask"] = enc
                    rewrite_dfljpg_meta(fp, meta)
                    with JOB.lock:
                        done["ok"] += 1
        except Exception:
            pass
        finally:
            with JOB.lock:
                done["n"] += 1
                JOB.done = done["n"]
                JOB.pct = int(done["n"] * 100 / max(1, total))
                if done["n"] % 20 == 0 or done["n"] == total:
                    JOB.msg = ("应用遮罩 %d / %d · 新写入 %d · 迁移 %d"
                               % (done["n"], total, done["ok"], done["moved"]))

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        list(pool.map(_one, files))

    with JOB.lock:
        JOB.running = False
        JOB.pct = 100
        JOB.msg = "应用遮罩完成：%d 张，新增 %d，迁移 %d" % (total, done["ok"], done["moved"])
    return done["ok"]
