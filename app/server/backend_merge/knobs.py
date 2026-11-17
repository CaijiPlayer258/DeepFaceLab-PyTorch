# -*- coding: utf-8 -*-
"""两个「值得让用户选」的性能旋钮：dfm 推理精度 / 合成后端。

合成链路上一共有 37 个 DFL_* 环境变量（清单与分类见 app/docs/27-ENV-KNOBS.md），
其中 35 个是**内部调优口**（已经量到最优，改了只会更慢）、排障开关（VRAM_* / OOM_INJECT /
ALLOW_TEST_HOOK）或有明确禁用理由的（DFM_TRT / DET_EP）。把它们摆到界面上，用户得到的是
「调坏了却不知道为什么变慢」—— 所以设置页只给下面这两个：

  · DFL_DFM_DTYPE  推理精度：直接影响画质 / 显存 / 速度，且没有「一台机器上唯一正确」的答案
                   （fp32 是对拍用的逃生门，bf16w 省显存，默认档见 dfm_torch.py）；
  · DFL_COMPOSE_GPU 合成后端：合成放 CPU 还是 GPU 取决于这张卡忙不忙（docs/26 实测：
                   overlay 持平、seamless −13.6%），所以要给用户一个「自动」以外的选择权。

三条口径（每条都有验收）：
  1. 值走**请求体**（body.dfm_dtype / body.compose_gpu），和其他设置项同一套；
  2. env 仍可覆盖：进程里显式设了 DFL_DFM_DTYPE / DFL_COMPOSE_GPU 就让它赢
     （排障、A/B 时不用改界面），请求体和 env 不一致会打一条 [eff-warn]；
  3. 「自动」= 请求体**不带这个键** —— 不是把「自动」写死成某个具体值发过去。
     请求体不带时这里把 env 还原成进程启动时的原值，避免上一单的选择残留到下一单。

回显：[eff] 里多两列 dfm_dtype / compose_backend（各带 *_src = ui | auto | default），
进度页「查看生效配置」抽屉里就是这两格。被流水线跳过时（选了 gpu 但 torch/cuda 起不来、
或 seamless 不进 GPU）由调用方打 [eff-warn]「选了 X、实际跑 Y、原因 Z」，并塞进
status.eff_warnings[] → 进度页顶部横幅。这就是检测器那次事故（界面写 TinyMog、日志跑
retinaface）的同类问题，同一个口径处理。

★ 两个值都不是 cfg 字段、而是**进程级**环境变量（dfm 会话 / 合成后端在各自的模块里读
  os.environ），所以这一层做的是「请求体 → env」的翻译 + 回显，别在流水线里再读一次 body。
"""
from __future__ import annotations

import os


# ---------------------------------------------------------------- 常量
DFM_ENV = "DFL_DFM_DTYPE"
GPU_ENV = "DFL_COMPOSE_GPU"

# 界面上给的 4 档；tf32 是内部对照档（核函数与 ORT 的 CUDA EP 同一数学模式），
# 只认 env、不上界面 —— 详见 dfm_torch.py 的表。
UI_DTYPES = ("auto", "fp32", "bf16", "bf16w")
DFM_DTYPES = ("auto", "fp32", "tf32", "bf16", "bf16w")

COMPOSE_MODES = ("auto", "cpu", "gpu")

# 界面上「自动」的显示名（回显/告警文案里要和界面长得一样，别写成 auto）
AUTO_LABEL = "自动"

# 进程启动时的原值：请求体不带这个键时**还原**成它。
#
# 为什么必须留一份：这些变量是进程级的，而请求体是**每一单**的。不还原的话，
# 「这一单选了 fp32」会一直粘到后面每一单上 —— 用户下一单看到的界面是「自动」，
# 实际却在跑 fp32，正是这次要根治的那类「界面和实际对不上」。
_ORIG = {
    "dtype": os.environ.get(DFM_ENV),
    "gpu": os.environ.get(GPU_ENV),
}


# ---------------------------------------------------------------- 小工具
def _dtype_of(value) -> str:
    """把请求体/env 里的精度写法归一；不认识的返回 ""（= 没说）。"""
    v = str(value or "").strip().lower()
    if v in ("fp16", "half"):       # 兼容老写法：8GB 卡上 fp16 定不了位，按 bf16 处理
        return "bf16"
    return v if v in DFM_DTYPES else ""


def _gpu_of(value):
    """compose_gpu：1/0（也认 true/false/on/off/gpu/cpu/auto）-> 1 / 0 / None。

    None = 「没说」（auto）。界面上「自动」时请求体不带这个键，所以这里是 None 最常见的一档。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        if int(value) == 0:
            return 0
        return 1 if int(value) > 0 else None
    v = str(value).strip().lower()
    if v in ("", "auto", "none", "-1"):
        return None
    if v in ("1", "on", "true", "yes", "gpu", "cuda"):
        return 1
    if v in ("0", "off", "false", "no", "cpu"):
        return 0
    return None


def _label(value: int) -> str:
    return "gpu" if value else "cpu"


# ---------------------------------------------------------------- 解析
def resolve(body) -> dict:
    """请求体 -> 这一单的两个旋钮（值 + 来源 + 需要提示的差异）。

    返回的 dict 会挂到 cfg["knobs"]，也是 [eff] 两个新列的唯一来源
    （回显和实际执行用的是同一份，别处不要再算一遍）：

        dfm_dtype / dfm_dtype_src        fp32 | bf16 | bf16w | auto    · ui | auto | default
        compose_backend / compose_backend_src   cpu | gpu              · ui | auto | default
        compose_gpu                      1 | 0 | ""      （写进 env 的原值；"" = 不动）
        warnings                         [[code, field, text], ...]   （env 覆盖了请求体）
    """
    body = body if isinstance(body, dict) else {}
    out = {
        "dfm_dtype": "auto", "dfm_dtype_src": "default",
        "compose_backend": "cpu", "compose_backend_src": "default",
        "compose_gpu": "",
        "dfm_dtype_env": False, "compose_env": False,
        # 人脸超分滑块（0..100）。不在这里做「能不能跑」的判断（那是 facesr 的事），
        # 这里只负责把界面上的值带到 env 去。
        "sr": _sr_level(body.get("sr")),
        "warnings": [],
    }

    # ---- ① 推理精度：env > 请求体 > 后端默认 ----
    env_dt = _dtype_of(_ORIG.get("dtype"))
    want_dt = _dtype_of(body.get("dfm_dtype"))
    if env_dt:
        out["dfm_dtype"], out["dfm_dtype_env"] = env_dt, True
        if want_dt and want_dt != env_dt:
            out["dfm_dtype_src"] = "default"
            out["warnings"].append(
                ["eff-env", "dfm_dtype",
                 "dfm_dtype 请求体写的是 %s，但进程里 %s=%s（环境变量优先）→ 实际跑 %s"
                 % (_show_dtype(want_dt), DFM_ENV, env_dt, env_dt)])
        elif want_dt:                     # 两边一致：界面上选的就是实际在跑的
            out["dfm_dtype_src"] = ("auto" if want_dt == "auto" else "ui")
        else:
            out["dfm_dtype_src"] = "default"
    elif want_dt:
        out["dfm_dtype"] = want_dt
        out["dfm_dtype_src"] = ("auto" if want_dt == "auto" else "ui")
    else:
        # 谁都没说：界面上是「自动」，这里也报「自动」——「落到哪一档」由 dfm 那一列
        # （torch · bf16w / onnx · fp32）回答，两个数不重复。
        out["dfm_dtype"], out["dfm_dtype_src"] = "auto", "default"

    # ---- ② 合成后端：env > 请求体 > 默认（= 现在的 CPU 路径）----
    env_g = _gpu_of(_ORIG.get("gpu"))
    want_g = _gpu_of(body.get("compose_gpu"))
    if env_g is not None:
        out["compose_backend"], out["compose_env"] = _label(env_g), True
        if want_g is not None and want_g != env_g:
            out["compose_backend_src"] = "default"
            out["warnings"].append(
                ["eff-env", "compose_backend",
                 "合成后端请求体写的是 %s，但进程里 %s=%d（环境变量优先）→ 实际跑 %s"
                 % (_label(want_g), GPU_ENV, env_g, _label(env_g))])
        elif want_g is not None:
            out["compose_backend_src"] = "ui"
        else:
            out["compose_backend_src"] = "default"
    elif want_g is not None:
        out["compose_backend"], out["compose_backend_src"] = _label(want_g), "ui"
        out["compose_gpu"] = want_g
    else:
        out["compose_backend"], out["compose_backend_src"] = "cpu", "default"
    return out


def _show_dtype(v: str) -> str:
    return "auto（自动）" if v == "auto" else str(v or "?")


def refine_compose(kn: dict, mode: str = "", fused: bool = True) -> dict:
    """按「这一单到底会不会上 GPU」再收一次合成后端（只动 gpu 这条）。

    和 pipeline_fused 里那份判定同一个顺序（先看分段链、再看 mode、最后看 torch/cuda），
    但**不碰 torch**：路由线程里 import torch 会把一条 POST 卡住好几秒。所以这里只做能
    确定的那一半，「torch/cuda 起不起来」留给流水线开跑时判，判完由它补一条 [eff-warn]
    （见 pipeline_fused._fused_run 里的 knobs 段）。

    fused=False（六阶段分段链）：那条路的合成只有 compose_frame（CPU 实现），
    选 gpu 也不会生效 —— 与其让 [eff] 说 gpu、实际跑 cpu，不如在这里就改回 cpu 并说明。
    """
    kn = dict(kn or {})
    why = ""
    if kn.get("compose_backend") == "gpu":
        if not fused:
            kn["compose_backend"] = "cpu"
            why = "分段链的合成只有 CPU 实现（GPU 合成只在快速导出里）"
        elif "seamless" in str(mode or "").strip().lower():
            # seamless 现在默认也走 GPU（compose_gpu.seamless_on 默认 True）；只有当用户显式
            # 设了 DFL_COMPOSE_GPU_SEAMLESS=0 时，才把这一单退回 CPU 并说明。
            try:
                from .compose_gpu import seamless_on as _seamless_on
            except Exception:                               # noqa: BLE001
                _seamless_on = None
            if _seamless_on is not None and not _seamless_on():
                kn["compose_backend"] = "cpu"
                why = "seamless 被 DFL_COMPOSE_GPU_SEAMLESS=0 显式关掉了 GPU 版"
    if why:
        kn["compose_why"] = why
        # 已经确定上不了 GPU 了：env 也写 0。不然流水线会白初始化一次 CUDA
        # （dfm 会话那条路本来就在用卡，多这一次只是噪音）。
        kn["compose_gpu"] = 0
        kn["warnings"] = list(kn.get("warnings") or []) + [
            ["eff-skipped", "compose_backend", skip_text("合成后端", "gpu", "cpu", why)]]
    return kn


# ---------------------------------------------------------------- 落地（写 env）
def apply(kn: dict) -> list:
    """把这一单的选择写进进程 env；**env 显式设过就一个字都不动**。

    返回需要 [eff-warn] 的文案（env 覆盖请求体那两条）。
    """
    kn = kn or {}
    notes = [str(w[2]) for w in (kn.get("warnings") or []) if len(w) > 2 and w[2]]

    if not kn.get("dfm_dtype_env"):
        src = str(kn.get("dfm_dtype_src") or "default")
        val = str(kn.get("dfm_dtype") or "auto")
        if src == "ui":
            _set_env(DFM_ENV, val)          # 界面钉死了档位
        elif src == "auto":
            _set_env(DFM_ENV, "auto")       # 显式选「自动」：后端按权重体量挑
        else:
            _set_env(DFM_ENV, None)         # 谁都没说：还原进程原值（不设 = 后端默认档）
    # 人脸超分：界面滑块值（0..100）。env 没被显式设过才由这一单决定。
    if not os.environ.get("DFL_SR_ENV_LOCKED"):
        lv = kn.get("sr")
        try:
            lv = int(float(lv)) if lv not in ("", None) else 0
        except (TypeError, ValueError):
            lv = 0
        lv = max(0, min(100, lv))
        if lv > 0:
            os.environ["DFL_SR"] = str(lv)
            os.environ.pop("DFL_SR_SIZE", None)      # 让滑块决定尺寸
        else:
            os.environ.pop("DFL_SR", None)
            os.environ.pop("DFL_SR_SIZE", None)
    if not kn.get("compose_env"):
        g = kn.get("compose_gpu")
        # "" = 请求体没说（自动）→ 还原进程原值（默认就是「关」，即 CPU 路径）
        _set_env(GPU_ENV, None if g in ("", None) else str(int(g)))
    return notes


def _sr_level(v) -> int:
    """请求体里的 sr（0..100，允许 "off/low/mid/high" 与字符串数字）→ int。"""
    if isinstance(v, bool) or v is None:
        return 0
    if isinstance(v, (int, float)):
        return max(0, min(100, int(round(float(v)))))
    txt = str(v).strip().lower()
    if txt in ("", "off", "none", "0"):
        return 0
    if txt == "low":
        return 18
    if txt == "mid":
        return 30
    if txt == "high":
        return 100
    try:
        return max(0, min(100, int(round(float(txt)))))
    except (TypeError, ValueError):
        return 0


def _set_env(name: str, value) -> None:
    """value=None 表示「还原成进程启动时的样子」（原来没设就删掉这个键）。"""
    if value is None:
        orig = _ORIG.get("dtype" if name == DFM_ENV else "gpu")
        if orig is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(orig)
        return
    os.environ[name] = str(value)


# ---------------------------------------------------------------- 回显 / 告警文案
def skip_text(field: str, want: str, got: str, why: str = "") -> str:
    """「选了 X、实际跑 Y、原因 Z」—— 三条都在，别只写一句「已回退」。

    field 用界面上的名字（合成后端 / 推理精度），want/got 用界面上的值（gpu / cpu）。
    """
    tail = ("，原因：%s" % why) if why else ""
    return ("%s：选了 %s、实际跑 %s%s（不是悄悄换的，这一条就是那个对不上的地方）"
            % (field, want or "?", got or "?", tail))


def warn_line(text: str) -> None:
    """打一行 [eff-warn]（桌面壳直接起 serve.py、stdout 不是 utf-8 时也不许吞掉）。

    和 eff._safe_print 同一套兜底；这里不 import eff（pipeline_fused ↔ eff 已经互相
    咬着了），复制六行比多一条环安全。
    """
    line = "[eff-warn] " + str(text)
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        try:
            import sys
            buf = getattr(sys.stdout, "buffer", None)
            if buf is not None:
                buf.write((line + "\n").encode("utf-8", "replace"))
                buf.flush()
        except Exception:                                   # noqa: BLE001
            pass
    except Exception:                                       # noqa: BLE001
        pass


def env_hint() -> str:
    """给人看的一行「这两个还能怎么改」—— 文档/日志里复用同一句。"""
    return ("%s=%s · %s=%s（env 显式设了优先于界面；全量清单见 app/docs/27-ENV-KNOBS.md）"
            % (DFM_ENV, os.environ.get(DFM_ENV) or "未设",
               GPU_ENV, os.environ.get(GPU_ENV) or "未设"))
