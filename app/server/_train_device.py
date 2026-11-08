"""训练「设备」选项（options["device"]）→ 子进程环境变量 / 命令行开关的唯一权威实现。

审计结论（为什么需要这个文件）：
    device 会被参数页写进 dat 的 options，但引擎和 _train_worker 里**没有任何地方读它**
    —— 用户在下拉框里选了 0 号卡，实际还是走 nn.DeviceConfig.BestGPU()（显存最大的那张），
    而且不报错、不提示。这里把「读 → 校验 → 映射」收成一组纯函数，启动路径和
    /api/train/status 共用同一份结果，避免两处各写一遍再对不上。

值的**真实类型**（不是猜的，是前端 <select> 的实际产物，见 app/web/js/app.js
buildDeviceOpts()：op.value = String(ov)）—— 一律是**字符串**：

    "auto"        自动选择（dat 里通常根本没有这个键）
    "cpu"         CPU
    "0" / "1"…    单卡；序号是 nvidia-smi 的**物理**序号
    "0,1"         多卡，逗号连接（只有本机 >1 张卡时前端才给出这一项）

人工改 dat / 老 Qt 界面还会留下别的写法，这里一并收敛：int 0、list [0, 1]、
-1（老约定里当 CPU 用）。

映射规则（「在映射后保持逻辑序号正确」）：
    CUDA_VISIBLE_DEVICES=0,2 之后，子进程里只看得见 2 张卡，编号变成 0 和 1。
    所以 --force-gpu-idxs 必须传**映射后的逻辑序号** 0,1，不能传物理序号 0,2 ——
    传 0,2 时 nn.DeviceConfig.GPUIndexes([0,2]) 只能取到逻辑 0 号那张卡，
    用户要的两卡训练就悄悄退化成单卡（又是一个「选了不生效」）。
"""
from __future__ import annotations

__all__ = ["DeviceOptionError", "plan", "gpus_brief", "raw_text",
           "AUTO", "CPU", "GPUS"]

AUTO = "auto"      # 不设 env、不加开关：和以前完全一样，由引擎自己挑
CPU = "cpu"        # --cpu-only
GPUS = "gpus"      # CUDA_VISIBLE_DEVICES + --force-gpu-idxs

# 老约定里 -1 表示 CPU（legacy/qt-ui 的 device 取值域里就有它）
_CPU_TOKENS = {"cpu", "c", "-1"}
_AUTO_TOKENS = {"", "auto", "automatic", "自动", "自动选择", "none", "null", "default"}


class DeviceOptionError(ValueError):
    """device 取值非法：启动前就该拒绝，不许静默回退到默认卡。

    message 是给用户看的完整句子（里面一定带可用设备列表），
    raw / available 留着给 /api/train/status 复用，前端不必再解析文本。
    """

    def __init__(self, message: str, raw=None, available=None):
        super().__init__(message)
        self.message = message
        self.raw = raw
        self.available = available or []

    def __str__(self):
        return self.message


def gpus_brief(gpus) -> str:
    """把 GPU 列表拼成一句话：「0:RTX 5070 Laptop GPU(8.0GB), 1:RTX 4090(24.0GB)」。"""
    out = []
    for g in (gpus or []):
        try:
            idx = int(g.get("index", g.get("idx")))
        except Exception:
            continue
        name = str(g.get("name") or "GPU").strip()
        mem = g.get("total_gb")
        if mem is None and g.get("mem_mb"):
            mem = round(float(g["mem_mb"]) / 1024.0, 1)
        out.append("%d:%s%s" % (idx, name, ("(%.1fGB)" % float(mem)) if mem else ""))
    return ", ".join(out)


def _index_set(gpus):
    ids = set()
    for g in (gpus or []):
        try:
            ids.add(int(g.get("index", g.get("idx"))))
        except Exception:
            continue
    return ids


def _coerce_list(raw):
    """把各种写法摊平成 (kind, indexes)；kind ∈ {AUTO, CPU, GPUS}。

    认不出来就抛 DeviceOptionError —— 不认识的值等于没设置，静默当 auto 是这个
    bug 的另一半（用户以为选了卡，其实什么都没发生）。
    """
    if raw is None:
        return AUTO, []
    if isinstance(raw, bool):
        # True/False 写进 device 只可能是手滑，但别把 False 当 0 号卡
        raise DeviceOptionError(
            "device=%r 不是有效的设备值：可用值 auto / cpu / 0…" % (raw,),
            raw=raw)
    if isinstance(raw, (list, tuple, set)):
        items = list(raw)
        if not items:
            return AUTO, []
    else:
        items = [raw]

    out = []
    for it in items:
        if isinstance(it, str):
            s = it.strip().lower()
            if s in _AUTO_TOKENS:
                continue
            if s in _CPU_TOKENS:
                return CPU, []
            # "0,1" / "0 1" / "0;1" / "0/1" 都当分隔符
            parts = [p for p in s.replace(";", ",").replace(" ", ",")
                     .replace("/", ",").split(",") if p]
            if len(parts) > 1:
                for p in parts:
                    pl = p.strip().lower()
                    if pl in _CPU_TOKENS:
                        return CPU, []
                    if pl in _AUTO_TOKENS:
                        continue
                    try:
                        out.append(int(float(pl)))
                    except Exception:
                        raise DeviceOptionError(
                            "device=%r 里有认不出的 GPU 序号 %r：可用值 auto / cpu / 0,1,…" % (raw, p),
                            raw=raw)
                continue
            try:
                out.append(int(float(s)))
            except Exception:
                raise DeviceOptionError(
                    "device=%r 认不出来：可用值 auto / cpu / 0…（GPU 序号，逗号分隔可多卡）" % (raw,),
                    raw=raw)
        elif isinstance(it, bool):
            raise DeviceOptionError("device=%r 里有布尔值，不是 GPU 序号" % (raw,), raw=raw)
        elif isinstance(it, (int, float)):
            if int(it) < 0:
                return CPU, []          # -1 / 负数：老约定 = CPU
            out.append(int(it))
        else:
            raise DeviceOptionError(
                "device=%r 的类型是 %s，认不出来：可用值 auto / cpu / 0…"
                % (raw, type(raw).__name__), raw=raw)

    if not out:
        return AUTO, []

    # 去重但保序：CUDA_VISIBLE_DEVICES="0,0" 在真机上会让 CUDA 一张卡都看不到
    # （实测 torch.cuda.device_count() == 0），所以重复序号必须先收掉。
    seen, dedup = set(), []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        dedup.append(i)
    return GPUS, dedup


def raw_text(raw) -> str:
    """原始值的可读形式：None 显示成「未设置」，和「用户选了 auto」区分开。"""
    if raw is None:
        return "未设置"
    if isinstance(raw, (list, tuple)):
        return ",".join(str(x) for x in raw)
    if isinstance(raw, bool):
        return str(raw).lower()
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    s = str(raw).strip()
    return s if s else "未设置"


def plan(raw, gpus, engine_cpu_only=True):
    """把 options["device"] 解析成启动子进程要用的具体动作。

    raw             : dat 里 options 的原始值（缺失 → 传 None）
    gpus            : app._gpu_list() 的结果（nvidia-smi 探到的物理卡）
    engine_cpu_only : 引擎的 train 子命令是否支持 --cpu-only（不支持要明确告警）

    返回 dict（可直接 JSON 回显）。取值非法时抛 DeviceOptionError，调用方
    （_train_worker / /api/train/start）负责**在起进程之前**拒绝。
    """
    gpus = list(gpus or [])
    avail = _index_set(gpus)
    kind, idxs = _coerce_list(raw)

    p = {"raw": raw_text(raw), "mode": kind, "physical": [], "logical": [],
         "cuda_visible": None, "force_gpu_idxs": None, "cpu_only": False,
         "banner": "", "warn": None, "available": gpus}

    if kind == AUTO:
        p["banner"] = ("[train] device=%s → CUDA_VISIBLE_DEVICES=(不设置，引擎自动选择)"
                       % p["raw"])
        return p

    if kind == CPU:
        p["cpu_only"] = True
        p["banner"] = ("[train] device=%s → CUDA_VISIBLE_DEVICES=(不设置) · --cpu-only"
                       % p["raw"])
        if not engine_cpu_only:
            p["warn"] = ("device=%s 想跑 CPU，但这个引擎的 train 子命令没有 --cpu-only，"
                         "实际仍可能用到 GPU" % p["raw"])
        return p

    # ---- GPUS：先校验，再映射 ----
    if not gpus:
        raise DeviceOptionError(
            "device=%s 指向 GPU，但这台机器探测不到可用 GPU（nvidia-smi 没返回任何卡）。"
            "可用值：auto / cpu。" % p["raw"], raw=raw, available=[])
    bad = [i for i in idxs if i not in avail]
    if bad:
        raise DeviceOptionError(
            "device=%s 里有本机不存在的 GPU 序号 %s。可用 GPU：%s（可用值：auto / cpu / %s）"
            % (p["raw"], ",".join(str(i) for i in bad), gpus_brief(gpus),
               ",".join(str(i) for i in sorted(avail))),
            raw=raw, available=gpus)

    p["physical"] = idxs
    # 关键：env 里放物理序号，--force-gpu-idxs 放映射后的**逻辑**序号 0..n-1
    p["logical"] = list(range(len(idxs)))
    p["cuda_visible"] = ",".join(str(i) for i in idxs)
    p["force_gpu_idxs"] = ",".join(str(i) for i in p["logical"])
    p["banner"] = "[train] device=%s → CUDA_VISIBLE_DEVICES=%s" % (p["raw"], p["cuda_visible"])
    if idxs != p["logical"]:
        p["banner"] += "（子进程内逻辑序号 %s · --force-gpu-idxs %s）" % (
            p["force_gpu_idxs"], p["force_gpu_idxs"])
    return p
