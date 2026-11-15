# -*- coding: utf-8 -*-
"""合成任务编排：settings 翻译 / DFM 解析 / 起 _merge_runner.py 子进程（从 app/server/app.py 切出）。

app.py 里这一串 _merge_* 是「合成」这一步的**运行控制**（谁跑、跑哪条命令、日志进哪），
合成核本身在 backend_merge/ 里。原样搬到公共层，函数体一行没改 —— 等 backend_merge 那边
腾出手再整体收编。

Path(__file__).with_name("_merge_runner.py") 原来指 app/server/，本文件在 app/server/_common/，
改成 Path(__file__).resolve().parents[1] / "_merge_runner.py"。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .bootstrap import _NO_WINDOW
from .jobs import _merge_cache, _merge_prep, _merge_prep_lock
from .maskcfg import _mask_mode_int
from .paths import WORKSPACE, _APP_ROOT, _DFL_ROOT, engine_python as _engine_python


# ============================================================
# 合成导出：调 4.0 的 main.py merge --pipeline（ONNX DFM，不需要 torch 权重）
# ============================================================
_merge_job = {"running": False, "log": [], "ok": None, "error": None, "out": "",
              "pct": 0, "msg": "",
              # 生效配置（见 server/_eff.py）：导出启动时写入，/export/status 回给界面
              "eff": None, "eff_warnings": [], "warnings": []}

def _merge_settings_to_config(settings: dict) -> dict:
    """把 4.1 界面的合成参数翻成 MergeStudio 的键名。

    两代键名都认（**新的优先**）：
        blend       = overlay / seamless / …      <-> mode
        ct_mode     = mt / rct / lct / mkl / sot  <-> color_transfer_mode
        mask_erode  = -100~100                    <-> erode_mask_modifier
        mask_blur   = -100~100                    <-> blur_mask_modifier

    默认值：ct_mode = "mt"、blend = "seamless"（和 /api/merge/frame、/merged、
    /preview 缺省 settings 时的结果**逐字节一致**，两边共用这一份默认值）。
    """
    settings = settings or {}

    def _pick(*names, default=None):
        for n in names:
            v = settings.get(n)
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            return v
        return default

    cfg = {
        "mode": _pick("blend", "mode", default="seamless"),
        "mask_mode": _mask_mode_int(settings.get("mask_mode"), 6),
        "color_transfer_mode": _pick("ct_mode", "color_transfer_mode",
                                     "color_transfer", default="mt"),
        "erode_mask_modifier": _pick("mask_erode", "erode_mask_modifier",
                                     "erode_mask", default=0),
        "blur_mask_modifier": _pick("mask_blur", "blur_mask_modifier",
                                    "blur_mask", default=0),
    }
    # 4.1 的键名有两代：界面现在发 erode_mask / blur_mask，别处会发 mask_erode /
    # mask_feather。两条都认，并且用别名给的值覆盖归一键 —— routes_merge 那边的
    # _settings_val 会先补一个默认 0 的 erode_mask_modifier，只转发别名的话，别名
    # 给的数会被那个 0 盖掉，等于没传。
    for _alias, _key in (("erode_mask", "erode_mask_modifier"),
                         ("mask_erode", "erode_mask_modifier"),
                         ("blur_mask", "blur_mask_modifier"),
                         ("mask_feather", "blur_mask_modifier")):
        _v = settings.get(_alias)
        if _v is None or (isinstance(_v, str) and not _v.strip()):
            continue
        cfg[_key] = _v
        if _alias in ("mask_erode", "mask_feather"):
            cfg[_alias] = _v            # 原样带上，_merge_runner 自己也归一这两个名字
    # face_type：4.1 的界面有这一项（mgFaceType），以前到这儿就丢了，
    # 导出日志里根本看不到它。
    _ft = settings.get("face_type")
    if _ft is not None and str(_ft).strip():
        cfg["face_type"] = _ft
    if settings.get("face_margin") is not None:
        cfg["face_margin"] = settings["face_margin"]
    return cfg


def _merge_source_video(input_dir: str = "") -> str:
    """这批帧的源视频（导出时取音轨用）；对不上就返回 ""，绝不瞎猜路径。

    两个来源，都要能证明「就是这批帧」才用：
      · /api/merge/open 打开视频 → _merge_cache["path"]（项目模式下被清成 ""）
      · /api/merge/prepare 抽帧 → _merge_prep["video"]，配 _merge_prep["project"]
    """
    exts = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv")

    def _as_video(p) -> str:
        text = str(p or "").strip()
        if not text:
            return ""
        try:
            pp = Path(text)
        except (OSError, ValueError):
            return ""
        if pp.suffix.lower() not in exts or not pp.is_file():
            return ""
        return str(pp)

    def _stem_hit(video: str) -> bool:
        """帧目录名和视频名对得上：prepare 的产物就是 <workspace>/merge/<视频名>/。"""
        name = Path(str(input_dir).strip()).name if str(input_dir or "").strip() else ""
        stem = Path(video).stem
        if not name or not stem:
            return False
        return name == stem or name.startswith(stem + "_")

    try:
        cache = dict(_merge_cache)
    except Exception:                                     # noqa: BLE001
        cache = {}

    hit = _as_video(cache.get("path"))        # 视频模式：path 就是原片
    if hit and _stem_hit(hit):
        return hit

    with _merge_prep_lock:
        prep = dict(_merge_prep)
    hit = _as_video(prep.get("video"))        # prepare 的项目：源视频只记在这儿
    if not hit:
        return ""
    proj = str(prep.get("project") or "").strip()
    mine = str(input_dir or "").strip() or str(cache.get("dir") or "").strip()
    if proj and mine:
        try:
            if os.path.normcase(os.path.abspath(proj)) == os.path.normcase(os.path.abspath(mine)):
                return hit
        except (OSError, ValueError):
            pass
    return hit if _stem_hit(hit) else ""


# ---------- 多模型导出（models 清单 → --settings <临时 JSON>） ----------
# 契约在 _merge_runner.py 那边：--settings 吃 {"models":[{"name","dfm","faces"}]}。
# 这边只干两件事：把 4.1 的 models 列表翻成那份 JSON，决定命令行给 --dfm 还是 --settings。


def _merge_log(msg: str):
    """往合成任务日志里追一行（前端日志面板读的就是 _merge_job["log"]）。"""
    try:
        with _merge_job_lock:
            log = _merge_job.get("log")
            if not isinstance(log, list):
                log = []
                _merge_job["log"] = log
            log.append(str(msg))
            if len(log) > 400:
                del log[:-400]
    except Exception:                                   # noqa: BLE001
        pass


def _merge_models_raw(models, settings):
    """models 的原始值：优先 models 实参，其次 settings["models"]。

    routes_merge 有两条接法（单独一个 models 字段 / 塞进 settings 一起给），两种都认。
    只认一种的话另一边会静默退回单模型 —— 用错模型出片，比直接失败更难查。
    """
    raw = models
    if raw is None and isinstance(settings, dict):
        raw = settings.get("models")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            raw = json.loads(text)
        except Exception:
            raise RuntimeError("多模型清单不是合法 JSON：%s" % text[:120])
    if isinstance(raw, dict):
        raw = raw.get("models") if "models" in raw else [raw]
    if not isinstance(raw, (list, tuple)):
        return None
    out = []
    for e in raw:
        if e is None or (isinstance(e, str) and not e.strip()):
            continue
        out.append(dict(e) if isinstance(e, dict) else {"dfm": e})
    return out


def _merge_faces_of(who: str, raw) -> list:
    """faces → 升序去重的 int 列表；空列表 = 这个模型兜底吃没被点名的脸。"""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return []
    if isinstance(raw, bool):
        raise RuntimeError("%s 的 faces 是布尔值，看不懂: %r" % (who, raw))
    if isinstance(raw, (int, str)):
        raw = [raw]
    if isinstance(raw, (set, frozenset)):
        raw = sorted(raw)
    if not isinstance(raw, (list, tuple)):
        raise RuntimeError("%s 的 faces 要是一个数组（人脸序号），收到 %s"
                           % (who, type(raw).__name__))
    out = set()
    for f in raw:
        if isinstance(f, bool):
            raise RuntimeError("%s 的 faces 里有布尔值: %r" % (who, f))
        if isinstance(f, int):
            n = f
        else:
            try:
                n = int(str(f).strip())
            except (TypeError, ValueError):
                raise RuntimeError("%s 的 faces 里有非整数: %r" % (who, f))
        if n < 0:
            raise RuntimeError("%s 的 faces 里有负数（脸序号从 0 起）: %d" % (who, n))
        out.add(int(n))
    return sorted(out)


def _merge_dfm_dir() -> Path:
    """按名字找 .dfm 的目录：workspace/model（训练页点「导出」就写这儿）。"""
    return Path(WORKSPACE) / "model"


def _merge_dfm_by_name(name: str, model_dir: Path) -> Path:
    """只给了名字 → <名字>.dfm，其次 <名字>-<迭代>.dfm（取迭代最高的那个）。

    找不到就报错。以前这种地方会悄悄退回"第一个 .dfm"，结果拿别的模型出片、
    谁都没发现 —— 宁可失败，也不能静默换模型。
    """
    hits = []
    try:
        for p in model_dir.iterdir():
            if not p.is_file() or p.suffix.lower() != ".dfm":
                continue
            if p.stem == name:
                return p
            if p.stem.startswith(name + "-"):
                hits.append(p)
    except OSError:
        hits = []
    if not hits:
        raise RuntimeError("模型「%s」没有 .dfm：%s 下既没有 %s.dfm 也没有 %s-*.dfm —— "
                           "先去训练页点「导出」" % (name, model_dir, name, name))

    def _iter(p):
        try:
            return int(p.stem.rsplit("-", 1)[-1])
        except (TypeError, ValueError):
            return -1

    hits.sort(key=_iter)
    return hits[-1]


def _merge_abs_dfm(raw_dfm, who: str) -> Path:
    """给出的 dfm 路径 → 存在的路径（相对路径按仓库根 / 模型目录补全）。"""
    p = Path(str(raw_dfm).strip())
    if not p.is_absolute():
        for base in (_APP_ROOT, Path(WORKSPACE), _merge_dfm_dir(), Path.cwd()):
            try:
                c = Path(base) / p
            except (OSError, ValueError):
                continue
            if c.is_file():
                p = c
                break
    if not p.is_file():
        raise RuntimeError("%s 的 dfm 不存在：%s" % (who, p))
    return p


def _merge_model_plan(models, settings, dfm: str = ""):
    """models 清单 → [{"name","dfm","faces"}]（dfm 已解析成绝对路径）；None = 单模型。

    传了 dfm 就只认那条路径；只给 name 才去 workspace/model 找 <name>.dfm /
    <name>-*.dfm。同一个 .dfm 出现两次 → 合成一项、faces 取并集。
    """
    raw = _merge_models_raw(models, settings)
    if not raw:
        return None
    model_dir = _merge_dfm_dir()
    plan, seen = [], {}
    for i, e in enumerate(raw):
        name = str(e.get("name") or "").strip()
        raw_dfm = ""
        for k in ("dfm", "dfm_path", "dfmPath", "path", "file", "dfm_file"):
            v = e.get(k)
            if v is not None and str(v).strip():
                raw_dfm = str(v).strip()
                break
        who = ("models[%d]「%s」" % (i, name)) if name else ("models[%d]" % i)
        if raw_dfm:
            path = _merge_abs_dfm(raw_dfm, who)
        elif name:
            path = _merge_dfm_by_name(name, model_dir)
        else:
            raise RuntimeError("%s 既没有 name 也没有 dfm 路径 —— 不知道要用哪个模型" % who)
        faces = _merge_faces_of(who, e.get("faces", e.get("face", e.get("indices"))))
        try:
            key = os.path.normcase(str(path.resolve()))
        except OSError:
            key = os.path.normcase(str(path))
        hit = seen.get(key)
        if hit is not None:
            hit["faces"] = sorted(set(hit["faces"]) | set(faces))
            continue
        ent = {"name": name or path.stem, "dfm": str(path), "faces": faces}
        seen[key] = ent
        plan.append(ent)
    return plan


def _merge_plan_label(plan) -> str:
    """「张元英[face 0], 韩素希[face 1]」—— 日志里点清谁换了哪张脸。"""
    parts = []
    for m in plan or []:
        faces = m.get("faces") or []
        if not faces:
            tag = "全部脸"
        elif len(faces) == 1:
            tag = "face %d" % int(faces[0])
        else:
            tag = "face " + ",".join(str(int(f)) for f in faces)
        parts.append("%s[%s]" % (m.get("name") or Path(m.get("dfm") or "").stem, tag))
    return ", ".join(parts)


def _merge_single_label(dfm, settings) -> str:
    """单模型日志里的名字：settings 里点过名就用名字，否则用 .dfm 的文件名。"""
    if isinstance(settings, dict):
        for k in ("model", "model_name", "dfm_name"):
            v = str(settings.get(k) or "").strip()
            if v:
                return v
    return Path(str(dfm or "")).stem or "（未命名模型）"


def _merge_write_models(plan) -> str:
    """整份 {"models":[...]} 落到系统 temp → 返回路径（调用方跑完负责删）。"""
    import tempfile
    payload = {"models": [{"name": str(m["name"]), "dfm": str(m["dfm"]),
                           "faces": [int(f) for f in (m.get("faces") or [])]}
                          for m in plan]}
    fd, path = tempfile.mkstemp(prefix="dfl_merge_models_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def _merge_models_args(plan, dfm):
    """→ (命令行片段, 临时清单路径)。

    多模型：["--settings", <temp json>] —— 整份清单走文件，命令行上不用塞中文 JSON；
    单模型：["--dfm", dfm]，和改动前逐字一致。
    """
    if plan:
        path = _merge_write_models(plan)
        return ["--settings", path], path
    return ["--dfm", dfm], ""


def _merge_worker(input_dir: str, out_mp4: str, aligned_dir: str, dfm: str,
                  settings: dict, mask_onnx: str, fps: float = 25.0, workers: int = 0,
                  encoder: str = "h264_nvenc", hwaccel: str = "auto", models=None,
                  detector: str = ""):
    """跑 _merge_runner.py（4.0 的 python + MergeStudio 导出管线）。

    以前这里是 "main.py merge"，而 mainscripts/Merger.py 依赖一个不存在的
    merger 包，跑到一半必崩；现在换成直接调 run_export_pipeline。

    models（可选）= [{name, dfm, faces}]：多模型导出。给了就整份写成系统 temp 里的
    JSON，命令行用 runner 的 --settings <文件>；没给就只有一次 --dfm，命令行与线上
    逐字一致。单项有 dfm 就只认那条路径，只有 name 时按 workspace/model 找
    <name>.dfm / <name>-*.dfm，找不到直接报错（绝不静默退回单模型）。
    """
    import subprocess
    py = _engine_python()
    runner = Path(__file__).resolve().parents[1] / "_merge_runner.py"
    if not py.is_file():
        raise RuntimeError("找不到 4.0 的 python: %s" % py)
    if not runner.is_file():
        raise RuntimeError("找不到合成桥脚本: %s" % runner)
    if not (_DFL_ROOT / "MergeStudio").is_dir():
        raise RuntimeError("4.0 根目录下没有 MergeStudio: %s" % _DFL_ROOT)

    # 多模型清单：先解析 —— 少一个 dfm 要在起进程之前就炸，
    # 别等跑到合成阶段才发现某张脸没换。
    try:
        plan = _merge_model_plan(models, settings, dfm)
    except Exception as e:                              # noqa: BLE001
        _merge_log("多模型导出失败: %s" % (str(e)[:300] or e.__class__.__name__))
        raise

    mode_word = "多模型导出" if plan else "单模型导出"
    mode_detail = _merge_plan_label(plan) if plan else _merge_single_label(dfm, settings)

    # 检测器：4.1 规范名 → 4.0 引擎名（转名只在 _common/det_names 一张表里做）。
    # 以前这里**一个字节都不传**，4.0 引擎就按它自己的默认 YOLOv8（最重那个）跑，
    # 而生效配置里硬写 tinymog —— 界面/日志说一套、实跑另一套。
    # 空 = 调用方没给（老调用点）：按 4.1 默认档显式传下去，别让引擎自己挑。
    from . import det_names as _det_names
    try:
        det_engine = _det_names.engine(detector, _det_names.DEFAULT)
    except ValueError as e:                       # 未知名字：起进程之前就炸
        raise RuntimeError(str(e))
    cmd_tail = ["--out", out_mp4,
                "--detector", str(det_engine),
                "--config", json.dumps(_merge_settings_to_config(settings), ensure_ascii=False),
                "--dfl-root", str(_DFL_ROOT),
                "--fps", str(float(fps or 25.0))]
    # 音轨：_merge_runner 会把 --source-video 的音轨合进成片。拿不到源视频就不传，
    # 让它在输出/帧目录旁边按 <名字>.mp4 自己找，找不到就静音出片。
    src_video = _merge_source_video(input_dir)
    if src_video:
        cmd_tail += ["--source-video", src_video]
    if mask_onnx:
        cmd_tail += ["--xseg", mask_onnx]
    if workers and int(workers) > 0:
        cmd_tail += ["--workers", str(int(workers))]
    cmd_tail += ["--encoder", str(encoder or "h264_nvenc")]
    cmd_tail += ["--image-format", str((settings or {}).get("image_format") or "jpg")]
    cmd_tail += ["--hwaccel", str(hwaccel if hwaccel is not None else "auto")]
    # 4.0 根下没有 ffmpeg/，用 4.1 自带的 imageio-ffmpeg（只读引用）
    try:
        import imageio_ffmpeg as _iff
        _ff = _iff.get_ffmpeg_exe()
        if _ff and Path(_ff).is_file():
            cmd_tail += ["--ffmpeg", _ff]
    except Exception:
        pass

    # 清单最后才落盘：上面任何一步抛异常都不该在 temp 里留文件
    model_args, settings_file = _merge_models_args(plan, dfm)
    cmd = [str(py), str(runner),
           "--frames", input_dir,
           "--aligned", aligned_dir] + model_args + cmd_tail

    with _merge_job_lock:
        _merge_job.update({"running": True, "log": [], "ok": None, "error": None,
                           "out": out_mp4, "pct": 0, "msg": "启动合成…"})
        _merge_job["log"].append("%s: %s" % (mode_word, mode_detail))
        if settings_file:
            _merge_job["log"].append("模型清单（--settings）: %s" % settings_file)
        if src_video:
            _merge_job["log"].append("源视频（音轨来源）: %s" % src_video)
        _merge_job["log"].append(
            "检测器: %s（4.1 名 %s · 4.0 引擎名 %s）"
            % (det_engine, _det_names.canon(detector, _det_names.DEFAULT), det_engine))
    try:
        env_note = "PYTHONIOENCODING=utf-8"
        import os as _os
        env = dict(_os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        global _merge_proc
        proc = subprocess.Popen(cmd, cwd=str(_DFL_ROOT), env=env, creationflags=_NO_WINDOW,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        _merge_proc = proc
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("@@P "):
                parts = line.split(" ", 3)
                try:
                    pct = int(float(parts[2]) * 100)
                except Exception:
                    pct = None
                try:
                    stage = int(parts[1])
                except Exception:
                    stage = -1
                with _merge_job_lock:
                    if pct is not None:
                        _merge_job["pct"] = max(0, min(100, pct))
                    if stage >= 0:
                        _merge_job["stage"] = stage
                    if len(parts) > 3:
                        _merge_job["msg"] = parts[3][:120]
                continue
            if line.startswith("@@DONE "):
                with _merge_job_lock:
                    _merge_job["out"] = line[7:].strip()
                continue
            if line.startswith("@@ERR "):
                with _merge_job_lock:
                    _merge_job["error"] = line[6:].strip()
                continue
            with _merge_job_lock:
                _merge_job["log"].append(line)
                if len(_merge_job["log"]) > 400:
                    _merge_job["log"] = _merge_job["log"][-400:]
        proc.wait()
        ok = proc.returncode == 0
        with _merge_job_lock:
            _merge_job["running"] = False
            _merge_job["ok"] = ok
            _merge_job["log"].append("%s%s: %s"
                                     % (mode_word,
                                        "完成" if ok else ("失败（退出码 %s）"
                                                           % proc.returncode),
                                        mode_detail))
            if ok:
                _merge_job["pct"] = 100
                _merge_job["msg"] = "合成完成"
            elif not _merge_job.get("error"):
                _merge_job["error"] = "合成进程退出码 %d" % proc.returncode
    except Exception as e:
        with _merge_job_lock:
            _merge_job["running"] = False
            _merge_job["ok"] = False
            _merge_job["error"] = str(e)
        _merge_log("%s失败: %s" % (mode_word, str(e)[:300]))
    finally:
        # 清单只给子进程用一次：跑完（成功/失败/取消）就把 temp 里的 JSON 删掉
        if settings_file:
            try:
                os.unlink(settings_file)
            except OSError:
                pass
import threading as _th2
_merge_job_lock = _th2.Lock()
_merge_proc = None          # 当前合成子进程，取消时用
