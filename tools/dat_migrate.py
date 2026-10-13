#!/usr/bin/env python3
"""dat_migrate.py — 老 DFL(TF) data.dat ↔ 深变(torch) 兼容体检与迁移工具。

背景（2026-10-12，Anna 实测两坑后沉淀）：
  convert_dfl_tf_to_torch.py 对 data.dat 是「Copy as-is」（第 987 行），
  权重转换有人管、options 清洗没人管。TF dat 直接给 torch 版用会踩：

  ① ct_mode=fs-aug        → torch 版值域白名单 {none,rct,mt,lct,mkl,idt,sot,lut}，
                             训练线程 ValueError: unknown ct_mode 直接崩（Anna 实踩）。
  ② 死键 10 个             → torch fork 零引用，静默躺在 dat 里误导排查：
                             auto_lr / mean_src_loss / mean_dst_loss / save_time /
                             retraining_samples / sample_ratio / lr_d_ratio /
                             gan_smoothing / preview_samples / force_full_preview
  ③ eyes_prio / mouth_prio → torch 版改用合并键 eyes_mouth_prio；
                             eyes_prio 会被 torch 主动 pop（无害但脏），mouth_prio 纯死键。
  ④ 新参数缺席             → lr_plateau_factor/patience/min_ratio、lr_total_steps、
                             random_color_alg、edge_enhance_power、preview_grade_alg、
                             eyes_mouth_prio、random_shadow 等 torch 新键。
                             --silent-start 不问问答，缺席即用内置默认
                             （plateau 默认关！要开必须种进去）。
  ⑤ random_flip            → 老键；torch 拆成 random_src_flip + random_dst_flip。
  ⑥ 优化器 lr 覆盖声明值    → torch 恢复训练时优化器状态里的 lr 优先于 options.lr
                             声明（Anna：10-11 热改 5e-05 未落盘，10-12 复跑实证）。
                             接手模型先 trainctl status 对 lr。

用法：
  python tools\\dat_migrate.py --doctor  <dat路径|模型目录>        # 单个体检
  python tools\\dat_migrate.py --scan    <模型目录>               # 目录批量体检
  python tools\\dat_migrate.py --fix     <dat路径|模型目录>       # 迁移演练（不落盘）
  python tools\\dat_migrate.py --fix ... --apply                  # 真正落盘（自动 .bak 备份）

  --model NAME   目录下含多个 dat 时指定模型名（默认取目录里唯一的那个）
  --plateau 0.3  fix 时种入 lr_plateau_factor（默认 0.3，用户习惯：永远开）
  --no-plateau   fix 时不种 plateau
  --lr X         fix 时顺带校准 lr 声明值（可选）

返回码：0=干净  1=有问题(--doctor/--scan)  2=参数/文件错误
fix 未 --apply 永不改盘。
"""
from __future__ import annotations

import argparse
import copy
import pickle
import shutil
import sys
from pathlib import Path

# ---------- 兼容规则表（唯一事实源，改这里） ----------

# TF 死键：torch fork 零引用，直接删除
DEAD_KEYS = [
    "auto_lr", "mean_src_loss", "mean_dst_loss", "save_time",
    "retraining_samples", "sample_ratio", "lr_d_ratio", "gan_smoothing",
    "preview_samples", "force_full_preview", "mouth_prio",
    # 2026-10-13 补：流通丹 dat 里的卖家信息/会话杂键（杨幂416 实见）
    "QQ", "专业模型制作", "模型制作", "session_name",
]

# 主动清除：torch 版代码里会 pop 的遗留键（eyes_prio 见 Model_pytorch.py:678）
POP_KEYS = ["eyes_prio"]

# 值域迁移：老值 → 新值
CT_MODE_MAP = {
    "fs-aug": "rct",   # fs-aug 仅老 TF 认识；torch 白名单外，起训即崩。Anna 修例用 rct
}
CT_MODE_ALLOWED = {"none", "rct", "mt", "lct", "mkl", "idt", "sot", "lut"}
LR_DROPOUT_MAP = {True: "y", False: "n"}

# 键改名：老 → 新（值语义等价或需拆分的单独处理）
RENAME_KEYS = {
    "random_flip": "random_src_flip",  # 老总开关 → src 侧；dst 侧按 torch 习惯补 True
}

# torch 新参数：缺席时种入（None=不主动种，保持 torch 默认；给出值=种）
# 注意：plateau 默认关是 torch 的「不静默替用户开启」设计；用户习惯是开，由 --plateau 控制
DEFAULTS_TORCH = {
    "lr_total_steps": 0,             # 0=单调退火关闭
    "lr_cos": 0,                     # 0=热重启关闭（与 plateau 互补，二选一）
    "random_color_alg": "none",
    "edge_enhance_power": 0.0,
    "preview_grade_alg": "same",
    "random_shadow": "none",
    "random_blur": False,
    "random_jpeg": False,
    "random_downsample": False,
}


def load_dat(path: Path):
    return pickle.loads(path.read_bytes())


def doctor(options: dict) -> list:
    """返回问题清单 [(级别, 键, 说明)]。级别: ERROR/WARN/INFO"""
    issues = []
    # ct_mode
    ct = options.get("ct_mode")
    if ct is None:
        issues.append(("WARN", "ct_mode", "缺席，torch 默认 rct"))
    elif ct not in CT_MODE_ALLOWED:
        new = CT_MODE_MAP.get(ct, "rct")
        issues.append(("ERROR", "ct_mode", f"{ct!r} 不在 torch 白名单，起训即崩 → 应改 {new}"))
    # 死键
    for k in DEAD_KEYS:
        if k in options:
            issues.append(("WARN", k, "torch fork 零引用（死键）→ 删"))
    for k in POP_KEYS:
        if k in options:
            issues.append(("INFO", k, "torch 加载时主动 pop → 可顺手清"))
    # lr_dropout 归一
    lrd = options.get("lr_dropout")
    if lrd is not None and lrd not in ("y", "n", "cpu"):
        issues.append(("ERROR", "lr_dropout", f"值 {lrd!r} 非法（须 y/n/cpu）"))
    # random_flip 未拆分
    if "random_flip" in options and "random_src_flip" not in options:
        issues.append(("WARN", "random_flip", "老总开关，torch 已拆 src/dst → 迁移"))
    # 新参数缺席（只报关键的，其余静默）
    if "lr_plateau_factor" not in options:
        issues.append(("INFO", "lr_plateau_factor", "缺席=平台期自动降 lr 关闭（--silent-start 不会问）"))
    if "lr_total_steps" not in options and "lr_cos" not in options:
        issues.append(("INFO", "lr_cos/lr_total_steps", "两个 lr 策略键都缺席，默认全关"))
    if "eyes_mouth_prio" not in options and ("eyes_prio" in options or "mouth_prio" in options):
        issues.append(("WARN", "eyes_mouth_prio", "老 eyes_prio/mouth_prio 在，合并键缺席 → 迁移"))
    # 架构键完整性（缺了会用默认值悄悄变架构，必须齐）
    for k in ("resolution", "archi", "ae_dims"):
        if k not in options:
            issues.append(("ERROR", k, "架构键缺席！torch 会用默认值悄悄改架构"))
    return issues


def migrate(options: dict, plateau: float | None, lr_override=None) -> tuple[dict, list]:
    """返回 (新options, 动作日志)。不改输入。"""
    o = copy.deepcopy(options)
    log = []
    # 1) ct_mode
    ct = o.get("ct_mode")
    if ct is not None and ct not in CT_MODE_ALLOWED:
        new = CT_MODE_MAP.get(ct, "rct")
        log.append(f"ct_mode: {ct!r} → {new!r}")
        o["ct_mode"] = new
    # 2) 死键/主动清除键
    for k in DEAD_KEYS:
        if k in o:
            del o[k]
            log.append(f"删除死键 {k}")
    for k in POP_KEYS:
        if k in o:
            del o[k]
            log.append(f"清除 {k}（torch 会 pop）")
    # 3) random_flip 拆分
    if "random_flip" in options and "random_src_flip" not in options:
        v = bool(options["random_flip"])
        o["random_src_flip"] = v
        # dst 侧：torch 习惯 True（泛化），老 dat 无对应概念
        o.setdefault("random_dst_flip", True)
        del o["random_flip"]
        log.append(f"random_flip={v} 拆分为 random_src_flip={v} + random_dst_flip={o['random_dst_flip']}")
    # 4) eyes_mouth_prio 合并
    if "eyes_mouth_prio" not in o:
        old = options.get("eyes_prio", options.get("mouth_prio"))
        v = bool(old) if old is not None else False
        o["eyes_mouth_prio"] = v
        log.append(f"eyes_mouth_prio={v}（由老 eyes_prio/mouth_prio 推导）")
    # 5) lr_dropout 布尔归一为字符串
    lrd = o.get("lr_dropout")
    if lrd in LR_DROPOUT_MAP:
        o["lr_dropout"] = LR_DROPOUT_MAP[lrd]
        log.append(f"lr_dropout {lrd!r} → {o['lr_dropout']!r}")
    # 6) lr 校准
    if lr_override is not None:
        old = o.get("lr")
        o["lr"] = float(lr_override)
        log.append(f"lr: {old} → {o['lr']}（显式指定）")
    # 7) 种 torch 新参数
    for k, v in DEFAULTS_TORCH.items():
        if k in o:
            continue
        o[k] = v
        log.append(f"补 torch 新键 {k}={v!r}")
    # 8) plateau（用户习惯：永远开）
    if plateau is not None:
        changed = o.get("lr_plateau_factor") != plateau
        o["lr_plateau_factor"] = float(plateau)
        o.setdefault("lr_plateau_patience", 0)
        o.setdefault("lr_plateau_min_ratio", 0.1)
        if changed:
            log.append(f"种入 lr_plateau_factor={plateau} (patience=0自动, min_ratio=0.1)")
    return o, log


def resolve_dat(path: Path, model_name: str | None) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        if model_name:
            p = path / f"{model_name}_data.dat"
            if p.exists():
                return p
            raise SystemExit(f"[dat_migrate] 目录里没有 {p}")
        cands = sorted(path.glob("*_data.dat"))
        if len(cands) == 1:
            return cands[0]
        names = ", ".join(c.name for c in cands) or "（无）"
        raise SystemExit(f"[dat_migrate] {path} 下有 {len(cands)} 个 dat，用 --model 指定：{names}")
    raise SystemExit(f"[dat_migrate] 路径不存在: {path}")


def main():
    ap = argparse.ArgumentParser(description="老 DFL(TF) data.dat → 深变(torch) 兼容体检/迁移")
    ap.add_argument("--doctor", metavar="PATH", help="体检查单个 dat 或模型目录")
    ap.add_argument("--scan", metavar="DIR", help="批量体检目录下全部 *_data.dat")
    ap.add_argument("--fix", metavar="PATH", help="迁移演练（不落盘，除非 --apply）")
    ap.add_argument("--apply", action="store_true", help="配合 --fix：真正写盘（自动 .bak-<date> 备份）")
    ap.add_argument("--model", help="目录下多 dat 时指定模型名")
    ap.add_argument("--plateau", type=float, default=0.3, help="fix 时种入的 lr_plateau_factor（默认 0.3）")
    ap.add_argument("--no-plateau", action="store_true", help="fix 不种 plateau")
    ap.add_argument("--lr", type=float, default=None, help="fix 时校准 lr 声明值")
    args = ap.parse_args()

    if not (args.doctor or args.scan or args.fix):
        ap.print_help()
        return 2

    rc = 0
    if args.scan:
        d = Path(args.scan)
        files = sorted(d.glob("*_data.dat"))
        if not files:
            print(f"[dat_migrate] {d} 下没有 *_data.dat")
            return 2
        bad = 0
        print(f"{'模型':<14}{'级别':<7}{'键':<22}说明")
        for f in files:
            dat = load_dat(f)
            opts = dat.get("options", {}) if isinstance(dat, dict) else {}
            name = f.name.replace("_data.dat", "")
            issues = doctor(opts)
            errs = [i for i in issues if i[0] == "ERROR"]
            bad += 1 if errs else 0
            if not issues:
                print(f"{name:<14}{'OK':<7}")
            for lvl, k, msg in issues:
                print(f"{name:<14}{lvl:<7}{k:<22}{msg}")
        print(f"\n共 {len(files)} 个 dat，{bad} 个含 ERROR（值域非法，起训即崩级别）")
        rc = 1 if bad else 0

    if args.doctor:
        p = resolve_dat(Path(args.doctor), args.model)
        dat = load_dat(p)
        opts = dat.get("options", {}) if isinstance(dat, dict) else {}
        print(f"[dat_migrate] 体检 {p}")
        print(f"  iter={dat.get('iter', '?')}  options 键数={len(opts)}")
        issues = doctor(opts)
        if not issues:
            print("  ✓ 干净，无兼容问题")
        else:
            rc = 1
            for lvl, k, msg in issues:
                mark = {"ERROR": "✗", "WARN": "!", "INFO": "·"}[lvl]
                print(f"  {mark} [{lvl}] {k}: {msg}")

    if args.fix:
        p = resolve_dat(Path(args.fix), args.model)
        dat = load_dat(p)
        opts = dat.get("options", {}) if isinstance(dat, dict) else {}
        plateau = None if args.no_plateau else args.plateau
        new_opts, log = migrate(opts, plateau, args.lr)
        print(f"[dat_migrate] 迁移方案 {p}")
        if not log:
            print("  （无需变更）")
        for line in log:
            print(f"  - {line}")
        if args.apply:
            import datetime
            stamp = datetime.date.today().strftime("%Y%m%d")
            bak = p.with_name(p.name + f".bak-{stamp}")
            if not bak.exists():
                shutil.copy2(p, bak)
                print(f"  备份 → {bak.name}")
            dat["options"] = new_opts
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_bytes(pickle.dumps(dat))
            tmp.replace(p)
            print(f"  ✓ 已落盘（{len(log)} 项变更）")
        else:
            print("  （演练模式，未落盘；加 --apply 生效）")
    return rc


if __name__ == "__main__":
    sys.exit(main())
