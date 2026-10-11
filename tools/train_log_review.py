# -*- coding: utf-8 -*-
"""train_log_review —— 经验提案器：读 train_log.jsonl 找可升级为习惯的模式（≥3 例才提案）

用法: python tools/train_log_review.py           # 输出候选提案（agent 转述给用户裁决）
原则：只发现、不写入；提案经用户确认后由 agent 追加进 skill 习惯节（带日期+样本数）。
"""
import json
from collections import Counter
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "workspace" / "train_log.jsonl"


def main():
    rows = [json.loads(l) for l in LOG.open(encoding="utf-8")] if LOG.exists() else []
    print(f"[review] 读取 {len(rows)} 条训练记录")
    proposals = []

    # 模式1：verdict 分布（收工习惯）
    vc = Counter(r.get("verdict") for r in rows if r.get("verdict"))
    for k, n in vc.items():
        if n >= 3:
            proposals.append(f"收工习惯：{n} 次训练以「{k}」结束——可写进习惯节的收工策略")

    # 模式2：改参类型分布（改参习惯）
    ch = Counter()
    for r in rows:
        for c in r.get("changes", []):
            raw = json.dumps(c, ensure_ascii=False)   # 兼容 {raw:...} 与 {at,lr,...} 两种格式
            for key in ("lr", "random_occlusion", "random_noise", "random_warp", "edge_enhance"):
                if key in raw:
                    ch[key] += 1
    for k, n in ch.items():
        if n >= 3:
            proposals.append(f"改参习惯：{k} 被调整 {n} 次——高频参数，值得写默认策略")

    # 模式3：最终 loss 水平（质量基线）
    finals = [r["final_loss"]["src"] for r in rows
              if r.get("final_loss") and r["final_loss"].get("src")]
    if len(finals) >= 3:
        finals.sort()
        proposals.append(f"质量基线：{len(finals)} 次训练 src 最终 loss "
                         f"min={finals[0]:.4f} mid={finals[len(finals)//2]:.4f} "
                         f"max={finals[-1]:.4f}——中期检查时可对照（超 max 提前预警）")

    if proposals:
        print("[review] 候选提案（需用户确认后才入册习惯节）：")
        for p in proposals:
            print("  -", p)
    else:
        print("[review] 暂无 ≥3 例的稳定模式（样本还少，继续积累）")


if __name__ == "__main__":
    main()
