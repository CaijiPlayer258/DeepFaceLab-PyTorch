#!/usr/bin/env python3
"""清理 DeepFaceLab Torch 模型的 data.dat。

常见用途：
- 重置 iter / loss_history（iter 默认设为 1，避免触发“模型首次运行”）
- 清理 options 里的“垃圾字段”（会出现在 *_summary.txt 的 Model Options 区域）
- 可选删除旧的 *_summary.txt（下次训练会重新生成干净的 summary）

示例：
  # 仅清理 options 里的字段，并删除 summary
	python tools/clear_Iter.py --data workspace/model/new_SAEHD_data.dat --remove-option-contains "模型训练|严禁转卖" --delete-summary

	# 重置 iter(=1) + loss_history，并清字段
	python tools/clear_Iter.py --data workspace/model/new_SAEHD_data.dat --reset-iter --clear-loss-history
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path


def _load_pickle_dict(path: Path) -> dict:
	obj = pickle.loads(path.read_bytes())
	if not isinstance(obj, dict):
		raise TypeError(f"{path} 不是 dict（data.dat 格式异常）")
	return obj


def _backup_file(path: Path) -> Path:
	ts = time.strftime("%Y%m%d-%H%M%S")
	bak = path.with_suffix(path.suffix + f".bak.{ts}")
	bak.write_bytes(path.read_bytes())
	return bak


def _derive_summary_path(data_dat: Path) -> Path:
	# new_SAEHD_data.dat -> new_SAEHD_summary.txt
	name = data_dat.name
	if name.endswith("_data.dat"):
		stem = name[: -len("_data.dat")]
		return data_dat.parent / f"{stem}_summary.txt"
	return data_dat.with_name(data_dat.stem + "_summary.txt")


def _parse_patterns(raw: str) -> list[str]:
	"""Parse a user string into substring patterns.

	Supported delimiters: ',', '，', '|'.
	"""
	raw = (raw or "").strip()
	if not raw:
		return []
	for delim in ("，", ","):
		raw = raw.replace(delim, "|")
	parts = [p.strip() for p in raw.split("|")]
	return [p for p in parts if p]


def _clean_options(options: dict, garbage_substrings: list[str]) -> tuple[dict, list[str]]:
	removed: list[str] = []
	out: dict = {}
	for k, v in options.items():
		ks = str(k)
		if any(s in ks for s in garbage_substrings):
			removed.append(ks)
			continue
		out[k] = v
	return out, removed


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description="清理/重置 DeepFaceLab 模型 data.dat")
	ap.add_argument("--data", type=Path, required=True, help="指向 *_data.dat 文件，例如 workspace/model_converted/new_SAEHD_data.dat")
	ap.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份（不推荐）")

	ap.add_argument("--reset-iter", action="store_true", help="将 iter 重置为 1（同时会清空 loss_history）")
	ap.add_argument("--clear-loss-history", action="store_true", help="清空 loss_history")
	ap.add_argument("--clear-preview-sample", action="store_true", help="清空 sample_for_preview")

	ap.add_argument(
		"--remove-option-contains",
		type=str,
		default="",
		help="删除 options 中 key 含指定子串的项。支持用 ',' / '，' / '|' 分隔多个子串，例如：\"模型训练|严禁转卖\"",
	)
	ap.add_argument("--delete-summary", action="store_true", help="删除同目录下的 *_summary.txt")

	args = ap.parse_args(argv)

	data_dat: Path = args.data
	if not data_dat.exists():
		raise FileNotFoundError(data_dat)

	if not args.no_backup:
		bak = _backup_file(data_dat)
		print(f"已备份: {bak}")

	d = _load_pickle_dict(data_dat)

	if args.reset_iter:
		d["iter"] = 1
		# reset-iter 常用于重新开始训练：同时清空 loss_history，方便从头查看曲线
		d["loss_history"] = []

	if args.clear_loss_history:
		d["loss_history"] = []

	if args.clear_preview_sample:
		d["sample_for_preview"] = None

	garbage = _parse_patterns(args.remove_option_contains)

	opts = d.get("options", {})
	if isinstance(opts, dict) and garbage:
		new_opts, removed = _clean_options(opts, garbage)
		d["options"] = new_opts
		if removed:
			print("已删除 options 字段:")
			for k in removed:
				print(f" - {k}")

	data_dat.write_bytes(pickle.dumps(d))
	print(f"写入完成: {data_dat}")
	print(f"iter={int(d.get('iter', 0) or 0)} loss_history_len={len(d.get('loss_history', []) or [])}")

	if args.delete_summary:
		sp = _derive_summary_path(data_dat)
		if sp.exists():
			sp.unlink()
			print(f"已删除 summary: {sp}")
		else:
			print(f"未找到 summary（可忽略）: {sp}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())