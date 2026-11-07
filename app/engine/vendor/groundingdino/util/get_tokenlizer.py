# -*- coding: utf-8 -*-
"""GroundingDINO 的文本编码器（tokenizer + BERT 权重）——只认本地目录。

上游原版这里是 `AutoTokenizer.from_pretrained("bert-base-uncased")` /
`BertModel.from_pretrained("bert-base-uncased")`，见 config 里的
`text_encoder_type = "bert-base-uncased"`。用户机器多半没网：那一句要么挂在 DNS /
TLS 上去几十秒，报一个跟「模型加载失败」毫不相干的错，要么在 HF_HUB_OFFLINE=1 下
直接 OSError。所以这里统一改成「先把名字翻译成本地目录，再 local_files_only」，
不保留任何联网回落——回落只会把超时伪装成「模型慢」。

目录里该有什么（weights/bert-base-uncased/ —— 权重全部平铺，只有它保留一个目录）：

    vocab.txt                               词表（tokenizer 必需，231 KB）
    tokenizer.json                          可选，有就走 fast tokenizer（466 KB）
    tokenizer_config.json                   可选（do_lower_case / model_max_length）
    config.json                             BertModel 结构（必需，570 B）
    pytorch_model.bin 或 model.safetensors  BertModel 权重（必需，420 MB）

目录优先级（第一个命中就用，命中后不再看后面的）：

    1) $DFL_GDINO_TEXT_ENCODER_DIR
    2) $DFL_WEIGHTS_DIR/<name>                   ← 平铺布局（默认）
    3) <仓库根>/weights/<name>                   ← worker 传了 --weights 就是这里
    4) $DFL_WEIGHTS_DIR/grounding_dino/<name>    ← 老的分目录布局，只作兜底
    5) <仓库根>/weights/grounding_dino/<name>

注意：GDINO 的 checkpoint 里**没有** bert.* 的权重（940 个 key 全是 backbone /
transformer / feat_map 的），文本编码器必须靠这几个文件单独提供，不能省。
"""
from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoTokenizer, BertModel, RobertaModel

# BertModel 的权重文件名，两个都认（transformers 5 优先 safetensors）
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

_DOWNLOAD_HINT = (
    "从镜像离线补：curl -L -o \"%s\" \"https://hf-mirror.com/bert-base-uncased/resolve/main/%s\""
)


def _cand_dirs(name):
    """可能放这个文本编码器的目录，按优先级排。"""
    out = []
    env = os.environ.get("DFL_GDINO_TEXT_ENCODER_DIR", "").strip()
    if env:
        out.append(Path(env))
    # app/engine/vendor/groundingdino/util/get_tokenlizer.py → parents[5] = 仓库根
    repo_weights = Path(__file__).resolve().parents[5] / "weights"
    env = os.environ.get("DFL_WEIGHTS_DIR", "").strip()
    if env:
        out.append(Path(env) / name)                       # 平铺：<weights>/bert-base-uncased
    out.append(repo_weights / name)                        # 平铺（默认）
    if env:
        out.append(Path(env) / "grounding_dino" / name)    # 老布局兜底
    out.append(repo_weights / "grounding_dino" / name)     # 老布局兜底
    return out


def _local_model_dir(name, need=("vocab.txt",)):
    """本地目录里凑齐 need 那几个文件就返回它，否则返回 ""。"""
    for cand in _cand_dirs(name):
        try:
            if cand.is_dir() and all((cand / f).is_file() for f in need):
                return str(cand)
        except OSError:
            continue
    return ""


def _name_of(text_encoder_type):
    """text_encoder_type 可能是字符串、SLConfig、dict，或者直接就是一个目录。"""
    if isinstance(text_encoder_type, str):
        return text_encoder_type
    if hasattr(text_encoder_type, "text_encoder_type"):
        return str(text_encoder_type.text_encoder_type)
    try:
        v = text_encoder_type.get("text_encoder_type", None)
        if v:
            return str(v)
    except Exception:
        pass
    raise ValueError("Unknown type of text_encoder_type: {}".format(type(text_encoder_type)))


def _pick_dir(text_encoder_type, need=("vocab.txt",)):
    """名字 → 本地目录。已经是目录就原样用；否则按优先级找，找不到抛带路径的错。"""
    name = _name_of(text_encoder_type)
    p = Path(name)
    try:
        if p.is_dir():
            return str(p)
    except OSError:
        pass
    got = _local_model_dir(name, need=need)
    if got:
        return got
    tried = "\n".join("    - " + str(c) for c in _cand_dirs(name))
    raise RuntimeError(
        "GroundingDINO 的文本编码器 %r 不在本地，也没有联网回落（用户机器可能没网）。\n"
        "把文件放到下面任一目录（缺 %s）：\n%s\n"
        % (name, " / ".join(need), tried)
    )


def get_tokenlizer(text_encoder_type):
    d = _pick_dir(text_encoder_type, need=("vocab.txt",))
    # local_files_only：目录已经定死了，禁止 hub 再去问一次网络
    return AutoTokenizer.from_pretrained(d, local_files_only=True)


def get_pretrained_language_model(text_encoder_type):
    name = _name_of(text_encoder_type)
    if name.startswith("bert"):
        cls, need = BertModel, ("config.json",)
    elif name.startswith("roberta"):
        cls, need = RobertaModel, ("config.json",)
    else:
        raise ValueError("Unknown text_encoder_type {}".format(name))

    # 先按「结构文件齐全」挑目录：只有 vocab.txt 的目录不算（BertModel 还要 config.json）
    d = _pick_dir(text_encoder_type, need=need)
    if not any((Path(d) / f).is_file() for f in WEIGHT_FILES):
        raise RuntimeError(
            "文本编码器权重缺文件：%s 里既没有 %s。\n%s\n"
            "（GDINO 的 checkpoint 不含 bert.* 权重，这几百兆省不掉）"
            % (d, " 也没有 ".join(WEIGHT_FILES),
               _DOWNLOAD_HINT % (Path(d) / WEIGHT_FILES[1], WEIGHT_FILES[1]))
        )
    return cls.from_pretrained(d, local_files_only=True)
