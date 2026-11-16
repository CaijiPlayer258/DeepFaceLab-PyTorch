"""合成管线的公共积木 —— merger.py 与 export_pipeline.py 共用同一份实现。

这里只收**两份调用点各抄过一遍、调用序列逐字一致**的东西。判据：把它抽出来之后，
两条路上的 imagelib 调用（函数名、参数、顺序）与抽出前完全相同，所以合成出的像素
逐位不变；roi_parity_accept.py / stage6_accept.py 是这条不变的验收。

现在只有一样东西满足这个判据：色彩迁移的分派表。

  · 4.0 的 merger.py 里叫 `_ct_table(mask_area)`，用 int 编号索引；
  · 4.0 的 export_pipeline.py::_phase_merge_worker 里是一串 `if _sct in ('rct','1')`
    的 if-elif 链，用字符串索引。

两张表指向的是同一批 imagelib 函数、同一组遮罩加权方式，只是索引类型不同。
合并成 `color_transfer(src, dst, mask_area, mode)`，两种索引都认。
"""
from core import imagelib

#: config.ctm_dict 的编号 ↔ 别名。编号是 merger 侧的老接口，别名是导出侧 cfg dict 里的值。
CT_NAMES = {
    0: "none", 1: "rct", 2: "lct", 3: "mkl", 4: "mkl-m",
    5: "idt", 6: "idt-m", 7: "sot-m", 8: "mix-m", 9: "mt", 10: "lut",
}

#: 别名 → 编号（导出侧还有一批历史拼写，一并认下）
_BY_NAME = {v: k for k, v in CT_NAMES.items()}


def ct_name(mode):
    """把任意一种色迁索引（int / 数字字符串 / 别名）归一成别名；认不出就是 'none'。

    4.0 的两份实现对「认不出的值」都是**静默跳过**（merger 是 dict.get 取不到；
    导出侧是 if-elif 全不匹配），所以这里返回 'none' 与老行为一致。
    """
    if mode is None:
        return "none"
    if isinstance(mode, str):
        s = mode.strip().lower()
        if s.isdigit():
            return CT_NAMES.get(int(s), "none")
        return s if s in _BY_NAME else "none"
    try:
        return CT_NAMES.get(int(mode), "none")
    except (TypeError, ValueError):
        return "none"


#: 别名 → 实现。mkl-m / idt-m / sot-m / mix-m 的遮罩加权、以及 rct/mt 的显式 mask 参数
#: 都照抄 4.0，别在这里「顺手统一」——它们是不同算法，不是同一件事的两种写法。
_IMPL = {
    "rct":   lambda s, t, m: imagelib.reinhard_color_transfer(s, t, target_mask=m, source_mask=m),
    "lct":   lambda s, t, m: imagelib.linear_color_transfer(s, t),
    "mkl":   lambda s, t, m: imagelib.color_transfer_mkl(s, t),
    "mkl-m": lambda s, t, m: imagelib.color_transfer_mkl(s * m, t * m),
    "idt":   lambda s, t, m: imagelib.color_transfer_idt(s, t),
    "idt-m": lambda s, t, m: imagelib.color_transfer_idt(s * m, t * m),
    "sot-m": lambda s, t, m: imagelib.color_transfer_sot(s * m, t * m, steps=10, batch_size=30),
    "mix-m": lambda s, t, m: imagelib.color_transfer_mix(s * m, t * m),
    "mt":    lambda s, t, m: imagelib.match_tone_np(s, t, src_mask=m, trg_mask=m),
    "lut":   lambda s, t, m: imagelib.color_transfer_lut(s, t),
}


def has_color_transfer(mode):
    """这个模式要不要做色迁（0 / 'none' / 认不出的值 → False）。"""
    return ct_name(mode) in _IMPL


def color_transfer(src, dst, mask_area, mode):
    """按色迁模式算一张新画布。

    src / dst 是 output_size 画布上的 float32 BGR（0..1），mask_area 是 2D 或 (S,S,1)
    的遮罩（调用方自己决定要不要二值化 —— 4.0 里 merger 二值化了、导出侧没有，
    这是**本来就不同**的两条路，别在这里替谁做主）。

    认不出的模式原样返回 src：4.0 两边都是静默跳过，返回值也就没人用。
    """
    fn = _IMPL.get(ct_name(mode))
    if fn is None:
        return src
    return fn(src, dst, mask_area)
