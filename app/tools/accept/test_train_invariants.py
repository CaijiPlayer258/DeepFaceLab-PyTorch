# -*- coding: utf-8 -*-
r"""训练不变量测试：专门抓「不报错、只是悄悄算错」的 bug。

跑法（统一环境）：
    runtime\unified\python.exe app\tools\accept\test_train_invariants.py

覆盖：
  1. 损失接线   —— masked_training 开关下 dst/src 重建损失都必须真的非零
                   （曾经 masked_training=False 时 dst 损失 = dssim(t,t) ≈ 4e-6）
  2. 优化器数学 —— Adam / AdaBelief 与 torch 同参同梯度逐步对齐
  3. 存档往返   —— 权重逐位相等；布局指纹不符必须拒绝加载，而不是串着读
  4. 调度曲线   —— warmup 首步 = lr/N；余弦退火末尾 ≈ 0；平台期按窗口衰减
  5. 梯度覆盖   —— 没有梯度的参数不能被更新（抓「参数漏出优化器」）
"""
import os
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ENGINE = os.path.join(ROOT, "app", "engine")
if ENGINE not in sys.path:
    sys.path.insert(0, ENGINE)

from core.leras import nn                      # noqa: E402
from core.leras.optimizers.Adam import Adam    # noqa: E402
from core.leras.optimizers.AdaBelief import AdaBelief  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail or ""))
    except AssertionError as e:
        RESULTS.append((False, name, str(e)))
    except Exception as e:                                   # noqa: BLE001
        RESULTS.append((False, name, "%s: %s" % (type(e).__name__, e)))


# ---------------------------------------------------------------- 1. 损失接线
def _loss_stub(masked_training):
    from models.Model_SAEHD.Model_pytorch import Model
    s = object.__new__(Model)
    s.resolution = 64
    s.masked_training = masked_training
    s.random_color_alg = "none"
    s.edge_enhance_power = 0.0
    s.vgg_perceptual_power = 0.0
    s.vgg_extractor = None
    s.gan_power = 0.0
    s.pretrain = False
    s.eyes_mouth_prio = False
    s.device = torch.device("cpu")
    s.mouth_scale = 1.0
    s.eyes_scale = 1.0
    s.options = {"face_style_power": 0.0, "bg_style_power": 0.0}
    s._em_split = types.MethodType(Model._em_split, s)
    s._scale_em_mask = types.MethodType(Model._scale_em_mask, s)
    return s


def test_loss_wiring():
    torch.manual_seed(0)
    n, c, h = 2, 3, 64
    tgt_s = torch.rand(n, c, h, h)
    tgt_d = torch.rand(n, c, h, h)
    m = torch.ones(n, 1, h, h) * 0.5
    fw = {
        "pred_src_src": torch.rand(n, c, h, h),
        "pred_src_srcm": torch.rand(n, 1, h, h),
        "pred_dst_dst": torch.rand(n, c, h, h),
        "pred_dst_dstm": torch.rand(n, 1, h, h),
        "pred_src_dst": None, "pred_src_dstm": None, "pred_src_dst_no_code_grad": None,
    }
    out = {}
    for mt in (True, False):
        s = _loss_stub(mt)
        r = s._recon_losses(tgt_s, tgt_d, m, m, m, m, fw)
        src_l = float(r[0].detach().mean())
        dst_l = float(r[1].detach().mean())
        out[mt] = (src_l, dst_l)
        assert src_l > 1e-3, "masked=%s 时 src 重建损失接近 0：%.3e" % (mt, src_l)
        assert dst_l > 1e-3, ("masked=%s 时 dst 重建损失接近 0：%.3e "
                              "（pred 侧被替换成了 target？）" % (mt, dst_l))
    return "src/dst loss masked=on %.3f/%.3f, off %.3f/%.3f" % (
        out[True][0], out[True][1], out[False][0], out[False][1])


# ------------------------------------------------------------ 2. 优化器数值
def _parity(cls, torch_cls, extra_cls_kw=None):
    torch.manual_seed(1)
    base = torch.randn(600)
    g = torch.randn(600) * 0.1
    p1 = torch.nn.Parameter(base.clone())
    opt1 = cls([p1], lr=1e-3, **(extra_cls_kw or {}))
    opt1.initialize_variables([p1])
    p2 = torch.nn.Parameter(base.clone())
    opt2 = torch_cls([p2], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(20):
        p1.grad = g.clone(); opt1.step(); opt1.zero_grad()
        p2.grad = g.clone(); opt2.step()
    d1 = (base - p1.detach())
    d2 = (base - p2.detach())
    err = float((d1 - d2).abs().max())
    assert err < 5e-7, "%s 与 torch 偏差 %.3e（缺偏差校正？）" % (cls.__name__, err)
    return "%s 与 torch 最大偏差 %.2e" % (cls.__name__, err)


def test_adam_parity():
    return _parity(Adam, torch.optim.Adam)


def test_adabelief_parity():
    """AdaBelief 论文公式（含偏差校正）的参考实现，逐位对比。

    AdaBelief 的稳态步长本来就 **不等于** lr：梯度恒定时 s_t 会趋于 0，
    步长被 eps 托住，所以判据只能是公式，不能是“步长≈lr”。
    """
    torch.manual_seed(2)
    base = torch.randn(300)
    g = torch.randn(300) * 0.1
    p = torch.nn.Parameter(base.clone())
    opt = AdaBelief([p], lr=1e-3, eps=1e-8)
    opt.initialize_variables([p])

    m = torch.zeros_like(base)
    v = torch.zeros_like(base)
    b1, b2, lr, eps = 0.9, 0.999, 1e-3, 1e-8
    ref = base.clone()
    for t in range(1, 21):
        p.grad = g.clone()
        opt.step(); opt.zero_grad()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g - m) ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        ref = ref - lr * m_hat / (torch.sqrt(v_hat) + eps)
        err = float((p.detach() - ref).abs().max())
        assert err < 5e-7, "第 %d 步与论文公式偏差 %.3e" % (t, err)
    return "20 步与论文公式逐位一致（最大偏差 <5e-7）"

# ------------------------------------------------------------ 3. 存档往返
def test_ckpt_roundtrip():
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(64))
    a = Adam([p], lr=1e-3, name="ck")
    a.initialize_variables([p])
    p.grad = torch.randn(64); a.step(); a.zero_grad()
    a.layout_tag = "enc1|AB1|B1|dec1"
    d = tempfile.mkdtemp()
    f = os.path.join(d, "opt.pth")
    a.save_weights(f)

    b = Adam([torch.nn.Parameter(p.detach().clone())], lr=1e-3, name="ck")
    b.initialize_variables(b.params)
    b.layout_tag = "enc1|AB1|B1|dec1"
    assert b.load_weights(f), "同布局加载失败：%s" % getattr(b.last_load_result, "detail", "")
    err = float((b.params[0].detach() - p.detach()).abs().max())
    assert err == 0.0, "往返后权重不一致：%.3e" % err

    c = Adam([torch.nn.Parameter(p.detach().clone())], lr=1e-3, name="ck")
    c.initialize_variables(c.params)
    c.layout_tag = "enc1|B1|dec1"
    assert not c.load_weights(f), "布局不符却接受了存档（会串动量）"
    return "往返逐位一致；布局不符已拒绝"


# ------------------------------------------------------------ 4. 调度曲线
def test_lr_schedule():
    torch.manual_seed(4)
    p = torch.nn.Parameter(torch.randn(8))

    # warmup：首步 = lr/N
    o = Adam([p], lr=1e-3, lr_warmup_steps=100)
    o.initialize_variables([p])
    o.iterations.fill_(1)
    v1 = o._get_lr_value()
    assert abs(v1 - 1e-3 / 100) < 1e-9, "warmup 首步 %.3e，期望 1e-5" % v1

    # 余弦退火：末尾 ≈ 0
    o2 = Adam([p], lr=1e-3, lr_total_steps=1000)
    o2.initialize_variables([p])
    o2.iterations.fill_(1000)
    v2 = o2._get_lr_value()
    assert v2 < 1e-12, "退火末尾 lr=%.3e，期望 ≈0" % v2
    o2.iterations.fill_(500)
    v3 = o2._get_lr_value()
    assert abs(v3 - 5e-4) < 1e-6, "退火中点 lr=%.3e，期望 5e-4" % v3

    # 平台期：连续两窗不降 loss → lr *= factor
    o3 = Adam([p], lr=1e-3, lr_plateau_factor=0.5, lr_plateau_patience=10)
    o3.initialize_variables([p])
    for i in range(10):
        o3.iterations.fill_(i + 1)
        o3.observe_loss(1.0)
        o3._get_lr_value()
    before = float(o3.lr_cur)
    for i in range(10, 30):
        o3.iterations.fill_(i + 1)
        o3.observe_loss(1.0)
        o3._get_lr_value()
    after = float(o3.lr_cur)
    assert after < before, "平台期没有降 lr：%.3e -> %.3e" % (before, after)
    return "warmup 首步 1e-05；退火中点 5e-04、末点 %.1e；平台期 lr %.1e→%.1e" % (v2, before, after)


# ------------------------------------------------------------ 5. 梯度覆盖
def test_grad_coverage():
    a = torch.nn.Parameter(torch.zeros(16))
    b = torch.nn.Parameter(torch.zeros(16))      # 永远没有 grad
    opt = Adam([a, b], lr=1e-2)
    opt.initialize_variables([a, b])
    for _ in range(3):
        a.grad = torch.ones(16)
        opt.step(); opt.zero_grad()
    da = float(a.detach().abs().mean())
    db = float(b.detach().abs().mean())
    assert da > 0.0, "有梯度的参数没被更新"
    assert db == 0.0, "没有梯度的参数被改动了（%.3e）" % db
    return "有梯度更新 %.3e；无梯度保持 0" % da


def main():
    check("1 损失接线（masked on/off）", test_loss_wiring)
    check("2 Adam 数值对齐 torch", test_adam_parity)
    check("2b AdaBelief 稳态步长", test_adabelief_parity)
    check("3 存档往返 + 布局指纹", test_ckpt_roundtrip)
    check("4 学习率调度曲线", test_lr_schedule)
    check("5 梯度覆盖", test_grad_coverage)

    ok = sum(1 for r in RESULTS if r[0])
    print("")
    for good, name, detail in RESULTS:
        print("%s  %-28s %s" % ("PASS" if good else "FAIL", name, detail))
    print("\n%d/%d 通过" % (ok, len(RESULTS)))
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
