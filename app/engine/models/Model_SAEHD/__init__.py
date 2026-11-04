"""Model_SAEHD package —— 只保留 PyTorch 实现。

以前这里是 try/except：优先 Model_pytorch，导入失败就悄悄退回 TF 版 Model.py。
那个兜底会把真正的导入错误吞掉，表现成 "nn.tf has no attribute tf" 这种
牛头不对马嘴的报错（实际上 torch 版压根没被导入成功）。
本项目只做 torch 训练，TF 版已删除，导入失败就该直接抛出来。
"""
from .Model_pytorch import Model
