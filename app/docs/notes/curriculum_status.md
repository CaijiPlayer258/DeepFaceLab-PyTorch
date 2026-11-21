# 分辨率课程：现状与卡点

目标：把 128 → 256 → 416 的逐级升维做出来，用 inter 重采样的结论（升采样 ρ=0，不丢信息）
把训练预算挪到前面去，比较「同墙钟时间下」的最终质量。

## 已经跑通的部分

```
1) 引擎能被无头拉起：
   python app/engine/main.py train --model-dir workspace/model --model SAEHD \\
       --training-data-src-dir ... --training-data-dst-dir ... --no-preview --silent-start \\
       --force-model-name _smoketrain_SAEHD
   → 打印「静默启动：已选择设备 NVIDIA GeForce RTX 5070 Laptop GPU」，模型加载成功

2) 训练器里已经挂了自动 resize 钩子（目录里只有一个模型权重时生效），
   多个权重时会打印：
   [RESIZE] 跳过：目录里有多个模型权重 ['New416_SAEHD', '_smoketrain_SAEHD']，无法判定要处理哪一个
3) 驱动脚本 app/tools/bench/curriculum_driver.py：数据集按档重采样 + 逐档起训练 + 到点停止
```

## 卡点（这才是没跑出结果的原因）

```
SampleGeneratorFace 要求 aligned 目录里的图与模型 resolution 一致。
本地数据集是 512² 的 1203 张，而模型是 416 → 生成器直接报「No training data provided」。
也就是说：课程训练的每一档都必须有一份对应分辨率的数据集，不能复用同一份。
驱动脚本里的 resample_dataset() 就是干这个的（cv2 缩放，512→128 约 1 分钟）。
```

## 成本模型（用于分配预算，尚未实测到 it/s）

同一个 SAEHD(df-udt) 主干、同一 batch，训练步耗时应大致按像素数线性增长：
128 → 256 是 4 倍，256 → 416 是 2.6 倍。
这决定了课程的预算应当「前重后轻」：在 128 上跑完大部分迭代，最后在 416 上只做收尾。

## 下一步（一条命令）

```bat
runtime\engine\python.exe app\tools\bench\curriculum_driver.py \\
    --plan 128:8000,256:4000,416:1500 \\
    --src workspace/data_src/aligned --dst workspace/data_src/aligned
```

跑完后用 `app/tools/bench/inter_study_run.py` 的评测路径（掩码内 PSNR/SSIM）比较
「课程最终模型」与「直接在 416 训练的等时长基线」。

## 为什么没有直接给出结论

这类实验的结论必须来自真实训练曲线；用合成数据或未收敛的短跑得出的数字没有意义。
本轮只把机械部分（数据重采样、逐档启动、到点停止）落地并验证了引擎可无头运行，
质量结论留给完整跑批。