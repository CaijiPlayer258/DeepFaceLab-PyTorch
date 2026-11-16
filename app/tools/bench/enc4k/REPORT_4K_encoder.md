# 4K 导出：libx264 vs NVENC（实测，本机 RTX 5070 Laptop + 8940HX 16C/32T）

**素材**：`workspace/9月29日 (5)_merged.mp4`（真实实拍 DFL 成片，1080×1920，含音轨）。
4K = 同片 lanczos 上采样到 **2160×3840（8.29 MP，与 3840×2160 同像素数）**，各取 80 帧。
旁证：另一条 3840×2160 动画片 x264 4K = 16.0 fps（vs 实拍 17.0）—— 4K 下 x264 是**像素驱动**的，内容难度几乎不影响。
口径与生产 `_fused_pipe_cmd` 逐字一致（rawvideo bgr24 管道 + `pad,format=rgb24,scale=out_range=tv` + `-c:a copy` + faststart），只换 `-c:v`。同进程按轮交替，各 2 轮，表为中位。

## 三臂表（4K，80 帧）

| 臂 | fps | ms/帧 | 写管道 ms | ffmpeg 占用 | GPU SM / NVENC | 体积 MB | Mbps | 逐像素差 vs x264 |
|---|---|---|---|---|---|---|---|---|
| A x264 crf16（现状） | **17.0** | **59.0** | 19.8 | **14.98 核** | 6.9% / — | 10.13 | 30.4 | 基准 |
| B1 h264_nvenc CQ19 | 45.7 | 21.9 | 19.5 | 5.28 核 | 18.3% / 42% | 13.76 | 41.3 | 均差 0.80 LSB，>8LSB 0.02%，PSNR 45.6 |
| B2 h264_nvenc CQ23 | 48.5 | 20.6 | 18.4 | 6.46 核 | 13.3% / 28% | 8.25 | 24.7 | 均差 0.86 |
| C1 hevc_nvenc CQ19 | 50.3 | 19.9 | 17.8 | 6.06 核 | 15.6% / 45% | **7.03** | 21.1 | 均差 0.86 |
| C2 hevc_nvenc CQ23 | 51.7 | 19.4 | 17.4 | 7.53 核 | 13.5% / 33% | 5.90 | 17.7 | 均差 0.89 |
| D 管道不编码（地板） | 64.1 | 15.7 | 15.3 | 4.69 核 | 7.0% / 5% | — | — | — |
| B3 h264_nvenc CQ21（默认档） | 48.1 | 20.8 | 21.0 | 5.23 核 | — | 10.73 | 32.2 | 体积≈crf16(+6%) |
| C3 hevc_nvenc CQ21（默认档） | 48.9 | 20.5 | 19.2 | 6.47 核 | — | 6.99 | 21.0 | 体积×0.69 |

1080×1920 同片：A 14.2 ms/帧（70 fps，14.8 核）· B1 7.8（129 fps，6.0 核）· C1 7.7（130 fps，5.7 核）· 地板 5.9（193 fps）。

## ① 4K 下 x264 是真墙

59 ms/帧 = 17 fps 上限，且**吃 15.0/16 物理核（94%）**。1080p 只要 14.2 ms 但同样占 14.8 核 —— 换到 4K 是「时间 ×4.2、核数不变（已满）」。整机只剩 1 核给检测/dfm/合成，流水线会被挤成串行：这不是"慢一点"，是抢占。

## ② NVENC 能拉回 2.7~3.0×，但 4K 的天花板是管道

h264 CQ19 21.9 ms（2.7×）、hevc CQ19 19.9 ms（3.0×），CPU 从 15 核降到 5~7.5 核（还 8~10 核给上游）。GPU 只吃点余量：SM 13~18%（上传/format 转换），NVENC 独立引擎 28~45%，不抢 dfm 的 SM。
但 **D 臂**说明地板在哪：不编码、只走管道+滤镜，4K 也要 15.7 ms/帧（64 fps、4.7 核）—— 24.9 MB/帧 过匿名管道 + swscale。所以 4K NVENC 的端到端上限 ≈ 46~52 fps，编码器之外还有 15 ms 拿不掉。

## ③ 1080p：NVENC 本来就更快，不是"只为 4K"

129 fps vs 70 fps（7.8 vs 14.2 ms/帧）。但 1080p 整条管线瓶颈在上游（生产实测 ~41 ms/帧），编码被盖住 —— 所以 1080p 的收益不是 fps，而是把 15 核还给合成/检测。1080p 可以不动（零风险），**4K 必须动**。

## ④ md5：换编码器必变，且必须显式

每档 2 轮 md5 完全相同（nvenc 在本机/本驱动上是确定性的），所以新基线是稳的；但它一定 ≠ crf16 的 md5。画质上 4K 五档全部落在 4:2:0 的噪声底里（vs 原始帧 PSNR：x264 43.42 / B1 43.67 / C1 43.33），逐像素均差 0.8 LSB、99% 像素差 <1、>8 LSB 的只占 0.02%，抠图肉眼无差 —— **换编码器不是画质退步，是换一条压缩曲线**。真正的代价是体积：CQ19 比 crf16 大 36%。

## CQ 与 crf 不等价

crf16 ≈ **h264 CQ21**（32.2 vs 30.4 Mbps，+6%）· CQ19 大 36% · CQ23 小 18% 但 PSNR −0.4 dB。hevc CQ21 = 21.0 Mbps（体积 ×0.69，同观感）。hevc 兼容性：mp4 默认打 `hev1`，Apple 系要 `-tag:v hvc1`；微信/老安卓 H5 可能黑屏；实测软解 4K h264 177 fps / hevc 174 fps，解码不是问题。

## 结论

**4K 该开 NVENC**：默认 `nvenc-h264 CQ21`（体积≈crf16、46 fps、5 核），要小 31% 且能接受 hevc 兼容面就 `nvenc-hevc CQ21`。1080p 维持 x264。
**auto 不要自动切**：保持 x264，只在新片 ≥6 MP 时**告警**一次（换编码器 = 换字节/体积/画质/兼容面，且 accept 测试以 md5 对拍，用户必须知情）。要自动切，另开 `DFL_FUSED_ENCODER_AUTO4K=1`，默认关。

落地开关设计见 `DESIGN_DFL_FUSED_ENCODER.md`（含 250 ms 预检、两条路径同源、accept 用例怎么改）。

## 复现

~~~powershell
$py = "runtime\server\Scripts\python.exe"          # 仓库根
$env:PYTHONIOENCODING="utf-8"
& $py app\tools\bench\enc4k\build_fixture.py     # 造 raw 帧（真实片 1080p / 2160x3840 / 动画 3840x2160）
& $py app\tools\bench\enc4k\bench4k.py           # 三臂矩阵（同进程、按轮交替、各 2 轮）
& $py app\tools\bench\enc4k\bench_cq21_clean.py  # CQ21 两档（默认档）干净计时 ×3 轮
& $py app\tools\bench\enc4k\quality.py           # 逐像素差 / PSNR / SSIM / 抠图
& $py app\tools\bench\enc4k\decode_cost.py       # 成片软解成本
& $py app\tools\bench\enc4k\summarize.py         # 中位汇总
~~~

产物都落在仓库根的 scratch 目录 `_enc4k_bench/`（脚本自己按 `__file__` 定位，第一次跑由 `build_fixture.py` 建目录）：
`results.json` / `results_cq21.json` / `quality.json` / `crop_*.png` / `out_*.mp4` / `f_*.raw`（约 4.3 GB，量完即删）。
未改任何生产代码。
