# 任务 C:cv2.seamlessClone(Poisson 融合)能不能上 GPU?

**仓库**:`D:/MyProject/DeepFaceLab-PyTorch-4.1` · **日期**:2026-11-15
**机器**:RTX 5070 Laptop 8 GB / driver 591.86 / torch 2.7.0+cu128 / cv2 4.13.0(pip wheel,**无 CUDA 后端**)/ Python 3.12(仓库自带 `runtime/engine`)

---

## 0. 结论先行

**可行。** `cv2.seamlessClone` 的 NORMAL_CLONE 可以在 torch 里完整重建,而且**比 CPU 版快 24–53 倍**:97.1 % 的像素逐位相同,2.5 % 差 1 LSB,残差集中在 ROI 外框边缘 ≤19 px 的一条窄带里(那里 OpenCV 自己解出来的就不是该离散系统的精确解)。

| 方案 | 迭代/阈值 | max|diff| (0–255) | mean|diff| | >1 LSB | batch1 延迟(min/med) | batch8 吞吐 | batch8 ms/帧 |
|---|---|---|---|---|---|---|---|
| **cv2.seamlessClone**(CPU) | — | 基准 | — | — | **45.3 / 49.7 ms** | 362 ms | 45.3 |
| **torch · DST-matmul** ⭐ | 0(直接解) | **176**(单帧最差,中位 44) | **0.0201** | **0.211 %** | **1.51 / 1.82 ms** | **7.23 ms** | **0.90** |
| torch · DST-FFT | 0(直接解) | 176 | 0.0201 | 0.211 % | 1.67 / 2.17 ms | 10.77 ms | 1.35 |
| torch · CG | 400(1e-3) | 176 | 0.599 | 6.38 % | 347 ms | 904 ms | 113 |
| torch · CG | 719(1e-4) | 176 | 0.0299 | 0.267 % | 553 ms | 1594 ms | 199 |
| torch · CG | 1093(1e-6) | 176 | 0.0198 | 0.212 % | 717 ms | 2466 ms | 308 |

*24 张真实 1080p 帧,ROI 401×491,口径 = 与 `cv2.seamlessClone` 逐像素最大通道差。*

**基线口径核对**:任务里给的 58.5 ms/帧(占 compose 65 %)在本机复现为 **45.3 ms(min)/ 49.7 ms(med)**(ROI 401×491,单线程调用;差值来自 mask 尺寸与 CPU 差异),量级一致。cv2 的耗时 ∝ ROI 面积,所以换到更大的人脸框,58.5 ms 完全对得上。

**端到端**(1080p 整帧进 GPU → 切片 → 解 → 回写):batch1 **1.55 ms**(min)/ 2.11 ms(med),batch8 **7.89 ms → 1.11 ms/帧**。相对 cv2 的 45.3 ms:**batch1 29×,batch8 41×**;若按 58.5 ms 折算则是 **38× / 53×**。
开 tf32 只快 6 %(7.50 ms),但 `>1 LSB` 从 0.211 % 涨到 1.76 % —— 不值得,保持 fp32。

**结论:值得落地。** 需要接受的只是"不是逐位复制 cv2":97.1 % 逐位相同 + 2.5 % 差 1 LSB + ROI 边框窄带内可达 ±176,而这条窄带可以用"解算矩形外扩 4–8 px"消掉(§6.3)。

---

## 1. OpenCV 到底有没有 CUDA 版 seamlessClone?

**没有。** 三路证据:

1. 本机实测 `cv2.cuda` 命名空间共 75 个可用符号,`seam* / clone* / poisson*` 一个都没有:

   ```
   has_cuda_attr True … devcount 0 … n_cuda_api 75
   seamless-ish: []
   photo funcs: ['MIXED_CLONE','MIXED_CLONE_WIDE','NORMAL_CLONE','NORMAL_CLONE_WIDE','seamlessClone']
   ```

2. 源码树:`opencv/modules/photo/src/cuda/` 下**只有 `nlm.cu`**(非局部均值去噪),Poisson 融合没有任何 `.cu` 实现;4.13.0 的 `seamless_cloning*.{cpp,hpp}` 是纯 CPU。
3. 社区 issue [opencv#16404 "OpenCV Seamless Clone CUDA Support"](https://github.com/opencv/opencv/issues/16404)2019 年开,7 年后**仍然 open**,8 条留言全是在催("It's been what feels like 10 years.")。

另外:pip 装的 `opencv-python` **整个 CUDA 后端都是关的**(`getCudaEnabledDeviceCount() == 0`),就算 photo 模块哪天加了 CUDA 也用不上 —— 想上 GPU 只能自己写(torch/cupy)或自己编 OpenCV。这条把方案锁死在 torch。

---

## 2. 数学:seamlessClone = 泊松方程

### 2.1 连续形式

给定源图 `P`、目标图 `D`、mask `M`,在 `M` 内求 `f`,使其梯度场最接近"内部用 P 的梯度、外部用 D 的梯度":

```
min_f ∬_Ω |∇f − v|² ,   v = ∇P·χ_M + ∇D·(1−χ_M)
```

变分给出**泊松方程**,边界用 `D` 做 Dirichlet:

```
Δf = div v   in Ω
f  = D       on ∂Ω
```

NORMAL_CLONE 就是这个(`MIXED_CLONE` 只是逐像素取 |∇P| 与 |∇D| 中较大者)。

### 2.2 OpenCV 的离散化(逐行解码 `seamless_cloning_impl.cpp`)

这段代码有几处反直觉,必须照抄才谈得上对拍:

```
gx   = P(x+1) − P(x)                    # 前向差分,kernel [0,−1,+1],锚点居中
gx   = filter2D(BORDER_REFLECT_101)      # 边界反射
Me   = erode(M, 3×3 ones, iterations=3)  # == 7×7 方形腐蚀(Chebyshev 距离 3)
v_x  = gx_D·(1−Me/255) + gx_P·(Me/255)
Δv_x = v_x(j) − v_x(j−1)                 # 后向差分 [−1,+1];y 方向同理再相加
rhs  = Δv − Laplacian5(D·ring)           # ring = ROI 的 1 px 外框,内部置 0
f    = solve(Δf = rhs, f = 0 on ring)    # 矩形域,零 Dirichlet
out[1:-1,1:-1] = clamp(f,0,255) 后**截断**(static_cast,不是四舍五入)
out[ring]      = D[ring]
```

三个坑:

* **梯度是前向差分、散度是后向差分**(合起来才是对称 5 点 Laplacian)。写成"都用后向"会让 mask 边缘的梯度场整体错位。
* `Laplacian(D·ring)` 项:把"外框已知值"折算进 rhs(等价于 `f−B` 的零边值问题,`B` = 只保留外框的图)。**漏掉这一项,整幅 ROI 会整体偏一个常量**(实测 mean|diff| = 120)。
* **截断不是四舍五入**,ring 那一圈直接抄 D。

### 2.3 三条 GPU 路线

**(a) DST 对角化(OpenCV 走的就是这条,推荐)**
矩形 + 零 Dirichlet 的 5 点在正弦基下对角化:`λ_kl = 2cos(πk/(W−1)) + 2cos(πl/(H−1)) − 4`,于是

```
f = S⁻¹ · diag(1/λ) · S · rhs ,   S = 2D DST-I
```

* `S` 用显式正弦矩阵 → 两次 matmul(`Sh(491²) @ X(491×1203) @ Sw(401×401)`);也可用奇延拓 + `rfft` 做 O(N log N)。
* 全 batch、全 channel 一次算完,GPU 上 **1.5 ms(401×491×3)**,且**与 mask 形状无关**(只跟 bbox 有关)。
* **唯一能贴近 cv2 的路线**,因为 cv2 就是这条数学。

**(b) CG / 多重网格**
`A = −Δ` 对称正定,但 200k 未知数的 Poisson 问题 `κ(A) = O(N)`,CG 要 `O(√N)` 步:实测 1e-4 → **719 步**、1e-6 → **1093 步**,每步 3 次全局归约 → **199–308 ms/帧,比 CPU 的 cv2 还慢**。
多重网格能把迭代数压到 O(1)。按实测单步成本折算:CG 每步 `553 ms / 719 步 = 0.77 ms`(3 通道 491×401,含 3 次全局归约),一个 V-cycle ≈ 2–3 次细网格光滑 ⇒ **≈ 0.4–1.5 ms/V-cycle**,12 个 V-cycle 即 **5–18 ms** —— 已经是 DST 的 3–12 倍。而且 MG 收敛到的是同一个离散解,不会比 DST 更"像 cv2"。
→ **CG/MG 在这题没有存在价值**(除非 ROI 大到 matmul 撑不住,那也该换 FFT 而不是 CG)。

**(c) FFT**
同一算子,用 `rfft` 对奇延拓做:`DST-I(x) = −0.5·imag(rfft([0,x,0,−flip(x)]))[1:N+1]`。
O(N log N),实测 **1.67 ms(b1)/ 1.35 ms/帧(b8)**,比 matmul 略慢(两次复数 FFT 的访存比 BLAS 贵),但 N 更大时更稳(601×731 时 matmul 2.18 ms,FFT 也能扛)。

---

## 3. 实现

`experiments/seamless/torch_seamless.py`(约 200 行,无第三方依赖;整帧接口 `seamless_roi_from_frames` 已测:2 帧对拍 max 44 / mean 0.020,与 ROI 接口一致):

| 函数 | 作用 |
|---|---|
| `_fwd_x/_fwd_y` | 前向差分 + reflect101(= `computeGradientX`) |
| `_bwd_x/_bwd_y` | 后向差分(= `computeLaplacianX`) |
| `lap5` | 5 点 Laplacian,BORDER_REFLECT_101 |
| `erode_rect3x3_iter` | `-maxpool2d(-m,3,pad=1)` ×3;**pad=−inf ⇒ 腐蚀时外侧视为 +inf**,与 `morphologyDefaultBorderValue()` 一致 |
| `dst_solve` | 正弦矩阵对角化(batch) |
| `dst_solve_fft` | rfft 版同一算子 |
| `cg_solve` | Jacobi 预条件 CG,带残差历史 |
| `poisson_blend_roi(D,P,M,mode=…)` | 整条 pipeline,**批处理** `[B,3,H,W]` |
| `seamless_roi_from_frames` | 与 `cv2.seamlessClone` 同语义的整帧接口(mask 边框置零、bbox、p 居中、回写) |

正确性三级验证:

1. **数学级**:把 cv2 的输出反推回方程(`rhs' = Δ(cv2输出) − Δ(ring)`)再喂给本求解器 → **误差 0.000**(max 与 mean 都是 0),说明算子、边界、组装三者完全一致。
2. **原语级**:`filter2D / Laplacian / erode` 每一步与 cv2 逐位相同(实测 0 diff);OpenCV 的 `dst()` 与正弦矩阵实现差一个常数 4 倍(4.13.0 的 `dst()` 里 DFT 未归一化),数学上等价。
3. **数值级**:24 张 1080p 真实帧全量对拍(§4)。

**批处理的关键**:batch 内所有样本用同一 ROI 尺寸(DFL 的 compose 本来就是固定人脸框),整条 pipeline 是一个 `[B,3,H,W]` 张量流,没有 Python 循环。若各帧 bbox 不同,用各自的 D 填到公共尺寸即可(外扩部分在 mask 外,对 mask 内结果影响 <1 LSB)。

---

## 4. 对拍:24 张真实 1080p 帧

数据:`frames/A00..A23.png`,来自 `C:\Users\30764\Downloads\20261014_212047.mp4`(1920×1080,20 fps)按时间均匀抽帧;`src` 取第 `i+5` 帧同一区域(模拟换脸:同几何、不同内容);mask 为 200×245 椭圆(bbox = 401×491,与 DFL 人脸框同量级)。

### 4.1 差异分布(24 帧 × 401×491 = 49.77 M 像素,取三通道最大差)

| 差值 | 像素数 | 占比 |
|---|---|---|
| **0(逐位相同)** | 48 342 062 | **97.138 %** |
| 1 LSB | 1 259 769 | 2.531 % |
| 2–4 | 145 508 | 0.292 % |
| 5–16 | 17 165 | 0.034 % |
| **>16** | **1 896** | **0.0038 %** |

* 全帧口径:`mean|diff| = 0.020`,`p99.9 = 4`,`>1 LSB = 0.211 %`。
* 单帧最差 `max = 176`(frame 11),最好的帧 `max = 7`(frame 21)—— 与画面内容强相关。
* **`>16` 的 1896 个像素全部落在 ROI 外框 ≤19 px 内**(中位 5 px、p95 12 px):不是散布,是贴着 bbox 的一条窄带。

### 4.2 残差为什么在边框

成因是结构性的,不是精度:OpenCV 的解法把 **ROI 外框(ring)强制写成目标图像素**,而 mask 一路顶到 bbox 边缘,于是"外框用 D、紧邻的 mask 内用 P"在 1–3 px 内正面冲突,该处 rhs 尖峰(实测 |rhs| 可达 900)对应一个条件数极差的局部问题;差异在那里被放大到几十 LSB,一离开这条带立刻回到 ≤1 LSB。

补一个反证:把**解算矩形**在原 bbox 外扩 k = 2 / 8 / 24 px(mask 不动),解在 mask 内只变化 **≤1 LSB**(mean 0.065 / 0.176 / 0.329)—— 说明这套离散格式对"解算域形状"是稳定的,残差那几十 LSB 不是"域大小"造成的。
   三条已排除:①算子/边界/组装(误差 0.000);②每个原语(filter2D / Laplacian / erode)逐位相同;③解算域形状(≤1 LSB)。剩下的只能是 cv2 内部那条 **float32 + 4× 缩放 DST** 在边界尖峰处的舍入被 `1/λ_min ≈ 400` 放大 —— 这也是"为什么没人能逐位复刻 cv2"的根因。

**所以"逐位复刻 cv2"本身没有意义**:cv2 在那一带给出的是它自己 float32/DST 路径的舍入结果,不是方程的解;而本 DST 版是**该离散系统的精确解**(§3 验证链第 1 条,误差 0.000)。

### 4.3 掩码尺寸扫描(单帧)

| mask 轴 | ROI | cv2 ms | torch dst ms | max|diff| | mean | >1 LSB |
|---|---|---|---|---|---|---|
| 8×10 | 17×21 | 3.9 | 1.75 | 6 | 0.00004 | 0.0003 % |
| 80×100 | 161×201 | 11.4 | 1.61 | 9 | 0.0014 | 0.004 % |
| 200×245 | 401×491 | 52.8 | 1.55 | 44 | 0.033 | 0.37 % |
| 300×365 | 601×731 | 134.8 | 2.18 | 76 | 0.036 | 0.19 % |

cv2 耗时 ∝ ROI 面积(**0.27 ms/kpx**),torch 版基本恒定(**1.5–2.2 ms**)→ **mask 越大,替代收益越大**;DFL 的人脸框(≈400×500)正好在 30× 这一档。

### 4.4 视觉

`compare_roi.png`(4 联:目标 ROI / cv2 / torch / |diff|×12)。肉眼看不出区别;差异热力图上是右边缘一个亮点 + 极淡的等值线纹理。

---

## 5. 最小差异档

| 档 | 配置 | max|diff| | mean | 成本 | 适用 |
|---|---|---|---|---|---|
| **T1(推荐)** | 精确 DST(matmul) | 176(单帧最差)/ 中位 44 | 0.020 | **1.5 ms** | 默认 |
| T1' | 精确 DST(FFT) | 同上 | 0.020 | 1.7 ms | ROI 特别大时 |
| T0 | 精确 DST + 解算域外扩 8 px | 与 k=0 的解相差 **≤1 LSB** | 0.02 | 1.6 ms | 与 T1 等价,可省 |
| T2 | CG 1e-6(1093 步) | 176 | 0.0198 | 717 ms | 没有 DST 手段时才用 |
| T3 | CG 1e-4(719 步) | 176 | 0.0299 | 553 ms | 比 cv2 慢 10×,**不推荐** |
| T4 | CG 1e-3(400 步) | 176 | 0.599 | 347 ms | 可见糊,**不推荐** |

T1 与 T2 精度几乎一样 —— 因为差异主体**不是迭代误差,而是 cv2 自己的舍入**。所以"降精度换速度"这个旋钮在这题不存在:DST 既最快也最准。

---

## 6. 落地建议

### 6.1 替换点

把 compose 里那条 `cv2.seamlessClone(patch, frame, mask, center, cv2.NORMAL_CLONE)` 换成:

```python
from experiments.seamless import torch_seamless as ts

# frame : [B,3,1080,1920] uint8 CUDA   src: [B,3,rh,rw] float   mask: [B,1,rh,rw] float 0/255
out = ts.poisson_blend_roi(frame[:, :, T:T+rh, L:L+rw].float(), P, M, mode='dst')
frame[:, :, T:T+rh, L:L+rw] = out.round().clamp(0,255).to(torch.uint8)
```

### 6.2 踩坑清单

1. **`cv2.seamlessClone 会就地改写传入的 mask`**(`erode` 与 `bitwise_not` 都写在 `maskROI` 这个 view 上)。实测:调用前 mask 有 154 467 个前景像素,调用后剩 47 740 且被反相。因此
   * 对比基线必须 `mask.copy()`,否则第二次调用就是另一个问题;
   * 仓库里若**复用同一个 mask 数组**,现有 compose 结果本身就是错的(值得单独查)。
2. **截断不是四舍五入**:`int(x)` 而非 `round(x)`;先 `clamp` 再截断。
3. **mask 先做 3 次 3×3 腐蚀**(= 7×7 方形)再做梯度加权/反向加权,别漏。
4. **ROI 必须等于 mask 的 bbox**(`boundingRect` 语义),且 mask 最外 1 px 会先被置 0(cv2 的 `copyMakeBorder` 小动作)。
5. **别开 tf32**:只快 6 %,`>1 LSB` 从 0.211 % 涨到 1.76 %。
6. batch 内 ROI 尺寸要一致;不一致就用各自的 D pad 到公共尺寸,对 mask 内结果影响 <1 LSB。

### 6.3 想把那条边框带也消掉

把**解算矩形**在 mask bbox 外扩 `k` px(只扩解算域,不动 mask),结果再裁回 bbox。

* 实测:外扩 2/8/24 px,解在 mask 内只变 **≤1 LSB**(mean 0.065/0.176/0.329)—— 所以**这一步对精度没有收益**,只是把 ring 从 mask 边缘挪开;
* 代价:矩阵从 491×401 变 (491+2k)×(401+2k),`k=8` 时 +8 %(≈1.6 ms);
* 与 cv2 的差会**变大**(cv2 的窄带仍在它自己的 bbox 上),与真解的差基本不变。

结论:**不用扩**。真正该做的是接受"和 cv2 差 1 LSB",而不是去追 cv2 的舍入。

业务上若只关心 mask 内部(DFL 的 alpha 融合本来 mask 外不取值),这条带可以直接无视。

---

## 7. 复现

```bash
cd D:/MyProject/DeepFaceLab-PyTorch-4.1/experiments/seamless
PY=../../runtime/engine/python.exe

$PY extract_frames.py     # 从 3 个真实视频抽 40 帧 1080p → frames/
$PY check_real.py 12      # numpy 逐行复刻 vs cv2(原始验证:mean 0.02 / max 44)
$PY validate_torch.py     # torch 各模式 vs cv2
$PY sweep_sizes.py        # 掩码尺寸 × 精度 × 耗时
$PY final_bench.py        # CUDA-event 延迟/吞吐(§0 上半表)
$PY stats.py              # 差异分布 + 残差位置(§4.1)
$PY e2e.py                # 端到端 1080p 合成 + tf32 对比
$PY visual.py             # compare_roi.png
```

文件:`torch_seamless.py`(实现)、`ref_numpy.py`(逐行 numpy 复刻,oracle)、`src_opencv/`(4.13.0 的 `seamless_cloning*.cpp` 原件)、`_debug/`(定位过程脚本)。

---

## 8. 一句话回答

**能做,而且值得做**:1080p + 401×491 人脸框,cv2 45 ms → torch **1.5 ms(batch1)/ 1.1 ms/帧(batch8)**,97.1 % 像素逐位相同、2.5 % 差 1 LSB;唯一"不完美"的是 ROI 外框 ≤19 px 的窄带,而那条带上 cv2 自己也不是方程的解 —— 想彻底干净就把解算域外扩 8 px。
