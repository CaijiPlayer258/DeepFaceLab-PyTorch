# GPU 线程逐帧时间线（CUDA event 级）· 40.7ms/帧 去哪了

未 commit / 未 push。**生产代码一行未改**：全部埋点是运行时 monkeypatch，插在真的
「pf._fused_run」里（「experiments/gpu_tl/gputl.py」）——
「pf._dfm_run_batch」（dfm 一次调用，取会话自报的 t_in/t_fwd/t_out/t_gpu 增量）、
「numpy.empty」（「flush()」里那句打包的起点）、「pf._fused_q_u8」（量化）、
「XSegMasker.mask_batch」（遮罩；CUDA event 记在 **TRT 自己那条流** 上，记在调用方流上只会量到一对空事件）。

口径：torch 2.9.1+cu128 · RTX 5070 Laptop 8GB（驱动 591.86）· 「runtime/server」venv ·
素材「workspace/9月29日 (5).mp4」第 200–349 帧（150 帧，1 脸/帧）· dfm「阿証証証証-416」
（bf16w，「DFL_DFM_BACKEND」未设 = torch + bf16w）· xseg「xseglite … int8」引擎 ·
seamless / mask 8 / erode 10 / blur 10 · 合成 CPU 8 线程 · raw 管道 ·
**「batch=1, fused_batch=1」钉死**（不钉的话「_pred_batch_pick」的跨轮滞回会在第 3 轮把一行变成 8 张脸）。

    runtime\server\Scripts\python.exe experiments\gpu_tl\gputl.py probe  --frames 150 --rounds 3 --tag c
    runtime\server\Scripts\python.exe experiments\gpu_tl\gputl.py replay --frames 150 --rounds 3 --tag g
    runtime\server\Scripts\python.exe experiments\gpu_tl\gil_dfm.py

---

## 结论（40.74ms/帧，稳态 127 帧中位）

**一句话**（稳态 127 帧的合计口径，中位在括号里）：每帧 **41.2ms**（中位 40.7）= **GPU 真跑 25.5ms**
（dfm kernel 17.6 + xseg 7.9 = **61.9%**）+ H2D 1.5（3.5%）+ **GPU 等主机提交 3.9**（9.5%）+
**host 不可藏 8.5**（20.7%：出参 2.2 + 出参解析 2.2 + u8 2.9 + 打包 1.2）+ 空等上游 1.1（2.8%）+
记账残差 0.7（1.7%）。
「dfm 多出来的 7~14ms」= **27.4 的调用（中位 26.1）− 17.6 的 kernel = 9.8ms**，**全在 host 侧、每帧一次、
串行**：H2D 1.5 + GPU 等启动 3.9 + 出参 pin/同步/numpy 2.2 + 「_dfm_run_batch」的出参解析 2.2。

**① dfm kernel 到底多久**：静默态**同一次** b1 前向的 event 窗口 **18.0~18.4ms**（三次独立复现：
18.01 / 18.08 / 18.39），生产里同一段窗口 **23.0ms**（中位 21.5）。差的那 3.9ms 不是 kernel 变慢 —— 是 GPU 在等
主机把下一个算子提交进来：156 个 aten 算子的**提交环从 5.0ms（静默）涨到 20.3~22.6ms（跑起来）**。

**② host 能不能藏到 kernel 后面**：分两半。**提交环不能藏**（GPU 收到算子才动，20.3ms 的提交
直接把帧压到 ≥20.3ms）；**能藏**的是「上一帧 D2H 尾巴 + 下一帧打包/H2D/u8/xseg」。实测（同会话
同输入、147 帧 × 3 轮交替）：串行 **25.95** → pinned 双缓冲 + 延迟同步 **24.25** → 再把 xseg 挪到
第二条流 **23.07** ms/帧。**现在确实在等同步**：静默态「t_dsync」13.4~13.9ms/帧（占该次调用 70%），
生产里只剩 0.58ms —— 等的是同一段，只是被记到提交环的账上了。

**③ xseg 与 dfm 两条流**：前提要纠正 —— 它们**不是两张独立的图**。xseg 的 prd 分支吃的是 dfm 输出
（「pipeline_fused.py:1703」「imgs = [f["pred_u8"]]」），只有 dst 分支（「f["dst_u8"]」，对齐后**未换脸**的
dst 脸）与 dfm 无关；且 mask_mode=8 时每张脸 **2 次前向**（引擎 maxbatch=1，「xseg.py:194」逐张走），
单次 ≈3.5ms。所以同帧内最多挪走 1 次（≈3.5ms）；**跨帧流水**（第 N 帧 xseg 跑在第 N+1 帧 dfm 的影子里）
实测 **23.07 vs 25.95 = 省 2.9ms/帧**，上限是 dfm kernel 本身：车道 20.1ms/帧（e 轮 dual）= 49.7 帧/秒。

**④ 「18.65ms 是静默态」单列**：「workspace/_pred_batch_pick.json」里 416-b1 的 **18.65ms** 是阶梯在
**没有别的负载**时量的（我此刻静默复现 18.0~18.4 的 event 窗口、19.5~22.7 的整次调用）；生产里同一次
调用 **27.4ms**（中位 26.1），差 **45%**。台架数不能当生产数用 —— 这就是「台架 18.9、生产 26~33」对不上的原因：
两者量的是**同一个 kernel**，两套不同的 host 环境。

**⑤ 35~40 帧/秒为什么够不着**：GPU 真忙已经 25.5ms（= 39.2 帧/秒，且要求 host 与空等全为 0）。要 35
帧/秒（28.6ms）得把 8.5ms host + 1.1 空等几乎全填进 kernel —— 只靠重叠做不到（提交环 20.3ms 自己就是下限），得先把
「每帧 156 次提交」变少。那属于改 dfm 的调用方式，不在本次「不改后端/精度」的范围，数据留在表里。

**（a）每帧 dfm 调用次数**：147 次调用 / 147 帧，「batch_actual.hist = {1: 147}」→ **每帧正好一次 b1
调用 = 1 次 156 算子图提交 + 1 次流同步**（「_outs」里那唯一一次）。xseg 每帧 1 次「mask_batch」，内部
因引擎 maxbatch=1 拆成 **2 次前向**。

**探针开销**：探针开/关同进程交替 3 轮 —— 19.74/20.52、16.84/20.24、20.49/20.08 fps，无系统性差异
（第 2 轮那对是降频：SM 1672 vs 1845 MHz）。逐帧方差大（p95 55.6ms）且**与谁在抢有关**：同一段代码
在流水线排空的尾部跑到 29.5~34.7ms/帧（表 3 第 136~147 帧），中段 50~60ms。

---