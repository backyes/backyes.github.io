# 论文分析报告 ·《FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling》

> 本报告是对 MLSys 2026 论文《FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling》（OpenReview ID: `mN5RtvuYl3`）的中文深度解读。同目录下的 PDF (`mN5RtvuYl3.pdf`) 是原文。本报告以原文 15 页 PDF 为唯一信息源，所有数字、公式、术语均对照原文，不做编造。

---

## 0. 元数据

| 字段 | 内容 |
| --- | --- |
| 题目 | FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling |
| 作者（按原文顺序） | Ted Zadouri\*, Markus Hoehnerbach\*, Jay Shah\*, Vijay Thakkar, Tri Dao（\* 表示 Equal contribution） |
| 所属单位 | Princeton University CS、Together AI、Meta、Colfax Research、Georgia Tech |
| 通讯作者 | Ted Zadouri (`tz6037@princeton.edu`)、Jay Shah (`jayhshah@colfax-intl.com`)、Tri Dao (`tri@tridao.me`) |
| 会议 / 轨道 | The 8th MLSys Conference, Santa Clara, CA, USA, 2025（论文头版称 2025；OpenReview 归档于 MLSys 2026 ） |
| 页数 | 15 页（含参考文献与附录 A、B） |
| OpenReview ID | `mN5RtvuYl3`（链接：<https://openreview.net/forum?id=mN5RtvuYl3>） |
| 本地 PDF 路径 | `/Users/backyes/Library/Mobile Documents/com~apple~CloudDocs/paper/mlsys2026/mlsys2026_papers/mN5RtvuYl3.pdf` |
| 代码仓库 | <https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute> |
| 实现语言 | CuTe-DSL（嵌入 Python，全 Python 实现，零 C++） |
| 目标硬件 | NVIDIA Blackwell B200 / GB200（datacenter SKU） |
| 主要数据类型 | BF16/FP16（FP8/INT4/FP4 等低精度路线由 SageAttention 系列负责） |

---

## 1. TL;DR（一句话三句话）

- **问题**：Blackwell 一代 GPU 出现"非对称硬件缩放"（asymmetric hardware scaling）——tensor core 吞吐相对 Hopper 翻倍到 2.25 PFLOPS BF16，但 SMEM 带宽（128 B/clk/SM）和 MUFU 指数单元吞吐（16 ops/clk/SM）几乎原地踏步，FlashAttention-3 的 Hopper 优化范式不再最优。
- **方案**：算法-内核协同设计——重做软件流水线吃满全异步 MMA 与 128×128 tile；用 FMA 多项式仿真 `2^x` 把 exp 算力借给 MUFU；引入条件 softmax rescaling；后向利用 256 KB TMEM、2-CTA MMA 模式把 SMEM 流量和 dQ atomic-add 各砍一半；全部用 CuTe-DSL Python 写完。
- **结果**：B200 BF16 上对 cuDNN 9.13 提速 1.1–1.3×，对 Triton 提速 2.1–2.7×，峰值 1613 TFLOPS（理论峰值的 71%）；编译时间相对 FA-3 的 C++ 模板减少 20–30×（前向 55s → 2.5s，后向 45s → 1.4s）。

---

## 2. 问题背景

### 2.1 为什么这件事重要

- Transformer 仍然是几乎一切 AI 应用的主干（LLM、ViT、多模态），attention 在长序列下是 O(N²) 计算瓶颈；长上下文能力（多文档推理、整 codebase 建模、高分辨率视频）的解锁直接受限于 attention kernel 的效率（p.1）。
- 加速器代际更替的"非对称缩放"已经成为一种长期趋势：MMA 单元每代翻倍甚至更多，但 SMEM、寄存器带宽、MUFU、ALU 等"非 matmul"单元的扩张被功耗与面积约束钉死。FlashAttention 系列因此必须每代重写，而不仅是重新调参。

### 2.2 FA-1 → FA-2 → FA-3 → FA-4 的演进脉络

| 版本 | 主要贡献 | 主硬件 |
| --- | --- | --- |
| FlashAttention (Dao et al., 2022) | tile + kernel fusion，消除 GMEM 中间读写 | A100/V100 类 |
| FlashAttention-2 (Dao, 2023) | 序列长度方向并行化，提高 SM occupancy | A100 |
| FlashAttention-3 (Shah et al., 2024) | 利用 Hopper 异步执行、warp specialization、FP8 | H100 |
| **FlashAttention-4** | 针对 Blackwell **非对称硬件缩放**进行算法-内核协同设计 | B200 / GB200 |

旁支路线（低精度量化方向）：SageAttention（INT8）、SageAttention2（INT4/FP8）、SageAttention3（FP4 on Blackwell consumer GPU），但主要面向消费级 GPU；而绝大多数 AI 算力部署在 datacenter GPU 上，这是 FA-4 的目标场景。

### 2.3 已有方案的痛点

- 直接把 FA-3 移植到 Blackwell：要么把性能留在桌面上，要么因为 Hopper MMA 指令在 Blackwell 没有 forward compatibility 而根本跑不起来（`FLASHATTENTION-3 does not run on B200`，p.11 脚注 1）。
- Triton/Gluon/cuDNN 的现有路径要么对 Blackwell 新硬件特性挖掘不够（TMEM、2-CTA MMA），要么是闭源黑盒（cuDNN）。

### 2.4 Blackwell 关键新硬件特性

- **Tensor Memory (TMEM)**：256 KB on-chip per SM，专门存 tensor core 中间结果，warp-synchronous，与 tensor core 紧耦合，MMA 直接写 TMEM 不再吃寄存器，缓解了 Hopper 上的极端寄存器压力，使更大 tile 成为可能。TMEM 以 32 列（16 KB）为最小分配粒度，需要程序员显式管理（p.3）。
- **MMA tile 翻倍**：每条 MMA 指令处理 128×N tile（典型 N=128 或 256），相比 Hopper 的 64×N，单 MMA 面积翻倍。
- **完全异步 MMA**：MMA 直接异步写 TMEM，不再阻塞寄存器写回路径，使 MMA 与其他操作的重叠空间更大。
- **2-CTA MMA 模式**：同一 thread block cluster 内一对 CTA 协同发射一次 MMA，A 与 accumulator 沿 M 划分，B 沿 N 划分，每个 CTA 只在自己的 SMEM 里 staging 一半 B。M 可达 128 或 256（单 CTA 最大 128）。

### 2.5 Bottleneck 转移（roofline 分析的关键结论）

- B200 BF16 MMA：8192 ops/clk/SM（Hopper 的 4096 翻倍）。
- B200 MUFU.EX2：16 ops/clk/SM（与 Hopper 相同；只有 B300/GB300 才翻倍到 32，但论文撰写时尚未铺货）。
- B200 SMEM read：128 B/clk/SM（与 Hopper 相同，按 microbenchmark 测得，引 Luo et al. 2025）。

结论：Blackwell 上 SMEM 流量与 exp 单元成为 attention 的真正瓶颈，超出 MMA compute 时间 25–60%。这把 FA-4 的设计目标从"喂饱 tensor core"改成了"省 SMEM、省 exp、并行其他单元"。

---

## 3. 核心思想 / 方法（最重要章节）

### 3.1 前向 pass 的 roofline 分析（§3.1.1，p.4）

设 Q/K tile 在序列方向的形状为 `M×N`，head dimension 为 `d`。前向有两次 MMA：

- `S = α QKᵀ`（M×d × d×N → M×N，**SS**：双操作数都来自 SMEM）
- `O += P V`（M×N × N×d → M×d，**TS**：A 来自 TMEM，B 来自 SMEM）

每次 MMA 共 `2MNd` FLOPs，tensor core 8192 ops/clk：

```
T_MMA = 4 M N d / 8192   (公式 1, p.4)
```

SMEM 流量：因为单 MMA 指令只能处理 128×128 tile，要算 M×N 输出需要 `⌈M/128⌉ × ⌈N/128⌉` 条指令，每条指令都会重复读一次 Q 和 Kᵀ（操作数无法跨 tile 复用）：

- QKᵀ (SS) 读 `⌈M/128⌉⌈N/128⌉ × 256d` 个 BF16 元素；
- PV (TS) 读 `⌈M/128⌉⌈d/128⌉ × 128N` 个 BF16 元素；

按 2 B/element、128 B/clk 折算：

```
T_smem = 3 M N d / 8192   (公式 2, p.4，假设 M,N,d 均为 128 的倍数)
```

Exp 单元：softmax 要对 `M×N` 个值取 exp，MUFU 16 ops/clk：

```
T_exp = M N / 16        (公式 3, p.4)
```

代入两个常用 tile（Table 1，p.4）：

| Resource | tile = 128³ | tile = 256×128² |
| --- | --- | --- |
| MMA compute | 1024 cycles | 2048 |
| Shared memory | 768 | 1536 |
| Exponential | 1024 | 2048 |

两种配置都呈现 **MMA 与 exp 几乎打平、两者同为主瓶颈**的局面，SMEM 反而略低（这是用 TMEM 把 PV 的 A 操作数下沉的结果）。这驱动 FA-4 同时做：(1) 用大 tile 提高复用并最大化 MMA-softmax 重叠；(2) 用其他硬件单元给 exp 加吞吐；(3) 砍掉非必要的非 matmul 操作。

### 3.2 前向新流水线：让 matmul 与 softmax 真正重叠（§3.1.2，Figure 1, p.5）

FA-4 沿用 FA-3 的 ping-pong：每个 thread block 算两块 output tile（"高 Q tile" `H` 与"低 Q tile" `L`），一块在 tensor core 跑 MMA 的同时，另一块跑 softmax。但 Blackwell 与 Hopper 的根本差异迫使 FA-4 做了以下重构：

1. **Accumulator 在 TMEM 而非寄存器**。Hopper 上 MMA 的 accumulator 用寄存器存（每行 4 个线程交错存放），FA-4 上 accumulator 全部在 TMEM 中。这意味着 softmax 所需的"读一整行"必须显式从 TMEM 加载到寄存器。
2. **128×128 单 tile**（Hopper 是 64×128）。配合两个 warpgroup（每组 128 线程）做 softmax，每个线程一整行——免去了 inter-warp shuffle 算 row-max 的开销，也省去了多份 statistics 寄存器。
3. **两个 softmax warpgroup 显式同步**关键区（exp 计算段）不重叠，与 FA-3 一致。每个 softmax warpgroup 流程：load 整行入寄存器 → row-max → softmax（减 max、rescale、exp、转回输入精度）→ row-sum。
4. **`P` 通过 TMEM 传递**（不像 FA-3 走寄存器）。这带来一个关键松绑：output rescale 可以解耦到一个独立的 "**correction warpgroup**"，从关键路径中拿掉。
5. **TMEM 划分**：head dim 128 时，两个 output tile 占掉一半 TMEM，剩下的另一半要装 S 和 P。可装 2 份 S 或 4 份 P（FP16/BF16 输入精度下）。FA-4 选择 "**2 份 S 与 P 重叠**" 而不是 "1 份 S + 2 份 P"，因为前者允许流水线开局立刻并发计算两个 S tile，并且还能挤出一小块 TMEM 用来给 correction warpgroup 通信 rescale 统计量。
6. **寄存器压力管控**：每个 softmax warpgroup 要常驻 128 个 BF16 输入寄存器 + 64 个输出寄存器 + 杂项/临时寄存器；并存在 4 个 warpgroup（2 softmax + 1 correction + 1 driver/TMA），每 SM 256 寄存器/线程的限制让这块极度紧张。FA-4 把 **存 P 拆成 stage**：前 3/4 一次性存（同时触发对应的 MMA），最后 1/4 单独存——这是个非常工程化的小招但对吞吐贡献明显。

> 图 1 (p.5)（前向 pipeline）：上下文中 H 和 L 上标分别表示两个 Q tile（每个 128 行 token），绿色 MMA、橙色 softmax、蓝色 correction、TMA load 在时间线上交错重叠。

### 3.3 Exp 单元瓶颈：FMA 多项式仿真 + 部分仿真（§3.1.3，p.5–6）

#### 3.3.1 为什么要仿真？

Exp 走 MUFU 单元，B200/GB200 仅 16 ops/clk/SM；softmax 是 exp 大户。把一部分 exp 卸载到 FMA 单元（与 MUFU 并行运行），相当于把"未被使用"的 FMA 算力直接借给 exp，等效提升 exp 总吞吐。

#### 3.3.2 算法：Cody-Waite range reduction + 多项式逼近

利用恒等式（公式 4, p.6）：

```
2^x = 2^⌊x⌋ · 2^(x − ⌊x⌋)
```

整数部分 `2^⌊x⌋` 用 IEEE-754 浮点 exponent 字段位移 + 加法（整数 ALU），分数部分 `x_frac ∈ [0,1)` 用 Horner 方法 + FMA 评估多项式（公式 5, p.6）：

```
2^x_frac ≈ Σ_{i=0..n} p_i · x_frac^i,    p_0 = 1.0
```

`p_i` 由 Sollya 工具（Chevillard et al., 2010）按最小化 `[0,1)` 上相对误差挑选。完整流程：

1. clamp `x ≥ −127`，避免 underflow；
2. 用 round-down 模式算 `⌊x⌋`：先 `x + 2^23 + 2^22` 把分数位强行挤到 mantissa，再减回去；
3. `x_frac = x − ⌊x⌋`；
4. Horner 多项式得 `2^x_frac`；
5. 把 `⌊x⌋` 移位到 exponent 字段，叠加 `2^x_frac` 的 mantissa。

#### 3.3.3 数值精度（Table 2, p.6）

| 方法 | FP32 max rel err | FP32 mean rel err | BF16 max rel err | BF16 mean rel err |
| --- | --- | --- | --- | --- |
| 理想 FP64→BF16 | — | — | 3.89e-3 | 1.41e-3 |
| 硬件 MUFU.EX2 | 1.41e-7 | 3.04e-8 | 3.89e-3 | 1.41e-3 |
| Degree 3 | 8.77e-5 | 5.43e-5 | 3.90e-3 | 1.41e-3 |
| Degree 4 | 3.05e-6 | 1.84e-6 | 3.89e-3 | 1.41e-3 |
| Degree 5 | 1.44e-7 | 5.48e-8 | 3.89e-3 | 1.41e-3 |

关键观察：在 FP32 层面 degree-3 比硬件大约 600× 差，但**一旦 round 到 BF16，BF16 的量化误差（约 3.9e-3）完全淹没多项式误差**——degree ≥ 3 的 BF16 误差与硬件不可区分；degree-3 在 99% 输入上与硬件 BF16 ULP 差不超过 1。Degree-5 把 FP32 max rel err 拉回到硬件 2× 以内，代价是每次评估多 2 条 FMA。

#### 3.3.4 部分仿真（Partial emulation）

完全用仿真会撞上寄存器压力上限（额外存中间值与多项式系数），还会增加寄存器带宽占用与延迟。FA-4 只把每行 softmax 的 **10–25% entry** 走仿真，其余仍走 MUFU.EX2，确切比例按 MMA / exp 吞吐比经验调优。这个比例就是 §B 消融表里 "no e2e" 与默认 kernel 的差距来源。

### 3.4 条件 softmax rescaling（§3.1.4，公式 6, p.7）

FA 在线 softmax 维护：`m_j = max(m_{j-1}, rowmax(S_j))`、`ℓ_j = e^{m_{j-1}-m_j} ℓ_{j-1} + rowsum(e^{S_j-m_j})`，输出更新带 `e^{m_{j-1}-m_j}` 这个 rescale 系数。

FA-4 的两点观察：

1. 只有 `m_j > m_{j-1}` 才真正需要 rescale；
2. 可以容忍 "slack"：只在 `m_j − m_{j-1} > τ` 时 rescale（τ 经验取 `log_2(256)=8.0`，对应 rescale 系数 256.0），其余情况延后 rescale，最终用真正的 `m_final` 与 `ℓ_final` 一次性归一化。

修改后（公式 6, p.7）：

```
若 m_j − m_{j-1} > τ:
    O_j = e^{m_{j-1}−m_j} O_{j-1} + e^{S_j − m_j} V_j
否则:
    O_j = O_{j-1} + e^{S_j − m_{j-1}} V_j   (维持 m_{j-1} 不变)
```

最终 `Output = O_final / ℓ_final`，由真正的 max/normalizer 修正。为避免 warp divergence，**只要 warp 内任一线程需要 rescale，整个 warp 都 rescale**。这一步显著减少了 rescale 调用次数，几乎不损精度。

> Ablation 印证：Table 11/12（p.14–15，"Always correction rescale"）显示长序列下吞吐相比默认 kernel 掉 ~15% 左右（1290 vs 1379 TFLOPS @ seqlen 2048 non-causal），证明条件 rescale 的实际收益。

### 3.5 后向 pass 的 roofline（§3.2.1，Table 3, p.8）

后向 5 次 MMA（recompute S、+ QK 与 PV 各自的两个梯度）：`S^T = K Qᵀ`、`dPᵀ = V dOᵀ`、`dV = Pᵀ dO`、`dK = dSᵀ Q`、`dQ = dS K`。其中 SS 三个、TS 两个。

```
T_MMA = 10 M N d / 8192          (公式 7, p.7)
T_smem,MMA = (4Md + 3Nd + MN) / 64 (公式 8, p.7)
T_smem 全部 = 上式 + MN/64 (dS 写) + Md/16 (dQ 写读) (公式 9, p.7)
T_exp = MN / 16                   (公式 10, p.7)
```

Table 3 对 `M=N=d=128`（1-CTA）/ `M=256, N=d=128`（2-CTA）两种配置：

| Resource (cycles) | 1-CTA M=128 | 2-CTA M=256 |
| --- | --- | --- |
| MMA compute | 2560 | 2560 |
| SMEM (MMA operands) | 2048 | 1536 |
| SMEM (dS write) | 256 | 256 |
| SMEM (dS DSMEM) | 0 | 384 |
| SMEM (dQ write+read) | 1024 | 512 |
| **Total SMEM** | **3328** | **2688** |
| Exponential | 1024 | 1024 |

后向 SMEM 是绝对瓶颈，比 MMA 多 ~30%（1-CTA），2-CTA 模式后压到 ~5%——**这一项就是 2-CTA 的设计动机**。

### 3.6 后向新流水线（§3.2.2，Figure 2, p.8）

FA-3 后向因为 accumulator 在寄存器、寄存器又紧张，本质上把计算图串行化（`S → dP → dV → dQ → dK`，TMA load 是唯一显著乱序的部分）。FA-4 借助 TMEM 实现了之前不可能的调度：

- 前向只需要让 softmax 与 dP MMA 重叠；后向 Blackwell 上至少需要"两个 MMA 同时跑"才能藏住 softmax 延迟。FA-4 用**前一次迭代的 `dQ` 与 `dK` MMA 与本次迭代的 softmax/dS 计算重叠**。
- TMEM 容量上**只能容下 4 个 128×128 accumulator tile**（不够 5 个）。dV 与 dK 必须独占（要全程累积），所以 FA-4 让 **S 与 P 共享一块 TMEM（offset 0）**，**dP、dS、dQ 共享另一块**。
- 图 2（p.8）展示了 1-CTA 模式后向的 prologue / main loop / tail 三段调度。

### 3.7 2-CTA 后向：减少 SMEM 流量、减少 atomic-add（§3.2.3，Figure 3, p.9）

#### 3.7.1 SMEM 流量减半

2-CTA MMA 让 CTA 对沿 M 划分输出 accumulator，每个 CTA 只 staging 一半 B；在 5 个 GEMM 中绝大多数（S、dP、dV、dK）使用 `M=256, N=K=128` 的 tile，operand B 流量近乎砍半。

#### 3.7.2 dQ 的难题与 DSMEM 解决方案

FlashAttention 后向把外层并行化到 N 个 KV CTA，内层流式扫 M tile；**`dQ` 的累加是沿 KV 序列方向的 reduction，落在外循环上**。但 2-CTA MMA 只切输出 tile（M），并不切 reduction 轴；`dQ` 的 reduction 维度恰恰是 N，即 CTA 对天然分掉的那一维——结果每个 CTA 仍然需要它所拥有那些行的完整 reduction。

FA-4 的破法：**用 DSMEM（distributed shared memory，cluster 内 CTA 间共享）交换一半 dS**，把 dS 沿非 reduction 轴重新分块，每个 CTA 拥有 M/2 行但持有完整 2N reduction。结果：

- S/dP/dV/dK 的 MMA 用 `M=256`；
- `dQ` 单独用 `M=128, 2N=256`（即每 CTA 算 (M/2, d) 的 dQ tile）。

为了藏住 DSMEM 延迟，FA-4 重排了 1-CTA 时的软件流水：**先算当前 tile 的 dP，再算上一 tile 的 dQ**；dQ tile 小到能塞进 TMEM 与 P 共驻一个 region（共享 S 的那块 TMEM），不再像 1-CTA 那样让 dP/dQ 共用同一 TMEM region。这样**当前 tile 的 elementwise dS 与上一 tile 的 dQ MMA 能够并行**（图 3, p.9）。

#### 3.7.3 dQ 的 atomic-add 减半

副产品：每个 CTA 现在只写 dQ 的一半，**全局 atomic reduction 的次数减半**。atomic 既引入非确定性又昂贵（每次内层迭代都要做），减半带来明显的吞吐收益。

### 3.8 Deterministic backward（§3.2.4，p.9）

后向因为 dQ（以及 GQA 下的 dK/dV）跨 CTA 在 GMEM 做 reduction，引入非确定性。RL training 等场景需要可复现，FA-4 提供 deterministic 模式：用 semaphore lock 串行化 reduction，每个写同一 dQ tile 的 CTA 按预设顺序拿锁、reduce、释放（incr counter）。

性能影响来自两点：(1) 用 memory fence 实现 acquire-release 语义；(2) CTA 等待前序 CTA 完成同一 tile reduction 时的 stall。在 load-imbalanced 场景下乱序选择会大幅劣化性能。FA-4 的策略：

- 一般情况下沿 head/batch 维度做 CTA swizzling（吃满 L2，对应 §3.3）；
- 因果 mask 时另加：**KV block 倒序发射 + Q block 从对角线开始正序遍历 + dQ reductions 按 Q block 倒序**——这是经典的 **Shortest-Processing-Time-First (SPT)** 调度，确保没有 CTA 在第一次 dQ 写时被 stall。

> 实测（图 7, p.12）：deterministic 后向在 causal head dim 128 上能达到 nondeterministic 1-CTA 后向的 ~75% 速度——已经相当接近。

### 3.9 调度（§3.3，p.9–10）

attention kernel 在 causal mask 或 varlen 场景下天然 load-imbalanced：不同 worktile 的 mainloop 长度不一。FA-4 使用经典 **Longest-Processing-Time-First (LPT)** 调度（Graham 1969），适用所有 GPU 架构（在 H100 上对 FA-3 也得到验证）。

#### 3.9.1 Causal mask 的 LPT

朴素网格 `(mblocks, heads, batches)` 左到右递增；causal mask 让对角线以上被掩掉，等于让 SM 从短到长处理 worktile（最差）。但**纯 LPT 也次优**——跨 batch 时 mainloop KV load 在 L2 失效；先加载所有 KV head 又会冲爆 L2。

FA-4 的折中：**始终把 batch 放最外层，按 head 做 swizzle**：head 切成不超 L2 的 section；scheduler 遍历顺序为 head（per section） → mblocks（reverse） → sections → batches。MQA/GQA 中始终在变 mblocks 之前先遍历完一个 KV head 的全部 query head。BF16 head dim 128：MHA +4–8% FLOPS、MQA-8 +7–14%（H200 测得）。

#### 3.9.2 Varlen 的 LPT

varlen 中 query/KV 长度由 device 上 metadata 给定。原始 batch 顺序可能极差（短 prefill → 长 context decode）。FA-4 的解法：**预处理 kernel 排序 batch（按 worktile 最大执行时间），输出 virtual→actual batch index 映射**，主 attention kernel 据此遍历；metadata 可缓存，零额外开销。

---

## 4. 实现 / 工程细节

### 4.1 CuTe-DSL（§4，p.10–11）

FA-4 是 FlashAttention 系列**第一次完全用 Python 写**：CuTe-DSL（NVIDIA 2025）embed 在 Python，编译路径为 `Python → CuTe-DSL compiler → PTX → ptxas → SASS`，没有任何 CUDA C++ 组件。

**与 CUTLASS C++ 编程模型同构**：保留全部低层表达力，PTX 作为 escape hatch（开发者可手写 PTX 补 DSL 尚未暴露的功能；FA-4 自身就用了若干 custom PTX sequence）。

**编译时间巨幅缩短**（Table 4, p.11）：

| 方法 | 前向 | 后向 |
| --- | --- | --- |
| FA-3 (C++ template) | 55s | 45s |
| FA-4 (CuTe-DSL) | 2.5s | 1.4s |
| 加速比 | 22× | 32× |

要知道 FA-2/FA-3 的发布需要预编译数百个 kernel（不同 head dim、causal/non-causal、各种 dtype 组合）；编译时间从一个 kernel 1 分钟降到 1–2 秒，等于把 kernel 设计-调优的 inner loop 缩短一个数量级，对开发者生产力是质变。

社区已经有人基于 FA-4 框架原样写出 FlexAttention 与 block-sparse 变体而不改核心；这是 CuTe-DSL 模块化的有力佐证。

### 4.2 寄存器与 SMEM 划分（前向）

- **4 个 warpgroup**：2 softmax、1 correction（rescale 输出与 statistics）、1 driver/TMA（驱动 tensor core 与 TMA 单元）；
- 每线程 256 寄存器是硬上限；softmax warpgroup 各自需要 ~128（输入）+ ~64（输出）+ 杂项；
- P 分阶段写（前 3/4 一次写、最后 1/4 分写）以缓解 register spill。

### 4.3 TMEM 划分（前向）

- 一半 TMEM 给两个 output accumulator tile；
- 另一半放：2 份 S（与 P 重叠）+ rescale statistics 通信缓冲；
- 这种"S 双份覆盖 P"布局允许流水开局立刻并行算两个 S，且为 correction warpgroup 的解耦留出空间。

### 4.4 TMEM 划分（后向）

- 4 个 128×128 tile 总额度；
- dV、dK 必须独占（全程累积）；
- S/P 共享 offset 0；dP/dS/dQ 共享另一块（1-CTA）；
- 2-CTA 模式下 dQ 与 P 共驻 S 那块 region（因为 dQ tile 减半到 (M/2, d)）。

### 4.5 与 ThunderKittens / cuDNN attention / FA-3 的差异

| 维度 | FA-3 (Hopper) | cuDNN (闭源 vendor) | ThunderKittens (Stanford 风格) | **FA-4** |
| --- | --- | --- | --- | --- |
| 主硬件 | H100 | H100/B200 | H100 系列 | B200/GB200 |
| Accumulator | 寄存器 | — | 寄存器 | **TMEM** |
| MMA 异步度 | 部分异步 | — | 部分异步 | **全异步**，写 TMEM |
| 2-CTA 利用 | ✗ | 部分 | ✗ | **✓**（5 GEMM 中 4 用 M=256） |
| Exp 仿真 | ✗ | ✗ | ✗ | **✓** FMA 多项式（部分仿真 10–25%） |
| 条件 rescale | ✗ | ✗ | ✗ | **✓** τ=log₂(256) |
| Deterministic | 有但慢 | — | — | **✓** SPT + LPT，可达 75% nondet |
| 实现语言 | C++ template (CUTLASS) | C++ 闭源 | C++ template | **Python (CuTe-DSL)** |
| 编译时间 | 数十秒/kernel | — | 类似 FA-3 | 1–3 秒/kernel |

cuDNN 9.13 与 9.14 起已与 cuDNN 团队合作把 FA-4 的部分技术 incorporate 进 cuDNN（附录 A.1, p.13），这意味着 FA-4 的影响已经直接渗透到 vendor 库。

### 4.6 Benchmark 系统

附录 A.1 (p.13)：B200 180GB SXM6（1000W），CUDA 13.1、FA 2.8.3、Triton 3.6、PyTorch 2.10.0、CuTe-DSL 4.4.1，主对比 cuDNN 9.13（同时报告 9.19.1.2）。Warmup 5 次、benchmark 重复 10 次取平均。

---

## 5. 评测

### 5.1 基线与设置

- 对比对象：PyTorch 原生、FlashAttention-2（FA-3 不能跑 B200）、Triton（带 B200 专用指令）、Gluon（比 Triton 更底层的 GPU 编程语言）、cuDNN（vendor 优化）。
- 配置：B200，BF16，causal/non-causal，head dim 64 / 128 / (192, 128)（DeepSeek V3 形状）；seqlen 1k–32k 扫描；batch 调到 token 总数 32k；hidden=2048，head=32（dim=64）或 16（dim=128）。
- FLOPs 约定：前向 `4 · seqlen² · headdim · nheads`；causal 除以 2；后向是前向 ×2.5（前向 2 个 matmul、后向 5 个 matmul，含重计算）。

### 5.2 前向吞吐（Figure 4, 6, p.10, 12）

- **non-causal head dim 128**（图 4 左, p.10）：FA-4 比 cuDNN 9.13 快 1.1–1.3×、比 Triton 快 2.1–2.7×；4k 及更长序列上对所有 baseline 一致领先；
- **causal head dim 128**（图 4 右）：增益更大（attribute to LPT 调度器）；
- **DeepSeek V3 形状 (192, 128) causal**（图 6, p.12）：相对 cuDNN 1.1–1.3×。

附录 B 默认 kernel 表（Table 5/6, p.13–14）：

| seqlen × batch | causal=False (TFLOPS) | causal=True (TFLOPS) |
| --- | --- | --- |
| 2048×16 | 1379.2 | 1033.3 |
| 4096×8 | 1466.0 | 1260.6 |
| 8192×4 | 1507.5 | 1401.0 |
| 16384×2 | 1561.0 | 1496.4 |
| 32768×1 | 1563.7 | 1544.6 |

正文摘要给出**最高 1613 TFLOPs/s（71% B200 BF16 理论峰值 2250 TFLOPS）**——这是 attention 类 kernel 在 Blackwell 上目前公开可见的最高利用率。

### 5.3 后向（Figure 5, 7, p.10–12）

后向（图 5）在长序列与 causal mask 下都有一致 speedup，验证 2-CTA 后向设计的正确性。

Deterministic backward（图 7, p.12 与图 8, p.15）：

- 默认实现 vs. naive（无 batch/head swizzle）vs. LPT（无 reverse mblock）vs. SPT（reverse mblock + descending dQ reduction）；
- SPT 最佳，达到 nondeterministic 1-CTA 后向的 ~75%。

### 5.4 关键消融（附录 B，p.13–15）

#### 5.4.1 q-stage = 1（关流水）

| seqlen | causal=False | causal=True |
| --- | --- | --- |
| 2048 | 993.5 vs 1379.2 (–28%) | 677.1 vs 1033.3 (–34%) |
| 32768 | 1013.2 vs 1563.7 (–35%) | 989.3 vs 1544.6 (–36%) |

→ 流水线（q-stage > 1，即 ping-pong）贡献巨大，关闭后吞吐掉 ~28–36%。

#### 5.4.2 关 exp 仿真（"No e2e"，仅 MUFU.EX2）

| seqlen | non-causal | causal |
| --- | --- | --- |
| 2048 | 1333.8 vs 1379.2 (–3%) | 1022.6 vs 1033.3 (–1%) |
| 8192 | 1428.9 vs 1507.5 (–5%) | 1380.9 vs 1401.0 (–1.4%) |
| 32768 | 1360.7 vs 1563.7 (–13%) | 1501.4 vs 1544.6 (–3%) |

→ exp 仿真在长序列、non-causal 下贡献最大（接近 13%）；causal 因为掩码减半本来就不卡 exp，仿真增益较小。

#### 5.4.3 关条件 rescale（"Always correction rescale"）

| seqlen | non-causal | causal |
| --- | --- | --- |
| 2048 | 1290.0 vs 1379.2 (–6%) | 964.4 vs 1033.3 (–7%) |
| 32768 | 1288.0 vs 1563.7 (–18%) | 1372.0 vs 1544.6 (–11%) |

→ 条件 rescale 在长序列收益巨大（最高 –18%）。

### 5.5 编译时间（Table 4, p.11）

FA-3 单 kernel 前向 55s / 后向 45s；FA-4 前向 2.5s / 后向 1.4s——前向 22×、后向 32× 加速。考虑 FA-2/FA-3 通常需要预编译数百 kernel（不同 head dim、causal、varlen、dtype 组合），这等于把 release pipeline 从数小时压缩到 5 分钟级。

---

## 6. 思想精读 / 启示

1. **算法-硬件协同设计已不是口号，而是必要条件**。Blackwell 把"非对称缩放"做到极致：MMA 翻倍、MUFU/SMEM 不动、TMEM 横空出世。如果继续把 GPU 当成一块"统一算力"，就会把 30%+ 性能留在桌面上。FA-4 证明了：每代 GPU 都需要重新"读硬件"，从 roofline 出发设计算法，而不是反过来。
2. **借用空闲单元**是面对 bottleneck 的最直接武器。Exp 走 MUFU 慢 → 借 FMA 写多项式仿真，等于把 exp 算力扩了几倍；这一招的意义远超 attention，凡是涉及 softmax / GeLU / SiLU 的 kernel 都可以套用。
3. **精度的"语境化"评估**：FP32 误差差 600× 听起来灾难，但在 BF16 下游中完全被量化误差淹没——FA-4 的 degree-3 多项式仿真就是这个原则的优秀应用。**别用 absolute precision 决定算法选型，要用 end-to-end precision**。
4. **TMEM 改变了 kernel 设计的自由度**。Hopper 时代寄存器是核心稀缺资源，FA-3 的整个调度本质上是在围绕寄存器约束作战；TMEM 把 accumulator 移出寄存器后，warp specialization 的边界、调度的并行度都能重新放开。这套思路对 Blackwell 上其他 kernel（如 GEMM-bias-activation 融合、MoE routing）同样适用。
5. **2-CTA 模式不仅是"两块 SMEM 凑一块"**：它真正的力量在于让你**重新规划 reduction 轴**，把全局 atomic 砍半。FA-4 的 dQ 重构（用 DSMEM 交换 dS、按非 reduction 轴重新切）是 2-CTA 用法的范例，未来 cluster-aware kernel 可借鉴。
6. **Skip vs. always**：条件 rescale 是 "lazy semantics" 在 GPU kernel 上的一个干净应用——以最终归一化吸收中间松弛，省掉绝大部分 vector multiply。GPU kernel 设计中"延迟生效"的思路应该被更广泛使用。
7. **Python/DSL 不是 toy**。FA-4 全 Python 实现，性能对标 cuDNN，编译快 22–32×。这意味着未来 kernel 开发的"门槛/产能曲线"会继续向 DSL 倾斜，C++ 模板元编程作为唯一选项的时代结束了。
8. **调度问题就是经典并行调度问题**。LPT、SPT、makespan 都是 Graham 1969 时代的结果，FA-4 在 attention kernel 里直接复用——研究者不要重新发明轮子，要善用经典调度文献。

---

## 7. 局限与开放问题

1. **硬件锁定**：FA-4 的设计深度耦合 Blackwell 的 TMEM、2-CTA、128×128 MMA。除了通用调度（LPT/SPT）能在 Hopper 上回流（4–14%），其他大部分技术（TMEM 划分、2-CTA dQ 重构、emulation 比例）都需要重新校准甚至重写，不能 free lunch。
2. **MUFU 不会永远紧张**：B300/GB300 已把 MUFU 翻倍到 32 ops/clk/SM，仿真带来的相对收益会缩小（虽然 FA-4 默认配置下也只有 5–13%）。是否仍需要仿真路径取决于未来代际继续走的"非对称"程度。
3. **数值精度权衡偏激进**：degree-3 仿真依赖于"BF16 量化掩盖一切"；如果训练精度路线转回 FP32-master 或 deterministic 数值再现要求严格场景，emulation 可能要换 degree-5 甚至 degree-6（每次评估多 2–4 条 FMA），收益变薄。
4. **Deterministic 仍有 25% 损失**：可复现是 RL/科研刚需，但 75% 速度比 nondet 还有显著缺口；semaphore + 顺序 reduction 这条路是否还有更好替代（例如 cluster-level 内 reduction tree）值得探索。
5. **未涵盖低精度 / 量化路线**：FA-4 主打 BF16/FP16，FP8/FP4 路线由 SageAttention3 等工作覆盖。两者的合流——Blackwell 上的低精度 attention 同时具备 FA-4 的 pipeline 优化——尚未发表。
6. **CuTe-DSL 仍在演化**：FA-4 对部分功能不得不写 PTX 补丁。DSL 表达力 vs. PTX escape hatch 的边界一旦发生变化，性能复刻成本会随之变化。
7. **CTA 调度的 LPT 排序需要 metadata pre-process kernel**：增加了一层 launcher 复杂度，对小 batch / 极短序列可能反而是 overhead；论文未量化这些 corner case。
8. **Roofline 只考虑了 tensor core / SMEM / MUFU 三资源**，没考虑 L2、寄存器带宽、HBM 拥塞等次要瓶颈（论文 §3.1.1 自承）；在某些非常规形状（极窄 head dim、超长 seqlen 单 batch）下结论可能要重做。
9. **附录 B 消融并未覆盖 2-CTA 后向 vs. 1-CTA 后向**的细粒度拆解，读者只能从 Table 3（roofline 估算）反推。
10. **Open source 与 PyTorch/Megatron 集成尚在进行**：论文承诺会集成但截稿时尚未完成（"are working to integrate"）。

---

## 8. 关键术语速查表

| 术语 | 含义 |
| --- | --- |
| **MMA** | Matrix Multiply-Accumulate；tensor core 的核心指令 |
| **TMEM** | Tensor Memory，Blackwell 新增 256 KB/SM 片上存储，专放 tensor core 中间结果，warp-synchronous |
| **SMEM** | Shared Memory，per-CTA 片上 banked cache，B200 read 128 B/clk/SM |
| **DSMEM** | Distributed Shared Memory，cluster 内 CTA 间互访的 SMEM |
| **MUFU** | Multi-Function Unit，硬件 exp/log/sin/recip 单元，B200 16 ops/clk/SM |
| **MUFU.EX2** | MUFU 的 base-2 exponential 指令 |
| **FMA** | Fused Multiply-Add，FP 算力主单元，被借来仿真 exp |
| **TMA** | Tensor Memory Accelerator，Hopper 起的异步 DMA 引擎，搬 GMEM ↔ SMEM |
| **CTA** | Cooperative Thread Array，等同 thread block |
| **2-CTA MMA** | Blackwell 新增模式，CTA 对协同执行一次 MMA，B 操作数沿 N 切，每 CTA stage 一半 |
| **Cluster** | thread block cluster，多 CTA 共驻 GPC，可访 DSMEM |
| **Warp** | 32 线程；**Warpgroup** = 4 warp = 128 线程 |
| **Warp specialization** | producer/consumer 模式，不同 warp 只发 load 或只发 compute |
| **Warp-synchronous** | 数据可见性以 warp 为粒度（与 thread/CTA 区分） |
| **Roofline** | 计算 vs. 带宽的上界分析框架，决定 kernel 是 compute-bound 还是 mem-bound |
| **Online softmax** | 流式维护 running max/normalizer 的 softmax 实现 |
| **Conditional rescale** | 仅当 m_j − m_{j-1} > τ 才更新 running max，否则延后归一化 |
| **Cody-Waite range reduction** | 把 `2^x` 拆成 `2^⌊x⌋ · 2^{x_frac}` 的经典数值技巧 |
| **Sollya** | 数值库，按最小化相对误差挑多项式系数 |
| **LPT** | Longest-Processing-Time-First scheduling（Graham 1969） |
| **SPT** | Shortest-Processing-Time-First scheduling |
| **CuTe-DSL** | NVIDIA CUTLASS 风格的 Python embedded DSL |
| **PTX/SASS** | NVIDIA 中间汇编 / 最终机器码 |
| **GQA / MQA** | Grouped / Multi-Query Attention，多 query head 共享少数 KV head |
| **varlen** | variable sequence length，混合 batch 不同长度 |

---

## 9. 关键页码索引

| 引用 | 位置 | 说明 |
| --- | --- | --- |
| 摘要 | p.1 | 全文核心数字：1.3× vs cuDNN 9.13、2.7× vs Triton、1613 TFLOPS、71%、20–30× 编译加速 |
| §1 Introduction 四点贡献 | p.2 | (1) redesigned pipeline；(2) exp emulation + conditional rescale；(3) TMEM + 2-CTA + atomic-add halving；(4) scheduling/register allocation |
| §2.1 公式（attention 前/后向）| p.2 | `S = αQKᵀ`、`O = PV`、`dV = PᵀdO`、`dQ = αdSK`、`dK = αdSᵀQ` |
| §2.2 Blackwell 硬件特性 | p.3 | TMEM、128×N MMA、2-CTA、bottleneck shifting；MMA 8192 ops/clk/SM、MUFU 16、SMEM 128 B/clk/SM |
| §3.1 公式 1–3 + Table 1 | p.4 | 前向 roofline；tile 128³ 与 256×128² |
| §3.1.2 Figure 1 | p.5 | 前向 ping-pong pipeline，H/L Q tile |
| §3.1.3 公式 4–5 + Table 2 | p.6 | exp 仿真：`2^x = 2^⌊x⌋·2^{x_frac}`；degree 3/4/5 精度对比 |
| §3.1.4 公式 6 | p.7 | 条件 rescale，τ = log₂(256) = 8.0 |
| §3.2.1 公式 7–10 + Table 3 | p.7–8 | 后向 roofline；1-CTA M=128 SMEM 3328 cycles vs 2-CTA M=256 SMEM 2688 |
| §3.2.2 Figure 2 | p.8 | 后向 1-CTA 计算图（5 MMA + 2 elementwise） |
| §3.2.3 Figure 3 | p.9 | 2-CTA 后向 dQ 步骤分解（DSMEM 交换 dS） |
| §3.2.4 deterministic backward | p.9 | semaphore lock、SPT 调度 |
| §3.3 LPT 调度 | p.9–10 | causal 收益 4–8% MHA / 7–14% MQA-8 |
| §4 + Table 4 | p.10–11 | CuTe-DSL；FA-3 vs FA-4 编译时间 55s→2.5s（前向）/ 45s→1.4s（后向） |
| §5 Figure 4 | p.10 | 前向 head dim 128 TFLOPS（causal/non-causal） |
| §5 Figure 5 | p.10 | 后向 head dim 128 TFLOPS |
| §5 Figure 6 | p.12 | 前向 head dim (192, 128) causal（DeepSeek V3） |
| §5 Figure 7 | p.12 | deterministic backward 消融 |
| 附录 A.1 系统配置 | p.13 | B200 180GB SXM6 1000W、CUDA 13.1、CuTe-DSL 4.4.1 |
| 附录 B Table 5–10 | p.13–14 | default kernel / q-stage 1 / no e2e 三组消融 |
| 附录 B Table 11–12 | p.14–15 | always correction rescale 消融 |
| 附录 B Figure 8 | p.15 | non-causal deterministic backward |

---

## 10. 一句话点评

> **FlashAttention-4 把"非对称硬件缩放"从一句吐槽变成了一套可执行的协同设计方法论：当 tensor core 翻倍而 MUFU/SMEM 原地踏步时，真正的工程不是更快地塞 MMA，而是把 MMA 偷的算力还给 softmax、把 atomic 还给 cluster、把寄存器压力还给 TMEM——再用 Python DSL 把整个 release cycle 砍掉一个数量级。**
