# 论文分析报告 ·《HipKittens: Fast and Furious AMD Kernels》

> 本报告基于 MLSys 2026 录用论文 `xxSSrndQrI.pdf` 的逐页精读，对 Stanford Hazy Research 团队联合 AMD 提出的 HIPKITTENS（简称 HK）框架进行详细技术拆解。报告以中文为主，AMD/GPU 关键术语（CDNA、wavefront、LDS、MFMA、VGPR、AGPR、XCD、LLC 等）保留英文，并按要求给出 10 节结构。引用页码均对应论文 PDF 中的物理页（页码 1–13）。

---

## 0. 元数据

| 字段 | 内容 |
| --- | --- |
| 题目 | HipKittens: Fast and Furious AMD Kernels |
| 作者 | William Hu, Drew Wadsworth, Sean Siddens, Stanley Winata, Daniel Y. Fu, Ryan Swann, Muhammad Osama, Christopher Ré, Simran Arora |
| 单位 | Stanford University (Hazy Research / 计算机系) · Advanced Micro Devices (AMD) · UC San Diego |
| 通讯作者 | William Hu `<willhu@stanford.edu>` · Simran Arora `<simarora@stanford.edu>` |
| 会议 / 轨道 | The 9th MLSys Conference (MLSys 2026), Bellevue, WA, USA — Main Track（系统/编译器/Kernel） |
| 总页数 | 13 页（正文 + 参考文献，附录 A–E 在仓库与 OpenReview 补充材料中） |
| OpenReview ID | `xxSSrndQrI` — https://openreview.net/forum?id=xxSSrndQrI |
| 本地 PDF 路径 | `/Users/backyes/Library/Mobile Documents/com~apple~CloudDocs/paper/mlsys2026/mlsys2026_papers/xxSSrndQrI.pdf` |
| 代码仓库 | https://github.com/HazyResearch/HipKittens（已在 AMD AITER 中产品化）|
| 评测硬件 | AMD MI325X (CDNA3) · AMD MI355X (CDNA4) · 对比基线运行于 NVIDIA B200 |
| 软件环境 | ROCm 7.0（`rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta`）|
| 主要基线 | AITER（AMD 手写汇编）· Composable Kernels (CK)· hipBLASLt · ROCm Triton · Mojo · PyTorch SDPA / compiled |
| 关键缩写 | HK = HIPKITTENS · TK = THUNDERKITTENS · MFMA = Matrix Fused Multiply-Add · LDS = Local Data Share（共享内存）· XCD = Accelerator Complex Die · CDNA = Compute DNA |

---

## 1. TL;DR

HIPKITTENS（HK）是首个面向 AMD CDNA3/CDNA4 GPU 的 **C++ 嵌入式、tile-based、PyTorch 风格** 高性能 AI Kernel DSL。论文的核心论断与贡献可总结为以下五点：

1. **"Tile 抽象是跨厂商可迁移的"**：在 NVIDIA Hopper/Blackwell 上验证有效的 ThunderKittens (TK) 三件套——tile 数据结构、bulk 算子（mma/exp/add 等）、grid scheduling——在 AMD GPU 上同样适用，无需推翻重来。HK 直接继承 TK 前端语义。
2. **"实例化层完全不同"**：尽管前端 API 一致，HK 必须为 AMD 重新设计三件事：(a) **register 显式 pin** 绕过 HIPCC 编译器对 AGPR 的限制；(b) 多种 MFMA 形状下的 LDS swizzle 算法（AMD 矩阵 layout 不像 NVIDIA 那样有 16×16 core matrix 的可组合结构）；(c) **8-WAVE PING-PONG / 4-WAVE INTERLEAVE** 两种调度范式，取代 NVIDIA 流行的 producer-consumer (wave specialization)。
3. **"Wave specialization 在 AMD 上不工作"**：因为 AMD 静态 register 分配 + 缺少 TMA / wgmma / mbarrier，producer wave 会消耗寄存器却不参与计算，导致每个 thread block 的 output tile 受限，arithmetic intensity 上不去；MI355X 上 wave specialization 仅取得峰值 BF16 GEMM 80% 的性能（页 2、6 Table 2）。
4. **"Chiplet/L2/LLC 联合调度"**：MI355X 有 8 个 XCD（每个 32 CU + 4MB 私有 L2），上面挂一个共享 LLC。论文给出 Algorithm 1 的 XCD swizzle 同时优化 L2 与 LLC 命中率，把行优先 36% L2 命中提升到 ~78%，整体带宽提升 19%（页 8 Table 4）。
5. **"实证：与汇编打平甚至超越"**：在 BF16/FP8 GEMM、MHA/GQA forward&backward、RoPE、LayerNorm 等十多个 workload 上，HK 与 AMD 手写汇编 AITER 持平或更快；在 d=64 attention、GQA non-causal backward、memory-bound 算子上比所有可用基线快 1.2×–10×；比 Triton-AMD GEMM 快 1.3×–3×；用 HK kernel 可成功预训练 Llama 1B 与 BERT 110M 至与 PyTorch/AITER 同等 perplexity（页 9–10）。

一句话：**HK 用一套 ~500 行 C++ DSL，证明了"统一 tile 编程模型 + AMD 专属调度算法"足以击败 AMD 顶尖工程师手写的汇编 kernel，从而打破"CUDA moat"**。

---

## 2. 问题背景

### 2.1 AMD 软件栈痛点：硬件领先，软件落后

论文在引言（页 1）与 Section 2（页 3–4）开门见山地指出当前 AI 算力市场的核心矛盾：

- **算力/带宽：AMD 已经反超 NVIDIA**。Table 2（页 3）给出：
  - BF16 矩阵：B200 = 2.2 PFLOPS vs MI355X = 2.5 PFLOPS
  - MXFP6：B200 = 4.5 PFLOPS vs MI355X = **10.1 PFLOPS**
  - MXFP4：B200 = 9.0 PFLOPS vs MI355X = 10.1 PFLOPS
  - 内存容量：180 GB vs **288 GB**
  - HBM 带宽：8 TB/s vs 8 TB/s（持平）
- **但软件难用**：作者称之为"hardware lottery / CUDA moat"（引用 Sara Hooker 2021；SemiAnalysis 2024）。AMD 上达到峰值性能的 kernel 几乎必须**手写汇编**——典型代表是 AITER 库（页 1 第 35 行）。手写汇编的代价是无法横向铺开到所有 AI workload：在 MI355X 上，AITER 与 PyTorch 的 Llama GQA backward 分别只达到 SoTA 的 30% 与 24%（页 1 第 38–39 行）。
- **NVIDIA 也曾经走过这条路**：从 H100 发布到第一个高性能开源 attention kernel，业界花了**两年**（Shah et al. 2024，FlashAttention-3）。论文借此暗示 AMD 不能等两年。

### 2.2 为什么 Triton/Mojo 之类的编译器还不够？

页 1（第 49–55 行）与页 4 Section 2.2 列出编译器路线的局限：

- **Triton/Mojo/TileLang** 基于 LLVM/MLIR，理论上可跨厂商，但实测：
  - Mojo 的 MHA kernel 在 MI355X 上 bank conflict 严重，仅达峰值 50%（页 4 footnote 5，作者用 `rocprofv3 --pmc SQLDSBANKCONFLICT,SQINSTSLDS` 实测）。
  - Triton-AMD 的 BF16 GEMM 比 HK 慢 1.3–3×（页 9 第 786 行）。
  - 编译器普遍**不让开发者精细控制寄存器寿命与指令调度**——这恰恰是 AMD 拿到 peak 性能的核心。
- **AI 自动设计 kernel**（Kevin、KernelBench 等）尚处早期，对新硬件特性支持差，并易"reward hacking"（页 1 第 55–56 行）。

### 2.3 为什么需要 ThunderKittens 的 AMD 版本？

ThunderKittens（TK，Spector et al., ICLR 2024）以及它的"后继者" CuTe DSL（NVIDIA 2025）、Gluon（Triton 2025）已经在 NVIDIA 上证明了一种 **C++ 嵌入式、轻量化、tile-first** 的 DSL 设计哲学的有效性。它们的三大支柱（页 1 末第 62–73 行）：

1. **Tiles**：基本数据类型是 tile，带优化的访存模式；TK 暴露 PyTorch 风格 bulk 算子（mma/exp 等）包装 PTX，让开发者显式管理各级 GPU 内存。
2. **Overlapping**：少量预设调度模板（典型如 producer-consumer / wave specialization）把 worker（NVIDIA 的 warp，AMD 的 wave）映射到执行单元上获得高 occupancy。
3. **Grid scheduling**：通过 thread block 分发顺序最大化非可编程 cache（L2/LLC）的复用。

**问题来了**：当前所有这类 C++ DSL 都只跑在 NVIDIA 上。AMD 用户要么写汇编，要么吃编译器性能损失。论文的研究问题（页 2 第 87 行）就是：

> *"Are entirely new programming primitives needed to simplify AMD kernel development, or do existing primitives suffice?"*

HK 的回答是：**前端原语够用，后端实例化必须重写**。

### 2.4 ROCm 现状速描

页 3 Section 2.1 给出的 AMD 软件 / 硬件 hierarchy（同时见 Figure 2 右图）：

- 编程语言层：raw assembly → HIP C++ → LLVM IR (+ hint) → Triton/Python 等高层接口；HIPCC 是默认 C++→ASM 编译器，但会自作主张做指令重排和寄存器寿命跟踪，破坏开发者意图。
- 库层：AITER（汇编）、Composable Kernels（CK，模板元编程 + 部分汇编）、hipBLASLt（GEMM 库）、rocFFT 等。
- 总结：缺一个**"NVIDIA CUTLASS / ThunderKittens 等价物"**——开放、易用、性能不输汇编、可让普通 AI 工程师快速覆盖新 workload。HK 就是来填这个坑。

---

## 3. 核心思想 / 方法（最详写）

HK 的整体技术框架可视为"**TK 前端 + 三个 AMD-native 后端模块**"：

```
┌──────────────────────────────────────────────────┐
│  前端：tile DSL（页 4 §3.1）                       │
│  - tile 数据类型（dtype, rows, cols, layout）      │
│  - load/store 跨内存层次                          │
│  - bulk 算子（mma/exp/add，wrap MFMA / HIP）       │
└──────────────────────────────────────────────────┘
            │ 实例化由以下三个模块负责
            ▼
┌──────────────┬───────────────┬────────────────────┐
│ §3.2 可编程   │ §3.3 调度算法  │ §3.4 非可编程       │
│ 内存（VGPR/   │ 8-WAVE & 4-   │ 内存（L2/LLC chiplet│
│ AGPR/LDS）    │ WAVE 替代      │ swizzle，Algo 1）   │
│ + pinned reg  │ wave special.  │                    │
│ + multi-shape │                │                    │
│ swizzle       │                │                    │
└──────────────┴───────────────┴────────────────────┘
```

下面逐模块剖析。

### 3.1 Tile 编程接口（页 4 §3.1）

HK 完全沿用了 TK 的 PyTorch/NumPy 风味 API，关键属性：

- **memory location**：tile 可建立在 register 或 shared memory（LDS）上。
- **类型参数**：`dtype` ∈ {FP32, BF16, FP16, FP8, FP6}、`rows`、`cols`、`layout` ∈ {row-major, column-major}。
- **形状约束**：rows / cols 必须是 matrix core shape 的整数倍（AMD MFMA 的输入/输出 layout 与 thread 拥有关系是固定的）。
- **基本操作**：
  - `load(global → shared)`、`load(shared → register)`、`store(...)`
  - bulk 算子：`mma(D, A, B, C)`、`exp`、`add`、`mul`、`max`、reduction 等
- **关键工程取舍**：bulk 算子是**轻量 wrap，不引入额外指令**——直接生成 CDNA 汇编 / HIP intrinsic（NVIDIA 上对应 PTX）。这与 Triton 的"自动调度"路线截然相反：HK 不试图智能化，而是**把控制权透明地还给开发者**。

页 4 第 323–326 行明言："Given these familiar programming primitives, HK automatically optimizes the memory access patterns for tiles."——也就是说 HK **不替开发者决定调度，但替他屏蔽 swizzle / phase / bank conflict 这些低层细节**。

### 3.2 CDNA3/CDNA4 上的可编程内存优化（页 4–6 §3.2）

#### 3.2.1 寄存器：HIPCC 的两个"罪状"与 EXPLICIT REGISTER SCHEDULING

AMD CDNA SIMD 上有 512 个 32-bit 寄存器；当一个 SIMD 上只有一个 wave 时，硬件把它们分成 **256 VGPR + 256 AGPR**（accumulator GPR，专为 MFMA 累加设计）。痛点（页 1 footnote 1，页 4 第 332–353 行）：

- **HIPCC 不让开发者把 AGPR 当作 MFMA 的 input operand 用**：尽管硬件支持，编译器会强行插入 `v_accvgpr_read` 把 AGPR 拷回 VGPR，浪费寄存器与发射槽。
- **HIPCC 还会做激进的寄存器寿命复用与指令重排**，导致开发者写好的"在 MFMA 之前预 load A、B"的 pattern 被打散。

HK 的解决方案是 **"绕过编译器"**：开发者用 HK 提供的 `pinned register tile` API，直接把 tile 的每一寄存器号显式 pin 到 inline asm。代码接口与"普通 compiler-managed tile"完全一致——开发者可以选择性地启用 pinning（页 5 第 374–376 行）。

实证（页 5 Table 1，4-wave MHA non-causal backward，batch=16, heads=16, head_dim=128）：

| 实现 | seq=4096 TFLOPS | seq=8192 TFLOPS |
| --- | --- | --- |
| HK（默认 HIPCC） | 855 | 909 |
| HK + pinned registers | **1024** | **1091** |
| AMD AITER（手写汇编） | 1018 | 1169 |

**洞察**：pinned register 让 HK 在 4096 上反超 AITER，在 8192 上达到 93% 汇编性能——而代码量比汇编低一个数量级。

#### 3.2.2 多种 MFMA 形状下的 tile layout 与 swizzle（页 5–6）

NVIDIA 与 AMD matrix layout 的根本差异（页 5 §3.2.2）：

- **NVIDIA**：所有 wmma/wgmma 形状（如 16×8×16, 64×128×16, 256×256×16 等）都基于一个 16×16 **core matrix** 重复盖章构成，因此 TK / Linear Layouts 用**单一 swizzle 策略**就能覆盖所有形状（页 5 Figure 4a）。
- **AMD**：每个 MFMA 形状（16×16×32、32×32×16、16×16×16 等）有**完全不同的 thread→element 拥有图**（Figure 4b/4c）。没有底层 core matrix 可组合。
- 加上 LDS bank 行为也依指令而异：`ds_read_b128` 走 64 bank（每个 4 byte）、4 phase；`ds_read_b96` 走 32 bank、8 phase；这些 phase 并非按 tid 顺序——AMD 的 ISA 文档没写，作者**自己写了一个 solver 反推 phase 分布**，并把结果汇总到附录 Table 7（页 5 footnote 6）。

HK 的处理策略（页 6 第 427–450 行）：

1. **Register tile**：默认采用最小的 MFMA 形状（如 16×16×32），因为这给后续指令调度最大自由度（小 tile 可灵活穿插）。但 HK 也允许用 `mfma_shape` 参数指定 32×32×16 等大形状供边角 case 使用——HK 的 attention backward 就**同时使用 16×16×32 与 32×32×16 两种 MFMA**（页 10 第 819 行）。
2. **Shared tile**：作者发现"为每种 layout 写专属 swizzle"代码量爆炸，因此采取**实用主义**：识别"经常共现"的 layout 组（如 16×32 row layout + 16×32 column layout），为这些组合设计**同时无 bank conflict** 的 swizzle。Figure 3（页 5）给出例子：把 16×32 BF16 tile 从第 8 行起把"前 8 列与后 8 列"对调，使 `ds_read_b128` 行布局与 `ds_read_b64_tr_b16`（转置加载，列布局）都无 bank conflict。详见附录 D.1。
3. **Global → shared 直传（async copy）**：AMD 提供类似 NVIDIA TMA 的 HBM→shared 异步 load，**绕过寄存器文件**。但与 TMA 不同——AMD 的指令以 **per-thread 地址** 作输入，所以**swizzle 需在 HBM 地址端做**，而非 shared 端（页 6 第 444–450 行）。这是 HK 与 TK 在 global load 处理上的最大实现差异。

### 3.3 调度算法：抛弃 wave specialization（页 6–7 §3.3）

这是 HK 最核心也最反常识的贡献。

#### 3.3.1 为什么 wave specialization 在 AMD 上不行？

NVIDIA wave specialization（FlashAttention-3、CUTLASS、TK 上 GEMM 等普遍采用）的成立条件：

- 专用 memory hardware：**TMA**（async global↔shared，绕过 register）。
- 大 SRAM：B200 的每 SM SRAM 比 MI355X 多 40%。
- async matmul 接受 shared/tensor memory operand：**wgmma** (Hopper)、**tcgen05** (Blackwell)。
- 寄存器再分配（producer 让出寄存器给 consumer）。
- 硬件同步原语 **mbarriers**。

AMD CDNA3/CDNA4 一项都没有：

- 没有 TMA-style 异步 copy（虽有 direct async load，但需 per-thread addr）。
- 没有 wgmma 等价物，MFMA 只接受寄存器输入。
- **没有寄存器再分配**——AMD 是**静态划分**（页 6 第 518 行 "AMD hardware statically divides registers across all waves"）。这意味着：producer wave 一旦 spawn 就永久占着寄存器，consumer 拿不到那部分。
- 没有 mbarrier，需要用 shared memory atomics 模拟。

实证（页 6 Table 2，BF16 GEMM M=N=K=8192）：

| 配置 | #Producer / #Consumer | MFMA shape | output tile | TFLOPS |
| --- | --- | --- | --- | --- |
| HK（带 P-C） | 4 / 8 | 16×16×32 | 128×256 | 893 |
| HK（带 P-C） | 4 / 12 | 16×16×32 | 192×256 | 1278 |
| HK（无 P-C，纯 consumer） | 0 / 8 | 16×16×32 | 192×256 | 1281 |
| HK（无 P-C，纯 consumer） | 0 / 8 | 16×16×32 | **256×256** | **1610** |
| TK (B200) | — | 256×256×16 | 256×256 | 1538 |
| CUTLASS profiler 选优 (B200) | — | 256×256×16 | 256×256 | 1570 |

关键观察：
- **去掉 producer 后 HK 在 MI355X 反而上到 1610 TFLOPS，超过 B200 的 TK / CUTLASS 数值**（注意硬件不同，但 BF16 峰值 2.5 vs 2.2 PFLOPS）。
- **output tile 大小是性能主导因子**，而非同步原语。在 192×256 上 atomics-based P-C 与 atomic-free 无 P-C 性能基本一致，证明 mbarrier 缺失并非 AMD 的瓶颈。
- producer 多出来的几十个 wave 都是**净负担**——它们消耗寄存器但不参与 MFMA。

#### 3.3.2 HK 的两个替代调度模式（页 7 §3.3.2）

HK 利用 AMD CU 的**4 SIMD/CU**结构来重新设计 overlap 模式：

##### 模式 A：8-WAVE PING-PONG（平衡型 workload）

- 每个 thread block 配 **8 wave**，每个 SIMD 上 resident **2 wave**。
- 8 wave 分两组（每组 4 wave，覆盖 4 个 SIMD）。
- **同 SIMD 上的两 wave 交替角色**：一个时刻 wave A 只发 MFMA，wave B 只发 LDS/HBM load；下一时刻互换。
- 角色切换由 **conditional barrier** 触发，代码极简（页 7 Figure 5）：

  ```cpp
  if (kittens::warpid() / 4 == 1) {
      __builtin_amdgcn_s_barrier();
  }
  ```

  半数 wave 走入 barrier 暂停一小段，使两组在 compute / memory 上错峰。
- **优势**：tile 粒度可以做大（与 wave specialization 等价的代码风格），代码紧凑可读；page 1 摘要给出的 Figure 1 profiler 可视化即此模式：每行一个 wave，相邻两 wave 共享 SIMD，一个执行 memory，另一个执行 compute，然后翻转。
- **适用场景**：compute / memory 时长大致相当的 workload——BF16/FP8 GEMM、MHA forward、GQA forward 等。

##### 模式 B：4-WAVE INTERLEAVE（不平衡 workload）

- 每个 thread block 配 **4 wave**，**1 wave 1 SIMD**。
- 每个 wave **同时**发 compute 与 memory 指令，但开发者要把指令"细粒度交错"（fine-grained staggered）排好。
- 为达到 ALU、MFMA、LDS 等多个 pipeline 都饱和，必须用**小 base tile** 编程（如 16×16×32 单步 MFMA），代码量显著膨胀。
- **适用场景**：compute-heavy 或 memory-heavy 的不平衡 workload，单 wave 可动态调整指令配比——典型如 attention backward。

页 7 Table 3 对比：

| Kernel | Pattern | LoC（hot loop） | TFLOPS |
| --- | --- | --- | --- |
| FP8 GEMM | 8-WAVE | 48 | 3222 |
| FP8 GEMM | 4-WAVE | 183 | 3327 |
| MHA backward | 8-WAVE | 331 | 894 |
| MHA backward | 4-WAVE | 989 | 1091 |

权衡明显：**4-WAVE 多 ~3× 代码量换 4–22% 性能**。HK 让开发者两种都能选。

> 论文给出的最重要的实证结论（页 2 第 162 行 + 页 7 第 600 行）：**8-WAVE 已经足够 match AMD 手写汇编**（BF16 GEMM、FP8 GEMM、attention forward），并在 GQA non-causal backward 上**超出基线 1.8×**；4-WAVE 在 backward 上更进一步达到 **2.3×**。这意味着对绝大多数 AI 任务，开发者只需要写 8-WAVE 代码，就能拿到汇编级性能。

#### 3.3.3 关于深 pipeline 的再权衡（页 6 第 521–553 行）

NVIDIA 用大 SRAM + 大 MFMA 形状（256×256×16）做深 pipeline，AMD 没有大 SRAM。**但 AMD 的小 MFMA 形状（16×16×32）反而提供另一条深 pipeline 路径**——以更细颗粒的 load/compute stage 切片。配合 2× 大的 register file，AMD 不用 wgmma 也能 hide latency。这是 HK 设计哲学的关键：**接受硬件差异，重新设计算法**，而不是模仿 NVIDIA。

### 3.4 非可编程内存优化：chiplet-aware grid swizzle（页 7–9 §3.4）

#### 3.4.1 为什么 chiplet 让 grid scheduling 变难

MI355X chiplet 结构（Figure 2 / 页 7 第 619–625 行）：

- **8 个 XCD**，每 XCD = 32 CU + **私有 4MB L2**（CDNA3 是 38 CU/XCD）。
- 8 个 XCD 共享一个 **LLC**（在 L2 与 HBM 之间）。
- 硬件 scheduler 用 **round-robin** 把 thread block 分发到 XCD 上。
- L2 miss penalty ≈ 300 ns；LLC miss penalty ≈ 500 ns。

带宽公式（页 7 公式 1）：

```
Bandwidth = LLC_BW × LLC_hit% + L2_BW × L2_hit%
```

朴素 row-major grid 下，同一 XCD 上的 block 加载的是 A、B 矩阵的**不重叠**子块——L2 几乎打不到 hit，且 XCD 之间也错开访问，LLC 也吃亏。Table 4 第 1 行（M=N=K=14592 BF16 GEMM）：L2 = 36%、LLC = 76%、有效带宽 10.7 TB/s、900 TFLOPS。

#### 3.4.2 两层目标：L2 reuse vs LLC reuse 的张力

- **L2 reuse**：同一 XCD 上的 block 应覆盖输出矩阵 D 的一个**矩形块**（"L2 tile"），让连续 block 复用 A 的同 row 与 B 的同 column。
- **LLC reuse**：跨 XCD 的访问足迹应在 A、B 上**相互重叠**（"LLC tile"），让多 XCD 共享 LLC 中的同一段数据。

二者**有冲突**：纯优化 L2（每 XCD 抢自己的矩形）会让不同 XCD 各自吃不同的 A/B 子块——LLC 命中惨跌。Table 4（页 9）M=N=K=9216 case 对比：

| 调度 | L2% | LLC% | 带宽 | TFLOPS |
| --- | --- | --- | --- | --- |
| Row-major | 55% | 95% | 15.1 TB/s | 1113 |
| XCD (W=7, C=216) 仅优化 L2 | **79%** | **24%**↓ | 14.9 TB/s | 991 |
| XCD (W=5, C=25) 联合优化 | 75% | 93% | **18.3 TB/s** | **1145** |

只优化 L2 反而**性能掉了 11%**——因为 LLC 命中崩了。HK 必须**联合优化**。

#### 3.4.3 Algorithm 1：XCD swizzle（页 8）

输入：原始 grid block 索引 `(b_x, b_y, b_z)`、grid 维度 `(g_x, g_y, g_z)`、XCD 数 `nXCD`、问题尺寸 `M, N`、tile 尺寸 `BLOCK_M, BLOCK_N`、窗口高 `W`、chunk 大小 `C`。

算法分**两阶段**：

**阶段 1：XCD grouping** —— 把 2D grid 拍平后重映射，使**连续 C 个 block ID resident 同一 XCD**：

```
blocks         = g_x × g_y                       # 单 batch 的 block 总数
xy             = b_x + g_x × b_y                 # 拍平
blocks_per_cyc = nXCD × C
limit          = ⌊blocks / blocks_per_cyc⌋ × blocks_per_cyc

if xy <= limit:
    xcd       = xy mod nXCD
    local     = ⌊xy / nXCD⌋
    chunk_idx = ⌊local / C⌋
    pos       = local mod C
    xy'       = chunk_idx × blocks_per_cyc + xcd × C + pos
else:
    xy' = xy   # tail 不重排
```

效果：消去硬件 round-robin 带来的"同 XCD block 在 grid 上间隔分布"现象，把 XCD 内部的 block 重新连续化，**减少 cross-chiplet 数据迁移**。

**阶段 2：Hierarchical windowed traversal** —— 在重映射后的索引上做"垂直窗口"遍历：

```
num_rows         = M / BLOCK_M
num_cols         = N / BLOCK_N
tids_per_group   = W × num_cols                  # 一个高度 W 的横条
group_id         = xy' / tids_per_group          # 第几个横条
first_row        = group_id × W
win_h            = min(num_rows - first_row, W)  # tail-safe
ε                = xy' mod tids_per_group
row              = first_row + (ε mod win_h)     # 快索引：列内向下走
col              = ε / win_h                     # 慢索引：W 行后跳列
return (row, col, b_z)
```

效果：把 block 的访问模式"折叠"成宽 `num_cols`、高 `W` 的矩形条带，使同一 XCD 上的 block 形成 W×k 的矩形（典型 8×4 或 4×8）——这就是 L2 tile。

**调参原则**（页 9 第 749–757 行）：

- **W 控制 L2 reuse**：MI355X 上 32 CU/XCD，经验上 8×4 或 4×8 最优。
- **C 控制 LLC reuse**：协调跨 XCD 是否在 A 的同 row 范围内工作。
- **L2 带宽 ≈ 3× LLC 带宽**，所以 W 应优先吃满 L2，再用 C 微调 LLC。

实证（Table 4 M=14592 case）：
- Row-major：L2=36%, LLC=76%, 10.7 TB/s, 900 TFLOPS
- XCD W=8/C=542：L2=79%, LLC=7%, 13.9 TB/s, 980 TFLOPS（LLC 几乎全失，仍因 L2 大幅提升而总体更快）
- XCD W=8/C=64：L2=78%, LLC=55%, 16.6 TB/s, **1068 TFLOPS**（联合最优，比 row-major 提升 19%）

注意论文指出：当输出矩阵宽度（以 tile 为单位）与 nXCD **互素**时（如 57 与 8），row-major 命中率最差，HK 算法收益最大。

---

## 4. 实现 / 工程细节

### 4.1 HIP / 内联汇编层

- HK 是 **C++17 头文件库**（沿用 TK 路线），所有"算子"都是 inline 模板函数，`__device__` 修饰。
- 真正发出 CDNA 指令通过两类机制：
  1. **AMD intrinsic**（`__builtin_amdgcn_*`）——例如 `__builtin_amdgcn_s_barrier`（页 7 Figure 5 中用于 8-WAVE PING-PONG 的条件 barrier）、`__builtin_amdgcn_mfma_*`（MFMA 调用）、`__builtin_amdgcn_ds_*`（LDS read/write）。
  2. **Inline asm + 显式 register clobber**——用于 EXPLICIT REGISTER SCHEDULING：tile 的每个 32-bit 槽被 pin 到具体的 `v0…v255` / `a0…a255`。这绕过 HIPCC 的 register allocator。

### 4.2 Async copy 实现

- 利用 CDNA 的 **direct global → LDS** 异步 load（接近 NVIDIA 的 `cp.async`）。
- 与 TMA 的根本差异：AMD 指令吃 **per-thread 32-bit / 64-bit 地址**，因此 swizzle **必须在 HBM 端就算好**，让每个 thread 的地址直接命中目标 LDS bank（页 6 第 444–450 行）。
- HK 把这部分抽象进 `tile.load_async(global_ptr, hbm_swizzle_func)`，开发者无需亲自计算 phase 与 bank。

### 4.3 LDS（shared memory）布局

- AMD MI355X 每 CU LDS 较小（相对 B200 SRAM 少 40%）——所以 HK 不能像 TK 那样囤大 stage。
- LDS bank 行为依指令而异，已在 §3.2.2 描述。HK 内部维护一个查找表：给定 (matrix shape, layout, mfma instruction)，返回选定的 swizzle pattern 与 phase 排布。
- 论文附录 D.1 给出无法用单一 swizzle 同时解所有 layout 的反例与证明（论文正文未展开，但仓库与附录可查）。

### 4.4 Register allocation 工程

- 默认路径：HK 标准 tile，HIPCC 自行分配 → 适合 90% 的简单场景。
- 高性能路径：**pinned register tile**——每个 tile 在声明时绑定到一段连续物理寄存器号（VGPR/AGPR），所有后续算子（mma/exp/...）在 inline asm 中显式使用这些编号，使 HIPCC 完全失去重排能力。这就是 Table 1（页 5）中"HK with pinned registers"的实现机制。
- 应用场景：**attention backward**——register 压力极大，HIPCC 自动调度会引入冗余的 `v_accvgpr_read`，pin 后 AGPR 被直接当 MFMA 输入，省 50%+ register traffic。

### 4.5 MFMA shape 选择策略

- 默认 16×16×32（最小颗粒，调度灵活度最高，与 8-WAVE PING-PONG 共生）。
- 边角 case（如 attention backward 的 Q·Kᵀ 转置内积）启用 32×32×16，并通过开发者参数 `mfma_shape_t` 显式指定。
- 多 shape 共存时（HK attention backward 即 16×16×32 + 32×32×16 混用），需要为每种 shape 各自维护 LDS swizzle 表——这正是 §3.2.2 提到的"多 layout swizzle"处理点。

### 4.6 Build / 部署

- 仓库 `https://github.com/HazyResearch/HipKittens` 同时提供：
  - 头文件库本体；
  - 一组示例 kernel（GEMM、attention、RoPE、LayerNorm、fused dropout-residual-LN 等）；
  - Python bindings（用 ctypes/torch.utils.cpp_extension 编译加载）；
  - benchmark 脚本（500 warmup + 100 measurement, N(0,1) 输入）。
- HK 已被 **AMD AITER 库产品化采用**——意味着 AMD 官方接受了 HK 的 kernel 实现作为部分汇编 kernel 的替代品（页 1 摘要末尾）。

---

## 5. 评测

### 5.1 实验环境

- 硬件：AMD MI325X（CDNA3）、AMD MI355X（CDNA4）；NVIDIA 对照用 B200 SXM5。
- 软件：ROCm 7.0 beta docker (`rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta`)。
- 度量：500 warmup + 100 timing run，输入采样 N(0,1)，报均值 TFLOPS/s。
- 基线：AITER（汇编）、Composable Kernels (CK)、hipBLASLt、ROCm Triton、Mojo（部分实验）、PyTorch SDPA、PyTorch compiled。

### 5.2 BF16 / FP8 GEMM（Figure 7，页 9）

- 在 MI325X、MI355X 上：HK 与 AITER、hipBLASLt 持平。
- HK 比 Triton-AMD GEMM 快 **1.3×–3.0×**。
- 用**单个 8-WAVE kernel** 即可覆盖所有评测形状，**无需逐 shape 调参**——这与 CUTLASS profiler 的"千 kernel 大杂烩"形成鲜明对比。

### 5.3 Attention Forward（Figure 8，页 10）

- 配置：batch=16, query_heads=64, kv_heads=8, head_dim ∈ {64, 128}，causal/non-causal。
- HK vs AITER（汇编基线）：**1.0×–2.1× 更快**。
- HK vs PyTorch SDPA：**1.3×–4.5×**；vs CK：**1.0×–1.4×**；vs Triton-AMD：**1.2×–4.5×**。
- HK attention forward 用 **8-WAVE PING-PONG**：compute cluster 中 wave 把 online-softmax 向量算子（max/sub/exp2/accumulate）与 MFMA 交错，**与 FlashAttention-3 在可比配置下性能持平**——尽管 MI355X 与 B200 体系结构差异巨大。

### 5.4 Attention Backward（Figure 9 + Figure 16，页 10）

- GQA causal/non-causal backward：HK 比所有基线快 **1.8×–2.5×**（含 AITER 汇编、CK、PyTorch SDPA、Triton）。
- MHA backward：与最强汇编基线持平。
- 关键工程要点（页 10 第 818–823 行）：
  - 同时使用 16×16×32 与 32×32×16 两种 MFMA；
  - shared tile 同时支持 row 与 column layout load（避免重新排序数据）；
  - 全程使用 explicit register pinning（否则 HIPCC 损失 ~10–20%）。

### 5.5 Memory-bound Kernels（Figure 10，页 11）

- Fused dropout-residual-layernorm（prenorm Transformer 用）+ rotary positional encoding。
- HK 比 AITER、PyTorch compiled 快 **1.1×–2.2×**。
- 这是论文最有"未被汇编覆盖到的 long tail" 意味的部分——即使 AITER 团队也没有为这些 fused 算子手写汇编版本。

### 5.6 端到端验证：模型预训练

- 用 HK kernel 训练 **Llama 1B + BERT 110M** on The Pile 数据集 10B tokens。
- 与 PyTorch + AITER 训练得到的 perplexity 在数值上**等价**——证明 kernel 的数值稳定性（无 NaN、无精度损失）（页 10 第 836–840 行）。

### 5.7 Producer-Consumer 微观对照（Table 2）

已在 §3.3.1 详述。核心信息再次强调：**0 producer + 8 consumer + 256×256 output tile = 1610 TFLOPS**，这是 HK 找到的 BF16 GEMM 最优点，并构成了 §3.3.2 的设计依据。

### 5.8 Chiplet swizzle 微观对照（Table 4）

已在 §3.4.2/§3.4.3 详述。两个 GEMM size 上 row-major vs HK XCD swizzle 的差距分别是 **+3%**（M=9216）和 **+19%**（M=14592）；与 problem size 与 nXCD 互素性相关。

---

## 6. 思想精读 / 启示

### 6.1 "前端通用 + 后端专属" 的 DSL 设计哲学

HK 的最大启示是**给跨厂商 DSL 的设计提供了实证模板**：

- **保持前端**（tile + PyTorch-style bulk ops + 三大调度模板）一致——这是开发者认知层面的"共同语言"。
- **替换后端**——swizzle、phase、MFMA shape、scheduling pattern、cache hierarchy strategy 全是硬件 specific，必须为每个厂商重新实现。

这与 LLVM/MLIR 的"统一 IR + 后端 codegen"思路在精神上是一致的，但 HK 选择**更轻量、更显式**的 C++ 模板路线：开发者写出的代码**直接对应** GPU 指令，不依赖编译器优化。这与 Hazy Research 提出的 "PyTorch-inspired DSL" 一脉相承（TK → HK → 未来可能的更多硬件版本）。

### 6.2 反思 wave specialization 的"hardware lottery"

NVIDIA 上 wave specialization 之所以成为标准，是因为 **TMA + wgmma + mbarrier** 三件套刚好匹配它的语义。论文用 AMD 的反例提醒我们：**调度模式不是 GPU programming 的"自然律"**，它是被 NVIDIA 硬件特性反向塑造的产物。当硬件特性不存在时（如 AMD），强行套用反而是反优化。

这给跨厂商 kernel 库设计者的实操教训是：**不要把 NVIDIA 特性当成 GPU programming 的范本**，而要回到第一性原理（output tile size × pipeline depth × register pressure）来选择调度模式。

### 6.3 Stanford Hazy 的 DSL 路线图

从公开资料看，Hazy Research 的 GPU DSL 演化轨迹：

- **TK (ICLR 2024)**：NVIDIA Hopper 上验证 tile-first 哲学。
- **No-bubbles megakernel (2025 blog)**：Llama-1B 端到端低延迟 megakernel。
- **HK (MLSys 2026)**：跨到 AMD CDNA3/4，验证可移植性。
- **未来方向（论文 §5）**：作者明示希望 HK 成为"universal software stack"的一步。考虑到 Apple Silicon、Intel Gaudi、Google TPU 等也在崛起，HK 的"前端通用 + 后端专属"模板可以继续扩展。

### 6.4 对 AI 训练基础设施市场的影响

- **打破 CUDA moat**：论文用 d=64 attention 和 GQA backward 这些 corner case 证明，AMD 在某些重要 workload 上**已经领先 NVIDIA**（前提是有好的 kernel）。
- **对 AMD 的影响**：HK 已被 AITER 库吸纳——这可能是 AMD 第一次大规模采用学术界 DSL 而非自家汇编。降低了 AMD GPU 上手门槛。
- **对编译器路线（Triton、Mojo）的挑战**：HK 用具体数据（Mojo MHA 仅达 50%、Triton GEMM 慢 3×）说明纯编译路线在新硬件上爬坡过慢。手写 DSL 反而更快达到 SoTA。

### 6.5 工程方法论上的启示

- **写 solver 反推未公开的硬件细节**（论文页 5 footnote 6 自述为 LDS phase 写了 solver）。
- **小心衡量"看似有用的功能的反作用"**：producer wave 在 NVIDIA 上提升性能、在 AMD 上下降性能——同一抽象在不同硬件下符号相反。
- **联合优化层间冲突目标**：L2 vs LLC reuse 的张力是普通开发者容易忽视的；HK 的 W、C 双参数算法是一个标杆。

---

## 7. 局限与开放问题

### 7.1 论文承认的局限

1. **代码量上 4-WAVE 比 8-WAVE 多 ~3×**（Table 3）：复杂 workload 仍需手写大量 base-tile 级代码。HK 没有给出"自动从 8-WAVE 升级到 4-WAVE"的工具。
2. **Pinned register 牺牲可移植性**：一旦 pin，代码与具体 SIMD 寄存器布局耦合，如未来 CDNA5 register file 变化（更大/重新分区），需要重写 pinned section。
3. **Swizzle 仅覆盖"常共现 layout 组合"**——不是所有 (matrix shape, instruction) 组合都无 bank conflict；边角 case 需开发者额外判断。
4. **Algorithm 1 的 W、C 仍需手调**：MI355X 上经验值 8×4/4×8，但其它 problem size 与 chiplet 配置（如 MI325X 的 38 CU/XCD）需重新搜参。论文未给自动 tuner。
5. **FP8 PyTorch 支持仍 experimental**（页 9 第 770 行）：HK FP8 kernel 用 Python bindings 调用，端到端集成体验未必平滑。

### 7.2 评审/读者可能追问的开放问题

- **CDNA5 / MI400 上的可迁移性**：HK 大量决策与 4 SIMD/CU、8 XCD/GPU、static register split 等强相关。下一代硬件若改 SIMD 数量或允许 register repartition，HK 调度模式是否仍最优？
- **与 Composable Kernels 的关系**：CK 的 template metaprogramming 也能写出高性能 kernel，但代码极复杂。HK 论文未给出 HK vs CK 的代码量对比，亦未直接承接 CK 的某些抽象（如 `BlockGemm`）。两条路线是否可融合？
- **是否支持 sparse / 量化的 fused kernel**：论文未涉及 W4A8、INT4、稀疏 attention 等场景。
- **多 GPU / NCCL / RCCL 集成**：HK 是单 GPU kernel 库，对 communication-overlap kernel（类似 COMET、TileLink）尚无设计。
- **AI auto-coding 友好度**：DSL 设计是否对 LLM 自动生成 kernel 友好？论文未做该方向消融。
- **Triton-AMD 的对比是否过时**：Triton 一直在快速迭代 AMD 后端（2025 起 AMD 大力投入），ROCm 7.x 之后差距是否会缩小？论文 benchmark 时间窗口（2025 年下半年）后续有待跟踪。
- **正确性测试覆盖**：论文用 Llama 1B + BERT 110M 验证训练 perplexity 一致，但未给 unit-test 数值精度对比表（FP8 等低精度尤其敏感）。
- **MoE / expert parallel kernel**：未覆盖。AMD 在 MI355X 上 FP6/FP4 算力极强，MoE 应是杀手级 use case。

---

## 8. 关键术语速查表

| 术语 | 全称 / 解释 | 与 NVIDIA 对应 |
| --- | --- | --- |
| **CDNA** | Compute DNA — AMD 数据中心 GPU 微架构系列；CDNA3 = MI300/MI325，CDNA4 = MI350/MI355 | 类比 NVIDIA Hopper / Blackwell |
| **CU** | Compute Unit — AMD GPU 的 SM 等价物，每 CU 4 个 SIMD | NVIDIA SM |
| **SIMD** | 一组 64 lane 的执行单元，wave 在其上 lockstep 运行 | NVIDIA 的"sub-partition / scheduler"近似 |
| **wave / wavefront** | 64 thread 一组，AMD 的最小调度单位 | NVIDIA warp（32 thread） |
| **VGPR** | Vector General-Purpose Register — 256 个/SIMD（单 wave 时）| NVIDIA "register file" 一部分 |
| **AGPR** | Accumulator GPR — 256 个/SIMD，专为 MFMA 累加器；HIPCC 不允许 AGPR 作 MFMA 输入，HK 用 pin 绕过 | 无直接对应，类比 Tensor Core 内部 accumulator |
| **LDS** | Local Data Share — AMD 的 shared memory，64 banks × 32-bit；不同 ds_read_* 指令的 phase / bank 行为不同 | NVIDIA shared memory |
| **MFMA** | Matrix Fused Multiply-Add — AMD 矩阵指令族，形状如 16×16×32、32×32×16；与 NVIDIA core matrix 不同，没有统一 16×16 building block | NVIDIA wmma / wgmma |
| **XCD** | Accelerator Complex Die — 一个 chiplet，含 32 CU (CDNA4) + 私有 4MB L2 | NVIDIA Blackwell 也用 chiplet，但通常 2 chip |
| **LLC** | Last-Level Cache — XCD 之间共享，位于 L2 与 HBM 之间 | NVIDIA L2（无显式 LLC 命名） |
| **HBM** | High Bandwidth Memory — 全局显存；MI355X 288 GB / 8 TB/s | 同概念 |
| **AITER** | AMD 官方手写汇编 kernel 库（github.com/ROCm/aiter）| 类比 cuBLAS + cuDNN 的"汇编版" |
| **Composable Kernels (CK)** | AMD 模板元编程 GEMM / conv 库（github.com/ROCm/composable_kernel）| 类比 CUTLASS |
| **hipBLASLt** | AMD 的"BLAS for Transformer"风格 GEMM 库 | NVIDIA cuBLASLt |
| **HIP** | Heterogeneous-Compute Interface for Portability — AMD 的 CUDA 等价物 + HIPCC 编译器 | CUDA + NVCC |
| **ROCm** | Radeon Open Compute — AMD 的 CUDA 平台等价物（驱动 + runtime + 库 + Triton）| CUDA toolkit |
| **wave specialization** | NVIDIA 主流调度模式：少量 producer wave 搬数据，consumer wave 计算；FA-3、TK GEMM 等都用 | 论文论证此模式在 AMD 失败 |
| **8-WAVE PING-PONG** | HK 提出：8 wave/block，2 wave/SIMD，交替 compute/memory 角色，conditional barrier 控制 | 无 NVIDIA 等价（NVIDIA 用 producer-consumer） |
| **4-WAVE INTERLEAVE** | HK 提出：4 wave/block，1 wave/SIMD，wave 内自己穿插 compute/memory；适合不平衡 workload | 类似 "warp-uniform schedule" 但更细 |
| **TMA** | NVIDIA Hopper 起的 Tensor Memory Accelerator — 异步 global↔shared 大块 copy | AMD 用 per-thread async copy 不完全等价 |
| **wgmma** | NVIDIA Hopper 异步 matmul，可读 shared memory 操作数 | AMD MFMA 仅支持 register 操作数 |
| **mbarrier** | NVIDIA Hopper 硬件同步原语 | AMD 无；HK 用 LDS atomic + s_barrier 模拟 |
| **swizzle** | 数据布局重排，使共享内存访问无 bank conflict | 同概念，但 AMD 因 phase 复杂度高一档 |
| **Linear Layouts (Gluon)** | Triton/Gluon 用矩阵论刻画 thread→element 映射的统一框架 | 论文论证 NVIDIA core matrix 让该方法可行；AMD 不行 |
| **THUNDERKITTENS (TK)** | Hazy Research 的 NVIDIA 版前作（ICLR 2024）| HK 的"祖宗" |
| **AITER 产品化** | AMD 已把 HK 整合进 AITER 仓库 | 论文摘要点出 |

---

## 9. 关键页码索引

| 主题 | 页码 / 章节 / 关键位置 |
| --- | --- |
| 摘要：AMD CUDA moat、HK 主张、AITER 已采用 | 页 1 摘要全文（第 10–26 行）|
| TK/HK 三大原语（tiles / overlapping / grid scheduling）| 页 1 末（第 62–73 行）|
| HK 三大模块概述（pinned reg / scheduling / chiplet swizzle）| 页 2（第 87–146 行）|
| 硬件参数对比表（B200 vs MI355X 算力/内存）| 页 3 Figure 2（左表）|
| AMD GPU 软件硬件层次结构图 | 页 3 Figure 2（右图）|
| 256 VGPR + 256 AGPR 划分 | 页 1 footnote 1，页 4 第 332–337 行 |
| CU / SIMD / wave / thread block 层级 | 页 3 第 218–228 行 |
| 256 CU、8 XCD、32 CU/XCD（CDNA4）| 页 3 第 226–228 行；页 7 第 619–622 行 |
| 软件栈 stack（asm / HIP / LLVM / Triton）| 页 4 第 251–260 行 |
| 相关工作：AITER、CK、TK、CuTe DSL、Gluon、Triton、Mojo | 页 4 §2.2（第 261–293 行）|
| Mojo MHA 仅 50% 性能与实测命令 | 页 4 footnote 5 |
| Tile 编程接口（dtype/rows/cols/layout、load/store、bulk ops）| 页 4 §3.1（第 305–325 行）|
| EXPLICIT REGISTER SCHEDULING / pinned register | 页 4–5（第 327–377 行）；Table 1 |
| Table 1：4-wave MHA backward, AITER vs HK with pin | 页 5 |
| AMD vs NVIDIA matrix layout 复杂度对比 | 页 5–6（第 391–422 行）+ Figure 4 |
| 多种 MFMA shape / multi-layout swizzle 处理 | 页 6（第 427–450 行）|
| LDS 16×32 BF16 swizzle 示意 | 页 5 Figure 3 |
| LDS bank phase 反推（solver）+ 附录 Table 7 | 页 5 footnote 6 |
| Async global→LDS 与 TMA 的差异（per-thread addr）| 页 6（第 444–450 行）|
| Wave specialization 在 AMD 失败的原因（静态 register、缺 TMA/wgmma/mbarrier）| 页 6 §3.3.1（第 492–520 行）|
| Table 2：BF16 GEMM Producer-Consumer 对比 | 页 6 |
| 8-WAVE PING-PONG 详细描述 + 条件 barrier 代码 | 页 7（第 556–583 行）+ Figure 5 |
| 4-WAVE INTERLEAVE 详细描述 | 页 7（第 584–592 行）|
| Table 3：8-WAVE vs 4-WAVE 代码量与 TFLOPS | 页 7 |
| Chiplet 缓存层次（L2 + LLC）+ 带宽公式 (1) | 页 7 §3.4（第 608–627 行）|
| Algorithm 1：XCD swizzle 全文 | 页 8 |
| Table 4：row-major vs XCD 调度 L2/LLC 命中率 + TFLOPS | 页 9 |
| Figure 6 / Figure 19：grid 调度可视化 | 页 8–9（Figure 6a/b/c）|
| 评测环境（500 warmup, 100 timing, ROCm 7.0 docker）| 页 9（第 762–777 行）|
| BF16 / FP8 GEMM 结果（Figure 7）| 页 9 |
| Attention forward 结果（Figure 8）| 页 10 |
| Attention backward 结果（Figure 9）| 页 10 |
| Memory-bound（fused dropout-residual-LN, RoPE）结果（Figure 10）| 页 11 |
| 端到端 Llama 1B + BERT 110M 训练验证 | 页 10（第 836–840 行）|
| Discussion / Conclusion | 页 10–11 §5 |
| Acknowledgements | 页 11 §6 |
| Contributions（作者分工）| 页 11 §7 |
| References 起始 | 页 11 |
| OpenReview ID | 文件名 `xxSSrndQrI.pdf` |
| 仓库地址 | 页 1 摘要末尾：`github.com/HazyResearch/HipKittens` |

---

## 10. 一句话点评

> **HipKittens 用一套 ~头文件级别的 C++ DSL 证明：跨 GPU 厂商的 tile-based 编程模型不是空想——保留前端的 PyTorch-style 抽象，针对 AMD 重写 register pinning、多形状 LDS swizzle、8-WAVE PING-PONG 调度、L2/LLC 联合 chiplet swizzle 这四个后端模块，就能在 MI325X/MI355X 上同时打平 AMD 手写汇编 AITER、且在 d=64 attention 与 GQA backward 等长尾 workload 上反超 1.2×–10×；这不仅打破了"CUDA moat"，也给所有未来异构 AI 加速器（Apple、Intel、TPU、国产 NPU）的 kernel DSL 设计者提供了"前端通用 + 后端专属"这条可复刻的工程范式。**

---

*报告生成时间：2026-06-18 · 基于 PDF 全文 13 页逐页精读 · 引用页码均对应 PDF 物理页 · 仓库链接、OpenReview ID、AITER 集成情况均与论文摘要一致。*
