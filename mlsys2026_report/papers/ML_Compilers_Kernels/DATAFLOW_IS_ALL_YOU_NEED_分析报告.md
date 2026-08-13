# 论文分析报告 ·《Dataflow Is All You Need》

> 本报告是对 MLSys 2026（Industry Track）SambaNova 团队论文 *Dataflow Is All You Need* 的中文深度解读。
> 同目录下的 `DATAFLOW_IS_ALL_YOU_NEED.pdf` 即原文，本文页码引用以原文为准。

---

## 0. 元数据

| 字段 | 内容 |
|---|---|
| **题目** | Dataflow Is All You Need |
| **作者** | Darshan Gandhi, Pushkar Nandkar, David Koeplinger, Nasim Farahini, Romy Tsoupidi, Samuel Rydh, Matheen Musaddiq, Tuowen Zhao, Reid Goodbar, Nathan Sheeley, Leon Zhang, Matthew Shaffer, John Long, Han Wang, Angela Wang, Arjun Sabnis, Joshua Brot, Yun Du, Hakan Zeffer, Mingran Wang, **Raghu Prabhakar**（通讯）（共 21 人） |
| **单位** | SambaNova Systems |
| **会议** | The 9th MLSys Conference 2026, Bellevue, WA — **Industry Track** |
| **OpenReview ID** | `7wOOhxkuN8` |
| **官方链接** | https://openreview.net/forum?id=7wOOhxkuN8 |
| **本地 PDF** | `mlsys2026/ML_Compilers_Kernels/DATAFLOW_IS_ALL_YOU_NEED.pdf`（同目录原文同步副本：`mlsys2026_papers/7wOOhxkuN8.pdf`） |
| **页数** | 15 页（含参考文献） |
| **生产部署** | 已上线于 cloud.sambanova.ai |

---

## 1. TL;DR（一句话三句话）

- **问题**：现代 LLM 推理的 *decode 阶段*是 memory-bandwidth-bound 的，GPU 上仅能榨出 ~21% HBM 带宽；megakernel、CUDA Graph、NVSHMEM 等现有缓解手段都无法消除「内核同步」与「计算–通信非重叠」两个根因。
- **方案**：在 SambaNova **SN40 Reconfigurable Dataflow Unit (RDU)** 上做软硬件协同设计，提出三项编译器级优化 —— **KernelLooping**（把 32 层 decoder 折成一次 kernel 调用）、**BatchStreaming**（让 batch 内样本跨层流水）、**ScheduleOffloading**（把多步 decode 调度搬到硬件）。
- **结果**：在多种规模/结构（Dense、MoE、混合）的开源大模型上 **达到 75% 以上 roofline**；speculative decoding 端到端提速 **6×**；在 HBM 带宽近似的前提下，SN40-16 对 DGX H100 上 Llama 3.1 70B SD 任务实现 **1.7× 总吞吐**，405B 在 H100 上 OOM 而 SN40 可运行。

> 一句话：**Decode 阶段的瓶颈不是算力，而是同步开销；用原生 dataflow 架构与编译器把同步消掉，就能逼近 HBM 屋顶（roofline）。**

---

## 2. 问题背景

### 2.1 Token 生成的两个阶段

| 阶段 | 性质 | 性能瓶颈 |
|---|---|---|
| Prefill | compute-bound（高算力比） | TFLOPs |
| **Decode** | **memory-bound**（低算力比） | **HBM 带宽 + 同步开销** |

Reasoning 模型（DeepSeek-R1 等）将 output:input token 比拉到接近 10:1，加上 context caching 把 TTFT 压低，**重心进一步压向 decode**。

### 2.2 GPU 的真实利用率有多差

论文图 1（p.2）：8×H100 跑 Llama 3.1 8B 稳态只能达到 ~300 tok/s，**仅占 24 TB/s HBM 带宽的 21%**。增加 GPU 数（2→4→8）也只换来 1.42× 的吞吐，scaling 极差。

图 2（p.2）拆解 TPOT 后发现两个主要凶手：
1. **Kernel boundary 处的强制同步**：每个 PyTorch op → kernel launch → 全局同步。
2. **Allreduce 不与计算重叠**：H100 用 NCCL 做 TP 通信，必须经由 HBM，吃带宽且阻塞。

MoE 场景更恶化（参考 Jiang et al. 2024b）：~68% 时间花在未重叠的 AllToAll/AllGather 上。

### 2.3 既有方案的痛处（论文 §1 综述）

| 方案 | 问题 |
|---|---|
| **Megakernel**（Spector 2025、Wu 2025、Aimuyo 2025） | 只面向单 GPU，未充分覆盖 collective；kernel 内仍要走 HBM；用 ≥200µs 全局内存 counter 做依赖跟踪，sub-ms 推理时占预算 20%+ |
| **CUDA Graph** | 静态图，与 MoE 动态 expert routing、speculative decoding 的运行时分支不兼容 |
| **NVSHMEM** | GPU↔GPU 仍只能经 global memory（HBM），bandwidth 敏感的 decode 雪上加霜 |
| **Speculative Decoding 在 H100** | 图 1b：跑 70B target + 8B draft，**72% 的时间花在 draft model**；CPU 在 draft↔target 间频繁介入，进一步掉性能 |

→ **结论：要根治这两个开销，必须换执行模型 —— Dataflow Is All You Need.**

---

## 3. SN40 RDU 架构速览（§3, p.4）

### 3.1 拓扑

```
  ┌──────────── SN40 socket (2-die, TSMC 5FF) ───────────┐
  │  Core ──┐  TLN  ┌── Core         (4 cores / socket) │
  │  Core ──┘       └── Core                            │
  │  ↕ Die-to-die                                       │
  │  HBM x4 (64 GB total, 1.6 TB/s)                     │
  │  DDR x6 (1.5 TB, >100 GB/s)                         │
  │  P2P 链路 (chip-to-chip, on-chip-mem 直通)          │
  └──────────────────────────────────────────────────────┘

  单 socket 算力：638 TFLOPS BF16；520 MB on-chip SRAM
```

### 3.2 单核内部

每个 socket 含 **1040 PCU + 1040 PMU**，通过可编程 **RDN（Reconfigurable Device Network）** 互联，由 **AGCU** 桥接到本地/远端内存：

| 单元 | 角色 |
|---|---|
| **PCU** (Pattern Compute Unit) | systolic + streaming 计算；可拼成 systolic array 跑 GEMM |
| **PMU** (Pattern Memory Unit) | 512 KB 程序管理 SRAM + 张量变换/地址生成硬件 |
| **AGCU** (Address Gen & Coalescing Unit) | 访问 HBM/DDR/host/远端 SN40，原生支持轻量 P2P 协议；**包含硬件图编排，能在不打扰 host 的情况下 launch kernel** |
| **TLN** (Top-Level Network) | 跨核与 IO 互联 |
| **RDN** | 单核内 PCU/PMU 间的可编程数据交换 |

### 3.3 编译产物：PEF

编译器把模型编为 **kernels**（融合后的子图）+ **schedules**（kernel 调用序列），二者打包成一个 **PEF**（Processor Execution Format）二进制文件。

### 3.4 与 GPU 的本质差别（一句话）

> **GPU**：靠全局内存（HBM 上的 counter）做同步 ⇒ 微秒级，逐设备/跨设备同步还要烧带宽。
>
> **SN40**：原生 dataflow，靠 **轻量控制握手**（lightweight control handshake） 做同步 ⇒ 纳秒级，且能与计算流水重叠到「零开销」。

这一条决定了下面三种优化能否真正落地。

---

## 4. 三大优化（论文 §4）

> 三者层次清晰：**KernelLooping** 削掉单 token 内的同步；**BatchStreaming** 削掉 batch 内 layer-边界的同步；**ScheduleOffloading** 削掉跨 token 的 host 介入。

### 4.1 KernelLooping（p.5–7）

**动机** —— 用 Llama 3.1-8B 举例：
- DGX H100 + TensorRT-LLM 上一个 decoder layer 被切成 **K1–K10 共 10 次 kernel 调用**（图 4a），32 层 ⇒ 320 次 launch + 320 个同步点。
- SN40 已经能把整个 decoder fuse 成单 kernel `K0` —— 但 32 个 K0 之间还有同步。

**做法**：把「同一 kernel 重复 N 次」直接重写为「一个带 outer loop 的 kernel」（图 4b iii）。最终 Llama-8B 的 decode 路径只剩 4 次 kernel 调用：`embedding → all_decoders_nosync → classifier → sampling`。

**编译实现**：
1. **Pattern Matching**：dataflow 分析判定调用是否是合规重复（输入/输出张量沿调用链按统一 pattern 出现，可被合并）。
2. **Transformation**：
   - 拼接/reshape 输入张量；
   - 引入外层迭代变量 `n` 并下沉到 kernel body；
   - 决定中间 buffer 放在 SRAM/HBM/DRAM 中的哪一级（图 6 给的例子里，中间结果用 on-chip `buf0`/`buf1` 双缓冲，独立的 weights `{w0,w1,w2}` 沿 `n` 维迭代访问）。

**为什么 GPU 复刻不了**：megakernel 也想做类似的事，但其「内部依赖跟踪」基于全局内存 counter，单次同步开销 ≥200 µs；SN40 的 PCU/PMU/AGCU 之间是控制握手 + on-chip 数据流，开销纳秒级且与计算流水重叠。

**额外副产物**：**HBM 通道始终保持忙碌**（图 11 的 HBM 带宽 trace：橙色 KernelLooping 把利用率拉满，蓝色 baseline 周期性掉到 0），因为消除了 layer 间的「热身死时间」。

**配套代价**：weights 必须按高维（layer 维度）打包；论文 §6 提到「checkpoint preprocessing」基础设施会读编译器元数据自动重排，**用户侧加 Python 装饰器即可**。

### 4.2 BatchStreaming（p.7–8）

**动机** —— KernelLooping 之后，batch 维度仍是同步的：layer N 必须等所有样本在 layer N-1 完成才能开工（图 7a）。这相当于**每层都付一次 pipeline warm-up**，对 1B/3B 这类小 draft 模型尤其致命（每层只有几 µs）。

**做法** —— 在两个 decoder 之间引入 **LoopBuffer**（一个由多个 PMU 拼出来的 logical scratchpad，按 batch 容量为单位）：
- decoder A 一旦处理完 sample 0，就把结果写进 LoopBuffer 的 sample-0 槽位；
- decoder B 一旦看到 sample 0 的数据 ready 就开始算它，**完全不等 sample 1–N**（图 7b）。
- 写/读由数据可用性驱动，**没有跨 layer 的全局同步**，但 read-after-write 依赖由硬件保障（图 8）。

**性能曲线**（图 12）：1B/3B/8B Llama 在 BS=2/4/8/16 上都有显著加速；BS=8 是甜蜜点；BS=16 时 1B/3B 因每层 warm-up 占比已经稀释而轻微回落。compute-to-memory 比越高（如 8B），收益相对越小。

**为什么 GPU 复刻不了**：GPU 上 producer/consumer 间的同步必须经内存总线，开销远超 layer 本身的几 µs；dataflow 体系下 handshake 与 compute 重叠到「免费」。

### 4.3 ScheduleOffloading（p.8–9）

**动机** —— 一份 PEF 中的 decode schedule 默认只生成 1 token；要生成 M 个 token 就得 M 次回 host 串接。host 的介入 = dataflow 体系的 Achilles heel。

**做法** —— SN40 runtime 支持 **schedule 动态拼接**：
- 应用告知 runtime「再 unroll M 次」；
- runtime 校验 schedule 间的输入/输出形状/类型，patch 内存地址；
- 整段 M-step decode 由硬件 graph orchestrator 直接编排（图 9 右）。

**对 speculative decoding 尤其友好**：draft 与 target 各自独立 unroll；M 是运行时决策，**与 CUDA Graph 静态化不兼容的 MoE 动态 expert routing 与 SD 启发式都能被覆盖**。论文实测 M ≈ 20 之后增益饱和；schedule 元数据不过几 MB，对 100s GB HBM 来说基本免费。

**收益曲线**（图 13）：在 k=9（每步 9 个 draft token）下，draft 越小、batch 越小，schedule offloading 的相对收益越高 —— 因为「相对分母」（host 占比）大。

---

## 5. 评测与性能（§5）

### 5.1 平台与模型矩阵（p.9，表 2/3）

| 模型 | 精度 | 角色 |
|---|---|---|
| GPT-OSS 120B | FP8 | target |
| DeepSeek R1 671B | FP8 | target |
| Llama 4 17B128E | FP8 | target |
| Llama 3.1/3.3 8B–405B | BF16 | draft / target |
| Llama 3.2 1B / 3B | BF16 | draft |
| Mixtral 8×7B | BF16 | target |
| Qwen 2.5 0.5B / 72B | BF16 | draft / target |

| 平台 | sockets | 总 BF16 TFLOPs | HBM BW |
|---|---|---|---|
| DGX H100 | 8 | 8000 | **24.0 TB/s** |
| **SN40-16** | 16 | 10208 | **25.6 TB/s** |

> H100 与 SN40-16 的 HBM 带宽几乎相等 —— 这是后续比较「是不是公平」的关键校准点。

### 5.2 单项优化收益

| 优化 | 收益 | 备注 |
|---|---|---|
| **KernelLooping** | geomean **1.6×**；Qwen2.5-72B 大批量近 **2×** | 横扫各种规模/结构/上下文长度（图 10） |
| **BatchStreaming** | Llama 1B/3B/8B 上 BS=2→8 单调上升；BS=16 轻微回落 | 8B 比 1B 收益略小（compute/mem 比更高） |
| **ScheduleOffloading** | k=9 下 draft 模型可大幅提速（图 13） | 小 draft、小 BS 收益最大 |

### 5.3 端到端 vs DGX H100（图 14，p.11）

任务：Llama 3.1 70B + 8B draft 做 SD（k=9）；Llama 3.1 405B + 8B draft（H100 OOM）。

- SN40-16 baseline → 加 KL → 加 BS → 加 SO，**逐层叠加** 收益清晰可见；
- vs Optimized DGX H100：在 70B SD 上 **60–80% 提速**；
- 405B：H100 直接 OOM，SN40 正常跑；
- **roofline 达成率 45–78%**，模型越大、效率越接近屋顶（70B → 405B 提升）。

> 论文坦白 H100 的估算公式（Eq. 4）忽略 SD 长序列退化，**对 H100 偏乐观**；实际 GPU 表现可能比图中更差 —— 也就是说论文给出的差距是**保守估计**。

### 5.4 时间分布（图 15）

把 SD 一步分成 4 块：`Roofline / SN40-Memory / SN40-Excess / Host`：
- BS 越大 → SN40-Excess 占比上升（compute-bound 抬头 + IO 开销）；
- 模型越大 → Host & Excess 占比下降（HBM 利用率改善 + SD 算法时间常数化，与模型规模无关）；
- k 越大 → Roofline 占比下降（更多时间花在 draft 上），但能换更高的 acceptance ⇒ 减少 target 调用次数。

### 5.5 Batched SD 综合表（表 4，p.12）

| 配置 | k=5 BS=1 | k=9 BS=1 | k=9 BS=16 | 趋势 |
|---|---|---|---|---|
| Llama 70B-1B (SS=4k) | 3.5× | **4.7×** | 3.8× | 大 ratio + 小 batch + 大 k 最香 |
| Llama 70B-8B (SS=4k) | 2.8× | 3.5× | 2.6× | 8B 太大，draft 不再「轻」 |
| Llama 405B-3B (SS=4k) | 4.0× | **6.1×** | 5.1× | 405B/3B 的 ratio 最佳，端到端 6× 来自这里 |

**四个杠杆的 trade-off** 一目了然：
1. **draft:target 比例越大 → 加速越大**（70B/1B > 70B/8B）；
2. **batch size 越大 → 加速衰减**（draft 模型 KV cache 占比变高，draft 与 target 性能差距缩小）；
3. **序列越长 → 加速衰减**（同上：draft 的 KV cache 长得更快）；
4. **k 越大 → 加速通常越大**（更多 step 走 draft），但 batch 大时 k 的边际收益变弱。

---

## 6. 思想精读 / 启示

### 6.1 「同步」才是 decode 的真正瓶颈

绝大多数关于 decode 的工作都把账算在「kernel launch overhead」「communication cost」「带宽」上。这篇论文把它们归约到一个根：**同步原语的物理开销与可重叠性**。
- GPU 的同步 = HBM counter 读写 = 数百 µs，且与计算同信道竞争；
- SN40 的同步 = on-chip 控制握手 = ns 级，与计算异信道。

只要这条根不变，megakernel/CUDA Graph/NVSHMEM 这些「在 GPU 体系内」的努力就只能「**逼近**」dataflow 体系的天花板，不可能**触达**。这是论文标题 *Dataflow Is All You Need* 的修辞里藏的硬主张。

### 6.2 三层「去同步」的方法论可移植

抛开 SN40 硬件细节，三个优化背后是一套 **decode 系统设计的去同步级联**：

| 同步层级 | 优化 | 类似的 GPU 思路 |
|---|---|---|
| **kernel-call 层** | KernelLooping | megakernel + outer loop |
| **layer-边界 + batch 层** | BatchStreaming | inter-layer pipeline / persistent kernels |
| **token / step 层** | ScheduleOffloading | CUDA Graph / persistent control plane |

任何想给 GPU decode 体系做 systems 工作的同学，都可以用这套分层框架去 audit 自己的栈：**哪一层的同步还在 host 或 HBM 上？**

### 6.3 P2P-on-chip 是 collective 通信的「正确抽象」

NVSHMEM / NCCL 的 collective 必须经 HBM；SN40 的 P2P 直通片上 PMU。这一点对 **TP/MoE 都是结构性优势** —— allreduce 不再吃 HBM 带宽，把带宽留给 weights 与 KV cache（图 5 的精髓）。GPU 阵营若要追平，要么靠 NVLink 域内的「片上 SRAM 直通」（如 H100 SHARP 一类），要么需要更激进的 in-network reduction。

### 6.4 Speculative Decoding 在 GPU 上「不太香」其实是被同步税吃掉了

图 1b 把这件事讲得最直白：8B draft 在 8×H100 上吃掉 72% 时间，因为 draft 模型小、kernel launch 占比高、CPU 介入频繁。SN40 三件套把这三块同步税一并消掉，所以 SD 在 SN40 上的边际收益反而更大（端到端 6×、对 H100 1.7×）。

> **推论**：一个加速器对 SD 的友好度，本质上由「小模型场景下的 launch/同步开销占比」决定，而非纯看 TFLOPS/HBM。

---

## 7. 局限与开放问题（批判读法）

1. **专有硬件 + 闭源编译器**：SN40 RDU 与其编译器栈（PEF、Pattern Matching pass）均不开源，论文层面无法复现；社区只能借鉴**思想**。
2. **基线选择窄**：仅对比 DGX H100 + TensorRT-LLM，没有与 H200 / B200 / MI300X / TPU v5e 等同代加速器横向比较，亦未与最新 megakernel 工作（Mirage、Hidet、Flash-Decoding++ 等）做端到端对比。
3. **GPU 估算偏乐观（作者自承）**：Eq. 4 的 H100 估算忽略 SD 算法本身的开销与长序列退化 —— 也就是说**真实的 GPU 性能可能比论文图中更低**。这对 SN40 是「保守差距」，对 NV 阵营是「下限」。
4. **配套工程量未充分披露**：KernelLooping 要 checkpoint 重打包、BatchStreaming 依赖 LoopBuffer 在 PMU 间的拼接、ScheduleOffloading 依赖 runtime patch 内存地址 —— 这些都是「平台工作量」，论文用一节生产部署经验粗略带过，工程债的真实规模无法判断。
5. **未涉及训练与超长序列**：全文聚焦 inference decode；训练（含 RL/SFT 的 backward）以及 1M+ token 上下文场景未覆盖。
6. **能效 / TCO 未量化**：标题与摘要谈到「PerfTCO」，但论文没给 W/token、$/Mtoken 之类的硬数字。

---

## 8. 关键术语速查表

| 术语 | 含义 |
|---|---|
| **TPOT** | Time Per Output Token（稳态生成 1 token 所需时间） |
| **TTFT** | Time To First Token（prefill 至首 token） |
| **k** | Speculative Decoding 中每步 draft 生成的 token 数 |
| **AR** | Acceptance Rate = (a+1)/(k+1)，target 接受 a 个 draft token 时的接受率 |
| **Roofline (Eq. 3)** | `Perf_RL = samples · HBM_bw / (weights_size + KV_cache · samples)` |
| **RDU** | Reconfigurable Dataflow Unit，SN40 的硬件抽象 |
| **PCU** | Pattern Compute Unit（systolic + streaming 算子） |
| **PMU** | Pattern Memory Unit（512 KB 可编程 SRAM + 张量变换硬件） |
| **AGCU** | Address Generation & Coalescing Unit（访存桥 + P2P + HW graph orchestrator） |
| **TLN / RDN** | Top-Level Network / Reconfigurable Device Network |
| **PEF** | Processor Execution Format，SN40 编译产物（kernels + schedules） |
| **KernelLooping** | 把同名 kernel 的 N 次重复调用编译为带 outer loop 的单 kernel |
| **BatchStreaming** | 用 LoopBuffer 让 batch 内样本跨 decoder 层流水线，消除 layer barrier |
| **ScheduleOffloading** | 把 M 步 decode 调度链 offload 给硬件 orchestrator，CPU 退出每 token 介入 |

---

## 9. 关键页码索引（便于回原文）

| 主题 | 原文位置 |
|---|---|
| GPU 21% HBM 利用、72% draft 占比 | Figure 1, p.2 |
| TPOT 拆解 | Figure 2, p.2 |
| Speculative Decoding 接受率表 | Table 1, p.4 |
| **Roofline 公式 Eq. 3** | p.4 |
| SN40 架构图 | Figure 3, p.4 |
| KernelLooping 调度对比与伪代码 | Figure 4, p.5 |
| AllReduce 与 Down GEMM 流式重叠 | Figure 5, p.6 |
| KernelLooping 编译变换示例 | Figure 6, p.7 |
| BatchStreaming 时序图 | Figure 7, p.7 |
| LoopBuffer 实现 | Figure 8, p.8 |
| ScheduleOffloading 示意 | Figure 9, p.8 |
| 模型 / 平台配置 | Tables 2, 3, p.9 |
| KernelLooping 全模型加速比 | Figure 10, p.10 |
| HBM 带宽 trace（base vs KL） | Figure 11, p.10 |
| BatchStreaming 加速比 | Figure 12, p.10 |
| ScheduleOffloading 加速比 | Figure 13, p.10 |
| **SN40 vs H100 vs Roofline** | Figure 14, p.11 |
| 时间分布拆解 | Figure 15, p.11 |
| Batched SD 综合表 | Table 4, p.12 |
| 部署经验（编译/运行时/checkpoint） | §6 Deployment Experience, p.12 |

---

## 10. 一句话点评

> 这是一篇 **「先架构后算法」** 的工业论文：所有结论都依赖 SN40 的 dataflow 硬件原语，但它给整个 LLM 推理社区贡献了一个非常清晰的视角 —— **decode 阶段的「最后一公里」性能由同步原语的物理实现决定，而非 TFLOPS、HBM 容量或单点优化**。GPU 阵营若想关上这道差距，恐怕需要在「片上 SRAM 直通的 collective」与「硬件级 graph orchestrator」上做更激进的取舍。

---

*生成时间：2026-06-18 · 作者：基于本地 PDF 全文抽取分析（pypdf 6.13.2 提取 15 页）*
