# 论文分析报告 ·《SHIP: SRAM-Based Huge Inference Pipelines for Fast LLM Serving》

> 体系结构 · 存储层级 · LLM 推理系统
> 来源：MLSys 2026 (Industry Track)
> 作者：Andrew Bitar, Aravind Vellora Vayalapra, Baorui Zhou, Matt Boyd, Charlie Wang, Sahil Parmar, Eugene Sha, Gautam Rayaprolu, Peter Hicks, Alex Bowe, Roberto DiCecco, Santosh Raghavan, Evan Patrick, Josip Smolcic, David Han, Kris Kang, Andy Rock, Josh Hay, Mohamed Eldafrawy, Mikhail Kandel, Daulet Zhanguzin, Omar Kilani, Liming Gong, Andrew Paprotskyi, Arash Taheri-Dezfouli, Josh Fender, Andrew Ling
> 通讯：abitar@nvidia.com（论文写作时已并入 NVIDIA — 此前在 Groq 完成本工作）
> OpenReview ID：IZaXDwDtL1

---

## 0. 元数据

| 项目 | 内容 |
|---|---|
| 标题 | SHIP: SRAM-Based Huge Inference Pipelines for Fast LLM Serving |
| 会议 | MLSys 2026, Industry Track, Bellevue, WA, USA |
| 主题归类 | Hardware Accelerators / LLM Serving / Memory Hierarchy |
| 关键硬件 | Groq LPU (Language Processing Unit) v1，GlobalFoundries 14 nm，230 MB on-chip SRAM，C2C 总带宽 235 GB/s（11 个逻辑端口） |
| 部署规模 | Groq 公有云：单实例 72 ~ 数千颗 LPU；每天服务“数千亿 tokens” |
| 关键模型 | Qwen3-32B、Qwen3-235B-A22B、gpt-oss-120B、Llama 3.3-70B、DeepSeek-V3 |
| 对比基准 | NVIDIA B200 DGX (TP=8)，SGLang v0.5.8，vLLM |
| 总页数 | 16 页（含附录与参考文献） |
| 章节结构 | §1 Intro / §2 Background / §3 Motivation / §4 Very Large Scaling / §5 Memory Capacity Management / §6 Dynamic Pipeline / §7 Discussion / §8 Related / §9 Conclusion |
| 核心贡献 | 首个公开的、生产级 SRAM-only LLM serving 部署的系统级 retrospective |

本文延续 Google TPU 论文（Jouppi et al., 2017; 2023）确立的“硬件回顾 + 真实生产数据”叙事范式，把 LPUv1 在 Groq 公有云上的 SHIP 部署经验完整公开。论文写作时下一代 LPU（Samsung 4 nm）已进入量产，这意味着 LPUv1 已是一个“完成态”系统，作者得以基于实际生产 trace 给出可信的工程总结。

---

## 1. TL;DR

SHIP（**S**RAM-based **H**uge **I**nference **P**ipelines）是 Groq 用 LPUv1 构建的、首个面向生产的全 SRAM LLM 推理部署。论文在三条主线上展开：

1. **存储层级革命**：把 weights 与 KV cache 全部放入 on-chip SRAM。SRAM 带宽是 HBM 的 10× 量级（LPU 单芯片 ~10 TB/s 级有效带宽），但容量极小（LPUv1 仅 230 MB），因此必须把模型横向切到数百~数千颗 LPU 才能容纳 frontier LLM。
2. **同步、低直径互连**：QuadFour 拓扑 + 编译期静态调度的 C2C 协议把单跳延迟压到 **300 ns**（GPU NCCL 通常 1~10 µs）；通过 propagated synchronization 让流水线无 host 介入。
3. **低 batch 高效推理**：在小 batch 下（low-batch / high-bandwidth ratio）维持高 FLOPs 利用率，靠 dynamic chunked prefill、fused context-batch、capacity-filling prefill、distributed prefix caching、speculative decoding 等手段把 TTFT 与 TPOT 同时压低。

**结果**：在 Qwen3-235B-A22B 上 SHIP（TP=16, PP=95, 并发=380）相比 SGLang on B200 DGX 在 TPOT 稳定性、绝对 ot/s/u、生产级 traffic mean TTFT 上全面占优，但 peak FLOPs 效率仍低于 NVIDIA。SHIP 系统单 LPU 配套功耗仅 388 W（B200 DGX 单 GPU 1788 W），通过 SRAM-centric 设计省掉 HBM/CoWoS/scale-up switch 等大头，从而在 perf/W 与 perf/TCO 上拉近差距。

---

## 2. 问题背景

### 2.1 LLM 推理为什么是 HBM-bound

现代 LLM serving 把推理切成 **prefill** 与 **decode** 两个工况：

- **Prefill**：处理用户 prompt 中所有 token，可并行，是 compute-bound；服务目标是 TTFT（time-to-first-token）。
- **Decode**：自回归一 token 一 token 生成，是 memory-bound；服务目标是 TPOT（time-per-output-token），即 1 / ot/s/u。

Decode 阶段每生成一个 token 都要把整套 weights + KV cache 从外存搬到片上，单 token 的 operational intensity (OI) 极低。论文 Figure 3 给出在 Qwen3-32B 与 gpt-oss-120B 下，OI 随 batch 与 context length 变化的曲线：

- Dense 模型（Qwen3-32B）：self-attention 的 QKᵀ 与 PV 在 decode 时 OI 与 batch 无关，导致 OI 形成上界（CL=131K 时仍只有 ~50）；
- MoE 模型（gpt-oss-120B）：tokens 路由到不同 experts，batch 难以摊分到所有 expert weight，要 batch ≥ 1000 才能让 OI 抬到几百。

而 HBM-based 加速器要逼近 compute saturation，OI 通常需要在数百~上千。这意味着传统 GPU 在低 batch 或长 context 下的 decode 永远是 memory-bound 的。**通过加 batch 来攒 OI 的思路，被 self-attention 的 batch-invariance 与 MoE 的 expert sparsity 两次封印。**

### 2.2 Reasoning model 让问题恶化

reasoning model 的 thinking traces 让单次请求的 output token 数比非 reasoning 高 2~10×（Artificial Analysis, 2025）。从用户体感看，TTFT 实际由 ot/s/u 决定（“等模型想完”的时延 ≈ 思考阶段长度 ÷ ot/s/u）。这把推理瓶颈进一步推向 memory bandwidth 一侧。

### 2.3 SRAM-centric 加速器的浪潮

论文把 SRAM-based 推理放进一个更大的格局：

- **Cerebras WSE-3**：wafer-scale，900K cores × 48 KiB SRAM，wafer 上 mesh NoC，wafer 间 RDMA-over-Ethernet。
- **Tenstorrent / Graphcore IPU / SambaNova**：dataflow + 大 SRAM 思路。
- **Chiplet Cloud (Peng et al., 2023)**：chiplet 形态的 SRAM-based 推理 supercomputer。
- **Groq LPU**：本论文主角，TSP（Tensor Streaming Processor）架构。

GPU 阵营（NVIDIA Blackwell、AMD MI300X、Google TPU、Meta MTIA、AWS Trainium/Inferentia）的共同假设是：模型 weights/KV cache 放外置 DRAM/HBM。SHIP 是第一个把这条假设彻底反转、并跑到“数千亿 tokens/天”的生产部署。

### 2.4 内存层级演化

| 层级 | 典型容量 | 典型带宽 |
|---|---|---|
| Register / SIMD lane | KB | 极高 |
| **on-chip SRAM** | 10–250 MB / die | **TB/s 级** |
| HBM3/HBM3e | 80–192 GB / package | 3–8 TB/s |
| host DDR4/DDR5 | 1–6 TB / node | ~20–100 GB/s |
| SSD / NVMe | TB | GB/s |

LPUv1 把 weights+KV cache 推到 SRAM 这一层，相当于把存储层级整体上移一档。LPU 单芯片 SRAM 230 MB、C2C 带宽 235 GB/s：**容量小到必须用“分布式 SRAM = 分布式 weights”**——即用数千颗 LPU 拼出一个模型实例。

### 2.5 P:D ratio 与 context length 的随机扰动

Figure 2 展示了 Groq Cloud 真实 trace：在 7 天内，prefill:decode token ratio（P:D）从 0 到 20+ 大幅波动，reasoning vs. non-reasoning 模型的分布也明显不同。任何静态调度策略（固定 chunk size、固定 batch）都难以应对这种动态性。这是 SHIP 设计 dynamic chunked prefill / fused context-batch 的直接动机。

---

## 3. 核心思想 / 方法

SHIP 不是一篇“算法论文”，而是一份系统-编译器-硬件协同设计的“配方”。其核心抽象是：

> **当 SRAM 带宽 ≫ 容量时，与其用大 batch 来摊薄 weight 加载成本（HBM 时代的传统答案），不如用小 batch 配合大规模分布式 SRAM 来同时压低 TTFT 与 TPOT。**

围绕这一中心，论文构建了三条支柱（§3.2 总结为 Low-Batch / Scale / Memory Capacity Efficiency）：

### 3.1 LPU 架构基础（§2.1）

LPU 是 **Tensor Streaming Processor (TSP)** 架构（Abts et al., 2020/2022）：

- 张量被表示为向量集合，水平“流过”一排 Functional Units (FU)；
- 每个 FU 是 320-element SIMD 单元，一周期一指令；
- FU 类型：MXM（高吞吐 MatMul）、VXM（非 MatMul）、MEM（高带宽 SRAM）、SXM（switch）；
- 互连是 **directional, arbitration-free 总线**：东向 + 西向各一条；
- 编译器 **静态调度** 全部指令派发；
- SRAM 访问 deterministic（无 refresh，无 cache 行为）；
- 与外部非 deterministic 组件（PCIe）交互时用一条轻量同步指令重新对齐。

该 determinism 并不止步于单芯片，而是经由 C2C 协议向上扩展到 **数千颗 LPU 同步执行**：自然时钟漂移用浅 FIFO + 静态 deskew 指令补偿；**LPU 自己充当 router**，没有专用网络芯片，使 compiler 可以同时 schedule 计算和网络 traffic（§2.1 / §4.1）。

每个 node = 8 LPU + dual-socket host + 1 TB DDR4，PCIe Gen4 x16 × 8。

### 3.2 三大支柱

1. **Low-Batch**：高 memory-to-compute bandwidth ratio 让 batch=1 仍能跑出高利用率，因此可以把 latency 直接最小化，而非牺牲 TPOT 换 throughput。
2. **Scale**：把 model partition 跨数千颗 LPU，TP + PP 共同上量。
3. **Memory Capacity Efficiency**：用 PagedAttention、prefix caching、speculative decoding 把 SRAM 容量榨干。

### 3.3 大规模并行：TP + PP 联合

#### 3.3.1 Tensor Parallelism (§3.2)

TP 在 SHIP 中是“广义 TP”，含 head parallel / context parallel / expert parallel / self-attention data parallel。每个 decoder layer 被切到多颗 LPU，目的是：
- 降低单 layer 计算时延；
- 扩大可用 SRAM 容量（weights + KV cache）。

代价：collective communication（AllReduce、AllGather）作为 sequential dep 暴露在关键路径上。低 batch 没法靠“另一个 batch 的独立计算”来掩盖，因此 collective 延迟的相对成本被放大（Davies et al., 2025）——这是 §4.2 中 collective 优化必须存在的根本原因。

#### 3.3.2 Pipeline Parallelism

当 TP 已经无法继续扩展（带宽、radix 见顶）时改用 PP：

- PP 提供更多 weight 容量、却 **几乎不缓解 KV cache 紧张**：每加一级 stage，要为新的 in-flight micro-batch 再开一份 KV cache（speculative decoding 部分缓解，详见 §5.3）；
- prefill 不需要多套 KV cache 来 saturate pipeline，因此可以把整个 SHIP 都压上单个 prefill job；
- PP 的关键风险是 stage 不平衡引入 bubble。LPU 的 determinism 让 compiler 在 compile time 就把 stage latency 配平。

#### 3.3.3 异构层混合的平衡

现代模型多种异构层共存（SWA + 标准 attention，dense + sparse MoE）。SHIP 通过 **per-stage TP size 不同** 来分配 compute/storage 资源：每个 partition 用多少 LPU 与该 partition 的 compute/storage 需求成比例（§3.2、§4.1）。

### 3.4 网络拓扑：QuadFour（§4.1）

QuadFour 是为 SHIP 设计的低直径、有序、可缩放、容错的拓扑：

- **Node 内**：8 LPU 全连接 K8（intra-node passive copper），单跳即可 all-to-all。
- **Node 间**：每个 node 用 32 条外部链路均分到前向 4 个邻居 + 后向 4 个邻居（每个邻居 4 条链路）。
- 整体形态：节点序列上的“linear sequence + 前后各 4 跳”。
- **bisection bandwidth**：853 GB/s（在一个 partition 内）。

属性：

- **同时为 TP 与 PP 服务**：partition 内用密集 intra-/inter-node 链路跑 collective，partition 间链路跑 pipeline transfer。
- **低直径**：72-LPU TP partition 直径 3 hops（16.56 GB SRAM）；136-LPU TP partition 直径 5 hops（31.28 GB SRAM）。每跳仅 300 ns。例：72-LPU 单层 partition 可承载 Qwen3-235B-A22B 的 weights + 1380 个 4K context 用户 KV cache。
- **可滑动**：partition 不绑定 rack 边界，72-LPU partition 可跨 rack。
- **propagated synchronization**：同步像“波”一样沿 pipeline 推进——前 partition 同步并算 step n+1 的同时，新 partition 加入同步域接 step n 的中间张量，前 partition 退出同步域开始下一次推理。这套机制省掉 host 调度，避免 pipeline bubble。
- **容错 (Resilient)**：partition 可绕过最多 3 个连续 down node。bandwidth 与 skip 数关系（Table 1）：N=0 → 427 GB/s, N=1 → 256, N=2 → 128, N=3 → 43。Transient error 重放对应 token；persistent error 把 node 下线并旋转 partition；多份 partition weight 缓存在 host DDR 加速旋转。

### 3.5 高效 collective（§4.2）

LPU 编译器对 collective 的“静态调度”带来三个优势：

1. **low-latency transfer**：lock-free，单跳 300 ns，远好于 NCCL/NCCLX/UCCL 的 1–10 µs（Hu 2025、Si 2025、Zhou 2025）。
2. **compute-aware pipelining**：Figure 6b 展示 8 芯片上 inner-dim split 的 MatMul + AllReduce，C2C 与 MAC 几乎完全重叠。
3. **routes 选择**：根据 tensor size 在 RedBcast / TiledRedBcast / PAARD（Ma 2021）之间切换；64-chip AllReduce 在 32 KiB 即达 50% 带宽饱和，80 KiB 达 90%——而 LLM serving 上 collective tensor 通常落在 32–256 KiB，正中靶心。

LPUv1 C2C 235 GB/s 远小于 B200 NVLink 1800 GB/s，但低延迟 + 静态调度让 LPU 在 small-tensor regime 全面胜出。

### 3.6 内存容量管理（§5）

#### 3.6.1 In-house PagedAttention (§5.1)

- 与 vLLM 风格一致，把 KV cache 切成 page，按需分配；
- page size 常用 128–512 tokens：小 page 减碎片但 mask encoding 大，大 page 反之；
- runtime 通过 host CPU 把一个紧凑的 mask encoding（描述哪些 page 是 active）打入 pipeline 头，每级 stage 把 mask 转发给下一级；
- 对应生产观测：reasoning / non-reasoning 各有 92% / 97% 的 request 总 token 数 < 8 Ki，说明大部分时间 context 都没被填满，pageable 设计正好捡回这些容量。

#### 3.6.2 Distributed Prefix Caching (§5.2)

SHIP 把 prefix cache 跨两层存储：

| 存储 | 容量 | 带宽 |
|---|---|---|
| SRAM (gpt-oss-120B 实例，72 nodes) | 69.7 GB（去掉 weights 后） | 10,616 TB/s（与 weights 共享） |
| DDR4（每 node 1 TB × 72 nodes） | **72 TB** | 18.4 TB/s |

- 数据结构：**BlockTrie** + 固定大小 token chunk 的 rolling hash，block size 与 PagedAttention page size 相等；
- SRAM 命中：**near-zero TTFT**（若全 prompt 命中可直接进 decode）；
- DRAM 命中：runtime 异步 schedule 把对应 page 拷回 SRAM；
- LRU 淘汰；
- TTL 量级：**hours**，因 DRAM 容量极大；
- 实测 Figure 7：DRAM 命中率长期稳定在 50–75%；
- 跨 organization 严格隔离 cache（多租户隐私）。

对比 HBM-only 系统：要复刻 SHIP 的 DRAM-as-prefix-cache 容量优势，需要 data parallel 多份 model 实例，但 prefix routing 必须一致（Cao 2025），调度更复杂。

#### 3.6.3 Speculative Decoding 当作“KV cache 压缩” (§5.3)

SD 的传统价值是降 TPOT；在 SHIP 中它被当成 **memory-capacity 优化**：

- 每个 KV cache 平均生成多个 token → 维持同样 FLOPs 利用率所需的总 KV cache 减少；
- draft 在 SHIP 中作为额外 pipeline stage 插入；
- Figure 8（Llama3.3-70B target）：1B/3B draft 显著提速，8B draft 反而拖累 efficiency；这与 GPU 上 SD 在高 batch 短 context 下 hurt efficiency 的结论一致（Liu 2024b、Su 2023）；
- 对 long context（self-attention 主导）SD 更有效（Sadhukhan 2025）。

### 3.7 动态平衡 pipeline（§6）

token 复杂度由两件事决定：(1) prefill or decode；(2) context position（越靠后 self-attention 越长）。pipeline 任何 stage 的微小 latency 抖动都会 propagate 成 bubble。SHIP 的应对：

#### 3.7.1 Dynamic Chunked Prefill

- chunk size 可小到 1 token（SRAM 让 self-attention OI 在 chunk=1 即可饱和：Llama3.3-70B 8、Qwen3-235B-A22B 16、DeepSeek-V3 128）；
- runtime 根据观测的 P:D ratio 与 prompt size distribution 动态选 chunk size；
- 通过 “task ID” 把同 task 的 token 融合成更大 chunk，attention mask 在线生成；
- chunk 可跨多个 non-adjacent pipeline stage，最大 chunk 可吃满整条 pipeline。

这与 SGLang/TensorRT-LLM 的固定 chunk size 形成鲜明对比——固定 chunk 必然要在 TTFT、TPOT、throughput 间硬选三选二（Agrawal 2023）。

#### 3.7.2 Fused Context-Batch

为消除“context position 不同 → stage latency 不齐 → bubble”：

1. 给定 TP×PP 系统，依据 max context length C 与 capacity/TPOT 目标选 batch size B；
2. 把 B 与 C 维度 fuse，切成 N 个 chunk（N 是任意 context length 下的 max batch）；
3. runtime 给 pipeline 一个 N×N mask，决定哪些 context-batch chunk fuse 到同一 unified context。

“B_max = ceil(num kv caches that fit / pipeline_depth)”——这是 SRAM capacity 直接限制 B 的硬约束。

直观地讲，fused context-batch 把“短 context tokens 的空闲时间用 batching 填满”，让 pipeline stage latency 严格相等。

#### 3.7.3 Capacity-Filling Prefill

调度器默认 **decode 优先**：
- decode 的限制是 KV cache 容量；
- 当容量不足导致 decode 留出空隙时，用 prefill token 填进去（prefill token 之间可共享 KV cache）；
- prefill-prioritized 会在 decode-heavy traffic 下饿死并造成 TPOT 抖动（§6.1 实验对照 SGLang 印证）；
- decode-priority 让 SHIP 在 SLO 下保证 TPOT 稳定，prefill chunk 仍然能吃满空隙保 TTFT。

#### 3.7.4 MoE Expert Imbalance

MoE 的 PP 执行有“每 stage activated experts 数不同”的天然不平衡：

- 在大 batch 下 expert usage 均匀化；
- 在小 batch（SHIP 工作点）下，**per-token expert 独立执行**反而平衡更好——代价是 expert MatMul OI=1，但 SRAM 高带宽足以撑住 OI=1。
- 下一代 LPU 容量增大后，B 更大，需要新策略。

---

## 4. 实现 / 工程细节（§7.1）

### 4.1 硬件

- **LPUv1**：14 nm GlobalFoundries，230 MB on-chip SRAM，11 个 logical C2C 端口、聚合带宽 235 GB/s；
- **Accelerator card**：dual-width / full-height / three-quarter-length PCIe Gen4 x16；含 12V→1V VRM、4 × QSFP（inter-node/rack 光链）、7 × intra-node 无源 copper；
- **Node**：4U server，8 块卡 + 双 socket AMD CPU + 1 TB DDR4 + 100 GbE NIC + 1 TB SSD + BMC；node 内 8 LPU all-to-all 由 passive twin-axial copper；
- **Rack**：9 nodes（72 LPU）；inter-node 用 quad-25G NRZ CDR QSFP；
- **功耗**：LPUv1 系统 **388 W / LPU**，B200 DGX 1788 W / GPU。

### 4.2 软件栈

- **编译器 + 运行时**：cycle-accurate 静态调度全局指令；compiler 决定计算 + 网络 traffic；
- **运行时**：管理 PagedAttention page 分配、prefix cache（BlockTrie + rolling hash）、speculative draft、dynamic chunk size、fused context-batch、partition rotation；
- **host 角色**：仅在 pipeline 起点投递请求与终点收集输出，**中间 stage 由 LPU 自调用**（与 host-driven GPU stack 形成对比）；
- **fault handling**：transient → token 重放；persistent → node 下线，partition 旋转，host DDR 缓存多 partition weights 加速旋转。

### 4.3 Power-Aware Scheduling (§7.1.2)

由于 LPU 程序执行 deterministic，作者从 RTL + 硅测据校准出 **instruction-level power model**：

- 把 instruction trace 转成 cycle-accurate power profile；
- compiler 根据 power 预算选 schedule，最大化 compute/I/O 利用率；
- 通过 reorder 独立指令或插 NOP 平滑电流瞬态、抑制 voltage droop；
- static-time inferrable sparsity 进一步降低 activity 估计。

代价：实际部署时 chip-to-chip variation 与 live temperature 让 model 校准颇耗时间，需保守 margin。

### 4.4 Synchronous Execution 的好处与挑战（§7.2）

**好处**：

- 静态依赖之间无同步等待；
- 计算与通信细粒度重叠；
- deterministic floating-point accumulation：debug 模型精度时直接复现，无延迟代价（GPU NCCL 复现需付额外延迟，Al Awar 2026）；
- 可做 power-aware scheduling。

**挑战**：

- variable context length 与 expert dispatch 等动态结构必须 “映射成 static shape + runtime mask”，再用 LPU 的 gather/scatter 实现；
- prefix caching 把 KV pages 从 host DDR via PCIe 拉回时引入非 deterministic，需要把 re-sync 开销与独立 compute overlap；
- power model 校准花了不少调参精力。

### 4.5 Disaggregated Serving（§7.3）

论文承认 disaggregation 是与 SHIP 互补的方向：

- **Prefill–Decode disagg**（DistServe / Splitwise / Dynamo）：解耦 prefill 与 decode；自然适合 “GPU 跑 prefill + LPU 跑 decode”；
- **Attention–FFN disagg**（Step-3, MegaScale-Infer）：高容量 GPU 存 KV cache，LPU 跑 memory-bound MoE-FFN；NVIDIA GTC 2026 已宣布相关“Inside NVIDIA Groq 3 LPX”形态。SHIP 在 KV cache capacity 上的瓶颈正好被 GPU HBM 补，SRAM 高带宽又让 batch 不需要太大就能高利用率，KV state 在 HBM 与 SRAM 之间来回搬的成本被降低。

---

## 5. 评测（§6.1 + Appendix D.2）

### 5.1 实验设置

- **模型**：Qwen3-235B-A22B（frontier MoE，LPU 与 SGLang 都成熟支持）；
- **SHIP**：TP=16, PP=95, max concurrency=380；
- **SGB200**：B200 DGX × 1, TP=8；mixed chunking; MBTB ∈ {16Ki（默认高吞吐）, 2Ki（低延迟）}；
- **指标**：system throughput (t/s), per-user input/output throughput (it/s/u, ot/s/u)；
- **traffic 维度**：P:D ∈ {…}, CL ∈ {2K, 8K, 32K, 128K}；并叠加两组生产 trace（reasoning：avg 3.4 K / max 99 K；non-reasoning：avg 1.4 K / max 24 K）。

### 5.2 主要结果（Figure 10）

#### System Throughput

- **SGB200 MBTB=16Ki**：在高 P:D 下吞吐高（prefill chunk 充裕，decode piggyback 受益）；P:D 低或 CL 大时下滑严重，因 KV cache 吃光 HBM 容量。NVL72 通过扩大 NVLink TP 域可缓解。
- **SHIP**：在 P:D ≥ 8 与 CL < 4 Ki 时吞吐稳定；dynamic chunked prefill（chunk=1 也高效）让平衡持续维持；CL 增长使 SRAM 容量见顶，但 capacity-filling prefill 在高 P:D 下补救；下一代 LPU 将通过更大 SRAM 与更高 C2C 带宽 + radix 进一步扩展低 P:D / 长 CL 工作点。

在生产 trace 上两系统都因长 CL 请求出现下滑（CL 二次方放大成本），但 **SHIP 相对 peak 的下滑更小，绝对吞吐更高**。

#### TPOT（ot/s/u）

- **SGB200**：preemption preference 给 prefill 以维持 TTFT，导致 TPOT 不稳：
  - 低 P:D 下 decode-only kernel 比 mixed-batch 快，越大 MBTB 越快地排空 prefill → 反而降 TPOT；
  - 高 P:D 下 mixed-batch 居多，小 MBTB 缩短 mixed-batch iter 反而降 TPOT；
- **SHIP**：decode-priority + capacity-filling prefill + fused context-batch → ot/s/u **跨 P:D / CL / 生产 trace 全程稳定**且绝对值更高；
- 这种稳定性对 reasoning 模型尤其关键（思考时延 ∝ 1/ot/s/u）。

#### TTFT（it/s/u）

- **SGB200**：prefill-priority 让 it/s/u 稳定；MBTB=2Ki 时 it/s/u 显著提升，因为 prompt 大多 < 8 Ki，16Ki 的预算反而把别的 prefill 凑进 batch、拉长 iter。
- **SHIP**：固定 traffic 下 it/s/u 稳定；高 P:D 一直强；最高 it/s/u 出现在低 P:D + 8 Ki context（capacity-filling prefill 抓住 KV cache 留下的“洞”填大 prefill chunk）；reasoning trace 下 it/s/u 明显下降（长 CL 二次方占走 compute），但 **mean TTFT 仍低于 SGB200**（包括 reasoning sample）。

### 5.3 单跳延迟与 collective bandwidth（Figure 6）

- 8-LPU/64-LPU AllReduce vs. 8×B200 NCCL：tensor 32 KiB 时 LPU 已到 50% 饱和，80 KiB 到 90%；GPU 在小 tensor regime 输给延迟 overhead；
- LPU collective 在 LLM serving tensor size（32–256 KiB）正中其有利区间。

---

## 6. 思想精读 / 启示

### 6.1 “存储层级革命”：从 HBM-bound 到 SRAM-distributed

SHIP 给出的最深刻命题是：**LLM serving 的瓶颈，与其说是 weights 太大，不如说是 weights 与 KV cache 离 compute 太远**。HBM 时代每 token 都要把数百 GB 数据从外存搬到 die；SRAM-distributed 时代把这些数据**拆碎放进数千颗 die 内部**，同时把芯片间互联做成 deterministic、ns 级跳点的同步 fabric——本质上是把“全局存储层级”重写为“分布式片上存储 + 同步互联”。

这与 Cerebras WSE-3（wafer-scale）走的是同一条路的两个极端：Cerebras 在硅基片层面把 SRAM 与互联“原生融合”；Groq LPU 在 die-级把同样的“片上 SRAM + 静态同步互联”理念扩展成 datacenter-scale。两者都默认 **batch=小、interconnect deterministic、compiler omniscient**。

### 6.2 Compiler-omniscient 是必要前提

SHIP 反复证明：只有当 compiler 静态拥有 cycle-accurate 视野，才有可能把 collective、pipeline、power 调度都做到极致。SambaNova SN40、Tenstorrent、ParallelKittens（CUDA-on-tensor-cores DSL）等也在不同切面尝试拉回“可预测性”。GPU 的 NVSHMEM、Flux（Chang 2024）等在做 partial 的 compiler-aware overlap，但因 dynamic kernel exec time 永远存在 “imperfect overlap”。SHIP 表明：**determinism 不是奢侈品，而是 SRAM-distributed 推理在低 batch 下保 SLO 的“底层假设”。**

### 6.3 Dataflow 思想的回归

LPU 的 TSP 架构、horizontal FU stream、东向/西向数据总线、static schedule，本质上是 **dataflow architecture**（Wavescalar、TRIPS 一脉的现代化）在 LLM 时代的工程化复活。SambaNova RDU 的 Reconfigurable Dataflow 是另一种 dataflow 商业化形态。SHIP 用真实生产数据告诉 GPU 阵营：当 batch=小时，dataflow + 静态调度可以把 collective overhead 推到 hardware fabric 极限以下。

### 6.4 与 ParallelKittens / CUDA Kernel 的对照

GPU 阵营的应对是把 kernel-level fusion 做到极致（Flash Attention、ParallelKittens），用更细致的 kernel 形态把 prefill/decode、attention/FFN 重叠起来。但 SHIP 的论证是：**只要 dynamic kernel launch 与 non-deterministic memory 仍存在，重叠就只能是 best-effort**。这给“纯 GPU 派”留了一个长期问题：能否在 NVL72 + 极致 NCCL 优化下，把 collective 的 latency 推到 < 1 µs / hop？

### 6.5 SRAM-disagg 的二阶推论

§7.3 提出的 attention/FFN disagg + GPU/LPU 异构是非常深刻的二阶设计：

- KV cache（capacity-bound）→ 给 HBM；
- FFN MatMul（bandwidth-bound, MoE 尤甚）→ 给 SRAM；
- 在 SHIP 内 KV cache 容量是 ot/s/u 上限的死结（Fig 10a），把 KV cache 扔到 GPU HBM 解开这一约束，同时让 LPU 的 SRAM 带宽继续吃 FFN。
- 这与 Step-3、MegaScale-Infer 的 attention/FFN disagg 思路一致，且是 NVIDIA Vera Rubin 平台“NVIDIA Groq 3 LPX”的设计前传（Aubrey & Ghodsian, 2026）。

### 6.6 SLO-driven 调度作为一等公民

SHIP 评测把 “system throughput / TPOT 稳定性 / TTFT 稳定性” 三件事都放到对比表上，而不是只看 peak throughput。这与 vLLM、SGLang 文献长期把 throughput 当 “唯一目标”形成对比。SHIP 的隐含信号是：**在生产中，SLO 稳定性比 peak 高一截更重要**——因为 P:D 与 CL 一直在波动，peak 永远是“调好那一刻”的数字。

---

## 7. 局限与开放问题

### 7.1 Peak FLOPs 利用率仍不及 NVIDIA

LPUv1 的 C2C bandwidth/radix 仍是限制 SHIP peak 的关键（Fig 6b 中暴露的 C2C 周期）。论文坦承：next-gen LPU（Samsung 4 nm）才会显著扩容 SRAM、增 C2C 带宽 + radix。

### 7.2 极长 context 的二次方代价

CL → 100K 时 SHIP 的 throughput / it/s/u 都明显下降（reasoning 生产 trace 上尤其明显）。dynamic chunk + capacity-filling 的“留洞”机制在长 CL 主导的 traffic 下被打破——因为长 CL 直接吞掉 compute capacity，而非简单地“留洞”。需要新的 adaptive scheduling heuristic。

### 7.3 SRAM 容量 vs 模型规模的矛盾

trillion-parameter MoE 在 LPUv1 (230 MB / die) 上需要数千 die 才能装下（72-LPU 单层 partition for Qwen3-235B 已是上限），意味着 BOM 与运营成本随模型规模线性放大；HBM 路径在 capacity 上仍然占便宜。

### 7.4 异构 disagg 的工程难度

§7.3 提出的 GPU+LPU 异构 disagg 引入更复杂的 KV cache transport、cross-domain scheduling、failure mode；论文只承诺 future retrospective。

### 7.5 Power model 的 dynamic 适应

power-aware scheduling 在 chip-to-chip 与温度变化下需要保守 margin，部署初期 calibration 成本高。

### 7.6 多租户隔离开销

prefix cache 跨 organization 隔离让 cache hit rate 受限（虽然 50–75% 已不错），但有否更细粒度的 trust 模型让公共 prompt 可共享（如 system prompt）是开放问题。

### 7.7 Speculative decoding 的 draft-target 工程化

draft 的容量、对齐度、acceptance rate 都直接决定 net efficiency；EAGLE-3 (Li 2025)、distillation (Zhou 2024) 是方向但未集成验证。

### 7.8 与 NVL72 / Ironwood / MI300X 的更细粒度对比

论文只比 SGLang on B200 DGX，未对 NVL72 / TPU Ironwood / Cerebras WSE-3 做完整 head-to-head；不同硬件的最优 software stack 选择对结果有较大影响。

---

## 8. 关键术语速查表

| 术语 | 解释 |
|---|---|
| **SRAM** (Static RAM) | on-chip 高带宽、低延迟、低密度存储；LPUv1 单 die 230 MB |
| **HBM** (High-Bandwidth Memory) | 通过 CoWoS interposer 与 GPU/加速器封装，密度大但带宽相对低；典型 80–192 GB/package |
| **DRAM** (DDR4 host) | host-attached 大容量内存；LPU node 1 TB/node，作为 prefix cache 二级 |
| **TSP** (Tensor Streaming Processor) | LPU 架构基础；张量在 horizontal FUs 上 stream，编译器静态调度 |
| **LPU** (Language Processing Unit) | Groq 的 SRAM-centric LLM 加速器；本论文 v1 是 14 nm 基础版 |
| **SHIP** | SRAM-based Huge Inference Pipelines，本论文核心系统抽象 |
| **C2C** (Chip-to-Chip) | LPU 间同步互连协议，单跳 300 ns，全 deterministic |
| **QuadFour** | LPU 间网络拓扑：node 内 K8 + node 间前后各 4 邻居 |
| **TP** (Tensor Parallel) | 单 layer 切跨多芯片，含 head/context/expert/data parallel |
| **PP** (Pipeline Parallel) | 多 layer 串行分到多芯片，每 stage 一份 KV cache |
| **OI** (Operational Intensity) | FLOPs / byte loaded；roofline 模型核心量 |
| **TTFT** (Time-To-First-Token) | 从请求到首 token 的延迟，prefill 主导 |
| **TPOT** (Time-Per-Output-Token) | 每个 output token 时间，decode 主导；ot/s/u = 1/TPOT |
| **P:D ratio** | prefill / decode token 比例，traffic 关键波动维度 |
| **CL** (Context Length) | 上下文长度 |
| **KV cache** | self-attention 历史 K/V 缓存，decode 时按 token 增长 |
| **PagedAttention** | KV cache 按 page 动态分配（Kwon 2023 / vLLM）；SHIP 自研版 page=128–512 tokens |
| **Prefix Caching** | 重用历史 prefix 的 KV cache；SHIP 用 BlockTrie + rolling hash |
| **Speculative Decoding (SD)** | 小 draft 模型预测多 token，target 模型并行验证 |
| **Chunked Prefill** | 把 prefill 切成 chunk 与 decode 共享一次 forward；SHIP 支持 chunk=1 |
| **Continuous Batching** | 迭代级重组 batch（Yu 2022 / Orca）以避免短 ctx 被长 ctx 阻塞 |
| **Mixed-Batch Token Budget (MBTB)** | SGLang/B200 上一次 iteration 内 mixed-batch token 数上限 |
| **Capacity-Filling Prefill** | SHIP 调度策略：decode 优先，prefill 填空 |
| **Fused Context-Batch** | 把 batch 与 context 维度 fuse 后切 N chunk，统一 stage latency |
| **Decode Piggybacking** | mixed-chunked-prefill 中把 decode token 借机塞进 mixed batch |
| **Wafer-scale** | Cerebras 风格的整片晶圆做加速器 |
| **Weight Stationary / Activation Stationary** | dataflow 优化方向：weight 不动 / activation 不动以减搬运 |
| **Dataflow** | 张量沿 FU pipeline 流动、静态调度的体系结构范式 |
| **Propagated Synchronization** | SHIP 跨数千 LPU 的“同步波”机制，避开 host 调度 |
| **CoWoS** | Chip-on-Wafer-on-Substrate；HBM 集成关键工艺，供给紧 |
| **NCCL / NCCLX / UCCL** | GPU collective 库；典型单跳延迟 1–10 µs |
| **NVSHMEM** | GPU device-initiated 通信库 |
| **Disaggregated Serving** | prefill / decode 或 attention / FFN 解耦到不同硬件池 |

---

## 9. 关键页码索引

| 页 | 内容 |
|---|---|
| **p.1** | Abstract + Intro：SHIP 全局命题、Fig 1 latency 对比 |
| **p.2** | LPU 架构基础（TSP, FU, MEM, MXM, VXM）；node = 8 LPU + 1 TB DDR4 |
| **p.2–3** | LLM serving prefill/decode 角色拆解；Fig 2 P:D 7-day trace |
| **p.3** | Motivation §3：roofline + Fig 3 OI vs batch/CL |
| **p.4** | TP / PP scaling 决策树；§4.1 QuadFour 介绍 |
| **p.5** | Fig 4 拓扑、Fig 5 partition 在 rack 上的分布；Table 1 skip-N 带宽 |
| **p.5–6** | §4.2 efficient collective：Fig 6a/b（MAC + C2C overlap） |
| **p.6** | §5.1 PagedAttention；§5.2 prefix caching 引入 |
| **p.7** | Table 2 SRAM/DDR4 容量&带宽；Fig 7 cache hit rate；§5.3 SD |
| **p.8** | Fig 8 SD acceptance length; §6 dynamic pipeline、Fig 9 pipeline 示意 |
| **p.9** | §6 Capacity-Filling Prefill / Expert Imbalance；§6.1 实验设置 |
| **p.10** | Fig 10：throughput / TPOT / TTFT 全矩阵；7 系列子图 |
| **p.11** | §7.1 power & cost；7.1.1 BOM 详情；7.1.2 power-aware scheduling |
| **p.11–12** | §7.2 同步执行 trade-off |
| **p.12** | §7.3 disaggregated serving；§8 related work（Cerebras / Chiplet Cloud） |
| **p.12** | §9 Conclusion |
| **p.13–16** | References |

---

## 10. 一句话点评

> **SHIP 用一份 production-scale 的实证答卷宣告：在低 batch、SLO-敏感的现代 LLM serving 时代，SRAM-distributed + deterministic 同步互连 + compiler-omniscient 的 dataflow 路线不仅可行，而且在 TPOT 稳定性与 mean TTFT 上系统性优于当今最强 GPU stack——它真正做到了把“存储层级”整体上移一档，并把这条路径变成了一个每天服务数千亿 token 的工业事实。**

---

> 备注：本分析报告基于 OpenReview ID `IZaXDwDtL1` 全文 16 页（含参考文献），并结合作者引用的 Abts 2020/2022（TSP/SDM）、Bitar 2022（Groq dataflow）、Kwon 2023（PagedAttention）、Zhong 2024（DistServe）、Mitra 2026（Dynamo）、Lie 2024（Cerebras WSE-3）、Peng 2023（Chiplet Cloud）等系列工作展开横向对比，重点突出 **体系结构 / 存储层级 / 推理** 三个维度的协同设计逻辑。
