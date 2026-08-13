# 论文分析报告 ·《veScale-FSDP: Flexible and High-Performance FSDP at Scale》

> MLSys 2026 (Industry Track) · ByteDance Seed · OpenReview ID: 3Lj8R0F48P
>
> 关键词：FSDP / ZeRO / RaggedShard / DTensor / Block-wise Quantization / Muon / 8-bit Adam / 万卡训练 / 字节跳动 veScale

---

## 0. 元数据 (ByteDance veScale 项目)

| 字段 | 内容 |
|---|---|
| 论文标题 | veScale-FSDP: Flexible and High-Performance FSDP at Scale |
| 会议 | MLSys 2026 (Industry Track), Bellevue, WA |
| 作者团队 | Zezhou Wang*, Youjie Li*, Zhiqi Lin*, Jiacheng Yang*, Cong Xie, Guanyu Feng, Zheng Zhong, Ziyue Huang, Hongyu Zhu, Zhi Zhang, Yanghua Peng, Xin Liu — 共 12 人，*为共同一作 |
| 单位 | ByteDance Seed（外加 University of Washington 实习生） |
| 通讯作者 | Youjie Li (youjie.li@bytedance.com), Yanghua Peng (pengyanghua.yanghua@bytedance.com) |
| 项目背景 | 字节跳动 Seed 大模型基础设施部门主导的开源训练框架 veScale 的核心 FSDP 子模块 |
| 代码开源 | https://github.com/volcengine/veScale (RaggedShard 已开源) |
| 部署规模 | 已在 ByteDance Seed 内部承担"绝大多数训练 workload"，最大 10K+ Hopper GPUs，2.4T 参数 MoE 模型 |
| 论文页数 | 14 页正文 + 参考文献 |
| 一句话定位 | 在 PyTorch FSDP2 后端基础上，引入 RaggedShard + Planning + DBuffer 三件套，让 FSDP 能原生支持块状量化和非按元素优化器，同时把万卡训练的吞吐和显存推到极限 |

veScale 是字节跳动开源的 PyTorch-Native 大规模训练框架的总称（与 MegaScale、VeOmni 同源），这篇论文是 veScale 中的 FSDP 子系统。从研究脉络看，它接续了 PyTorch 团队从 FSDP1 → FSDP2 → Megatron-FSDP 的演进路径，但跳出"只优化对齐和拷贝"的局部修补思路，重新设计了 sharding 抽象。

---

## 1. TL;DR

现有的 FSDP（包括 DeepSpeed ZeRO、PyTorch FSDP1、PyTorch FSDP2、Megatron-FSDP）在两个维度上同时面临瓶颈：

- **灵活性**：要么按元素切（element-wise，DeepSpeed/FSDP1），要么按行均匀切（row-wise even shard，FSDP2/Megatron-FSDP）。这两种切法都无法表达"块对齐"的切片语义，因此对 DeepSeek-V3 的 128×128 块量化、Shampoo/Muon 这类非按元素优化器都需要侵入式改模型代码或额外通信。
- **性能**：FSDP2 的 per-parameter Shard(0) 设计带来 interleaved Copy-In/Copy-Out 开销（最高占一次迭代的 14%）；Megatron-FSDP 虽然零拷贝但需要大量 padding（MoE 上有 33% 通信量膨胀）；DeepSpeed/FSDP1 的 collectives 又零碎且不对齐 NCCL buffer。

veScale-FSDP 的解法：

1. **RaggedShard**：一种新的 DTensor placement，允许任意块粒度（element / row / 多维 block）+ 任意分布（每个设备可以装不同数量的 block，参考 JaggedTensor/NestedTensor 的"长短不齐"语义）。
2. **Structure-aware Planning（NP-hard，DP 启发式解）**：把通信 buffer 形式化为"最小化每设备 buffer 大小 S"的优化问题，三个约束（块不被切碎、tensor 在 buffer 中连续、各设备负载均衡），给出 O(|T|² m log E log(|T|m)) 的多项式时间 DP 算法。
3. **DBuffer (Distributed Buffer)**：把所有 RaggedShard tensor 映射到一个全局 buffer 的切片上，实现 zero-copy AllGather/ReduceScatter、批量内存分配、组级算子融合（add/scale/zero/copy 一次 kernel），并配合显式 stream 依赖管理减少碎片。

实测：在 1024 块 H800 上对 Llama-3-70B、GPT-OSS-120B、内部 160B MoE，veScale-FSDP 比 DeepSpeed/FSDP1/FSDP2/Megatron-FSDP 快 5–66%、显存少 16–30%；在万卡 Hopper 上训练 800B–2.4T MoE 模型保持线性扩展；用极少代码即可同时支持 8-bit Adam 和分布式 Muon 优化器。这是 ByteDance Seed 当前生产环境的主力训练系统。

---

## 2. 问题背景

### 2.1 大模型训练的并行化拼图

现代万亿参数 LLM 训练通常组合多种并行策略：

- **DP / FSDP / ZeRO**：按 batch 维度分片，把参数/梯度/优化器状态切到不同设备，通过 AllGather + ReduceScatter 通信。
- **TP (Tensor Parallel)**：把单个矩阵切到多卡，通常按 hidden 维或 head 维。
- **PP (Pipeline Parallel)**：按 layer 切到不同 stage。
- **EP (Expert Parallel)**：MoE 模型把专家分散到不同卡。
- **SP (Sequence Parallel)** / Context Parallel：长序列拆分。

FSDP 之所以是工业界第一选择，原因有三：
1. 内存效率高（参数/梯度/优化器状态全部切片）；
2. 编程模型简单（对模型代码几乎零侵入）；
3. 与其他并行策略可以正交叠加（FSDP × TP，FSDP × EP，HSDP）。

### 2.2 各家 FSDP 的痛点（论文最浓墨重彩的一段）

#### 2.2.1 DeepSpeed ZeRO（2020）
- **思路**：把一层所有的 tensor（params/grads/optstates）concat 在一起，再切到 N 张卡，切片边界完全任意。
- **问题**：
  - 切片是 element-wise 的，会切碎单个 tensor 的内部结构；
  - 一次 AllGather 实际上发了很多 fragmented collectives（GitHub Issue #5047），网络利用率低；
  - 内存管理依赖 PyTorch 的 record_stream，导致 caching allocator 难以复用 buffer，比 veScale-FSDP 多用 ~20% reserved memory。

#### 2.2.2 PyTorch FSDP1（Zhao et al. 2023）
- 第一版 PyTorch native ZeRO，沿用了 concatenated element-wise sharding，但优化了 AllGather 的批量化。
- **遗留问题**：ReduceScatter 慢（pre-PR）、不对齐 NCCL buffer、依然没解决 element-wise 切碎结构的问题。

#### 2.2.3 PyTorch FSDP2 / fully_shard（2024，DTensor 时代）
- **重大改变**：从 concatenated 切片转向 per-parameter sharding，每个参数表示成 Shard(0) DTensor，按行均匀切。
- **优势**：DTensor 抽象使 TP/EP/checkpoint 与 FSDP 自然组合。
- **致命开销**：interleaved Copy-In / Copy-Out。
  - 论文 Figure 2 说明：因为 FSDP2 把每个 param 都切成 Shard(0)，AllGather 之后每个 param 在 output buffer 里是 interleaved 内存（不连续），必须为每个 param 做一次 Copy-Out 到连续地址才能算。
  - 论文 Table 1（GPT-OSS-120B / 64 H800）实测：
    - Shard(0): AllGather 43.71 ms vs Copy-Out 5.22 ms；ReduceScatter 94.24 ms vs Copy-In 12.37 ms。
    - Shard(1)（避免 padding 用）: Copy-Out 13.72 ms / Copy-In 23.14 ms（更糟）。
  - 这些拷贝累计可达单次迭代时间的 14%。
- **scaling 怪事**：GPT-OSS-120B 在 128 卡能跑，到 256 卡 OOM。原因是 128 个 expert 切到 256 卡，padding 让 AllGather buffer 直接翻倍。

#### 2.2.4 Megatron-FSDP（2025）
- **思路**：放弃 FSDP2 的 per-parameter，回到 FSDP1 的 concatenated sharding 来避免 Copy 开销（zero-copy）。
- **代价**：为了让 concat-shard 在 checkpoint 时还能伪装成 Shard(0) DTensor（与 PyTorch 上游 DCP 兼容），它必须插入 padding 把切片对齐到行边界。MoE 上 padding 直接膨胀 33%。
- **本质局限**：仍是 row-wise，无法满足 block-wise 量化的块对齐要求；也无法支持任意块粒度。

### 2.3 字节实战触发的真实需求

论文很清楚地把"为什么 FSDP 必须改"摆在桌面上，原因来自一线工程：

1. **Block-wise FP8 量化**：DeepSeek-V3 等用 128×128 block 做 FP8 weight 量化。如果切片边界不和 block 对齐，每张卡需要互相交换 scaling factor 元数据，破坏了 communication-free 量化的核心收益。
2. **Matrix Optimizers (Shampoo / Muon)**：这些优化器对原始 2D 矩阵做 Newton–Schulz 等矩阵级运算，必须要拿到完整的 2D 参数，而不是切碎的 element shard。Muon 已被证实比 AdamW 收敛更快，工业上要支持。
3. **8-bit Adam**：optimizer state 做 block-wise INT8 量化，依赖块边界对齐。
4. **GPU 显存是更紧的约束**：在共享集群里，OOM 或贴着内存上限运行（频繁 device free）会强制 over-provision，浪费 GPU。论文反复强调这是产线痛点。
5. **万卡级 scaling**：达到 10K+ GPU 时，padding、buffer alignment、kernel launch 这些"小问题"全部变成大问题。

### 2.4 为什么不直接用 Megatron-LM？
论文在 Lessons Learned 里直接吐槽：Megatron-LM 把模型代码和并行策略紧耦合，研究员每改一次 architecture 都要重写一遍系统层，对架构创新极不友好。FSDP 的解耦优势在 LLM 架构高速演化的当下变得格外重要。

---

## 3. 核心思想 / 方法

veScale-FSDP 的整体架构（论文 Figure 3）由四层组成：

```
PyTorch-Native fully_shard API   ← 用户侧 0 改动
        ↓
RaggedShard DTensor placement     ← 灵活性
        ↓
Structure-aware Planning Algorithm ← 把灵活性翻译成高效通信
        ↓
Distributed Buffer (DBuffer)       ← 零拷贝、零碎片的底层执行
        ↓
10K+ Devices
```

### 3.1 RaggedShard：灵活性的核心

#### 3.1.1 现有 sharding 形式的能力光谱（Figure 4）
| 格式 | 非按元素计算 | Redistribute | Block-wise 量化（无通信） |
|---|---|---|---|
| Element-wise Shard（DeepSpeed/FSDP1） | ✗ | ✗ | ✗ |
| Row-wise Even Shard（FSDP2/Megatron） | ✓ | ✓ | ✗ |
| **Row-wise RaggedShard** | ✓ | ✓ | 部分 |
| **Block-wise RaggedShard** | ✓ | ✓ | ✓ |

#### 3.1.2 RaggedShard 的两大自由度

借鉴 JaggedTensor / NestedTensor 的"长短不齐"语义：

1. **任意 sharding 粒度（granularity）**：原子不可切的 block 大小可以由用户定义，可以是 element、row、二维 block、甚至高维 plane。
2. **任意 sharding 分布（distribution）**：每个设备上 block 的数量可以不同，不需要均匀分布。

最一般化形式是 **Block-wise RaggedShard**：把 tensor 切成形状自定义的多维 block，每个 device 装若干个 block。这是真正的"sharding 通用形式"——其他所有格式都是它的特例。

#### 3.1.3 与现有 DTensor placement 的组合（Figure 5）

DTensor 已经支持 Replicate / Partial / Shard(dim)，veScale-FSDP 把 RaggedShard 作为额外 placement 注册进去：

- **与 Replicate / Partial 正交**：直接组合即可。
- **与 Shard(dim) 的协同**：
  - 对 `Shard(0)`：引入 `StridedRaggedShard` 携带 reorder/stride 元信息，在 materialize 完整 tensor 时做 reshuffle（解决 PyTorch placement list 的"逆序应用"语义混乱）。
  - 对 `Shard(dim>0)`：把 ragged 粒度调整为该维度 stride 与用户粒度的 LCM，避免切到 stride 内部。

这一节是论文工程上最绕的部分，但也是它能"无缝接入既有 TP/EP 生态"的关键。Checkpointing 直接复用 PyTorch DCP，因为 RaggedShard 本质是 DTensor 的扩展。

### 3.2 Structure-aware Planning：性能的核心

#### 3.2.1 Naïve 路径的三大坑（Figure 6a）

如果只是简单把 RaggedShard tensors concat 到通信 buffer，会遇到：

- **Sharded Block**：block 被切到不同 device 上，破坏 block 抽象，量化要补做通信；
- **Non-contiguous Tensor Memory**：为了满足 NCCL alignment / 等大小约束的 padding 落到 tensor 内部，又制造了 interleaved copy；
- **Imbalanced Load**：各 device buffer 不一样大，collective 失去对称性，带宽用不满。

#### 3.2.2 优化目标（Figure 6b）

veScale-FSDP 的 planner 做两步：先 **permute tensors**（给 tensor 排个好顺序），再 **pad between tensors**（填充加在 tensor 之间，而不是 tensor 内部）。形式化为：

设 T = {t₁,…,tₙ} 为 RaggedShard tensor 集合，切到 m 个设备。tensor t 的 block 大小 g_t、总元素数 e_t、block 数 u_t = e_t/g_t。在全局 buffer 上每个 t 占连续区间 [ℓ_t, r_t)。每个 device k 拥有 [(k−1)S, kS) 这一段。

```
min   S
s.t.  r_t − ℓ_t = e_t,   r_t ≤ mS               (尺寸正确)
      区间互不重叠                                 (tensor 连续 + 不相交)
      (kS ≤ ℓ_t) ∨ (kS ≥ r_t) ∨ ((kS − ℓ_t) ≡ 0 mod g_t)   (block 不被切)
```

三个约束分别对应：**Non-Sharded Block**、**Contiguous Tensor Memory**、**Balanced Load**。

#### 3.2.3 NP-hardness 与启发式
- 可从经典 Partition 问题归约 → NP-hard。
- ILP 求解器在生产规模（百级参数组、十万级 device）上要跑几十分钟甚至超时，不实用。
- 因此 veScale-FSDP 给出 **多项式时间 DP 启发式算法（Algorithm 1）**：
  - 时间复杂度 O(|T|² m log(E) log(|T|m))。
  - 利用 transformer 参数的"高度规则性"：linear weights 主导参数量、各层 block size 一致。
  - 实践中只考虑三种 tensor 排序：默认顺序、按 block size 排、按 shape 排。统计显示这三种就能给出最优或近最优解，最终采用默认顺序（便于调试）。
  - 核心 DP 状态 dp(t, i) = 放完前面所有 tensor + 当前 tensor t 的前 i 个 block 所需的最小 device 数。
  - 利用 dp(t, i) 在 i 上单调的性质，把连续相同值的 i 段合并跳过，得到上述复杂度。
  - 对每个 tensor 与 shard 边界的关系做 case 分析：
    - case (1) 完全落在单个 shard 内；
    - case (2) 跨两个 shard 但不完整包含任何 shard；
    - case (3) 完整包含至少一个 shard。
  - 如果只有 case (1)(2)，S 在最小对齐单位上单调可行 → 二分搜索；
  - 如果存在 case (3)，S 必须是 L = LCM{g_t : t in case (3)} 的倍数，按 L 的倍数二分。
  - 对 case (3) tensor 集合，按 element 数排序后只考虑 prefix → 2-approximation，避免指数枚举。

实测：所有实验中 planner 自身运行时间 < 0.3 秒，初始化阶段一次性，可忽略。

### 3.3 DBuffer：执行层的零拷贝底盘

DBuffer 是 RaggedShard plan 的运行时执行体（Figure 7）：

1. **N 维全局 buffer 抽象**：模仿 DTensor 的 placement 概念，DBuffer 自己也有 N 维度上的 sharding spec，自然支持 2D/3D 通信（如 FSDP × EP，gradient 的 (Partial, Partial) → (Replicate, Shard) 实现 ReduceScatter+AllReduce 组合）。
2. **Group-level 算子**：tensor 通信前的 add/scale/zero/copy 在 DBuffer 上做一次 fused kernel，避免每个 tensor 单独 launch CUDA kernel 阻塞 NCCL stream。
3. **Zero-copy 访问**：每个 RaggedShard tensor 的数据指针直接映射到 DBuffer 的某一段，永久地址映射；通信前后无需拷贝。这正是 FSDP2 interleaved Copy 的根除方案。
4. **In-place 通信和计算**：减少额外内存分配。
5. **批量分配与显式 stream 管理**：批量分配减少碎片；显式 stream 依赖（不依赖 PyTorch 的 record_stream 隐式机制）让 caching allocator 能预测性地复用 buffer，相比 DeepSpeed/FSDP1 节省 ~20% reserved memory，相比 FSDP2 再省 ~12%。

### 3.4 三者协同的语义闭环

- **RaggedShard** 提供"用户想要什么样的切片"的描述能力（语义层）；
- **Planning** 决定"为了高效通信，tensor 应该怎么排、padding 加在哪里"（编译层）；
- **DBuffer** 提供"按规划放置 + 零拷贝执行"（运行时层）。

三者缺一不可，且边界清晰——这是论文一个工程美学上的亮点。

---

## 4. 实现 / 工程细节

### 4.1 代码体量与集成方式

- 7.6K 行 Python 代码。
- 透明替换 PyTorch FSDP2 后端，对外暴露同一套 `fully_shard` API（用户 0 修改）。
- 即插即用 Python 模块，兼容 PyTorch 标准 distributed runtime 和广泛的 PyTorch 版本。
- Mixed precision 支持：默认 FP32 master weights + BF16 forward/backward。

### 4.2 与 PyTorch DTensor 的关系

- RaggedShard 注册为 DTensor 的一种 placement，与 Shard/Replicate/Partial 并列。
- Checkpointing 直接复用 PyTorch Distributed Checkpoint（DCP），享受其 communication-free sharded checkpoint 等优化。
- 论文披露：RaggedShard 已经被列入 PyTorch 官方 2026 H1 路线图（即将合入主线）。

### 4.3 Plan-Execute 分离的思想

虽然论文没有显式用"Plan-Execute"这个词，但整体架构本质就是：
- **Plan 阶段（一次性）**：用户用 fully_shard 包装模型 → 系统采集所有 RaggedShard tensor 元信息 → 跑 Algorithm 1 → 生成全局 buffer 布局。
- **Execute 阶段（每个 step）**：DBuffer 按既定布局执行 AllGather / ReduceScatter / 计算，全程零拷贝。

这与 Megatron 的"代码内手写 collective"路线形成鲜明对比。

### 4.4 通信调度与 overlap

- AllGather / ReduceScatter 与 forward / backward 计算 overlap（FSDP 通用做法，veScale 在 DBuffer 上做得更彻底）。
- 跨节点 EP 在大规模时被用来缓解 FSDP 通信压力（论文 Strong Scaling 部分提到这个 trade-off：EP 减 FSDP 通信，但加了 token exchange 和 kernel 效率下降）。
- NCCL buffer alignment：FSDP1/FSDP2 不强制对齐导致 collectives 性能退化；veScale 通过 planner 显式对齐 NCCL preferred unit size g_coll（Algorithm 1 第 19 行 `g ← g_coll`）。

### 4.5 Activation Checkpoint / Mixed Precision

论文未在这两点上开新章节，因为它们在 PyTorch FSDP 体系里已经成熟。veScale-FSDP 的贡献在于：
- 不像 Megatron 那样为了支持 mixed precision 持久化低精度 buffer（导致 Llama-3 实验多耗 24% 显存），DBuffer 的 batched allocation 让低精度 buffer 按需创建释放。
- Activation checkpoint 与 fully_shard API 自然兼容。

### 4.6 8-bit Adam 的实现接口

- 暴露 `orig_param_policy` 接口，让用户为每个参数指定量化粒度。
- 论文 setup：32×32 block，给 matrix 参数指定 32-row block 粒度。
- 每个设备独立量化本地 shard，无需通信。block 边界完美保留。
- 用户代码量：「few lines of code」级别。

### 4.7 Distributed Muon 的实现（Algorithm 2）

```
for w in 2D 参数:
    g ← grad(w)
    u ← MomentumUpdate(g, m)
    p ← placement(u)
    r ← SelectRoot()              # 负载均衡选个 root rank
    o ← Redistribute(u, RaggedShard(r))   # 用 redistribute 把完整 tensor unshard 到 root
    o ← NewtonSchulz(o)           # 只在 root 上跑矩阵迭代，其他 rank 是 no-op
    o ← Redistribute(o, p)        # update 重新 redistribute 回原 placement
    w ← w − η o
```

关键点：
- RaggedShard 让 unshard-to-one-rank 这件事可以用标准 DTensor `redistribute` SPMD 写出来；
- 异步 redistribute 与计算 overlap；
- 配合 `torch.compile` 增强 Newton-Schulz 的算力密度，最终在 256 Hopper GPU 上达到 47.3% MFU。

---

## 5. 评测

### 5.1 硬件与基线

- §6.1, §6.4, §6.5：NVIDIA H800 集群，每节点 8×H800（979 BF16 TFLOPS, 80 GB HBM, 400 GB/s NVLink）。
- §6.2, §6.3：NVIDIA Hopper 集群（具体型号未披露，推测仍是 H 系列）。
- 基线：DeepSpeed ZeRO v0.17.6 / PyTorch 2.7.1 FSDP1 / PyTorch 2.7.1 FSDP2 / Megatron-FSDP。
- 统一 ZeRO-3 + mixed precision (FP32 master + BF16 fwd/bwd)。

### 5.2 端到端性能（§6.1, Figure 8, 1024 GPU）

模型：
- Llama-3-70B（dense）
- GPT-OSS-120B（sparse MoE，128 experts）
- Internal-Model-160B（内部 MoE）

测试 4 种配置：FSDP@128 / FSDP@256 / HSDP 2×256 / HSDP 4×256。

**吞吐**：
- 在两个 MoE 模型上，veScale-FSDP 比所有 baseline 快 11~66%。
- 在 Llama-3-70B 上，比 DeepSpeed/FSDP1/FSDP2 快约 5%，略胜 Megatron-FSDP。
- 收益来源：DBuffer 零拷贝 collective、planner 减 padding、通信 overlap、buffer alignment。

**显存**：
- veScale-FSDP 比 baseline peak reserved memory 低 16–30%。
- DeepSpeed/FSDP1：record_stream 隐式释放 → caching allocator 难复用 → 多用 ~20%。
- FSDP2：per-parameter eager allocation → veScale 比它再省 ~12%。
- Megatron：MoE padding inflation → +33% memory；mixed precision 持久 buffer → Llama-3 +24% memory。
- **GPT-OSS-120B 显著事件**：FSDP2 在 128 卡能跑，到 256 卡 OOM（128 expert × 2 padding → AllGather buffer 翻倍）。Megatron 在 256+ 卡也 OOM/ERROR。veScale-FSDP 全部正常运行。

### 5.3 Scalability（§6.2, Figure 9）

测试在 800B–2.4T 内部 MoE 模型 + 1K~10K Hopper GPU 上。

**Weak Scaling（Figure 9a, 1K→8K GPU, 固定 per-GPU 2K~16K tokens）**
- 接近线性扩展。原因：FSDP 通信成本与计算成本都不随 GPU 数变化。
- ByteDance 用 64 GPU profiling 即可外推到千卡（Lesson 1 验证）。

**Strong Scaling（Figure 9b/9c, 1K→8K GPU, 固定 16M~128M global tokens）**
- 120M token global batch 在 10K GPU 仍线性。
- 16M token batch 从 1K → 8K 仍有 3.4× 加速。
- 当 token/GPU 减少时，FSDP 通信开始 dominate → 启用 cross-node EP 来 cap collective group size，但 EP 自身有 token exchange 开销，所以超大规模会有轻微下降。

**Model Scaling（Figure 9d, 1K GPU, 400B→2.4T 参数）**
- MFU 随模型增大略微提升（compute intensity 上升）。
- DBuffer 内存效率使 1K GPU 能训练 2.4T 参数模型而无性能退化。

### 5.4 8-bit Adam 与 Muon（§6.3, Figure 10）

- **8-bit Adam**：veScale-FSDP 与 DDP 8-bit Adam 的 loss 曲线高度吻合，仅有 reduced-precision 特征性 spike。
- **Distributed Muon**：veScale-FSDP Muon 与 DDP Muon 几乎重合；Muon 比 AdamW 在 ~80B token 后稳定低 0.01 loss，复现了 Wen et al. 2025 的结论。
- **核心论点**：veScale-FSDP 让这些 advanced optimizer 用"几行代码"就能落地，而其他 FSDP 实现需要"侵入式改模型/优化器"或"手写 collective"。

### 5.5 Planner 质量（§6.4, Figure 11）

模型：DeepSeek-V3-671B、GPT-OSS-120B；row 粒度扫 1× / 16× / 128×。

- 1× / 16× row 粒度：在所有 FSDP size 下 padding < 3%。
- 128× row（DeepSeek 128×128 tile）：DeepSeek-V3 大体 < 3%，GPT-OSS 出现台阶式波动，最高 18%。
- 原因：GPT-OSS 把所有 expert 融合为单 tensor，没法在 expert 之间插 padding；DeepSeek-V3 每个 expert 独立 tensor 可以 expert 间 padding。
- 台阶现象的根因：shard size 必须按 LCM(粒度, NCCL 对齐) 取整，跨过 LCM 倍数时跳跃。
- **实践指南**：
  - 不要用过大 FSDP group，用 HSDP 分层并行；
  - 用 offline simulation 选 FSDP size 减少 LCM rounding；
  - 模型 hidden size 用小复合数（divisible by 多个小因子），别用大互质数。

### 5.6 Component Ablation（§6.5, Table 2，32 GPU + GPT-OSS-style + 8-bit Adam）

| 配置 | Normalized Throughput |
|---|---|
| 完整 veScale-FSDP | 100.0% |
| 关掉 DBuffer | 92.8% |
| 关掉 Planning | 65.4% |
| 关掉 RaggedShard | N/A（无法跑） |

- DBuffer 贡献 7.2% 吞吐（来自 Copy-In/Out 的消除）；
- Planning 贡献 34.6% 吞吐（关掉后 quantization block 跨 device，被迫 fallback 到 DTensor redistribute 做额外通信）；
- RaggedShard 是抽象本身——没有它，8-bit Adam 根本写不出来（除非侵入式改）。

---

## 6. 思想精读 / 启示

### 6.1 中国大厂训练系统的工程能力跃迁

veScale-FSDP 体现了字节训练系统团队几个值得记录的能力点：

1. **抽象层提升**：不是修补 FSDP2 的 Copy 开销（这是个工程优化思路），而是重新设计 sharding format（这是个抽象层思路）。三个组件的边界清晰：RaggedShard（语义）/ Planning（编译）/ DBuffer（运行时）。这种分层是研究型系统的特征。
2. **PyTorch 上游协作**：作者明确感谢 TorchTitan 团队和 Edward Z. Yang，且 RaggedShard 已进入 PyTorch 官方 2026 H1 路线图。这意味着 ByteDance 在主导 PyTorch 分布式生态的演进，而不是在 fork 上做单方面优化。
3. **生产-研究闭环**：论文在生产环境跑了 10K+ GPU，2.4T 参数模型，且"承担 ByteDance Seed 大多数训练 workload"。这与 MegaScale (NSDI'24)、VeOmni (2025) 是同一团队连续输出。
4. **务实的算法选择**：planner 用 DP 启发式而非 ILP，因为生产规模下 ILP 不可用；用三种 tensor 排序中的"默认顺序"，因为 transformer 规则性高且便于调试。这是工程驱动的算法设计。

### 6.2 FSDP 与 Megatron 路线的边界

- **Megatron 路线**（Megatron-LM 主分支）：把并行策略嵌进模型代码，TP/PP/SP 高度优化但模型迭代慢。
- **FSDP 路线**（veScale-FSDP / FSDP2 / Megatron-FSDP）：保持模型代码 SPMD，把并行下沉到框架。

veScale-FSDP 通过 RaggedShard 把 FSDP 的能力上界往 Megatron 推进了一大步——以前需要 Megatron-LM 才能做的"非按元素优化器、块状量化"，现在 FSDP 也能做。
但是 TP/PP/SP 这些更细颗粒度的 model parallelism 仍由 Megatron/DTensor 的 TP/EP 提供，veScale-FSDP 选择与之组合而非替代。

这是一个清晰的"分层协作"信号：FSDP 负责 batch / parameter sharding，TP/PP 负责模型内并行，DTensor 是它们的公共抽象。

### 6.3 RaggedShard 的"通用 sharding format"野心

从 Figure 4 可以看出，论文把 element-wise / row-wise / block-wise 都视为 RaggedShard 的特例。这个观点其实可能改变 PyTorch DTensor 的核心 placement 设计——如果未来 DTensor 用 RaggedShard 作为 base placement，Shard(dim) 反而成为一个特殊情况。论文在 Lesson 2 也明确说了"Design system abstractions on the shoulders of giants"，把 RaggedShard 设计为 DTensor 的扩展而非替代。

### 6.4 启示给其他训练框架

1. **Padding 不要加在 tensor 内部**：interleaved padding = interleaved copy。永远在 tensor 之间加 padding。
2. **NCCL alignment 是一等公民**：不要让 collective 在不对齐 buffer 上跑。
3. **Stream 依赖显式化**：不要依赖 record_stream 的隐式 free，caching allocator 会哭。
4. **Plan-Execute 分离**：一次性 planning 解 NP-hard，每 step 零开销执行。这是编译型系统的标准范式。

---

## 7. 局限与开放问题

### 7.1 论文承认的局限

1. **GPT-OSS 128× row 粒度的 padding 波动**：单 fused tensor 模型不能跨 expert 插 padding，最差 18%。需要靠"调整 FSDP group size + 调整 hidden size"才能缓解。
2. **超大规模强 scaling 收益递减**：8K+ GPU 时 token/GPU 太少，FSDP 通信 dominate；EP 救场但有自己的开销。
3. **Planner 是启发式的**：理论上 NP-hard，DP 给出 2-approximation；最优解保留可能性，但生产中"足够好"。
4. **依赖 transformer 规则性**：默认 tensor 顺序对 transformer 工作良好，非 transformer 架构需要插入自定义 ordering 启发。

### 7.2 论文没讨论但值得思考的问题

1. **故障容错**：10K GPU 上 MTBF 短，论文只提到 DCP checkpoint 复用，没讨论 elastic FSDP 和 fault recovery 的细节。
2. **跨 region / 多集群训练**：通信拓扑不均匀时 planner 是否需要扩展？
3. **与 PP 的组合**：论文重点是 FSDP × EP，对 FSDP × PP 涉及较少。
4. **量化范围进一步收窄（FP4/INT4）**：当量化 block 更小时（如 32×32），padding overhead 是否还能保持？
5. **Compile/Inductor 集成**：论文只在 Muon 部分提到 torch.compile，FSDP 主路径下 compile graph 与 DBuffer 的交互还有空间。
6. **Planner 对模型动态变化的响应**：FSDP 包装是 static 的，但训练中如果有 dynamic shape 或 dynamic expert routing 怎么办？
7. **NCCL 之外的通信库**：是否能扩展到 RCCL / 国产 GPU 通信库？字节内部肯定有相关需求但论文未涉及。
8. **真实 padding 数值**：论文给出 padding 比例但没给绝对显存数字，对小模型/边缘情况的影响不明。

### 7.3 工程层面的开放点

- 用户需要根据 model + RaggedShard granularity 做 offline simulation 选 FSDP group size，这增加了上线成本。是否可以自动化？
- `orig_param_policy` 接口的人机工程学：每个参数指定粒度对工程师友好，但对纯研究员可能仍嫌繁琐。

---

## 8. 关键术语速查表

| 术语 | 含义 | 在论文中的位置 |
|---|---|---|
| **FSDP (Fully Sharded Data Parallel)** | PyTorch 实现的 ZeRO 风格数据并行：参数/梯度/优化器状态全部按设备切片，每层 forward/backward 前 AllGather，backward 后 ReduceScatter | §1, §2.3 |
| **ZeRO-3** | DeepSpeed 提出的 stage-3 优化：参数 + 梯度 + 优化器状态全切。FSDP 对应 ZeRO-3 | §6 baseline 配置 |
| **FSDP1** | PyTorch 第一版 native ZeRO，concatenated element-wise sharding | §2.3 |
| **FSDP2 / fully_shard** | PyTorch 第二版，per-parameter Shard(0) DTensor，state of the art | §2.3, §6 |
| **Megatron-FSDP / Mcore Custom FSDP** | NVIDIA Megatron 实现的 FSDP，零拷贝但有 padding 膨胀 | §2.3 |
| **DTensor** | PyTorch 分布式 tensor 抽象，三种 placement: Shard(dim) / Replicate / Partial，提供 redistribute API | Figure 1, §2.2 |
| **Shard(dim)** | DTensor 沿 dim 维度均匀切片的 placement | §2.2 |
| **Replicate** | DTensor 在每个 device 全副本的 placement | §2.2 |
| **Partial** | DTensor 每个 device 持有部分值，要 reduce 才得完整 tensor | §2.2 |
| **JaggedTensor / NestedTensor** | PyTorch / TensorFlow 用于表示最后维度长度不齐的 tensor，是 RaggedShard 的灵感来源 | §2.2 |
| **RaggedShard** | 本文新提出的 DTensor placement，任意块粒度 + 任意分布 | §4 |
| **Block-wise RaggedShard** | RaggedShard 最一般化形式，支持自定义多维 block | §4 |
| **StridedRaggedShard** | RaggedShard 与 Shard(0) 组合时携带 reorder/stride 元信息的子类 | §4 末 |
| **DBuffer (Distributed Buffer)** | 本文新提出的全局通信 buffer 抽象，支持 N 维度 sharding spec、group-level 算子、零拷贝 | §5 末, Figure 7 |
| **AllGather** | NCCL collective：每个 rank 把本地 tensor 收集到所有 rank | FSDP forward/backward 前 |
| **ReduceScatter** | NCCL collective：每个 rank 收到 sum 后再分散一份 | FSDP backward 后 |
| **AllReduce** | AllGather + Reduce 的组合，DDP 标准操作 | §6.3 baseline |
| **Interleaved Copy-In / Copy-Out** | FSDP2 因 Shard(0) 拼接到通信 buffer 时产生的非连续地址拷贝 | Figure 2, Table 1 |
| **TP (Tensor Parallel)** | 把单个矩阵切到多卡，Megatron-LM 经典方案，DTensor 用 Shard(0)/Shard(1) 表达列/行切 | §4 末 |
| **PP (Pipeline Parallel)** | 按 layer 切到不同 stage 的并行 | §1 |
| **EP (Expert Parallel)** | MoE 中按 expert 切，DTensor 用 expert 维 Shard(0) | §4 末, §6.2 |
| **HSDP (Hybrid Sharded Data Parallel)** | FSDP + DP 副本组合的分层并行 | §6.1 |
| **MFU (Model FLOPS Utilization)** | 实际算力 / 理论算力 | §6.2 |
| **Block-wise Quantization** | 把 tensor 按固定 block（如 128×128）做缩放因子量化，DeepSeek-V3 / 8-bit Adam 都用 | §2.1, §6.3 |
| **Matrix Optimizer** | Shampoo/Muon 等需要原始 2D 参数矩阵参与更新的优化器 | §2.1 |
| **Newton-Schulz** | Muon 优化器中的矩阵迭代算子，需要完整 2D 参数 | Algorithm 2 |
| **NP-hardness (Partition Reduction)** | RaggedShard planner 的优化问题可由经典 Partition 问题归约 | §5 |
| **LCM Rounding** | shard size 必须是粒度与 NCCL alignment 的最小公倍数倍数，导致 padding 阶梯 | §6.4 |
| **record_stream** | PyTorch caching allocator 的隐式释放机制，DeepSpeed/FSDP1 依赖它，会导致 buffer 难以复用 | §6.1 memory 分析 |
| **DCP (Distributed Checkpoint)** | PyTorch 分布式检查点，RaggedShard 直接复用其栈 | §4 末 |
| **veScale** | ByteDance 开源的 PyTorch-Native 训练框架总称，FSDP 是其子模块 | github.com/volcengine/veScale |

---

## 9. 关键页码索引

| 主题 | 页码 |
|---|---|
| Abstract（一句话总结 5–66% 吞吐、16–30% 显存） | p.1 |
| 各家 FSDP 痛点综述（DeepSpeed/FSDP1/FSDP2/Megatron-FSDP） | p.1 末 – p.2 上 |
| 四个核心 contribution bullet（RaggedShard / Planning / DBuffer / 部署） | p.2 上 |
| Structure-aware Training 动机：Shampoo/Muon、block-wise 量化 | p.2 |
| DTensor 三种 placement + redistribute（Figure 1） | p.2 末 |
| FSDP2 interleaved Copy 开销（Table 1, Figure 2） | p.3 |
| 系统 Overview（Figure 3） | p.3 末 – p.4 上 |
| 已有 sharding format 比较（Figure 4） | p.4 |
| Block-wise RaggedShard 概念 | p.4 中 |
| RaggedShard 与现有 placement 组合，StridedRaggedShard | p.5 上 |
| Naive grouping 三大坑（Figure 6a） | p.5 末 |
| 优化问题形式化与 NP-hardness | p.5 末 – p.6 上 |
| Algorithm 1 DP 启发式 | p.7 |
| DBuffer 设计（Figure 7） | p.7 末 |
| End-to-end 性能对比（Figure 8, 1024 GPU） | p.8 |
| Memory 分析（FSDP2 GPT-OSS OOM, Megatron padding 膨胀） | p.8 末 – p.9 |
| 万卡 scaling（Figure 9, 8K-10K GPU, 2.4T 模型） | p.9 |
| 8-bit Adam + Muon（Figure 10, Algorithm 2） | p.10 |
| Planner padding 质量（Figure 11） | p.11 |
| Component Ablation（Table 2） | p.11 |
| Lessons Learned（小规模 profile 外推、站在 DTensor 肩膀上、解耦模型与系统） | p.12 |

---

## 10. 一句话点评

> **veScale-FSDP 用 RaggedShard 把 FSDP 的 sharding 抽象一举提升到"任意块粒度 + 任意分布"的通用形式，再用 NP-hard 的 planning 算法和零拷贝 DBuffer 把它翻译成万卡级别的极致性能——这不是对 FSDP2 的修补，而是 FSDP 路线在大模型时代的范式重构，也代表了字节跳动 Seed 训练系统团队"既能写论文也能扛万卡生产"的工程成熟度。**

— 完 —
