# 论文分析报告 ·《SuperInfer: SLO-Aware Rotary Scheduling and Memory Management for LLM Inference on Superchips》

> 体系结构 + 超节点总线 + 大模型推理：当 GPU 与 CPU 通过 NVLink-C2C 紧耦合成 Superchip 之后，传统基于 PCIe 的 KV cache offloading 软件栈为什么只能用到 <5% 的链路带宽？SuperInfer 给出了一套从调度（RotaSched）到数据通路（DuplexKV）的协同设计答案。本文对该工作进行深入剖析，重点关注其在 Superchip 体系结构上的范式意义。

---

## 0. 元数据

| 项目 | 内容 |
| --- | --- |
| 论文标题 | SuperInfer: SLO-Aware Rotary Scheduling and Memory Management for LLM Inference on Superchips |
| 作者 | Jiahuan Yu, Mingtao Hu, Zichao Lin, Minjia Zhang |
| 单位 | Siebel School of Computing and Data Science, University of Illinois Urbana-Champaign (UIUC SSAIL Lab) |
| 通讯邮箱 | jiahuan2@illinois.edu, mingtao4@illinois.edu, zichaol3@illinois.edu, minjiaz@illinois.edu |
| 会议 / 年份 | 9th MLSys Conference, Bellevue, WA, USA, 2026 |
| OpenReview | https://openreview.net/forum?id=RuslSHdIHa |
| 代码仓库 | https://github.com/Supercomputing-System-AI-Lab/SuperInfer |
| Artifact DOI | https://doi.org/10.5281/zenodo.18971768 (artifact)，https://doi.org/10.5281/zenodo.19394229 (code) |
| 测试硬件 | NVIDIA GH200 NVL2（每节点 2 个 Grace–Hopper Pair，HBM3 144GB + Grace LPDDR5X 480GB，NVLink-C2C 900GB/s 双向） |
| 关键基线 | vLLM v0.6.6.post1（V1 engine）、TensorRT-LLM v1.1.0、LightLLM v1.1.0、LTR、NEO |
| 评测模型 | LLaMA-3-8B、Qwen2.5-32B、Mixtral-8x7B（MoE）|
| 评测数据集 | ShareGPT、LMSYS-Chat-1M |
| SLO 设定 | TTFT SLO = 5s，TBT SLO = 100ms |
| 论文页数 | 19 页（含 Reference 与 Artifact Appendix） |
| 致谢资助 | NSF Grant No. 2441601；ACCESS（Delta、DeltaAI、Jetstream2）；Google ML & Systems Junior Faculty Award；IBM、Amazon、AMD 礼物资助 |

---

## 1. TL;DR

SuperInfer 是首个面向 **Superchip（NVIDIA GH200 等紧耦合 GPU+CPU 体系结构）** 的 SLO-aware LLM 推理系统。它围绕一条核心论点展开：

> Superchip 给 LLM 服务带来了 PCIe 时代不可想象的 swap 带宽（NVLink-C2C 双向 900GB/s），但**直接把 PCIe 时代的 offloading 代码移植到 GH200，只能利用不到 5% 的 C2C 链路带宽**。瓶颈不在硬件，而在以 PagedAttention 为代表的软件栈：碎片化的 64KB segment 拷贝 + 数千次 cudaMemcpyAsync kernel 启动开销 + 单向半双工的 swap 路径。

为弥合这一软硬件鸿沟，SuperInfer 提出两个相互耦合的核心组件：

1. **RotaSched**：受 OS 启发的 SLO-aware 主动旋转调度器。它把请求看作"线程"，HBM 看作"片上 cache"，Grace DRAM 看作"主存"，引入 **Virtual Lag Time (VLT)** 度量请求相对 SLO 的"滞后"程度，并按 **Largest-VLT-First (LVF)** 策略主动把长跑请求换出到 DRAM、把濒临违约的等待 / rotary 请求换入 HBM，进入一个新的 *transient* "rotary" 状态。
2. **DuplexKV**：高带宽 KV cache 旋转引擎。通过 **eager block rotation**（消除 H2D / D2H 间的 data race）+ **block-first KV layout**（把 64KB 小段合并到 4MB 大块）+ **batched cudaMemcpyBatchAsync** + **跨迭代流水线**，把 GH200 上有效 swap 带宽从 ~10GB/s 提升到 ~180+180 GB/s（双向），逼近 Grace DRAM 半双工 384GB/s 的硬件极限。

实验在 GH200 NVL2 上覆盖 LLaMA-3-8B、Qwen2.5-32B、Mixtral-8x7B 三种模型，在 ShareGPT/LMSYS-Chat-1M 工作负载上：

- **TTFT SLO 命中率最高提升 74.7%**（相对 vLLM、TensorRT-LLM、LTR、LightLLM、NEO）；
- **TBT SLO 命中率持平或略优**于最强基线；
- **吞吐持平甚至最高 +29.2%**（高 RPS 下 chunked prefill 因为 fast rotation 拿到更多 batching 机会）；
- DuplexKV 实现单向 ~239/270 GB/s 的 D2H/H2D 带宽（MS+MK 配置）和约 180+180 GB/s 的双工带宽（达到 ideal 384GB/s 半双工的 94%）。

一句话总结：**Superchip 的硬件红利不会自动落到 LLM 推理服务上，必须用 OS 思维重写调度器 + 重新组织 KV layout，才能把 NVLink-C2C 当作"扩展 HBM"用。**

---

## 2. 问题背景

### 2.1 LLM 推理的延迟 SLO 与 KV cache 内存压力

LLM 在线服务（chat、search、code copilot、agent）越来越普遍，其延迟 SLO 通常拆为两类：

- **TTFT (Time-To-First-Token)**：从请求到达到第一个 token 返回的延迟。直接决定用户体感的"响应性"。论文设为 5s。
- **TBT (Time-Between-Tokens)** / TPOT：相邻两个 token 之间的间隔，决定流式输出的"流畅度"。论文设为 100ms（行业典型值 50–100ms）。

LLM 推理的难点在于 **KV cache 随序列长度线性增长**：每个 layer、每个 head 都要存历史 token 的 K/V。即便 H100 80GB 或 GH200 144GB HBM 也只能装下有限的并发请求；一旦 RPS 超出容量，新请求堆积在 waiting queue 中，**HOL (head-of-line) blocking** 导致 TTFT SLO 大规模违约。

为缓解 KV 压力，业界主要有两条路：

1. **离线压缩 / 选择性保留**（CacheGen 量化压缩、InfiniGen 重要性裁剪、H2O 等）：lossy，泛化困难。
2. **Offloading**（FlexGen、DeepSpeed-Inference、FastDecode、NEO、HeteGen、CachedAttention、FlashGen、NanoFlow、Pie、Mooncake 等）：把 KV cache 或部分计算搬到 CPU DRAM 或 SSD。

### 2.2 PCIe offloading 的两大致命短板

论文在 §3 用三个动机实验讲清楚为什么 PCIe offloading 路线已经走到尽头。

**短板一：static SLO-unaware 策略两面不讨好（Insight #1）。** vLLM / SGLang 这类生产框架普遍采用：
- **Waiting-First (WF)**：抢占 running 请求让 waiting 请求先入队 → TTFT 改善但 TBT 灾难（running 被反复打断）。
- **Swapped-First (SF)**：恢复 swap 请求优先 → TBT 保住但行为退化为 FCFS，swap 空间利用不足。

Fig. 1 (Qwen2.5-32B, ShareGPT) 显示，在 RPS=20 时 WF 的 P99 TTFT 比 FCFS 改善约一个数量级，但 P99 TBT 退化两个数量级。**两个静态策略互相冲突，没有"中间态"**。

**短板二：PCIe 带宽天花板限制响应性（Insight #2）。** 即便策略最优，PCIe Gen5 x16 单向 ~64GB/s 也无法跟上高 RPS 下的 swap 需求。Fig. 2 (Qwen2.5-32B, RPS=20) 表明：把 swap 带宽从 ~50GB/s 提升到 150GB/s，可让 P99 TTFT 减半、P99 TBT 显著下降。背后的因果链 (Fig. 3)：
- 低带宽 → 抢占 / 换出慢 → 等待队列 backlog 不能及时清理 → TTFT 违约；
- 低带宽 → 换入慢 → swapped 队列产生**新的 HOL blocking** → TBT 违约。

**结论：PCIe 是 offloading 范式的硬上限**。

### 2.3 Superchip 趋势：紧耦合 GPU+CPU 的体系结构

Superchip 是当前体系结构最重要的趋势之一。代表性产品：

| 平台 | GPU 侧 | CPU 侧 | 互连 | 单边带宽 | 一致性 |
| --- | --- | --- | --- | --- | --- |
| **NVIDIA GH200** | Hopper (H100 等价) 96/144GB HBM3 | Grace ARM Neoverse V2，480GB LPDDR5X | NVLink-C2C | 900 GB/s 双向 | cache-coherent |
| **NVIDIA GB200 NVL72** | 2× Blackwell + 1× Grace per chip，72 GPU 一个机柜 | Grace ARM | NVLink-C2C + NVLink 5 | 1800 GB/s C2C | cache-coherent |
| **AMD MI300A** | CDNA3 GPU + Zen4 CPU 共封装 | 共享 HBM3 128GB | Infinity Fabric | ~5.3TB/s 内部 | unified memory |

Superchip 的范式特征：
1. **超出 PCIe 一个数量级的链路带宽**（C2C 900GB/s vs PCIe Gen5 x16 ~64GB/s）；
2. **cache-coherent**：GPU 可以直接发 load/store 到 CPU DRAM，无需显式 DMA（论文中提到 GH200 的 ATS Address Translation Services）；
3. **统一物理地址空间 / Unified Memory (UM)**：硬件可在 HBM 和 DRAM 间迁移页（GH200 是基于 hardware access counter 的迁移，不是 page-fault 触发）；
4. **NUMA + DRAM 带宽瓶颈**：Grace 每个 NUMA 节点 384GB/s DRAM 带宽，**比 C2C 链路本身更紧的瓶颈**；
5. **半双工 DRAM**：D2H + H2D 同时进行时被 384GB/s 的 DRAM 带宽限制（C2C 链路本身可以 450+450 双工）。

GH200 在 LLM 训练（SuperOffload、Lian et al. 2025）和初步推理（Pie）上有探索，但**针对 SLO-aware 推理的系统几乎为零**。这正是 SuperInfer 的契机。

### 2.4 GH200 上的"反直觉现象"：C2C <5% 利用率（Insight #3）

最让人意外的是论文 Fig. 4 / Fig. 5 / Fig. 12 展示的**软硬件错配**：

- vLLM 的 offloading engine 在 GH200 上跨 LLaMA-3-8B / Qwen2.5-32B / Mixtral-8x7B 实测带宽都只有 **~10GB/s**，<5% 的 C2C 理论峰值；
- 用 nvbandwidth 微基准测得 C2C 双向各方向 ~200+200 GB/s 即可饱和（受 Grace DRAM 384GB/s 半双工限制）；
- 单 segment 大小 ≤ 64KB 时带宽急剧跌到 <10GB/s，因为 cudaMemcpyAsync 的 kernel launch 时间 ≥ 实际拷贝时间（在 ≤ 4MB 时尤其明显）。

**为什么会差距这么大？**核心原因有二：
1. **PagedAttention 的 layer-first 布局把 KV cache 切成 NL × NB 个 64KB 小段**：以 Qwen2.5-32B 为例（NL=64 layer，C=4 bytes/token KV，P=16 token/block），单 segment = 64KB，整个 block 4MB；这些段在物理内存上是非连续的；每段都要单独发起 cudaMemcpyAsync。
2. **vLLM/SGLang 的 swap-in 与 swap-out 串行**，且没人尝试过双工 D2H+H2D（因为存在 HBM block 共享导致的 data race）。

简言之，**Superchip 把硬件瓶颈消除了，但软件栈仍然在按 PCIe 时代的方式 offload，C2C 链路实际上是空闲的**。这是 SuperInfer 论文最重要的体系结构 insight。

---

## 3. 核心思想 / 方法

SuperInfer 的设计可以归纳为两句话：
- **调度层（RotaSched）**：把"被动抢占"改造成"主动旋转"，以 OS 时间片调度为隐喻，用 VLT 衡量请求是否"落后"，按 LVF 策略让濒临 SLO 违约的请求优先占用 HBM；
- **数据层（DuplexKV）**：把碎片化 KV layout 改造成块优先布局 + 批量 kernel + 提前 sync + 全双工流水线，使得 RotaSched 的高频大体积旋转在带宽上是"免费"的。

### 3.1 RotaSched：从 passive preemption 到 active rotation

#### 3.1.1 OS 类比

SuperInfer 把 GH200 LLM 推理栈看作 OS：

| LLM 服务概念 | OS 概念 |
| --- | --- |
| Request | Thread |
| KV cache | Thread data |
| Hopper HBM | On-chip cache |
| Grace DRAM | Main memory |
| GPU SM | CPU core |
| 调度迭代 | OS 时间片 |

OS 上的 CFS / EEVDF 等调度器之所以能在数百线程间优雅切换，本质是因为 **(a) 抢占自由、(b) 上下文切换成本低（硬件管 cache 预取）、(c) 公平度量（virtual runtime / lag）足够便宜**。

LLM 推理上做不到 (a)(b)(c) 中的任何一项（KV cache 上下文切换是 GB 级数据搬运），但 GH200 把 (b) 的成本降到几十毫秒甚至更低；这就给了把 OS 调度思路搬过来的空间。

#### 3.1.2 三态状态机：waiting / running / rotary

SuperInfer 引入了一个新的瞬态 **rotary** 状态：

- **waiting**：刚到达，KV 都还没建（甚至 prompt 还没 prefill）；
- **running**：当前在 HBM 上跑 decode；
- **rotary**：被主动换出，KV 从 HBM 搬到 DRAM 暂存，等下一轮被换回。

与传统 swap 不同，**rotary 不是出错路径**（不是 OOM 才进），而是**调度器主动选择的瞬态**——只要某个 running 请求"已经超额占用 HBM 时间"且某个 waiting/rotary 请求"已经接近 SLO 违约"，就会触发 rotation。

#### 3.1.3 Virtual Lag Time (VLT)

VLT 是 SuperInfer 的"调度货币"。其受 EEVDF 的 lag 思想启发但适配了 LLM 推理的两点特殊性：(i) 上下文切换不是免费的，是 GB 级搬运；(ii) 同时要管两个 SLO（TTFT 与 TBT）。

定义如下（论文 §4.2.2 公式）：

```
            ┌  α · ReLU(t_now − t_last  − β_B · S_B),  if rotary
VLT(req) = ┤   ReLU(t_now − t_arr  − β_F · S_F),       if waiting
            └  −(t_now − t_run),                        if running
```

其中：
- `S_B` = TTFT SLO（论文 5s），`S_F` = TBT SLO（论文 100ms）；注意论文文本中 β_B 对应 TTFT、β_F 对应 TBT，但公式形式上是把 `β · S` 当成"容忍偏移"来减；
- `α ≥ 0`：TBT/TTFT 灵敏度权重，论文默认 α = 3；
- `β_B, β_F ∈ R`：分别为 TTFT 与 TBT 的容忍系数，论文默认 β_B = 0、β_F = 0.5；
- `t_now`：当前时间；
- `t_last`：rotary 请求上次生成 token 的时间；
- `t_arr`：waiting 请求到达时间；
- `t_run`：running 请求开始 running 的时间。

直觉解读：
- **running 请求 VLT 永远 ≤ 0**，且越跑越负 → 越是"占用过久"的 running 请求，VLT 越小，越优先被踢出；
- **waiting / rotary 请求 VLT 从 0 开始**，超过容忍阈值后转正并继续增大 → "等得越久"的请求 VLT 越大，越优先被换入；
- **α** 控制 TBT 相对 TTFT 的紧迫度。论文实验显示 α 越大 TBT 越好但 TTFT 退化（rotary 优先 → 新到达请求被推迟）；
- **β_B, β_F** 控制对 SLO 容忍多少：β_F 越大 → waiting 请求"心理预期"等得起 → 进入 VLT > 0 越晚 → TTFT 越差但 TBT 越好；β_B 越小（甚至负）→ rotary 请求被早早判为"落后"→ TBT 越好。

这种参数化让运维者可以根据场景偏置（summarization 偏 TTFT、chat 偏 TBT）做权衡。

#### 3.1.4 Largest-VLT-First (LVF) 调度算法

LVF 在每个 engine iteration 里执行四步（论文 Algorithm 1）：

```
输入：Q_R (running), Q_W (waiting), Q_S (rotary), 块数函数 blk(·)
      transfer budget B_xfer，当前 HBM 空闲块数 B_HBM
输出：preempted set P，prioritized set R

1. P ← ∅, R ← ∅, B_left ← B_HBM + B_xfer
2. L ← Q_R ∪ Q_W ∪ Q_S
3. ① Contention Check：若 B_HBM ≥ Σ blk(r), r ∈ Q_W ∪ Q_S → fall back FCFS
4. ② Sort：按 VLT 降序排 L
5. ③ Prioritize：从 L 头扫，VLT(r) ≥ 0 且 blk(r) ≤ B_left → 加入 R，更新 B_left
6. ④ Preempt：B_swap = B_xfer − B_left；从 L 尾扫（最负的 VLT）→ VLT(r)<0 且 B_swap>0 → 加入 P，释放 blk(r)
7. return (P, R)
```

关键设计点：

1. **当 HBM 够用时直接 fall back 到 FCFS**，避免无谓 overhead；
2. **prioritize 同时把 waiting 与 rotary 视为"可执行候选"**，按 VLT 排序，**没有 WF/SF 的偏置**——这是相对生产框架的根本改进；
3. **transfer budget B_xfer 是带宽的旋钮**：每 iteration 最多 swap 多少 block。在 PCIe 系统里 B_xfer 受限（论文实验中 B_xfer 从 300 提到 4800 时 P99 TTFT 显著下降），在 GH200 上可放大到 2400（论文默认）；
4. **preempt 与 prioritize 在同一 iteration 内一起决策**，不存在"先驱逐再决定换入"的两段式。

#### 3.1.5 conceptual 例子（论文 Fig. 9）

4 个请求 R1–R4，HBM 只能装 2 个。VLT 数值随时间演化：
- t=0：R1 arr，R1 running（VLT=0），其他还没到；
- 进展中 R3、R4 陆续进入 waiting，VLT 上升为正；R1、R2 running 的 VLT 越来越负；
- 当 R3 的 VLT 超过 R1 的负 VLT 绝对值时，LVF 把 R1 移到 rotary，把 R3 拉到 running；
- 后续每隔一段时间继续旋转，避免任何单个请求"霸占"HBM 太久。

这就是"OS 时间片"在 LLM 推理上的等价物——但片长不是固定的 quantum，而是被 SLO 进度（VLT）触发。

### 3.2 DuplexKV：高带宽 KV cache 旋转引擎

LVF 之所以可行，前提是 swap 必须"足够便宜"。否则 transfer budget 一大就会把模型计算阻塞——论文 Fig. 17 显示 SuperInfer w/o DuplexKV 在大 B_xfer 下 TBT 反而塌方。DuplexKV 解决三件事：

#### 3.2.1 为什么 NVLink-C2C 利用率 <5%（论文 §4.3.1）

回顾 §2.4 的瓶颈：
- **layer-first KV layout** 让单次 contiguous segment 只有 64KB（NL=64 时），远小于 NVLink-C2C 进入"高带宽区"所需的 ~8MB；
- **每段都用一个独立 cudaMemcpyAsync** → kernel launch 时间在 segment ≤ 4MB 时大于实际拷贝时间；
- **swap-in / swap-out 串行**：vLLM、SGLang 都没做双工。

#### 3.2.2 Block-First KV Layout（数据布局变换）

把原本 (N_L, N_B, S_seg) 三维张量从 **layer-first**：
```
顺序：(layer=0, block=0), (layer=0, block=1), ..., (layer=0, block=NB-1),
      (layer=1, block=0), ...
```
改为 **block-first**：
```
顺序：(block=0, layer=0), (block=0, layer=1), ..., (block=0, layer=NL-1),
      (block=1, layer=0), ...
```

- 一个 block 的所有 layer 连续 → 单次连续区域从 64KB 涨到 NL × 64KB = 4MB（Qwen2.5-32B）；
- PagedAttention kernel 需要扩展：原来 block-i 与 block-j 之间的 stride 是 `(j−i)·S_seg`，现在变为 `(j−i)·N_L·S_seg`；论文修改了 PagedAttention kernel 以支持新 stride，同时保留 paging 抽象（block table、free list 等不变）；
- 这一变换让 transfer 进入 NVLink-C2C 的"高带宽区"。

#### 3.2.3 cudaMemcpyBatchAsync：批量 kernel launch

把同一方向（HBM→DRAM 或 DRAM→HBM）的所有 transfer descriptor 装进一个 cudaMemcpyBatchAsync 调用：
- 消除每个 segment 一次 launch overhead；
- 与 block-first layout 配合后，launch 次数从"NL × N_blocks"降到 "1 per direction per iteration"。

#### 3.2.4 Eager Block Rotation：消除 H2D / D2H 的 data race

朴素地为 H2D 和 D2H 各开一条 CUDA stream 是不行的，因为：
- swap-in 的目的 HBM block 可能正是某个 swap-out 释放的源 block；
- 这种 RAW/WAR 依赖会让 swap-in stream 等 swap-out 完成 → 退化为串行。

DuplexKV 的关键观察：**KV cache 是 incremental 的**。一个 block 一旦被写满（"synced"）就不会再被改，直到请求完成。这意味着可以**提前**把 synced block 复制到 DRAM——即"eager rotation"。

机制：
1. 给每个 block 标记 **dirty / synced**：dirty 是当前正在追加 token 的 block，synced 是已经写满、不会再变的 block；
2. 后台流持续把 synced block 从 HBM 拷贝到 DRAM，**即便请求还没被抢占**；
3. 当请求真正被 LVF 选中要 preempt 时：synced block 已经在 DRAM 有副本 → **直接 discard HBM 拷贝**（不用再传），只需 swap-out 当前 dirty block 一个；
4. swap-in 拿到的目的 HBM block 与 swap-out 释放的源 block **永远不重叠**（因为后者 dirty block 已经搬完后整个请求就走了），data race 消失，**两个 stream 真正并发**；
5. CPU 侧的备份保证正确性，HBM 上 discard 不带来额外内存压力。

block table（论文 Fig. 6 右侧）跟踪每个块的 (HBM 地址, DRAM 地址, dirty/synced 标记)。

#### 3.2.5 跨迭代流水线（cross-iteration pipeline）

最后用一条流水线把所有事情藏到 GPU 计算后面（论文 Fig. 15）：
- 在 iteration t，GPU 跑 iteration (t−1) 准备好的 batch；
- 同时 RotaSched 与 DuplexKV 在 host 侧并发准备 iteration (t+1) 的 batch，包括调度决策、D2H、H2D 三件事用两个 CUDA stream 完成；
- 通过 batch t+1 的 prepare 完全 hidden 在 model exec t 后面，**只要 sched + transfer < model exec，GPU 就不 stall**。

论文实测 79% 调度开销 + 16% 转移 + 70% 模型执行（实际数字：scheduling 7.63ms / KV transfer 15.8ms / model exec 69.82ms），**只有 0.021% 的 iteration 出现 overlap 失败**。

### 3.3 整体协同：RotaSched × DuplexKV

RotaSched 和 DuplexKV 不是简单堆叠，而是相互成就：

- **RotaSched 需要高 B_xfer 才能有效**：transfer budget 越大 → 每 iter 可以旋转越多 block → 越能及时清理 backlog（§3.2 Insight #2）；
- **大 B_xfer 在 vLLM 引擎下会反噬**：因为 transfer 太慢、不能 hidden 在 model exec 后 → 反而让 TBT 更差（Fig. 17 SuperInfer w/o DuplexKV (H)）；
- **DuplexKV 把 transfer 时间压到 model exec 的 1/4 左右**：让大 B_xfer 真正可用；
- **没有 RotaSched，DuplexKV 也不知道该 swap 哪些 block**：单靠双工链路只能同时搬两个方向的随机数据，不解决 SLO 问题。

二者构成一个闭环：**调度提需求、数据通路兑现需求、需求兑现使更激进的调度成为可能**。

---

## 4. 实现 / 工程细节

### 4.1 软件实现

- **基线代码库**：vLLM v0.6.6.post1（V1 engine），约 12 GB 额外运行时内存预算与原 vLLM 持平；
- **语言**：Python + C++（CUDA kernel 与 DuplexKV 的批量 transfer 控制路径）；
- **CUDA / GCC**：CUDA 12.8、GCC 13.3.0；
- **PyTorch**：2.5.1；
- **kernel 修改**：扩展 vLLM 的 PagedAttention kernel 以支持 block-first layout 的新 stride；
- **License**：Apache 2.0；
- **artifact**：已提交 Zenodo（DOI: 10.5281/zenodo.18971768），并通过 Docker 镜像 `monsoon235/superinfer_ae_public` 提供完整复现环境。

### 4.2 硬件平台

- **GH200 NVL2**：每个节点 2 个 Grace–Hopper Pair，单 Pair：
  - Hopper GPU 144GB HBM3
  - Grace CPU 480GB LPDDR5X
  - NVLink-C2C 900GB/s 双向
  - DRAM 带宽 384GB/s（半双工）
  - 两个 Hopper 之间用 NVLink 900GB/s 互联（用于 TP=2 实验）
- **NUMA 配置**：用 `numactl --cpunodebind=0 --membind=0` 把所有内存绑到同一 Superchip，避免跨 NUMA 流量；
- **OS**：Ubuntu 24.04.3 LTS，kernel 6.8.0-100-generic-64k；
- **存储**：约 500GB（模型权重 + 数据集 + Docker 镜像）。

### 4.3 NVLink-C2C 与 PCIe 的体系结构差别

论文 Fig. 5 用 nvbandwidth 工具（v0.8，Test ID 2 host→device bidirectional, Test ID 3 device→host bidirectional）实测：

| 链路 | segment ≤ 64KB | segment 1MB | segment 8MB+ | 双工合计 |
| --- | --- | --- | --- | --- |
| PCIe Gen5 x16 (H200) | <5 GB/s | ~30 GB/s | ~60 GB/s | ~120 GB/s（双工） |
| NVLink-C2C (GH200) | <10 GB/s | ~80 GB/s | ~200 GB/s | ~384 GB/s（半双工 DRAM 限制） |

关键观察：
- segment 大小 < 8MB 时，C2C 的优势被 launch overhead 抵消，**反而比 PCIe 没强多少**；
- segment 大于 8MB 后 C2C 才进入"应许之地"；
- 因此 SuperInfer 把整个 block（4MB，64 layer × 64KB segment）合并是非常关键的工程细节。

### 4.4 Chunked Prefill 与 KV cache 管理

- **Chunked prefill** (Sarathi-Serve)：把长 prompt 切成 chunk 与 decode 混合 batch，平滑 TTFT/TBT 抖动。SuperInfer 与所有基线都启用这一特性；
- **KV cache block size**：16 token / block（vLLM 默认）；
- **transfer budget B_xfer = 2400 blocks/iter**（默认，约 38400 token / iter）；
- **DRAM 预留**：480GB DRAM 中拨 400GB 给 KV cache offload，80GB 留 OS 和 runtime；
- **block table** 维护 `<request_id, block_id> → (HBM_addr or NULL, DRAM_addr or NULL, dirty/synced)`，支持 O(1) 状态查询；
- **eager rotation 策略**：所有 synced block 在 swap budget 允许范围内尽量早地复制到 DRAM，作为后台 best-effort。

### 4.5 LVF 的参数选择

论文给出的参考值（Qwen2.5-32B + ShareGPT，TTFT SLO 5s、TBT SLO 100ms）：
- `α = 3`（TBT/TTFT 灵敏度，TBT 较紧时 α ≥ 3）；
- `β_B = 0`（TTFT 不容忍）；
- `β_F = 0.5`（TBT 容忍 50ms）；
- `B_xfer = 2400 blocks`。

并通过参数扫描（Fig. 18–20）给出经验法则：
- **TTFT-sensitive** 任务（summarization、长文翻译）：α ≤ 1；
- **TBT-sensitive** 任务（chat 流式输出）：α ≥ 3，β_B 取负（如 −10）；
- α = 3 是 sweet spot，再大边际收益递减但损害 TTFT 显著。

### 4.6 多 GPU（TP=2）扩展

RotaSched 和 DuplexKV 都是 **per-GPU local** 的：
- RotaSched 决策只用本机请求队列与 HBM/DRAM 状态；
- DuplexKV 在每个 Grace–Hopper pair 内独立工作；
- 跨 GPU 的 NVLink 流量与 SuperInfer 完全正交。

因此 TP=2 实验（Qwen2.5-32B、Mixtral-8x7B）显示与 vLLM 对比的 SLO 改善依然显著，证明设计**与并行策略解耦**。

---

## 5. 评测

### 5.1 实验设置

- **模型**：LLaMA-3-8B（小密集）、Qwen2.5-32B（大密集）、Mixtral-8x7B（MoE，NEO 不支持因此该列被排除 NEO）；
- **数据集**：ShareGPT、LMSYS-Chat-1M；每次实验采样 120 × RPS 个请求模拟 120s 流量，到达间隔服从 Poisson；
- **基线**：vLLM v0.6.6（V1 engine）、TensorRT-LLM v1.1.0、LightLLM v1.1.0（Past-Future scheduler）、LTR（学习排序近似 SJF）、NEO（KV+Attention 部分卸载到 CPU）；
- **未对比**：Pie、HeteGen、Select-N（无开源代码）、FlexGen（缺 PagedAttention 与 chunked prefill 不公平）、InfiniGen（lossy 不可比）；
- **指标**：TTFT SLO 命中率、TBT SLO 命中率、Throughput（token/s）、有效 swap 带宽（GB/s）、E2E 拷贝时间（ms）。

### 5.2 主结果（Fig. 16）

跨 6 个 (model, dataset) 组合在不同 RPS 下绘制 TTFT 与 TBT SLO 命中率曲线：

- **TTFT SLO 命中率**：在高 RPS（接近系统饱和点）下 SuperInfer 比最强基线高 **最多 74.7%**；
- **TBT SLO 命中率**：与 TensorRT-LLM、LightLLM 相当或更优；
- **低 RPS 下**：所有方法都接近 100%，SuperInfer 没有引入额外开销，验证收益完全来自 SLO-aware offloading；
- **LTR**：TTFT 强但 TBT 塌方，因为静态 deadline 优先级；
- **LightLLM**：TBT 随 RPS 升高反而稳定（Past-Future scheduler 避免有害驱逐，CDF 几乎不变 — 见 Appendix C 与 Fig. 25）；
- **TensorRT-LLM**：TBT 不错但 TTFT 在高 RPS 下显著退化（lazy preemption + offload 拖慢新请求 prefill）；
- **NEO**：作为代表性 CPU offload 方案在高 RPS 下两个指标都不如 SuperInfer。

### 5.3 消融实验（Fig. 17）

四档配置在 Qwen2.5-32B + ShareGPT 上：
1. **vLLM (FCFS)**：基线；
2. **SuperInfer w/o DuplexKV (L)**：B_xfer = 300（小预算，对应 vLLM offload engine 真实带宽）；
3. **SuperInfer w/o DuplexKV (H)**：B_xfer = 2400（大预算，但没有 DuplexKV 兜底）；
4. **SuperInfer 完整版**：RotaSched + DuplexKV，B_xfer = 2400。

观察：
- (2) 已经显著改善 TTFT（证明 RotaSched 单独有效）；
- (3) 反而把 TBT 拖垮（transfer 来不及 hidden）；
- (4) TTFT 比 (2) 进一步提升，TBT 不退化；
- 印证：**没有 DuplexKV，激进调度反而有害；二者协同才解锁 Superchip**。

### 5.4 DuplexKV 带宽实测（Table 1）

在 Qwen2.5-32B、双向各 8GB（32768 token）的 KV cache 上：

| 方法 | D2H | H2D | E2E (ms) | 备注 |
| --- | --- | --- | --- | --- |
| Naive | 10.75 (U) | 9.86 (U) | 1556.15 | 复刻 vLLM，64KB/段 |
| MS（block-first，merged segments） | 80.05 (U) | 133.51 (U) | 159.87 | 单方向带宽飙升 |
| MS+MK（merged kernels） | 238.95 (U) | 269.69 (U) | 63.14 | 单方向已逼近 NVLink-C2C 物理上限 |
| **DuplexKV（MS+MK+eager rotation）** | 180.99 (B) | 179.37 (B) | **46.80** | 双工，达到 ideal 384GB/s 半双工 ~94% |
| Ideal（理论） | 192.00 (B) | 192.00 (B) | 41.66 | DRAM 半双工 384GB/s 极限 |

亮点：
- naive 只有 ideal 的 5.6% 带宽、37.4× 时间；
- MS+MK 单方向已经几乎达到 NVLink-C2C 单边物理峰；
- **eager rotation 是真正打开双工的关键**，让总带宽再上一个台阶。

### 5.5 swap 带宽对延迟的敏感性（Fig. 21）

固定其他参数变 B_xfer（300 → 4800）：
- B_xfer = 4800 时 P99 TTFT 比 B_xfer = 300 低一个数量级；
- P99 TBT 也单调下降；
- 说明：**就算调度算法完美，没有 NVLink-C2C 提供的高带宽，offloading 仍然救不了高 RPS**。

### 5.6 多 GPU 扩展（Fig. 22）

TP=2 配置（Qwen2.5-32B、Mixtral-8x7B）下 SuperInfer 相对 vLLM 的优势依旧。RotaSched/DuplexKV 是本地决策，不引入额外跨 GPU 通信，**与并行策略正交**。

### 5.7 吞吐量（Fig. 23）

- LLaMA-3-8B、Qwen2.5-32B、Mixtral-8x7B 三个模型在不同 RPS 下；
- SuperInfer 与 vLLM 持平到 **+29.2%**（高 RPS）；
- 原因：fast rotation 让 prefill 请求获得更多 chunked prefill batching 机会，避免 long request 长期独占 HBM。

### 5.8 与 GH200 Unified Memory (UM) 的对比（Appendix D）

读者一个自然的问题是："既然 GH200 有硬件管理的 UM，何不直接把 KV cache 放进 UM？"论文专门做了 ablation（Fig. 26）：vLLM + UM 的 TBT 显著恶化。原因：

- GH200 的 UM 不是 page-fault 触发，而是 **hardware access counter** 触发后台迁移；
- KV cache 访问模式短而瞬时（一次 attention 计算之后就走），counter 还没攒够频率，迁移就不会发生；
- 那些 KV 块永远在 DRAM 里被 GPU 通过 ATS 访问，只能拿到 384GB/s（C2C 链路 + DRAM bw），远低于 HBM 4TB/s → **bandwidth cliff**；
- 整个请求处理期都在"warming up"已经凉掉的数据。

**结论：把决策交给硬件 UM 不可行，必须由软件显式调度（RotaSched + DuplexKV）。**

---

## 6. 思想精读 / 启示

### 6.1 Superchip 体系结构对推理系统设计的范式影响

SuperInfer 对系统社区最大的启示是：**Superchip 不是 PCIe 的"快一点版本"，而是一个新计算模型**。具体到几个层面：

1. **链路带宽与 DRAM 带宽的相对关系翻转**。在 PCIe 时代，瓶颈是链路（PCIe Gen5 ~64GB/s vs DDR5 ~400GB/s）；在 Superchip 时代，瓶颈是 DRAM（NVLink-C2C 900GB/s vs Grace LPDDR5X 384GB/s 半双工）。系统设计原则要从"压榨链路"转为"压榨 DRAM"，包括 NUMA 局部性、半双工避让、双工流水线。
2. **GPU 不再是孤岛**。GPU 可以不通过 DMA、不通过显式 cudaMemcpy，直接 load/store CPU DRAM。但这不意味着"统一编程模型让一切自动美好"——UM 实验已经反证。**软件依然要决定何时迁移、迁移什么粒度**，只是迁移本身变便宜了。
3. **OS 范式回归**。当 context switch 成本从"不可接受"变成"可接受"，整个调度词典（lag、fairness、time-slice、preemption、CFS、EEVDF）都重新可用。RotaSched 是这个回归的第一个具体实例，可以预期 GB200 时代会出现更激进的 OS-style serving stack。
4. **数据布局是新的一阶问题**。PagedAttention 在 PCIe 时代是好设计（防内存碎片），在 Superchip 时代成了瓶颈（碎片化的 KV segment 让 C2C 跑不满）。block-first layout 不只是局部优化，而是**为新链路重新设计内存抽象**。这一思路可推广到其他被 PagedAttention 隐藏的瓶颈，例如分布式 KV cache 路由。

### 6.2 一致内存空间是否真的有用？

论文给出的回答是**有用，但不能依赖硬件 UM 自动管理**。核心论据：

- **有用**：cache-coherent + ATS 让 GPU 直接访问 DRAM 的 attention 计算成为可能（即便慢，但不需要 page fault）；让"discard on preemption"机制可行（HBM 副本可以丢，因为 DRAM 上有可读副本，且语义一致）；
- **不能自动**：KV cache 的访问模式（短、稀疏、瞬时）和 access-counter 驱动的 UM 迁移机制不匹配；UM 不知道哪些 block 是 SLO-critical；UM 也不知道 prefill / decode 的访问差异。

启示：**"GPU 与 CPU 共享地址空间"是必要条件不是充分条件**；要让 LLM 推理利用这个条件，仍然需要应用层的语义介入（RotaSched 知道哪个请求快违约 SLO；DuplexKV 知道哪个 block synced 可以安全 discard）。这与 OS 历史上"虚拟内存 + page replacement"需要应用 hint（madvise）的结论一致。

### 6.3 与传统 PCIe 架构对比

| 维度 | PCIe 架构 | Superchip 架构 |
| --- | --- | --- |
| GPU↔CPU 链路 | PCIe Gen5 x16 ~64GB/s | NVLink-C2C 900GB/s 双向 |
| 一致性 | 非 cache-coherent，需 DMA | cache-coherent，可直接 load/store |
| Offloading 范式 | 静态 / 反应式 | 主动 / 旋转式 |
| KV layout 关键性 | 中（被 PCIe 链路掩盖） | 高（决定能否进入高带宽区） |
| 调度复杂度上限 | 低（swap 太贵不能频繁） | 高（OS 风格调度可行） |
| DRAM 带宽 | 不是瓶颈 | 新瓶颈 |
| Programming model | CUDA + 显式 memcpy | UM / ATS + 显式 memcpy 共存 |

SuperInfer 实质上是在证明：**整个 LLM serving 软件栈需要为 Superchip 重写，而不是补丁**。

### 6.4 与 Mooncake / Pie / NEO 等系统的差异

- **Pie**：同样在 GH200 上做 KV spilling 和 on-the-fly 内存重分配，但**没有 SLO-aware 调度，也没有专门的全双工 transfer engine**。SuperInfer 把这两个维度补齐。
- **Mooncake**：核心是分布式 KVCache 化的 P/D 分离架构，更强调跨节点的 KV 复用；与 SuperInfer 的本地 GPU-CPU 旋转思路是正交关系，可以叠加（SuperInfer 处理单 Superchip 内部，Mooncake 处理跨节点）。
- **NEO**：把 attention 与 KV 都搬到 CPU，PCIe 时代的折衷方案。在 GH200 上仍然不如 SuperInfer，因为没有针对 NVLink-C2C 的 layout 优化。
- **SuperOffload**：同实验室前作，针对训练；SuperInfer 是推理版本的自然延伸。
- **Aqua**：把 KV offload 到其他 GPU（用 NVLink），与 SuperInfer 的 CPU DRAM 思路互补——前者扩展集群级共享 KV pool，后者扩展单机内层级。

### 6.5 对 GB200 NVL72 / MI300A 的可迁移性

GB200 NVL72：每 chip 1× Grace + 2× Blackwell，C2C 带宽翻倍至 1800GB/s；72 GPU 通过 NVLink-5 互联。SuperInfer 的设计**直接可迁移**，且以下几点会更突出：
- 单机 KV 容量更大 → 旋转空间更大 → LVF 收益更大；
- C2C 带宽更高 → B_xfer 可以再上一个台阶；
- DRAM 带宽仍然可能是瓶颈，需要 block size 重新调参。

MI300A：CPU+GPU 共享 HBM3，没有 host DRAM 这一额外层级。SuperInfer 的"双层旋转"在 MI300A 上退化为"批内换出/换入"，但 RotaSched 的 VLT 思路依然适用——它本质上是个 SLO-aware 公平调度器，与具体的内存层级解耦。

### 6.6 OS 调度思想 → LLM Serving 的更深层启示

如果把 SuperInfer 视为"把 EEVDF 搬进 LLM serving"的具体实例，下一步问题自然延伸：
- **多 SLO**（如混合 chat + agent + RAG）：能否在 VLT 上引入多维度 lag？
- **带优先级类**：能否模仿 Linux SCHED_RT/SCHED_OTHER 设计 LLM 的优先级类？
- **公平性 + SLO**：与 Sheng et al. 2024 的公平 serving、Equinox 的 holistic fair scheduling 如何统一？
- **Token-level preemption**：当前 rotation 粒度是请求级；token 级 preemption 是否能进一步降低尾延迟？
- **Heterogeneous Superchip cluster**：GH200 + H200 + GB200 异构集群下如何全局 VLT？

这些都是 SuperInfer 之后非常自然的开放问题。

---

## 7. 局限与开放问题

### 7.1 论文显式提到的局限

1. **参数手动调**：α、β_B、β_F、B_xfer 仍需运维针对场景调；论文明确"SuperInfer 不尝试预测查询分布"，把调参留给应用方；
2. **TTFT/TBT 二分法**：SLO 模型只考虑 TTFT 与 TBT，未涵盖 e2e latency、token throughput-per-user 等其他 SLO 指标；
3. **无 lossy 优化**：未与 quantization、sparse attention、speculative decoding 结合，留作未来工作；
4. **未对比 Pie / HeteGen / Select-N / FlexGen / InfiniGen**：因代码不开源或评测不公平。

### 7.2 我看到的局限

1. **DRAM bandwidth 仍是天花板**：DuplexKV 已经接近 384GB/s 半双工，但 DRAM bw 不是免费的，model weight 加载、gradient（推理无）、其他系统进程都要分摊；论文没有评估在多租户 GH200 上是否会被其他 workload 抢 DRAM bw。
2. **块粒度的代价**：4MB block 意味着即便 swap 一个仅有几百 token 的小请求，也得搬 4MB；对超短请求可能有放大效应（论文未量化）。
3. **Eager rotation 的功耗 / DRAM bw 成本**：后台持续把 synced block 拷贝到 DRAM 占带宽；对于"很快就完成、根本不会被 preempt"的请求，这是浪费。论文没有给出"虚拟拷贝率"统计。
4. **VLT 模型的可解释性 / 收敛性**：α 和 β 的最优解是否随 workload mix 漂移？是否有自动化策略（RL 调参、gradient-free search）？论文留空。
5. **跨 NUMA / 跨 Superchip 行为**：GH200 NVL2 包含 2 个 Grace–Hopper pair，论文用 numactl 把内存绑死在一对内；如果 KV cache 必须跨 pair（HBM 联合大模型），Superchip-to-Superchip C2X bandwidth、同一 chassis 内 GPU 间 NVLink 的扮演角色都没讨论。
6. **TBT 长尾在低 RPS 下的细节**：消融实验主要在 RPS=10–20，对极低 RPS（RPS<5）下 RotaSched 是否引入不必要 rotation overhead 的研究不足。
7. **rotary 状态的状态机复杂度**：增加一个状态对系统正确性证明有压力；论文没有证明 rotation 不会饿死任何请求（理论上 VLT 增长保证不饿死，但缺乏形式证明）。
8. **artifact 重现成本**：完整复现需要 GH200 NVL2 + 30 小时 + 500GB 存储，不是普通研究者能复现的——这是 Superchip 工作不可避免的代价。

### 7.3 自然延伸方向

- **Token-level rotation**：现在 rotation 单位是请求；能否做 token 级（保留前缀 KV，只 swap 增量部分）？
- **多请求联合 layout**：当多个被 preempt 的请求 block 同时换出时，能否在 DRAM 侧直接 group 起来便于后续批量换入？
- **跨节点 RotaSched**：在 NVL72 等 rack-scale 域内做全局 VLT 调度；
- **SLO-aware 编译**：把 RotaSched 的语义编译进 Triton / CUDA Graph，进一步压低 schedule overhead；
- **与 P/D 分离结合**：SuperInfer 是单实例，在 P/D 分离 + Superchip 部署中能否替换 prefill 实例的 KV pool 管理。

---

## 8. 关键术语速查表

| 术语 | 含义 |
| --- | --- |
| **Superchip** | 把 GPU 与 CPU 通过高带宽 cache-coherent 总线紧耦合到同一封装/同一节点的体系结构。代表：NVIDIA GH200、GB200 NVL72、AMD MI300A。|
| **NVLink-C2C** | NVIDIA Chip-to-Chip 互连，GH200 上提供 900GB/s 双向带宽，cache-coherent，是 NVLink 的 die-to-die 化。|
| **CXL.mem** | Compute Express Link 内存子协议，目标是让 CPU/GPU/加速器以 cache-coherent load/store 访问扩展内存。语义上类似 NVLink-C2C，但开放标准、目前带宽较低。|
| **Coherent memory / Cache-coherent** | 多设备访问同一物理地址时保持一致性视图，不需显式 flush/invalidate。|
| **Unified Memory (UM)** | NVIDIA 的硬件管理跨 CPU/GPU 内存抽象。GH200 上是 access-counter 驱动迁移，不再是 page-fault 驱动。|
| **ATS (Address Translation Services)** | PCI-SIG 标准，让设备能通过 IOMMU 访问 CPU 虚拟地址空间。在 GH200 上让 GPU 可直接 load/store CPU DRAM。|
| **HBM (High Bandwidth Memory)** | GPU 片上堆叠 DRAM，GH200 是 HBM3，144GB，~4TB/s 带宽。|
| **LPDDR5X** | Grace CPU 使用的低功耗 DDR5，GH200 480GB，单 NUMA 384GB/s。|
| **KV cache** | Transformer 自回归生成时缓存历史 token 的 K/V，避免重复计算；随序列长度线性增长；是 LLM 推理主要内存压力来源。|
| **PagedAttention** | vLLM 提出的 KV cache 分页机制，把 KV 切成固定大小 block 减少内存碎片。代价：layer-first 布局让 KV segment 散落，对 C2C 不友好。|
| **Prefill** | 处理输入 prompt 的第一阶段，并行计算所有 token 的 K/V，计算密集。|
| **Decode** | 自回归生成阶段，每次 forward 一个 token，受 KV cache 内存带宽限制。|
| **Chunked prefill** | Sarathi-Serve 提出的把长 prompt 切片与 decode 混合 batch 的技术，平滑 TTFT/TBT。|
| **TTFT (Time-To-First-Token)** | 从请求到达到首 token 返回的延迟。论文 SLO=5s。|
| **TBT (Time-Between-Tokens) / TPOT** | 相邻 token 间的延迟，决定流式输出体感。论文 SLO=100ms。|
| **SLO (Service Level Objective)** | 服务等级目标，如 P99 TTFT < 5s。SLO 命中率 = 满足 SLO 的请求百分比。|
| **HOL blocking (Head-of-Line blocking)** | 队列首部的慢请求阻塞后面所有请求，是高 RPS 下 TTFT 违约的主因。|
| **Continuous batching** | Orca 提出的逐 iteration 动态加入 / 退出请求的 batching，提高吞吐但不直接控延迟。|
| **EEVDF (Earliest Eligible Virtual Deadline First)** | Stoica & Abdel-Wahab 1995 提出的比例公平 OS 调度器，跟踪 lag 选择最早 deadline。Linux 6.6+ 采用。|
| **VLT (Virtual Lag Time)** | SuperInfer 提出的请求 SLO 偏离度量，受 EEVDF 启发，三种状态（waiting/running/rotary）有不同公式。|
| **LVF (Largest-VLT-First)** | 按 VLT 降序优先执行的策略，VLT 最负的 running 请求是 preempt 候选。|
| **Rotary state** | SuperInfer 引入的瞬态请求状态：KV 已 swap 到 DRAM、等待下一次 rotation。|
| **Eager block rotation** | 把 synced block 提前异步拷贝到 DRAM，preempt 时直接 discard HBM 拷贝，破除 H2D/D2H 的 data race。|
| **Block-first layout** | 把 KV cache 布局从 (layer, block) 改为 (block, layer)，让一个 block 的所有 layer 连续，单段 4MB。|
| **cudaMemcpyBatchAsync** | CUDA 12 引入的批量异步拷贝 API，一次 launch 提交多个 transfer，消除 per-kernel 启动开销。|
| **B_xfer (transfer budget)** | 每个 engine iteration 最多 swap 多少 block，反映系统的 swap 带宽预算。|
| **Half-duplex DRAM** | Grace LPDDR5X 在 D2H + H2D 同时进行时被 384GB/s 限制（共享物理通道），这是 NVLink-C2C 链路本身可以承载更高（450+450）的瓶颈所在。|
| **NUMA (Non-Uniform Memory Access)** | 多 socket 系统的内存访问局部性概念。GH200 NVL2 中两个 Grace–Hopper pair 是两个 NUMA node。|

---

## 9. 关键页码索引

| 主题 | 页 / 节 |
| --- | --- |
| Abstract（74.7% TTFT 提升） | p.1 Abstract |
| Insight #1：静态 SLO-unaware 策略两面不讨好 | p.3 §3.1, Fig.1 |
| Insight #2：PCIe 带宽天花板 | p.4 §3.2, Fig.2, Fig.3 |
| Insight #3：C2C <5% 利用率 | p.4 §3.3, Fig.4, Fig.5 |
| 整体架构图 | p.5 Fig.6 |
| OS 类比（线程 ↔ 请求） | p.5 §4.2.1, Fig.7 |
| VLT 公式与可视化 | p.6 §4.2.2, Fig.8 |
| LVF 算法流程图 | p.6 Fig.10 |
| LVF 概念例子（4 请求 / 2 槽） | p.6 Fig.9 |
| LVF 伪代码 | p.7 Algorithm 1 |
| layer-first 布局问题 | p.7 §4.3.1, Fig.11 |
| kernel launch vs transfer time 实测 | p.7 Fig.12 |
| Eager block rotation 数据竞争图 | p.8 Fig.13 |
| Block-first layout / merged kernels | p.8 Fig.14 |
| 跨迭代流水线图 | p.9 Fig.15 |
| 主结果（Fig.16，3 模型 × 2 数据集） | p.10 §5.2, Fig.16 |
| 消融（Fig.17） | p.10 §5.3.1 |
| α 扫描（Fig.18） | p.10 §5.3.2 |
| β_F 扫描（Fig.19） | p.10 |
| β_B 扫描（Fig.20） | p.11 §5.3.2 |
| DuplexKV 带宽实测 Table 1 | p.11 §5.3.3 |
| B_xfer 扫描（Fig.21） | p.11 §5.3.4 |
| 流水线 stall 仅 0.021% | p.11 |
| TP=2 多 GPU（Fig.22） | p.11 §5.3.5 |
| Throughput（Fig.23） | p.12 §5.3.6 |
| 结论 | p.12 §6 |
| Appendix A：FCFS vs SJF-Oracle | p.16 |
| Appendix B：nvbandwidth 测量方法 | p.16 |
| Appendix C：LightLLM 异常分析（Fig.25） | p.16 |
| Appendix D：UM 无效（Fig.26） | p.16-17 |
| Appendix E：Artifact 详细信息 | p.18-19 |

---

## 10. 一句话点评

> **SuperInfer 揭示了一个被忽略的事实：Superchip 的硬件红利不会自动落到 LLM 推理服务上——必须把 PagedAttention 的 KV 布局换成 block-first、把抢占式调度换成 OS 风格的 VLT 主动旋转、把 H2D/D2H 改造成全双工，才能把 NVLink-C2C 的 900GB/s 真正用起来；它是 PCIe 时代向 Superchip 时代过渡阶段最完整的一份"软硬协同"答卷，也为 GB200 NVL72 / MI300A 时代的 serving 系统设计立下了可借鉴的范式。**

