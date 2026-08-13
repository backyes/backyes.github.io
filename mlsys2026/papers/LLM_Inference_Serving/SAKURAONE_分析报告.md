# 论文分析报告 ·《SAKURAONE: An Open Ethernet–Based AI HPC System and Its Observed Workload Dynamics in a Single-Tenant LLM Development Environment》

> 本报告是对 MLSys 2026 (Industry Track) 论文《SAKURAONE: An Open Ethernet–Based AI HPC System and Its Observed Workload Dynamics in a Single-Tenant LLM Development Environment》(Konishi 等, SAKURA Internet) 的中文深度解读。同目录下 PDF 是原文。本文件聚焦"超节点总线 + 网络体系结构"维度，并辅以 LLM 训练 workload dynamics 的工程观察。

---

## 0. 元数据

| 项目 | 内容 |
| --- | --- |
| 标题 | SAKURAONE: An Open Ethernet–Based AI HPC System and Its Observed Workload Dynamics in a Single-Tenant LLM Development Environment |
| 作者 | Fumikazu Konishi*, Yuuki Tsubouchi*, Hirofumi Tsuruta* (* 同等贡献) |
| 单位 | SAKURA Internet Inc., Research Center, Japan |
| 通讯作者 | Fumikazu Konishi `<f-konishi@sakura.ad.jp>` |
| 会议 | The 9th MLSys Conference (2026), **Industry Track**, Bellevue, WA, USA |
| 论文页数 | 15 页 (含参考文献) |
| OpenReview ID | `n7o6C3p3wk` |
| OpenReview URL | https://openreview.net/forum?id=n7o6C3p3wk |
| 本地 PDF 路径 | `/Users/backyes/Library/Mobile Documents/com~apple~CloudDocs/paper/mlsys2026/mlsys2026_papers/n7o6C3p3wk.pdf` |
| 系统部署单位 | SAKURA Internet Research Center (日本) |
| 系统命名 | **SAKURAONE**, 基于 KOKARYOKU PHY 裸金属 GPU 平台 |
| TOP500 排名 | ISC 2025 第 49 名 (HPL Rmax = 33.95 PFLOP/s) |
| 资助 | Cross-ministerial Strategic Innovation Promotion Program (SIP), Grant JPJ012425 |

特别说明：SAKURAONE 是 ISC 2025 TOP500 中唯一进入前 100 且采用**完全开放网络栈**(800 GbE + SONiC) 的系统。这一点是论文最重要的工业意义之一 (页 1 摘要)。

---

## 1. TL;DR

- **目标**：在不依赖 NVIDIA InfiniBand / Spectrum-X 等闭源网络栈的前提下，构建一个能够媲美专用 IB 集群、可投入大规模 LLM 持续预训练 (CPT) 与微调 (SFT/LoRA) 的开放以太网 AI HPC 系统。
- **方法**：采用 **800 GbE (2× 400 GbE) RoCEv2** 作为 GPU fabric，**SONiC** (Open Compute Project 开源 NOS) + **Broadcom Tomahawk 5** ASIC 构建 rail-optimized leaf–spine 拓扑；存储平面物理隔离 (200 GbE/400 GbE) + 全闪 Lustre 2 PB；H100 SXM 800 卡 / 100 节点。
- **结果**：HPL Rmax = **33.95 PFLOP/s** (43.31 TFLOP/s 每卡, 78.3% GEMM 效率)；HPL-MxP (FP8) = **339.86 PFLOP/s**；HPCG = **396.295 TFLOP/s**；MLPerf Training v4.1 GPT-3 175B 与 Llama 2 70B LoRA 与 NVIDIA Eos (DGX H100 SuperPOD + IB) **同节点数差距 2–17%** (页 7 表 12)。
- **观察**：在 9 个月单租户 LLM 项目的运行中：
  - 73.5% GPU-time 来自用户主动 CANCELLED (调参/早停)；
  - 1–2 节点小作业占总数 76.9%，但只占 GPU-time 的 1.8%；17+ 节点大作业仅占数量 3.3%，却占 GPU-time 的 73.3%；
  - 17–32 节点的 CPT 作业 GPU 利用率中位数 98.4%，1–2 节点小作业中位数仅 23.4%/17.7%；
  - 资源使用从 CPT 阶段 (大作业) 向 fine-tuning 阶段 (中作业) 自然迁移；
  - 21 起故障中 GPU 类故障占 42.9%，多数通过节点级重启 (warm/cold reboot) 在数分钟内恢复；
  - 单端口 NIC 峰值 19–23 GB/s，部分作业出现 inter-rail 不均衡 (8 GB/s vs 18.9 GB/s)。
- **核心结论**：**开放以太网 + SONiC + RoCEv2 已能在中等规模 (800-GPU 量级) AI HPC 上提供与 IB 相当的端到端性能**，前提是对 ECN/PFC、NCCL channel striping、固件版本进行精细 cross-layer 调优。

---

## 2. 问题背景

### 2.1 AI 集群网络栈现状

当下大规模 AI HPC 互连存在三条主流路线：

1. **NVIDIA InfiniBand (闭源)**：DGX SuperPOD 默认互联，CX-7 NDR 400Gbps，附带 SHARP in-network reduction、自适应路由、credit-based flow control。NVIDIA Eos 即此路线的旗舰参考 (论文页 7 用作对比基线)。
2. **NVIDIA Spectrum-X**：以太网范式但仍是 NVIDIA 闭环 (Spectrum-4 ASIC + BlueField-3 DPU + 专用拥塞控制 RTT-CC + 自适应路由)，给 AI 集群提供"以太网外形 + 类 IB 性能"。
3. **开放 Ethernet + RoCEv2 + 开源 NOS (SONiC)**：基于 Open Compute Project 抽象 (SAI)，硬件可选 Broadcom Tomahawk / Marvell Teralynx / Cisco Silicon One，控制面常用 SONiC + FRRouting。这是 SAKURAONE 选择的路线。

论文 §2 (页 2) 强调：**自 2015 年起，OCP/SAI 推动了交换平台软硬件解耦**；超大规模数据中心 (Microsoft Azure 等) 已大规模部署 SONiC；对 AI/HPC fabric 而言，SONiC 提供了 RoCEv2 必需的 lossless 以太网构件 (PFC, ECN) 与 EVPN/VXLAN 多租户 overlay 能力。

### 2.2 超节点总线设计的工业现状 (论文背景的隐含语境)

论文虽未明确比较 NVL72 / GB200 NVLink Switch System 等"超节点总线"路线，但读者必须理解 SAKURAONE 的体系结构定位：

- **节点内 (Scale-Up)**：H100 SXM × 8，第 4 代 NVLink + NVSwitch，提供节点内 GPU-GPU 直连 (论文页 4，表 1)。这部分 SAKURAONE **完全沿用 NVIDIA 节点内 fabric**，没有自研超节点总线。
- **节点间 (Scale-Out)**：这是 SAKURAONE 真正"开放"的部分——8× ConnectX-7 400 GbE NIC + RoCEv2 + SONiC leaf–spine。
- **未涉及更激进的 Scale-Up**：如 NVL72 (72 GPU NVLink 域)、UALink (Ultra Accelerator Link)、Tenstorrent/Rebellions 自研总线，论文未触及。这意味着 SAKURAONE 是"**节点内闭源 NVLink + 节点间开放以太网**"的混合形态，是当前工业界最务实可落地的开放路径。

### 2.3 为什么需要开放以太网方案

论文 §3 Motivation (页 2) 的核心动机：

1. **供应链韧性 / 反 vendor lock-in**：解耦 NOS 与 ASIC，硬件多源采购可行。
2. **国家 AI 主权**：日本工业界长期受限于共享 HPC (ABCI 3.0 / TSUBAME 4.0)，缺乏稳定可预测的私有 LLM 训练资源；开放栈降低进入门槛。
3. **成本效率**：白盒交换 + 开源 NOS 降低 TCO。
4. **创新速度**：SONiC 容器化架构与 FRR 控制面让特性快速迭代 (页 2 引 Yuan, 2018)。
5. **生命周期透明度**：开放遥测、可观测性符合 mission-critical 治理要求。

### 2.4 单租户 LLM workload 特点

论文 §1 (页 1–2) 强调：单租户 (single-tenant) + 单项目 (single-project) 设置消除了多租户调度的混杂因子 (cross-tenant contention, heterogeneous policy)，使得作者能"干净地"观察一个 LLM 开发生命周期 (CPT → SFT → 评估) 内的工作负载演化。这是与 Microsoft Philly (Jeon et al. 2019)、Meta (Kokolis et al. 2025) 这类多租户超大规模研究的关键差异。论文也补充：**中等规模 (~100–1000 GPU) 的 LLM 工作负载公开数据极少**——这个 800-GPU 数据点本身就是贡献。

---

## 3. 核心思想 / 系统设计

### 3.1 总体五子系统架构 (页 3 §4)

SAKURAONE 由 5 个子系统组成：

1. **Compute** - 100 节点 / 800× H100。
2. **Interconnect** - rail-optimized 800 GbE leaf–spine，RoCEv2，full bisection。
3. **Storage** - 全闪 Lustre 2 PB，独立 200/400 GbE storage plane。
4. **Secure access** - VPN/FW + 交互式 front-end node。
5. **Observability** - 带外 (out-of-band) 只读遥测节点。

关键的设计原则：**training network 与 storage network 物理且逻辑分离**，避免 collective burst 与 checkpoint I/O 的相互干扰 (页 3, §4)。

### 3.2 Interconnect 需求 (页 3, §4.2)

论文系统地列出 5 条互连需求：

#### (a) Direct GPU-to-GPU Data Movement
- 必须支持 GPUDirect RDMA (NIC 直接 DMA 到 GPU 显存，绕过 host memory + CPU)。
- 节点内通过 NVLink/NVSwitch 暴露统一拓扑，使节点内带宽不会成为节点间瓶颈。

#### (b) Collective-Oriented Performance (Clos & Rail Topologies)
- 高 bisection 带宽 + 低 tail latency + 同步 burst 下稳定。
- **Rail-optimized**：将多条独立 rail 映射到不同 channel，缓解拥塞、提升吞吐。
- NCCL 等库可以**跨 rail stripe**，使用 hierarchical collective 算法。
- Transport 级 ECN 控制缓解大 incast。

#### (c) Performance Targets
- 通信时间应只占 step time 一小部分；每 GPU 有足够的 BW 与 bounded collective latency。

#### (d) Ethernet/RoCEv2 Engineering
- 要做到 lossless RDMA，需要**端到端 ECN + DCQCN** 主导拥塞控制，而不过度依赖 PFC (避免 head-of-line blocking)。
- QoS 隔离 background flow。
- 高速链路 + multirail。

#### (e) Scalability & Topology-Aware Orchestration
- Rack→pod 无需重设计；调度器与 runtime 必须看见 rail/domain 拓扑信息，把通信限制在高带宽域内。

这 5 条需求直接对应了第 5 节的实现选择 (NCCL channel striping、ECN 阈值调优、rail-optimized 物理布线)。

### 3.3 Rail-optimized Leaf–Spine 拓扑 (核心拓扑设计)

论文 §5.2 (页 4 表 3, 图 2) 定义：

| 维度 | 配置 |
| --- | --- |
| 物理介质 | Ethernet 800 GbE (实现为 2× 400 GbE) |
| 协议 | RoCEv2 |
| 拓扑 | rail-optimized leaf–spine (两 pod，每 pod 8 leaf) |
| 交换机型号 | **Edgecore AIS800-64O** (开放白盒) |
| 交换容量 | 51.2 Tb/s 全双工 |
| 软件栈 | **SONiC** |
| 交换 ASIC | **Broadcom Tomahawk 5** |
| Leaf 总数 | 16 chassis (2 pod × 8 leaf) |
| Spine 总数 | 8 chassis |
| Leaf–Spine 链路 | 800 GbE inter-switch |
| 节点上行 | 8× 400 GbE 进入同一 pod 的 leaf 集合 |

#### 拓扑的"rail" 含义

每个 H100 节点的 8 张 400 GbE NIC，对应 8 条 rail。同一 rail 编号的 NIC 在不同节点上**总是接到同一组 leaf 上**——这意味着：
- 节点间同 rail 的 collective (通常对应 tensor/data parallel 的 Allreduce 子组) 走最短路径，可能只过 1 跳 leaf；
- 跨 rail 才需要 spine 转发；
- NCCL 可以把 ring 沿 rail 切片，让多条 ring/tree 并行不抢占。

每 pod 8 leaf × 8 spine = 全连 leaf–spine。所有 leaf 都连接所有 spine (full bisection)。两个 pod 通过 spine 互联，从而保证 **uniform shortest-path connectivity between pods** (页 4 §5.2)。

> 在"超节点总线 + 网络体系结构"维度，这个拓扑就是把 NVLink 的"轨道化思想"投射到以太网域：rail 0 ~ rail 7 各自封闭，再由 spine 提供少量跨 rail 流量出口。这是当前公开的开放以太网设计中，与 Meta RoCEv2 训练网络 (Gangidi et al., SIGCOMM 2024)、MIT/Microsoft 的 Rail-only (Wang et al., HOTI 2024) 同源的工程范式。

### 3.4 节点内 GPU-NIC affinity (页 4 表 1, 表 2)

每节点 11 张 NIC，通过 `nvidia-smi topo -mp` profile：

| NIC 设备 | 用途 | GPU 亲和性 |
| --- | --- | --- |
| NIC0 (mlx5_0) | inter-node RoCEv2 | PIX → GPU0 |
| NIC1 (mlx5_1) | inter-node RoCEv2 | PIX → GPU1 |
| NIC2 (mlx5_2) | inter-node RoCEv2 | PIX → GPU2 |
| NIC4 (mlx5_4) | inter-node RoCEv2 | PIX → GPU3 |
| NIC5 (mlx5_5) | inter-node RoCEv2 | PIX → GPU4 |
| NIC6 (mlx5_6) | inter-node RoCEv2 | PIX → GPU5 |
| NIC7 (mlx5_7) | inter-node RoCEv2 | PIX → GPU6 |
| NIC9 (mlx5_11) | inter-node / management | PIX → GPU7 |
| NIC3 (mlx5_3) | secondary/reserved | NODE (GPU3 affinity) |
| NIC8 (mlx5_8) | storage I/O | NODE (GPU7 affinity) |
| NIC10 (mlx5_bond_0) | storage (bonded) | logical multi-bridge |

注意：
- **PIX** = NIC 与 GPU 共享同一 PCIe switch (最佳亲和性)；
- **NODE** = 同一 CPU socket 但跨 PCIe switch；
- 8 张 PIX-NIC 严格对应 8 个 GPU，构成 8 条 inter-node rail；
- 存储 NIC (NIC8/NIC10) 单独走存储 plane，避免与 GPU fabric 串扰；
- NIC3 作为热备 (reserved)，体现工业级冗余设计。

### 3.5 协议栈与拥塞控制设计

论文 §8.2 (页 11 表 15) 公布了真实生产环境采用的 RoCEv2 拥塞控制参数：

| 参数 | 值 |
| --- | --- |
| ECN min / max | 2 MB / 10 MB |
| ECN max marking probability | **1%** |
| PFC priority queue | 3 (DSCP-based QoS) |
| PFC Xoff threshold | 36,570,285 bytes |
| PFC Xon offset | 18,432 bytes |
| PFC headroom | 36 MB (shared, all ports) |
| Shared-buffer mode | Dynamic (alpha = 1, 66%) |

设计方法学：
- 在简化的两层 leaf–spine 上做厂商验证测试；
- 分别用 RingAllReduce 与 AlltoAll 流量模式扫 ECN min/max/marking probability；
- PFC buffer 用厂商默认值 (论文明确建议 **不要随便调 PFC buffer**)；
- ECN 阈值要按交换机 buffer 容量比例设置 — 阈值过低会让 DCQCN 提前进入 100% mark-rate 饱和，导致不必要的吞吐损失。

> 1% 的 ECN max marking probability 看似保守 (更依赖 PFC 而非 ECN 速率回退)，但作者在验证中观察到该取值给出最高吞吐——这是一个非常有价值的工业经验数据点 (页 11)。

### 3.6 软件栈 (页 4–5 §5.4)

- **OS**: Rocky Linux 9.4 (RHEL 兼容)；
- **环境管理**: module 系统 (多版本编译器/CUDA/库共存)；
- **并行模型**: MPI + OpenMP + CUDA 12.x；
- **GPU 库**: cuDNN, NCCL；
- **容器**: Singularity/Apptainer + Pyxis (Slurm 集成)；
- **调度**: Slurm 22.05.9 (优先级、保留、依赖)；
- **监控**: 与 Slurm 集成的实时遥测。

---

## 4. 实现 / 工程细节

### 4.1 计算节点 (页 4 表 1)

| 组件 | 规格 |
| --- | --- |
| 机箱 | Supermicro **SYS-821GE-TNHR** (8U, 风冷) |
| CPU | 2× **Intel Xeon Platinum 8580+** (5th Gen, 60C/120T) — 共 120 核/节点 |
| 内存 | **1.5 TB DDR5-5600** |
| GPU | 8× **NVIDIA H100 SXM** 80 GB + NVLink/NVSwitch |
| 系统盘 | 2× 372 GB SAS (mirrored) |
| 本地 scratch | 4× 7.68 TB NVMe |
| GPU fabric NIC | **8× ConnectX-7, 400 GbE** |
| Storage NIC | 2× ConnectX-7, 400 GbE (I/O plane) |
| 管理 | 1 GbE |

### 4.2 整机规模

- 节点数：100
- GPU 总数：**800× H100 SXM** (每张 80 GB HBM3)
- CPU 总核心：12,000 (100 × 120)
- 总内存：**150 TB** DDR5
- GPU fabric 总端口：100 × 8 = **800 条 400 GbE**，等效 **400 Tb/s** GPU fabric 上行总带宽
- 存储池：2 PB 全闪 Lustre

### 4.3 网络硬件 (页 4 表 3)

- Leaf 16 台 + Spine 8 台 = 24 台 Edgecore AIS800-64O
- 每台 51.2 Tb/s
- ASIC: Broadcom Tomahawk 5 (业界 51.2T 一代主力)
- 软件: SONiC (开源, 容器化 + FRR 控制面)

### 4.4 存储系统 (页 4 表 4 + 页 5 图 2)

- **DDN ES400NVX2** × 4 chassis, 全闪
- 每 chassis: active dual-controller, 24× NVMe Gen4 bay, 30.72 TB TLC SSD/槽位
- 每 controller: 200 GbE × 2
- vOSS/MDS 在每个 controller 上虚拟化 (object/metadata service 可共存)
- 每节点: 2× 400 GbE → 双 storage switch (冗余 + 负载均衡)
- 端到端目标: ~100 GB/s sustained，支撑同时 checkpoint + data generation
- 容量目标: 1 PB 预期产出 × 2 安全余量 = 2 PB

### 4.5 软件栈细节

- NCCL 通过 8 条 rail 上 stripe，hierarchical algorithm 让 intra-node TP collective 留在 NVLink，inter-node DP/PP 走 RoCEv2；
- Slurm 与遥测集成，提供 per-job GPU 利用率 / NCCL 时间分解；
- 论文使用 PyTorch Profiler 对 32-node / 64-node GPT-3 175B 训练做 NCCL kernel 级分解 (页 7 表 10)。

### 4.6 性能复现要素 (论文披露)

- HPL: HPL-NVIDIA 25.4.0, N = 2,706,432, NB = 1024, P×Q = 16×49 (784 GPU 实测)；
- HPL-MxP: HPL-MxP-NVIDIA 25.4.0, N = 2,989,056, NB = 4096, P×Q = 24×32 (768 GPU 实测), Sloppy FP8 (type=1)；
- HPCG 3.1: 4096×3584×3808 全局网格, 784 进程 × 16 线程；
- IO500: 10-node 与 96-node 两组对比；
- MLPerf Training v4.1: GPT-3 175B (32/64/96 节点) + Llama 2 70B LoRA (1/8/64/96 节点)。

---

## 5. 观测到的 workload dynamics

论文 §7 (页 8–11) 通过 9 个月 (2024-06 到 2025-03) 的日本医疗 LLM 项目运行数据，给出 7 项关键观察。CPT 阶段集中在 2024-12 到 2025-03，对 Llama-3.1-70B-instruct 与 Qwen2.5-72B-instruct 做 continued pretraining，再做 EHR→standard-code mapping 的 instruction tuning。

### Observation 1: 用户主动取消主导 GPU-time，FAILED 几乎不耗资源 (页 8)
- **CANCELLED 占 GPU-occupied time 的 73.5%**：LLM 训练中难以预先确定最佳 step 数，常见做法是设保守上限 + 实时盯 loss/eval，到收敛或饱和就 kill。
- FAILED 作业占 **数量 16.9%** 但**仅占 GPU-time 0.3%**——绝大多数失败发生在作业早期。

### Observation 2: 小作业占多数，大作业占多数资源 (页 8–9, 图 4)
- 1 节点作业 = 76.9% 数量 / 1.8% GPU-time
- ≤4 节点作业 = 86.4% 数量 / 4.6% GPU-time
- ≥17 节点作业 = 3.3% 数量 / **73.3% GPU-time**

这与 Jeon et al. (Microsoft Philly, 2019) 和 Kokolis et al. (Meta, HPCA 2025) 的多租户研究**一致**——长尾分布是生产 GPU 集群的共性，而非多租户独有现象。

### Observation 3: 大作业 GPU 利用率持续高，小作业大量低利用率时间 (页 9, 图 5)
- 17–32 节点 (CPT 主力): **median utilization 98.4%**，仅 1.1% 时间在低利用率 (<20%)；
- 3–16 节点 (FT 等): 中等利用 42.0%–92.2%；
- 1 节点 / 2 节点: median 23.4% / 17.7%，**69.2% / 75.9% 时间在低利用率** — 反映数据预处理、评测、调试等 CPU/IO 主导的开发任务。

### Observation 4: 作业运行时间长尾 (页 9, 图 6)
- 大多数作业数十分钟内结束；
- **17–32 节点的作业里 13.6% 跑超过一周**，对应 CPT 长跑。

### Observation 5: 资源使用从大尺度向中尺度迁移 (页 9–10, 图 7)
- 2025-01 中 ~2025-03 初: 17–32 节点大作业持续 (CPT 阶段)；
- 2025-02 中起: 3–16 节点中作业逐渐占据主流 (fine-tuning 阶段)。
- 这是**典型 LLM 开发生命周期模式**: 大规模 CPT → 中等规模 SFT/任务适配。

### Observation 6: 故障以 GPU 类为主，多数节点级重启即恢复 (页 10 表 13)
3 个月共 21 起故障：

| 故障类别 | 数量 | 占比 |
| --- | --- | --- |
| GPU (ECC / HW error / unresponsive) | 9 | 42.9% |
| NVLink / NVSwitch / PCIe switch | 4 | 19.0% |
| NIC / transceiver | 1 | 4.8% |
| Interconnect switch (leaf/spine) | 5 | 23.8% |
| Storage switch | 1 | 4.8% |
| Misconfiguration | 1 | 4.8% |

- 时间分布: 1 月 13 起 (burn-in), 2 月 5 起, 3 月 3 起；
- 21 起中 10 起靠节点级 warm/cold reboot 解决；
- 3 起需要厂商更换硬件 (GPU tray, NVLink module, NIC transceiver)，lead time 数天到数周；
- 一起 leaf switch 的 MAC-learning 异常表现为 **cross-rail communication degradation**——这是 rail-optimized 拓扑特有的故障形态，值得高度关注。

### Observation 7: 单端口 NIC 峰值 19–23 GB/s + inter-rail 不均衡 (页 10–11 表 14)

| 指标 | Job A (64 nodes) | Job B (32 nodes) |
| --- | --- | --- |
| 单 NIC 端口峰值 | 22.6 GB/s (8 端口均衡) | 18.9 GB/s (6 端口) + 8.0 GB/s (2 端口) |
| 每 GPU NVLink (TX+RX) | 502.0 GB/s | 114.5 GB/s |
| 每 GPU PCIe (TX+RX) | 74.5 GB/s | 17.4 GB/s |

每 400 GbE 端口名义全双工 100 GB/s，22.6 GB/s 折合**约 22.6% 的端口利用率** (60 秒计数器平均，会平滑掉亚秒级 burst)。Job B 的 rail 不均衡 (2 GPU PCIe 仅 8.5 GB/s，其它 6 GPU 约 20.3 GB/s) 跨 NIC/PCIe/NVLink 一致，提示是**全栈一致的非均衡**——可能是模型并行 mapping 的非对称性。

Pod-level switch 计数器只能给出**上下文性证据** (受同 pod 其它作业污染)。论文坦言：
- **没有采集 ECN marking rate 和 PFC pause counter**——无法直接量化拥塞贡献；
- 60 秒采样会平滑掉真正的 burst 峰值；
- 这是后续工作的一个开放点。

---

## 6. 评测

### 6.1 HPL (页 5–6, 表 5)
- N = 2,706,432, NB = 1024, P×Q = 16×49, **784 GPU**
- **Rmax = 33.95 PFLOP/s** (43.31 TFLOP/s/GPU)
- 单 GPU GEMM peak = 55.34 TFLOP/s → 每 GPU 效率 78.3%
- 运行 389.23 s，scaling 高效

### 6.2 HPCG (页 6, 表 6)
- HPCG 3.1, 784 进程 × 16 线程
- 全局网格 4096×3584×3808 (~55.9B unknowns, 1.51T nonzeros)
- 总内存 39.96 TB, observed peak BW 3.316 TB/s
- 验证后结果 **396.295 TFLOP/s** ✓

### 6.3 HPL-MxP (FP8, 页 6 表 7)
- N = 2,989,056, NB = 4096, P×Q = 24×32 (**768 GPU**)
- Sloppy FP8 (type=1)
- **Rmax = 339.86 PFLOP/s** (442.52 TFLOP/s/GPU)
- LU-only = **539.19 PFLOP/s** (702.07 TFLOP/s/GPU)
- 残差 5.01×10⁻⁵ ≪ 1.6×10¹，**PASSED**

### 6.4 IO500 (页 6–7 表 8)
对比 10 节点 vs 96 节点：

| 指标 | 10 节点 | 96 节点 |
| --- | --- | --- |
| Bandwidth score | 133.03 GiB/s | 139.80 GiB/s |
| IOPS score | 248.74 kIOPS | **327.84 kIOPS** |
| Total IO500 | 181.91 | **214.09** |

观察: 带宽侧已接近后端饱和 (10 节点 ior-easy-write 已 262.91 GiB/s)；metadata 路径 (mdtest, find) 节点数越多越好。

### 6.5 MLPerf Training v4.1 (页 6–8, 表 9–12)

#### GPT-3 175B Pretrain (Unverified)
| Scale | TT-train (min) | MFU | Tokens/s/GPU | TFLOPS/GPU |
| --- | --- | --- | --- | --- |
| 32 N (256 GPU) | 105.31 | 38.3% | 707.62 | 757.13 |
| 64 N (512 GPU) | 58.30 | 41.2% | 758 | 815 |
| 96 N (768 GPU) | 41.86 | 35.9% | 714.23 | 710.73 |

H100 SXM dense Tensor Core peak = 1979 TFLOPS (no sparsity)。32→64 节点跨 pod，spine 出现，通信占比 16.4%→19.3%，overlap 72.3%→67.2%；96 节点 TP 拓宽到 8、MFU 降到 35.9%。

#### NCCL kernel 分解 (页 7 表 10)
- SendRecv (PP, 16-stage + VP=6) **主导 NCCL 时间** (32N: 91.2%, 64N: 89.1%) — 体现 pipeline parallel 的频繁跨节点 P2P；
- ReduceScatter / AllReduce / AllGather (TP/DP) 都很小：TP 留在 NVLink。

#### Llama 2 70B LoRA (Unverified, 页 8 表 11)
| Scale | TT-train (min) |
| --- | --- |
| 1 N | 28.44 |
| 8 N | 4.79 |
| 64 N | 1.94 |
| 96 N | 1.26 |

#### 与 NVIDIA Eos (DGX H100 SuperPOD + InfiniBand) 对比 (页 8 表 12)

| Benchmark | Scale | SAKURAONE | Eos | Ratio (Ours/Eos) |
| --- | --- | --- | --- | --- |
| GPT-3 175B | 32 N | 105.31 | 96.66 | 1.09× |
| GPT-3 175B | 64 N | 58.30 | 49.80 | 1.17× |
| GPT-3 175B | 96 N | 41.86 | 33.20† | 1.26× |
| Llama 2 LoRA | 1 N | 28.44 | 27.93 | 1.02× |
| Llama 2 LoRA | 8 N | 4.79 | 4.57 | 1.05× |

†96-node Eos 数据是从 64-node 线性外推 (对 Eos 偏乐观)。

**关键判读**：
- **小规模 / LoRA 接近持平 (1.02×–1.05×)**——节点内 NVLink 主导；
- **GPT-3 175B 32/64 节点 1.09×–1.17×**——以太网+RoCEv2 vs IB 的差距在 LLM 训练真实负载下保持在 ≤17%；
- 96 节点 1.26× 部分来自 Eos 的乐观外推；
- 这是公开数据中**开放以太网栈最有竞争力的端到端 LLM 训练对比之一**。

---

## 7. 思想精读 / 启示

### 7.1 开放以太网 vs InfiniBand 的趋势

SAKURAONE 给出了一个明确的工业判据：**在 800-GPU / 中等规模 / 单租户场景下，开放 Ethernet (SONiC + Tomahawk 5 + RoCEv2) 已能在 LLM 训练上做到与 IB 集群差距 ≤17%**。这呼应了 Ultra Ethernet Consortium (UEC) 的方向——主流以太网生态正在通过 link-level retransmission、selective acknowledgement、modern congestion control (类 DCQCN/HPCC/Swift) 与 RDMA 增强吞吐与可预测性，目标就是**全面对标 IB**。论文的结果可以视为"IB 不再是不可替代"的早期证据。

### 7.2 单租户 vs 多租户

论文做了非常清晰的"setting-specific" vs "tenancy-independent"区分 (页 12):
- 单租户专享：消除排队延迟、跨租户竞争——这些不会迁移到多租户；
- **跨租户也成立的发现**：长尾资源消耗模式、CPT→FT 阶段迁移、用户主动 cancel 主导 GPU-time——这些是 LLM 工作流本身的特点。

### 7.3 Ultra Ethernet 与 NVLink72 的边界

- **节点内 (Scale-Up)**：当前 SAKURAONE 仍依赖 NVLink+NVSwitch (闭源)。NVL72 / GB200 NVLink Switch System 把 NVLink 域扩到 72 GPU，向"超节点总线"演进。开放阵营的 UALink (UALink Consortium 2024) 试图给 Scale-Up 提供一条开放替代，但生态还很早期。
- **节点间 (Scale-Out)**：**这才是 SAKURAONE 真正"开放"的领域**。论文不试图打破 NVLink，而是聚焦在 100+ 节点尺度的 fabric 开放化——这是当前最有产业落地价值的边界划分。
- **未来超大尺度 (100k+ GPU)**：NVLink 域的扩张 (Scale-Up) + Ultra Ethernet (Scale-Out) 将共存；rail-optimized 拓扑会与 NVLink72 节点内 fabric 共同协同设计。

### 7.4 单租户 LLM workload 的调度启示 (页 11–12 §8.5)

- **Checkpoint-aware preemption** (Tiresias / Themis 思路)：用大作业的 checkpoint 完成事件作为"安全打断点"，临时插入小作业、再从 checkpoint 恢复；可在不损失多日训练进度的前提下显著降低小作业等待时间。
- 集群配置不应静态——应能随项目阶段调整。
- 用户主动 cancel 高比例反映"反馈驱动"的训练实践，传统批处理 HPC 调度器不直接友好，需要更交互的 orchestration。

### 7.5 故障容忍与模块化

GPU 类故障占 42.9%，与 Meta (Kokolis 2025) 一致，但 **node-level restart 解决了 21 起中的 10 起**——这验证了"模块化服务器 + Slurm drain 机制"在中等规模下足以隔离爆炸半径。下一步是**自动化 pre-job GPU health check** (Kokolis 2025)。

---

## 8. 局限与开放问题

论文 §8.8 (页 13) 自陈：

1. **单租户单项目** — 多用户竞争下的 workload 模式可能不同，scheduling/queueing 结论不可直接推广。
2. **9 个月时间窗** — 仅覆盖 LLM training + fine-tuning，**没有覆盖 multimodal、RAG、推理 serving** 工作负载。
3. **缺乏更细遥测** — 计划补充 GPU 利用率细粒度、I/O latency、能耗，做 power-to-throughput 评估。
4. **未采集 ECN marking 与 PFC pause counter** (页 11) — 无法直接归因带宽到拥塞。
5. **60 秒采样平滑了 burst** — 真实瞬时峰值可能被低估。
6. **MLPerf 结果未官方提交** — 仅声明遵循规范，"unverified"。
7. **96-node Eos baseline 是线性外推**，对 Eos 偏乐观 — SAKURAONE 实际差距可能更小。
8. **缺乏纯推理负载评测** — 论文是开发期 (training/FT) 数据，对 LLM serving 集群的指导有限。
9. **没有覆盖更激进的 Scale-Up (NVL72/UALink)** — 节点内仍完全依赖 NVIDIA NVLink/NVSwitch。
10. **inter-rail 不均衡的根因分析不充分** — Job B 的非对称是模型 mapping 还是网络拥塞，作者未给定论。

---

## 9. 关键术语速查表

| 术语 | 解释 |
| --- | --- |
| **RDMA** | Remote Direct Memory Access — 远端直接内存访问，绕过 CPU/OS kernel，提供低延迟高带宽数据传输。 |
| **RoCEv2** | RDMA over Converged Ethernet v2 — 把 RDMA 跑在 IP/UDP over Ethernet 上，是当前以太网 RDMA 的主流形态。 |
| **DCQCN** | Data Center Quantized Congestion Notification (Zhu et al., SIGCOMM 2015) — 基于 ECN 的 RDMA 拥塞控制算法，是 RoCEv2 的事实标准。 |
| **PFC** | Priority Flow Control (IEEE 802.1Qbb) — 链路级 per-priority pause，让 Ethernet 变成 lossless；过度依赖会导致 head-of-line blocking 与 PFC storm。 |
| **ECN** | Explicit Congestion Notification — 在 IP 头中标记拥塞，让端点感知并降速；DCQCN 的核心信号。 |
| **SHARP** | Scalable Hierarchical Aggregation and Reduction Protocol (NVIDIA/Mellanox) — IB 交换机在网络内完成 reduction 聚合，加速 Allreduce；以太网域无对应。 |
| **Spectrum-X** | NVIDIA 的以太网 AI fabric 解决方案 (Spectrum-4 + BlueField-3 + 自适应路由)，提供"类 IB"的以太网体验，仍是闭源。 |
| **Ultra Ethernet (UEC)** | Ultra Ethernet Consortium — Linux Foundation 旗下，AMD/Arista/Broadcom/Cisco/Meta/Microsoft 等主导的开放规范，目标全面替代 IB 在 AI HPC 中的地位。 |
| **SONiC** | Software for Open Networking in the Cloud — 微软主导的开源 NOS，容器化 + FRR，兼容 OCP/SAI。 |
| **SAI** | Switch Abstraction Interface — OCP 提出的 ASIC 抽象层，让 NOS 与硬件解耦。 |
| **NIC (ConnectX-7)** | NVIDIA Mellanox 第 7 代网卡，单端口 400 Gb/s (NDR / 400 GbE)，支持 RoCEv2 + GPUDirect RDMA。 |
| **fat-tree / Clos** | 无阻塞或低阻塞多级拓扑，bisection 带宽随规模线性扩展，是 HPC/AI 的事实拓扑。 |
| **Rail-optimized** | 把每节点 N 张 NIC 映射到 N 条独立 rail，每条 rail 在交换层独立——降低跨 rail 路径上的争抢。 |
| **NVLink/NVSwitch** | NVIDIA 节点内 GPU-GPU 高带宽总线 (H100 SXM 第 4 代 NVLink)；NVL72 把 NVLink 域扩到 72 GPU。 |
| **GPUDirect RDMA** | NIC 直接 DMA 到 GPU 显存，免去 host memory staging。 |
| **PIX / NODE / SYS** | `nvidia-smi topo -mp` 中 GPU-NIC 邻近度等级：PIX=同 PCIe switch，NODE=同 NUMA 但跨 PCIe switch，SYS=跨 NUMA。 |
| **MFU** | Model FLOPS Utilization — 实测 FLOPS / 理论 dense peak FLOPS。H100 SXM 1979 TFLOPS 为基准。 |
| **CPT** | Continued Pre-Training — 在已有 base model 上做继续预训练。 |
| **LoRA** | Low-Rank Adaptation — 通过低秩矩阵微调，显著降低 fine-tuning 资源需求。 |
| **HPL / HPCG / HPL-MxP** | TOP500 / Green500 系列 benchmark：稠密线性 / 稀疏 + 通信受限 / 混精 AI Linpack。 |
| **IO500** | 并行存储基准 (IOR + mdtest)，几何平均得分。 |
| **DDN ES400NVX2** | DDN 全闪 NVMe 存储设备，支持 NDR200 / 400 GbE 网络接入。 |
| **Lustre** | 高性能分布式 POSIX 文件系统，HPC 主流，分 OSS/MDS。 |

---

## 10. 关键页码索引

| 主题 | 页码 |
| --- | --- |
| 摘要 + TOP500 第 49 名声明 | 1 |
| 系统总览图 (Figure 1) | 1 |
| 单租户 vs 多租户讨论 + 中等规模数据空白 | 2 |
| Background: SONiC/SAI/RoCEv2/PFC/ECN | 2 |
| Motivation: 开放网络 + 多租户 overlay (EVPN/VXLAN) | 2 |
| 系统五子系统架构 §4 | 3 |
| Interconnect 5 大需求 (rail-optimized + GPUDirect RDMA + DCQCN/ECN) | 3 |
| Storage 需求 (2PB Lustre, 100 GB/s) | 3 |
| 计算节点表 1 + GPU-NIC 表 2 (PIX 亲和) | 4 |
| 互连网络表 3 (Tomahawk 5, SONiC, AIS800-64O) | 4 |
| 存储系统表 4 (DDN ES400NVX2 ×4) | 4 |
| 系统软件 §5.4 (Rocky 9.4 / Slurm 22.05.9 / Apptainer / NCCL) | 4–5 |
| 系统详图 (Figure 2, leaf–spine + storage 双 plane) | 5 |
| HPL 表 5 (33.95 PFLOP/s, 78.3% 效率) | 6 |
| HPCG 表 6 (396.295 TFLOP/s) | 6 |
| HPL-MxP 表 7 (FP8, 339.86 PFLOP/s) | 6 |
| IO500 表 8 (10 vs 96 节点) | 7 |
| MLPerf GPT-3 175B 表 9 + PyTorch Profiler 表 10 | 7 |
| MLPerf Llama 2 LoRA 表 11 + Eos 对比表 12 | 8 |
| Observation 1–2 (cancel/long-tail) + Figure 3, 4 | 8–9 |
| Observation 3 (大作业 98.4% 利用率) Figure 5 | 9 |
| Observation 4 (long-tail runtime) Figure 6 | 9–10 |
| Observation 5 (CPT → FT 阶段迁移) Figure 7 | 9–10 |
| Observation 6 (故障表 13) | 10 |
| Observation 7 (NIC 22.6 GB/s + rail 不均衡 表 14) | 10–11 |
| RoCEv2 拥塞控制参数表 15 + 1% ECN marking | 11 |
| 调度建议 (checkpoint-aware preemption) | 11–12 |
| 单租户 vs 多租户 generalizability 讨论 | 12 |
| AI–HPC co-design 4 原则 | 12–13 |
| 局限 §8.8 | 13 |
| Related Work (DCQCN, Rail-only, HammingMesh, Meta RoCEv2) | 13 |
| Conclusion | 13 |
| 参考文献 | 13–15 |

---

## 11. 一句话点评

> **SAKURAONE 用一套 800 GbE + SONiC + RoCEv2 的完全开放以太网栈，在 800× H100 / 中等规模 / 单租户 LLM 真实生产环境下实测出与 NVIDIA Eos (DGX SuperPOD + IB) 端到端差距 ≤17% 的 LLM 训练性能，并公开了 ECN/PFC 调优值、rail-optimized leaf–spine 设计与 9 个月 workload telemetry——这是当前少有的、把"开放以太网能否替代 IB" 的工业问题回答得相当具体的论文，也为 Ultra Ethernet 时代的 AI HPC 网络设计提供了一份难得的、可复制的中等规模参考实现。**
