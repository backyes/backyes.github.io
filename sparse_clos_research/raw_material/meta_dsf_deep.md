# Meta DSF + Llama4 网络深化调研（2026-07-12）

## 来源
1. Meta DSF 工程博客：https://engineering.fb.com/2025/10/20/data-center-engineering/disaggregated-scheduled-fabric-scaling-metas-ai-journey/
2. Llama4 arXiv 论文：https://arxiv.org/abs/2510.20171

---

## 一、DSF（Disaggregated Scheduled Fabric）核心设计

### 1.1 动机：传统 IP fabric 的三大问题
"We encountered these three types of problems: ... elephant flows ... low entropy traffic patterns ... hash collisions and sub-optimal load distribution"

已尝试的方案均不可行：
- BGP pinning（steady state 可但 failover 回退到 ECMP）
- 感知负载的 ECMP（难 tuning 且产生乱序 packets——对 RDMA 有害）
- 集中式流量工程（规模增大后太复杂，且响应故障慢）

### 1.2 核心创新：双域架构（Two-Domain Architecture）
> "The fundamental innovation of DSF lies in its two-domain architecture, which separates the network into the Ethernet domain, where servers and traditional networking protocols operate, and the 'fabric' domain, where packets will be broken into cells, sprayed across the fabric, and subsequently reassembled at the hardware before being delivered back to the Ethernet domain."

**Ethernet 域**：服务器 + 传统网络协议
**Fabric 域**：packet 被切分为 cell，spray 到整个 fabric，硬件端重组后再送回 Ethernet 域

### 1.3 组件
| 组件 | 名称 | 角色 |
|------|------|------|
| RDSW | Rack Disaggregated Switch (Interface Node/IN) | 面向外部网络，处理路由/L3 |
| FDSW | Fabric Disaggregated Switch (Fabric Node/FN) | 内部高速交换，无 L3 路由 |
| SDSW | Super DSF Spine（推测） | 跨 DSF 集群互联 |

> "To the external network infrastructure, this distributed collection of INs and FNs appears as a single, unified switch, with the total number of external ports equivalent to the aggregate of all external ports across all INs, effectively creating a virtual chassis switch that scales far beyond the physical limitations of traditional designs."

### 1.4 流量管理机制
**Packet Spraying**（非 ECMP 哈希）：
> "Unlike conventional Ethernet fabrics that rely on hash-based approaches, DSF utilizes packet spraying that distributes traffic across all available paths through the fabric. Such a feature is enabled by the hardware's ability to reassemble packet cells at the interface nodes."

**Credit-based 拥塞控制**：
> "orchestrated through a credit-based allocation scheme where ingress INs dynamically request credit tokens from egress INs, allowing the system to make real-time decisions based on current path availability, congestion levels, and bandwidth utilization."

**VOQ（Virtual Output Queuing）**确保 lossless 交付。

### 1.5 Input Balanced Mode（关键创新）
> "Input Balanced Mode is a critical feature that supports balanced traffic throughout the network in the face of remote link failures. The feature avoids severe congestion on the fabric and spine layer of the DSF network."

**目标**：确保任何 DSF 设备的输入带宽 ≤ 输出带宽，**零过订阅**（no oversubscription），即使链路故障也不超卖。

**故障传播机制**（三层递归）：
1. **RDSW↔FDSW 链路故障**：RDSW 停广播可达性 → FDSW 停广播 → SDSW 收到降级信息 → 随机选择上游链路进行 Input Balanced Mode 剪枝
2. **FDSW↔SDSW 链路故障**：双向传播 — FDSW 侧降输入（RDSW 端）、SDSW 侧降输入（其他集群 FDSW 端）
3. **示例量化**：单链路故障下，受影响 RDSW 保留 50% 容量转发跨集群流量

### 1.6 DSF 的层次化扩展
- **DSF Fabric（单级）**：RDSW + FDSW
- **DSF Dual-Stage Fabric**：双级 DSF fabric
- **DSF Region**：多组 DSF 跨域互联  
- **L3 Super Spine**：iBGP + eBGP 连接 DSF L2 zones，交换聚合路由

> "Given that L3 spine is used, some of the problems, including entropy and fat flow, tend to reappear; however, at this network tier where there's much less traffic, those problems are less profound."

### 1.7 与 Sparse Clos 的关系
DSF 不是"稀疏 Clos"，但在两个维度上与 Sparse Clos 精神相通：
1. **解聚**打破单体 chassis 的物理极限——如同多级 Clos 用小芯片堆出大交换
2. **免 ECMP 的 spraying 替代哈希静态路由**——与 Sparse Clos 需动态负载均衡的配套刚需一致

---

## 二、Llama4 网络架构

### 2.1 三层 Clos 架构
> "The network within a DC adopts a 3-layer Clos architecture. Each DC is partitioned into multiple AI Zones. The Rack Training Switch (RTSW) connects GPUs within a rack, while the Cluster Training Switches (CTSW) connect all racks within an AI Zone. Aggregator Training Switches (ATSW) connect CTSWs across the DC."

| 层 | 交换机 | 负责范围 |
|----|--------|---------|
| L1 | RTSW（Rack Training Switch） | 机架内 GPU 互联 |
| L2 | CTSW（Cluster Training Switch） | AI Zone 内所有机架 |
| L3 | ATSW（Aggregator Training Switch） | 跨 AI Zone（跨 DC） |

### 2.2 过订阅比演进
> "Compared to the Llama 3 network, we reduced the cross-AI-Zone over-subscription ratio from **1:7 to 1:2.8** to provide higher bandwidth and better support multi-dimensional parallelism training at larger scale."

**关键判断**：Llama3→Llama4 不是"更稀疏"而是"更密集"。随着模型和集群规模增长，跨区带宽成为多维并行（TP×PP×DP×EP）的瓶颈，Meta 选择收紧过订阅比。

### 2.3 跨 DC mesh
> "To interconnect multiple DCs, we use a fully connected mesh between the ATSW layers of different DC buildings ... Inter-DC traffic experiences the same over-subscription ratio as cross-AI-Zone traffic (1:2.8). This architecture is extensible and can scale to hundreds of thousands of GPUs."

### 2.4 拓扑感知优化
**Job placement**：调度器分配连续 rank 到网络距离最近的节点，<code>users can specify constraints ... at different network-topology levels (e.g., rack, AI zone, and DC)</code>

**集合通信**：对过订阅网络，farthest-first 递归倍增 all-gather 被确认为最优策略（实证击败 nearest-first 和 hybrid）

### 2.5 故障容错与拥塞控制
DQPLB（Dynamic Queue-Pair Load Balancing）+ VOQ（Virtual Output Queuing）调参：
> "Together with network load balancing improvements ... and spine switch Virtual Output Queuing (VOQ) tuning, compared to Llama3 training, we reduce switch buffer build-up by an order of magnitude in the RoCE network used in Llama4 training."

跨区（FTAR/Farther-Than-A-Rack）通信受高过订阅比和有限二分带宽约束，强制使用 Ring 算法限制并发流量。
