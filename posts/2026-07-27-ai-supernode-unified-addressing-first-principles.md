---
title: "AI 超节点统一编址的第一性原理 —— 从 Hopper 到 NVL72，再到 AI Native Memory Fabric"
date: 2026-07-27
tags: ["unified-addressing", "nvlink", "hopper", "nvl72", "memory-fabric", "moe", "fusion-operator", "location-transparency", "ai-infra", "first-principles"]
excerpt: "统一编址不是构建覆盖整个超节点的统一地址空间，而是为当前计算任务建立位置透明的访问语义。从融合算子出发，重新理解 Hopper 的三层 Fabric 架构。"
---

# AI 超节点统一编址的第一性原理

## 一个思维转折

过去讨论统一编址，我们习惯从 CUDA UVA 或 CPU 虚拟内存出发。但对于 AI 超节点，这已经偏离了问题本身。

经过对 Hopper 架构、NVL72 拓扑、以及 [DeepEP 编址设计的深度对抗分析](deep-ep/deep_dive/addressing_deep_dive.html)，我发现真正推动统一编址发展的，不是虚拟内存，而是 ==**AI Workload 的变化**==。

---

## 1. 为什么 AI 超节点需要统一编址？

### 1.1 融合算子改变了通信的本质

过去 GPU Kernel 访问本地 HBM，GPU 间通信由 NCCL、MPI 完成，**计算和通信是两个独立阶段**。

但今天的 AI 算法开始采用 ==**融合算子（Fusion Operator）**==：

| 融合算子 | 远程访问模式 |
|---|---|
| **MoE Fusion Kernel** | 直接访问不同 Expert 所在 GPU 的参数 |
| **Cross-GPU Attention** | 直接读取远端 GPU 上的 KV Cache |
| **Engram Memory / Embedding Cache** | Kernel 内部大量随机远程访问 |

**通信已经融合进计算本身**。

### 1.2 融合算子希望表达什么？

对一个融合算子而言，它真正希望表达的是：

```cpp
// 理想：位置透明
float val = load(ptr);
```

而不是：

```cpp
// 现实：显式指定位置，破坏计算语义
float val = load(gpu7, hbm2, offset);
```

更不希望退化成：

```
Fusion Kernel → Suspend → CPU Runtime → Address Lookup → Network → Resume Kernel
```

因为这意味着 GPU 必须退出执行流，由 CPU 完成地址解析。这不仅增加延迟，更==**破坏 GPU 流水线**==，使远程 Load 无法像本地 Load 一样连续发射。

### 1.3 第一性原理：Location Transparency

因此，统一编址真正解决的问题不是"统一地址"，而是：

> ==**让 GPU 能够像访问本地 HBM 一样访问远端 Memory。**==

换句话说，它提供的是 **Location Transparency（位置透明）**。Kernel 不应该关心数据在哪张 GPU、属于哪个 HBM Stack、是否经过几个 NVSwitch。

一个统一编址系统，本质上是在 GPU 和 Memory 之间增加了一层 ==**Location Service**==：

```
   AI Object
      │
  Location Service
      │
 Physical Location
      │
   Transport
      │
 Remote Memory
```

这里对应三个彼此独立的问题：

| 问题 | 职责 | Hopper 对应 |
|---|---|---|
| **Location** | 对象在哪里？ | Control Fabric |
| **Translation** | 逻辑地址 → 可访问位置 | Memory Fabric |
| **Transport** | 如何把请求送过去？ | Transport Fabric |

---

## 2. Hopper 的三层 Fabric 架构

理解第一性原理后，再看 Hopper，就会发现 NVIDIA 并没有把统一编址放到 NVLink，而是划分成三个职责完全不同的层次：

```
              CUDA Runtime
                   │
──────────────────────────────────────
        Control Fabric
──────────────────────────────────────
Fabric Manager / NVLSM / Driver
──────────────────────────────────────
        Memory Fabric
──────────────────────────────────────
GMMU / TLB / Page Table /
Memory Aperture / Peer Mapping
──────────────────────────────────────
       Transport Fabric
──────────────────────────────────────
NVLink / NVSwitch / HBM
```

### 2.1 Control Fabric：管理拓扑，不管理内存页

Control Fabric 负责 GPU Discovery、Topology Discovery、Partition、Routing Programming、Peer Registration。Fabric Manager 和 NVLSM 维护的是整个 NVL72 的**资源拓扑**——它们知道 GPU 在哪里、Switch 如何连接，但并不知道 Tensor、KV Cache 或 Expert。

### 2.2 Memory Fabric：建立内存语义，不负责控制

当 GPU Kernel 执行 `load(ptr)` 时，请求进入 GMMU，通过 TLB 和 Page Table 找到 PTE。PTE 中已记录目标 Peer GPU 和 HBM Offset。==**Memory Fabric 本身没有控制逻辑，它只是 Control Fabric 在 GPU 上留下的执行结果。**==

### 2.3 Transport Fabric：纯粹的数据平面

NVLink 和 NVSwitch 看到的只是 `(Destination GPU, HBM Offset, Payload)`，它们并不知道 Virtual Address。==**NVLink 的职责始终是 Transport，不是 Address Translation。**==

### 2.4 完整数据路径

```
  Control Fabric
        │
  建立 Peer Mapping
        │
  Memory Fabric
  (GMMU / Aperture)
        │
  解析目标 GPU
        │
  Transport Fabric
     (NVLink)
```

---

## 3. Hopper Server 与 NVL72：两个尺度的区别

| 维度 | Hopper Server (8 GPU) | NVL72 (72+ GPU) |
|---|---|---|
| **Control Scope** | 单机 | 机柜 |
| **Control Plane** | CUDA Driver | Fabric Manager + NVLSM |
| **Memory Fabric** | 单机 Global Address | 机柜 Global Address |
| **Peer Mapping** | 8 个 Peer | 72+ 个 Peer |
| **TLB 压力** | 低 | 高 |
| **Partition** | 不支持 | 支持（MIG / Multi-tenant） |

在 Hopper Server 中，8 GPU 的 Peer Mapping 规模可控，一次性建立 Global Address Space 是合理选择。

但 NVL72 是一个 ==**GPU Fabric**==，不是服务器。Control Fabric 需要管理整个机柜的多级拓扑、多个 OS Domain、Partition、故障恢复。

两者采用相同的编址策略：

```
     Global Fabric
          │
  Global Memory Fabric
          │
 Global Address Space
```

Control Fabric 和 Memory Fabric 的作用域都覆盖整个 NVL72，保持一致。

---

## 4. 为什么 Hopper 采用全局编址？

这来自 HPC 的基本假设：

> 传统 HPC 中，任何 MPI Rank 都可能与任何 Rank 通信，因此最简单的软件模型就是一次性建立整个 Fabric 的 Global Address Space。

这是一种典型的 ==**用初始化成本换运行时效率**== 的设计。Kernel 永远面对统一 Pointer，运行过程中不需要修改 Page Table。

对于几十张 GPU 的静态集群，这非常合理。

---

## 5. AI Workload 改变了这一假设

AI 工作负载不再遵循"所有 GPU 都可能通信"：

- 一个 Attention Kernel 可能只涉及 GPU4~GPU7
- 一个 MoE Kernel 可能只访问几个 Expert 所在 GPU
- 一个推理实例甚至只使用 NVL72 的一小部分

==**AI 的基本通信单位从整个 Fabric 变成了 Communication Domain。**==

### 5.1 类比 RDMA

RDMA 并不是天然拥有远端地址。一次 RDMA Read 能直接执行，前提是控制平面已完成 Memory Region 注册、QP 建立、地址和权限交换。

NVLink 的统一编址同理：GPU 能执行 `load(ptr)`，不是因为 NVLink 天然知道远端地址，而是因为系统已提前建立好 ==**Address Resolution Service**==。

### 5.2 真正必要的是什么？

真正必要的不是一个覆盖整个 NVL72 的 Global Address Space，而是：

> ==**在开始通信之前，为当前计算任务建立地址透明化能力。**==

Global Address Space 只是实现 Address Transparency 的一种方式，而且是一种"奢侈"的方式。

地址透明化可以在不同时刻建立：

| 建立时机 | 作用域 | 适用场景 | Hopper 支持 |
|---|---|---|---|
| 系统启动时 | 整个 NVL72 | HPC 全对全 | ✓（当前方案） |
| NCCL 初始化 | Communicator | 集合通信 | 部分 |
| 推理实例启动 | 实例级 Domain | 推理服务 | ✗ |
| 融合算子启动 | Communication Domain | MoE Fusion | ✗ |

> ==**真正不可缺少的是 Address Resolution Service，不是 Global Address Space。**==

---

## 6. Hopper 今天的局限

从这个角度看，Hopper 真正的限制不在 NVLink，也不在 NVSwitch，而在 ==**Memory Fabric 的作用域**==。

今天，Control Fabric 管理整个 NVL72，Memory Fabric 也覆盖整个 NVL72。即使一个 Fusion Kernel 只访问 4 张 GPU，每张 GPU 的 GMMU 仍然维护整个机柜的 Peer Mapping。

==**Fabric Scope 与 Address Scope 被绑定在了一起。**==

| 成本项 | 当前状态 | 问题 |
|---|---|---|
| Page Table | 覆盖整个 NVL72 | 与实际通信域不匹配 |
| TLB Miss | 全局映射导致压力大 | 远程访问延迟增加 |
| Peer Mapping | 系统启动时一次性建立 | 无法按需释放 |
| Fabric Manager | 维护全局视图 | 扩展性瓶颈 |

---

## 7. Next：Memory Fabric 从静态资源走向动态服务

未来不需要推翻 Hopper 的三层架构。Control / Memory / Transport 的职责划分仍然合理，真正需要变化的是 ==**Memory Fabric 的生命周期**==。

```
   Communication Domain
            │
            ▼
  Address Resolution Service
       (Control Plane)
            │
            ▼
    Address Domain
  (Page Table / Peer Mapping)
            │
            ▼
     Memory Fabric
  (GPU 可直接 load/store)
            │
            ▼
    Transport Fabric
  (NVLink / UALink / UB)
```

这里真正动态的是 **Address Resolution Service**。Memory Fabric 只是地址解析关系在 GPU GMMU 和页表中的具体体现。

### 演进路径

| 阶段 | 编址对象 | 映射关系 | 服务对象 |
|---|---|---|---|
| **CPU 时代** | Virtual Page | VA → PA | 进程 |
| **Hopper 时代** | Virtual Address | VA → GPU + HBM Offset | 整个 Fabric |
| **AI Native 时代** | Communication Domain | Domain → Address Domain | 计算任务 |
| **未来** | AI Object | Object → Location | Fusion Kernel |

未来，统一编址可能从 `VA → GPU + Offset` 提升到 `AI Object → GPU + Offset`，围绕 KV Block、Expert、Embedding 建立 ==**Object Directory Service**==。届时，统一编址将不再只是内存管理机制，而会成为 AI Native Memory Fabric 的核心基础设施。

---

## 8. 总结：统一编址不是统一地址，而是统一位置

> ==**统一编址并不是构建一个覆盖整个超节点的统一地址空间，而是为当前计算任务建立位置透明的访问语义，使 GPU 能够像访问本地 HBM 一样访问远端内存。**==

三个核心结论：

1. **第一性原理**：统一编址的第一性原理是 ==**Location Transparency**==，不是 Unified Address。目标是让 Kernel 用 `load(ptr)` 表达远端访问。

2. **Hopper 的合理与局限**：三层 Fabric 架构（Control / Memory / Transport）是合理的，但 ==**Fabric Scope 与 Address Scope 的绑定**== 是历史遗留问题。

3. **动态化是方向**：真正不可缺少的是 ==**Address Resolution Service**==，不是 Global Address Space。建立时机、作用域、维护者都是可自由权衡的设计参数。

Hopper 选择在系统启动阶段建立覆盖整个 NVL72 的 Global Memory Fabric，是面向 HPC 的优秀工程实现；但它并不是唯一实现。对于未来 AI Native Supernode，更合理的方向是在 Runtime 驱动下，根据 Communication Domain 动态建立 Address Resolution Service，将统一编址从"静态的机柜级资源"演进为"按需构建的运行时能力"。

---

## 延伸阅读

- [DeepEP 编址服务深度对抗分析](deep-ep/deep_dive/addressing_deep_dive.html) —— 四 Agent 对抗讨论实录
- [DeepEP 综合分析报告](deep-ep/DeepEP_Final_Analysis_Report.html) —— 三视角完整分析
- [HBM / CXL / Memory 市场调研](../hbm-cxl/report.html) —— 内存层级全景
