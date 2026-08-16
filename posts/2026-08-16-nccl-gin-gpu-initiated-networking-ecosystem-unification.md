---
title: "Deepdive GPU Drive Communication: From GPUDirect RDMA to GDAKI"
date: 2026-08-16
tags: ["GPUDirect", "GDAKI", "GIN", "NVSHMEM", "NCCL", "RDMA", "IBGDA", "GPUNetIO", "DOCA", "DeepEP", "MoE"]
excerpt: "GPU-driven networking evolved through three generations: GPUDirect RDMA (CPU-controlled), GPUDirect Async (GPU-triggered, CPU-configured), GDAKI (GPU full-control). GIN's plugin architecture aims to build a GPU-centric communication ecosystem with multi-vendor backends. Deep dive into the 5 fundamental challenges GIN solves: data transfer, ordering, consistency, network parallelism, and memory flexibility."
---

# Deepdive GPU Drive Communication: From GPUDirect RDMA to GDAKI

## Thesis

**The three generations of GPU-driven networking answer one question: who owns the Queue Pair lifecycle?** GPUDirect RDMA gave QP control to the CPU. GPUDirect Async let the GPU trigger pre-configured operations. GDAKI (GPU-Driven Async Kernel-Initiated) hands full QP ownership to GPU kernels. ==The physical layer has converged; the battle is now at the runtime abstraction layer.==

> "GIN builds on a three-layer architecture: host-side APIs for device communicator setup, device-side APIs for remote memory operations callable from CUDA kernels, and a network plugin architecture with dual semantics (GDAKI and Proxy)."
> — [GPU-Initiated Networking for NCCL, arXiv:2511.15076](https://arxiv.org/abs/2511.15076)

---

## 1. Three Generations: The QP Control Transfer

| Generation | Technology | QP Creator | Operation Trigger | CPU in Hot Path | Year |
|------------|-----------|------------|-------------------|-----------------|------|
| **Gen 1** | GPUDirect RDMA | CPU (`ibv_create_qp`) | CPU (`ibv_post_send`) | Every operation | ~2013 |
| **Gen 2** | GPUDirect Async (IBGDA) | CPU (pre-configured) | GPU (doorbell MMIO) | Setup only | ~2016 |
| **Gen 3** | GDAKI / GPUNetIO | GPU kernel | GPU kernel | None | ~2024 |

The critical metric is **per-operation control overhead**:

```
Gen 1 path:  GPU → CPU syscall → kernel transition → MMIO doorbell → NIC    (~1-2μs)
Gen 2 path:  GPU → MMIO doorbell → NIC                                       (~300ns, but QP static)
Gen 3 path:  GPU → create QP + MMIO doorbell → NIC                           (~100-300ns, fully dynamic)
```

As the [GIN paper](https://arxiv.org/abs/2511.15076) states: *"GPUDirect Async introduced partial control-path offload: GPU threads trigger pre-configured network operations writing to NIC doorbell registers that are memory-mapped into the GPU address space. However, the CPU must pre-construct communication descriptors, limiting operations to those pre-configured by the host and preventing fully autonomous device-driven networking."*

**Gen 2's fundamental limitation**: QP is static — fixed count, fixed configuration, no runtime adaptation, no GPU-side error recovery.

---

## 2. Concept Hierarchy

```
GPUDirect (NVIDIA umbrella brand)
├── GPUDirect RDMA     — NIC DMA directly to GPU HBM (zero-copy data path)
├── GPUDirect Async    — GPU triggers pre-configured RDMA via doorbell
│   └── IBGDA          — InfiniBand-specific implementation
├── GPUDirect Storage  — SSD DMA directly to GPU HBM (GDS)
└── GDAKI              — GPU full control of RDMA lifecycle
    ├── GIN (NCCL)     — NCCL 2.28+ public API name
    └── GPUNetIO (DOCA)— DOCA SDK API name

Software Ecosystem
├── NCCL GIN    — Collective communication + GIN device API (3 modes: LSA, Multimem, GIN)
├── NVSHMEM     — PGAS one-sided put/get (≤3.6: IBGDA, 3.7.0+: GPUNetIO)
└── DeepEP      — MoE communication library (V1: NVSHMEM, V2: NCCL GIN)
```

### GIN's Three Device API Modes (NCCL 2.28+)

Per the [arXiv paper](https://arxiv.org/abs/2511.15076), NCCL 2.28 Device API supports three distinct modes:

| Mode | Interconnect | Mechanism | Use Case |
|------|-------------|-----------|----------|
| **LSA** (Load/Store Accessible) | NVLink / PCIe | Direct memory load/store | Intra-node P2P |
| **Multimem** | NVLink SHARP | Hardware multicast | Intra-node broadcast |
| **GIN** | InfiniBand / RoCE | GPU-initiated RDMA | Inter-node communication |

---

## 3. NVSHMEM vs GIN: Convergent Bottom, Divergent Top

### 3.1 NVSHMEM ≤3.6: IBGDA Backend

- CPU creates QP at init time; GPU triggers via doorbell
- `nvshmemi_ibgda_put_nbi` — GPU kernel issues RDMA put
- Limitation: QP count static, error handling requires CPU

### 3.2 NVSHMEM 3.7.0+: GPUNetIO Backend

- GPU kernel creates/destroys QP dynamically via DOCA GPUNetIO
- Full lifecycle control: `gpunetio_create_queue` → `gpunetio_workq_complete`
- CPU completely off the hot path

### 3.3 NCCL GIN: Dual Backend

The [GIN paper](https://arxiv.org/abs/2511.15076) reveals a critical design insight: **two interchangeable backends** for different hardware:

| Backend | Hardware Requirement | Mechanism |
|---------|---------------------|-----------|
| **GDAKI** | ConnectX-6 Dx+ / BlueField (DOCA GPUNetIO capable) | GPU directly programs NIC via device verbs |
| **Proxy** | Any RDMA-capable NIC | GPU enqueues 64-byte descriptors to lock-free queue; CPU proxy thread executes |

> "The GDAKI backend leverages DOCA GPUNetIO for direct GPU-to-NIC communication, while the Proxy backend provides equivalent functionality via lock-free GPU-to-CPU queues over standard RDMA networks."
> — [arXiv:2511.15076](https://arxiv.org/abs/2511.15076)

This dual-backend design means **GIN is not limited to latest hardware** — it degrades gracefully on older NICs via Proxy.

### 3.4 Convergence Point

Both NVSHMEM 3.7.0+ GPUNetIO and NCCL GIN GDAKI backend converge on the same mechanism:

```
GPU kernel → DOCA GPUNetIO device verbs → MMIO doorbell → NIC hardware
```

The hardware requirement is identical: NIC must expose memory-mapped queue pairs and doorbell registers via PCIe BAR (ConnectX-6 Dx and later).

---

## 4. GIN's Ecosystem Play: Plugin Architecture for Multi-Vendor Expansion

GIN 的核心战略不只是提供一套 GPU 发起通信的 API，而是**构建一个以 GPU 为中心的通信系统生态**。这一战略最清晰地体现在其三层软件分层设计和可扩展的网络插件架构中。

### 4.1 三层架构：关注点分离

GIN 的三层架构（NCCL Core → Device GIN API → GIN Network Plugin）每一层都有明确的职责边界：

| 层 | 职责 | 关键设计 |
|---|---|---|
| **NCCL Core** (Host) | 通信子初始化、内存窗口注册、资源分配 | 扩展现有 NCCL 基础设施，不破坏兼容性 |
| **Device GIN API** (GPU) | GPU 内核可调用的 put/signal 原语 | 统一接口，后端透明切换（GDAKI 或 Proxy） |
| **GIN Network Plugin** | 远程数据移动的具体实现 | 双语义（GDAKI + Proxy），支持厂商自定义 |

> "GIN builds on a three-layer architecture: i) NCCL Core host-side APIs for device communicator setup and collective memory window registration; ii) Device-side APIs for remote memory operations callable from CUDA kernels; and iii) A network plugin architecture with dual semantics (GPUDirect Async Kernel-Initiated and Proxy) for broad hardware support."
> — [arXiv:2511.15076](https://arxiv.org/abs/2511.15076)

### 4.2 双后端 → 多厂商扩展

GIN 的网络插件架构从设计之初就为**多厂商参与**预留了接口：

| 后端类型 | 控制路径归属 | 硬件要求 | 当前支持 |
|---|---|---|---|
| **GDAKI** | 插件全权拥有（createContext + device code） | ConnectX-6 Dx+ / DOCA GPUNetIO | NVIDIA InfiniBand |
| **Proxy** | NCCL Core 拥有控制结构，插件仅提供 CPU 数据路径 | 任意 RDMA 网卡 | NVIDIA InfiniBand |
| **External Plugin** | 通过 `libnccl-net.so` 动态加载 | 厂商自定义 | 开放给第三方厂商 |

> "NCCL's InfiniBand transport implements both, while external vendors may supply their own. Under the Proxy interface, NCCL Core owns control structures, device-side queuing logic, and device API implementations, while plugins provide only CPU-based put, signal, test, and regMr operations, enabling networks without GPU-direct capabilities and lowering the barrier to GIN adoption."
> — [arXiv:2511.15076](https://arxiv.org/abs/2511.15076)

关键机制是 `NCCL_NET_PLUGIN` 环境变量——外部插件在运行时动态加载，无需修改 NCCL 核心代码。这意味着：

- **AWS EFA**（Elastic Fabric Adapter）可以通过实现 Proxy 语义的 `libnccl-net.so` 插件接入 GIN 生态
- **其他网卡厂商**（Marvell/Cisco/Mellanox 等）只需提供 CPU 数据路径的 4 个基础操作（put/signal/test/regMr），即可获得完整的 GIN Device API 能力
- **不要求 GPU 直通能力**——Proxy 后端让不具备 DOCA GPUNetIO 的网卡也能参与生态

### 4.3 生态目标：以 GPU 为中心的统一通信层

GIN 的生态逻辑可以概括为：

```
               ┌─────────────────────────────────┐
               │     Application Layer           │
               │  (PyTorch, vLLM, SGLang, TRT)   │
               └──────────────┬──────────────────┘
                              │
               ┌──────────────▼──────────────────┐
               │     NCCL Device GIN API          │  ← 统一编程接口
               │  (put / signal / flush / ...)    │
               └──────────────┬──────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼─────┐        ┌────▼─────┐         ┌────▼─────┐
   │  GDAKI   │        │  Proxy   │         │ External │
   │ (NVIDIA  │        │ (通用    │         │ (AWS EFA,│
   │  IB/RoCE)│        │  RDMA)   │         │  其他)   │
   └──────────┘        └──────────┘         └──────────┘
```

> "By providing both hardware-direct and CPU-assisted plugin interfaces, GIN offers functionality across diverse deployment scenarios while maintaining compatibility with NCCL's existing ecosystem."
> — [arXiv:2511.15076](https://arxiv.org/abs/2511.15076)

GIN 的生态赌注是：**当所有网卡厂商都通过插件接入时，NCCL GIN 就成为 GPU 通信的"USB 接口"——统一、即插即用、硬件无关**。这与 NVSHMEM 的路径截然不同：NVSHMEM 是单一厂商（NVIDIA）的垂直整合方案，而 GIN 试图构建一个水平分层的多厂商生态。

---

## 5. What Problems Does GIN Solve? The Five Fundamental Challenges

GIN 的设计本质上是在回答一个问题：**GPU 直接发起网络通信，需要解决哪些基础问题？**论文将 GIN 的设计元素归纳为 5 个核心功能，它们分别对应单边通信中 3 个根本性挑战 + 2 个编程易用性需求。

### 5.1 三个根本性挑战

在任何单边通信系统（GIN、NVSHMEM、OpenSHMEM）中，必须解决三个基础问题：

| # | 挑战 | 核心问题 | GIN 的解决 |
|---|------|---------|-----------|
| **1** | **Data Transfer** | GPU 如何远程写入/读取数据？ | One-Sided Semantics（put / put with signal） |
| **2** | **Ordering** | 对方按什么顺序看到数据更新？ | Ordering Semantics（signal 保证前置 put 的可见顺序） |
| **3** | **Consistency** | 什么时候对方能看到数据？ | Asynchronous Completion Tracking（Signal/Counter 远程可见性通知） |

> "These components enable device-initiated one-sided communication through several key design elements: one-sided semantics for unilateral data movement, symmetric memory windows for zero-copy remote access, and asynchronous completion tracking with flexible ordering semantics."
> — [arXiv:2511.15076](https://arxiv.org/abs/2511.15076)

### 5.2 两个编程易用性需求

在解决三个基础问题之上，GIN 还需要让程序员**容易地写出正确的通信代码**：

| # | 需求 | 核心问题 | GIN 的解决 |
|---|------|---------|-----------|
| **4** | **Network Parallelism** | 如何充分利用多网卡/多端口/多 QP？ | GIN Contexts（每个 context 抽象一个 GPU↔NIC 通道） |
| **5** | **Memory Flexibility** | 是否必须对称内存？ | Windows-based (A)Symmetric Memory（支持非对称容量） |

> "GIN windows are designed to support asymmetry in capacity: each rank may register different buffer sizes. This flexibility proves essential for disaggregated serving architectures where prefill ranks require larger buffers than decode ranks."
> — [arXiv:2511.15076](https://arxiv.org/abs/2511.15076)

### 5.3 科普：为什么 Ordering 和 Consistency 是单边通信的基石？

用一个简单的例子说明这两个概念为什么重要。

**场景**：GPU A 需要向 GPU B 发送一组数据，并告诉 B"数据准备好了"。

```
GPU A (发送方)                    GPU B (接收方)
─────────────                    ─────────────
put(data[0] → B)  ──────→  (网络中...)
put(data[1] → B)  ──────→  (网络中...)
signal(B, ready)  ──────→  (网络中...)
                                 ...
                            B 如何知道数据真的到了？
```

**问题 1 — Ordering（顺序性）**：

如果 signal 先于 data 到达 B（网络乱序），B 会认为"数据准备好了"，但实际上 data[0]、data[1] 还在路上。B 读到的是**旧数据**。

GIN 的解决方案：
> "When a signal operation completes at the destination, it guarantees that all preceding put operations to that peer on the same context have completed and are visible to remote GPU threads."

即 signal 是一个**顺序锚点**：signal 可见 = 之前所有 put 都可见。程序员只需在最后一次 put 后 attach 一个 signal，然后 waitSignal，就能保证整批数据的顺序可见性。

**问题 2 — Consistency（一致性）**：

即使数据到达了 B 的 HBM，B 的 GPU 内核也不一定能立即看到——因为 GPU 有 write-back cache 和 relaxed memory model。

GIN 区分两种完成状态：
- **Local completion**（`flush`）：源端缓冲区可以安全重用（数据已离开发送方）
- **Remote completion**（`waitSignal`）：数据已到达目的地且对远程 GPU 线程可见

> "The flush operation ensures only local completion—all pending operations have been consumed and source buffers can be safely reused—but makes no guarantees about remote visibility."

**一句话总结**：Ordering 解决"**按什么顺序看到**"，Consistency 解决"**什么时候能看到**"。两者缺一不可——没有 Ordering，数据可能乱序到达；没有 Consistency，数据到了也不一定能读到。GIN 通过 signal 机制将两者统一：signal 既保证顺序（之前所有 put 完成），又保证可见（对远程 GPU 线程可见）。

### 5.4 GIN vs NVSHMEM：相同的问题，不同的解法

值得注意的是，NVSHMEM 作为成熟的 PGAS 库，同样需要解决上述 5 个问题。两者的对比：

| 设计元素 | GIN | NVSHMEM |
|---|---|---|
| **Data Transfer** | `put(team, peer, win, off, ...)` | `put_nbi(dst_ptr, src_ptr, count, pe)` |
| **Ordering** | Signal-based（ID 地址，轻量） | `fence` / `quiet`（QP 级别） |
| **Consistency** | Counter（本地）+ Signal（远程） | `quiet()` per QP |
| **Network Parallelism** | GIN Context（显式多通道） | QP per PE |
| **Memory Model** | Window-based（支持非对称容量） | Symmetric Heap（全局地址空间） |

GIN 的 signal-based ordering 相比 NVSHMEM 的 fence/quiet 更轻量——它只保证"同一 context 内、同一 peer 的 put-signal 顺序"，而不是全局顺序。这种选择性保证换来了更高的网络效率。

---

## 6. Runtime Ecosystem: Where the Real Difference Lies

Since the physical layer is identical, the differentiation is purely at the runtime semantic level.

### NCCL: Collective-First

- **API**: AllReduce, AllGather, ReduceScatter, Broadcast + device-side put/signal primitives
- **Topology**: Deep NVLink/IB hierarchy awareness (rail/plane/world)
- **In-network compute**: SHARP hardware reduction inside IB switches
- **Memory model**: Collective symmetric memory via `ncclCommWindowRegister`
- **Elasticity**: Dynamic rank join/leaving for production fault tolerance

### NVSHMEM: PGAS-First

- **API**: One-sided put/get/atomics (OpenSHMEM compatible)
- **Memory model**: Symmetric heap (`nvshmem_alloc`) — all ranks see same global address space
- **HPC compatibility**: OpenSHMEM standard API — existing HPC codes port with minimal changes
- **Transport**: IBGDA (≤3.6) or GPUNetIO (3.7.0+) for inter-node, symmetric memory for intra-node

### Key Semantic Difference

| Property | NCCL | NVSHMEM |
|----------|------|---------|
| Communication pattern | Collective (all ranks participate) | One-sided (initiator-only) |
| Synchronization | Implicit (collective barrier) | Explicit (fence, barrier, quiet) |
| Address space | Window-based (rank-relative offset) | Global symmetric address |
| Completion tracking | Signal (remote) + Counter (local) | fence / quiet / barrier |
| Standard compatibility | NVIDIA-proprietary | OpenSHMEM standard |

---

## 7. The DeepEP Signal: Migration from NVSHMEM to NCCL GIN

DeepEP's version history is the clearest signal of ecosystem convergence:

| Version | Symmetric Memory Allocation | Communication Backend |
|---------|---------------------------|----------------------|
| V1 Legacy | `nvshmem_alloc` | NVSHMEM IBGDA (`nvshmemi_ibgda_put_nbi`) |
| V2 Elastic | `ncclMemAlloc` / CUDA Driver API | NCCL GIN (`ncclGin` put/signal) |

The [GIN paper](https://arxiv.org/abs/2511.15076) explicitly uses DeepEP as the integration demonstration: *"We demonstrate GIN's practicality through integration with DeepEP, an MoE communication library."*

**Why the migration matters**: DeepEP V1 proved that GPU-initiated communication is essential for MoE. DeepEP V2 proves that **GIN can match NVSHMEM's device-initiated capability while retaining NCCL's production infrastructure** (hierarchical communicators, elasticity, fault tolerance).

---

## 8. Future Trajectory

### Open Questions

1. **If bottom converges, what decides the winner?**
   - The runtime abstraction's ability to exploit GPU-initiated capability — not the raw mechanism itself.
   - NCCL's structural advantage: SHARP in-network reduction + topology-aware collectives.
   - NVSHMEM's structural advantage: OpenSHMEM standard + one-sided semantics for irregular patterns.

2. **Will NVSHMEM be absorbed into NCCL?**
   - Short term (1-2yr): No. OpenSHMEM compatibility is independent value for HPC.
   - Medium term: Partial. NCCL may expose one-sided put/get primitives, eroding NVSHMEM's differentiation.
   - Key signal: Does NCCL add a one-sided API? If yes, NVSHMEM's reason to exist shrinks dramatically.

3. **Or long-term coexistence?**
   - Most likely: **unified bottom + divided top**. GPU-initiated RDMA as common substrate; NCCL for ML collectives, NVSHMEM for HPC PGAS. Ecosystem inertia > technical merit.

---

## 9. References

| # | Source | URL |
|---|--------|-----|
| 1 | **GPU-Initiated Networking for NCCL** (NVIDIA, arXiv:2511.15076) | https://arxiv.org/abs/2511.15076 |
| 2 | NVIDIA GPUDirect RDMA Documentation | https://docs.nvidia.com/cuda/gpudirect-rdma/ |
| 3 | NVIDIA Blog: Improving Network Performance using GPUDirect Async | https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/ |
| 4 | NVIDIA DOCA GPUNetIO | https://docs.nvidia.com/doca/archive/doca-v2.5.0/gpunetio/index.html |
| 5 | NVSHMEM Installation Guide | https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html |
| 6 | DeepEP GitHub | https://github.com/deepseek-ai/DeepEP |
| 7 | NCCL Documentation | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/ |

### Key Source Paths (DeepEP)

| File | Purpose |
|------|---------|
| `deep_ep/symmetric.hpp` | V1 NVSHMEM alloc vs V2 NCCL `ncclMemAlloc` / CUDA Driver API |
| `deep_ep/nccl.cu` | NCCL GIN context creation (`ncclDevCommCreate` with `ginContextCount`) |
| `deep_ep/handle.cuh` | `gin.put`, `gin.signal` — GPU kernel-side remote operations |
| `deep_ep/internode.cu` | V1 `nvshmemi_ibgda_put_nbi_warp` vs V2 GIN put |

---

### Key GIN Paper Sections

| Section | Content |
|---------|---------|
| § III-A | Key GIN Design Elements (one-sided semantics, symmetric memory windows, GIN contexts, completion tracking, ordering semantics) |
| § III-B | Device-Side API and Programming Model (ncclGin class, API organization, usage workflow) |
| § III-C | Backend Implementations (GDAKI vs Proxy comparison, Table I) |
| § IV | DeepEP Integration (integration requirements, backend strategy, Table II API mapping) |
| § V | Performance Evaluation (point-to-point microbenchmarks, DeepEP HT/LL benchmarks) |

---

> **Date**: 2026-08-16
> **Method**: Source analysis (DeepEP) + NVIDIA official docs + arXiv:2511.15076 (full paper deep read)
