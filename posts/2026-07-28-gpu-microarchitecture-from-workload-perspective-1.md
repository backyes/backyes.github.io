---
title: "Understanding GPU Microarchitecture from a Workload Perspective (1): The End of Block Independence and the Rise of On-chip Dataflow"
date: 2026-07-28
tags: ["GPU", "Microarchitecture", "Hopper", "CUDA", "Thread-Block-Cluster", "Shared-Memory", "DSM", "TMA", "Warp-Specialization", "AI-Dataflow", "Persistent-GEMM", "FlashAttention", "MoE"]
excerpt: "GPU's traditional execution model assumes Thread Blocks are independent — but modern AI workloads break this assumption. Hopper's two key evolutions — raising the Resident Block limit and introducing Thread Block Cluster — represent a fundamental shift: from optimizing single-Tensor-Core utilization to optimizing on-chip dataflow across the entire GPU."
---

# Understanding GPU Microarchitecture from a Workload Perspective (1): The End of Block Independence and the Rise of On-chip Dataflow

## The Foundational Assumption: Block Independence

GPU's traditional execution model was built on graphics rendering workloads, whose core assumption is simple: **Thread Blocks are independent of each other**. CUDA codifies this assumption into a unified boundary — the Block serves simultaneously as the unit of resource allocation, Shared Memory ownership, synchronization scope, and scheduling. A Block monopolizes one SM's Shared Memory; Blocks cannot directly share on-chip data and must exchange data through L2 and HBM.

This design served Graphics, traditional HPC, and early GEMM workloads well. Data reuse happened primarily *within* a Block, and there was almost no need for fine-grained inter-Block collaboration. The Block was a self-contained island of compute and memory.

## AI Workloads Break the Assumption

With the rise of Transformers, MoE, FlashAttention, Persistent GEMM, and fused AI kernels, this foundational assumption is increasingly violated. The bottleneck in modern AI kernels is shifting — less from Tensor Core FLOPS, more from ==**On-chip Dataflow**==.

Consider the pattern: multiple Blocks often need to collaboratively complete a single tile, a pipeline stage, or a fusion operator. A Producer Block generates data that a Consumer Block needs *immediately*. Under the traditional CUDA model, this intermediate result must traverse the path:

```
Shared Memory → L2 → HBM → L2 → Shared Memory
```

This not only consumes precious HBM bandwidth but also adds L2 traffic and access latency. The data takes a detour through the entire memory hierarchy when it should have moved a few millimeters on-chip.

Meanwhile, Transformer-era kernels increasingly use small Blocks — a Block of 64–128 threads (2–4 Warps) is common for LayerNorm, Softmax, Activation, and Embedding kernels. If an SM can still only resident 16 Blocks, then at most ~32 Warps can be resident on an SM that supports 64 Warps. Half the scheduling capacity sits idle. ==**SM Occupancy becomes the new utilization bottleneck.**==

## Hopper's Two Representative Evolutions

Against this backdrop, Hopper introduced two microarchitecturally significant changes.

### Evolution 1: Resident Block Limit Raised from 16 to 32

This change is not about supporting larger kernels — it is about adapting to increasingly small Block kernels. In the past, a typical GEMM Block had 256–512 threads; an SM would quickly hit Warp, register, or Shared Memory limits, making the Resident Block ceiling irrelevant. But in the Transformer era, lightweight kernels use Blocks of only 2–4 Warps. With a 16-Block limit, an SM could resident at most ~32 Warps — leaving half of Hopper's 64-Warp capacity unused.

Raising the limit to 32 allows small-Block kernels to resident more Warps, significantly improving Occupancy, hiding HBM latency, and boosting SM utilization. ==**This is fundamentally an AI small-granularity kernel-driven resource scheduling optimization.**==

### Evolution 2: Thread Block Cluster

A natural question arises: if multiple Blocks need to collaborate, why not simply make Blocks larger — merge multiple Blocks into a "Super Block"?

The answer lies in the Block's quadruple responsibility in CUDA: **Scheduling Unit, Resource Allocation Unit (Registers / Shared Memory), Synchronization Domain (Barrier Scope), and Shared Memory Ownership**. Simply expanding a Block would mean: one Block monopolizes an entire SM (destroying Occupancy); Barriers must wait for more Warps (increasing synchronization cost); register and Shared Memory fragmentation worsens; and load imbalance across tiles becomes more severe.

GPU's design principle has always been ==**Small Scheduling Unit + Large Collaboration Domain**==. Hopper did not enlarge the Block — it added a new collaboration layer *above* the Block: the **Cluster**.

## Cluster: SRAM Pooling + Fabric

From a first-principles perspective, **Cluster's essence is: Shared Memory Pooling + On-chip Topology Fabric**. It adds no new SRAM, no new "Cluster Memory" chip. Instead, it connects the previously isolated Shared Memories of multiple SMs through a Cluster Fabric, forming a logical ==**Distributed Shared Memory (DSM)**==.

Concretely, Cluster accomplishes three things:

1. **Logical Pooling** of multiple SMs' Shared Memory
2. **On-chip Fabric** providing remote Shared Memory access paths
3. **Memory Scope expansion** from Block to Cluster

From software's perspective, DSM appears as a new Memory Tier between Local Shared Memory and L2 Cache. From hardware's perspective, it is more precisely an ==**SRAM Pooling + Fabric**== design — not new physical storage.

| Before Hopper | After Hopper |
|---|---|
| Each SM's Shared Memory = isolated resource island (Private SRAM) | Multiple SMs' Shared Memory = logical on-chip resource pool (On-chip SRAM Pool) |
| Inter-Block data exchange: Shared Memory → L2 → HBM → L2 → Shared Memory | Inter-Block data exchange: DSM direct access (on-chip only) |
| Shared Memory scope = Block | Shared Memory scope = Cluster |

This design philosophy is highly consistent with system-level resource pooling trends like CXL Memory Pool and NVLink Memory Fabric — except the pooled object is on-chip SRAM rather than DRAM/HBM.

## What Cluster Actually Solves

Cluster optimizes not compute, but ==**inter-Block data flow**==.

The publicly known large-scale adopters of Cluster are a small number of highly optimized AI kernels: **CUTLASS Persistent GEMM, FlashAttention-3, Transformer Engine, and CUDA Cooperative Algorithms (Histogram, Reduction, etc.)**. These kernels share a common characteristic: compute itself is not the bottleneck; the true bottleneck is frequent intermediate result exchange between multiple Blocks.

Cluster allows Producer Blocks to retain data in DSM for Consumer Blocks to read immediately — bypassing L2 and HBM entirely — enabling cross-SM pipelining. This is why Cluster always appears alongside Hopper's other two critical features: ==**Tensor Memory Accelerator (TMA)**== and ==**Warp Specialization**==:

| Component | Responsibility |
|---|---|
| **TMA** | Efficient Tile movement (HBM ↔ Shared Memory) |
| **Warp Specialization** | Pipeline stage assignment (Producer / Consumer / Compute) |
| **Cluster** | Cross-Block data sharing and synchronization (DSM) |

Together, they form Hopper's core optimization system for AI Dataflow.

## The Arc of GPU Architecture

Looking at the full trajectory of GPU evolution, a clear thread emerges: GPU does not keep enlarging the Block — it progressively **decouples the Block's responsibilities**.

| Era | Block's Role | Optimization Target |
|---|---|---|
| **Pre-AI** | Unified: scheduling + resources + sync + memory | Single Tensor Core utilization |
| **Hopper** | Lightweight Execution Unit | On-chip Dataflow across SMs |
| **Future** | Execution Unit only | Resource & Collaboration Domain shifts to Cluster and beyond |

The Block becomes a lightweight execution unit; the Cluster becomes the fundamental domain of resource sharing and collaborative execution. This shift signals that GPU's optimization target has evolved from ==**"improving single Tensor Core utilization"**== to ==**"optimizing on-chip dataflow across the entire GPU"**==.

Shared Memory is no longer a Block's private cache — it is evolving into an on-chip resource pool shared among multiple compute units. This not only explains why Hopper introduced Cluster, but also reveals the future direction of GPU microarchitecture: ==**compute resources maintain fine-grained scheduling, while on-chip storage and dataflow continuously move toward Pooling, Fabric, and larger-scale collaborative computation.**==

---

*Analysis based on NVIDIA Hopper architecture whitepaper, CUTLASS documentation, FlashAttention-3 paper, and CUDA Programming Guide. Views are my own.*

---

*© 2026 backyes · Follow me on [Zhihu](https://www.zhihu.com/people/nono-nono-66) & [LinkedIn](https://www.linkedin.com/in/yanfei-wang-5081b4126/) for more AI infrastructure insights*

### References

- [NVIDIA Hopper Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) — Thread Block Cluster, DSM, TMA, and Warp Specialization
- [CUDA Programming Guide — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters) — Official specification of Cluster scope and remote Shared Memory access
- [CUTLASS Persistent GEMM](https://github.com/NVIDIA/cutlass) — CUDA Templates for Dense Linear Algebra, large-scale Cluster adoption
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Pipelining](https://arxiv.org/abs/2407.08608) — Cross-SM pipelining via Cluster and Warp Specialization
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) — FP8 training engine leveraging Cluster for inter-Block communication
