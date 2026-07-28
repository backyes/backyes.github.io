---
title: "Understanding GPU Microarchitecture from a Workload Perspective (3): DeepEP's First Principles"
date: 2026-07-28
tags: ["GPU", "DeepEP", "MoE", "Data-Movement", "Dataflow", "All-to-All", "Expert-Parallelism", "Warp-Specialization", "Buffer-System", "NVLink", "RDMA"]
excerpt: "DeepEP is not an All-to-All communication library — it is a Data Movement Runtime for MoE. Its core challenge is converting dynamic sparse Token-Expert mappings into continuous GPU-friendly data layouts. From this first-principles perspective, DeepEP's Buffer system, Normal/Low-Latency kernels, and Warp Specialization all become coherent design choices."
---

# Understanding GPU Microarchitecture from a Workload Perspective (3): DeepEP's First Principles

## Introduction: Re-understanding DeepEP

In many descriptions, DeepEP is simply characterized as:

> "A high-performance All-to-All communication library for MoE."

This is not wrong, but it fails to explain *why* MoE communication became a core bottleneck, and *why* DeepEP needs complex kernels, buffers, and Warp pipelines.

From a first-principles perspective, MoE's real challenge is not "how to send data from GPU A to GPU B."

The truly difficult problem is:

> **How does the dynamic sparse Token-Expert mapping produced by the Router get transformed into data layouts that both GPU communication hardware and Tensor Core compute hardware can process efficiently?**

MoE's core tension can be expressed as:

```
Dynamic sparse Token-Expert data flow
              ↓
Continuous regular GPU Tensor data flow
```

There is a structural gap between the two.

The Router outputs `Token → Expert`, but:
- The network wants **Destination-major**
- Expert GEMM wants **Expert-major**
- The next Transformer layer still wants **Token-major**

Therefore, every MoE Layer must perform a dynamic data layout transformation.

---

From this perspective:

**DeepEP is not primarily a complete MoE Runtime — it is a Runtime specialized in solving MoE Expert Parallelism data movement problems.**

It handles:
- Token Dispatch
- Expert Routing data organization
- Inter-GPU communication
- Data rearrangement
- Combine recovery

It connects:

```
Router → DeepEP → Expert Buffer → Expert GEMM
```

Where Expert GEMM is typically handled by DeepGEMM, CUTLASS, or custom CUDA kernels.

A more accurate definition:

> **DeepEP is a MoE-oriented Data Movement Runtime that transforms dynamic sparse Token flows into continuous data flows that Expert computation can consume, through data layout transformation, communication pipelining, and asynchronous scheduling.**

---

# 1. Dispatch / Combine: Dynamic Data Layout Transformation

## 1.1 Why Does MoE Need Dispatch?

In ordinary Transformers, the input naturally maintains Token-major layout:

```
T0, T1, T2, T3, ...
```

Attention and MLP both compute following this layout.

But in MoE, the Router dynamically selects Experts per Token based on semantics:

```
T0 → E2, E7
T1 → E1, E5
T2 → E7, E3
```

This produces a dynamic sparse `Token → Expert` mapping. The problem: different GPU modules need different layouts.

### Communication Phase Wants Destination-major

GPU communication asks: *which GPU should this data go to?*

```
GPU0: T0 → GPU1, T1 → GPU3, T2 → GPU1
```

Communication needs:

```
GPU1: T0, T2
GPU3: T1
```

This forms **Destination-major** layout — contiguous sends.

### Expert GEMM Wants Expert-major

After the destination GPU receives data, Expert GEMM does not want source-GPU ordering. It wants:

```
Expert2: T0, T20
Expert3: T2, T30
```

This forms **Expert-major** layout — a continuous `[M, K]` matrix for Tensor Core.

---

Therefore, Dispatch actually performs:

```
Token-major → Destination-major → Expert-major
```

This is not simple communication. It is a ==**dynamic data layout transformation process**==.

## 1.2 Combine: Expert-major Back to Token-major

After Expert computation, the output remains Expert-major:

```
Expert2 Output: T0, T20
Expert3 Output: T2, T30
```

But the model's next layer needs Token-major. Combine must reverse:

```
Expert-major → Destination-major → Token-major
```

While also recovering Router semantics:

```
T0: Expert2 weight=0.73, Expert7 weight=0.27
Output(T0) = 0.73 × Expert2(T0) + 0.27 × Expert7(T0)
```

Combine is not merely the inverse of communication. It is ==**data layout recovery + semantic recovery**==.

---

# 2. DeepEP Buffer System: Data Flow Organization Under Different Kernels

DeepEP does not use identical data paths in all scenarios. MoE workloads differ significantly:

- **Training / Prefill** care about throughput
- **Decode** cares about latency

Therefore, Normal Kernel and Low-Latency Kernel employ different dataflow designs.

## 2.1 Normal Kernel: Throughput-Optimized Complete Pipeline

Normal Kernel targets large-batch inference, Prefill, and Training. Goal: maximize communication throughput and hide communication overhead.

Typical data path:

```
Token Buffer → Dispatch Buffer → Chunk Buffer → NVLink / RDMA Pipeline → Receive Buffer → Expert Buffer
```

### Token Buffer
Stores Hidden States after Router output. Layout: **Token-major**.

### Dispatch Buffer
Completes the first layout transformation: **Token-major → Destination-major**. Purpose: make communication data contiguous.

### Chunk Buffer
Chunk is critically important in Normal Kernel. The network is unsuitable for single-Token sends — each Token triggering RDMA individually produces small packets, high startup overhead, and low bandwidth utilization. Therefore, Tokens are first aggregated:

```
Token Stream → Chunk → Network Transfer
```

Here, Token is the **scheduling granularity**; Chunk is the **communication granularity**.

### Receive Buffer
The destination GPU receives Chunks from multiple GPUs. Still in communication layout.

### Expert Buffer
Completes the final transformation: **Destination-major → Expert-major**, forming Expert GEMM input.

---

## 2.2 Low-Latency Kernel: Decode-Oriented Short Path

Decode characteristics: small batch, few Tokens, single-Token latency sensitive. Throughput is not the primary concern. Waiting for Chunk aggregation反而 increases queuing time, buffer latency, and pipeline depth.

Therefore, Low-Latency Kernel reduces intermediate layers:

```
Token Buffer → Direct RDMA → Receive Buffer → Expert Buffer
```

Compared to Normal Kernel: reduced Chunk aggregation, forwarding, and intermediate buffering. Goal: minimize ==**end-to-end Token latency**==.

---

# 3. Normal vs Low-Latency: Two Communication Philosophies

DeepEP's two kernels embody two system optimization directions.

| | Normal Kernel | Low-Latency Kernel |
|---|---|---|
| **Primary Scenario** | Training / Prefill | Decode |
| **Goal** | Maximize throughput | Minimize latency |
| **Core Tension** | Bandwidth | Latency |
| **Chunk** | Critical | Reduced |
| **Pipeline** | Deep | Shallow |
| **Communication Path** | NVLink + RDMA coordination | Direct RDMA |

## 3.1 Normal Kernel: Communication Pipelining

In multi-GPU nodes, GPU-NIC topology is not fully symmetric. Some GPUs are closer to the NIC; others require GPU-to-GPU forwarding. Therefore, communication paths may include:

```
IB Sending → IB-to-NVLink Forwarding → NVLink Receiving
```

Forming a three-stage communication pipeline.

---

# 4. Warp Specialization: Pipelined Execution Inside the Communication Kernel

A clarification: Warp Specialization in DeepEP is *not* about:
- GPU role assignment
- SMs dedicated to compute or communication
- Warps executing Expert GEMM

It is primarily used for **parallelizing different stages within the communication Kernel**.

For example, in Normal Kernel, different Warp Groups can handle:

```
Warp Group A: IB Sending
Warp Group B: IB-NVLink Forwarding
Warp Group C: NVLink Receiving
```

Forming:

```
Send → Forward → Receive
```

---

# 5. FIFO: From Synchronous to Streaming Pipeline

Without FIFO: before the previous stage completes, the next stage must wait:

```
Send → Barrier → Forward → Barrier → Receive
```

Inefficient.

With FIFO:

```
Send → FIFO → Forward → FIFO → Receive
```

Each stage only cares about its own write/read. No need to wait for the entire Batch.

==**FIFO's essence: transforms the communication pipeline from synchronous execution to streaming execution.**==

---

# 6. Metadata: How Dynamic Routing Becomes Contiguous Access

Two analytical abstractions. Note: **Layout Metadata / Identity Metadata are not official DeepEP source code terms — they are conceptual models proposed for understanding MoE Runtimes.**

## 6.1 Layout Metadata: Where Should Data Go?

Core question: **Where?**

Describes data layout. Typical information: count, prefix sum, offset.

```
Expert2: 3 tokens
Expert3: 5 tokens

Prefix: Expert2 offset=0, Expert3 offset=3
```

Then: `dst = prefix[expert]++` completes contiguous writes.

Solves: ==**dynamic mapping → contiguous addresses**==.

## 6.2 Identity Metadata: Who Is This Data?

Core question: **Who?**

Describes Token identity. Includes: token id, expert id, gate weight, top-k slot.

```
Position 100 → Token17 → Expert2 → weight=0.73
```

During Combine, recover:

```
Token17 = 0.73 × Expert2 + 0.27 × Expert7
```

---

# 7. Key Details in MoE Runtime

## 7.1 Does Sort Exist?

Yes, but not traditional sorting. Core process:

```
Count → Prefix Sum → Scatter
```

Essentially **Counting Sort / Bucketization**. Goal: not to sort by size, but to produce contiguous layout.

## 7.2 Why Must Top-K Information Be Preserved?

Because MoE is not one Token to one Expert — it's Top-K Experts:

```
T0: Expert2 weight=0.73, Expert7 weight=0.27
```

Combine must know: which Experts, corresponding weights, ordering. Therefore, Top-K slot information must be preserved.

## 7.3 Why Is Memory Reorganization Unavoidable?

Because the three stages are inherently different:

- Communication: **Destination-major**
- Compute: **Expert-major**
- Model: **Token-major**

Rearrangement is not an implementation defect — it is a **necessary transformation** produced by MoE architecture itself. The optimization direction is not to eliminate it, but to ==**make data rearrangement, communication, and computation as pipelined as possible**==.

---

# 8. From DeepEP to Mega MoE: MoE Runtime Evolution

DeepEP solves **Communication + Data Movement**. But system trends are further fusing **Communication + Compute**.

This direction is not future speculation — in DeepGEMM, Mega MoE Kernel has already demonstrated a new execution pattern targeting Blackwell SM100 and specific MoE workloads, further fusing:

```
Token Routing → Dispatch → Linear1 → Activation → Linear2 → Combine
```

into an integrated execution flow.

Distinction:
- **DeepEP**: MoE Data Movement Runtime
- **Mega MoE**: Communication + Compute Fusion Runtime

The former solves how data moves efficiently; the latter further solves how data movement and computation can be co-scheduled.

---

# 9. Summary: DeepEP's True System Significance

DeepEP's value is not just improving All-to-All performance. It solves:

> **How to transform dynamic sparse Token-Expert dataflows into continuous dataflows that GPU hardware prefers.**

Core mechanisms: Dispatch / Combine, Data Layout Transformation, Buffer Pipeline, Chunk Streaming, FIFO Stage Decoupling, Warp Specialization, Metadata-driven Mapping.

Final abstraction:

```
Router → Token-Expert Stream → DeepEP (Data Movement Runtime) → Expert Buffer → Expert GEMM → Combine
```

Further evolution:

```
DeepEP + DeepGEMM + Fusion Kernel → Unified MoE Dataflow Runtime
```

==**DeepEP's important significance is that it represents MoE systems' evolution from "communication optimization" to "dataflow execution."**==

It does not simply replace NCCL — it solves a more fundamental problem:

> **How to make dynamic sparse AI workloads ultimately flow into the continuous compute streams that GPUs excel at processing.**

---

*Analysis based on DeepEP paper and source code, CUTLASS documentation, and NVIDIA NVLink/RDMA architecture. Views are my own.*

---

*© 2026 backyes · Created by backyes*
