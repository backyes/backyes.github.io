---
title: "DeepEP: From Expert Parallelization to Every Parallelization"
date: 2026-08-07
tags: ["DeepEP", "Expert-Parallelism", "Collective-Communication", "NCCL", "Interconnect", "Rack-Scale", "MoE", "Data-Movement", "Infrastructure-Abstraction"]
excerpt: "DeepEP began as an EP acceleration library for MoE models. Its rebranding to "Every Parallelization" signals a structural ambition: becoming the thin data-transfer abstraction layer that unifies operators, QoS, and direct-drive capabilities across rack- and cluster-scale compute. This post traces that trajectory, draws the NCCL parallel, and argues why a new unified memory-data-movement service layer is emerging."
---

# DeepEP: From Expert Parallelization to Every Parallelization

## Thesis

**DeepEP's repositioning from "Expert Parallelization" to "Every Parallelization" is not marketing — it is a structural claim about where the infrastructure stack is heading.** The thin data-transfer layer that once served MoE all-to-all is evolving into a rack- and cluster-scale abstraction that unifies operators, QoS, and direct-drive capabilities for *all* parallelism strategies: data, pipeline, tensor, memory-pooling, and beyond.

> The center of gravity is shifting: from "EP needs fast all-to-all" to "every parallelism needs a unified data-movement fabric."

---

## 1. The Original DeepEP: MoE's Communication Accelerator

DeepEP's founding problem was narrow and well-defined: **MoE (Mixture-of-Experts) Expert Parallelism requires massive all-to-all communication**, and vanilla NCCL was not optimized for the fine-grained, latency-sensitive token dispatch pattern that EP demands.

| Property | Standard EP (NCCL) | DeepEP (Original) |
|---|---|---|
| Primitive | `AllToAll` / `AllGather` | Low-latency all-to-all |
| Target workload | Data/pipeline parallel | MoE expert dispatch |
| Optimization goal | Bulk throughput | ==Low-latency, small-message== |
| Transport | IB/RoCE via NCCL | IB/RoCE with custom scheduling |
| Granularity | Tensor-level | Token-level |

**The core insight**: MoE's expert dispatch is not a "big tensor" problem — it is a ==high-frequency, small-payload, latency-critical== problem. NCCL's bulk-transfer semantics (optimized for AllReduce of gradient tensors) leave EP's tail latency exposed. DeepEP attacked this directly with kernel-level optimizations, RDMA one-sided operations, and bypass paths that avoid NCCL's synchronization tax.

This was a *point solution* — brilliant, but narrow.

---

## 2. The Pivot: "Every Parallelization" as Infrastructure

The rebranding to **DeepEP = "Deep Every Parallelization"** reveals a fundamentally different ambition. The new positioning targets:

- **Low-latency, high-bandwidth Pod-scale fabrics** (IB and RoCE domains)
- **Rack-scale compute infrastructure** (NVL72-class and beyond)
- **A unified data-transfer layer** that sits *below* parallelism strategies and *above* raw transport

```
┌─────────────────────────────────────────────────┐
│         Parallelism Strategies                   │
│   Data / Pipeline / Tensor / EP / Memory-Pool   │
├─────────────────────────────────────────────────┤
│              DeepEP Abstraction Layer            │
│   Operators │ QoS │ Direct-Drive │ Scheduling   │
├─────────────────────────────────────────────────┤
│         Transport Fabric (IB / RoCE)             │
└─────────────────────────────────────────────────┘
```

**What changed**: DeepEP is no longer "the EP library." It is positioning as the ==thin data-transfer interface== that:

1. **Abstracts underlying operators** — all-to-all, all-gather, reduce-scatter, and custom MoE primitives behind a unified API
2. **Enforces QoS** — latency-critical EP traffic vs. bulk gradient traffic get differentiated scheduling
3. **Exposes direct-drive capabilities** — applications can bypass the generic stack for deterministic, kernel-level data movement

This is the difference between a *library* and a *framework*. A library solves one problem well. A framework defines the boundary of a subsystem.

---

## 3. The NCCL Parallel: From Baidu to Cluster-Scale Dominance

DeepEP's trajectory has a striking historical parallel — **NCCL itself**.

### 3.1 NCCL Was Not Born at NVIDIA

NCCL (NVIDIA Collective Communications Library) was ==originally invented by Baidu's US research lab== in 2015-2016, not by NVIDIA. Baidu developed the first optimized multi-GPU collective communication primitives for their internal PaddlePaddle framework, recognizing that GPU-to-GPU data movement was becoming the binding constraint for distributed training.

NVIDIA's response was swift and strategic:

| Phase | Timeline | Action | Scope |
|---|---|---|---|
| **Internal** | 2015-2016 | Baidu invents optimized collective comms | Single machine, multi-GPU |
| **Absorption** | 2017-2018 | NVIDIA launches NCCL 1.x/2.x | Intra-node (PCIe/NVLink) |
| **Expansion** | 2019 | NVIDIA acquires Mellanox (IB) | Inter-node, cluster-scale |
| **Dominance** | 2020-2024 | NCCL becomes de facto standard | Full cluster stack |
| **Convergence** | 2025+ | Supernode / rack-scale integration | NVL72, NVSwitch fabrics |

**The pattern**: A communication primitive invented for one scale (intra-node) was absorbed, productized, and then ==extended across scale boundaries== via acquisition (Mellanox IB) and architectural integration (NVLink → NVSwitch → NVL72).

### 3.2 The Structural Similarity

DeepEP is following the same playbook — but starting from the *opposite end* of the stack:

| Dimension | NCCL Trajectory | DeepEP Trajectory |
|---|---|---|
| Origin | Intra-node (Baidu) | EP-specific (MoE) |
| First expansion | Inter-node via IB | Pod/rack-scale fabrics |
| Acquisition/integration | Mellanox IB | — (organic growth) |
| End state | Cluster-wide collective standard | ==Every-parallelization abstraction== |

**The key difference**: NCCL expanded *up* from intra-node to cluster. DeepEP is expanding *outward* from a specific parallelism (EP) to all parallelisms. Both converge on the same destination: ==a unified data-movement layer that abstracts the underlying fabric==.

---

## 4. Why Now: The Convergence Forces

Four structural forces are driving the need for a DeepEP-style abstraction layer:

### 4.1 Supernode → Rack-Scale → Bus Fusion

The industry is converging on ==rack-scale compute domains== where compute, memory, and networking are co-packaged as a single unit (NVL72, GB200 NVL72, future Rubin NVL72). Within these domains, the traditional boundaries between intra-node (NVLink) and inter-node (IB) are blurring into a unified bus fabric.

**Implication**: A data-transfer layer that can operate *across* this unified fabric — not just within one scale — becomes essential.

### 4.2 Multi-Media, Tightly-Coupled Memory Movement

New workload patterns are creating ==multi-tier, heterogeneous data-movement demands== that a single-purpose library cannot serve:

| Pattern | Data Movement Characteristic | Traditional Layer |
|---|---|---|
| **n-gram speculative decoding** | Small, frequent KV lookups | NCCL (poor fit) |
| **KV-Cache offloading** | HBM ↔ DDR ↔ Storage | Custom (no standard) |
| **P/D disaggregation** | Prefill ↔ Decode KV transfer | RDMA (low-level) |
| **Incremental KV caching** (decode side) | Streaming, persistent state | No standard |
| **Memory pooling** (CXL/fabric) | Shared, remote memory access | No standard |

**The common thread**: These are all ==tightly-coupled, latency-sensitive, multi-media data-movement problems== that share the same underlying fabric (IB/RoCE) but have radically different semantic requirements. A unified abstraction layer can optimize across them — a point solution cannot.

**DeepEP V2: the shift is already happening.** DeepEP's evolution from V1 to V2 demonstrates this expansion in practice — the library has already moved well beyond EP-specific all-to-all:

| DeepEP V2 Primitive | Target Parallelism | What It Does |
|---|---|---|
| **Engram** | KV-Cache access | ==Zero-SM RDMA remote KV-Cache read== — bypasses GPU compute units entirely for KV retrieval, freeing SMs for compute |
| **pp_send / pp_recv** | Pipeline Parallelism | Dedicated pipeline-parallel send/recv primitives, optimizing the inter-stage tensor handoff pattern |
| **AGRS** (All-Gather Reduce-Scatter) | Data / Tensor Parallelism | Unified all-gather + reduce-scatter for data-parallel and tensor-parallel communication patterns |
| **NCCL Gin backend** | All patterns | V2 switched from NVSHMEM to ==NCCL Gin backend==, reducing ==SM occupancy by up to 4×** — the same primitives, drastically lower overhead |

**The significance**: DeepEP V2 is not merely "EP library plus some extras." The addition of Engram (KV-Cache), PP primitives (pipeline), and AGRS (data/tensor) means DeepEP already covers ==all four major parallelism strategies== (EP, PP, data, tensor) plus KV-Cache storage access. The "Every Parallelization" rebranding reflects what the codebase already does — it is a retrospective acknowledgment, not aspirational marketing.

### 4.3 The "Bus Load" Problem

As these patterns multiply, the ==bus itself becomes the shared bottleneck==. Every data-movement pattern competes for the same IB/RoCE bandwidth and latency budget. Without a unified layer:

- EP traffic starves KV-cache transfers
- Bulk gradient AllReduce disrupts latency-critical speculative decoding
- Memory-pooling traffic has no QoS guarantees

**DeepEP's value proposition**: A single layer that ==schedules, prioritizes, and optimizes== all data-movement across the shared bus fabric.

### 4.4 The Abstraction Tax Is Falling

Historically, a unified abstraction layer was too expensive — the overhead of generality exceeded the benefit of optimization. But:

- **Kernel-level optimizations** (RDMA one-sided, GPU-initiated communication) reduce the abstraction penalty
- **Workload diversity** means the *cost of not* abstracting (fragmented, conflicting data-movement) now exceeds the cost of abstracting
- **Scale** (hundreds of GPUs per pod, thousands per cluster) makes manual per-pattern optimization infeasible

> The abstraction tax has crossed the threshold: a unified layer is now cheaper than the sum of point solutions.

---

## 5. AMD and NVIDIA's Communication Infra: Diverging Philosophies

As DeepEP expands toward "every parallelization," it's worth contextualizing how AMD and NVIDIA — the two dominant GPU vendors — approach the communication infrastructure layer. Their historical paths and current strategies reveal different philosophies about where the abstraction should live.

### 5.1 AMD: A Separate Branch

AMD has essentially ==pulled its own branch== in communication infra, vertically integrated but vendor-specific:

- **RCCL** (ROCm Communication Collective Library) is AMD's NCCL equivalent, tightly coupled to AMD's GPU and CPU ecosystem
- **Infinity Fabric** serves as both intra-chip (chiplet-to-chiplet) and inter-node fabric — a unified physical layer across scales
- **Historical pattern**: AMD's approach mirrors its broader strategy — build a coherent, self-consistent stack that works optimally within AMD's own ecosystem

**The trade-off**: AMD's branch is ==self-consistent but isolated==. It works well for all-AMD deployments, but multi-vendor environments (common in cloud and enterprise) face friction. AMD is essentially betting that ecosystem coherence beats openness.

### 5.2 NVIDIA: Open at the Lower Level

NVIDIA has taken a ==more open but lower-level== approach:

- **NCCL** remains the de facto collective communication standard, but NVIDIA increasingly exposes *lower-level primitives* (NVLink-C2C, NVSwitch, GPUDirect RDMA) rather than only high-level collective APIs
- **Historical pattern**: NVIDIA's communication strategy evolved from absorbing Baidu's intra-node invention → acquiring Mellanox for inter-node IB → integrating NVLink/NVSwitch for supernode fabrics. Each step exposed more of the underlying transport to ecosystem builders
- **Current direction**: NVIDIA is ==opening the lower layers== — providing building blocks (transport, memory semantics, chip-to-chip interfaces) and letting ecosystem layers build abstractions on top

**The strategic logic**: NVIDIA wins regardless of which higher-level abstraction wins, because it owns the underlying transport fabric. DeepEP on NVIDIA hardware is complementary — it builds *on top of* NVIDIA's primitives rather than competing with them.

### 5.3 NVIDIA's ICMS: KV-Cache Storage Infrastructure

NVIDIA has also developed **ICMS (Inference Context Memory Storage)** — a storage platform specifically built for LLM inference scenarios, designed as ==dedicated KV-Cache/context storage infrastructure==. Unlike general-purpose storage or memory fabrics, ICMS targets the unique demands of inference workloads: massive KV-Cache capacity, high-bandwidth streaming access, and tight coupling with the compute fabric.

**The significance**: ICMS represents NVIDIA's recognition that ==KV-Cache storage is distinct enough from general memory hierarchies to warrant its own optimized infrastructure==. This is the same structural force driving DeepEP's expansion — the realization that different data-movement patterns (EP dispatch vs. KV-Cache storage/retrieval vs. gradient AllReduce) have fundamentally different requirements, and a one-size-fits-all approach leaves performance on the table.

**The open question**: Will ICMS remain a KV-Cache-specific storage tier, or will it expand into a broader "every parallelization" memory-data-movement platform? If NVIDIA opens ICMS semantics upward, it could absorb the abstraction layer that DeepEP is building toward. If it stays KV-Cache-specific, it becomes one more specialized tier that a unified abstraction layer (like DeepEP) would need to subsume and optimize across.

---

## 6. The Endgame: A Unified Memory-Data-Movement Service

The structural endpoint is becoming clear: **a new abstract service layer for unified memory-data-movement optimization**.

This service would:

1. **Unify operators** — all-to-all, all-gather, reduce-scatter, KV-dispatch, memory-pool access behind a single API
2. **Enforce cross-workload QoS** — EP latency vs. gradient throughput vs. KV-cache bandwidth, all scheduled against a shared policy
3. **Abstract the fabric** — IB, RoCE, NVLink, CXL, and specialized storage infra like NVIDIA's ICMS (Inference Context Memory Storage) — behind a single transport interface
4. **Enable multi-media data movement** — HBM, DDR, CXL memory, storage — as a unified tiered space

**Why this is inevitable**:

- The ==bus is the shared resource== — all workloads compete for it
- The ==workload diversity== is too high for point solutions
- The ==abstraction cost== has fallen below the fragmentation cost
- The ==scale== (pod → rack → cluster) demands a single optimization surface

**Why DeepEP is well-positioned**:

- It already solves the hardest subproblem (low-latency EP all-to-all)
- Its rebranding signals the strategic intent to expand
- It sits at the right layer — above transport, below parallelism strategies
- The MoE workload that created it is the fastest-growing parallelism in LLMs

---

## 7. Open Questions

| Question | Consideration |
|---|---|
| **Will NVIDIA absorb DeepEP's abstraction into NCCL?** | Possible — but NCCL's bulk-transfer DNA may resist the low-latency, fine-grained semantics DeepEP requires |
| **Can DeepEP remain hardware-agnostic?** | Critical for adoption — but deep kernel optimizations are often vendor-specific |
| **Does ICMS/CMX define a competing abstraction?** | Open question — currently KV-Cache storage-specific, but could expand into a broader memory-data-movement layer |
| **Will AMD build a competing "every parallelization" layer?** | RCCL's current scope is narrower — but the strategic logic is the same |
| **Is the abstraction layer a library, a daemon, or a service?** | The trend is toward a ==daemon/service model== (persistent state, cross-workload scheduling) |

---

## 8. Conclusion

DeepEP's journey from "Expert Parallelization" to "Every Parallelization" mirrors the industry's broader trajectory: **the data-movement layer is becoming the central abstraction of AI infrastructure**.

The NCCL story — from Baidu's intra-node invention to NVIDIA's cluster-scale standard — shows how communication primitives absorb upward through scale. DeepEP is running the same playbook in reverse: starting from a specific parallelism and expanding outward to all parallelisms.

The structural forces are aligned: rack-scale compute convergence, multi-media memory movement, bus-load competition, and falling abstraction costs all point toward a ==unified memory-data-movement service layer==.

The question is not *whether* this layer will exist, but *who* will define it — and whether it will be open or proprietary, a library or a service, a NVIDIA product or an independent standard.

> **The thin data-transfer layer is no longer thin. It is becoming the most strategic surface in AI infrastructure.**

---

## References

- DeepEP: Efficient Expert Parallelism for MoE Training and Inference (DeepSeek, 2024-2025)
- DeepEP V2 — DeepEP (DeepEveryParallel): Engram (zero-SM RDMA KV read), pp_send/pp_recv (Pipeline Parallelism), AGRS (All-Gather Reduce-Scatter), NCCL Gin backend — [https://github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- NCCL: NVIDIA Collective Communications Library — [https://developer.nvidia.com/nccl](https://developer.nvidia.com/nccl)
- Baidu's early collective communication work (PaddlePaddle, 2015-2016)
- NVIDIA Mellanox acquisition (2019) — [https://nvidianews.nvidia.com/news/nvidia-to-acquire-mellanox-for-6-9-billion](https://nvidianews.nvidia.com/news/nvidia-to-acquire-mellanox-for-6-9-billion)
- NVLink-C2C and NVSwitch architecture (NVIDIA, 2024-2025)
- ICMS / CMX (Inference Context Memory Storage → Context Memory eXtension) — NVIDIA pod-level KV-Cache storage based on BlueField-4, announced at CES 2026
- Prefill-Decode disaggregation: vLLM, SGLang, Mooncake (2025-2026)
- CXL memory pooling and fabric-attached memory (2025-2026)
