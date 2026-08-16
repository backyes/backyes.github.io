---
title: "Deepdive GPU Drive Communication: From GPUDirect RDMA to GDAKI"
date: 2026-08-16
tags: ["GPUDirect", "GDAKI", "GIN", "NVSHMEM", "NCCL", "RDMA", "IBGDA", "GPUNetIO", "DOCA", "DeepEP", "MoE"]
excerpt: "GPU-driven networking evolved through three generations: GPUDirect RDMA (CPU-controlled), GPUDirect Async (GPU-triggered, CPU-configured), GDAKI (GPU full-control). NVIDIA's GIN (NCCL 2.28) and DOCA GPUNetIO now converge on the same hardware mechanism — but serve different runtime ecosystems."
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

## 4. Runtime Ecosystem: Where the Real Difference Lies

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

## 5. The DeepEP Signal: Migration from NVSHMEM to NCCL GIN

DeepEP's version history is the clearest signal of ecosystem convergence:

| Version | Symmetric Memory Allocation | Communication Backend |
|---------|---------------------------|----------------------|
| V1 Legacy | `nvshmem_alloc` | NVSHMEM IBGDA (`nvshmemi_ibgda_put_nbi`) |
| V2 Elastic | `ncclMemAlloc` / CUDA Driver API | NCCL GIN (`ncclGin` put/signal) |

The [GIN paper](https://arxiv.org/abs/2511.15076) explicitly uses DeepEP as the integration demonstration: *"We demonstrate GIN's practicality through integration with DeepEP, an MoE communication library."*

**Why the migration matters**: DeepEP V1 proved that GPU-initiated communication is essential for MoE. DeepEP V2 proves that **GIN can match NVSHMEM's device-initiated capability while retaining NCCL's production infrastructure** (hierarchical communicators, elasticity, fault tolerance).

---

## 6. Future Trajectory

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

## 7. References

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

> **Date**: 2026-08-16
> **Method**: Source analysis (DeepEP) + NVIDIA official docs + arXiv:2511.15076
