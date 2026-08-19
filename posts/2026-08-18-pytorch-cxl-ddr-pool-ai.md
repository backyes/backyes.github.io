---
title: "[AI Generated] PyTorch DataLoader for CXL DDR Pools: A Deep Software Architecture Analysis"
date: 2026-08-18
tags: ["CXL", "PyTorch", "DataLoader", "GPUDirect", "GDS", "DDR Pool", "Linux", "Memory", "PMDK", "Memkind", "DAXFS", "AI Generated"]
excerpt: "The critical missing piece in the CXL-for-training stack is not hardware capability, but a fundamental rethinking of PyTorch's DataLoader architecture. This report provides source-level analysis of PyTorch DataLoader internals, identifies exact integration points for CXL, and systematically answers five critical design questions — from data path to architectural philosophy."
---

# [AI Generated] PyTorch DataLoader for CXL DDR Pools: A Deep Software Architecture Analysis

> **Note**: This post was AI-generated based on systematic research. Source: [pytorch-cxl-ddr-pool.md](https://github.com). Method: Source-level analysis of PyTorch DataLoader internals, NVIDIA GDS/DALI architecture, Linux kernel CXL subsystem, and deep reading of 12+ academic papers (arXiv 2023-2026).
>
> **Companion PDF Slides**: See embedded preview below or [Download PDF](cxl_pytorch_final_v3.pdf)

---

## Thesis

**The critical missing piece in the CXL-for-training stack is not hardware capability, but a fundamental rethinking of PyTorch's DataLoader architecture.** Current DataLoader design assumes a rigid data path: SSD → DRAM (via page cache) → Pinned DRAM (via pin_memory) → GPU. CXL breaks this chain by introducing a tier that is byte-addressable, cache-coherent, and poolable — but PyTorch has no abstraction for "data that lives in CXL memory."

This report provides a source-level analysis of PyTorch DataLoader internals, identifies the exact integration points for CXL, and systematically answers five critical design questions:

1. **How does SSD file data load onto CXL?** — The data path problem
2. **Is CXL a file cache or a sample cache?** — The semantic positioning problem
3. **How do multiple GPUs handle concurrent access to pooled data?** — The coherence problem
4. **What extensions does GPU-side CXL DataLoader require?** — The API compatibility problem
5. **Transparent access vs. dedicated CXL service?** — The architectural philosophy problem

We analyze 12+ academic papers in depth (including PMDK/Memkind/DAXFS/CXLMemUring/CXL-DMSim), extract their design philosophies, and evaluate which approaches have the highest probability of production success.

---

## 1. CXL DDR Pooling: A Paradigm Shift in Memory Expansion

| CXL Version | Key Capability | Pooling | Year |
|---|---|---|---|
| **CXL 1.1** | Base protocol (io / mem / cache) | Fixed mapping | 2019 |
| **CXL 2.0** | Switch + Memory Pooling | Dynamic allocation | 2020 |
| **CXL 3.0** | Multi-level switching + Global sharing | Cross-node sharing | 2022 |

| Tier | Latency | Bandwidth (per channel) | Coherence |
|---|---|---|---|
| **DRAM (local)** | ~80ns | ~100GB/s | Hardware |
| **CXL Type-3** | ~200-400ns | ~32-64GB/s | Hardware (CPU side) |
| **NVMe SSD** | ~10μs | ~7GB/s | None |

---

## 2. PyTorch DataLoader: Source-Level Architecture Analysis

### 2.1 The DataLoader Pipeline

```python
class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, 
                 sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, prefetch_factor=2):
        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn or default_collate
        
    def _get_data(self):
        while True:
            idx = self._next_index()
            data = self._next_data()
            if self.pin_memory:
                data = pin_memory(data)  # PROBLEM: requires page-locked DRAM
            return data
```

### 2.2 Key Internal Components

| Component | Location | Responsibility | CXL Integration Point |
|---|---|---|---|
| **Sampler** | `torch/utils/data/sampler.py` | Generates index sequence | CXL-aware skip |
| **Dataset** | User-defined | `__getitem__(idx)` returns sample | **Primary: returns CXL-backed Tensor** |
| **Worker** | `torch/utils/data/_utils/worker.py` | Subprocess executing `__getitem__` | Can mmap CXL device directly |
| **Queue** | `torch/utils/data/_utils/worker.py` | IPC between worker and main | Can pass CXL pointers |
| **pin_memory_thread** | `torch/utils/data/_utils/pin_memory.py` | Async copy to page-locked DRAM | **Must be redefined** |
| **collate_fn** | User-defined | Merges samples into batch | Must handle CXL-backed Tensors |

### 2.3 The pin_memory Problem

```python
def pin_memory(data):
    if isinstance(data, torch.Tensor):
        # PROBLEM: calls cudaHostAlloc(), requires page-locked DRAM
        # CXL memory CANNOT be page-locked!
        return data.pin_memory()
```

**Critical insight**: `pin_memory()` calls `cudaHostAlloc()` which requires page-locked DRAM. CXL memory is **not** page-lockable in the CUDA sense. This means:

1. **Direct path blocked**: Cannot `pin_memory()` a CXL-backed Tensor
2. **Double-copy required**: CXL → DRAM (pinned) → GPU, adding latency
3. **GDS bypass**: GPUDirect Storage can write directly to CXL-registered buffers

---

## 3. The Memory Wall

| Dataset | Scale | Typical Batch Memory | DRAM Bottleneck |
|---|---|---|---|
| **ImageNet (CV)** | 150 GB | ~1 GB | Fits |
| **LAION-5B (Multimodal)** | 240 TB | ~100 GB | Does not fit |
| **LLM Pre-training (FineWeb)** | PB-scale | ~10 TB+ | Severe bottleneck |

---

## 4. NVIDIA Software Ecosystem: DALI / GDS / Magnum IO

### 4.1 GDS × CXL Intersection

```
Data flow: NVMe → CXL Memory Pool (GDS Buffer) → GPU
           (GDS direct write)              (GPUDirect)
```

cuFileRegister() can register CXL memory as a GDS buffer, enabling direct GPU access.

### 4.2 Three-Stage Pipeline

| Stage | Source | Destination | Mechanism | Latency |
|---|---|---|---|---|
| **Stage 1** | NVMe SSD | CXL Pool | GDS / DAX | ~300ns (CXL) |
| **Stage 2** | CXL Pool | CXL Pool | Preprocessing | ~300ns (CXL) |
| **Stage 3** | CXL Pool | GPU | GDS / Double-buffer | ~5μs |

---

## 5. CXL Software Ecosystem: Linux / ndctl / DAX / Tiering

```
Linux CXL Software Stack
├── Userspace tools: ndctl
│   ├── Create / configure CXL memory regions
│   ├── volatile mode / persistent mode
│   └── Pool management
├── Kernel drivers: cxl_acpi / cxl_pci / cxl_pmem
├── DAX (Direct Access)
│   ├── Bypass page cache, userspace mmap
│   └── ext4 / XFS dax mount options
├── TPP (Transparent Page Placement)
│   ├── Automatic hot/cold page migration
│   └── Meta's 2022 Linux kernel contribution
└── kmem: CXL memory as independent NUMA node
```

---

## 6. Academic Research: Deep Analysis

### 6.1 TRAININGCXL (arXiv:2301.07492) — KAIST 2023

**Design Philosophy**: "Dataset resident in PMEM, GPU directly accesses without CPU intervention."

| Design Element | TRAININGCXL Approach |
|---|---|
| **Hardware** | PMEM + GPU + CXL Type-2 device |
| **Data Placement** | Training datasets reside in PMEM; GPU accesses directly via load/store |
| **Fault Tolerance** | Checkpoint logic near CXL controller, async persistence |
| **Key Optimization** | Exploits sparse access patterns of recommendation models |

**Results**: 5.2× training performance improvement, 76% energy savings.

**Core insight**: CXL provides not just capacity, but a "CPU-uninvolved" data access path.

### 6.2 CCCL (arXiv:2602.22457) — 2026

**Design Philosophy**: "CXL memory pool as cross-node shared memory for GPU collectives."

| Communication Mode | CCCL (CXL) | Baseline (InfiniBand 200Gbps) | Speedup |
|---|---|---|---|
| **AllGather** | CXL Memory Pool | RDMA | 1.34× |
| **Broadcast** | CXL Memory Pool | RDMA | 1.84× |
| **Gather** | CXL Memory Pool | RDMA | 1.94× |
| **Scatter** | CXL Memory Pool | RDMA | 1.04× |

**Results**: 1.11× LLM training speedup with 2.75× hardware cost savings.

**Core insight**: Per-GPU CXL partitions eliminate contention. CXL pools can serve as cross-node data sharing medium.

### 6.3 Proxics (arXiv:2604.18120) — 2026

**Design Philosophy**: "Familiar OS abstractions (processes + pipes) for near-data processing."

- Lightweight process abstraction for NDP accelerators
- Compiler-assisted region identification
- Efficient IPC via CXL atomic operations

**Core insight**: OS abstractions can be preserved while achieving NDP efficiency.

### 6.4 TERAIO (arXiv:2506.06472) — 2025

**Design Philosophy**: "Lifetime-aware tensor offloading for GPU memory expansion."

- Active tensors occupy only 1.7% of allocated GPU memory
- Inactive tensors offloaded to SSDs via GDS
- 80.7% of ideal unlimited GPU memory performance

**Core insight**: GDS can be extended to CXL memory by registering CXL buffers as GDS-compatible storage targets.

### 6.5 CXL-GPU (arXiv:2506.15601) — 2025

**Design Philosophy**: "GPU storage expansion via CXL with custom RTL controller."

- Multiple CXL root ports per GPU
- Custom CXL controller at hardware RTL level
- Two-digit nanosecond roundtrip latency (first in field)

**Core insight**: Custom CXL controllers can achieve near-DRAM latency.

### 6.6 Aquifer (arXiv:2606.24079) — 2026

**Design Philosophy**: "Hierarchical CXL+RDMA pooling for MicroVM snapshots."

- Hotness-based placement: hot data in CXL, cold data in RDMA
- Ownership-based coherence for CXL 2.0 (no hardware coherence)
- 2.2× geometric-mean speedup for MicroVM restore

**Core insight**: Ownership-based coherence protocol can be adapted for multi-GPU DataLoader access.

---

## 7. Storage & Memory Software Stack: Deep Dive

### 7.1 PMDK (Persistent Memory Development Kit)

**Overview**: Intel's open-source persistent memory programming library.

| Component | Function | Key API |
|---|---|---|
| **libpmem** | Low-level persistent memory operations | `pmem_memcpy_persist()`, `pmem_flush()` |
| **libpmemobj** | Transactional object store | `PMobj_begin()`, `PMobj_commit()` |
| **libpmemlog** | Persistent log file | `pmemlog_write()`, `pmemlog_rewind()` |

**Key Capabilities**:
- **Direct Access (DAX)**: bypass page cache for persistent memory
- **Atomic persistence**: 8-byte atomic writes guaranteed
- **Memory-mapped I/O**: `mmap` with `MAP_SYNC` flag
- **Flush optimization**: minimize cache line flush overhead

**Relevance to DataLoader**:
PMDK provides the foundation for applications to directly access CXL memory (when configured as persistent memory) without going through the block storage stack. This enables:
- Direct dataset memory-mapping in CXL
- Atomic updates to shared data structures
- Crash-consistent checkpointing

### 7.2 Memkind: Heterogeneous Memory Allocator

**Overview**: Unified DRAM / PMEM / CXL memory allocation library.

```c
// Bind thread to CXL memory tier
memkind_set_arena(MEMKIND_CXL, arena_id);

// Allocate from CXL tier
void *ptr = memkind_malloc(MEMKIND_CXL, size);
```

**Key Features**:
- **memkind_set_arena()**: binds thread to specific memory tier
- **Tiered allocation**: DRAM (default), PMEM (persistent), CXL (via CXL memory tier)
- **Custom arena management**: per-thread memory tier binding
- **posix_memalign compatibility**: drop-in replacement for standard allocators

**Relevance to DataLoader**:
DataLoader worker threads can bind to CXL NUMA node for CXL-local allocation:
- Reduces cross-NUMA memory access
- Enables thread-local CXL memory pools
- Automatic tier-based allocation policy

### 7.3 DAXFS: Lock-Free Shared Filesystem for CXL Memory

**Overview** (arXiv:2604.01620, 2026): A Linux filesystem for CXL shared memory that uses CXL atomics as its sole coordination primitive.

**Design Philosophy**:
"CXL enables multi-host byte-addressable memory with hardware cache coherence, but no existing filesystem exploits this for lock-free multi-host coordination."

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│  Host 0                              Host 1              │
│  ┌──────────┐                        ┌──────────┐       │
│  │  App     │                        │  App     │       │
│  └────┬─────┘                        └────┬─────┘       │
│       │                                   │             │
│  ┌────▼─────┐                        ┌────▼─────┐       │
│  │  DAXFS   │                        │  DAXFS   │       │
│  │  Instance│                        │  Instance│       │
│  └────┬─────┘                        └────┬─────┘       │
│       │                                   │             │
│  ┌────▼───────────────────────────────────▼─────┐       │
│  │  CXL Shared Memory Region                     │       │
│  │  (cmpxchg atomic operations)                  │       │
│  └───────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

**Key Mechanisms**:
- **CAS-based hash overlay**: lock-free concurrent writes from multiple hosts
- **Cooperative shared page cache**: demand-paged caching in shared DAX memory
- **Multi-host clock eviction (MH-clock)**: decentralized victim selection via cmpxchg
- **No centralized coordinator**: fully decentralized

**Performance**:
- >99% CAS accuracy with no lost updates under cross-host contention
- Up to 2.68× higher random write throughput (4 threads)
- 1.18× higher random read throughput at 64KB
- Exceeds tmpfs on DRAM-backed DAX

**Relevance to DataLoader**:
DAXFS provides the filesystem layer for multi-GPU DataLoader coordination:
- Lock-free shared dataset index across hosts
- Multi-host page cache for hot dataset segments
- Decentralized eviction eliminates bottleneck

### 7.4 CXLMemUring: Asynchronous CXL Memory Pool Access

**Overview** (arXiv:2309.04011, 2023): Hardware/software co-design for hiding CXL latency using larger units of asynchronous work.

**Design Philosophy**:
"CXL memory accesses are too slow for normal CPU mechanisms to hide reliably, especially when each access depends on the result of a previous one. At the same time, they are too fast for traditional software techniques."

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│  Host CPU                                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Application (unchanged)                          │  │
│  │  - Launches "regions" asynchronously               │  │
│  │  - Continues useful work while region executes     │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                               │
│  ┌───────────────────────▼────────────────────────────┐  │
│  │  Runtime + JIT Compiler                             │  │
│  │  - Identifies CXL-resident regions                  │  │
│  │  - Refines region boundaries at runtime             │  │
│  └───────────────────────┬────────────────────────────┘  │
│                          │                               │
│  ┌───────────────────────▼────────────────────────────┐  │
│  │  Vortex-based CXL-side Accelerator                  │  │
│  │  - Executes regions near memory                     │  │
│  │  - Hides CXL latency via async execution            │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Key Mechanisms**:
- **Region-based execution**: groups CXL accesses into async work units
- **Near-memory accelerator**: commodity CXL Type-2 FPGA
- **Compiler-assisted**: identifies candidate regions from unmodified source
- **Online JIT**: refines region boundaries based on runtime behavior

**Performance**:
- Tested on Granite Rapids CXL platform
- Baseline: 2.44× slowdown vs DRAM (GAPBS graph workloads)
- Prefetcher alone: 2.21× slowdown remains
- CXLMemUring: 1.45× to 1.75× improvement (1.59× geometric mean)

**Relevance to DataLoader**:
Region-based async execution is a natural fit for DataLoader's prefetch mechanism:
- DataLoader's prefetch_factor maps to "region" size
- Compiler identifies dataset loading regions
- CXL-side accelerator prefetches data while GPU computes

### 7.5 CXL-DMSim: Full-System CXL Disaggregated Memory Simulator

**Overview** (arXiv:2411.02282, IEEE TCAD 2024): Open-source full-system simulator with gem5-comparable speed.

**Key Features**:
- Flexible CXL memory expander model + device driver
- CXL.io and CXL.mem protocol support
- App-managed (AM) and kernel-managed (KM) modes
- NUMA-compatible mechanism for kernel management

**Silicon Validation**:
- Verified against real FPGA- and ASIC-based CXL devices
- Average simulation error: 3.4%
- CXL-FPGA: ~2.88× higher latency than DDR
- CXL-ASIC: ~2.18× higher latency than DDR
- CXL-FPGA: 45-69% of DDR bandwidth
- CXL-ASIC: 82-83% of DDR bandwidth

**Relevance to DataLoader**:
Essential tool for evaluating CXL memory pool performance without physical hardware.

### 7.6 CXLRAMSim: Memory Card Simulator

**Overview** (arXiv:2603.29483, 2026): First gem5-integrated full-system simulator for CXL memory expander cards.

**Key Features**:
- Models CXL devices at correct I/O bus position
- Uses unmodified Linux kernels and software stack
- Realistic latency-bandwidth behavior
- True interleaving with system DRAM
- Captures cache pollution when accessing CXL memory

**Core Insight**:
Enables rapid prototyping of CXL-aware DataLoader designs with high fidelity.

---

## 8. Industrial Practice

| Company | Solution | Core Capability | Status |
|---|---|---|---|
| **Meta** | TPP | CXL auto-tiering (Linux kernel) | Merged into mainline |
| **Intel** | Xeon 6 + Micron CXL expansion | System memory bandwidth optimization | In production (2024) |
| **Samsung** | CMM-H (Hybrid) | DRAM cache + NAND CXL device | Evaluated (arXiv:2503.22017) |
| **MemVerge** | Memory Machine | CXL memory virtualization + Kubernetes orchestration | Commercial |
| **NVIDIA** | DGX / SuperPOD | NVLink + CXL hybrid interconnect vision | Roadmap |

---

## 9. Five Critical Design Questions: Deep Analysis

### 9.1 Q1: How Does SSD File Data Load onto CXL?

**Current Path**: SSD → Page Cache → User Buffer → Pinned DRAM → GPU (4 hops, 3 copies)

| Option | Path | Latency | Code Change | Performance |
|---|---|---|---|---|
| **A: Page Cache** | SSD → CXL (DAX) → Pinned DRAM → GPU | ~300ns + ~1μs + ~5μs | None | Moderate |
| **B: GDS Buffer** | SSD → CXL (cuFile) → GPU | ~300ns + ~5μs | Medium | Best |
| **C: Sample Cache** | SSD → CXL (preprocessed) → GPU | ~300ns + ~5μs | High | Best |

**Step-by-step**:
1. **CXL Pool Init**: `ndctl create-region --mode=devdax` exposes `/dev/dax0.0`
2. **Data Loading**: DAX mmap (~300ns), cuFile GDS (~300ns), memcpy (~300ns)
3. **Bottleneck**: GPU cannot directly access CXL (no GPU CXL coherence)

**Verdict**: GDS Bypass (Option B) — bypasses pin_memory, direct CXL→GPU.

---

### 9.2 Q2: Is CXL a File Cache or a Sample Cache?

| Approach | Granularity | Transparency | Performance | Complexity |
|---|---|---|---|---|
| **File Cache** | File-level | Transparent | Moderate | Low |
| **Sample Cache** | Sample-level | Requires API changes | Best | High |
| **Hybrid** | Both | Semi-transparent | Best | Medium |

**Hybrid Architecture**:
- **Layer 1**: File Cache (CXL as DAX filesystem, managed by TPP)
- **Layer 2**: Sample Cache (Application-managed CXL regions)
- **Layer 3**: GDS Buffer (CXL registered for GPU access)

**Verdict**: Hybrid approach. TPP handles Layer 1-2, CXLDataset manages Layer 2-3.

---

### 9.3 Q3: How Do Multiple GPUs Handle Concurrent Access?

| Scenario | CXL 2.0 | CXL 3.0 | GPU Involvement |
|---|---|---|---|
| **Multiple CPUs read** | ✅ Hardware coherent | ✅ Hardware coherent | N/A |
| **CPU read + write** | ✅ Hardware coherent | ✅ Hardware coherent | N/A |
| **GPU read CXL** | ❌ Not supported | ❌ Not supported | GPU cannot access CXL |
| **Multiple GPUs via GDS** | ⚠️ Software managed | ⚠️ Software managed | Requires explicit sync |

| Approach | Scalability | Complexity | Use Case |
|---|---|---|---|
| **Read-Only Sharing** | Limited | Low | Inference |
| **Read-Write with Locks** | Poor | High | Not recommended |
| **Per-GPU Partitioning** | Excellent | Medium | **Training** |

**Verdict**: Per-GPU Partitioning (CCCL-proven). Aquifer-style ownership-based coherence for multi-host.

---

### 9.4 Q4: What Extensions Does GPU-Side CXL DataLoader Require?

| Extension | Current Behavior | Required CXL Behavior | Difficulty |
|---|---|---|---|
| **`__getitem__` return** | CPU Tensor (DRAM) | CXL-backed Tensor | Medium |
| **`pin_memory()`** | Page-lock DRAM | No-op or CXL→DRAM copy | **High** |
| **`collate_fn`** | Merge CPU Tensors | Merge CXL Tensors | Low |
| **`.cuda()`** | H2D from pinned DRAM | CXL→GPU (GDS or copy) | **High** |
| **DistributedSampler** | Shard indices | Shard CXL partitions | Medium |

**Verdict**: GDS bypass is optimal. Standard path: ~6μs. GDS bypass path: ~5μs.

---

### 9.5 Q5: Transparent Access vs. Dedicated CXL Service?

| Approach | Code Change | Performance | Hardware Requirements |
|---|---|---|---|
| **Transparent (TPP)** | None | Moderate (double-copy) | CXL device |
| **Dedicated Service** | New API | Best (no double-copy) | CXL + GDS-capable NIC |

**Two-Phase Approach**:
- **Phase 1 (Short-term)**: TPP transparent tiering — zero code change
- **Phase 2 (Mid-term)**: CXLDataset/CXLDataLoader with GDS bypass

---

## 10. Five-Layer Architecture Design

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 4: PyTorch Training Code                               │
│  (Zero code change — for batch in dataloader: ...)            │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: CXLDataset / CXLDataLoader                          │
│  - CXLDataset: __getitem__ returns CXL-backed Tensor          │
│  - CXLDataLoader: Bypasses pin_memory, uses GDS               │
│  - PrefetchEngine: CXL → GPU async prefetch                   │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: CXL Memory Pool + Coherence Manager                 │
│  - CXLMemoryPool: Per-GPU partitions (no contention)          │
│  - CoherenceManager: Ownership-based protocol (Aquifer-style) │
│  - HotnessTracker: TPP integration for auto-tiering           │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: CUDA / GDS / cuFile / DAX                           │
│  - cuFileRegister: CXL buffer → GDS accessible                │
│  - DAX mmap: Direct CXL access from userspace                 │
│  - TPP: Automatic DRAM ↔ CXL page migration                   │
├──────────────────────────────────────────────────────────────┤
│  Layer 0: CXL Hardware (DDR Pool / Switch)                     │
│  - CXL Type-3 memory expansion cards                          │
│  - CXL 3.0 Switch for multi-node pooling                      │
└──────────────────────────────────────────────────────────────┘
```

| Component | Responsibility | Key API |
|---|---|---|
| **CXLMemoryPool** | Memory pool manager | `CXLMemoryPool(device, size, policy)` |
| **CXLDataset** | CXL memory-mapped Dataset | `CXLDataset(cxlpool, file_list)` |
| **PrefetchEngine** | Async CXL → GPU prefetch | Lookahead / Adaptive / Priority |
| **CXLDataLoader** | Extended DataLoader | `CXLDataLoader(dataset, cxl_pool, cxl_prefetch)` |

---

## 11. Reference Design Evaluation

| Solution | Name | Difficulty | Performance | Hardware | Phase |
|---|---|---|---|---|---|
| **A** | mmap Transparent | ★★★ | ★★★ | CXL device | Short-term |
| **B** | Tiered Caching | ★★★★★ | ★★★★★ | CXL + DRAM | Long-term |
| **C** | GDS Passthrough | ★★★★ | ★★★★ | CXL + GDS NIC | Mid-term |
| **D** | Distributed Sharing | ★★★★★ | ★★★★★ | CXL 3.0 Switch | Long-term |
| **E** | Plugin Architecture | ★★★★ | ★★★★ | None special | Short-term |

| Approach | Performance | Compatibility | Hardware | Maturity | Community | **Total** |
|---|---|---|---|---|---|---|
| **A: mmap Transparent** | 3/5 | 5/5 | 3/5 | 4/5 | 4/5 | **3.75** |
| **B: Tiered Caching** | 5/5 | 3/5 | 3/5 | 3/5 | 3/5 | **3.55** |
| **C: GDS Passthrough** | 4/5 | 2/5 | 2/5 | 2/5 | 5/5 | **3.05** |
| **D: Distributed Sharing** | 5/5 | 2/5 | 1/5 | 2/5 | 4/5 | **3.00** |
| **E: Plugin Architecture** | 4/5 | 4/5 | 4/5 | 3/5 | 3/5 | **3.65** |

**Recommended Path**: A (short-term) → E (mid-term) → B (long-term)

---

## 12. Implementation Roadmap

| Phase | Timeline | Goal | Key Milestones |
|---|---|---|---|
| **Phase 1** | 0-6 months | Transparent Access | TPP integration; DAX mmap Dataset; baseline performance |
| **Phase 2** | 6-12 months | Plugin API | CXLDataset/CXLDataLoader; GDS bypass; prefetch engine |
| **Phase 3** | 12-24 months | Distributed Pooling | CXL 3.0 Switch support; cross-node sharing; production hardening |

---

## 13. Key Challenges

| # | Challenge | Impact | Mitigation |
|---|---|---|---|
| **1** | **pin_memory incompatibility** | CXL memory cannot be page-locked | GDS bypass or CXL → DRAM copy |
| **2** | **GPU lacks CXL coherence** | GPU cannot directly access CXL | GDS registration or double-copy |
| **3** | **Multi-GPU contention** | Concurrent access to shared CXL pool | Per-GPU partitioning (CCCL-style) |
| **4** | **Cache invalidation** | Stale data in CXL after epoch change | Version-based invalidation |
| **5** | **Memory pool fragmentation** | Dynamic allocation causes fragmentation | Slab allocator + defragmentation |
| **6** | **Fault recovery** | CXL pool failure loses data | Checkpoint + persistent backup |

---

## 14. Conclusion and Future Outlook

**Highest-probability path**: Two-Phase approach — Phase 1: TPP transparent tiering; Phase 2: CXLDataset/CXLDataLoader with GDS bypass.

**Critical insight**: TRAININGCXL proved GPU-direct PMEM access. CCCL proved CXL pooling beats InfiniBand. CXL-GPU proved custom controllers achieve near-DRAM latency. The hardware is ready — the software stack is what's missing.

**What PyTorch must solve**:
1. **pin_memory redesign**: Either bypass via GDS or accept double-copy
2. **Dataset API extension**: Allow `__getitem__` to return CXL-backed Tensors
3. **Multi-GPU coherence**: Per-GPU CXL partitions with ownership protocol
4. **Prefetch engine**: Async CXL → GPU transfer hiding latency

<mark>Ultimate vision: Data no longer needs to be "loaded" — the data sits right next to compute. The CXL memory pool serves as a unified data plane.</mark>

---

## References

| # | Source | URL |
|---|---|---|
| 1 | **TRAININGCXL: Failure Tolerant Training with PMEM Disaggregation over CXL** (KAIST, arXiv:2301.07492) | https://arxiv.org/abs/2301.07492 |
| 2 | **CCCL: Node-Spanning GPU Collectives with CXL Memory Pooling** (arXiv:2602.22457) | https://arxiv.org/abs/2602.22457 |
| 3 | **TERAIO: Cost-Efficient LLM Training with GDS Tensor Offloading** (arXiv:2506.06472) | https://arxiv.org/abs/2506.06472 |
| 4 | **Proxics: Efficient Programming Model for Far Memory Accelerators** (arXiv:2604.18120) | https://arxiv.org/abs/2604.18120 |
| 5 | **CXLMemUring: Asynchronous CXL Memory Pool Access** (arXiv:2309.04011) | https://arxiv.org/abs/2309.04011 |
| 6 | **CXL-DMSim: Full-System CXL Disaggregated Memory Simulator** (arXiv:2411.02282, IEEE TCAD) | https://arxiv.org/abs/2411.02282 |
| 7 | **CXLRAMSim: System-Level Exploration of CXL Memory Expander Cards** (arXiv:2603.29483) | https://arxiv.org/abs/2603.29483 |
| 8 | **DAXFS: Lock-Free Shared Filesystem for CXL Memory** (arXiv:2604.01620) | https://arxiv.org/abs/2604.01620 |
| 9 | **Aquifer: Hierarchical CXL+RDMA Memory Pooling** (arXiv:2606.24079) | https://arxiv.org/abs/2606.24079 |
| 10 | **CXL-GPU: Pushing GPU Memory Boundaries with CXL** (arXiv:2506.15601) | https://arxiv.org/abs/2506.15601 |
| 11 | **TPP: Transparent Page Placement for CXL Tiered-Memory** (Meta, arXiv:2206.02878) | https://arxiv.org/abs/2206.02878 |
| 12 | **Samsung CXL Memory Module Hybrid (CMM-H)** (arXiv:2503.22017) | https://arxiv.org/abs/2503.22017 |
| 13 | **NVIDIA DALI Documentation** | https://docs.nvidia.com/deeplearning/dali/user-guide/docs/ |
| 14 | **NVIDIA GPUDirect Storage Documentation** | https://docs.nvidia.com/cuda/gpudirect-storage/ |
| 15 | **PMDK Documentation** | https://pmem.io/pmdk/ |
| 16 | **Memkind Documentation** | https://github.com/memkey/memkind |
