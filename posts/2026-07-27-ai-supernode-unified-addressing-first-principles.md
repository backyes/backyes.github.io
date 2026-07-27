---
title: "First-Principles Thinking on AI Supernode Unified Addressing — From Hopper to NVL72, and AI Native Memory Fabric"
date: 2026-07-27
tags: ["unified-addressing", "nvlink", "hopper", "nvl72", "memory-fabric", "moe", "fusion-operator", "location-transparency", "ai-infra", "first-principles"]
excerpt: "Unified addressing is not about building a single address space spanning the entire supernode — it is about establishing location-transparent access semantics for the current compute task. Starting from fusion operators, rethinking Hopper's three-layer Fabric architecture."
---

# First-Principles Thinking on AI Supernode Unified Addressing

## A Mental Shift

When discussing unified addressing, we traditionally start from CUDA UVA or CPU virtual memory. But for AI supernodes, this misses the point.

Through deep analysis of the Hopper architecture, NVL72 topology, and [DeepEP's addressing design](deep-ep/deep_dive/addressing_deep_dive.html), I have come to realize that what truly drives unified addressing is not virtual memory — it is ==**the evolution of AI workloads**==.

---

## 1. Why Do AI Supernodes Need Unified Addressing?

### 1.1 Fusion Operators Have Changed the Nature of Communication

In the past, GPU kernels accessed local HBM, and inter-GPU communication was handled by NCCL and MPI. ==**Compute and communication were two distinct phases.**==

But today's AI algorithms increasingly adopt ==**Fusion Operator**== designs:

| Fusion Operator | Remote Access Pattern |
|---|---|
| **MoE Fusion Kernel** | Directly access parameters on different Expert GPUs |
| **Cross-GPU Attention** | Directly read KV Cache on remote GPUs |
| **Engram Memory / Embedding Cache** | Massive random remote access inside kernels

**Communication has begun to merge into compute itself.**

### 1.2 What Does a Fusion Operator Want to Express?

For a fusion operator, what it truly wants to express is:

```cpp
// Ideal: location-transparent access
float val = load(ptr);
```

Not:

```cpp
// Reality: explicit location breaks compute semantics
float val = load(gpu7, hbm2, offset);
```

And certainly not the degenerate path:

```
Fusion Kernel → Suspend → CPU Runtime → Address Lookup → Network → Resume Kernel
```

Because this means the GPU must exit its execution flow and let the CPU resolve addresses. This not only adds latency — more importantly, it ==**breaks the GPU pipeline**==, preventing remote memory loads from being issued as continuously as local loads.

### 1.3 First Principles: Location Transparency

Therefore, the real problem unified addressing solves is not "unified addresses" — it is:

> ==**Enabling GPUs to access remote memory as if it were local HBM.**==

In other words, it provides **Location Transparency**. The kernel should not care which GPU data resides on, which HBM stack it belongs to, or how many NVSwitches it traverses.

A unified addressing system essentially adds a layer of ==**Location Service**== between GPU and memory:

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

This maps to three independent questions:

| Question | Responsibility | Hopper Mapping |
|---|---|---|
| **Location** | Where is the object? | Control Fabric |
| **Translation** | Logical address → accessible location | Memory Fabric |
| **Transport** | How to deliver the request? | Transport Fabric |

---

## 2. How Does Hopper Implement Unified Addressing?

With the first-principles lens, we can see that NVIDIA did not put unified addressing into NVLink — instead, the system is divided into three layers with entirely distinct responsibilities:

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

### 2.1 Control Fabric: Manages Topology, Not Memory Pages

The Control Fabric handles GPU Discovery, Topology Discovery, Partition, Routing Programming, and Peer Registration. Fabric Manager and NVLSM maintain the *resource topology* of the entire NVL72 — they know where GPUs are and how switches are connected, but they know nothing about Tensors, KV Caches, or Experts.

### 2.2 Memory Fabric: Establishes Memory Semantics, No Control Logic

When a GPU kernel executes `load(ptr)`, the request enters the GMMU, traverses the TLB and Page Table to find the PTE. The PTE records the target Peer GPU and HBM Offset. ==**The Memory Fabric itself has no control logic — it is merely the execution result of the Control Fabric left on the GPU.**==

### 2.3 Transport Fabric: Pure Data Plane

NVLink and NVSwitch see only `(Destination GPU, HBM Offset, Payload)`. They have no knowledge of Virtual Addresses. ==**NVLink's responsibility has always been Transport, not Address Translation.**==

### 2.4 Complete Data Path

```
   Control Fabric
         │
   Establish Peer Mapping
         │
   Memory Fabric
   (GMMU / Aperture)
         │
   Resolve Target GPU
         │
   Transport Fabric
      (NVLink)
```

---

## 3. Hopper Server vs. NVL72: Two Scales of Addressing

| Dimension | Hopper Server (8 GPU) | NVL72 (72+ GPU) |
|---|---|---|
| **Control Scope** | Single node | Rack |
| **Control Plane** | CUDA Driver | Fabric Manager + NVLSM |
| **Memory Fabric** | Node-global Address | Rack-global Address |
| **Peer Mapping** | 8 peers | 72+ peers |
| **TLB Pressure** | Low | High |
| **Partition** | Not supported | Supported (MIG / Multi-tenant) |

In a Hopper Server, 8 GPUs make Peer Mapping tractable — establishing a Global Address Space at boot is reasonable.

But NVL72 is a ==**GPU Fabric**==, not a server. The Control Fabric must manage rack-wide multi-level topology, multiple OS Domains, Partitions, and fault recovery.

Both employ the same addressing strategy:

```
      Global Fabric
           │
   Global Memory Fabric
           │
  Global Address Space
```

The Control Fabric and Memory Fabric share the same scope — the entire NVL72.

---

## 4. Why Does Hopper Use Global Addressing?

This stems from a fundamental HPC assumption:

> In traditional HPC, any MPI Rank may communicate with any other Rank. The simplest software model is to establish a Global Address Space for the entire Fabric at once.

This is a classic ==**trade initialization cost for runtime efficiency**== design. Kernels always face unified pointers; no Page Table modifications or Peer Mapping re-establishment is needed at runtime.

For static clusters of dozens of GPUs, this is entirely reasonable.

---

## 5. AI Workloads Have Changed This Assumption

AI workloads no longer follow the "all GPUs may communicate" assumption:

- An Attention Kernel may only involve GPU4~GPU7
- A MoE Kernel may only access a few Expert GPUs
- A推理 instance may use only a fraction of NVL72

==**The fundamental communication unit of AI has shifted from the entire Fabric to the Communication Domain.**==

### 5.1 Analogy: RDMA Establishment

RDMA does not natively possess remote addresses. A RDMA Read can execute directly only because the control plane has already completed Memory Region registration, Queue Pair establishment, and address/permission exchange.

NVLink's unified addressing works identically: a GPU can execute `load(ptr)` not because NVLink natively knows remote addresses, but because the system has pre-established an ==**Address Resolution Service**==.

### 5.2 What Is Truly Necessary?

What is truly necessary is not a Global Address Space spanning the entire NVL72, but:

> ==**Establishing address-transparent capabilities for the current compute task before communication begins.**==

Global Address Space is merely *one* way to achieve Address Transparency — and a "luxurious" one.

Address transparency can be established at different moments:

| When | Scope | Use Case | Hopper Support |
|---|---|---|---|
| System boot | Entire NVL72 | HPC all-to-all | ✓ (current) |
| NCCL init | Communicator | Collective comms | Partial |
| Inference instance start | Instance-domain | Inference serving | ✗ |
| Fusion operator start | Communication Domain | MoE Fusion | ✗ |

> ==**What is truly indispensable is the Address Resolution Service, not the Global Address Space.**==

---

## 6. Hopper's Limitation Today

From this perspective, Hopper's real limitation is not in NVLink, nor in NVSwitch — it is in the ==**scope of the Memory Fabric**==.

Today, the Control Fabric manages the entire NVL72, and therefore the Memory Fabric also covers the entire NVL72. Even if a Fusion Kernel only accesses 4 GPUs, each GPU's GMMU still maintains Peer Mapping for the entire rack.

==**Fabric Scope and Address Scope are bound together.**==

| Cost Item | Current State | Problem |
|---|---|---|
| Page Table | Covers entire NVL72 | Mismatched with actual communication domain |
| TLB Miss | Global mapping causes pressure | Increased remote access latency |
| Peer Mapping | Established once at boot | Cannot be released on demand |
| Fabric Manager | Maintains global view | Scalability bottleneck |

---

## 7. Next: Memory Fabric from Static Resource to Dynamic Service

The future does not require overthrowing Hopper's three-layer architecture. The Control / Memory / Transport division remains sound. What must change is the ==**lifecycle of the Memory Fabric**==.

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
   (GPU can directly load/store)
             │
             ▼
     Transport Fabric
   (NVLink / UALink / UB)
```

What is truly dynamic is the **Address Resolution Service**. The Memory Fabric is merely the concrete manifestation of address-resolution relationships in GPU GMMUs and page tables.

### Evolution Path

| Phase | Addressing Object | Mapping | Serves |
|---|---|---|---|
| **CPU Era** | Virtual Page | VA → PA | Process |
| **Hopper Era** | Virtual Address | VA → GPU + HBM Offset | Entire Fabric |
| **AI Native Era** | Communication Domain | Domain → Address Domain | Compute task |
| **Future** | AI Object | Object → Location | Fusion Kernel |

In the future, unified addressing may evolve from `VA → GPU + Offset` to `AI Object → GPU + Offset`, building an ==**Object Directory Service**== around KV Blocks, Experts, and Embeddings. At that point, unified addressing will no longer be merely a memory management mechanism — it will become the core infrastructure of the AI Native Memory Fabric.

---

## 8. Summary: Unified Addressing Is Not Unified Addresses — It Is Unified Location

> ==**Unified addressing is not about building a single address space spanning the entire supernode — it is about establishing location-transparent access semantics for the current compute task, enabling GPUs to access remote memory as if it were local HBM.**==

Three core conclusions:

1. **First Principles**: The first principle of unified addressing is ==**Location Transparency**==, not Unified Address. The goal is to let kernels express remote access via `load(ptr)`.

2. **Hopper's Soundness and Limitation**: The three-layer Fabric architecture (Control / Memory / Transport) is sound, but the ==**binding of Fabric Scope and Address Scope**== is a legacy constraint.

3. **Dynamicization Is the Direction**: What is truly indispensable is the ==**Address Resolution Service**==, not the Global Address Space. When to establish it, at what scope, and who maintains it are all freely tradeable design parameters.

Hopper chooses to establish a Global Memory Fabric covering the entire NVL72 at system boot — an excellent engineering implementation for HPC workloads. But it is not the only implementation. For the future AI Native Supernode, the more reasonable direction is to dynamically establish the Address Resolution Service driven by the runtime, evolving unified addressing from "static rack-level resource" to "on-demand runtime capability."

---

## Further Reading

- [DeepEP Addressing Design — Adversarial Analysis](deep-ep/deep_dive/addressing_deep_dive.html) — Four-agent adversarial discussion
- [DeepEP Comprehensive Analysis Report](deep-ep/DeepEP_Final_Analysis_Report.html) — Three-perspective complete analysis
- [HBM / CXL / Memory Market Research](../hbm-cxl/report.html) — Memory hierarchy landscape
