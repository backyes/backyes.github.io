---
title: "When Do We Need a Global Address Service? — Beyond Programming Ease"
date: 2026-07-28
tags: ["Address-Service", "Unified-Addressing", "RDMA", "UVA", "Engram", "Parameter-Server", "KVCache", "Memory-Fabric", "MoE", "Kernel-Fusion", "Latency-Taxonomy", "AI-Infra"]
excerpt: "Unified addressing is often discussed as a programming-ease feature. But the real reason we need an independent, global address service is deeper: it is about amortizing the cost of establishing communication pipes, enabling data services at every latency tier to exchange data with lower overhead. From Redis to NVLink, the history of AI distributed systems is a history of address services evolving toward lower latency."
---

# When Do We Need a Global Address Service? — Beyond Programming Ease

We frequently discuss unified addressing services — CUDA UVA, POSIX shared memory (`shmem`), NVLink global address space — and the conversation almost always lands on the same point: *programming ease*. A single pointer, a flat address space, no explicit data movement. The developer doesn't need to know where data lives.

I've come to believe this framing is fundamentally limiting. Programming ease is a *benefit* of unified addressing, but it is not the *reason* we need an independent, global address service that outlives any single application. The real reason is about ==amortizing the cost of establishing and maintaining communication pipes== — and enabling data services at every latency tier to exchange data with lower overhead, higher IOPS, and better resource utilization.

To see why, let me walk through what "address service" actually means across the stack, then show you a concrete workload pattern — Engram — where the need for a global address service is not about ease at all, but about economics and physics.

---

## 1. An Address Service Taxonomy

Before arguing *why* we need a global address service, let me establish that address services already exist at every layer of the system — most of them invisible because we don't call them "address services."

### RDMA: Address as a Communication-Lifetime Service

A common misconception is that RDMA "natively" accesses remote memory. In reality, before any RDMA Read can execute, the control plane must complete: (1) Memory Region registration — pin the buffer, assign `lkey`/`rkey`; (2) Queue Pair establishment — create the communication channel; (3) Address and permission exchange — share virtual addresses, rkeys, and QP numbers between peers [1](#ref-1).

This is not unified addressing in the traditional sense. There is no single flat address space visible to the NIC. But it *is* an address service — a mapping from local virtual address to remote-accessible physical address, established at communication setup and torn down at communication teardown. ==The address service lifecycle follows the communication domain, not the application.== It requires no external third-party service. It is self-contained, scoped, and ephemeral.

As I analyzed in the [unified addressing first-principles post](ai-supernode-unified-addressing-first-principles.html#12-the-rdma-analogy-address-capability-as-a-contro), this reveals a fundamental point: **[address transparency is not an inherent property of the data plane — it is a capability established by the control plane before communication begins](ai-supernode-unified-addressing-first-principles.html#12-the-rdma-analogy-address-capability-as-a-contro)**.

### NVIDIA UVA: Two Kinds of Address Service

CUDA Unified Virtual Addressing (UVA), introduced in CUDA 4.0 (2011), provides a single virtual address space spanning host and device memory. But UVA actually encompasses two distinct address service models:

**Model A — Runtime Address Exchange (cudaMallocManaged / Managed Memory):** When you call `cudaMallocManaged`, the CUDA runtime coordinates between host and device. Page faults trigger migration. The runtime acts as a *per-allocation address broker* — it establishes mappings on demand, per buffer, per device. This is analogous to the RDMA model: address service scoped to the allocation.

**Model B — Boot-Time Unified Page Table (UVA with explicit device memory):** Once `cudaMalloc` returns a device pointer, that pointer is meaningful across all GPUs in the same UVA context. The system has established a *global page table* at initialization. This is a **third-party address service** — independent of any single allocation, independent of any single communication relationship. It outlives the individual `cudaMalloc` call. It persists for the lifetime of the CUDA context.

The difference matters. Model A is a convenience. Model B is infrastructure. When we talk about "needing a global address service," we are talking about Model B — an address resolution capability that exists independently of any single data transfer.

### Bus-Level Address Services: The Control Plane as Address Authority

Moving down the stack, bus protocols themselves embody address services. PCIe uses a flat address space with a memory management unit (IOMMU) that translates device-visible virtual addresses to physical addresses. CXL goes further: CXL 3.x introduces *routing-based address resolution* where a CXL switch (or fabric manager) maintains a mapping between a device's address window and the actual memory target [3](#ref-3).

As I discussed in the [AI Supernode Unified Addressing post](ai-supernode-unified-addressing-first-principles.html#8-address-resolution-service-who-owns-the-authorit), the bus control plane is fundamentally an address service: it maintains the authoritative mapping between "what address the requester sees" and "where the data physically lives." The bus doesn't just transport data — it *resolves* addresses.

This pattern repeats at every scale:

| Layer | Address Service | Scope | Lifetime |
|-------|----------------|-------|----------|
| **RDMA QP** | Memory Region + rkey exchange | Per peer pair | Per connection |
| **CUDA UVA** | Unified page table | Per CUDA context | Per context |
| **PCIe IOMMU** | ATS/PRI translation | Per device | Per device assignment |
| **CXL Fabric** | Routing-based address resolution | Per fabric | Per fabric initialization |
| **NVLink/NVSwitch** | Peer mapping + aperture | Per fabric partition | Per Fabric Manager session |

==At every layer, there is an address service. The question is never "do we need an address service" — we always do. The question is: at what scope, at what lifetime, and at what latency cost should it operate?==

---

## 2. Is Address Service Only About Programming Ease?

The standard narrative: unified addressing exists so developers don't have to manually manage data movement. True, but insufficient. Let me show you a workload where the need for a global address service is driven not by programming ease, but by *economics and physics*.

### The Engram Pattern: Five Contradictory Requirements

When analyzing Engram-style memory services (a term I use for the disaggregated, medium-bandwidth, fine-grained random-access memory tier sitting between HBM and SSD), five characteristics emerge simultaneously:

| Property | Requirement | Implication |
|----------|-------------|-------------|
| **Capacity** | Large (100 GB – 10 TB per instance) | Cannot fit entirely in HBM |
| **Bandwidth** | Low (relative to HBM) | DDR is "good enough" — but this means DDR capacity is underutilized |
| **Granularity** | Fine (80 B – 1 KB random access) | Cannot amortize with large transfers |
| **IOPS** | Very high (millions of QPS in low-latency inference) | Each access must be sub-microsecond |
| **Latency** | Strict (tens to hundreds of µs for multi-MB data transfer) | Rules out SSD/NAND; DDR is the only medium |

The contradiction: **low latency + high IOPS forces DDR** (NAND/SSD cannot deliver millions of sub-microsecond random accesses). But **low bandwidth means the DDR's full capacity is never utilized** — you're paying for DDR's bandwidth capability while only using its capacity and random-access capability.

> ==DDR is being used in the worst possible way: paying for bandwidth and capacity, but only exploiting its low-latency characteristic.==

### The Sharing Insight from Recommendation Systems

In recommendation inference, there is a classic pattern: the **embedding service** is shared across multiple model serving instances. A single embedding table (often 100 GB – 50 TB) is too large to replicate per instance, but too hot to centralize on one node. The solution: a shared embedding service, accessed remotely by all inference instances [4](#ref-4)[5](#ref-5).

The key architectural decision: the embedding service has its *own address space*, independent of any single inference instance. Inference instances don't embed the table locally — they issue remote lookups against a shared address domain. The address service (mapping from embedding ID to physical location) lives in the embedding service, not in the inference runtime.

Now apply the same pattern to Engram. If Engram's characteristics are large capacity + low bandwidth + fine granularity + high IOPS + low latency, then:

1. **Replicating Engram per inference instance wastes DDR** (the low-bandwidth characteristic means each instance uses only a fraction of available DDR bandwidth).
2. **Sharing Engram across instances amortizes DDR cost** — the same DDR serves more workload, improving utilization.
3. **Sharing requires an address service independent of any single inference instance** — because the Engram service must be addressable by multiple, ephemeral, independently-scheduled inference workloads.

This is the crux: ==when you need to share a memory tier across multiple independent compute domains, you need an address service that is itself independent — a third party that outlives any single compute domain.==

### What Does This Independent Address Service Look Like?

In traditional distributed systems, this pattern is well-established:

- **Parameter Server** (Chen et al., 2015 [6](#ref-6); Li et al., 2014 [7](#ref-7)): The parameter server maintains the authoritative mapping from parameter key to value location. Workers don't know where parameters live — they issue `pull(key)` requests, and the parameter server resolves the address. The parameter server *is* the address service, independent of any single worker.

- **KVCache Memory Pool** (vLLM's PagedAttention [8](#ref-8), LMCache [9](#ref-9)): The KVCache manager maintains the mapping from `(request_id, layer, block_index)` to physical memory location. The attention kernel doesn't know where the KV block lives — it goes through the block table. The block table *is* the address service, independent of any single attention kernel invocation.

- **Unified Addressing (NVLink, UALink, CXL)**: The fabric address resolution mechanism maintains the mapping from virtual address to physical location across the domain. The GPU kernel issues `load(ptr)`, and the fabric resolves it. The fabric address service *is* the address service, independent of any single kernel.

The pattern is identical. The only differences are scope, lifetime, and latency.

---

## 3. Recalling Past "Address Services" in AI Systems

If we define an address service as "the mechanism by which a compute entity finds where data lives," then the history of AI distributed systems is a history of address services evolving toward lower latency. Each tier solves the same problem — "how do I find the data?" — at a different point on the latency spectrum.

<a id="ref-10"></a>Let me lay out the full taxonomy:

| Tier | Address Service | Typical Latency | Use Case | Sharing Model |
|------|----------------|-----------------|----------|---------------|
| **a. Redis KV** | Feature store (key → value lookup) | Milliseconds (1–10 ms) | Recommendation feature serving | Multi-instance shared |
| **b. Parameter Server (Training)** | Key → parameter shard mapping | Milliseconds (1–10 ms) | Distributed training parameter storage | Multi-worker shared |
| **c. Parameter Server (Serving)** | Embedding ID → embedding vector | Milliseconds (1–10 ms) | Recommendation online serving, embedding sharing across inference instances | Multi-instance shared |
| **d. RDMA/ROCE** | Virtual address + rkey → remote physical | Hundreds of µs (100–500 µs) | Remote memory access, storage disaggregation | Per connection |
| **e. GPU Direct (RDMA/IB/GDR)** | GPU-direct remote memory access | Hundreds of µs (50–200 µs) | Kernel fusion across nodes, compute-communication overlap | Per communicator |
| **f. GPU Bus (NVLink, UB, PCIe P2P)** | Hardware page table → peer HBM | Tens of µs (1–50 µs) | Kernel-level distributed computing: MoE expert dispatch, fine-grained fusion, remote table lookup | Per fabric domain |

### Reading the Taxonomy

Three observations:

**1. Every tier is an address service.** Redis resolves keys to values. Parameter servers resolve parameter IDs to shard locations. RDMA resolves virtual addresses to remote physical pages. NVLink resolves virtual addresses to peer HBM offsets. They are the same abstraction at different latencies.

**2. Lower latency enables finer-grained sharing.** At millisecond latency, you share *tables* (entire embedding tables, parameter shards). At hundred-microsecond latency, you share *buffers* (activation buffers, gradient buffers). At ten-microsecond latency, you share *individual cache lines and parameters inside a running kernel*. The address service latency determines the granularity of what can be shared.

**3. Sharing drives the need for independence.** In tiers (a), (b), (c), the address service is *necessarily* independent of any single compute instance — because the whole point is sharing. In tiers (d), (e), (f), the address service can be either coupled (per-connection RDMA) or independent (NVLink global address space). ==The trend is toward independence as latency decreases==, because lower latency enables more instances to share the same resource, which requires a more durable address service.

### The Latency Amortization Argument

Here is the core insight I want to leave you with:

> ==A global address service is not primarily about programming ease. It is about amortizing the cost of establishing communication pipes.==

Every time two compute entities need to exchange data, they must establish *some* form of address mapping. In RDMA, this is QP setup + MR registration + address exchange — microseconds to milliseconds of control-plane overhead. In NVLink, this is peer mapping establishment at Fabric Manager initialization — milliseconds at boot, then free at runtime.

If the address service is *per-connection* (RDMA model), you pay the setup cost per communication relationship. If the address service is *global* (NVLink model), you pay the setup cost once, amortized across all future communication within that domain.

The higher the communication frequency, the more you amortize. The lower the communication latency, the more frequently you communicate, and the more you benefit from a pre-established global address service.

This is why NVLink's global address space exists: not because NVIDIA wanted to make programming easier (though it does), but because ==at 10 µs latency, you are communicating so frequently that per-connection address setup would dominate your overhead==. The global address service is a performance optimization disguised as a programming convenience.

---

## 4. Bringing It Back to Engram

Now we can frame the Engram question precisely:

**The problem:** Engram needs DDR (for latency + IOPS), but DDR bandwidth is underutilized (because Engram's bandwidth requirement is low). DDR cost is high. DDR power is significant.

**The solution:** Share Engram across multiple inference instances. Amortize DDR cost. Improve utilization.

**The requirement:** A shared Engram tier needs an address service that is:
- **Independent** of any single inference instance (because instances come and go, scale up and down)
- **Low-latency** (because Engram's access pattern is fine-grained random access at millions of QPS)
- **High-IOPS** (because the address resolution itself must not become the bottleneck)

**The architectural options:**
- *Option A*: Per-instance Engram with explicit data movement. Simple, but no sharing, no DDR amortization.
- *Option B*: Shared Engram with a global address service (RDMA-based, CXL-based, or NVLink-based depending on latency requirements). Complex, but DDR cost amortized.

==The decision between A and B is not about programming ease. It is about whether the workload's DDR economics justify the complexity of a global address service.== For large-scale inference serving where DDR cost dominates, the answer is increasingly yes.

---

## 5. Conclusion

The history of AI distributed systems is address services pushing down the latency stack — each order-of-magnitude in latency unlocks a finer sharing granularity:

| Latency | Address Service | Sharing Granularity |
|---------|----------------|---------------------|
| **ms** | Parameter Server, Redis | Tables, embedding shards |
| **100 µs** | RDMA, GPU Direct | Buffers, activation tensors |
| **10 µs** | NVLink, UB | Cache lines, kernel parameters |

At each step, the address service becomes more independent, more global, and more deeply integrated into the hardware. And at each step, the justification shifts: it starts as a *necessity* (you must share, so you need an address service), becomes a *performance optimization* (pre-established pipes amortize setup cost), and ends as a *programming convenience* (the address service is so transparent that you forget it exists).

But the fundamental driver is not convenience. It is this: ==when you need to share a resource across multiple independent compute domains, and the resource is accessed frequently enough that per-access setup cost matters, you need an address service that outlives any single domain.==

The question is never "do we need an address service." We always do. The question is: at what latency tier, at what scope, and with what sharing model?

---

*Analysis based on publicly available technical documentation and architectural patterns. Views are my own. Not investment advice.*

---

*© 2026 backyes · Created by backyes*

### References

<a id="ref-1"></a>[1](#ref-1) [RDMA Aware Networks Programming User Manual — NVIDIA Docs](https://docs.nvidia.com/networking/display/rdmaawarenetworksprogrammingv120) — Memory Region registration, QP establishment, and address exchange protocol for RDMA.

<a id="ref-2"></a>[2](#ref-2) [First-Principles Thinking on AI Supernode Unified Addressing — backyes](ai-supernode-unified-addressing-first-principles.html) — Analysis of RDMA as control-plane address service, and the scope-vs-lifetime design space for address transparency.

<a id="ref-3"></a>[3](#ref-3) [CXL 3.0 Specification — Compute Express Link Consortium](https://computeexpresslink.org/) — CXL routing-based address resolution and fabric manager address service model.

<a id="ref-4"></a>[4](#ref-4) [DeepRec: An Industrial-Scale Recommendation Engine — Alibaba](https://arxiv.org/abs/2409.12415) — Embedding service architecture for recommendation inference with shared embedding tables.

<a id="ref-5"></a>[5](#ref-5) [NVIDIA Merlin HugeCTR — Distributed Embedding](https://github.com/NVIDIA-Merlin/HugeCTR) — Distributed embedding table serving for recommendation systems with parameter server pattern.

<a id="ref-6"></a>[6](#ref-6) [Communication Efficient Distributed Stochastic Gradient Descent — Li et al., 2014](https://arxiv.org/abs/1409.4066) — Parameter server architecture for distributed training with key-addressable parameter storage.

<a id="ref-7"></a>[7](#ref-7) [Scaling Distributed Machine Learning with the Parameter Server — Li et al., 2014](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf) — OSDI'14 paper defining the parameter server as an independent address service for distributed ML.

<a id="ref-8"></a>[8](#ref-8) [Efficient Memory Management for Large Language Model Serving with PagedAttention — Kwon et al., SOSP'23](https://arxiv.org/abs/2309.17453) — vLLM's PagedAttention: block table as address service for KVCache memory management.

<a id="ref-9"></a>[9](#ref-9) [LMCache: An Efficient KV Cache Layer for Large Language Models](https://github.com/LMCache/LMCache) — KVCache disaggregation and sharing across inference instances via independent address service.

<a id="ref-10"></a>[10](#ref-10) [NVIDIA NVLink and NVSwitch Architecture — NVIDIA Developer](https://developer.nvidia.com/blog/nvidia-nvlink-the-scale-up-network-for-ai-factories/) — Hardware-level address service via NVLink peer mapping and global address space.
