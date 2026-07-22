---
title: "[Draft] NVIDIA ICMS vs Google's KV Cache Strategy — Two Philosophies for AI Memory Infrastructure"
date: 2026-07-22
tags: ["Google-Cloud", "Rapid-Storage", "Managed-Lustre", "NVIDIA-ICMS", "KV-Cache", "AI-Inference", "Storage-Hierarchy"]
excerpt: "NVIDIA's ICMS puts storage at the high-speed bus edge. Google's Managed Lustre takes a different path — centralized shared storage for KV Cache. Two approaches, two bets on where the memory wall should be solved."
---

# NVIDIA ICMS vs Google's KV Cache Strategy

## Two directions for AI memory

At CES 2026 and GTC 2026, NVIDIA announced **ICMS** (Inference Context Memory Storage, also branded **CMX**) — a pod-level, flash-backed context tier for the Vera Rubin platform [8][9]. The idea: put high-bandwidth storage *close* to the compute, bridging GPU HBM and general-purpose network storage. Powered by BlueField DPUs in dedicated flash enclosures, ICMS delivers up to ==5×== higher tokens-per-second and ==5×== power efficiency over traditional secondary storage, using Spectrum-X Ethernet for RDMA-accelerated transport [8].

NVIDIA's bet: the KV Cache bottleneck is best solved by pushing storage *toward* the accelerator — at the high-speed bus edge.

Google took a different path. Instead of placing storage at the bus edge, Google's Managed Lustre solution puts KV Cache on a *centralized, shared parallel file system* — accessed via RDMA from any node. The bet: KV Cache externalization is a *data sharing and capacity* problem, not just a *proximity* problem.

These two approaches represent fundamentally different views on where the memory wall should be solved.

---

## 1. NVIDIA ICMS: Storage at the Bus Edge

**Architecture:**

| Component | Detail |
|---|---|
| **Name** | ICMS / CMX (Context Memory Storage) |
| **Platform** | NVIDIA Vera Rubin (2026) |
| **Processors** | BlueField-4 DPUs in dedicated flash enclosures |
| **Fabric** | Spectrum-X Ethernet, RDMA-accelerated |
| **Tier** | G3.5 — between in-Pod HBM (G3) and off-Pod network storage (G4) |
| **Performance** | Up to 5× tokens/sec vs traditional secondary storage |
| **Software** | NVIDIA DOCA, Dynamo, NIXL for KV block pre-staging |

**The philosophy:** KV Cache should be staged *as close to the GPU as possible* — close enough to act as a memory extension, far enough to break the HBM capacity wall. ICMS is essentially a "memory bus-attached flash tier."

---

## 2. Google's Answer: Managed Lustre for External KV Cache

Google's official position: "We believe that Google Cloud Managed Lustre should be your primary storage solution for external KV Cache." [2]

The approach is fundamentally different from ICMS. Instead of placing storage at the bus edge, Lustre provides a *centralized, shared parallel file system* that any node can access via RDMA.

### 2.1 Why Lustre, Not Local SSD?

Google's blog [2] makes the case:

> "On GPUs, Lustre is assisted by locally attached SSDs. And on TPUs, where local SSDs are not available, Lustre's role is even more central."

> "While Local SSDs and CPU RAM are effective node-local tiers, they are fixed in size and cannot be shared. Managed Lustre provides a parallel file system to act as the massive, high-throughput external storage, making it a great solution for large-scale, multi-node, and multi-tenant AI inference workloads where the cache exceeds the capacity of the host machine."

The key insight: local SSDs *can* be shared in distributed setups, but their I/O bandwidth is limited by host NIC capacity. Lustre provides higher aggregate bandwidth and centralized sharing — critical for long-context workloads where KV Cache sizes reach terabytes.

### 2.2 Performance Data

Google's experiments (DeepSeek-R1 on A3-Ultra: 8× H200, 8× 141 GB HBM, 50K token context, ~75% cache hit rate, ~3.4 TiB KV Cache) [2]:

| Metric | Improvement |
|---|---|
| **Inference throughput** | ==+75%== vs host memory only |
| **Mean time to first token (MTTF)** | ==−40%+== vs host memory only |
| **I/O parallelism** | 32 worker threads for KV reads |

### 2.3 TCO Analysis: 35% Savings

For a workload processing 1 million TPS using A3-Ultra VMs + Managed Lustre [2]:

| Metric | Value |
|---|---|
| **TCO reduction** | ==35%== vs no external storage |
| **Accelerator savings** | ==~40%== fewer GPUs needed |
| **Key insight** | "Storage replaces compute" — offloading KV to I/O frees accelerators for other work |

### 2.4 Configuration Details

| Parameter | Value |
|---|---|
| **Lustre bandwidth** | 1000 MB/s per TiB (Performance Tier) |
| **Capacity per machine** | 18 TiB Lustre per A3-Ultra VM |
| **Cluster size** | 73 A3-Ultra machines for 1M TPS |
| **GPU** | 8× H200 per VM, 8× 141 GB HBM |
| **Software tuning** | `o_direct` flag, high I/O parallelism (32 threads) |

### 2.5 The Core Argument

> "Our experiment demonstrated that with configuration tuning and an improvement in KV Cache software to adopt more I/O parallelism, Managed Lustre can substantially improve inference performance."

> "You can handle a specific number of queries per second with ~40% fewer accelerators, resulting in direct cost savings."

Google's thesis: **I/O bandwidth is the bottleneck**, and parallel storage (Lustre) can deliver more aggregate bandwidth than node-local solutions when properly tuned. The storage cost is offset by compute savings.

---

## 3. Rapid Storage: What It Actually Does

Rapid Storage (Rapid Bucket + Rapid Cache) is Google's high-performance object storage layer — built on the Colossus distributed system that powers Gemini and YouTube [3].

| Product | Bandwidth | Latency | Primary Use |
|---|---|---|---|
| **Rapid Bucket** | 15 TB/s | Sub-ms (open file) | Training data, checkpoint write/restore |
| **Rapid Cache** | 2.5 TB/s | Sub-ms | Inference model weight loading (cold start) |

**Rapid Storage is not designed for KV Cache externalization.** Its strengths are:
- Accelerating training data loading (50% GPU blocked time reduction)
- Checkpoint write/restore (3.2× write, 5× restore speedup)
- Eliminating inference cold start (model weight pre-warming)

For KV Cache, Google's official recommendation remains **Managed Lustre** [1][2].

---

## 4. Comparing the Three Approaches

| Dimension | NVIDIA ICMS | Google Managed Lustre | Google Rapid Storage |
|---|---|---|---|
| **Architecture** | Bus-attached flash (DPU-powered) | Centralized parallel file system (RDMA) | Distributed object storage (Colossus) |
| **Proximity** | Pod-level (close to GPU) | Rack/cluster-level (network-accessed) | Regional |
| **Sharing** | Pod-scoped | Multi-node, multi-tenant | Zonal |
| **KV Cache role** | Primary tier for KV staging | Primary tier for external KV | Not recommended for KV |
| **Core bet** | Proximity solves the wall | Shared bandwidth solves the wall | Throughput solves training, not inference |
| **Best for** | Single-pod, latency-sensitive KV | Multi-tenant, long-context, shared KV | Training data, model weights |

---

## 5. What This Means for AI Infrastructure

The NVIDIA-Google divergence reflects a deeper question: **where should the KV Cache bottleneck be solved?**

- **NVIDIA says:** at the bus edge — push storage as close to the GPU as possible, use DPUs to manage the flash tier.
- **Google says:** in the network — provide a centralized, high-bandwidth shared tier that any node can access, and optimize I/O parallelism in software.

Both approaches acknowledge the same reality: at 42.7:1 storage:compute ratio [1], the KV Cache *is* the workload. The question is whether to solve this with proximity (ICMS) or with shared bandwidth (Lustre).

For infrastructure planners, the implication is clear: **there is no one-size-fits-all.** The optimal KV Cache architecture depends on workload characteristics — single-pod vs multi-tenant, latency-sensitive vs throughput-bound, shared vs isolated contexts.

---

## References

[1] [backyes — The Million-Token Bill](https://backyes.github.io/posts/million-seq-storage-vs-compute.html) — 42.7:1 storage:compute ratio, 77× cache hit cost difference

[2] [Google Cloud Blog — Choosing Managed Lustre for External KV Cache](https://cloud.google.com/blog/products/storage-data-transfer/choosing-google-cloud-managed-lustre-for-your-external-kv-cache) — 75% throughput improvement, 35% TCO savings, 40% fewer GPUs

[3] [Google Cloud Blog — Storage innovations at Next '26](https://cloud.google.com/blog/products/storage-data-transfer/next26-storage-announcements) — Rapid Storage official positioning and performance specs

[4] [Google Cloud Documentation — Storage services for AI and ML workloads](https://docs.cloud.google.com/ai-hypercomputer/docs/storage) — Official storage tier recommendations

[5] [Google Cloud Blog — AI infrastructure at Next '26](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26) — "Dedicated KV Cache scalable storage subsystem" announcement

[6] [Google Cloud Blog — TPU 8t/8i technical deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) — TPU 8i 384 MB SRAM designed for KV Cache

[7] [Blocks & Files — Google's cloud storage gets faster and smarter for AI](https://www.blocksandfiles.com/public-cloud/2026/04/22/googles-cloud-storage-gets-faster-and-smarter-for-ai/5218551) — Managed Lustre as shared KV Cache

[8] [NVIDIA Developer — Introducing BlueField-4-Powered CMX](https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/) — ICMS architecture, 5× performance, Spectrum-X fabric

[9] [NVIDIA Newsroom — BlueField-4 Powers AI-Native Storage](https://nvidianews.nvidia.com/news/nvidia-bluefield-4-powers-new-class-of-ai-native-storage-infrastructure-for-the-next-frontier-of-ai) — ICMS launch announcement

[10] [NVIDIA Investor Relations — BlueField-4](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-BlueField-4-Powers-New-Class-of-AI-Native-Storage-Infrastructure-for-the-Next-Frontier-of-AI/default.aspx) — ICMS/CMX branding

---

*The full research report (including all accessed/failed sources) is available at: [Google Rapid Storage Deep Research Report](/posts/research/google-rapid-storage/research_report.html)*
