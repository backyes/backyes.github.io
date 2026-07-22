---
title: "[Draft] NVIDIA ICMS vs Google KV Cache Strategy — Two Philosophies for AI Memory Infrastructure"
date: 2026-07-22
tags: ["Google-Cloud", "Rapid-Storage", "Managed-Lustre", "NVIDIA-ICMS", "KV-Cache", "AI-Inference"]
excerpt: "NVIDIA ICMS puts storage at the bus edge. Google Lustre uses centralized shared storage. Two bets on where the KV Cache wall should be solved."
---

# NVIDIA ICMS vs Google KV Cache Strategy

NVIDIA's ICMS (Inference Context Memory Storage, aka CMX) is a pod-level flash tier for Vera Rubin [8][9]. BlueField-4 DPUs in dedicated enclosures, Spectrum-X RDMA fabric, ==5×== token throughput over traditional storage. The bet: push storage *toward* the GPU — solve the wall at the bus edge.

Google's bet is different. Managed Lustre puts KV Cache on a *centralized parallel file system*, accessed via RDMA from any node [2]. The wall is a *capacity and sharing* problem, not just a *proximity* problem.

Two philosophies. Both acknowledge the same reality: at 42.7:1 storage:compute ratio [1], KV Cache *is* the workload.

---

## 1. NVIDIA ICMS: Storage at the Bus Edge

| Component | Detail |
|---|---|
| Platform | Vera Rubin (2026) |
| Processors | BlueField-4 DPUs in flash enclosures |
| Fabric | Spectrum-X Ethernet, RDMA |
| Tier | G3.5 — between HBM (G3) and network storage (G4) |
| Performance | ==5×== tokens/sec vs traditional storage |
| Software | DOCA, Dynamo, NIXL for KV pre-staging |

**Philosophy:** KV Cache should sit as close to the GPU as possible — a memory bus-attached flash tier.

---

## 2. Google Managed Lustre: Shared Bandwidth

> "We believe that Google Cloud Managed Lustre should be your primary storage solution for external KV Cache." [2]

### Why not local SSD?

> "On GPUs, Lustre is assisted by locally attached SSDs. On TPUs, where local SSDs are not available, Lustre's role is even more central. While Local SSDs and CPU RAM are effective node-local tiers, they are fixed in size and cannot be shared." [2]

Local SSDs *can* be distributed, but host NIC bandwidth becomes the bottleneck. Lustre delivers higher aggregate bandwidth with centralized sharing.

### Performance (DeepSeek-R1, A3-Ultra, 8× H200, 50K context, ~75% hit rate, ~3.4 TiB KV) [2]:

| Metric | Result |
|---|---|
| Inference throughput | ==+75%== vs host memory only |
| MTTF | ==−40%+== |
| I/O parallelism | 32 worker threads |

### TCO (1M TPS, A3-Ultra + Lustre) [2]:

| Metric | Result |
|---|---|
| TCO reduction | ==35%== vs no external storage |
| Accelerator savings | ==~40%== fewer GPUs |

### Configuration [2]:

| Parameter | Value |
|---|---|
| Lustre bandwidth | 1000 MB/s per TiB |
| Capacity | 18 TiB per A3-Ultra VM |
| Cluster | 73 VMs for 1M TPS |
| Tuning | `o_direct`, 32 I/O threads |

**Thesis:** "Storage replaces compute" — offload KV to I/O, free accelerators for other work.

---

## 3. Rapid Storage: Not for KV Cache

| Product | Bandwidth | Use Case |
|---|---|---|
| Rapid Bucket | 15 TB/s | Training data, checkpoints |
| Rapid Cache | 2.5 TB/s | Model weight loading |

Rapid Storage accelerates training and cold start. For KV Cache, Google recommends Managed Lustre [1][2].

---

## 4. Comparison

| | NVIDIA ICMS | Google Lustre | Rapid Storage |
|---|---|---|---|
| Architecture | Bus-attached flash (DPU) | Centralized parallel FS (RDMA) | Object storage (Colossus) |
| Proximity | Pod-level | Cluster-level | Regional |
| Sharing | Pod-scoped | Multi-tenant | Zonal |
| KV role | Primary KV staging | External KV | Not for KV |
| Core bet | Proximity | Shared bandwidth | Training throughput |

---

## 5. The View

The NVIDIA-Google divergence is a question: **where to solve the wall?**

- NVIDIA: at the bus edge, storage close to GPU, DPU-managed.
- Google: in the network, centralized shared tier, software-optimized I/O.

No one-size-fits-all. Single-pod latency-sensitive → ICMS. Multi-tenant long-context shared → Lustre. Training data → Rapid Storage.

---

## References

[1] [backyes — The Million-Token Bill](https://backyes.github.io/posts/million-seq-storage-vs-compute.html) — 42.7:1 storage:compute ratio
[2] [Google Cloud Blog — Managed Lustre for KV Cache](https://cloud.google.com/blog/products/storage-data-transfer/choosing-google-cloud-managed-lustre-for-your-external-kv-cache) — 75% throughput, 35% TCO, 40% fewer GPUs
[3] [Google Cloud Blog — Next '26 Storage](https://cloud.google.com/blog/products/storage-data-transfer/next26-storage-announcements) — Rapid Storage specs
[4] [Google Cloud Docs — Storage for AI](https://docs.cloud.google.com/ai-hypercomputer/docs/storage) — Tier recommendations
[5] [Google Cloud Blog — Next '26 AI](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26) — Dedicated KV Cache subsystem
[6] [Google Cloud Blog — TPU 8t/8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) — 384 MB SRAM for KV
[7] [Blocks & Files](https://www.blocksandfiles.com/public-cloud/2026/04/22/googles-cloud-storage-gets-faster-and-smarter-for-ai/5218551) — Lustre as shared KV
[8] [NVIDIA — BlueField-4 CMX](https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/) — ICMS architecture
[9] [NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-bluefield-4-powers-new-class-of-ai-native-storage-infrastructure-for-the-next-frontier-of-ai) — ICMS launch
[10] [NVIDIA Investor Relations](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-BlueField-4-Powers-New-Class-of-AI-Native-Storage-Infrastructure-for-the-Next-Frontier-of-AI/default.aspx) — CMX branding
