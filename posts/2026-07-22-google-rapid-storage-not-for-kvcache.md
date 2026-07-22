---
title: "[Draft] How Google Thinks About KV Cache Storage — And Why It Matters"
date: 2026-07-22
tags: ["Google-Cloud", "Managed-Lustre", "LMCache", "KV-Cache", "AI-Inference", "Storage-Hierarchy"]
excerpt: "Google's KV Cache strategy isn't one solution — it's two complementary approaches: centralized Lustre for multi-tenant sharing, and node-local tiered storage (HBM + CPU RAM + Local SSD) for single-node efficiency. Both prove the same point: storage is the bottleneck."
---

# How Google Thinks About KV Cache Storage

## The question

At 42.7:1 storage:compute ratio [1], KV Cache *is* the workload. How does Google design the storage system around it?

Google's answer is not one solution — it's two complementary approaches, each optimized for different workload characteristics.

---

## Approach 1: Node-Local Tiered Storage (HBM + CPU RAM + Local SSD)

Google's collaboration with LMCache [11] demonstrates a tiered storage approach on GKE:

**Setup:** A3 mega node, 8× H100 Mega 80GB, Llama-3.3-70B-Instruct

| Tier | Capacity | Role |
|---|---|---|
| **HBM (Tier 1)** | 640 GiB | Fastest, smallest |
| **CPU RAM (Tier 2)** | 1 TiB | Mid-speed extension |
| **Local SSD (Tier 3)** | 5 TiB | Largest, slowest node-local |

**Results (4M–4.3M token cache, exceeding HBM):**

| Context | Best Setup | TTFT | Throughput | Latency |
|---|---|---|---|---|
| 5K | HBM + CPU RAM | −18% | +16% | −14% |
| 10K | HBM + CPU RAM | −44% | +50% | −33% |
| 50K | HBM + CPU RAM + SSD | −68% | +179% | −64% |
| 100K | HBM + CPU RAM + SSD | −79% | +264% | −73% |

**Key insight:** At 100K context, node-local tiered storage delivers ==2.6×== throughput and ==73%== latency reduction vs HBM-only. The cache stays on the node — no network round-trips.

**Limitation:** Node-local storage is fixed in size and cannot be shared across nodes.

---

## Approach 2: Centralized Shared Storage (Managed Lustre)

For workloads that exceed node-local capacity or require multi-tenant sharing, Google recommends Managed Lustre [2]:

> "We believe that Google Cloud Managed Lustre should be your primary storage solution for external KV Cache."

**Why not just local SSD?**

> "While Local SSDs and CPU RAM are effective node-local tiers, they are fixed in size and cannot be shared. Managed Lustre provides a parallel file system to act as the massive, high-throughput external storage."

Local SSDs are limited by host NIC bandwidth. Lustre provides higher aggregate bandwidth and centralized sharing via RDMA.

**Performance (DeepSeek-R1, A3-Ultra, 8× H200, 50K context, ~75% hit rate, ~3.4 TiB KV):**

| Metric | Result |
|---|---|
| Inference throughput | ==+75%== vs host memory |
| MTTF | ==−40%+== |
| TCO reduction | ==35%== (1M TPS workload) |
| Accelerator savings | ==~40%== fewer GPUs |

**Configuration:** 18 TiB Lustre per VM, 1000 MB/s per TiB, 73 VMs for 1M TPS, `o_direct` + 32 I/O threads.

---

## Rapid Storage: Not for KV Cache

| Product | Bandwidth | Use Case |
|---|---|---|
| Rapid Bucket | 15 TB/s | Training data, checkpoints |
| Rapid Cache | 2.5 TB/s | Model weight loading |

Rapid Storage accelerates training and cold start. For KV Cache, Google recommends Lustre or tiered node-local storage [1][2].

---

## Comparison

| | Node-Local Tiered (LMCache) | Centralized Lustre |
|---|---|---|
| Architecture | HBM → CPU RAM → Local SSD | Parallel file system (RDMA) |
| Sharing | Node-scoped | Multi-tenant |
| Best for | Single-node, latency-sensitive | Multi-node, capacity-bound |
| Throughput gain | ==+264%== (100K context) | ==+75%== (50K context) |
| Context ceiling | ~100K (node capacity) | Unlimited (shared pool) |

---

## The View

Google's two-pronged approach reflects a simple principle: **match the storage architecture to the workload.**

- Single-node, long-context, latency-sensitive → tiered node-local (HBM + CPU RAM + SSD)
- Multi-tenant, capacity-bound, shared contexts → centralized Lustre

Both prove the same point: at 42.7:1 storage:compute ratio, the storage layer determines inference cost and performance. The compute is almost an afterthought.

---

## References

[1] [backyes — The Million-Token Bill](https://backyes.github.io/posts/million-seq-storage-vs-compute.html) — 42.7:1 storage:compute ratio
[2] [Google Cloud Blog — Managed Lustre for KV Cache](https://cloud.google.com/blog/products/storage-data-transfer/choosing-google-cloud-managed-lustre-for-your-external-kv-cache) — 75% throughput, 35% TCO
[3] [Google Cloud Blog — Next '26 Storage](https://cloud.google.com/blog/products/storage-data-transfer/next26-storage-announcements) — Rapid Storage specs
[4] [Google Cloud Docs — Storage for AI](https://docs.cloud.google.com/ai-hypercomputer/docs/storage) — Tier recommendations
[11] [LMCache Blog — LMCache on GKE](https://blog.lmcache.ai/en/2025/10/07/lmcache-on-google-kubernetes-engine-boosting-llm-inference-performance-with-kv-cache-on-tiered-storage/) — 264% throughput at 100K context
