---
title: "[Draft] How Google Thinks About KV Cache Storage — And Why It Matters"
date: 2026-07-22
tags: ["Google-Cloud", "Managed-Lustre", "LMCache", "Rapid-Storage", "KV-Cache", "AI-Inference", "Storage-Hierarchy"]
excerpt: "Google's KV Cache strategy isn't one solution — it's two complementary approaches: node-local tiered storage (HBM + CPU RAM + Local SSD) for single-node efficiency, and centralized Lustre for multi-tenant sharing. Both prove the same point: storage is the bottleneck."
---

# How Google Thinks About KV Cache Storage

## The question

At 42.7:1 storage:compute ratio [1], KV Cache *is* the workload. How does Google design the storage system around it?

Google's answer is not one solution — it's two complementary approaches, each optimized for different workload characteristics.

---

## Approach 1: Node-Local Tiered Storage (HBM + CPU RAM + Local SSD)

Google's collaboration with LMCache [11] demonstrates a tiered storage approach on GKE. The idea: extend KV Cache from GPU HBM to larger, cost-effective node-local tiers.

**Setup:** A3 mega node, 8× H100 Mega 80GB, Llama-3.3-70B-Instruct, LMCache v0.3.3

| Tier | Capacity | Role |
|---|---|---|
| **HBM (Tier 1)** | 640 GiB | Fastest, smallest |
| **CPU RAM (Tier 2)** | 1 TiB | Mid-speed extension |
| **Local SSD (Tier 3)** | 5 TiB | Largest, slowest node-local |

**Results (4M–4.3M token cache, exceeding HBM capacity):**

| Context | Best Setup | TTFT | Throughput | Latency |
|---|---|---|---|---|
| 1K | HBM only | Baseline | Baseline | Baseline |
| 5K | HBM + CPU RAM | −18% | +16% | −14% |
| 10K | HBM + CPU RAM | −44% | +50% | −33% |
| 50K | HBM + CPU RAM + SSD | −68% | +179% | −64% |
| 100K | HBM + CPU RAM + SSD | −79% | +264% | −73% |

**At 100K context:** node-local tiered storage delivers ==2.6×== throughput and ==73%== latency reduction vs HBM-only. The cache stays on the node — no network round-trips.

**Larger cache (12.6M–13.7M tokens, saturating HBM + CPU RAM):**

| Context | Best Setup | TTFT | Throughput | Latency |
|---|---|---|---|---|
| 10K | HBM + CPU RAM + SSD | +121% | +23% | −19% |
| 50K | HBM + CPU RAM + SSD | +48% | +69% | −41% |
| 100K | HBM + CPU RAM + SSD | −3% | +130% | −57% |

**Key insight:** At 100K context with large cache, tiered storage still delivers ==+130%== throughput despite SSD spillover. But TTFT degrades when cache exceeds comfortable capacity — signaling the limit of node-local approaches.

**Limitation:** Node-local storage is fixed in size and cannot be shared across nodes.

---

## Approach 2: Centralized Shared Storage (Managed Lustre)

For workloads that exceed node-local capacity or require multi-tenant sharing, Google recommends Managed Lustre [2]:

> "We believe that Google Cloud Managed Lustre should be your primary storage solution for external KV Cache."

**Why not just local SSD?**

> "On GPUs, Lustre is assisted by locally attached SSDs. And on TPUs, where local SSDs are not available, Lustre's role is even more central. While Local SSDs and CPU RAM are effective node-local tiers, they are fixed in size and cannot be shared. Managed Lustre provides a parallel file system to act as the massive, high-throughput external storage, making it a great solution for large-scale, multi-node, and multi-tenant AI inference workloads where the cache exceeds the capacity of the host machine."

Local SSDs *can* be distributed, but host NIC bandwidth becomes the bottleneck. Lustre provides higher aggregate bandwidth and centralized sharing via RDMA.

**Performance (DeepSeek-R1, A3-Ultra, 8× H200, 8× 141 GB HBM, 50K context, ~75% hit rate, ~3.4 TiB KV):**

| Metric | Result |
|---|---|
| Inference throughput | ==+75%== vs host memory only |
| MTTF | ==−40%+== vs host memory only |
| I/O parallelism | 32 worker threads for KV reads |

**TCO Analysis (1M TPS, A3-Ultra + Lustre):**

| Metric | Result |
|---|---|
| TCO reduction | ==35%== vs no external storage |
| Accelerator savings | ==~40%== fewer GPUs needed |
| Key insight | "Storage replaces compute" — offloading KV to I/O frees accelerators |

**Configuration:**

| Parameter | Value |
|---|---|
| Lustre bandwidth | 1000 MB/s per TiB (Performance Tier) |
| Capacity per machine | 18 TiB Lustre per A3-Ultra VM |
| Cluster size | 73 A3-Ultra machines for 1M TPS |
| GPU | 8× H200 per VM, 8× 141 GB HBM |
| Software tuning | `o_direct` flag, 32 I/O threads |

**Google's thesis:** "Our experiment demonstrated that with configuration tuning and an improvement in KV Cache software to adopt more I/O parallelism, Managed Lustre can substantially improve inference performance. You can handle a specific number of queries per second with ~40% fewer accelerators, resulting in direct cost savings."

---

## Rapid Storage: What It Actually Does

| Product | Bandwidth | Latency | Primary Use |
|---|---|---|---|
| Rapid Bucket | 15 TB/s | Sub-ms (open file) | Training data, checkpoint write/restore |
| Rapid Cache | 2.5 TB/s | Sub-ms | Inference model weight loading (cold start) |

**Rapid Storage is not designed for KV Cache externalization.** Its strengths:
- Accelerating training data loading (50% GPU blocked time reduction)
- Checkpoint write/restore (3.2× write, 5× restore speedup)
- Eliminating inference cold start (model weight pre-warming)

For KV Cache, Google's official recommendation remains **Managed Lustre** or **node-local tiered storage** [1][2][11].

---

## Comparing the Three Approaches

| | Node-Local Tiered (LMCache) | Centralized Lustre | Rapid Storage |
|---|---|---|---|
| **Architecture** | HBM → CPU RAM → Local SSD | Parallel file system (RDMA) | Object storage (Colossus) |
| **Proximity** | Node-local | Cluster-level (network) | Regional |
| **Sharing** | Node-scoped | Multi-tenant | Zonal |
| **KV Cache role** | Primary (single-node) | Primary (multi-node) | Not for KV |
| **Throughput gain** | ==+264%== (100K context) | ==+75%== (50K context) | N/A |
| **Context ceiling** | ~100K (node capacity) | Unlimited (shared pool) | N/A |
| **Best for** | Single-node, latency-sensitive | Multi-tenant, capacity-bound | Training data, model weights |

---

## The View

Google's two-pronged approach reflects a simple principle: **match the storage architecture to the workload.**

- **Single-node, long-context, latency-sensitive** → tiered node-local (HBM + CPU RAM + SSD)
- **Multi-tenant, capacity-bound, shared contexts** → centralized Lustre
- **Training data and model weights** → Rapid Storage

Both KV Cache approaches prove the same point: at 42.7:1 storage:compute ratio, the storage layer determines inference cost and performance. The compute is almost an afterthought.

---

## References

[1] [backyes — The Million-Token Bill](https://backyes.github.io/posts/million-seq-storage-vs-compute.html) — 42.7:1 storage:compute ratio, 77× cache hit cost difference
[2] [Google Cloud Blog — Managed Lustre for KV Cache](https://cloud.google.com/blog/products/storage-data-transfer/choosing-google-cloud-managed-lustre-for-your-external-kv-cache) — 75% throughput, 35% TCO, 40% fewer GPUs
[3] [Google Cloud Blog — Next '26 Storage](https://cloud.google.com/blog/products/storage-data-transfer/next26-storage-announcements) — Rapid Storage specs
[4] [Google Cloud Docs — Storage for AI](https://docs.cloud.google.com/ai-hypercomputer/docs/storage) — Tier recommendations
[5] [Google Cloud Blog — Next '26 AI](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26) — Dedicated KV Cache subsystem
[6] [Google Cloud Blog — TPU 8t/8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) — 384 MB SRAM for KV
[7] [Blocks & Files](https://www.blocksandfiles.com/public-cloud/2026/04/22/googles-cloud-storage-gets-faster-and-smarter-for-ai/5218551) — Lustre as shared KV
[11] [LMCache Blog — LMCache on GKE](https://blog.lmcache.ai/en/2025/10/07/lmcache-on-google-kubernetes-engine-boosting-llm-inference-performance-with-kv-cache-on-tiered-storage/) — 264% throughput at 100K context
