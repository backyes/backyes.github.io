---
title: "Google Chooses Lustre on SSD for KV Cache: TTFT, Throughput, and Latency Analysis — Beyond Local SSD"
date: 2026-07-22
tags: ["Google-Cloud", "Managed-Lustre", "LMCache", "Rapid-Storage", "KV-Cache", "AI-Inference", "Storage-Hierarchy"]
excerpt: "Google's KV Cache strategy isn't one solution — it's two complementary approaches: node-local tiered storage (HBM + CPU RAM + Local SSD) for single-node efficiency, and centralized Lustre for multi-tenant sharing. Both prove the same point: storage is the bottleneck."
---

# Google Chooses Lustre on SSD for KV Cache: TTFT, Throughput, and Latency Analysis — Beyond Local SSD

## NVIDIA ICMS: Storage at the Bus Edge

At CES/GTC 2026, NVIDIA announced **ICMS** (Inference Context Memory Storage, aka **CMX**) — a pod-level, flash-backed KV Cache tier for the Vera Rubin platform [8][9]. BlueField-4 DPUs in dedicated flash enclosures, Spectrum-X Ethernet for RDMA transport, delivering ==5×== token throughput over traditional secondary storage [8]. The bet: push storage *toward* the GPU — solve the memory wall at the bus edge.

ICMS is essentially a G3.5 tier: faster than network storage (G4), slower than HBM (G3), managed by DOCA/Dynamo/NIXL for KV block pre-staging.

## How Google Thinks About Long-Context KV Cache Systems

Our previous analysis found that ==99.1%== of tokens in a long-context Agent session are "remembered," not "computed" [1]. The storage:compute ratio reaches ==42.7:1==. As Google's own blog puts it: "The KV Cache's immense GPU memory footprint, amplified by system prompt size, is the primary bottleneck in LLM serving, directly constraining context length, concurrency, and overall system throughput." [2]

This is Google's concern: at 42.7:1 storage:compute ratio [1], KV Cache *is* the workload. Compute is almost an afterthought.

How does Google design the storage system around it? The answer is not one solution — it's two complementary approaches, each optimized for different workload characteristics.

---

## Approach 1: Node-Local Tiered Storage (HBM + CPU RAM + Local SSD)

Google's collaboration with LMCache [11] demonstrates a tiered storage approach on GKE. The core idea: extend KV Cache from GPU HBM to larger, cost-effective node-local tiers, keeping all data on the accelerator node.

**Experiment setup:** A3 mega node, 8× H100 Mega 80GB GPUs, Llama-3.3-70B-Instruct, LMCache v0.3.3, vLLM serving.

| Tier | Capacity | Role |
|---|---|---|
| **HBM (Tier 1)** | 640 GiB | Fastest, smallest — fits ~1.1M–1.3M tokens |
| **CPU RAM (Tier 2)** | 1 TiB | Mid-speed extension — total ~4M–4.3M tokens |
| **Local SSD (Tier 3)** | 5 TiB | Largest, slowest node-local — total ~12M–13M tokens |

**Benchmark command:**

> `python3 sglang/bench_serving.py --host=${IP} --port=${PORT} --dataset-name='generated-shared-prefix' --model=$MODEL --tokenizer=$MODEL --backend=vllm --gsp-num-groups=80 --gsp-prompts-per-group=20 --gsp-system-prompt-len=1000 --gsp-question-len=256 --gsp-output-len=512 --request-rate=800 --max-concurrency=200`

**Test 1: Cache fits in HBM (1.1M–1.3M tokens).** Adding slower storage tiers offered no advantage. HBM-only is optimal.

**Test 2: Cache exceeds HBM (4.0M–4.3M tokens).**

![LMCache Test 2 — TTFT, Throughput, Latency vs Context](https://backyes.github.io/posts/assets/lmcache_test2.png)

*Test 2: A3 mega, 8× H100, Llama-3.3-70B. Cache exceeds HBM (640 GiB) but fits within HBM + CPU RAM (1.7 TiB total). TTFT blue, Throughput green, Latency amber.*

| Context | Best Setup | TTFT | Throughput | Latency |
|---|---|---|---|---|
| 1K | HBM only | Baseline | Baseline | Baseline |
| 5K | HBM + CPU RAM | −18% | +16% | −14% |
| 10K | HBM + CPU RAM | −44% | +50% | −33% |
| 50K | HBM + CPU RAM + SSD | −68% | +179% | −64% |
| 100K | HBM + CPU RAM + SSD | −79% | +264% | −73% |

At 100K context, node-local tiered storage delivers ==2.6×== throughput (==+264%==) and ==73%== latency reduction vs HBM-only. The cache stays on the node — no network round-trips.

---

**Test 3: Cache saturates HBM + CPU RAM (12.6M–13.7M tokens), spills to Local SSD.**

![LMCache Test 3 — TTFT, Throughput, Latency vs Context](https://backyes.github.io/posts/assets/lmcache_test3.png)

*Test 3: Same setup. Cache saturates HBM + CPU RAM, spills to Local SSD (5 TiB). Throughput still gains, but TTFT degrades at 10K–50K — the cost of SSD spillover.*

| Context | Best Setup | TTFT | Throughput | Latency |
|---|---|---|---|---|
| 1K | HBM + CPU RAM | +5% | +1% | −1% |
| 5K | HBM + CPU RAM | −6% | +27% | −21% |
| 10K | HBM + CPU RAM + SSD | +121% | +23% | −19% |
| 50K | HBM + CPU RAM + SSD | +48% | +69% | −41% |
| 100K | HBM + CPU RAM + SSD | −3% | +130% | −57% |

Even with SSD spillover, 100K context still delivers ==+130%== throughput. But TTFT degrades when cache exceeds comfortable capacity — signaling the practical limit of node-local approaches.

> **🔍 Open question: Why does TTTT degrade at 10K–50K but recover at 100K, while throughput keeps improving?**
>
> **One hypothesis (worth discussing):** At 10K–50K, the cache is in a "worst of both worlds" regime — too large for fast memory, too small to amortize SSD I/O costs. By 100K, the workload may reach a new steady state where massive cache hit rates and batch efficiency outweigh the SSD latency penalty. But this is speculation — the actual cause could be vLLM scheduling behavior, SSD read-ahead effects, or KV Cache eviction dynamics. The data raises the question; it doesn't answer it.

**Key insight:** Tiered node-local storage works best when the cache fits within HBM + CPU RAM. SSD spillover still helps throughput but adds latency. The ceiling is node capacity.

**Limitation:** Node-local storage is fixed in size and cannot be shared across nodes. For multi-tenant or larger-than-node workloads, a different approach is needed.

---

## Approach 2: Centralized Shared Storage (Managed Lustre)

For workloads that exceed node-local capacity or require multi-tenant sharing, Google recommends Managed Lustre [2]:

> "We believe that Google Cloud Managed Lustre should be your primary storage solution for external KV Cache."

**Why not just local SSD?**

Google's blog [2] makes the case:

> "On GPUs, Lustre is assisted by locally attached SSDs. And on TPUs, where local SSDs are not available, Lustre's role is even more central."

> "While Local SSDs and CPU RAM are effective node-local tiers, they are fixed in size and cannot be shared. Managed Lustre provides a parallel file system to act as the massive, high-throughput external storage, making it a great solution for large-scale, multi-node, and multi-tenant AI inference workloads where the cache exceeds the capacity of the host machine."

The key insight: local SSDs *can* be shared in distributed setups, but their I/O bandwidth is limited by host NIC capacity. Lustre provides higher aggregate bandwidth and centralized sharing — critical for long-context workloads where KV Cache sizes reach terabytes.

**Performance data [2]:** DeepSeek-R1 on A3-Ultra (8× H200, 8× 141 GB HBM), ==50K== token context, ==~75%== cache hit rate, ==~3.4 TiB== total KV Cache.

| Metric | Result |
|---|---|
| Inference throughput | ==+75%== vs host memory only |
| Mean time to first token (MTTF) | ==−40%+== vs host memory only |
| I/O parallelism | 32 worker threads for KV reads |

> "In an experiment with a 50K token context and a high cache hit rate (about 75%), using Managed Lustre improved total inference throughput by ==75%== and reduced MTTF by more than ==40%== compared to using KV Cache in host memory alone." [2]

**TCO analysis [2]:** Workload processing ==1 million TPS== using A3-Ultra VMs + Managed Lustre.

| Metric | Result |
|---|---|
| TCO reduction | ==35%== vs no external storage |
| Accelerator savings | ==~40%== fewer GPUs needed |
| Key insight | "Storage replaces compute" — offloading KV to I/O frees accelerators |

> "TCO analysis yielded a ==35%== savings from using an external attention/KV Cache for a workload processing 1 million Tokens per Second (TPS)... when compared to a workload leveraging no external storage." [2]

> "You can handle a specific number of queries per second with ==~40%== fewer accelerators, resulting in direct cost savings." [2]

**Configuration details [2]:**

| Parameter | Value |
|---|---|
| Lustre bandwidth | ==1000 MB/s per TiB== (Performance Tier) |
| Capacity per machine | ==18 TiB== Lustre per A3-Ultra VM |
| Cluster size | ==73== A3-Ultra machines for 1M TPS target |
| GPU | 8× H200 per VM, 8× 141 GB HBM |
| Software tuning | `o_direct` flag (bypass FS cache), ==32== I/O worker threads |

> "Our experiment demonstrated that with configuration tuning and an improvement in KV Cache software to adopt more I/O parallelism, Managed Lustre can substantially improve inference performance." [2]

> "You can handle a specific number of queries per second with ==~40%== fewer accelerators, resulting in direct cost savings." [2]

**The core argument:** I/O bandwidth is the bottleneck. At ==1000 MB/s per TiB==, Lustre delivers ==18 TB/s== aggregate bandwidth per VM — far exceeding what host NIC-attached local SSDs can provide. The storage cost is offset by compute savings: ==40%== fewer accelerators for the same workload.

---

## Rapid Storage: What It Actually Does

Rapid Storage (Rapid Bucket + Rapid Cache) is Google's high-performance object storage layer — built on Colossus, the distributed system that powers Gemini and YouTube [3].

| Product | Bandwidth | Latency | Primary Use |
|---|---|---|---|
| **Rapid Bucket** | 15 TB/s | Sub-ms (open file) | Training data, checkpoint write/restore |
| **Rapid Cache** | 2.5 TB/s | Sub-ms | Inference model weight loading (cold start) |

**Rapid Storage is not designed for KV Cache externalization.** Its strengths are:
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

## What Comes Next?

Both KV Cache approaches prove the same point: at 42.7:1 storage:compute ratio, the storage layer determines inference cost and performance.

But Agentic AI is heading toward million-token contexts with ==90–99%== cache hit rates. Every percentage point of hit rate improvement means more data to store and more data to move. At 99% hit rates, a single workload needs ==1.3×== the I/O bandwidth of a 75% hit workload at the same context length.

**The open question: when KV Cache scales to millions of tokens, can Lustre keep up?**

Google's Lustre solution delivers ==75%== throughput improvement today at 50K context. But it runs on a shared RDMA fabric with fixed per-VM bandwidth (==18 TB/s== per A3-Ultra). At million-token scales with thousands of concurrent agents, the fabric becomes the contention point. Node-local tiered storage, meanwhile, hits a hard ceiling at node capacity (5 TiB SSD).

Current solutions are already at their limits. The next generation of AI infrastructure will be designed around bandwidth — not capacity alone.

---

## References

[1] [backyes — The Million-Token Bill](https://backyes.github.io/posts/million-seq-storage-vs-compute.html) — 42.7:1 storage:compute ratio, 77× cache hit cost difference

[2] [Google Cloud Blog — Choosing Managed Lustre for External KV Cache](https://cloud.google.com/blog/products/storage-data-transfer/choosing-google-cloud-managed-lustre-for-your-external-kv-cache) — 75% throughput improvement, 35% TCO savings, 40% fewer GPUs

[3] [Google Cloud Blog — Next '26 Storage Announcements](https://cloud.google.com/blog/products/storage-data-transfer/next26-storage-announcements) — Rapid Storage official positioning and performance specs

[4] [Google Cloud Documentation — Storage services for AI and ML workloads](https://docs.cloud.google.com/ai-hypercomputer/docs/storage) — Official storage tier recommendations

[5] [Google Cloud Blog — AI infrastructure at Next '26](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26) — "Dedicated KV Cache scalable storage subsystem" announcement

[6] [Google Cloud Blog — TPU 8t/8i technical deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) — TPU 8i 384 MB SRAM designed for KV Cache

[7] [Blocks & Files — Google's cloud storage gets faster and smarter for AI](https://www.blocksandfiles.com/public-cloud/2026/04/22/googles-cloud-storage-gets-faster-and-smarter-for-ai/5218551) — Managed Lustre as shared KV Cache

[11] [LMCache Blog — LMCache on GKE: Boosting LLM Inference with KV Cache on Tiered Storage](https://blog.lmcache.ai/en/2025/10/07/lmcache-on-google-kubernetes-engine-boosting-llm-inference-performance-with-kv-cache-on-tiered-storage/) — 264% throughput at 100K context, full benchmark data
