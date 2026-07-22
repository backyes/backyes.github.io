---
title: "[Draft] Google Rapid Storage Was Not Built for KV Cache — Dissecting the Storage Hierarchy of Google's AI Hypercomputer"
date: 2026-07-22
tags: ["Google-Cloud", "Rapid-Storage", "KV-Cache", "TPU-8i", "Managed-Lustre", "AI-Inference", "Storage-Hierarchy"]
excerpt: "When Google announced Rapid Storage, many assumed it was purpose-built for LLM KV Cache. A deeper investigation reveals this is a misconception. Google's solution for KV Cache externalization is Managed Lustre — plus an undisclosed dedicated subsystem. Rapid Storage solves a different problem entirely: making sure data never becomes the bottleneck for accelerators."
---

# [Draft] Google Rapid Storage Was Not Built for KV Cache

## The Birth of a Misconception

At Google Cloud Next '26 in April 2026, Google announced Cloud Storage Rapid (Rapid Bucket and Rapid Cache) alongside TPU 8i — the latter tripling on-chip SRAM to 384 MB, explicitly designed to "host massive KV Caches entirely on silicon" [5].

These two announcements, landing in the same keynote, wrapped in the "AI Hypercomputer for the agentic era" narrative, produced a natural inference: **Rapid Storage must also be built for KV Cache.**

That inference is wrong.

> ==Rapid Storage was not built for KV Cache.== It solves "how to get data to accelerators fast enough"; KV Cache externalization requires "how to share and restore inference state across nodes."

Understanding this distinction means understanding the full storage hierarchy Google designed for AI Hypercomputer.

---

## 1. Google's Storage Hierarchy: A Four-Layer Architecture

Google Cloud's official documentation [1] explicitly maps storage products to AI workloads:

| Use Case | Primary Recommendation | Why |
|---|---|---|
| **Training data loading** | Rapid Bucket | 15 TB/s bandwidth + EB-scale capacity + cost efficiency |
| **Checkpoint write/restore** | Managed Lustre | Sub-ms latency + parallel data access |
| **Inference model weight loading** | Rapid Cache / Rapid Bucket | Eliminate inference cold start |
| ==**KV Cache offloading**== | ==**Managed Lustre**== | ==Sub-ms latency + cross-node pull== |

The key finding: **KV cache offloading's primary recommendation is Managed Lustre, not Rapid Storage.**

Google's own words:

> "We believe that Google Cloud Managed Lustre should be your primary storage solution for external KV Cache." [2]

---

## 2. What Rapid Storage Actually Delivers

Rapid Storage is built on Colossus — Google's internal distributed storage system that powers Gemini and YouTube [3]. It ships as two products:

### Rapid Bucket

| Metric | Value | vs. Traditional Object Store |
|---|---|---|
| Bandwidth | ==15 TB/s== | 7-15x |
| Request rate | ==20M ops/s== | 200x |
| Checkpoint write | — | ==3.2x== faster |
| Checkpoint restore | — | ==5x== faster |
| GPU blocked time | — | ==50%== reduction |

### Rapid Cache

| Metric | Value |
|---|---|
| Aggregate read throughput | ==2.5 TB/s== |
| Target workload | Bursty loads (inference model loading) |
| Code changes | Zero (transparent to existing buckets) |

These numbers point in one direction: **training throughput and accelerator utilization**, not inference-time KV Cache access.

---

## 3. Why Rapid Storage Is Architecturally Unsuitable for KV Cache

The mismatch is fundamental:

| Requirement | KV Cache Externalization | Rapid Storage (Object Store) |
|---|---|---|
| **Access granularity** | ~KB random reads | ~MB block reads/writes |
| **Consistency** | Strong (state cannot be lost) | Eventual consistency |
| **Latency** | Sub-ms (directly impacts TTFT) | Sub-ms (open file only) |
| **Access pattern** | Token-indexed random access | Large-file sequential I/O |
| **Shared access** | Multiple workers simultaneously | Limited |

Managed Lustre, as a POSIX-compliant parallel file system, natively supports everything KV Cache externalization demands: fine-grained random access, cross-node sharing, RDMA direct paths.

Managed Lustre's KV Cache performance [2]:
- ==40%+== reduction in TTFT (vs. host memory only)
- ==75%== inference throughput improvement
- Built on DDN EXAScaler with TPUDirect RDMA support

---

## 4. Google's KV Cache Strategy: "Avoid External Storage If You Can"

TPU 8i's architecture reveals Google's KV Cache philosophy:

| Tier | Storage | Capacity per Chip | KV Cache Role |
|---|---|---|---|
| **L0** | On-chip SRAM | ==384 MB== | ==Optimal==: preferred residence |
| **L1** | HBM | ==288 GB== | ==Secondary==: main capacity layer |
| **L2** | Managed Lustre | ==80 PB== | ==Externalization==: cross-node sharing |
| **L3** | Rapid Storage | ==EB-scale== | ==Unsuitable==: training data/weights |

TPU 8i's pod-level SRAM: ==384 MB × 1,152 chips = 432 GB==. This is Google's first line of defense for KV Cache — keep it on silicon whenever possible.

The logic is clear: **rather than shuttling KV Cache in and out of external storage, keep it on-chip.** Only when KV Cache grows too large for HBM does Managed Lustre serve as the L2 externalization tier. Rapid Storage plays no role in this chain.

---

## 5. Where the Misconception Comes From

**1. Simultaneous announcement, conflated context**
Rapid Storage and TPU 8i's 3x SRAM increase launched at the same event, both wrapped in the "AI Hypercomputer for the agentic era" narrative. But TPU 8i's SRAM is for KV Cache; Rapid Storage is not.

**2. The "Agentic Era" narrative redefines storage**
Google stated: "During AI inference, storage is the access layer that makes it responsive" [3]. The word "context" here is misread as KV Cache, when it actually refers to the full runtime context of an Agent — far more than just KV Cache.

**3. The name implies low latency**
"Rapid" suggests low latency, which maps easily to the latency requirements of KV Cache externalization. But Rapid refers to object-store-level throughput and IOPS improvements, not fine-grained random access latency.

**4. Industry trend projection**
In 2026, KV Cache externalization is the industry's hottest topic (LMCache, Mooncake, NVIDIA KV offload). Any new "AI storage" product is presumed to be "for KV Cache" by default.

---

## 6. What About the "Dedicated KV Cache Scalable Storage Subsystem"?

In the Next '26 announcement [4], Google listed a separate, undisclosed component:

> "Dedicated KV Cache scalable storage subsystem"

It was announced alongside TPU 8t/8i, Virgo Network, and Managed Lustre. As of this writing (July 2026), Google has released no technical details.

If it exists as a separate product, *that* would be Google's purpose-built KV Cache storage — but it is not Rapid Storage.

---

## 7. Implications for Model Designers

**If you design long-context models:**
TPU 8i's 384 MB SRAM means your model's KV Cache can reside entirely on-chip. This is a structural advantage over GPU-based approaches (limited HBM, mandatory offload). For million-token context windows, TPU 8i's storage hierarchy delivers significantly lower KV Cache access latency than GPU alternatives.

**If you design Agent systems:**
Managed Lustre's KV Cache sharing enables multiple Agents to share context state, eliminating redundant computation. This is Google's core storage advantage in the Agentic AI era.

**Rapid Storage has minimal direct impact on your model design:**
It primarily affects training efficiency and inference cold-start time, not KV Cache management during inference.

---

## 8. In One Sentence

> ==Google Rapid Storage is the "highway" for AI data pipelines — getting data to accelerators fast. KV Cache externalization needs a "vault" — the shared, durable, fine-grained state storage that Managed Lustre provides. They complement each other but are not interchangeable.==

Rapid Storage solves a supply-side problem (keeping TPUs/GPUs fully utilized); KV Cache externalization solves a state-management problem (moving inference state efficiently across nodes). In Google's AI Hypercomputer architecture, each has its role — neither was built *for* the other.

---

## References

[1] [Google Cloud Documentation — Storage services for AI and ML workloads](https://docs.cloud.google.com/ai-hypercomputer/docs/storage) — Official storage tier recommendations; KV cache offloading → Managed Lustre

[2] [Blocks & Files — Google's cloud storage gets faster and smarter for AI](https://www.blocksandfiles.com/public-cloud/2026/04/22/googles-cloud-storage-gets-faster-and-smarter-for-ai/5218551) — Managed Lustre as shared KV Cache: 40% TTFT improvement

[3] [Google Cloud Blog — Storage innovations at Next '26](https://cloud.google.com/blog/products/storage-data-transfer/next26-storage-announcements) — Rapid Storage official positioning and performance specs

[4] [Google Cloud Blog — AI infrastructure at Next '26](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26) — "Dedicated KV Cache scalable storage subsystem" announcement

[5] [Google Cloud Blog — TPU 8t/8i technical deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) — TPU 8i 384 MB SRAM designed for KV Cache

[6] [IO Fund — Google TPU v8 vs Nvidia](https://io-fund.com/ai-stocks/google-tpu-v8-vs-nvidia-inference-rewrites-ai-market) — Analyst perspective on TPU 8i inference advantages

[7] [StorageNews — Rapid cloud storage fixes AI training bottlenecks](https://storagenews.top/posts/rapid-cloud-storage-cuts-gpu-idle-time-by-half/) — Managed Lustre for KV Cache; Smart Storage for data selection

---

*The full research report (including all accessed/failed sources) is available at: [Google Rapid Storage Deep Research Report](/posts/research/google-rapid-storage/research_report.html)*
