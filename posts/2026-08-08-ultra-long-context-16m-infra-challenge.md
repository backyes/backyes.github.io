---
title: "Does Ultra-Long Context Exist? How Infrastructure Responds (1) — At 16M Context, Can Sparse Attention Scale Inference Cost and Performance?"
date: 2026-08-08
tags: ["Ultra-Long-Context", "Sparse-Attention", "HSA", "DeepSeek-V4", "KV-Cache", "Prefill", "Memory-Wall", "Infrastructure", "16M-Token"]
excerpt: "Ant Group's HSA-UltraLong demonstrates 16M token context via Hierarchical Sparse Attention. But sparsity dilutes during prefill — when sequence length grows 10×, a new storage tier with TB capacity and 500GB–1TB bandwidth becomes mandatory. This post analyzes the infra implications."
---

# Does Ultra-Long Context Exist? How Infrastructure Responds (1) — At 16M Context, Can Sparse Attention Scale Inference Cost and Performance?

## Thesis

**16M ultra-long context is real — but it demands sparse attention architecture. The catch: sparsity dilutes during prefill. When sequences grow 10×, current storage hierarchies break. A new tier with TB capacity and 500GB–1TB bandwidth is not optional — it is mandatory.**

> Sparse attention enables ultra-long context in model architecture. But sparsity itself is not a solved problem — it requires continued micro-architecture optimization. And system architecture must pay the bandwidth bill that sparsity cannot dodge during prefill. Two complementary approaches can address this: **storage-for-compute** (new media like HBF that trade storage density for bandwidth) and **compute-as-cache** (recomputing KV on-the-fly when agentic AI's short-append patterns make it cheaper than loading from memory).

---

## 1. The Papers: Ultra-Long Context Is an Active Frontier

The community is converging on ultra-long context as a critical engineering challenge. Two recent works highlight this trend:

- **Ant Group** — *Every Token Counts: Generalizing 16M Ultra-Long Context*[^1]: 8B MoE with Hierarchical Sparse Attention (HSA), extrapolating from 8K training to 16M inference — a ==500× extrapolation==.
- **Tencent / Tsinghua** — *FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention*[^2]: LSA on DeepSeek-V4, achieving 90% KV cache reduction via predictive lookahead indexing.

Both demonstrate that sparse attention is the architectural path to ultra-long context — but neither fully solves the infrastructure cost of prefill.

**The sparsity challenge remains open**[^2][^5]. FlashMemory-DS-V4 exposes a critical failure mode on MRCR (Multi-Range Context Retrieval): accuracy drops from 76% to 48%. MRCR requires dense global memory — even providing 50% of true golden chunks still causes 2% accuracy drop. This reveals that sparsity is not a solved problem: micro-architecture innovations (better indexers, retrieval mechanisms, attention patterns) are still needed to handle dense workloads. Infrastructure alone cannot fix what the model cannot retrieve.

### 1.1 HSA-UltraLong: 500× Extrapolation

| Property | HSA-UltraLong | Notes |
|---|---|---|
| **Architecture** | 8B MoE (1B activated params) | Based on Ling-2.0 MoE design |
| **Training Data** | 8 trillion tokens | Multi-stage: warmup → pretrain → mid-train → anneal |
| **Pre-training Context** | 8K tokens | Standard short-context pretraining |
| **Mid-training Context** | 32K tokens | Long-context adaptation |
| **Evaluation Range** | 4K → 16M tokens | **500× extrapolation** from training |
| **Core Innovation** | Hierarchical Sparse Attention (HSA) | Chunk-based sparse + retrieval fusion |
| **Key Result** | >90% accuracy on NIAH at 16M | Near-perfect needle-in-haystack retrieval |

**The critical insight**: HSA-UltraLong was *pre-trained* on only 8K context and *mid-trained* up to 32K. Yet it generalizes to **16M** — a ==500× extrapolation==. This is not incremental improvement; it is a qualitative leap in what context lengths are achievable.

---

## 2. Why Sparse Attention Is a Necessity, Not a Choice

The paper argues that ultra-long context requires three properties:

| Property | Why It Matters | How HSA Achieves It |
|---|---|---|
| **Sparsity** | Full attention is O(n²) — impossible at 16M | Each token retrieves only top-k chunks |
| **Random-Access Flexibility** | Must retrieve relevant fragments anywhere in 16M | Landmark-based chunk retrieval, end-to-end learned |
| **Length Generalization** | Cannot pretrain on 16M (too expensive) | Learn retrieval on short → extrapolate to long |

### 2.1 HSA Mechanism: Attention as Mixture-of-Experts

HSA's core design draws a direct analogy to MoE:

```
MoE:     Router selects top-k Experts → FFN(x) weighted fusion
HSA:     Retrieval selects top-k Chunks → Attention(x) weighted fusion
```

Formally, for each token $x_t$:
1. **Retrieval**: Compute dot-product scores against chunk landmarks → select top-k chunks
2. **Intra-chunk Attention**: Run standard attention within each retrieved chunk independently
3. **Inter-chunk Fusion**: Weighted sum of attention outputs using softmax-normalized retrieval scores

The chunk size is 64 tokens (hardware-aligned), and top-k is 64 chunks — giving a fixed 4,096-token historical window per HSA layer. This fixed window is the key to length generalization: the retrieval mechanism learned on 8K context transfers to 16M because the *retrieval vocabulary* (chunks) scales linearly while the *computation per token* stays constant.

### 2.2 NoPE: The Extrapolation Enabler

A critical design choice: **HSA layers use No Positional Encoding (NoPE)**, while sliding-window layers use RoPE. RoPE's periodic nature causes OOD degradation at lengths beyond training. NoPE removes this bottleneck entirely — the retrieval mechanism is content-based, not position-based.

---

## 3. The Business Context: Why 16M?

The paper frames the motivation as building **"Machines that Can Remember"** — a direct reference to the long-term memory problem in AI agents.

**The scenario**: A personalized AI agent that accumulates unique experiences over a user's lifetime. Human memory spans birth-to-present; machine memory requires ultra-long context. If Transformers could handle infinite-length contexts:
- World knowledge lives in context, not compressed into parameters
- Skills and latest information acquired via in-context learning, not retraining
- Each user gets a truly personalized agent with full history

**The business driver** (Ant Group's perspective): Financial services, insurance, compliance — domains where a customer's complete interaction history (potentially millions of tokens across years) must be accessible for accurate reasoning.

> This is not about processing a 16M-token document once. It is about an agent that *accumulates* 16M tokens of context over its lifetime and must reason over all of it, continuously.

---

## 4. The Prefill Problem: Why Sparsity Dilutes

Here is where the infrastructure story becomes critical. The paper demonstrates that HSA achieves sparsity during **decoding** — each generated token retrieves only top-k chunks from the full history. But **prefill** (processing a long input sequence) behaves fundamentally differently.

### 4.1 Single-Token Decode vs. Multi-Token Prefill

| Phase | KV Cache Access Pattern | Effective Sparsity |
|---|---|---|
| **Decode** (1 token) | Query attends to top-k chunks only | High sparsity — only ~10-13.5% of KV cache resident |
| **Prefill** (N tokens) | Each of N tokens retrieves top-k chunks | **Union of retrievals → dense** |

During decode, a single query token needs only its relevant chunks. The FlashMemory-DeepSeek-V4 paper confirms: ==over 90% of requests with contexts >64K can be resolved using only the last 8K tokens==. This means the active KV cache footprint during decode can be ~10% of the total.

But during prefill of a long sequence (e.g., loading a 16M context into the agent's memory), each of the N input tokens retrieves different top-k chunks. The **union** of all retrieved chunks across N tokens grows with N. At the limit, the effective attention pattern becomes dense — you need access to nearly the entire KV cache.

### 4.2 The Bandwidth Calculation

Let's trace the bandwidth scaling from concrete numbers. **Critical distinction**: KV cache *storage capacity* ≠ prefill *read bandwidth*. The latter includes chunk retrieval overhead, multi-head expansion, and burst transfer amplification.

**DeepSeek-V4 Pro baseline** (KV cache compression frontier):
- 1M context KV cache (full, no sparsity): ==4.8 GB== (with MLA compression)
- Per-token KV cache: ~4.8 KB/token

**Compute platform**: ==4P@FP4== (4 PetaFLOPS at FP4 precision)

**Prefill read bandwidth baseline** (1X = 4P@FP4):
- 1M context: ~==40 GB/s== prefill read bandwidth (at 4P@FP4, defined as 1X)
- This is the data volume that must be moved/processed during prefill per unit time

**The overhead ratio**:
```
Prefill bandwidth / Storage capacity = 40 GB/s / 4.8 GB ≈ 8×
```
This 8× factor captures: chunk-level retrieval scanning, multi-head KV expansion, attention score computation staging, and burst transfer granularity.

### Introducing Sparsity into KV Cache Storage

Sparsity is not uniform — it dilutes as context grows. At short contexts, most tokens can be skipped. At long contexts, the union of per-token retrievals covers more of the history.

### Prefill Bandwidth: Growth via Sparsity + Compute Scaling

Prefill bandwidth is determined by **two forces**: (1) sparsity determines how much data must be accessed, and (2) compute time grows with sequence length. Bandwidth = Data / Time.

**Sparsity semantics**: Sparsity here means the *fraction of KV cache that must be queried* during prefill.

| Context Length | Full KV Cache | Sparsity (fraction queried) | Data to Access | Compute Time (vs 1M) |
|---|---|---|---|---|
| 1M | $4.8 GB$ | 90% | $4.32 GB$ | 1× (baseline) |
| 10M | ~48 GB | 70% | ~33.6 GB | 1 + 10%×10 = **2×** |
| 16M | ~77 GB | 60% | ~46.2 GB | 1 + 10%×16 = **2.6×** |

**Compute time model**: Baseline 1X + 10% of length ratio as *incremental*. At 10M (10× length), compute = 1 + 0.1×10 = 2×. At 16M, compute = 1 + 0.1×16 = 2.6×. This reflects that sparse attention makes compute sublinear to sequence length.

**Bandwidth calculation**: `Bandwidth = Data to Access / Compute Time`

From the 1M baseline:
- 1M bandwidth = 40 GB/s (given)
- 1M compute time = Data / Bandwidth = 4.32 GB / 40 GB/s = 0.108 s

| Context Length | Data to Access | Compute Time | Prefill Bandwidth | Bandwidth Growth (vs 1M) |
|---|---|---|---|---|
| 1M | $4.32 GB$ | 0.108 s (1×) | ==40 GB/s** | 1× (baseline) |
| 10M | ~33.6 GB | 0.216 s (2×) | ~33.6 / 0.216 = ==~156 GB/s== | 3.9× |
| 16M | ~46.2 GB | 0.281 s (2.6×) | ~46.2 / 0.281 = ==~164 GB/s== | 4.1× |

**The two forces**:

```
Force 1 — Data growth (sparsity dilution):
  1M → 10M: 4.32 GB → 33.6 GB = 7.8× more data
  1M → 16M: 4.32 GB → 46.2 GB = 10.7× more data

Force 2 — Compute time growth (10% incremental):
  1M → 10M: 1 + 10%×10 = 2×
  1M → 16M: 1 + 10%×16 = 2.6×

Result — Bandwidth = Data / Time:
  1M → 10M: 7.8× / 2× = 3.9× bandwidth growth
  1M → 16M: 10.7× / 2.6× = 4.1× bandwidth growth
```

**Key insight**: Data volume grows ~7-10× from 1M to 10-16M, and compute time grows 2-2.6× (10% incremental model). The compute time increase absorbs some data growth, yielding 3.9-4.1× bandwidth growth. This is **sublinear scaling** — bandwidth grows slower than context length.

| Metric | 1M → 10M | 1M → 16M |
|---|---|---|
| Context length | 10× | 16× |
| Data to access (sparsity) | 7.8× | 10.7× |
| Compute time (10% incremental) | 2× | 2.6× |
| **Bandwidth required** | **3.9×** | **4.1×** |

> **The key insight**: With ==sparse attention==, compute time grows sublinearly (1 + 10% of length ratio), so bandwidth growth is moderated by compute time absorption. The infrastructure must deliver ~156 GB/s per request at 10M.

### Compute Scaling: Bandwidth Scales Linearly with Compute

Prefill bandwidth scales **linearly** with compute power. The baseline **1X = 4P@FP4 = 40 GB/s** means doubling compute doubles available bandwidth:

| Compute Scaling | Platform | Available Prefill BW | vs 1X baseline |
|---|---|---|---|
| **1X** | 4P@FP4 | 40 GB/s | 1× |
| **2X** | 8P@FP4 | 80 GB/s | 2× |
| **4X** | 16P@FP4 | 160 GB/s | 4× |
| **8X** | 32P@FP4 | 320 GB/s | 8× |

**Compute scaling vs context length coverage**:

带宽需求随上下文长度递增：1M (40 GB/s) → 10M (156 GB/s) → 16M (164 GB/s)。对每一级算力，需覆盖所有更短上下文累加带宽。

| Compute Available BW | 1M 需求 | 10M 需求 | 16M 需求 | 累加总需求 | 能否覆盖 |
|---|---|---|---|---|---|
| **1X**: 40 GB/s (4P@FP4) | 40 GB/s | 156 GB/s | 164 GB/s | **360+ GB/s** | ❌ 仅覆盖 1M |
| **2X**: 80 GB/s (8P@FP4) | 40 GB/s | 156 GB/s | 164 GB/s | **360+ GB/s** | ❌ 仅覆盖 1M |
| **4X**: 160 GB/s (16P@FP4) | 40 GB/s | 156 GB/s | 164 GB/s | **360+ GB/s** | ⚠️ 覆盖 1M+10M，16M 边界 |
| **8X**: 320 GB/s (32P@FP4) | 40 GB/s | 156 GB/s | 164 GB/s | **360+ GB/s** | ✅ 全覆盖 |

> **The compute-bandwidth coupling**: 1X 算力 (40 GB/s) 仅能满足单个 1M 请求。10M 单请求需 156 GB/s → 需 4X (16P@FP4, 160 GB/s)。16M 单请求需 164 GB/s → 4X 边界，需 8X (32P@FP4, 320 GB/s) 才能舒适覆盖。若需同时服务 1M+10M+16M 混合负载，总带宽需求 360+ GB/s → 需 8X 以上。算力翻倍确实带来带宽翻倍，但随上下文长度递增，累加需求使得单纯算力扩展经济上不可行。

---

## 5. The Storage Hierarchy Breaks

Current GPU-CPU-SSD hierarchy cannot deliver the combined capacity + bandwidth for 10M+ prefill:

| Tier | Capacity | Sustainable Read Bandwidth | Verdict for 10M Prefill |
|---|---|---|---|
| **GPU HBM** (H800) | 80 GB | 3.35 TB/s | ❌ Capacity insufficient (need ~34GB effective per request, TB for concurrency) |
| **CPU DRAM** | 1-2 TB | 50-100 GB/s (per socket) | ❌ Bandwidth insufficient for single request (~156 GB/s needed) |
| **NVMe SSD** | 10+ TB | 10-14 GB/s | ❌ Bandwidth far insufficient |
| **CXL/Pooled Memory** | TB-scale | 100-200 GB/s | ✅ Can meet requirement, but high cost |

**The gap**: We need a storage tier with:
- **Capacity**: TB-scale (10M context × 34 GB effective × multiple concurrent requests)
- **Prefill Bandwidth**: ~156 GB/s per request at 10M, ~164 GB/s at 16M
- **Position**: Between GPU HBM and CPU DRAM in the memory hierarchy

**Why this is hard**: A single 10M request needs ~34 GB effective storage (70% sparsity) and ~156 GB/s prefill bandwidth. CPU DRAM bandwidth (~50-100 GB/s) is *insufficient* even for a *single* request.

**Compute scaling projection**: On a 4P@FP4 platform (1X, 40 GB/s aggregate bandwidth):
- A single 1M request needs ~40 GB/s → consumes 100% of 1X bandwidth budget
- A single 10M request needs ~156 GB/s → needs 4X (16P@FP4, 160 GB/s)
- A single 16M request needs ~164 GB/s → needs 4X-8X (16-32P@FP4)
- **Concurrency killer**: 4X compute can serve only ONE 10M request at a time. Two concurrent 10M requests need 8X compute (32P@FP4).

This means compute scaling alone cannot solve the problem — it merely shifts the bottleneck from "bandwidth insufficient" to "compute massively over-provisioning". **Storage tier innovation (HBF/CXL) is mandatory** to provide both capacity AND bandwidth without linearly scaling compute.

CXL 3.0 pooled memory is the development target that can meet these requirements (TB-scale capacity + ~160 GB/s bandwidth), but at significant cost.

### 5.1 The DeepSeek-V4 Evidence

FlashMemory-DeepSeek-V4's own data supports this extrapolation. Their system uses:
- **P-server** (Prefill): Standard prefill, exports full KV cache to D-server
- **D-server** (Decode): LSA recall from CPU cold mirror (DRAM) to GPU HBM
- **Hardware**: 8×H20 GPUs, PD-disaggregated serving

At 1M context, the KV cache is 3.73 GB (DS-V4-Flash) / 4.8 GB (DS-V4-Pro). The recall mechanism transfers chunks from CPU DRAM to GPU HBM every τ=64 decode steps. The bandwidth for this transfer is manageable because only ~10% of chunks are recalled at each step — **but this is decode, not prefill**.

Scale to 10M context (with bandwidth driven by sparsity + minimal compute growth):
- Full KV cache storage: ~48 GB
- Effective storage (70% sparsity): ~34 GB
- **Prefill read bandwidth**: ~156 GB/s (3.9× from 1M's 40 GB/s, due to 7.8× data / 2× time)
- During prefill on P-server: must process all 10M tokens with heavy KV access
- KV transfer from P-server to D-server: 34 GB effective + burst bandwidth for prefill computation

> **The P-server becomes the bottleneck.** Prefill requires heavy KV computation (sparsity dilutes with multi-token processing), and because compute time grows sublinearly (1 + 10% of length ratio), bandwidth growth is moderated but still substantial. The 40 GB/s @ 1M baseline (1X = 4P@FP4) already proves that prefill bandwidth far exceeds storage capacity. At 10M, a single request needs ~156 GB/s — requiring 4X compute (16P@FP4) for a single request, leaving zero headroom for concurrency. At 16M, 4X is borderline (160 GB/s ≈ 164 GB/s), demanding 8X compute (32P@FP4) for comfortable headroom.

---

## 6. How System Addresses the Ultra-Long Context Challenge

### 6.1 Quantitative Analysis & System Direction

| Context Length | Full KV Cache | Sparsity | Data to Access | Compute Time | Prefill Bandwidth | vs 1M |
|---|---|---|---|---|---|---|
| **1M** | $4.8 GB$ | 90% | $4.32 GB$ | 1× | **40 GB/s** (1X, 4P@FP4) | 1× |
| **10M** | ~48 GB | 70% | ~33.6 GB | 2× | ==**~156 GB/s**== (4X, 16P@FP4) | 3.9× |
| **16M** | ~77 GB | 60% | ~46.2 GB | 2.6× | ==**~164 GB/s**== (4X, 16P@FP4) | 4.1× |

**Key ratios**:
- 1M → 10M: Context length 10×, data volume 7.8×, compute time 2×, **bandwidth 3.9×**
- 1M → 16M: Context length 16×, data volume 10.7×, compute time 2.6×, **bandwidth 4.1×**

**Compute-bandwidth coupling**: Prefill bandwidth scales linearly with compute power (per **1X = 4P@FP4 → ~40 GB/s** @ 1M baseline). Doubling compute doubles available bandwidth. But the coupling is a double-edged sword:

| Scenario | Compute Needed | Problem |
|---|---|---|
| Single 10M request | 4X (16P@FP4) → 160 GB/s | ✅ Feasible, but 100% bandwidth utilization |
| Single 16M request | 4X-8X (16-32P@FP4) | ⚠️ Massive compute over-provisioning |
| 2× concurrent 10M | 8X (32P@FP4) → 320 GB/s | ❌ Compute scales linearly with concurrency |
| 4× concurrent 10M | 16X (64P@FP4) | ❌ Economically unviable |

**The fundamental tension**: Compute scaling can provide bandwidth "for free" (linear coupling), but it scales linearly with both context length AND concurrency. At 10M+, the compute cost of sustaining prefill bandwidth becomes prohibitive for multi-tenant serving.

**System direction judgment**: The ~160 GB/s per-request bandwidth at 10-16M exceeds CPU DRAM capability (~50-100 GB/s) and approaches HBM bandwidth density. The future system architecture needs a **three-pronged approach**: (1) introduce a new storage tier (HBF, CXL pooled memory) for capacity + bandwidth, (2) exploit compute-as-cache to trade increasingly cheap compute for scarce I/O bandwidth, and (3) leverage compute-bandwidth coupling for headroom — but recognize its linear scaling limit for concurrency.

### 6.2 Thinking on System for Ultra-Long Context

**Current storage hierarchy breaks at 10M+**:

| Storage Tier | Capacity | Sustained Read Bandwidth | Can Serve 10M Prefill? |
|---|---|---|---|
| **GPU HBM** (H800) | 80 GB | 3.35 TB/s | ❌ Capacity insufficient |
| **CPU DRAM** | 1-2 TB | 50-100 GB/s | ❌ Bandwidth insufficient (~156 GB/s needed) |
| **NVMe SSD** | 10+ TB | 10-14 GB/s | ❌ Bandwidth far insufficient |
| **CXL/Pooled Memory** | TB-scale | 100-200 GB/s | ⚠️ Can meet requirement, but high cost |

**Three complementary approaches**:

1. **Storage-for-Compute (以存换算)**: Develop new storage media like ==HBF (High-Bandwidth Flash)== that offer TB-scale capacity with ~160 GB/s sustained read bandwidth. This trades storage density for bandwidth, providing a new tier between DRAM and SSD. CXL 3.0 pooled memory is another option for the medium-term. **This is the primary solution** for capacity + bandwidth that doesn't scale linearly with compute.

2. **Compute-as-Cache (以算缓存)**: For agentic AI's short-append patterns (where only a small context delta is added per turn), recompute KV cache on-the-fly using on-chip compute rather than loading from external memory. When compute becomes cheaper relative to memory bandwidth, this can be more economical than fetching from DRAM.

3. **Compute-Bandwidth Coupling (算力带宽耦合)**: Leverage the linear relationship between compute and prefill bandwidth (1X = 4P@FP4 → 40 GB/s, 4X = 16P@FP4 → 160 GB/s, 8X = 32P@FP4 → 320 GB/s) for headroom. But recognize its limit: compute scales linearly with concurrency, so it cannot be the sole solution for multi-tenant serving.

**Roadmap**:

| Phase | Context Length | Effective KV Storage | Prefill Bandwidth | Compute Needed | Enabling Technology |
|---|---|---|---|---|---|
| **Current** | 1M | $4.32 GB$ | 40 GB/s | 1X (4P@FP4) | GPU HBM + CPU DRAM |
| **Near-term** | 4M | ~14 GB | ~100 GB/s | 2-3X (8-12P@FP4) | CPU DRAM (borderline) |
| **Medium-term** | 10M | ~34 GB | ==~156 GB/s== | 4X (16P@FP4) | **HBF / CXL 3.0 pooled memory** + compute coupling |
| **Target** | 16M | ~46 GB | ==~164 GB/s== | 4-8X (16-32P@FP4) | **HBF / CXL 3.0 pooled memory** + compute coupling |
| **Long-term** | 100M+ | ~350 GB | ~1.6 TB/s | 40X+ (unviable alone) | **Storage tier mandatory** + compute-as-cache |

---

## 7. Conclusion: The Two Frontiers of Ultra-Long Context

**Frontier 1 — Model Architecture**: Partially solved
- Sparse attention (HSA, LSA) enables 16M extrapolation (500×)
- But sparsity itself remains challenging (MRCR failure), requiring continued micro-architecture optimization

**Frontier 2 — System Architecture**: Open problem
- Prefill sparsity dilution drives data volume growth (7.8× from 1M to 10M)
- Compute time grows sublinearly (1 + 10% of length ratio) → bandwidth growth 3.9-4.1×
- 10M requires ~156 GB/s, 16M requires ~164 GB/s per request
- CPU DRAM is insufficient even for a single request; **new storage tier mandatory**
- **Compute-bandwidth coupling**: Linear compute growth linearly increases prefill bandwidth (1X = 4P@FP4 → 40 GB/s, 4X = 16P@FP4 → 160 GB/s, 8X = 32P@FP4 → 320 GB/s). This provides headroom but scales linearly with concurrency — 4X serves one 10M request, but 8X is needed for two. Compute alone cannot solve multi-tenant ultra-long context serving.
- **Three-pronged solution**: (1) Storage tier innovation (HBF/CXL) for capacity + bandwidth, (2) compute-as-cache for append-heavy workloads, (3) compute-bandwidth coupling for headroom — all three are necessary because no single approach scales economically.

> **The bottom line**: The bottleneck for 16M context is no longer the model — it is the infrastructure. Sparse attention solves computational complexity but cannot dodge the prefill bandwidth demand. Storage bandwidth (not compute) becomes the critical resource, and compute-bandwidth coupling alone cannot scale economically for concurrency. A new storage tier (HBF/CXL) is mandatory, complemented by compute-as-cache and strategic compute over-provisioning.

---

## 8. Limitations & Open Questions

**8.1 Algorithmic deficiency feedback to infrastructure**: The MRCR failure (76% → 48%) reveals that current sparse indexers lose information on dense global memory tasks. If precision requirements force "dynamic dense fallback" or "multi-level hybrid attention," prefill bandwidth pressure worsens further. Infrastructure must provision redundancy bandwidth for this algorithmic worst case — but how much remains an open question.

**8.2 Prefill chunk-sharing and kernel-level optimization**: Our analysis assumes the union of per-token retrievals approaches dense. In practice, adjacent tokens exhibit high overlap in retrieved chunks (spatial locality). Techniques like hierarchical aggregation, tile-based prefill, or chunk-level prefix caching could reduce union inflation. Software/kernel-layer mitigations may meaningfully lower effective bandwidth demand.

**8.3 Crossover point: Storage-for-Compute vs Compute-as-Cache**: The choice between provisioning new storage media (HBF/CXL) and recomputing KV on-the-fly depends on the relative cost trajectory of compute vs. memory bandwidth. A quantitative crossover model — at context length $L$ with append delta $\Delta L$, when does recompute become cheaper than load? — would provide sharper engineering guidance. This remains a promising direction for future analysis.



[^1]: [Every Token Counts: Generalizing 16M Ultra-Long Context](https://arxiv.org/abs/2511.23319) — Ant Group & Westlake Univ., 2025
[^2]: [FlashMemory-DeepSeek-V4: Lightning Index Ultra-Long Context via Lookahead Sparse Attention](https://arxiv.org/abs/2606.09079) — Tencent & Tsinghua, 2025
[^3]: [DeepSeek-V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) — DeepSeek-AI, 2026
[^4]: [RULER Benchmark](https://arxiv.org/abs/2404.06654)
[^5]: [MRCR Benchmark](https://arxiv.org/abs/2409.12640)

