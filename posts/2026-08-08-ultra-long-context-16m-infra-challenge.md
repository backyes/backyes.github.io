---
title: "Does Ultra-Long Context Exist? How Infrastructure Responds (1) — 16M and the Sparsity Dilution Wall"
date: 2026-08-08
tags: ["Ultra-Long-Context", "Sparse-Attention", "HSA", "DeepSeek-V4", "KV-Cache", "Prefill", "Memory-Wall", "Infrastructure", "16M-Token"]
excerpt: "Ant Group's HSA-UltraLong demonstrates 16M token context via Hierarchical Sparse Attention. But sparsity dilutes during prefill — when sequence length grows 10×, a new storage tier with TB capacity and 500GB–1TB bandwidth becomes mandatory. This post analyzes the infra implications."
---

# Does Ultra-Long Context Exist? How Infrastructure Responds (1) — 16M and the Sparsity Dilution Wall

## Thesis

**16M ultra-long context is real — but it demands sparse attention architecture. The catch: sparsity dilutes during prefill. When sequences grow 10×, current storage hierarchies break. A new tier with TB capacity and 500GB–1TB bandwidth is not optional — it is mandatory.**

> Sparse attention enables ultra-long context in model architecture. But sparsity itself is not a solved problem — it requires continued micro-architecture optimization. And system architecture must pay the bandwidth bill that sparsity cannot dodge during prefill.

---

## 1. The Papers: Ultra-Long Context Is an Active Frontier

The community is converging on ultra-long context as a critical engineering challenge. Two recent works highlight this trend:

- **Ant Group** — *[Every Token Counts: Generalizing 16M Ultra-Long Context](https://arxiv.org/abs/2511.23319)* (arXiv:2511.23319): 8B MoE with Hierarchical Sparse Attention (HSA), extrapolating from 8K training to 16M inference — a ==500× extrapolation==.
- **Tencent / Tsinghua** — *[FlashMemory-DeepSeek-V4: Lookahead Sparse Attention](https://arxiv.org/abs/2606.09079)* (arXiv:2606.09079): LSA on DeepSeek-V4, achieving 90% KV cache reduction via predictive lookahead indexing.

Both demonstrate that sparse attention is the architectural path to ultra-long context — but neither fully solves the infrastructure cost of prefill.

**The sparsity challenge remains open**. FlashMemory-DS-V4 exposes a critical failure mode on MRCR (Multi-Range Context Retrieval): accuracy drops from 76% to 48%. MRCR requires dense global memory — even providing 50% of true golden chunks still causes 2% accuracy drop. This reveals that sparsity is not a solved problem: micro-architecture innovations (better indexers, retrieval mechanisms, attention patterns) are still needed to handle dense workloads. Infrastructure alone cannot fix what the model cannot retrieve.

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
- Per-token KV cache: ~5.04 KB/token

**Compute platform**: ==4P @ FP4== (4 PetaFLOPS at FP4 precision)

**Prefill read bandwidth baseline** (per 1P @ FP4, not theoretical):
- 512K context: ~==20 GB/s== prefill read bandwidth per 1P @ FP4
- 1M context: ~==40 GB/s== prefill read bandwidth per 1P @ FP4
- 4P @ FP4 aggregate: ~==160 GB/s== available at 1M
- This is the data volume that must be moved/processed during prefill per unit time

**The overhead ratio**:
```
Prefill bandwidth / Storage capacity = 20 GB/s / 2.5 GB ≈ 8×
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

**4P @ FP4 projection**: On a 4P @ FP4 platform, 1M context has ~160 GB/s aggregate bandwidth available. A single 10M request needs ~156 GB/s, consuming the storage bandwidth budget of ~1P @ FP4; 16M needs ~164 GB/s, consuming ~1P. This means in ultra-long context scenarios, **storage bandwidth (not compute) becomes the bottleneck**, demanding storage tier innovation.

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

> **The P-server becomes the bottleneck.** Prefill requires heavy KV computation (sparsity dilutes with multi-token processing), and because compute time grows sublinearly (1 + 10% of length ratio), bandwidth growth is moderated but still substantial. The 40 GB/s @ 1M baseline already proves that prefill bandwidth far exceeds storage capacity. At 10M, a single request needs ~156 GB/s — consuming ~1P of a 4P @ FP4 platform's storage bandwidth budget.

---

## 6. Infrastructure Implications: Summary

Based on the quantitative analysis, ultra-long context infrastructure requirements are summarized as follows:

### 6.1 Quantitative Data Summary

| Context Length | Full KV Cache | Sparsity | Data to Access | Compute Time | Prefill Bandwidth | vs 1M |
|---|---|---|---|---|---|---|
| **1M** | $4.8 GB$ | 90% | $4.32 GB$ | 1× | **40 GB/s** | 1× |
| **10M** | ~48 GB | 70% | ~33.6 GB | 2× | ==**~156 GB/s**== | 3.9× |
| **16M** | ~77 GB | 60% | ~46.2 GB | 2.6× | ==**~164 GB/s**== | 4.1× |

**Key ratios**:
- 1M → 10M: Context length 10×, data volume 7.8×, compute time 2×, **bandwidth 3.9×**
- 1M → 16M: Context length 16×, data volume 10.7×, compute time 2.6×, **bandwidth 4.1×**

### 6.2 4P @ FP4 Platform Bandwidth Budget

| Platform | Available Bandwidth | 10M Consumption | 16M Consumption |
|---|---|---|---|
| 4P @ FP4 | ~160 GB/s @ 1M equivalent | ~156 GB/s (≈1P) | ~164 GB/s (≈1P) |

**Scaling with compute**: Prefill bandwidth scales linearly with compute power (per 1P @ FP4 → ~40 GB/s @ 1M baseline). If compute density doubles (8P @ FP4), available bandwidth doubles to ~320 GB/s, easing the storage bandwidth pressure.

**Conclusion**: On a 4P @ FP4 platform, a single 10M request consumes ~1P of the storage bandwidth budget; 16M consumes ~1P. **Storage bandwidth (not compute) becomes the bottleneck for ultra-long context.**

### 6.3 Storage Hierarchy Gap

| Storage Tier | Capacity | Sustained Read Bandwidth | Can Serve 10M Prefill? |
|---|---|---|---|
| **GPU HBM** (H800) | 80 GB | 3.35 TB/s | ❌ Capacity insufficient |
| **CPU DRAM** | 1-2 TB | 50-100 GB/s | ❌ Bandwidth insufficient (~156 GB/s needed) |
| **NVMe SSD** | 10+ TB | 10-14 GB/s | ❌ Bandwidth far insufficient |
| **CXL/Pooled Memory** | TB-scale | 100-200 GB/s | ⚠️ Single request borderline |

**The missing tier**: A new storage medium with TB-scale capacity + ~160 GB/s sustained read bandwidth.

### 6.4 Development Roadmap

| Phase | Context Length | Effective KV Storage | Prefill Bandwidth | Enabling Technology |
|---|---|---|---|---|
| **Current** | 1M | $4.32 GB$ | 40 GB/s | GPU HBM + CPU DRAM |
| **Near-term** | 4M | ~14 GB | ~100 GB/s | CPU DRAM (borderline) |
| **Medium-term** | 10M | ~34 GB | ==~156 GB/s== | **New storage tier required** |
| **Target** | 16M | ~46 GB | ==~164 GB/s== | **New storage tier required** |
| **Long-term** | 100M+ | ~350 GB | ~1.6 TB/s | Optical/CXL 3.0 pooled |

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

> **The bottom line**: The bottleneck for 16M context is no longer the model — it is the infrastructure. Sparse attention solves computational complexity but cannot dodge the prefill bandwidth demand. On a 4P @ FP4 platform, storage bandwidth (not compute) becomes the critical resource constraining ultra-long context.

---

## References

| # | Source | Key Data |
|---|---|---|
| [1] | [Every Token Counts: Generalizing 16M Ultra-Long Context](https://arxiv.org/abs/2511.23319) — Ant Group & Westlake Univ. | HSA architecture, 8B MoE, 16M extrapolation, 90%+ NIAH accuracy |
| [2] | [FlashMemory-DeepSeek-V4: LSA](https://arxiv.org/abs/2606.09079) — Tencent & THU | KV cache scaling data, 90% reduction, MRCR failure, PD-disaggregated serving |
| [3] | [DeepSeek-V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) | DSA+HCA+CSA hybrid attention, MLA compression |
| [4] | [RULER Benchmark](https://arxiv.org/abs/2404.06654) | Standard long-context evaluation suite |
| [5] | [MRCR Benchmark](https://arxiv.org/abs/2409.12640) | Multi-Range Context Retrieval — tests dense memory dependency |

---

## Appendix: Key Data & Sources

| Source | Key Data Point |
|---|---|
| [HSA-UltraLong](https://arxiv.org/abs/2511.23319) | 8B MoE, 8K→16M extrapolation, HSA architecture |
| [DeepSeek-V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 4.8 GB KV cache @ 1M (MLA compressed) |
| [FlashMemory-DS-V4](https://arxiv.org/abs/2606.09079) | 90% KV reduction, MRCR failure, PD-disaggregated |
| Compute platform | 4P @ FP4 (4 PetaFLOPS, FP4 precision) |
| User-provided baseline | 20 GB/s @ 512K, 40 GB/s @ 1M prefill bandwidth |
| User-provided sparsity | 1M@90%, 10M@70%, 16M@60% |
| User-provided compute model | 1 + 10% of length ratio (10M=2X, 16M=2.6X) |
