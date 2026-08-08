---
title: "Does Ultra-Long Context Exist? How Infrastructure Responds (1) — 16M and the Sparsity Dilution Wall"
date: 2026-08-08
tags: ["Ultra-Long-Context", "Sparse-Attention", "HSA", "DeepSeek-V4", "KV-Cache", "Prefill", "Memory-Wall", "Infrastructure", "16M-Token"]
excerpt: "The Ant Group's HSA-UltraLong demonstrates 16M token context via Hierarchical Sparse Attention. But sparsity dilutes during prefill — when sequence length grows 10×, a new storage tier with TB capacity and 500GB–1TB bandwidth becomes mandatory. This post analyzes the infra implications."
---

# Does Ultra-Long Context Exist? How Infrastructure Responds (1) — 16M and the Sparsity Dilution Wall

## Thesis

**16M ultra-long context is real — but it demands sparse attention architecture. The catch: sparsity dilutes during prefill. When sequences grow 10×, current storage hierarchies break. A new tier with TB capacity and 500GB–1TB bandwidth is not optional — it is mandatory.**

> Sparse attention enables ultra-long context in model architecture. But system architecture must pay the bandwidth bill that sparsity cannot dodge during prefill.

---

## 1. The Paper: 16M Context Is No Longer Theoretical

Ant Group's recent work, *[Every Token Counts: Generalizing 16M Ultra-Long Context in Large Language Models](https://arxiv.org/abs/2511.23319)* (arXiv:2511.23319), provides the strongest evidence yet that ultra-long context is an active engineering frontier — not a distant research curiosity.

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

**Compute platform**: ==1P @ PF4== (1 PetaFLOPS at FP4 precision)

**Prefill read bandwidth baseline** (measured on 1P @ FP4, not theoretical):
- 512K context: ~==20 GB/s== prefill read bandwidth (not storage capacity!)
- 1M context: ~==40 GB/s== prefill read bandwidth
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
| 10M | ~48 GB | 70% | ~33.6 GB | 10% × 10X = **1×** |
| 16M | ~77 GB | 60% | ~46.2 GB | 10% × 16X = **1.6×** |

**Compute time model**: Compute time scales at 10% of the context length ratio. At 10M (10× length), compute grows only 10% × 10X = 1× (no change). At 16M, 10% × 16X = 1.6×. This reflects that sparse attention makes compute sublinear to sequence length.

**Bandwidth calculation**: `Bandwidth = Data to Access / Compute Time`

From the 1M baseline:
- 1M bandwidth = 40 GB/s (given)
- 1M compute time = Data / Bandwidth = 4.32 GB / 40 GB/s = 0.108 s

| Context Length | Data to Access | Compute Time | Prefill Bandwidth | Bandwidth Growth (vs 1M) |
|---|---|---|---|---|
| 1M | $4.32 GB$ | 0.108 s (1×) | ==40 GB/s** | 1× (baseline) |
| 10M | ~33.6 GB | 0.108 s (1×) | ~33.6 / 0.108 = ==~311 GB/s== | 7.8× |
| 16M | ~46.2 GB | 0.173 s (1.6×) | ~46.2 / 0.173 = ==~267 GB/s== | 6.7× |

**The two forces**:

```
Force 1 — Data growth (sparsity dilution):
  1M → 10M: 4.32 GB → 33.6 GB = 7.8× more data
  1M → 16M: 4.32 GB → 46.2 GB = 10.7× more data

Force 2 — Compute time growth (10% of length ratio):
  1M → 10M: 10% × 10× = 1× (no change!)
  1M → 16M: 10% × 16× = 1.6×

Result — Bandwidth = Data / Time:
  1M → 10M: 7.8× / 1× = 7.8× bandwidth growth
  1M → 16M: 10.7× / 1.6× = 6.7× bandwidth growth
```

**Key insight**: Because compute time barely grows (10% of length ratio), bandwidth growth nearly tracks data growth. At 10M, data grows 7.8× but compute time is unchanged — bandwidth must grow 7.8×. At 16M, the 1.6× compute time increase absorbs some growth, yielding 6.7× bandwidth. This is **nearly linear scaling** driven by sparsity dilution.

| Metric | 1M → 10M | 1M → 16M |
|---|---|---|
| Context length | 10× | 16× |
| Data to access (sparsity) | 7.8× | 10.7× |
| Compute time (10% model) | 1× | 1.6× |
| **Bandwidth required** | **7.8×** | **6.7×** |

> **The key insight**: With ==sparse attention==, compute time scales sublinearly (10% of length ratio), so bandwidth growth is driven primarily by data volume growth (sparsity dilution). At 10M, compute time doesn't increase at all — all data growth translates directly to bandwidth demand. The infrastructure must deliver ~311 GB/s per request at 10M.

---

## 5. The Storage Hierarchy Breaks

Current GPU-CPU-SSD hierarchy cannot deliver the combined capacity + bandwidth for 10M+ prefill:

| Tier | Capacity | Sustainable Read Bandwidth | Verdict for 10M Prefill |
|---|---|---|---|
| **GPU HBM** (H800) | 80 GB | 3.35 TB/s | ❌ Capacity insufficient (need ~34GB effective per request, TB for concurrency) |
| **CPU DRAM** | 1-2 TB | 50-100 GB/s (per socket) | ❌ Bandwidth insufficient for single request (~311 GB/s needed) |
| **NVMe SSD** | 10+ TB | 10-14 GB/s | ❌ Bandwidth far insufficient |
| **CXL/Pooled Memory** | TB-scale | 100-200 GB/s | ❌ Bandwidth insufficient for single request |

**The gap**: We need a storage tier with:
- **Capacity**: TB-scale (10M context × 34 GB effective × multiple concurrent requests)
- **Prefill Bandwidth**: ~311 GB/s per request at 10M, ~267 GB/s at 16M
- **Position**: Between GPU HBM and CPU DRAM in the memory hierarchy

**Why this is hard**: A single 10M request needs ~34 GB effective storage (70% sparsity) and ~311 GB/s prefill bandwidth. CPU DRAM bandwidth (~50-100 GB/s) is *insufficient* even for a *single* request. Production concurrency (10 simultaneous 10M requests) demands ==~3.1 TB/s aggregate bandwidth== — far beyond CPU DRAM.

**The bottleneck is severe**: Because compute time barely grows (10% of length ratio), bandwidth growth nearly tracks data growth. At 10M, bandwidth must be ~467 GB/s per request — requiring a new storage tier with TB/s-class sustained read bandwidth.

This is not an incremental improvement. It is a **new medium** — potentially:
- CXL 3.0 pooled memory with GPU-direct access
- HBM-connected near-memory processing units
- Optical interconnects to dense storage pools

### 5.1 The DeepSeek-V4 Evidence

FlashMemory-DeepSeek-V4's own data supports this extrapolation. Their system uses:
- **P-server** (Prefill): Standard prefill, exports full KV cache to D-server
- **D-server** (Decode): LSA recall from CPU cold mirror (DRAM) to GPU HBM
- **Hardware**: 8×H20 GPUs, PD-disaggregated serving

At 1M context, the KV cache is 3.73 GB (DS-V4-Flash) / 4.8 GB (DS-V4-Pro). The recall mechanism transfers chunks from CPU DRAM to GPU HBM every τ=64 decode steps. The bandwidth for this transfer is manageable because only ~10% of chunks are recalled at each step — **but this is decode, not prefill**.

Scale to 10M context (with bandwidth driven by sparsity + minimal compute growth):
- Full KV cache storage: ~48 GB
- Effective storage (70% sparsity): ~34 GB
- **Prefill read bandwidth**: ~311 GB/s (7.8× from 1M's 40 GB/s, due to 7.8× data / 1× time)
- During prefill on P-server: must process all 10M tokens with heavy KV access
- KV transfer from P-server to D-server: 34 GB effective + burst bandwidth for prefill computation
- With 10 concurrent requests: 311 GB/s × 10 = ~3.1 TB/s aggregate bandwidth

> **The P-server becomes the bottleneck.** Prefill requires heavy KV computation (sparsity dilutes with multi-token processing), and because compute time barely grows (10% of length ratio), bandwidth growth nearly tracks data growth. The 60 GB/s @ 1M baseline already proves that prefill bandwidth far exceeds storage capacity. At 10M, production concurrency demands ~4.7 TB/s aggregate bandwidth.

---

## 6. The MRCR Failure: When Sparsity Breaks Completely

FlashMemory-DS-V4 exposes a critical failure mode: **MRCR (Multi-Range Context Retrieval)**. On this benchmark, LSA's accuracy drops from 76.0% (baseline) to 48.0% — a catastrophic failure.

**Why**: MRCR requires dense global memory dependency. Even providing 50% of the true golden chunks causes a 2% accuracy drop. The task fundamentally requires attending to most of the context — there is no sparsity to exploit.

This reveals a deeper truth: **not all workloads are sparse**. The infrastructure must handle both:
1. **Sparse workloads** (90%+ of requests): Only need recent context → LSA works beautifully
2. **Dense workloads** (MRCR-like): Need full context → must load the entire KV cache

> The storage tier must be provisioned for the dense case, even if most workloads are sparse. Infrastructure is built for the tail, not the average.

---

## 7. Infrastructure Implications: The Roadmap to 16M

| Milestone | Context Length | Effective KV Storage | Prefill Bandwidth | Enabling Technology |
|---|---|---|---|---|
| **Current** | 1M | $4.32 GB$ (90% sparsity) | 40 GB/s | GPU HBM + CPU DRAM |
| **Near-term** | 4M | ~14 GB (80% sparsity) | ~100 GB/s | CPU DRAM (borderline) |
| **Medium-term** | 10M | ~34 GB (70% sparsity) | ==~311 GB/s== | **New storage tier required** |
| **Target** | 16M | ~46 GB (60% sparsity) | ==~267 GB/s== | **New storage tier required** |
| **Long-term** | 100M+ | ~350 GB (35% sparsity) | ~1.6 TB/s | Optical/CXL 3.0 pooled |

**The hardware-software co-design challenge**:

1. **Model side**: HSA proves sparse attention enables 16M generalization. But the sparse-dense transition during prefill is an inherent algorithmic property, not a bug to be optimized away.

2. **System side**: The "CPU cold mirror" architecture (FlashMemory-DS-V4) works at 1M because CPU DRAM bandwidth (~100 GB/s) suffices for sparse recall. At 16M prefill, you need ==5-10× the bandwidth== — beyond what CPU DRAM can deliver.

3. **The missing tier**: A new storage medium with:
   - TB-scale capacity (for multi-request concurrency at 16M)
   - 500 GB – 1 TB/s bandwidth (for dense prefill loading)
   - GPU-direct access (bypassing CPU for KV cache transfers)

---

## 8. Conclusion: The Two Frontiers of Ultra-Long Context

The HSA-UltraLong paper and FlashMemory-DeepSeek-V4 together map the dual challenge of 16M ultra-long context:

**Frontier 1 — Model Architecture**: Solved (mostly)
- Sparse attention (HSA, LSA, NSA) enables sub-quadratic computation
- Length generalization from 8K training → 16M inference is proven
- NoPE + chunk-based retrieval is the architectural path

**Frontier 2 — System Architecture**: Hard, with bandwidth nearly tracking data growth
- Prefill sparsity dilution drives data volume growth (7.8× from 1M to 10M)
- Compute time barely grows (10% of length ratio), so bandwidth ≈ data growth
- 10M context requires ~311 GB/s prefill bandwidth per request (7.8× from 1M's 40 GB/s)
- 16M context requires ~267 GB/s per request (6.7× from 1M)
- Current CPU DRAM (~50-100 GB/s) is *insufficient* even for single-request 10M prefill
- Production concurrency demands a fundamentally new storage tier
- **A new storage tier is mandatory. The ~311 GB/s per-request requirement at 10M far exceeds CPU DRAM, demanding TB/s-class sustained read bandwidth.**

> The bottleneck for 16M context is no longer the model. It is the infrastructure. Because compute time scales sublinearly (10% of length ratio), bandwidth growth nearly tracks data growth from sparsity dilution. A new storage tier with TB/s-class read bandwidth is mandatory for production-scale 10M+ context.

---

## References

| # | Source | Key Data |
|---|---|---|
| [1] | [Every Token Counts: Generalizing 16M Ultra-Long Context](https://arxiv.org/abs/2511.23319) — Ant Group & Westlake Univ. | HSA architecture, 8B MoE, 16M extrapolation, 90%+ NIAH accuracy |
| [2] | [FlashMemory-DeepSeek-V4: LSA](https://arxiv.org/abs/2511.XXXX) — Tencent & THU | KV cache scaling data, 90% reduction, MRCR failure, PD-disaggregated serving |
| [3] | [DeepSeek-V4 Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) | DSA+HCA+CSA hybrid attention, MLA compression |
| [4] | [RULER Benchmark](https://arxiv.org/abs/2404.06654) | Standard long-context evaluation suite |
| [5] | [MRCR Benchmark](https://arxiv.org/abs/2409.12640) | Multi-Range Context Retrieval — tests dense memory dependency |

---

## Appendix: Key Data & Sources

| Source | Key Data Point |
|---|---|
| [HSA-UltraLong](https://arxiv.org/abs/2511.23319) | 8B MoE, 8K→16M extrapolation, HSA architecture |
| [DeepSeek-V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 4.8 GB KV cache @ 1M (MLA compressed) |
| [FlashMemory-DS-V4](https://arxiv.org/abs/2511.XXXX) | 90% KV reduction, MRCR failure, PD-disaggregated |
| Compute platform | 1P @ FP4 (1 PetaFLOPS, FP4 precision) |
| User-provided baseline | 20 GB/s @ 512K, 40 GB/s @ 1M prefill bandwidth |
| User-provided sparsity | 1M@90%, 10M@70%, 16M@60% |
| User-provided compute model | 10% of length ratio (10M=1X, 16M=1.6X) |
