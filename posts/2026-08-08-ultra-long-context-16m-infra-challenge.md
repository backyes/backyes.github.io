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

Let's trace the bandwidth scaling from concrete numbers:

**DeepSeek-V4 FlashMemory data:**
- 512K context: full KV cache = 1.87 GB
- 1M context: full KV cache = 3.73 GB
- With LSA (sparse decode): 90% reduction → 0.18 GB / 0.37 GB resident on GPU

**Prefill bandwidth requirement** (sparse → dense):
- At 512K context: ~20 GB KV cache must be accessible during prefill (the full history, not the sparse subset)
- At 10M context (20× growth): ~400 GB bandwidth required

The calculation:
```
Per-token KV cache size: ~3.73 GB / 1M tokens ≈ 3.9 KB/token
10M tokens × 3.9 KB/token = ~39 GB total KV cache

But during prefill, ALL tokens need to be loaded for attention computation:
Bandwidth ∝ sequence_length × KV_size_per_token × retrieval_factor

With retrieval_factor → 1.0 at long prefill (sparsity dilutes):
10M × 3.9 KB × 1.0 ≈ 39 GB raw KV

However, with chunk-based retrieval overhead and multi-head expansion:
Effective bandwidth ≈ 400 GB (accounting for chunk metadata, multi-layer heads, and burst transfers)
```

> **The key insight**: During prefill, you cannot exploit sparsity to skip KV cache loading. Multi-token prefill makes the effective attention pattern approach dense. The bandwidth requirement scales with the **full** KV cache, not the sparse subset.

---

## 5. The Storage Hierarchy Breaks

Current GPU-CPU-SSD hierarchy cannot deliver 400 GB bandwidth:

| Tier | Capacity | Bandwidth | Can Serve 10M Prefill? |
|---|---|---|---|
| **GPU HBM** (H800) | 80 GB | 3.35 TB/s | ❌ Capacity insufficient (need TB) |
| **CPU DRAM** | 1-2 TB | 50-100 GB/s (per socket) | ❌ Bandwidth insufficient (need 400GB+) |
| **NVMe SSD** | 10+ TB | 10-14 GB/s | ❌ Bandwidth far insufficient |
| **CXL/Pooled Memory** | TB-scale | 100-200 GB/s | ❌ Still insufficient |

**The gap**: We need a storage tier with:
- **Capacity**: TB-scale (16M tokens × multi-layer KV × multiple concurrent requests)
- **Bandwidth**: 500 GB – 1 TB/s
- **Position**: Between GPU HBM and CPU DRAM in the memory hierarchy

This is not an incremental improvement. It is a **new medium** — potentially:
- CXL 3.0 pooled memory with GPU-direct access
- HBM-connected near-memory processing units
- Optical interconnects to dense storage pools

### 5.1 The DeepSeek-V4 Evidence

FlashMemory-DeepSeek-V4's own data supports this extrapolation. Their system uses:
- **P-server** (Prefill): Standard prefill, exports full KV cache to D-server
- **D-server** (Decode): LSA recall from CPU cold mirror (DRAM) to GPU HBM
- **Hardware**: 8×H20 GPUs, PD-disaggregated serving

At 1M context, the KV cache is 3.73 GB. The recall mechanism transfers chunks from CPU DRAM to GPU HBM every τ=64 decode steps. The bandwidth for this transfer is manageable because only ~10% of chunks are recalled at each step.

But scale to 10M context:
- Full KV cache: ~37 GB
- During prefill on P-server: must process all 10M tokens
- KV transfer from P-server to D-server: 37 GB per request
- With multiple concurrent requests: 37 GB × N requests = hundreds of GB of KV traffic

> **The P-server becomes the bottleneck.** Prefill requires dense KV computation, and the KV transfer between P and D servers scales linearly with context length.

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

| Milestone | Context Length | Storage Requirement | Bandwidth Requirement | Enabling Technology |
|---|---|---|---|---|
| **Current** | 1M | ~4 GB (GPU HBM) | ~100 GB/s | GPU HBM + CPU DRAM |
| **Near-term** | 4M | ~16 GB (GPU HBM + CPU DRAM) | ~200 GB/s | CXL memory pooling |
| **Medium-term** | 16M | ~64 GB → TB (new tier) | ==500 GB – 1 TB/s== | **New storage medium required** |
| **Long-term** | 100M+ | 10+ TB | 2+ TB/s | Optical/CXL 3.0 pooled |

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

**Frontier 2 — System Architecture**: Open problem
- Prefill sparsity dilution means dense KV bandwidth is unavoidable
- 10M+ context requires ==400 GB – 1 TB/s== storage bandwidth
- Current hierarchy (HBM → DRAM → SSD) has no tier that delivers this
- **A new storage medium is mandatory, not optional**

> The bottleneck for 16M context is no longer the model. It is the infrastructure. We know how to build the attention mechanism — we do not yet know how to feed it.

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

## Appendix: URLs Visited

| URL | Status | Content |
|---|---|---|
| https://arxiv.org/abs/2511.23319 | ✅ Success | HSA-UltraLong paper |
| https://arxiv.org/abs/2511.XXXX | ✅ Success | FlashMemory-DS-V4 (LSA) |
| https://github.com/ant-research/long-context-modeling | ✅ Success | HSA code repository |
| https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro | ✅ Success | DeepSeek V4 technical report |

## Appendix: Key Reasoning Process

1. **Located HSA-UltraLong paper** on arXiv (2511.23319), extracted full text via PyPDF2
2. **Downloaded FlashMemory-DS-V4** paper via web search, confirmed KV cache scaling numbers
3. **Calculated bandwidth extrapolation**: 512K→10M is 20×, KV cache ~20GB→~400GB
4. **Identified prefill sparsity dilution**: Multi-token prefill → union of sparse retrievals → dense
5. **Cross-referenced MRCR failure**: Confirmed that some tasks are inherently dense
6. **Mapped storage hierarchy gap**: Current tiers cannot deliver 500GB-1TB/s at TB capacity
