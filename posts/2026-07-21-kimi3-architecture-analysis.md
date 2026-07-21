---
title: "Understanding Kimi3 (1): Linear Attention, Sparse Attention, and the Architecture War at Million-Token Scale"
date: 2026-07-21
tags: ["Kimi3", "linear-attention", "sparse-attention", "DSA", "MLA", "architecture", "long-context"]
excerpt: "Kimi3 represents the frontier of linear attention architectures — but can it scale? This post analyzes the micro-architecture war between linear attention (Kimi3, Qwen3) and sparse attention (DeepSeek V4, GLM52, LongCat), with quantitative analysis of their scaling ceilings."
---

# Understanding Kimi3 (1): Linear Attention, Sparse Attention, and the Architecture War at Million-Token Scale

## The Question That Matters

At million-token context lengths, the cost structure of AI inference fundamentally shifts. Our previous analysis showed that ==storage (KV-Cache) dominates the bill== — 67-78% of total cost across vendors. This reality makes one question critical: **which model micro-architecture can deliver the best quality-cost tradeoff at extreme sequence lengths?**

Two paths have emerged:

| Path | Representative | KV Strategy | Scaling Claim |
|---|---|---|---|
| **Linear Attention** | Kimi3, Qwen3, Xiaomi SWA | Compress via linear attention | ~linear with context |
| **Sparse Attention (DSA)** | DeepSeek V4, GLM52, LongCat | Sparse index + compression | ~near-linear with index |

This post dives deep into Kimi3's architecture, compares it with the DSA path, and offers a perspective on which approach has the higher scaling ceiling.

---

## 1. Kimi3 Architecture: The Linear Attention Frontier

### 1.1 Hybrid Architecture: KDA + ResNet + MLA

Kimi3 employs a ==layerwise hybrid architecture== combining three components:

| Component | Role | Sequence Scaling |
|---|---|---|
| **KDA (Kimi Delta Attention)** | Linear attention module | O(n) — linear with context |
| **MLA (Multi-head Latent Attention)** | Compressed full attention | O(n) with 32× compression |
| **ResNet** | Gradient path stabilization | Independent of sequence |

The key innovation is the ==3:1 ratio of KDA to MLA== — three linear attention layers for every one full attention layer. This hybrid design claims to reduce computation by ==75%== compared to pure full attention, while maintaining quality through strategic full-attention checkpoints.

### 1.2 KDA: The Core Linear Attention Module

At the heart of Kimi3 lies **Kimi Delta Attention (KDA)**, a hardware-efficient linear attention module that extends Gated DeltaNet with a ==channel-wise forget gate==.

Traditional linear attention uses a single scalar forget gate per head. KDA introduces finer-grained control:

```
Standard:  gate ∈ ℝ          (1 value per head)
KDA:       gate ∈ ℝ^C        (per-channel, C = channel dim)
```

This channel-wise gating allows the model to selectively retain or forget information at a granular level — critical for long-context tasks where different channels may carry information at different timescales.

### 1.3 MoE Sparsity: 1.8% Activation

Kimi3 pushes MoE sparsity to extreme levels:

| Model | Activated Params | Total Params | Sparsity |
|---|---|---|---|
| **Kimi3** | 3B | 48B | ==1.8%== |
| DeepSeek V4 Flash | — | — | ~1.8% |
| Typical MoE | 10-30% | — | 10-30% |

This ultra-low sparsity means only 1.8% of parameters are active per token — a design choice that trades training efficiency for inference cost savings.

---

## 2. The Two Architecture Paths

### 2.1 Linear Attention Path

**Adopters:** Kimi3, Qwen3 series, Xiaomi SWA

**Core idea:** Replace O(n²) full attention with O(n) linear attention, using recurrent state compression.

**Strengths:**
- True O(n) scaling for the linear portion
- No index structure needed
- Simpler hardware utilization

**Weaknesses:**
- ==Must retain full attention layers== to maintain quality (the 3:1 KDA:MLA ratio)
- Full attention layers are the ==scaling bottleneck== — they remain O(n²)
- Cross-media coordination (HBM → DRAM → SSD) is extremely difficult due to bandwidth requirements

### 2.2 DSA (Dense-Sparse Attention) Path

**Adopters:** DeepSeek V4, GLM52, LongCat

**Core idea:** Use sparse indexing to select only the most relevant tokens, achieving near-linear scaling while preserving full-attention quality.

**Strengths:**
- ==Near-linear scaling== with sparse index (S² complexity, still a bottleneck but better than O(n²))
- Naturally suited for ==cross-media coordination== (HBM/DRAM/SSD)
- DeepSeek V4 demonstrates real-world viability

**Weaknesses:**
- Sparse indexing introduces ==random memory access== patterns
- Index computation adds overhead (S² complexity)
- Hardware support for gather/scatter still maturing

### 2.3 The Scaling Ceiling: A Quantitative View

| Architecture | Compute Scaling | Memory Scaling | Cross-Media | Practical Ceiling |
|---|---|---|---|---|
| Full Attention | O(n²) | O(n) | Poor | ~128K |
| Linear (Kimi3) | O(n) for 75% of layers | O(n) | Poor | ~500K (estimated) |
| DSA (DeepSeek V4) | O(S²) + O(n×S) | O(n) | Good | ~1M+ |

> **Key insight:** Linear attention's full-attention layers (the 25% MLA portion) become the scaling ceiling. At 1M tokens, even 25% full attention is computationally expensive. DSA's sparse index, while not perfect, has a higher ceiling because it avoids the O(n²) wall entirely.

---

## 3. Efficiency Numbers: What Kimi's Linear Paper Actually Shows

### 3.1 The 2× Decoding Speedup Claim

From the Kimi Linear paper:

> "Our proposed KDA delivers competitive performance compared to the full-attention baseline (Table 3) and achieves ==over a 2× decoding speedup== at the one-million-token context (Figure 7b)."

**Experimental setup:**
- Model: 3B activated / 48B total parameters
- Baseline: Full attention
- Context: 1M tokens

**Critical analysis:**
- 2× speedup at 1M is modest — full attention at 1M is nearly infeasible, so the baseline is already degraded
- The comparison is against full attention, not against DSA architectures
- Real-world deployment constraints (batch size, memory bandwidth) are not captured

### 3.2 The 6× Decoding Throughput Claim

> "For decoding at 1M context length, Kimi Linear is ==6× faster== than full attention."

**Experimental setup:**
- Same 3B/48B model
- Pretrained on 1.4T tokens
- Batch size: 1 (2.2× speedup), larger batches show 6×

**Critical analysis:**
- 6× at batch=1 is impressive, but batch=1 is not production-realistic
- The comparison baseline (full attention at 1M) is already impractical
- No comparison against DSA architectures (DeepSeek, GLM, LongCat)

### 3.3 The 1.16× Computational Efficiency Claim

> "Kimi Linear achieves ∼==1.16× computational efficiency== compared to the MLA baselines with compute-optimal training."

**What this means:** To achieve the same model quality, Kimi Linear requires 1.16× less compute than MLA baselines.

**Critical analysis:**
- This is a training efficiency metric, not inference efficiency
- The 1.16× improvement is modest — within noise for most practical purposes
- Compute-optimal training involves many hyperparameters; the fair comparison is unclear
- ==This metric is not what inference engineers should care about==

### 3.4 Summary of Efficiency Claims

| Claim | Value | Baseline | Relevance |
|---|---|---|---|
| Decoding speedup @1M | 2× | Full attention | Modest — baseline already degraded |
| Decoding throughput @1M | 6× | Full attention (BS=1) | Good but not production-realistic |
| Compute efficiency | 1.16× | MLA (training) | Low — training metric, not inference |

> **Bottom line:** Kimi3's efficiency claims are real but measured against weak baselines. The absence of DSA comparisons makes it impossible to assess relative merit.

---

## 4. Deployment Reality: The 200K Wall

### 4.1 Advertised vs Actual Sequence Lengths

| Model | Advertised | Actual (Claude Code) | Constraint |
|---|---|---|---|
| DeepSeek V4 | 1M | ~200K | Client-side |
| GLM52 | 1M | ~200K | Client-side |
| LongCat | 1M | ~200K | Client-side |
| Kimi3 | 1M | Unknown | Not yet widely deployed |

**Observation:** All models hit the ==200K wall== in real-world deployment via Claude Code. This may be a client-side constraint (Claude Code's context window) rather than a model limitation.

### 4.2 Why 200K? The System-Level Bottleneck

Even if the model supports 1M, the system stack imposes constraints:

1. **Client-side context window:** Claude Code may limit input to ~200K
2. **Memory bandwidth:** At 1M tokens, even O(n) memory access saturates HBM bandwidth
3. **Latency:** Linear attention's recurrent state processing adds sequential latency
4. **Cost:** Our analysis showed $200+/session for Kimi at 480M tokens — 1M would be prohibitive

> **Reality check:** The gap between "supports 1M" and "usefully deploys at 1M" is vast. DeepSeek is the only vendor that has publicly demonstrated system-level scalability data.

---

## 5. The Cost of Memory: Why Architecture Choice Matters

### 5.1 The Agent Economy Is a Memory Economy

In multi-turn Agent scenarios:

| Metric | Value |
|---|---|
| Cache hit tokens | 480.4M (our measurement) |
| Compute tokens | 11.3M |
| Storage:Compute ratio | 42.7:1 |
| Cost share (Kimi) | 78% storage |
| Cost share (DeepSeek) | 21% storage |

> At million-token scale, ==you're buying memory, not compute==. The architecture that stores KV-Cache most efficiently wins.

### 5.2 DSA's Cross-Media Advantage

The memory hierarchy for million-token KV-Cache:

| Tier | Capacity | Bandwidth | Cost/GB | Architecture Fit |
|---|---|---|---|---|
| **HBM** | 80-128GB | 3-5 TB/s | $$$ | Hot cache (recent tokens) |
| **CXL/DRAM** | 1-4 TB | 100-200 GB/s | $$ | Warm cache (recent context) |
| **SSD/NVMe** | 10+ TB | 5-15 GB/s | $ | Cold cache (historical context) |

**DSA's advantage:** Sparse indexing naturally maps to this hierarchy:
- Index resides in HBM (small, fast)
- Selected tokens fetched from DRAM/SSD on demand
- ==Random access patterns are tolerable== with modern memory semantics

**Linear attention's problem:** Full-attention layers require ==all tokens in HBM simultaneously==:
- No natural tiering — every token must be accessible
- Cross-media coordination requires massive bandwidth
- Latency-sensitive scenarios (real-time chat) are especially challenging

### 5.3 The Full-Attention Bottleneck

Kimi3's 3:1 KDA:MLA ratio means 25% of layers are full attention. At 1M tokens:

```
Full attention compute per layer: O(n²) = O(10¹²)
Linear attention compute per layer: O(n) = O(10⁶)
Ratio: 10⁶× difference per layer
```

Even with MLA's 32× compression, the full-attention layers dominate at extreme scales. This is the ==fundamental ceiling== of hybrid linear architectures.

---

## 6. The Road Ahead: Convergence or Divergence?

### 6.1 Kimi3's Own Prediction

From the Kimi3 paper's future work section:

> "Future work could explore ==hybrid models== that integrate the strengths of both, leveraging the compression and generalization capabilities of linear attention with the fine-grained retrieval advantages of sparse attention."

This is a remarkable admission — the linear attention camp acknowledges that pure linear is not enough.

### 6.2 The Convergence Thesis

The industry is converging on a hybrid approach:

| Layer Type | Function | Scaling |
|---|---|---|
| **DSA (Sparse)** | Long-range retrieval | O(S²) — index-based |
| **MLA (Compressed)** | Mid-range attention | O(n) — compressed |
| **Linear (KDA)** | Local processing | O(n) — recurrent |

This hybrid combines:
- DSA's cross-media coordination capability
- MLA's compression efficiency
- Linear attention's local processing speed

### 6.3 Why DeepSeek Is Still Ahead

DeepSeek remains the only vendor that has demonstrated:
1. **System-level scalability data** — not just model claims
2. **Real-world deployment** at scale
3. **DSA + MLA + MTP integration** — all three innovations working together
4. **Industry adoption** — competitors are adopting DSA principles

> **Bottom line:** The architecture war is not over, but DSA has a higher ceiling. Linear attention (Kimi3) is impressive but faces fundamental limits at extreme scales. The future likely belongs to hybrid architectures that combine DSA's sparse indexing with linear attention's efficiency — and DeepSeek is best positioned to deliver this.

---

## 7. What to Watch Next

In the next post, we'll quantify Kimi3's actual cost-efficiency:
- Compute-optimal training cost analysis
- Inference cost projection at 1M tokens
- Comparison with DeepSeek V4's published efficiency data
- The "memory wall" — when does storage cost exceed compute cost?

---

*Based on Kimi Linear paper, DeepSeek V4 technical report, and real-world deployment data. All calculations reproducible — see data tables above.*
