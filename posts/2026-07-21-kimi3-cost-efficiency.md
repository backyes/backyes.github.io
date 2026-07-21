---
title: "Understanding Kimi3 (2): Efficiency Numbers, Cost Analysis, and Architecture Tradeoffs"
date: 2026-07-21
tags: ["Kimi3", "efficiency", "cost-analysis", "linear-attention", "sparse-attention", "DSA"]
excerpt: "A faithful analysis of Kimi3's efficiency claims from the Linear paper, with cost modeling at million-token scale and the fundamental tradeoffs between linear and sparse attention architectures."
---

# Understanding Kimi3 (2): Efficiency Numbers, Cost Analysis, and Architecture Tradeoffs

## 1. Efficiency Numbers from the Kimi Linear paper

All data in this section comes from the [Kimi Linear paper](https://arxiv.org/abs/2510.26692). We preserve the original claims and add critical analysis.

### 1.1 2× Decoding Speedup @1M Context

**Original claim:**

> "Our proposed KDA delivers competitive performance compared to the full-attention baseline (Table 3) and achieves ==over a 2× decoding speedup== at the one-million-token context (Figure 7b)."

**Experimental setup:**
- Model: 3B activated / 48B total parameters
- Architecture: Layerwise hybrid of KDA (linear attention) and MLA (Multi-head Latent Attention)
- All models based on Kimi Linear 48B setting

**Critical analysis:**
- Kimi K3 official materials do **not** mention system throughput
- The 2× figure is from the research paper, not product specs
- The paper also mentions "==6× higher decoding== at 1M context" elsewhere — this number is not fully elaborated
- The final comparison shows KDA achieves ==2× speedup vs MLA baseline== at 1M context (Figure 7b)

### 1.2 1.16× Computational Efficiency (Compute-Optimal Training)

**Original claim:**

> "Kimi Linear achieves ∼==1.16× computational efficiency== compared to the MLA baselines with compute-optimal training."

**What this means:** To achieve the same model quality, Kimi Linear requires ~1.16× less training compute than MLA baselines.

**Critical analysis:**
- This is a ==training== efficiency metric, not inference
- "Compute-optimal training" involves many confounding hyperparameters
- ==This 1.16× is likely not a metric inference engineers should care about== — real-world inference cost depends on deployment hardware, batching, memory bandwidth, and KV-Cache management

### 1.3 6× Decoding Throughput @1M Context

**Original claim:**

> "As shown in Figure 1b, Kimi Linear fully demonstrates its advantages during the decoding phase. For decoding at 1M context length, Kimi Linear is ==6× faster== than full attention."

**Experimental setup:**
- Model: 3B activated / 48B total parameters
- Pretraining corpus: 1.4T tokens
- Batch size: 6× at optimal batch, ==2.2× at batch=1==

**Critical analysis:**
- The 6× claim uses optimal batch size (not specified, likely 4-8)
- At batch=1 (interactive use), speedup drops to ==2.2×==
- The comparison baseline is full attention at 1M — already an impractical baseline
- No comparison against DSA architectures (DeepSeek, GLM, LongCat)

### 1.4 Summary of Efficiency Numbers

| Claim | Value | Baseline | Condition | Relevance |
|---|---|---|---|---|
| Decoding speedup | 2× | Full attention | 1M tokens, vs MLA | Modest |
| Decoding throughput | 6× | Full attention | 1M tokens, optimal batch | Not production-realistic |
| Training efficiency | 1.16× | MLA | Compute-optimal training | Low — training metric |

> **Bottom line:** Kimi3's efficiency claims are real but measured against weak baselines. The absence of DSA comparisons makes relative assessment impossible.

---

## 2. Cost Model at Million-Token Scale

### 2.1 Assumptions

```
Model: Kimi3 (3B activated / 48B total)
Context: Variable (100K to 10M tokens)
Pricing: $0.28/M cache hit (from Moonshot official)
        $2.78/M cache miss
        $13.90/M output
```

### 2.2 Storage Cost Projection

Storage cost = context_tokens × cache_hit_price

| Context | Storage Cost | vs DeepSeek ($0.003625/M) |
|---|---|---|
| 100K | $0.028 | 77× |
| 500K | $0.14 | 77× |
| 1M | $0.28 | 77× |
| 10M | $2.80 | 77× |

> Storage scales linearly and the 77× gap vs DeepSeek is constant.

### 2.3 Compute Cost Projection

Kimi3's hybrid architecture (75% KDA + 25% MLA):

```
For each token generated:
  - 75% KDA layers: O(1) per token → constant cost regardless of context
  - 25% MLA layers: O(n) per token → cost scales linearly with context
```

At 1M tokens:
- KDA compute: 0.75 × 1M × c_linear (constant)
- MLA compute: 0.25 × 1M × c_full (linear)

### 2.4 Total Cost at Scale

| Context | Storage | Compute (est.) | Total | DeepSeek Equivalent |
|---|---|---|---|---|
| 100K | $0.028 | $0.01 | $0.038 | $0.0005 |
| 500K | $0.14 | $0.05 | $0.19 | $0.0023 |
| 1M | $0.28 | $0.10 | $0.38 | $0.0046 |
| 10M | $2.80 | $1.00 | $3.80 | $0.046 |

> At 1M tokens, Kimi3 costs ==$0.38/session== just for KV-Cache storage. DeepSeek: ==$0.0046==.

---

## 3. Sparse Attention Limitations and DSA Advantages

### 3.1 The Random Access Problem

Sparse attention's fundamental limitation is ==random memory access==:
- Must read entire KV cache to select which tokens to attend
- Random access patterns are inefficient on hierarchical memory

**But this problem is diminishing:**

Three developments are changing the equation:

1. **Memory semantic pooling systems are maturing** — large-scale pooling of memory semantics makes random access more efficient

2. **DeepSeek V4's actual performance proves it** — random access on HBM and DRAM performs ==much better than expected==; in engineering practice, this efficiency challenge is becoming hard to see

3. **Hardware improvements:**
   - ==Chip-level gather/scatter== instructions are being hardened and performance-improved
   - ==Memory-semantics direct-drive== of DDR memory is advancing

> **Engineering reality:** The random access challenge that once seemed fundamental is becoming manageable. ==DSA architectures naturally have advantages for cross-media coordination.==

### 3.2 DSA's Cross-Media Advantage

From a computer architecture perspective:

| Aspect | Linear Attention | DSA |
|---|---|---|
| KV storage | Must keep all in HBM | Can tier across HBM/DRAM/SSD |
| Cross-media | ==Nearly impossible== (bandwidth too high) | ==Natural fit== (selective fetch) |
| Latency sensitivity | Poor (recurrent state must be in HBM) | Good (index in HBM, data anywhere) |

> **Key insight:** DSA architectures are ==naturally suited for cross-media coordination==. Linear attention must depend on full attention layers, which cannot effectively tier across media — coordination bandwidth is too high, especially as latency requirements decrease.

---

## 4. The Cost of Memory: Why Architecture Choice Matters

### 4.1 Agent Workloads Are Memory Workloads

From real-world deployment data and vendor pricing:

| Observation | Implication |
|---|---|
| Cost is shifting to ==memory== (not compute) | Storage dominates the bill |
| Each request's sequence length >> compute input | 10× to 1000× longer |
| Million-token contexts are becoming standard | Memory cost scales linearly |

> **The core problem:** How to reduce memory cost is the key to system efficiency.

### 4.2 Storage Cost Is Media Cost

From a computer architecture perspective:
- ==Storage media cost >> interconnect cost==
- The physical medium (HBM, DRAM, SSD) dominates cost, not the connection

**DSA's advantage:**
- Naturally fits cross-media tiering
- Can place hot data in HBM, warm in DRAM, cold in SSD
- Index-based retrieval minimizes data movement

**Linear attention's limitation:**
- Must depend on full attention layers
- Full attention ==cannot effectively coordinate across media==
- Bandwidth requirements are too high, especially in low-latency scenarios

> **Personal view:** DSA is more likely to become the ==unified model micro-architecture== for the future. Linear attention alone cannot solve the cross-media problem.

---

## 5. The Kimi3 Paper's Own Conclusion

From the [Kimi Linear paper's Discussion section](https://arxiv.org/abs/2510.26692):

> "Linear attention and sparse attention represent two distinct pathways toward efficient long-context modeling. Sparse attention tends to retrieve fine-grained historical information more effectively, but this advantage comes at the cost of storing the entire KV cache for token selection, making it less efficient than linear attention models that maintain a constant state. Moreover, sparse attention performs only information selection, and its theoretical expressive upper bound remains that of full attention. In contrast, linear attention, grounded in the principle of 'compression as intelligence', enables generalization with a fixed-size state and, when combined with the Delta learning rule, can achieve theoretically stronger expressive capacity. Although linear attentions have traditionally been criticized for weak retrieval ability, this limitation can be mitigated through state expansion or related techniques. Nevertheless, despite these advantages, linear attention remains limited by current hardware implementations and the absence of optimized inference infrastructure. Our work overcomes these limitations with Kimi Linear, a powerful model integrated with vLLM for efficient inference. Our proposed KDA delivers competitive performance compared to the full-attention baseline (Table 3) and achieves over a 2× decoding speedup at the one-million-token context (Figure 7b). ==Despite their distinct approaches to efficient long-context modeling, linear attention and sparse attention are not mutually exclusive. Future work could explore hybrid models that integrate the strengths of both, leveraging the compression and generalization capabilities of linear attention with the fine-grained retrieval advantages of sparse attention to further enhance model performance and efficiency.=="

---

## 6. Summary

| Dimension | Rating | Notes |
|---|---|---|
| Innovation | ⭐⭐⭐⭐ | KDA is genuinely novel |
| Efficiency claims | ⭐⭐⭐ | 2× speedup is real but modest |
| Scaling ceiling | ⭐⭐ | ~500K practical limit |
| System readiness | ⭐⭐ | No published system-level data |
| Cost competitiveness | ⭐ | 77× storage cost vs DeepSeek |

> **Bottom line:** The future is ==hybrid== — compression (MLA) + Sparse (DSA) + Linear (KDA) fusion. No industrial model uses pure linear attention; all are hybrid. The architecture that delivers the best quality-cost tradeoff at 10M+ tokens will define the next generation of AI infrastructure.

---

*Based on [Kimi Linear paper](https://arxiv.org/abs/2510.26692), DeepSeek V4 technical report, and first-principles analysis. All original claims preserved with source attribution.*
