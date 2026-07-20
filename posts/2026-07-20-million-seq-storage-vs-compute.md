---
title: "The Million-Token Bill: Why Your AI Agent Is Actually Paying for Storage, Not Compute"
date: 2026-07-20
tags: ["compute-cost", "long-context", "DeepSeek", "Kimi", "KV-Cache", "inference"]
excerpt: "I ran an Agent session that accumulated 160 million tokens of context history. The bill revealed a truth that changes how we should think about AI infrastructure: at scale, storage costs dominate compute costs by orders of magnitude."
---

# The Million-Token Bill

## A number that changed how I think about AI inference

Last week, I was reviewing the consumption logs of a production multi-turn Agent session. The task was routine — a long-running research assistant that maintained context across hundreds of turns. But one number stopped me cold.

**160.4 million cache hit tokens.**

That's not the total tokens processed. That's just the tokens *recalled from KV-Cache memory* — the historical context that the model "remembered" rather than recomputed. The actual compute (cache misses + output)? Just 1.467 million tokens.

The ratio: **110 storage tokens for every 1 compute token.**

This is the new reality of long-context AI. And it fundamentally changes which resource dominates your bill.

> **99.1% of tokens in a long-context Agent session are "remembered," not "computed."**

---

## The Scenario: One Million Tokens of Remembered Context

Here's the real-world consumption profile from that session:

| Metric | Value | What It Means |
|---|---|---|
| Cache Hit Tokens | 160.4M | KV-Cache reads: historical context reused |
| Cache Miss Tokens | 1.02M | New tokens requiring actual computation |
| Output Tokens | 0.447M | Model-generated responses |
| Total Compute | 1.467M | What you'd traditionally call "processing" |
| Storage:Compute Ratio | ~110:1 | Each compute token "serves" 110 cached |

The context window had reached a steady state of **200K–450K tokens** — every request carried this full history. And the vast majority of the "work" was simply keeping that history available in memory, not computing new things.

---

## The Receipt: What Different Vendors Actually Charge

I calculated the bill using official pricing from four major providers. The differences are staggering.

| Provider | Cache Hit /M | Cache Miss /M | Output /M |
|---|---|---|---|
| **DeepSeek Pro** | **¥0.025** | ¥3.00 | ¥6.00 |
| Kimi K3 | ¥2.00 | ¥20.00 | ¥100.00 |
| Claude Sonnet 4.5 | ¥21.6 | ¥21.6 | ¥108 |
| GPT-5 | ¥9.0 | ¥9.0 | ¥72 |

Read that again. **DeepSeek charges 80× less per cache hit than Kimi.** This isn't a temporary promotion — it reflects a structural cost advantage rooted in architecture.

### So, what did each vendor charge for the identical workload?

**DeepSeek Pro:**
- Storage cost: 160.4M × ¥0.025 = **¥4.01**
- Compute cost: 1.467M × ~¥4.00 = **¥5.74**
- **Total: ¥9.42** (storage is 43% of bill)

**Kimi K3:**
- Storage cost: 160.4M × ¥2.00 = **¥320.80**
- Compute cost: 1.467M × ~¥42.00 = **¥65.10**
- **Total: ¥377.03** (storage is 85% of bill)

**For the exact same token consumption, Kimi costs 40× more than DeepSeek.**

> Kimi K3's storage bill alone (¥320.80) is **80× DeepSeek's storage bill** (¥4.01). This is the price of architectural choices.

---

## The Visualization: Two Different Cost Philosophies

```
DEEPSEE P PRO                         KIMI K3

Storage ████ ¥4.01 (43%)             Storage ██████████████████ ¥320.80 (85%)
Compute █████ ¥5.74 (57%)            Compute ███ ¥65.10 (15%)

Total: ¥9.42                         Total: ¥377.03
```

DeepSeek has achieved **cost inversion** — compute now costs more than storage. This is the hallmark of an architecture optimized for the long-context era.

Kimi (and most traditional models) operate under the old paradigm: storage is expensive, so your bill is dominated by memory costs at scale.

---

## The Root Cause: One Architecture Choice, 80× Cost Difference

The 80× gap in cache hit pricing traces back to a single design decision: **how to store the KV-Cache.**

| Architecture | KV-Cache Size / Token | Compression | Used By |
|---|---|---|---|
| **Multi-head Latent Attention (MLA)** | ~576 bytes | ~32× | DeepSeek |
| Grouped-Query Attention (GQA) | ~4.5 KB | ~4× | LLaMA-3, Qwen |
| Standard Multi-head Attention (MHA) | ~18 KB | 1× | Most others |

DeepSeek's MLAttention projects the full key-value cache into a low-rank latent space, compressing it by ~32×. This means:

- 32× less memory capacity needed per token
- 32× less bandwidth to read the cache
- Marginal storage cost approaches zero at scale

> The 128× difference in KV-Cache size translates to 80× difference in pricing — the rest is economics of scale.

---

## Why This Matters More Than You Think

### The Agent Economy Is a Storage Economy

For multi-turn Agent systems — the architecture behind every coding assistant, research agent, and autonomous workflow — the primary cost driver is **not** "how smart is the model" but "how cheaply can it remember what happened 100 turns ago."

Consider the scaling:

| Context Length | DeepSeek Bill | Kimi Bill | Ratio |
|---|---|---|---|
| 1M tokens | ¥0.58 | ¥22 | 38× |
| 10M tokens | ¥5.74 | ¥220 | 38× |
| 100M tokens | ¥57.40 | ¥2,200 | 38× |
| 160M tokens | ¥9.42 | ¥377 | 40× |

The gap *widens* with context length. For any serious Agent deployment — million-token contexts are becoming standard — the economics are decisive.

### The Infrastructure Implication

This pricing reality cascades up the stack:

- **Hardware procurement**: Memory bandwidth and capacity matter more than FLOP density
- **System design**: KV-Cache reuse across sessions becomes the primary optimization target
- **Vendor selection**: Cache hit pricing should weight more than compute pricing in RFPs

---

## What Comes Next

**Near-term (2026–2027):** Expect competitors to either subsidize cache pricing (taking a loss on storage) or accelerate MLA-like compression research. The ¥2.00/M cache hit price point is economically unsustainable at scale.

**Medium-term (2027–2029):** CXL-based memory expansion enables TB-scale KV-Cache per node, further driving down per-token storage costs. Persistent KV-Cache across sessions makes marginal cost approach zero.

**Long-term:** The distinction between "storage" and "compute" dissolves. Processing-in-memory architectures treat KV-Cache as the primary compute resource — reading *is* computing.

---

## The Bottom Line

I started this investigation curious about one number: 160 million cache hit tokens. It revealed a fundamental shift in how AI inference economics work.

Three conclusions:

1. **At million-token scale, you're buying storage, not compute** — 85% of a traditional bill goes to memory
2. **Architecture determines economics** — MLA's 32× compression translates to 80× pricing advantage
3. **The gap widens with scale** — the longer your context, the more you pay for inefficient storage

The future belongs to systems that treat memory as the primary compute resource. DeepSeek has shown it's possible. The rest of the industry will follow — or pay 40× more.

---

*Based on real-world consumption data (160M+ cache hit tokens). Rate cards current as of July 2026. All calculations are reproducible — see the data tables above.*

*— backyes | 2026-07*
