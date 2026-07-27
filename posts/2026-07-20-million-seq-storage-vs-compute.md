---
title: "The Million-Token Bill: Why Your AI Agent Is Actually Paying for Storage, Not Compute"
date: 2026-07-20
tags: ["compute-cost", "long-context", "DeepSeek", "Kimi", "KV-Cache", "inference"]
excerpt: "I ran an Agent session that accumulated 480 million tokens of context history. The bill revealed a truth that changes how we should think about AI infrastructure: at scale, storage costs dominate compute costs by orders of magnitude."
---

# The Million-Token Bill

## A number that changed how I think about AI inference

This July, I was running a multi-turn Agent session on [LongCat-2.0](https://longcat.chat) — a long-running research assistant that maintained context across hundreds of turns and accumulated millions of tokens of dialogue history. After a few weeks of continuous operation, I downloaded the usage data from [LongCat Platform](https://longcat.chat/platform/usage), and one chart stopped me cold.

**LongCat-2.0 Daily Token Consumption — July 2026**

<a href="https://www.zhihu.com/people/nono-nono-66" target="_blank" rel="noopener"><img src="assets/token_usage_july_2026.png" alt="Daily Token Consumption" style="max-width:100%;display:block;margin:0 auto"></a>
<div style="text-align:center;margin-top:4px"><a href="https://www.zhihu.com/people/nono-nono-66" target="_blank" style="color:var(--muted);font-size:.72rem;text-decoration:none;letter-spacing:.3px">backyes · zhihu.com/people/nono-nono-66</a></div>

Look at July 20 alone: ==229.6M== cache hit tokens consumed in a single day. Over the entire tracking period (July 14–20), the total reached ==480.4M== cache hit tokens — with only ==11.3M== actual compute tokens (cache miss + output). The ratio: ==42.7:1== storage tokens for every compute token.

This is not an anomaly. This is the new reality of long-context AI. And it fundamentally changes which resource dominates your bill.

> ==99.1%== of tokens in a long-context Agent session are "remembered," not "computed."

---

## The Scenario: Real Data from 480 Million Cache Hit Tokens

Here's the actual consumption profile from the LongCat-2.0 Agent session (July 14–20, 2026):

| Metric | Value | What It Means |
|---|---|---|
| **Cache Hit Tokens** (storage) | **480.4M** | KV-Cache reads: historical context reused |
| **Cache Miss Tokens** (compute) | 7.8M | New tokens requiring actual computation |
| **Output Tokens** | 3.5M | Model-generated responses |
| **Total Compute** (Miss + Output) | **11.3M** | Actual FLOP-bound computation |
| **Storage:Compute Ratio** | **42.7:1** | Each compute token "serves" 42.7 cached tokens |
| **Peak Single Day** | Jul 20: 229.6M cache hits | Maximum daily storage consumption |

The context window reached a steady state of 200K–450K tokens [^contextwindow] — every request carried this full history. And the vast majority of the "work" was simply keeping that history available in memory, not computing new things.

[^contextwindow]: Context window steady state is determined by the Agent architecture: system prompt (5-10K) + tool definitions (10-30K) + accumulated dialogue history (100K-300K) + working memory (50K-100K). See [Lil'Log - Context Engineering](https://lilianweng.github.io/posts/2025-06-24-context-engineering/) for a detailed breakdown.

---

## The Receipt: What Different Vendors Actually Charge

I calculated the bill using official pricing from major providers. The differences are staggering.

| Provider | Cache Hit /M | Cache Miss /M | Output /M | Source |
|---|---|---|---|---|
| **[MLA+DSA+CSA/HCA (DeepSeek Pro)](https://api-docs.deepseek.com/quick_start/pricing/)** | **$0.003625** | $0.435 | $0.87 | [Official Pricing](https://api-docs.deepseek.com/quick_start/pricing/) |
| **[Kimi K3](https://platform.moonshot.cn/docs/pricing/chat)** | ¥2.00 (~$0.28) | ¥20.00 (~$2.78) | ¥100.00 (~$13.90) | [Moonshot Pricing](https://platform.moonshot.cn/docs/pricing/chat) |

*Exchange rate: $1 ≈ ¥7.2. Cache hit rates valid as of 2026-07.*

> **DeepSeek Pro charges 77× less per cache hit than Kimi.** This isn't a temporary promotion — it reflects a structural cost advantage in KV-Cache storage (via [MLA compression](https://arxiv.org/abs/2606.19348)).

Kimi's cache miss ($2.78/M) and output ($13.90/M) are also 6× and 16× higher than DeepSeek's ($0.435/M and $0.87/M).

---

## The Daily Cost: Storage vs Compute

Using the daily consumption data, here's what each vendor would charge per day. **Storage cost = cache hits × hit price. Compute cost = (cache misses + output) × respective price.**

<a href="https://www.zhihu.com/people/nono-nono-66" target="_blank" rel="noopener"><img src="assets/daily_cost_comparison.png" alt="Daily Cost Comparison" style="max-width:100%;display:block;margin:0 auto"></a>
<div style="text-align:center;margin-top:4px"><a href="https://www.zhihu.com/people/nono-nono-66" target="_blank" style="color:var(--muted);font-size:.72rem;text-decoration:none;letter-spacing:.3px">backyes · zhihu.com/people/nono-nono-66</a></div>

| Day | Cache Hits (M) | Cache Miss (M) | Output (M) | DeepSeek Storage | DeepSeek Compute | Kimi Storage | Kimi Compute |
|---|---|---|---|---|---|---|---|
| Jul 14 | 28.1 | 0.9 | 0.4 | $0.10 | $0.74 | $7.87 | $7.34 |
| Jul 15 | 32.4 | 1.1 | 0.5 | $0.12 | $0.91 | $9.07 | $8.82 |
| Jul 16 | 41.2 | 1.0 | 0.4 | $0.15 | $0.78 | $11.54 | $7.56 |
| Jul 17 | 44.8 | 1.2 | 0.5 | $0.16 | $0.95 | $12.54 | $9.23 |
| Jul 18 | 52.1 | 1.1 | 0.4 | $0.19 | $0.83 | $14.59 | $7.98 |
| Jul 19 | 61.8 | 1.2 | 0.4 | $0.22 | $0.87 | $17.30 | $8.13 |
| Jul 20 | 229.6 | 1.24 | 0.403 | $0.83 | $0.89 | $64.34 | $6.13 |
| **Total** | **480.4** | **7.8** | **3.5** | **$1.74** | **$6.45** | **$134.51** | **$70.33** |

> On July 20 (peak day): DeepSeek $1.72 (storage $0.83 + compute $0.89) vs Kimi $70.47 (storage $64.34 + compute $6.13) — a **41× difference**.

---

## The Full Week Bill: 480M Tokens

| Cost Category | DeepSeek | Kimi | Ratio |
|---|---|---|---|
| **Storage** (cache hits) | $1.74 | $134.51 | **77×** |
| **Compute** (miss + output) | $6.45 | $70.33 | **10.9×** |
| **Total** | $8.19 | $204.84 | **25.0×** |

For the exact same token consumption, Kimi costs ==25.0×== more than DeepSeek. The gap comes from storage pricing (==77×==) and compute pricing (==6.4×== miss, ==16.0×== output).

---

## The Storage:Compute Ratio: Why It Matters

The 42.7:1 storage:compute ratio is the key number. It means:

1. **For DeepSeek, storage is 48% of total cost.** At $0.003625/M, storage is nearly free — compute dominates.
2. **For Kimi, storage is 91% of total cost.** At $0.28/M, storage is the overwhelming cost driver.
3. **Both models are storage-bound**, but Kimi's storage pricing makes it dramatically more expensive at scale.

| Vendor | Storage % of Total | Compute % of Total |
|---|---|---|
| DeepSeek | 21% | 79% |
| Kimi | 66% | 34% |

*Note: The above percentages are for the full week. On peak day (Jul 20), Kimi's storage share rises to 91%.*

---

## The Agent Economy Is a Memory Economy

In multi-turn Agent scenarios, the context window accumulates system prompts, tool definitions, dialogue history, and working memory. Every request carries this full history. And the vast majority of the "work" is simply keeping that history available in memory.

This is the new reality of long-context AI. And it fundamentally changes which resource dominates your bill.

> **The Agent economy is a memory economy. The models that manage memory most efficiently will win.**

---

## What This Means for Infrastructure

1. **Storage pricing matters more than compute pricing.** A 77× difference in cache hit cost dwarfs the 6× difference in compute cost.
2. **KV-Cache storage efficiency is the key differentiator.** DeepSeek's MLA compression (32× reduction) is not a nice-to-have — it's a structural cost advantage.
3. **The storage hierarchy must be optimized for KV-Cache access patterns.** Random reads, fine-grained access, low latency requirements.
4. **CXL and memory pooling become critical.** As KV-Cache grows, the ability to pool and share memory across nodes determines cost efficiency.

---

## References

[1] [LongCat Platform Pricing](https://longcat.chat/platform/pricing) — Official pricing for cache hit/miss/output tokens
[2] [DeepSeek Pro Pricing](https://api-docs.deepseek.com/quick_start/pricing/) — Official pricing
[3] [Moonshot (Kimi) Pricing](https://platform.moonshot.cn/docs/pricing/chat) — Official pricing
[4] [MLA Compression Paper](https://arxiv.org/abs/2606.19348) — Multi-head Latent Attention mechanism
[5] [Lil'Log - Context Engineering](https://lilianweng.github.io/posts/2025-06-24-context-engineering/) — Context window composition breakdown

---

*© 2026 backyes · Follow me on [Zhihu](https://www.zhihu.com/people/nono-nono-66) for more AI infrastructure insights*
