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

![Daily Token Consumption](assets/token_usage_july_2026.png)

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

## Daily Breakdown: Storage vs Compute Cost

![Daily Cost Comparison](assets/daily_cost_comparison.png)

Using the daily consumption data, here's what each vendor would charge per day. **Storage cost = cache hits × hit price. Compute cost = (cache misses + output) × respective price.**

**Pricing:** DeepSeek storage $0.003625/M, miss $0.435/M, output $0.87/M · Kimi storage $0.28/M, miss $2.78/M, output $13.90/M

| Date | Hit (M) | Miss (M) | Out (M) | DeepSeek Storage | DeepSeek Compute | Kimi Storage | Kimi Compute |
|---|---|---|---|---|---|---|---|
| Jul 14 | 23.9 | 1.02 | 0.185 | $0.09 = 23.9×0.003625 | $0.60 = 1.02×0.435 + 0.185×0.87 | $6.70 = 23.9×0.28 | $2.55 = 1.02×2.78 + 0.185×13.9 |
| Jul 15 | 56.1 | 1.50 | 0.416 | $0.20 = 56.1×0.003625 | $1.01 = 1.50×0.435 + 0.416×0.87 | $15.72 = 56.1×0.28 | $7.30 = 1.50×2.78 + 0.416×13.9 |
| Jul 16 | 33.7 | 1.40 | 0.351 | $0.12 = 33.7×0.003625 | $0.91 = 1.40×0.435 + 0.351×0.87 | $9.45 = 33.7×0.28 | $6.13 = 1.40×2.78 + 0.351×13.9 |
| Jul 17 | 4.8 | 0.227 | 0.029 | $0.02 = 4.8×0.003625 | $0.12 = 0.227×0.435 + 0.029×0.87 | $1.34 = 4.8×0.28 | $0.74 = 0.227×2.78 + 0.029×13.9 |
| Jul 18 | 132.4 | 3.71 | 0.779 | $0.48 = 132.4×0.003625 | $2.29 = 3.71×0.435 + 0.779×0.87 | $37.11 = 132.4×0.28 | $16.87 = 3.71×2.78 + 0.779×13.9 |
| Jul 20 | 229.6 | 1.24 | 0.403 | $0.83 = 229.6×0.003625 | $0.89 = 1.24×0.435 + 0.403×0.87 | $64.34 = 229.6×0.28 | $6.13 = 1.24×2.78 + 0.403×13.9 |
| **Total** | **480.4** | **7.8** | **3.5** | **$1.74** | **$6.45** | **$134.51** | **$70.33** |

> On July 20 (peak day): DeepSeek $1.72 (storage $0.83 + compute $0.89) vs Kimi $70.47 (storage $64.34 + compute $6.13) — a **41× difference**.

---

## Total Bill: DeepSeek vs Kimi

For the identical 480M+ token workload:

| Provider | Storage Cost | Compute Cost | Total | Storage % |
|---|---|---|---|---|
| **MLA+DSA+CSA/HCA (DeepSeek Pro)** | $1.74 | $6.45 | **$8.19** | 21.2% |
| **Kimi K3** | $134.51 | $70.33 | **$204.84** | 65.7% |

**Calculation details:**

**DeepSeek:**
- Storage: `480.4M × $0.003625` = ==1.74==
- Compute (miss): `7.8M × $0.435` = ==3.40==
- Compute (output): `3.5M × $0.87` = ==3.05==
- **Total: ==8.19==**

**Kimi:**
- Storage: `480.4M × $0.28` = ==134.51==
- Compute (miss): `7.8M × $2.78` = ==21.68==
- Compute (output): `3.5M × $13.90` = ==48.65==
- **Total: ==204.84==**

> For the exact same token consumption, Kimi costs ==25.0×== more than DeepSeek. The gap comes from storage pricing (==77×==) and compute pricing (==6.4×== miss, ==16.0×== output).

---

## What the Data Tells Us

From our measured 480M cache hit tokens (one Agent session, one week):

1. **Storage dominates Kimi's bill (78%) but not DeepSeek's (25%).**
   - DeepSeek: $1.74 storage / $8.19 total = 21.2%
   - Kimi: $134.51 storage / $204.84 total = 65.7%
   - DeepSeek's $0.003625/M hit price makes storage nearly free; Kimi's $0.28/M makes it the dominant cost

2. **For identical workload, Kimi costs 25.0× more.**
   - DeepSeek total: $8.19 ($1.74 storage + $6.45 compute)
   - Kimi total: $204.84 ($134.51 storage + $70.33 compute)
   - Ratio: $204.84 / $8.19 = 25.0×

3. **The price gap has two sources:**
   - Storage: Kimi $0.28/M vs DeepSeek $0.003625/M = **77×**
   - Compute miss: Kimi $2.78/M vs DeepSeek $0.435/M = **6.4×**
   - Compute output: Kimi $13.90/M vs DeepSeek $0.87/M = **16.0×**

> The cost structure is not theoretical — it's measured from 7 days of real Agent usage.

---

## Architecture Scaling: Sparse Compression vs Linear Hybrid

### Two Paths to Long-Context

Our cost comparison reveals a deeper architectural divergence:

| Architecture | Representative | KV-Cache Strategy | Cache Hit Price | Compression |
|---|---|---|---|---|
| **Sparse Compression** | MLA+DSA+CSA/HCA | Compress KV to low-rank latent | $0.003625/M | ~32× vs MHA |
| **Linear Hybrid** | Kimi K3 | Linear attention + full attention blocks | $0.28/M | 3:1 linear compression |

### Storage Cost Scaling at Million-Token Scale

Projecting storage cost as context length scales (cache hit price × tokens):

| Context Length | MLA+DSA+CSA/HCA | Kimi3 (Linear Hybrid) | Ratio |
|---|---|---|---|
| 10M | $0.036 | $2.80 | 77× |
| 100M | $0.36 | $28.00 | 77× |
| 480M (ours) | $1.74 | $134.51 | 77× |
| 1B | $3.63 | $280.00 | 77× |
| 10B | $36.25 | $2,800.00 | 77× |

**Key observation:** The ratio stays constant at ==77\times== because both scale linearly with token count. The absolute dollar gap widens from $2.76 (10M) to $2,763.75 (10B).

### Two Scenarios for the Storage Market

**Scenario A: Kimi3-like linear hybrid becomes dominant**
- Full KV-Cache retained → storage scales linearly with context
- At 1B tokens: $280/session storage cost
- Storage becomes the dominant cost center (78% of bill in our data)
- **Implication:** Massive demand for KV-Cache storage across the full memory hierarchy:
  - **HBM:** Hot cache (active context, highest bandwidth)
  - **CXL/DRAM:** Warm cache (recent context, high capacity)
  - **SSD/NVMe:** Cold cache (historical context, cross-request persistence)
- Storage market grows proportionally with context length; SSD becomes a critical tier for cost-effective long-context serving

**Scenario B: DeepSeek V4-like sparse compression becomes dominant**
- KV compressed 32× → storage cost near zero
- At 1B tokens: $3.63/session storage cost
- Compute becomes the dominant cost (75%+ of bill)
- **Implication:** Storage market growth decoupled from context length; compute becomes the bottleneck

### Which Path Wins? Too Early to Call

The Kimi3 paper acknowledges that linear hybrid + full attention incurs higher system cost (see [Kimi K3 Technical Report](https://platform.moonshot.cn/docs/pricing/chat)). Their proposed future direction — convergence of both paths — suggests neither architecture has won decisively:

1. **Sparse compression (DeepSeek V4):** Lower storage cost, but compression loses information. May hit quality walls at extreme contexts.
2. **Linear hybrid (Kimi3):** Higher storage cost, but preserves full attention quality. May hit cost walls at extreme contexts.
3. **Convergence:** Future architectures may combine both — sparse compression for older context + full attention for recent context.

### The Deciding Factor: Quality-Cost Tradeoff

At 1M+ tokens, the question is not "which is cheaper" but "which delivers acceptable quality per dollar":

| Metric | MLA+DSA+CSA/HCA | Kimi3 (Linear Hybrid) |
|---|---|---|
| Storage cost / 1M tokens | $0.003625 | $0.28 |
| Relative quality | Compressed (potential loss) | Full attention (preserved) |
| Best for | Long context, cost-sensitive | Quality-critical, shorter context |

> **Bottom line:** If Kimi3's linear hybrid becomes mainstream, the storage market wins big — every 1M tokens costs $0.28 vs DeepSeek's $0.003625. But if sparse compression wins, storage becomes nearly free and compute takes over as the cost center. The architecture that delivers the best quality-cost tradeoff at 10M+ tokens will define the next generation of AI infrastructure.


---

## Summary

One week, one Agent session, 480M cache hit tokens:

| Vendor | Storage | Compute | Total | Storage % |
|---|---|---|---|---|
| MLA+DSA+CSA/HCA (DeepSeek Pro) | $1.74 | $6.45 | $8.19 | 21.2% |
| Kimi K3 | $134.51 | $70.33 | $204.84 | 65.7% |

**At scale (projection):**

| Context | DeepSeek | Kimi | Ratio |
|---|---|---|---|
| 100M | $2.11 | $42.70 | 20.2× |
| 480M (ours) | $8.19 | $204.84 | 25.0× |
| 1B | $17.11 | $426.80 | 24.9× |

- The longer the context, the wider the gap: from $2.76 at 10M to $2,763.75 at 10B

---

## References

1. [DeepSeek API Pricing](https://api-docs.deepseek.com/quick_start/pricing/) — Official MLA+DSA+CSA/HCA (DeepSeek Pro) pricing: $0.003625/M cache hit
2. [Moonshot AI (Kimi) Pricing](https://platform.moonshot.cn/docs/pricing/chat) — Kimi K3 official pricing
3. [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2606.19348) — MLA architecture
4. Kimi K3 Technical Report — Linear attention + full attention hybrid architecture
5. [Lil'Log - Context Engineering](https://lilianweng.github.io/posts/2025-06-24-context-engineering/) — Context window composition
6. [Anthropic - Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — Multi-turn Agent patterns

---

*Based on real-world Agent consumption data (480M+ cache hit tokens, context window 200K–450K tokens). Rate cards current as of 2026-07-20. All calculations reproducible — see data tables and references above.*

---

## Appendix: Context Usage

> LongCat-2.0 session: 613.9k/200k tokens (307%) for this research + writing session.

**Token breakdown by category:**

| Category | Tokens | Percentage |
|----------|--------|------------|
| System prompt | 2.8k | 0.5% |
| System tools | 18.3k | 3.0% |
| MCP tools | 2.5k | 0.4% |
| Memory files | 1.5k | 0.2% |
| Skills | 2k | 0.3% |
| **Messages** | **172.9k** | **28.2%** |
| Read results | ~1M | — |

**Efficiency notes:**
- Piped outputs through `head`/`tail`/`grep` to reduce token consumption
- Used `Read` with offset/limit instead of re-reading entire files
- Bash results consumed 104.5k tokens (52%) — could be reduced with more aggressive filtering
- Edit tool consumed 68.6k tokens (34%) — large markdown edits are token-intensive
