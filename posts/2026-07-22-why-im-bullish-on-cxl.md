---
title: "[Draft] Why I'm Bullish on CXL — A Market Game Theory Perspective"
date: 2026-07-22
tags: ["CXL", "UALink", "Memory-Pooling", "AI-Inference", "Marvell", "Broadcom", "Interconnect", "Open-Standard"]
excerpt: "CXL is not just another interconnect protocol. From a market game theory lens, I believe CXL's long-term positioning is structurally underappreciated. Four forces are converging to make CXL the inevitable backbone of open-memory infrastructure."
---

# [Draft] Why I'm Bullish on CXL — A Market Game Theory Perspective

## The setup

The CXL component market was worth roughly ==$1.3–2.1B== in 2025. Forecasts put it at ==$12.3B== by 2030 — a ~32% CAGR [1]. That's strong growth, but the number itself isn't why I'm bullish. Plenty of technologies have strong TAM projections and still fail.

What makes CXL different is its *position in the market game*. Four structural forces are converging, and CXL sits at the intersection of all of them.

---

## 1. Storage is marching into the "bus domain"

Here's the fundamental shift: AI training and inference are forcing a structural reorganization of the entire memory-storage stack.

In the old world, storage sits *below* the compute — you pull data up through PCIe, process it, write it back. The hierarchy was clean: DRAM → SSD → HDD → Tape.

That hierarchy is collapsing.

The physics are simple: at million-token context lengths and trillion-parameter MoE models, the working dataset doesn't fit in one box. It barely fits in one rack. Data must be *infinitely close* to compute — not just physically, but topologically. Every hop adds latency. Every copy wastes power.

> ==Storage is no longer a tier below compute. It's becoming a peer on the bus.==

This is what I mean by "泛存储全面总线化" — all forms of storage are being pulled onto the interconnect fabric as first-class citizens. NVMe over Fabrics was step one. CXL is step two: memory itself becomes a fabric-attached resource.

CXL 4.0 (sampling in 2026) enables ==100+ TB== shared memory pools with cache-coherent access [2]. That's not a niche accelerator feature — that's a fundamental re-architecture of the server.

The trend is irreversible because the physics are irreversible. AI models will only get bigger. Context windows will only get longer. The bus is the only place where the access latency works.

---

## 2. The open camp needs a counterweight

Let's be blunt: ==everyone is tired of being locked in.==

NVIDIA's "全家桶" approach — where GPU, NIC, switch, and software stack are all bundled — is the most profitable monopoly in the history of computing infrastructure. And every hyperscaler knows it.

But knowing it and being able to do something about it are different things.

The problem: if you want to decouple your supply chain — mix vendors, negotiate pricing, avoid single-source risk — you need *open standards* at every layer. And the memory layer has been the hardest to crack.

Think about it. You can buy a white-box switch. You can run open-source SONiC. You can pick AMD or Intel for CPUs. But memory? Until recently, DDR was soldered to the CPU's memory controller. No poolability. No disaggregation. No negotiation leverage against DRAM vendors.

> ==CXL is the first open standard that cracks the memory layer for disaggregation.==

Broadcom and Marvell aren't charity organizations. Their aggressive push into CXL controllers and switches isn't altruism — it's a *second ecosystem*. A market where hyperscalers can buy from multiple vendors, negotiate pricing, and avoid being locked into any single platform's memory architecture [3].

This is the same dynamic that drove the open networking movement (white-box switches, SONiC). CXL is open networking for memory. The demand is there because the pain of the alternative is there.

---

## 3. The "last mile" problem has one answer

Here's where it gets interesting. The open camp has a core technical problem:

> You've got a fabric bus (UALink, PCIe, Ethernet). You've got DDR memory sitting in a server. How do you efficiently pool and extend that memory across nodes *without* destroying latency?

This is the "last mile" of memory disaggregation. And the answer keeps coming back to CXL.

Marvell's acquisition of CXL controller talent and their next-gen CXL switch launch tell you everything [4]. They're not investing billions into CXL because it's a nice-to-have. They're investing because ==cross-node memory pooling over a bus fabric has an extremely high technical barrier==, and CXL is the protocol that clears it.

UALink's architecture makes this explicit: ==UALink and CXL are designed as complementary, not competitive.== [5]

| Protocol | Role | Layer |
|---|---|---|
| **UALink** | Accelerator-to-accelerator communication, GPU fabric | Compute fabric |
| **CXL** | Memory expansion, pooling, disaggregation | Memory fabric |

UALink handles the GPU-to-GPU traffic. CXL handles the memory-to-CPU/GPU traffic. You need both. UALink Consortium and CXL Consortium are not at war — they're *partitioning the problem*.

> ==UALink choosing to coexist with CXL is the market's answer to the memory last-mile question.== CXL is the hub. Everything else is a spoke.

Marvell's CXL switch delivers ==sub-microsecond== access to a near-local shared memory pool [4]. That's the kind of latency budget that makes memory pooling actually work for AI inference workloads. No other open protocol is hitting those numbers in 2026.

---

## 4. Ecosystem gravity: PCIe's native cousin

Now let's talk about why the *other* independent memory protocols are going to lose.

There have been several attempts to build alternative memory interconnect standards. Most of them are now dead or dying. The survivors all share one trait: they're fighting an uphill battle against CXL's ecosystem gravity.

Here's why. CXL is not a from-scratch protocol. ==CXL is built on PCIe physical layer.== [6] It inherits:

- PCIe's massive ecosystem (every CPU, every OS, every BIOS)
- PCIe's manufacturing scale (billions of units, cost amortized)
- PCIe's backward compatibility (same electrical interface, same form factors)
- PCIe's vendor ecosystem (Broadcom, Marvell, Microchip, Astera Labs all building on PCIe expertise)

Any alternative protocol has to rebuild all of that from zero. The technical uncertainty alone is a killer — hyperscalers won't bet their infrastructure on a protocol that might not have a second source.

> ==CXL's "native cousin" relationship with PCIe is a structural moat.== Not a technical differentiator — a *supply chain* moat.

Look at the deployment roadmap:

| Timeline | Milestone |
|---|---|
| 2025 | CXL 3.x in production (Intel Sapphire Rapids, AMD Genoa) |
| Late 2026 | CXL 4.0 products begin sampling [5] |
| Late 2026 | UALink switches reach data centers |
| 2027 | CXL 4.0 multi-rack systems, 100+ TB pools [2] |
| 2030 | $12.3B component market, $30B+ ecosystem [1] |

The train is moving. The fabs are booked. The controllers are shipping. Alternative protocols don't have time to catch up.

---

## Putting it all together

Four forces. One convergence.

| Force | Direction | CXL's Position |
|---|---|---|
| **Storage → Bus** | Data must be close to compute | CXL puts memory on the fabric |
| **Open vs. Lock-in** | Hyperscalers need supply chain independence | CXL is the open memory standard |
| **Memory Last Mile** | Fabric must pool DDR efficiently | CXL is the hub, UALink confirms |
| **Ecosystem Gravity** | PCIe infrastructure is irreplaceable | CXL rides PCIe's coattails |

Each force alone would make CXL interesting. Together, they make CXL *structurally inevitable* — at least for the open camp.

Will NVIDIA try to maintain NVLink's proprietary advantage? Of course. But even NVIDIA's customers (hyperscalers) are pushing for open alternatives. The market is bigger than one vendor's preferences.

---

## What could go wrong?

Intellectual honesty requires listing the bear cases:

1. **Deployment slower than expected.** CXL 2.0/3.0 adoption has been gradual. Enterprise inertia is real.
2. **NVIDIA's counter-move.** If NVLink/NVSwitch adds pooling features, the open-camp urgency decreases.
3. **UALink eats into CXL's scope.** If UALink expands to include memory semantics, CXL could be marginalized. (Current signals suggest complementarity, not competition.)
4. **DDR5/DDR6 bandwidth improvements reduce the pooling need.** If HBM keeps scaling, maybe local memory is "good enough."

These are real risks. But none of them change the structural dynamics. Even if CXL grows slower than the 32% CAGR forecast, the direction is clear: memory disaggregation is happening, and CXL is the open-standard vehicle.

---

## Bottom line

> ==I'm bullish on CXL not because of the TAM number, but because of the game theory. Every major player — hyperscalers, chip vendors, switch makers — has a structural incentive to make CXL succeed. That's rare in infrastructure.==

When Broadcom builds CXL controllers, Marvell builds CXL switches, Samsung builds CXL memory modules, UALink aligns with CXL, and hyperscaler after hyperscaler joins the CXL Consortium — that's not a technology bet. That's a *market coordination outcome*.

CXL won by choosing to be open at exactly the moment the market needed an open memory standard. The rest is execution.

---

## References

[1] [Strategic Market Research — Compute Express Link (CXL) Component Market](https://www.strategicmarketresearch.com/market-report/compute-express-link-component-market) — $12.3B by 2030, 32% CAGR; [Market Intelo — CXL Memory Expansion Market](https://marketintelo.com/report/cxl-memory-expansion-market) — $1.3B in 2025 → $11.8B by 2034, 28.7% CAGR; [Dataintelo — CXL Memory Module Market](https://dataintelo.com/report/cxl-memory-module-market) — $2.8B in 2025 → $28.6B by 2034, 29.4% CAGR

[2] [Introl — CXL 4.0 Infrastructure Planning Guide](https://introl.com/blog/cxl-4-0-infrastructure-planning-guide-memory-pooling-2025) — 100+ TB shared memory pools for AI inference

[3] [Marvell — Next-gen CXL Switch](https://www.marvell.com/company/newsroom/marvell-next-gen-cxl-switch-memory-pooling-breaks-ai-memory-wall.html) — Sub-microsecond access, near-local shared memory pool

[4] [Marvell CXL Controller/Switch investments](https://www.marvell.com/company/newsroom/marvell-next-gen-cxl-switch-memory-pooling-breaks-ai-memory-wall.html) — Aggressive positioning in CXL controller market

[5] [Introl — CXL 4.0 and the Interconnect Wars](https://introl.com/blog/cxl-4-specification-interconnect-wars-ai-memory-december-2025) — UALink and CXL as complementary; CXL 4.0 sampling late 2026

[6] [Compute Express Link Consortium](https://computeexpresslink.org/) — CXL specification built on PCIe physical layer

[7] [Samsung Semiconductor — Breaking AI Memory Limits with CXL Memory Pooling](https://semiconductor.samsung.com/news-events/tech-blog/breaking-ai-memory-limits-with-cxl-memory-pooling/) — CXL switch + memory devices → shared memory pool

[8] [Goldman Sachs — AI to drive 165% increase in data center power demand by 2030](https://www.goldmansachs.com/insights/articles/ai-to-drive-165-increase-in-data-center-power-demand-by-2030/) — Structural AI infrastructure demand driver

---

*This is a draft post. Views are my own analysis based on publicly available market data. Not investment advice.*
