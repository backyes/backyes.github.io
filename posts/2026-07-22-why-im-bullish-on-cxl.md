---
title: "[Draft] Why I'm Bullish on CXL — A Market Game Theory Perspective"
date: 2026-07-22
tags: ["CXL", "UALink", "Memory-Pooling", "AI-Inference", "Marvell", "Broadcom", "Interconnect", "Open-Standard"]
excerpt: "CXL is not just another interconnect protocol. From a market game theory lens, I believe CXL's long-term positioning is structurally underappreciated. Four forces are converging to make CXL the inevitable backbone of open-memory infrastructure."
---

# [Draft] Why I'm Bullish on CXL — A Market Game Theory Perspective

## The setup

The CXL component market was worth roughly ==$1.3–2.1B== in 2025 [1a][1b]. Forecasts put it at ==$12.3B== by 2030 — a ~32% CAGR [1c]. That's strong growth, but the number itself isn't why I'm bullish. Plenty of technologies have strong TAM projections and still fail.[^tam]

[^tam]: **TAM (Total Addressable Market)** = 可寻址市场总规模，指某一产品或服务在理论上能达到的最大市场收入。TAM 高不代表一定能赢——技术路线、生态、执行时机同样关键。

What makes CXL different is its *position in the market game*. Four structural forces are converging, and CXL sits at the intersection of all of them.

---

## 1. Storage is marching into the "bus domain"

Here's the fundamental shift: AI training and inference are forcing a structural reorganization of the entire memory-storage stack.

In the old world, storage sits *below* the compute — you pull data up through PCIe, process it, write it back. The hierarchy was clean: DRAM → SSD → HDD → Tape.

That hierarchy is collapsing.

The physics are simple: the working dataset doesn't fit in one box. It barely fits in one rack. Even NVIDIA — the most HBM-rich vendor — is hitting the wall. Their LPX chip pushes SRAM usage to the extreme, because for MoE models the expert routing creates *random, fine-grained memory access patterns* that HBM bandwidth alone can't feed. The only way to keep the compute fed is to put more memory closer — closer than HBM, closer than a remote node. Data must be *infinitely close* to compute — not just physically, but topologically. Every hop adds latency. Every copy wastes power.

> ==Storage is no longer a tier below compute. It's becoming a peer on the bus.==

This is what I mean by "泛存储全面总线化" — all forms of storage are being pulled onto the interconnect fabric as first-class citizens. NVMe over Fabrics was step one. CXL is step two: memory itself becomes a fabric-attached resource.

CXL 4.0 (released Nov 2025, products sampling late 2026) enables ==100+ TB== shared memory pools with cache-coherent multi-rack access [2]. Its bundled ports aggregate up to ==1.5 TB/s== bandwidth — roughly 30% of HBM3e's bandwidth, sufficient for memory expansion where capacity matters more than peak bandwidth [2]. That's not a niche accelerator feature — that's a fundamental re-architecture of the server.

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

Marvell's acquisition of XConn (a CXL switch startup) and their Structera S 30260 launch tell you everything [4a][4b]. The Structera S delivers ==sub-460ns round-trip== memory access — that's sub-microsecond, near-local latency — across a shared pool of up to ==48 TB== of memory [4c]. They're not investing billions into CXL because it's a nice-to-have. They're investing because ==cross-node memory pooling over a bus fabric has an extremely high technical barrier==, and CXL is the protocol that clears it.

UALink's architecture makes this explicit: ==UALink and CXL are designed as complementary, not competitive.== [5a]

| Protocol | Role | Layer |
|---|---|---|
| **UALink** | Accelerator-to-accelerator communication, GPU fabric | Compute fabric |
| **CXL** | Memory expansion, pooling, disaggregation | Memory fabric |

UALink handles the GPU-to-GPU traffic. CXL handles the memory-to-CPU/GPU traffic. You need both. UALink Consortium and CXL Consortium are not at war — they're *partitioning the problem*.

Industry analysis confirms this: "CXL focuses on rack-level and multi-rack memory expansion and pooling, while UALink optimizes high-speed, accelerator-to-accelerator scaling for AI clusters" [5b]. Emerging architectures even propose "CXL over XLink/UALink" to pair high-performance local memory with shared composable pools for LLM inference [5c].

> ==UALink choosing to coexist with CXL is the market's answer to the memory last-mile question.== CXL is the hub. Everything else is a spoke.

No other open protocol is hitting sub-microsecond latency with real silicon in 2026. That's the kind of latency budget that makes memory pooling actually work for AI inference workloads — where a 70B model with 128K context and batch size 32 can require ==150+ GB== just for KV cache alone [2].

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
| Late 2024-2025 | CXL 2.0 in production — Samsung CMM-D (256 GB) [7a], SK Hynix CMM (128 GB) [7b], Micron CZ120 (256 GB) [7c] shipping |
| 2025 | CXL 3.x fabric switches — XConn Apollo, Panmnesia CXL 3.2 (first PBR implementation, up to 4,096 nodes) [7d] |
| Late 2026 | CXL 4.0 products begin sampling [5b]; UALink switches reach data centers |
| 2027 | CXL 4.0 multi-rack systems, 100+ TB pools — target production deployment [2] |
| 2030 | $12.3B component market, $30B+ ecosystem [1c] |

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

## NVIDIA's counter-move: NVLink Fusion

NVIDIA is not standing still. At Computex 2025 and GTC 2026, NVIDIA announced **NVLink Fusion** — opening the proprietary NVLink fabric to third-party custom silicon (MediaTek, Marvell, Fujitsu, Qualcomm) via the **NVLink-C2C** chip-to-chip interconnect [9a][9b]. NVLink-C2C delivers up to ==1.8 TB/s== per GPU with cache-coherent shared memory access, enabling pooled memory subsystems across the NVLink fabric [9c].

In essence, NVIDIA is building a *proprietary alternative to CXL for memory disaggregation* — using NVLink as the pooling bus instead of PCIe/CXL.

![NVIDIA NVLink Fusion Architecture — NVLink-C2C enables third-party chip integration and memory pooling over NVLink fabric](assets/nvlink-fusion-cxl-pooling.png)

*NVIDIA NVLink Fusion architecture: third-party chips (custom CPUs, accelerators) connect via NVLink-C2C and UCIe bridge into the NVLink fabric, enabling cache-coherent memory pooling as a proprietary alternative to CXL [9a][9b][9e]*

Academic research confirms NVLink's viability for memory disaggregation — a 2025 paper from Hasso Plattner Institute benchmarks Grace CPU accessing GPU memory via NVLink for disaggregated memory workloads [9d].

This is a real competitive response. But note the key limitation: **NVLink Fusion still requires NVIDIA's blessing**. Every third-party chip needs an NVIDIA bridge chip and licensing. For hyperscalers who want *true* supply chain independence — multiple sources, no single point of control — CXL remains the only open path. NVIDIA's move validates the market need for memory pooling but doesn't eliminate the demand for an open-standard alternative.

---

## Bottom line

> ==I'm bullish on CXL not because of the TAM number, but because of the game theory. Every major player — hyperscalers, chip vendors, switch makers — has a structural incentive to make CXL succeed. That's rare in infrastructure.==

When Broadcom builds CXL controllers, Marvell builds CXL switches, Samsung builds CXL memory modules, UALink aligns with CXL, and hyperscaler after hyperscaler joins the CXL Consortium — that's not a technology bet. That's a *market coordination outcome*.

CXL won by choosing to be open at exactly the moment the market needed an open memory standard. The rest is execution.

---

## References

### Market Size & Forecasts

[1a] [Market Intelo — CXL Memory Expansion Market](https://marketintelo.com/report/cxl-memory-expansion-market) — CXL memory expansion market valued at ==$1.3B in 2025==, projected to reach $11.8B by 2034 (28.7% CAGR)

[1b] [Dataintelo — CXL Memory Module Market](https://dataintelo.com/report/cxl-memory-module-market) — CXL memory module market valued at ==$2.1–2.8B in 2025==, projected to reach $28.6B by 2034 (29.4% CAGR)

[1c] [Strategic Market Research — CXL Component Market](https://www.strategicmarketresearch.com/market-report/compute-express-link-component-market) — CXL component market: $1.9B (2024) → ==$12.3B by 2030==, ==32% CAGR==; wider ecosystem $30B+ by 2030

### CXL 4.0 Technical Capabilities

[2] [Introl — CXL 4.0 Infrastructure Planning Guide](https://introl.com/blog/cxl-4-0-infrastructure-planning-guide-memory-pooling-2025) — CXL 4.0 enables ==100+ TB== shared memory pools; bundled ports deliver ==1.5 TB/s==; multi-rack memory pooling target production late 2026-2027; KV cache offloading use case (70B model, 128K context, batch 32 → 150+ GB); memory utilization 50-60% → 85%+; latency 200-500 ns (DRAM-like)

### Marvell CXL Strategy

[3] [Futurum Group — Marvell's XConn Acquisition](https://futurumgroup.com/insights/marvells-xconn-buy-yields-a-two-pronged-open-fabric-play-against-nvlink/) — Marvell's acquisition of XConn creates two-pronged open fabric play against NVLink; CXL 3.0 memory pooling delivers sub-microsecond latency at rack scale

[4a] [Marvell Official — Next-gen CXL Switch Launch](https://www.marvell.com/company/newsroom/marvell-next-gen-cxl-switch-memory-pooling-breaks-ai-memory-wall.html) — Structera S CXL switch: near-local shared memory pool with ==sub-microsecond access==, eliminates multi-hop data movement

[4b] [Yahoo Finance — Marvell Structera S 30260](https://finance.yahoo.com/markets/stocks/articles/marvell-mrvl-launches-next-gen-155330376.html) — Structera S 30260: ==260 lanes==, ==4 TB/s== cumulative bandwidth, up to ==48 TB== shared memory, sub-microsecond latency

[4c] [HyperFRAME Research — Marvell Structera Analysis](https://hyperframeresearch.com/2026/06/28/marvell-structera-driving-cxl-hardware-optimization-and-ai-memory-efficiency/) — Sub-460ns round-trip latency; integrated Structera S switches + Structera X expanders + Structera A accelerators

### UALink & CXL Complementary

[5a] [Introl — CXL 4.0 and the Interconnect Wars](https://introl.com/blog/cxl-4-specification-interconnect-wars-ai-memory-december-2025) — "CXL focuses on rack-level and multi-rack memory expansion and pooling, while UALink optimizes high-speed, accelerator-to-accelerator scaling"

[5b] [Google Cloud AI Overview (search result)](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26) — UALink and CXL as complementary open standards; CXL 4.0 sampling late 2026

[5c] [Blocks & Files — Panmnesia Unified Memory](https://www.blocksandfiles.com/ai-ml/2025/07/18/panmnesia-pushes-unified-memory-and-interconnect-design-for-ai-superclusters/1602353) — "CXL over XLink/UALink" architectures for LLM inference and RAG

### CXL Built on PCIe

[6] [Compute Express Link Consortium](https://computeexpresslink.org/) — CXL specification built on PCIe physical layer; CXL 4.0 based on PCIe 7.0 (128 GT/s)

### Ecosystem — Memory Vendors

[7a] [Samsung Semiconductor — CXL Memory Pooling](https://semiconductor.samsung.com/news-events/tech-blog/breaking-ai-memory-limits-with-cxl-memory-pooling/) — Samsung CMM-D: ==256 GB== CXL 2.0, mass production 2025

[7b] [Introl — CXL 4.0 Planning Guide (vendor table)](https://introl.com/blog/cxl-4-0-infrastructure-planning-guide-memory-pooling-2025) — SK Hynix CMM-DDR5: 128 GB; SK Hynix CMS: 512 GB (compute-enabled); Micron CZ120: 256 GB

[7c] [Astera Labs — Leo CXL Controller](https://www.asteralabs.com/) — CXL 2.0 smart memory controller shipping

[7d] [Panmnesia — CXL 3.2 Fabric Switch](https://panmnesia.com/news/en/) — First PBR implementation, up to 4,096 nodes, sampling Nov 2025

### NVIDIA's Counter-Move

[9a] [NVIDIA Newsroom — NVLink Fusion](https://nvidianews.nvidia.com/news/nvidia-nvlink-fusion-semi-custom-ai-infrastructure-partner-ecosystem) — NVLink Fusion announced May 2025, opens NVLink fabric to custom silicon via NVLink-C2C; partners include MediaTek, Marvell, Fujitsu, Qualcomm

[9b] [Counterpoint Research — NVLink Fusion: NVIDIA's Response to UALink?](https://counterpointresearch.com/en/insights/post-insight-research-notes-blogs-nvlink-fusion-nvidias-response-to-ualink) — Analysis of NVLink Fusion as competitive response to open standards

[9c] [NVIDIA Developer — NVLink: The Scale-Up Network for AI Factories](https://developer.nvidia.com/blog/nvidia-nvlink-the-scale-up-network-for-ai-factories/) — NVLink-C2C cache-coherent interconnect, 1.8 TB/s per GPU in GB200/GB300 NVL72

[9d] [ACM Digital Library — Towards Memory Disaggregation via NVLink C2C](https://dl.acm.org/doi/10.1145/3723851.3723853) — Academic benchmarking of NVLink for memory disaggregation (Grace CPU accessing GPU memory via NVLink)

[9e] [SDxCentral — NVIDIA opens NVLink fabric to hyperscaler custom silicon](https://www.sdxcentral.com/news/nvidia-opens-nvlink-fabric-to-hyperscaler-custom-silicon/) — UCIe interface for third-party chip integration

---

*This is a draft post. Views are my own analysis based on publicly available market data and technical documentation. Not investment advice.*
