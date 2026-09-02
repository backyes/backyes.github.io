---
title: "[Under Review] Revisiting BaM's '× Cost Advantage' — The Denominator That Got Skipped"
date: 2026-09-02
tags: ["BaM", "GPU-Initiated-Storage", "GDS", "SSD", "IOPS", "Cost-Analysis", "Memory-Wall", "PCIe", "Optane", "System-Architecture"]
excerpt: "BaM claims 4.4×~21.8× cost advantage over DRAM-only systems. But this conclusion rests on $/GB — while the real bottleneck for these workloads is IOPS. When IOPS demand scales beyond the tested 10-SSD configuration, the cost curve may bend in ways the paper never measured."
---

# [Under Review] Revisiting BaM's "× Cost Advantage" — The Denominator That Got Skipped

## Thesis

**BaM's cost advantage is real within its tested envelope — but the paper proves a $/GB claim and sells it as a $/IOPS conclusion. The gap between these two metrics only widens as IOPS demand scales, and that scaling behavior is precisely what the paper leaves unexamined.**

> BaM (Big accelerator Memory) lets GPU threads bypass the CPU entirely to initiate storage I/O against SSDs — a clean architectural idea with impressive engineering. Its cost claim — ==4.4×~21.8× cost advantage== over pure DRAM — has been widely cited. This post does not argue that claim is *wrong*. It argues the claim is *narrower than it sounds*, and the jump from "measured at 10 SSDs" to "system-level cost advantage" skips a question that matters most for the people who would push BaM-style architectures to their limit: ==what happens to that ratio when you need 50 or 100 SSDs' worth of IOPS?==

---

## 1. What BaM Actually Built

BaM's core idea is straightforward: GPU threads issue storage requests directly to SSDs via PCIe, without CPU mediation in the hot path. The software stack handles:

| Component | Mechanism | Purpose |
|-----------|-----------|---------|
| **Ticket allocation** | GPU-managed circular buffer with producer/consumer indices | Ordering and concurrency control without CPU involvement |
| **Mark bitmap** | Per-block completion tracking | Fine-grained I/O completion detection |
| **Doorbell coalescing** | Batched MMIO writes to SSD submission queues | Amortize PCIe transaction overhead |
| **Request batching** | GPU threads aggregate small requests into larger SSD commands | Approach SSD peak efficiency (larger queue depths) |

The result: BaM achieves ==~90% PCIe bandwidth utilization== on its tested SSD configurations. This is genuine systems engineering — getting within 10% of theoretical PCIe throughput from a GPU-initiated storage stack is not trivial.

### 1.1 Tested Configuration

| Parameter | BaM's Setup |
|-----------|-------------|
| **GPU** | NVIDIA A100 (40/80 GB HBM) |
| **Optane SSDs** | Up to ×10 (Intel P5800X, PCIe 4.0) |
| **Consumer NVMe** | Up to ×4 (Samsung 980 Pro or similar) |
| **Workloads** | Graph analytics (BFS, SSP), recommendation embedding lookups, data analytics |
| **Dataset scale** | Exceeds DRAM capacity — the entire point of using SSDs |

---

## 2. The Two Pillars of BaM's Cost Argument

BaM's "× cost advantage" rests on two sub-claims, each individually defensible:

### Pillar ①: Capacity Cost ($/GB)

This is the easy one. The paper presents a $/GB comparison:

| Media | Approximate $/GB (2021–2022 pricing) | Ratio vs DRAM |
|-------|--------------------------------------|---------------|
| **DRAM (HBM/DDR)** | ~$15–30/GB | 1× (baseline) |
| **Intel Optane (P5800X)** | ~$3–5/GB | ~5–10× cheaper |
| **Consumer NVMe (TLC)** | ~$0.10–0.20/GB | ~100–200× cheaper |

BaM's reported 4.4×~21.8× advantage reflects a *system-level* comparison (not raw media cost) — accounting for the fact that BaM needs some DRAM as a cache/buffer layer, and that the comparison is against a DRAM-only system provisioned for the same *capacity*. The 4.4× end corresponds to Optane (expensive per GB but low-latency), the 21.8× end to consumer NAND (cheap per GB, higher latency).

**This pillar holds.** For workloads whose datasets exceed economically feasible DRAM capacity — BaM targets graph analytics on hundred-billion-edge graphs, recommendation tables with terabyte-scale embeddings, scan-heavy data analytics — SSDs are unambiguously cheaper per GB. No argument here.

### Pillar ②: Performance Efficiency (IOPS Delivered vs. IOPS Possible)

This is where BaM's engineering shines. The paper demonstrates that its software stack can drive SSDs toward their hardware limits:

| Metric | BaM's Achievement |
|--------|-------------------|
| **PCIe bandwidth utilization** | ~90% of theoretical |
| **Per-SSD IOPS** | Near vendor-spec peak (with sufficient queue depth) |
| **CPU offload** | CPU not in hot path — free for compute |

The key insight: traditional CPU-initiated storage stacks waste SSD capability through software overhead (context switches, kernel transitions, descriptor setup). BaM's GPU-initiated path eliminates these bottlenecks, letting the SSDs actually *do what they're capable of*.

**This pillar also holds** — within the tested configuration. BaM genuinely extracts near-peak IOPS from each SSD.

---

## 3. The Logical Gap: Where ① × ② ≠ ③

The paper's implicit argument structure is:

```
① Capacity cost advantage (SSD cheaper per GB)     ✓ Proven at tested scale
② Performance loss可控 (software stack near-optimal) ✓ Proven at tested scale
─────────────────────────────────────────────────
③ ∴ System-level cost advantage = several ×         ?
```

The leap from ① and ② to ③ assumes that the *cost per IOPS* scales linearly with the *cost per GB* — that the ratio you measure at 10 SSDs holds at 50 or 100 SSDs. This assumption is never tested.

### 3.1 The Core Problem: DRAM's IOPS Is "Free"

Here is the asymmetry the paper does not address:

```
DRAM:  Pay for capacity → IOPS comes along at zero marginal cost
       (DRAM bandwidth is ~500-1000 GB/s per GPU — any byte is equally accessible)

SSD:   Pay for capacity → IOPS is FIXED per drive
       To get 2× IOPS → need 2× SSDs → 2× $ + 2× PCIe lanes + 2× physical slots
```

DRAM's IOPS is a *byproduct of its capacity* — the same HBM that stores your data also delivers TB/s of bandwidth. You do not buy DRAM for capacity and then separately provision IOPS. But with SSDs, capacity and IOPS are *independent cost dimensions*: a 4 TB consumer SSD might give you 100K IOPS — if you need 1M IOPS, you need 10 of them, and you now have 40 TB you may not need.

### 3.2 The Two Cost Curves

```
Cost
 │
 │                          ╱ SSD ($/IOPS — linear in drive count)
 │                        ╱
 │                      ╱    ····· DRAM ($/IOPS — near-zero marginal)
 │                    ╱   ···
 │                  ╱  ···
 │                ╱ ···
 │              ╱···
 │··········· ╱
 │          ╱
 │        ╱
 │──────╱──────────────────────────────── IOPS Demand
       Low          Medium         High
       (BaM tested)  (untested)     (untested)
```

At the *low-to-medium* IOPS range BaM tested (10 Optane or 4 consumer SSDs), the SSD cost curve sits well below DRAM. The 4.4×~21.8× advantage is real *in this region*.

But the curves have different shapes:
- **DRAM $/IOPS**: essentially flat — you already paid for bandwidth when you bought capacity
- **SSD $/IOPS**: linear — each additional 100K IOPS costs one more SSD + PCIe lane + slot + power

At some IOPS demand — the paper never identifies where — the SSD line crosses above where DRAM would be. Not because SSDs got more expensive, but because you're paying for IOPS you could have gotten "for free" with a DRAM-only system of equivalent bandwidth.

---

## 4. The Physical Constraints That Don't Appear at ×10

At 10 SSDs, several costs are invisible or negligible. They become dominant at scale:

### 4.1 PCIe Lane Budget

| Resource | Available (typical H100/A100 node) | Consumed by 10× SSDs | Consumed by 50× SSDs |
|----------|------------------------------------|-----------------------|----------------------|
| **PCIe lanes (CPU)** | 128 (×16 for GPU) + 64 (chipset) | ~40 (×4 per SSD) | ~200 — **exceeds platform** |
| **PCIe switch ports** | Limited by topology | Fits | Needs external switches |
| **GPU PCIe root complex** | 16 lanes (typically all to GPU NIC) | N/A (SSD via CPU) | N/A |

At 10 SSDs, you can route them through the CPU's PCIe root complex and a couple of switches. At 50–100 SSDs, you need PCIe switches ($$), riser cards, external chassis — each adding cost, latency, and failure points that the $/GB table does not capture.

### 4.2 Chassis and Power

| Factor | 10 SSDs | 50 SSDs | 100 SSDs |
|--------|---------|---------|----------|
| **Drive bays** | Fits in 2U server | Needs 4U+ or JBOD | Needs dedicated storage enclosure |
| **Power (~15W/active SSD)** | ~150W | ~750W | ~1500W |
| **Cooling** | Standard server airflow | Enhanced cooling needed | Storage-grade thermal design |
| **Cost impact** | Negligible | Significant (JBOD $, power $, cooling $) | Dominates TCO |

### 4.3 The Optane Escape Hatch Shrinks

When IOPS demand exceeds what consumer NAND can deliver per dollar, the natural move is to Optane (or Z-NAND) — lower latency, higher IOPS per drive, fewer drives needed. But:

| Media | $/GB Ratio vs DRAM | $/IOPS Ratio vs DRAM |
|-------|--------------------|-----------------------|
| **Consumer NAND** | ~100× cheaper | ~20× cheaper (BaM's 21.8×) |
| **Optane** | ~5–10× cheaper | ~4× cheaper (BaM's 4.4×) |

The cost advantage *compresses by 5×* when you move from the capacity-optimized to the IOPS-optimized media. And that compression is a signal: ==the $/GB and $/IOPS curves diverge, and the paper's headline number tracks the more favorable of the two.==

---

## 5. What the Paper Would Need to Show

This critique does not demand that BaM be wrong — it demands that the *scope* of the claim match the *scope* of the evidence. To support a general "× cost advantage" claim, the paper would need:

| Missing Analysis | Why It Matters |
|------------------|----------------|
| **$/IOPS curve across drive counts** | Show the cost ratio at 10, 20, 50, 100 SSDs — does it stay flat or bend? |
| **PCIe topology cost model** | Include switch/riser/enclosure costs at scale |
| **Break-even IOPS analysis** | At what IOPS demand does DRAM become cheaper? (Answer: probably very high, but "very high" ≠ "infinity") |
| **Workload IOPS requirement characterization** | What IOPS does each target workload actually need? Does it stay within the flat region? |
| **Sensitivity to SSD generation** | PCIe 5.0 SSDs double per-drive IOPS — does this help or hurt the ratio? |

Without these, the 4.4×~21.8× number is best read as: *"at the specific IOPS demand level of our tested workloads, with our specific SSD configuration, under our specific assumptions."* That is a valid and useful result — it is just not the sweeping claim the abstract implies.

---

## 6. When Does This Gap Actually Matter?

For fairness, let me note when the gap *doesn't* matter:

| Scenario | BaM's Claim Holds? | Why |
|----------|-------------------|-----|
| **Dataset >> DRAM, IOPS demand moderate** | ✓ Yes | This is BaM's sweet spot — exactly what was tested |
| **Graph analytics on 100B-edge graphs** | ✓ Likely | BFS/SSSP are bandwidth-bound, not IOPS-bound at extreme scale |
| **Recommendation embedding tables** | ✓ Likely | Large capacity, access patterns amenable to caching |
| **High-frequency trading / real-time inference** | ✗ No | IOPS-bound, latency-sensitive — Optane required, advantage shrinks to 4.4× |
| **50+ GPU cluster, each needing 1M+ IOPS** | ✗ Unclear | Aggregate IOPS demand may push beyond tested regime |

The gap matters most for anyone considering BaM-style architectures for *IOPS-bound* workloads at *large scale* — precisely the scenario where you'd want to deploy such a system most aggressively.

---

## 7. The Deeper Lesson: $/GB Is the Wrong Metric for Storage-Class Memory

BaM is not alone in this analytical pattern. The entire "storage-class memory" discourse — CXL memory, SSD-as-memory, computational storage — tends to anchor on $/GB because it is the easy number to cite. But for GPU-accelerated workloads:

```
The right metric chain:

  Workload characteristic
        │
        ▼
  What limits throughput? ──→ Capacity? → $/GB matters
        │                      IOPS?     → $/IOPS matters
        │                      Latency?  → $/access matters
        ▼
  Provision the bottleneck resource
        │
        ▼
  Compare total cost at equivalent bottleneck performance
```

BaM's workloads are *capacity-bound* (datasets exceed DRAM) — so $/GB is indeed the right primary metric for them. The paper's conclusions are valid for its tested workloads. The problem is one of *rhetoric*, not *results*: the abstract's "× cost advantage" reads as a universal claim, when the evidence supports a narrower one.

---

## 8. Conclusion

BaM is a well-engineered system that demonstrates GPU-initiated storage can approach hardware bandwidth limits. Its 4.4×~21.8× cost advantage is real *within the tested regime* — moderate IOPS demand, 10 or fewer SSDs, capacity-bound workloads.

The critique is not that BaM is wrong. It is that ==the paper proves a point about capacity cost and narrates it as a point about system cost==, skipping the question of how that number behaves when IOPS demand grows beyond the tested envelope. The $/GB curve and the $/IOPS curve are different shapes — and the gap between them is where the real cost analysis lives.

For practitioners: if your workload looks like BaM's targets (large graph, large embedding table, scan-heavy analytics), the cost advantage is probably real. If you are pushing toward higher IOPS or larger scale, ==recompute the ratio at your actual IOPS target== — the answer may be different from 21.8×.

---

## References

- **BaM Paper**: "GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture" — Qin et al., MICRO 2022
- **Related**: NVIDIA GPUDirect Storage (GDS) — the production-grade realization of GPU-initiated storage I/O
- **Related**: Intel Optane P5800X specifications — the low-latency SSD used in BaM's evaluation
- **Context**: CXL memory expansion — an alternative approach to the capacity-cost problem that avoids the IOPS-scaling issue entirely
