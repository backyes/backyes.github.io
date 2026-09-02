---
title: "[Under Review] Revisiting BaM's '× Cost Advantage' — The Denominator That Got Skipped"
date: 2026-09-02
tags: ["BaM", "GPU-Initiated-Storage", "GDS", "SSD", "IOPS", "Cost-Analysis", "Memory-Wall", "PCIe", "Optane", "System-Architecture"]
excerpt: "BaM claims 4.4×~21.8× cost advantage over DRAM-only systems. But this conclusion rests on $/GB — while the real bottleneck for these workloads is IOPS. When IOPS demand scales beyond the tested 10-SSD configuration, the cost curve may bend in ways the paper never measured."
---

# [Under Review] Revisiting BaM's "× Cost Advantage" — The Denominator That Got Skipped

## Thesis

**BaM proves a $/GB claim and sells it as a system-level cost conclusion. The gap between these two metrics only widens as IOPS demand scales — and that scaling behavior is what the paper leaves unexamined.**

> BaM (Big accelerator Memory) lets GPU threads bypass the CPU to initiate storage I/O directly against SSDs. Its cost claim — ==4.4×~21.8× cost advantage== over pure DRAM — has been widely cited. This post does not argue the claim is *wrong*. It argues the claim is ==narrower than it sounds==: the jump from "measured at 10 SSDs" to "system-level cost advantage" skips a question that matters when you push BaM-style architectures to their limit.

---

## 1. What BaM Built

BaM's idea: GPU threads issue storage requests directly to SSDs via PCIe, no CPU in the hot path. Through careful queue design (ticket allocation, mark bitmap, doorbell coalescing), BaM achieves ==~90% PCIe bandwidth utilization== — genuine systems engineering.

**Tested setup**: NVIDIA A100, up to ×10 Intel Optane SSDs or ×4 consumer NVMe, running graph analytics (BFS, SSP), recommendation embedding lookups, and data analytics.

---

## 2. The Cost Argument — Two Pillars

BaM's "× cost advantage" combines two claims:

**① Capacity cost ($/GB)**: SSD is unambiguously cheaper per GB than DRAM. Consumer NAND is ~100× cheaper; Optane ~5–10× cheaper. For datasets exceeding DRAM capacity — BaM's target — this is true.

**② Performance efficiency**: BaM's software stack drives SSDs toward their hardware limits. ~90% PCIe utilization, near peak-IOPS per drive. Also true — at the tested scale.

---

## 3. The Gap: ① + ② ≠ System-Level Cost Advantage

The paper's logic: "SSD cheaper per GB" + "performance loss可控" → "system costs several × less."

The missing step: ==cost per GB and cost per IOPS are two different curves.==

**DRAM's IOPS is "free"** — buy capacity, get TB/s bandwidth at zero marginal cost. The same HBM that stores your data also serves it.

**SSD's IOPS is fixed per drive** — need 2× IOPS? Buy 2× SSDs, 2× PCIe lanes, 2× physical slots. Capacity and IOPS are independent cost dimensions.

At 10 SSDs (BaM's tested scale), the SSD cost curve sits well below DRAM. But the curves have different shapes: DRAM $/IOPS is flat, SSD $/IOPS is linear in drive count. ==The paper measures at one point on the curve and extrapolates to a universal claim.==

---

## 4. What Happens at Scale

At 10 SSDs, several costs are invisible. At 50–100 SSDs, they dominate:

- **PCIe lanes**: 10 SSDs need ~40 lanes (fits). 50 SSDs need ~200 (exceeds platform — needs external switches, $$).
- **Chassis and power**: 10 SSDs fit in 2U. 100 SSDs need dedicated storage enclosures, ~1500W, storage-grade cooling.
- **The Optane escape hatch**: When consumer NAND can't deliver enough IOPS, move to Optane. But Optane's cost advantage over DRAM shrinks from ~20× (NAND) to ~4.4×. ==The more IOPS you need, the less "cheap" SSD becomes.==

None of this appears in the paper's $/GB table.

---

## 5. The Bottom Line

BaM is well-engineered. Its 4.4×~21.8× advantage is real *within the tested regime* — moderate IOPS demand, ≤10 SSDs, capacity-bound workloads.

But ==the paper proves a point about capacity cost and narrates it as a point about system cost.== The $/GB curve and the $/IOPS curve diverge as IOPS demand grows. The paper never measures where.

**For practitioners**: if your workload matches BaM's targets (large graph, large embedding table, scan-heavy analytics), the advantage is real. If you need higher IOPS or larger scale, recompute the ratio at your actual target — the answer may be different from 21.8×.

---

## References

- **BaM Paper**: "GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture" — Qin et al., MICRO 2022
- **Related**: NVIDIA GPUDirect Storage (GDS) — production-grade GPU-initiated storage I/O
- **Related**: Intel Optane P5800X — the low-latency SSD used in BaM's evaluation
