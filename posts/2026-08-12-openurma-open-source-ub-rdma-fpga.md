---
title: "OpenURMA: Open-Source FPGA Implementation of a UB-class RDMA Transport"
date: 2026-08-12
tags: ["RDMA", "FPGA", "OpenURMA", "UB", "SmartNIC", "Interconnect", "Open-Source", "Load-Store"]
excerpt: "OpenURMA is an open-source FPGA implementation of the UB (Ultra-Bus) connectionless RDMA transport, achieving ≈500ns remote load/store latency on Alveo U50 (vs 2236ns on RoCE, 4.47× faster), while fully implementing the UB-Base-Specification 2.0.1 transaction and transport layers."
---

# OpenURMA: Open-Source FPGA Implementation of a UB-class RDMA Transport

> **Links:**
> - Paper: [arXiv:2605.28717](https://arxiv.org/abs/2605.28717)
> - Code: [github.com/bojieli/OpenURMA](https://github.com/bojieli/OpenURMA)
> - Architecture Guide: [docs/architecture.md](https://github.com/bojieli/OpenURMA/blob/main/docs/architecture.md)
> - Evaluation: [EVAL.md](https://github.com/bojieli/OpenURMA/blob/main/EVAL.md)

## What Is This

[OpenURMA](https://github.com/bojieli/OpenURMA) is an **open-source FPGA implementation** of the [UB (Ultra-Bus)](https://www.ub.org/) connectionless RDMA transport protocol, built as `.clnp` elements on top of [OpenClickNP](https://github.com/OpenClickNP). It fully implements what *UB-Base-Specification 2.0.1* defines:

- **Transaction Layer**: BTAH/ATAH headers, 18 transaction opcodes, all four service modes (ROI/ROT/ROL/UNO), all three execution-order tags (NO/RO/SO), application Fence, and both completion-order modes.
- **Transport Layer**: RTP (PSN/GoBackN retransmission), UTP for UNO mode, simplified CETPH echo.

On top of that, `libopenurma` exposes the URMA verb surface from *UB-Software-Reference-Design-for-OS-2.0* §5.3.

## Three Architectural Pillars

The OpenURMA paper ([arXiv:2605.28717](https://arxiv.org/abs/2605.28717)) summarizes the design philosophy of the UB protocol stack in a single figure — three mutually reinforcing pillars:

1. **Transport / Transaction split.** State scales as O(local Jetties) + O(remote endpoints), not their product. This is what lets the controller sit on the on-chip bus (rather than behind PCIe).

2. **Native load/store latency.** Because the NIC's working set fits in on-chip SRAM, the controller lives on the on-chip bus next to the CPU, so a CPU load/store reaches remote memory directly — collapsing the four PCIe traversals of an RDMA READ into a **single on-chip-bus crossing**. This is the headline result: ==a 64-byte remote fetch completes in ≈500ns== end-to-end versus ==2236ns== on the matched RoCEv2 baseline (**4.47× faster**).

3. **Graded ordering.** OpenURMA implements the full §7.3 surface — four service modes × three execution tags × Fence × two completion modes — so applications opt into precisely the consistency they need (it rides on the per-application counters that pillar 1 provisions, so it costs nothing on operations that don't request gating).

## Not Just RTL

A standout feature of OpenURMA is the **full software-stack validation**:

- The same `.clnp` design compiles both to Alveo U50 hardware RTL and to a cycle-accurate SystemC NIC.
- The **unmodified official openEuler UMDK stack** (`liburma → uburma.ko → ubcore.ko → openurma_ubcore.ko`) drives it end-to-end inside a full-system **gem5** Linux guest.
- Real applications run on it: the official `urma_perftest`, the URPC `umq` RPC framework, a KV store (up to 60KB values), distributed atomic counters, many-client concurrency, and §7.3 ordering workloads.
- All three transport modes (RM / RC / UM) are verified.

## Reproduction Paths

```bash
git clone https://github.com/bojieli/OpenURMA
./reproduce.sh doctor   # check toolchains
./reproduce.sh smoke    # build + 17 tests + verify headline numbers (~2 min)
./reproduce.sh paper    # full dataset + all figures + rebuild PDF (~15 min)
```

## Why It Matters

In the context of AI supernode interconnects, OpenURMA provides an interesting point of comparison:

- **RoCE/RDMA** takes the "message passing" path (Send/Write/Read), requiring multiple PCIe traversals.
- **UB load/store** takes the "memory semantics" path (load/store directly reaches remote memory), with an order-of-magnitude lower latency.
- OpenURMA proves this difference is **not theoretical** — it runs on real FPGA hardware and is validated with a production software stack.

For those studying the first principles of URMA protocol implementation, its key concepts, and full-stack vertical integration, this project offers a valuable open-source reference.
