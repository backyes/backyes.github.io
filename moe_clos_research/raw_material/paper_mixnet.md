# MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed MoE Training

> SIGCOMM 2025
> Authors: Xudong Liao, Yijun Sun, Han Tian, Xinchen Wan, et al. (HKUST, et al.)
> URL: https://arxiv.org/abs/2501.03905

## Core Thesis
MoE's dynamic communication has strong locality → regional OCS reconfiguration suffices → no need for full CLOS.

## Key Results
- Cost efficiency improvement: **1.2-1.5x** (100 Gbps), **1.9-2.3x** (400 Gbps)
- Performance comparable to non-blocking fat-tree
- Working prototype on **32 A100 GPUs**
- Large-scale simulation validates at scale

## How It Works
- **Regional reconfiguration**: Only reconfigure within local region, not globally
- **OCS (Optical Circuit Switching)**: Runtime circuit reconfiguration
- Exploits MoE's **communication locality** - tokens tend to go to nearby experts
- Electrical + optical hybrid fabric

## Significance
First system that dynamically reconfigures network topology DURING MoE training. Shows that MoE's locality means you don't need static CLOS non-blocking fabric.
