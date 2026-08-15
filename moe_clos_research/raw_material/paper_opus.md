# Opus: Photonic Rail-Optimized Fabric in ML Datacenters

> arXiv 2602.12521, 2026
> Authors: Eric Ding, Barry Lyu, Bhaskar Kataria, Rachee Singh (Microsoft)
> URL: https://arxiv.org/abs/2602.12521

## Core Thesis
Replace electrical switches in rail fabric with optical circuit switches. Exploit non-overlapping communication phases of different parallelism dimensions.

## Key Insight
Different parallelism dimensions (TP, EP, DP) have **non-overlapping communication phases** within a training iteration. This allows time-multiplexing a single set of physical ports across circuit configurations tailored to each phase.

## Key Results
- Network power reduction: **23×**
- Network cost savings: **4×**
- Tested on Perlmutter supercomputer and simulated at 2048 GPUs
- Only modest training overhead at production OCS reconfiguration latencies

## Significance
Photonic rails achieve order-of-magnitude power savings while maintaining performance. The parallelism-driven reconfiguration insight is generalizable.
