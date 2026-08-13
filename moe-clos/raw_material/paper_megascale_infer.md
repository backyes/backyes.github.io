# MegaScale-Infer: Serving MoE with Disaggregated Expert Parallelism

> SIGCOMM 2025
> Authors: Ruidong Zhu, Ziheng Jiang, Chao Jin, Peng Wu, et al.
> URL: https://arxiv.org/abs/2504.02263

## Core Thesis
Disaggregate attention and FFN modules within each MoE layer → independent scaling + tailored parallelism for each.

## Key Techniques
1. **Ping-pong pipeline parallelism**: Partition request batch into micro-batches, shuttle between attention and FFN
2. **M2N communication library**: Eliminates GPU-CPU copies, group init overhead, GPU sync
3. **Heterogeneous deployment**: Different hardware for attention vs FFN modules

## Key Insight
MoE inference shifts FFN from compute-intensive to **memory-intensive** → disaggregation leverages this asymmetry.

## Key Results
- Per-GPU throughput: **up to 1.90×** vs state-of-the-art
- Effectively hides communication overhead

## Significance
Shows the fundamental shift in bottleneck (compute→memory) enables new parallelism strategies that reduce network pressure.
