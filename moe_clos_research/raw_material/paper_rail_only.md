# Rail-only: A Low-Cost High-Performance Network for Training LLMs with Trillion Parameters

> IEEE Hot Interconnects 2024
> Authors: W Wang, M Ghobadi, K Shakeri (MIT CSAIL)
> URL: https://ieeexplore.ieee.org/abstract/document/10664412/

## Core Thesis
LLM training generates sparse communication patterns → does NOT require any-to-any full-bisection CLOS network.

## Key Results
- **Spine layer eliminated** - This IS the "sparse CLOS" concept: removing the spine tier in traditional CLOS
- Network cost reduction: **38-77%**
- Network power reduction: **37-75%**
- For MoE all-to-all traffic: only **4.1-5.6% completion time overhead** (via forwarding)
- Same training performance as full CLOS
  
## How It Works
- Eliminates spine layer; keeps only rail-level (leaf) switches
- MoE tokens forwarded through remaining rails with minimal overhead
- Key insight: LLM/MoE traffic does not need any-to-any full-bisection bandwidth

## Significance
First paper to explicitly show that **the spine layer of CLOS is wasted for LLM workloads**, especially for sparse MoE models.
