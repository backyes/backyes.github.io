# Rethinking Network Topologies for Cost-Effective MoE LLM Serving

> arXiv 2605.00254, Apr 2026
> Authors: Junsun Choi, Sam Son, Sunjin Choi, Hansung Kim, Yakun Sophia Shao, Scott Shenker, Sylvia Ratnasamy, Borivoje Nikolic (UC Berkeley)
> URL: https://arxiv.org/abs/2605.00254

## Core Thesis
**Question**: Is expensive high-bandwidth scale-up network (CLOS-based) necessary for MoE LLM serving?
**Answer**: No. Lower-cost switchless topologies are more cost-effective across all scenarios.

## Key Findings

### 1. Cost-Effectiveness Improvement
- Switchless topologies improve cost-effectiveness by **20.6-56.2%** over scale-up (CLOS) topology
- 3D full-mesh is **Pareto-optimal** in performance-cost tradeoff
- Current scale-up link bandwidths are **over-provisioned**: reducing bandwidth improves throughput per cost by up to **27%**

### 2. Topology Types Compared
| Topology | Description | Switch Cost | A2A Bandwidth |
|----------|-------------|-------------|---------------|
| Scale-up (CLOS) | High-bandwidth switched (NVSwitch, NVLink) | High | High |
| Scale-out (Ethernet/IB) | Standard CLOS fat-tree | Medium | Medium |
| 3D Torus | Switchless, mesh with wrap-around (TPU) | Low | Lower |
| 3D Full-Mesh | Direct connect within dimension (Huawei UB-Mesh) | Low | Medium |

### 3. Why Switchless Wins for MoE
- MoE's sparse expert activation means all-to-all communication is already limited
- Computation-communication overlap (dual-batch overlap, DBO) further reduces exposed A2A time
- Speculative decoding expands DBO-friendly regime
- Saved network cost → deploy more clusters → higher aggregate throughput

### 4. Forward-Looking Analysis
- Advantage persists across **NVIDIA Blackwell and Rubin** GPU generations
- Compute/memory scaling won't eliminate the benefit of switchless networks
- 150-300 GB/s link bandwidth is the sweet spot (vs current 450 GB/s)

### 5. Software Optimization Impact
- **Dual Batch Overlap (DBO)**: Effective at large batch sizes, nearly eliminates exposed communication time
- **Speculative Decoding**: Shifts batch sizes into DBO-friendly regime
- Combined, they push the cost-optimal point toward lower bandwidths

## Significance for Sparse CLOS × MoE
This paper provides the strongest evidence that for MoE inference:
- **Full CLOS bisection bandwidth is unnecessary**
- **Switchless/sparse topologies exploit MoE's communication sparsity**
- **The cost benefit comes from avoiding expensive spine switches**
- **MoE dispatch/gather A2A is not bandwidth-hungry enough to justify CLOS**
