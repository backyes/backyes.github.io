# DeepEP 分析报告索引

## 主报告

| 文件 | 内容 | 行数 |
|------|------|------|
| [DEEP_EP_ANALYSIS_POST.md](DEEP_EP_ANALYSIS_POST.md) | 完整深度分析报告（理论+源码） | 2420 |

## 分问题详细分析

| # | 文件 | 问题 | 行数 |
|---|------|------|------|
| Q1 | [01_q1_moe_communication_fundamental.md](01_q1_moe_communication_fundamental.md) | MoE通信不是普通All-to-All | 114 |
| Q2 | [02_q2_deepep_vs_gemm.md](02_q2_deepep_vs_gemm.md) | DeepEP不是优化GEMM | 87 |
| Q3 | [03_q3_dispatch_combine_difficulty.md](03_q3_dispatch_combine_difficulty.md) | Dispatch/Combine为什么难 | 106 |
| Q4 | [04_q4_permutation_elimination.md](04_q4_permutation_elimination.md) | 消除permutation带来的额外搬运 | 115 |
| Q5 | [05_q5_ibgda_foundation.md](05_q5_ibgda_foundation.md) | IBGDA是共同底座 | 104 |
| Q6 | [06_q6_warp_specialization.md](06_q6_warp_specialization.md) | Warp specialization | 146 |
| Q7 | [07_q7_sm_resource_overlap.md](07_q7_sm_resource_overlap.md) | SM资源与overlap | 121 |
| Q8 | [08_q8_forwarding_gpu.md](08_q8_forwarding_gpu.md) | Forwarding GPU | 139 |
| Q9 | [09_q9_nvshmem_v1_v2.md](09_q9_nvshmem_v1_v2.md) | NVSHMEM角色演变 | 156 |
| Q10 | [10_q10_data_flag_primitives.md](10_q10_data_flag_primitives.md) | Data/Flag两套原语 | 187 |
| Q11 | [11_q11_release_acquire.md](11_q11_release_acquire.md) | Release-Acquire机制 | 187 |
| Q12 | [12_q12_v1_to_v2_evolution.md](12_q12_v1_to_v2_evolution.md) | V1→V2架构演进 | 221 |

## 总结

| 文件 | 内容 | 行数 |
|------|------|------|
| [SUMMARY.md](SUMMARY.md) | 总结 | 82 |
| [README.md](README.md) | 问题去重清单 | 43 |

## 验证结论

**所有 12 个核心论断都验证为正确** ✅

