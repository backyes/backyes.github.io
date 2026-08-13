# Sparse CLOS × MoE 推理调研 - 来源清单

> 调研日期：2026-07-12
> 主题：MoE 稀疏性带来的 CLOS 网络效率问题 & "sparse CLOS" 方案的性能/成本收益

## 1. 核心论文（已下载 PDF）

| # | 标题 | URL | 来源质量 |
|---|------|-----|---------|
| 1 | Rethinking Network Topologies for Cost-Effective MoE LLM Serving | https://arxiv.org/abs/2605.00254 | primary (UC Berkeley) |
| 2 | Rail-only: A Low-Cost High-Performance Network for Training LLMs with Trillion Parameters | https://ieeexplore.ieee.org/abstract/document/10664412/ | primary (MIT CSAIL) |
| 3 | MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed MoE Training | https://arxiv.org/abs/2501.03905 | primary (HKUST, SIGCOMM 2025) |
| 4 | UBEP: Re-architecting Expert Parallelism Communication Library for Production Superpods | https://arxiv.org/abs/2607.06202 | primary (2026) |
| 5 | Speculative MoE: Communication Efficient Parallel MoE Inference | https://arxiv.org/abs/2503.04398 | primary (2025) |
| 6 | Opus: Photonic Rail-Optimized Fabric in ML Datacenters | https://arxiv.org/abs/2602.12521 | primary (Microsoft, 2026) |
| 7 | Switching Efficiency: A Framework for Dissecting AI DC Network Efficiency | https://arxiv.org/abs/2604.14690 | primary (2026) |
| 8 | MegaScale-Infer: Serving MoE with Disaggregated Expert Parallelism | https://arxiv.org/abs/2504.02263 | primary (SIGCOMM 2025) |

## 2. 相关论文（引用/搜索发现，未下载 PDF）

| # | 标题 | URL | 来源质量 |
|---|------|-----|---------|
| 9 | Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling | https://arxiv.org/abs/2503.04398 (v2+ redirect) | primary |
| 10 | RailS: Load Balancing for All-to-All in Distributed MoE Training | https://ieeexplore.ieee.org/abstract/document/11450478/ | primary (IEEE ToN 2026) |
| 11 | Spine-free Networks for LLM Training | https://ieeexplore.ieee.org/abstract/document/10884699/ | primary (IEEE Micro 2025) |
| 12 | RailX: Flexible, Scalable, Low-Cost Network for LLM Training | https://arxiv.org/abs/2507.18889 | primary (2025) |
| 13 | LINA: Accelerating Distributed MoE Training and Inference | https://www.usenix.org/conference/atc23/presentation/li-jiamin | primary (ATC 2023) |
| 14 | Janus: Unified Distributed Training for Sparse MoE | https://dl.acm.org/doi/abs/10.1145/3603269.3604869 | primary (SIGCOMM 2023) |
| 15 | Astral: Datacenter Infrastructure for LLM Training at Scale | https://dl.acm.org/doi/abs/10.1145/3718958.3750521 | primary (SIGCOMM 2025) |
| 16 | PROBE: Co-Balancing Computation and Communication in MoE Inference | https://arxiv.org/abs/2602.00509 | primary (2026) |
| 17 | Grace-MoE: Grouping and Replication with Locality-aware Routing | https://arxiv.org/abs/2509.25041 | primary (2025) |
| 18 | InfiniteHBD: DC-scale HBD for LLM with OCS | https://dl.acm.org/doi/abs/10.1145/3718958.3750468 | primary (SIGCOMM 2025) |
| 19 | Megascale-moe: Large-scale Communication-efficient Training of MoE | https://dl.acm.org/doi/abs/10.1145/3767295.3769325 | primary (2026) |
| 20 | Insights into DeepSeek-V3: Scaling Challenges and Hardware Reflections | https://dl.acm.org/doi/abs/10.1145/3695053.3731412 | primary (2025) |
| 21 | MoESys: Distributed MoE Training and Inference for Internet Services | https://ieeexplore.ieee.org/abstract/document/10528887/ | primary (IEEE TPDS 2024) |
| 22 | Shortcut-connected Expert Parallelism for MoE | https://arxiv.org/abs/2404.05019 | primary (2024) |
| 23 | Reducing Cross-Pod Communication for MoE with Hybrid Parallelism | https://ieeexplore.ieee.org/abstract/document/11417440/ | primary (IEEE TPDS 2026) |
| 24 | DFS: Dynamic Flow Spraying with Bounded Reordering for AI Clusters | https://dl.acm.org/doi/abs/10.1145/3805621.3807657 | primary (2026) |
| 25 | Survey: Communication Optimization in Distributed Training | https://ieeexplore.ieee.org/abstract/document/11139179/ | primary (2025) |
| 26 | Serving LLMs on Huawei CloudMatrix384 | https://arxiv.org/abs/2506.12708 | primary (2025) |
| 27 | Switching Efficiency: A Framework for Dissecting AI DC Network Efficiency | https://arxiv.org/abs/2604.14690 | primary (2026) |
| 28 | A Survey on Inference Optimization for MoE Models | https://dl.acm.org/doi/abs/10.1145/3794845 | primary (ACM CS 2026) |

## 3. Google Scholar 搜索结果页

| # | 搜索关键词 | URL | 状态 |
|---|-----------|-----|------|
| S1 | MoE mixture of experts all-to-all communication bottleneck sparse expert parallelism | https://scholar.google.com/scholar?q=MoE+mixture+of+experts+all-to-all+communication+bottleneck+sparse+expert+parallelism&hl=en&as_sdt=0%2C5&as_ylo=2023&as_yhi=2026 | ✅ |
| S2 | "sparse MoE" CLOS network topology expert parallelism inference | https://scholar.google.com/scholar?q=%22sparse+MoE%22+CLOS+network+topology+expert+parallelism+inference&hl=en&as_sdt=0%2C5&as_ylo=2023&as_yhi=2026 | ✅ |
| S3 | rail-only network MoE all-to-all topology sparse | https://scholar.google.com/scholar?q=rail-only+network+MoE+all-to-all+topology+sparse&hl=en&as_sdt=0%2C5&as_ylo=2023&as_yhi=2026 | ✅ |
| S4 | "speculative MoE" "communication efficient" inference | https://scholar.google.com/scholar?q=%22speculative+MoE%22+%22communication+efficient%22+inference&hl=en&as_sdt=0%2C5&as_ylo=2024&as_yhi=2026 | ✅ |
| S5 | "sparse CLOS" "mixture of experts" inference | https://scholar.google.com/scholar?q=%22sparse+CLOS%22+%22mixture+of+experts%22+inference&hl=en&as_sdt=0%2C5&as_ylo=2023&as_yhi=2026 | ✅ |
| S6 | MoE sparse all-to-all traffic bursty imbalanced network utilization | https://scholar.google.com/scholar?q=MoE+sparse+all-to-all+traffic+bursty+imbalanced+network+utilization&hl=en&as_sdt=0%2C5&as_ylo=2023&as_yhi=2026 | ✅ |

## 4. 其他搜索页

| # | 描述 | URL | 状态 |
|---|------|-----|------|
| W1 | Google: MixNet arXiv | https://www.google.com/search?q=MixNet+runtime+reconfigurable+optical+electrical+fabric+MoE+training+arXiv | ✅ |
| W2 | Google: DeepSeek V3 A2A CJL | https://www.google.com/search?q=DeepSeek+V3+all-to-all+communication+CJL+lossless+network+expert+parallelism | ✅ |
| W3 | Google: MegaScale-Infer arXiv | https://www.google.com/search?q=MegaScale-Infer+efficient+MoE+serving+disaggregated+expert+parallelism+arxiv | ✅ |

## 5. 直接访问的论文页

| # | 论文 | URL | 状态 |
|---|------|-----|------|
| D1 | Rethinking Network Topologies (arXiv abs) | https://arxiv.org/abs/2605.00254 | ✅ |
| D2 | Rethinking Network Topologies (arXiv HTML) | https://arxiv.org/html/2605.00254v1 | ✅ |
| D3 | Rail-only (IEEE) | https://ieeexplore.ieee.org/abstract/document/10664412/ | ✅ |
| D4 | UBEP (arXiv) | https://arxiv.org/abs/2607.06202 | ✅ |
| D5 | MixNet (ACM) | https://dl.acm.org/doi/abs/10.1145/3718958.3750465 | ✅（paywall） |
| D6 | MixNet (arXiv) | https://arxiv.org/abs/2501.03905 | ✅ |
| D7 | Speculative MoE (arXiv v1) | https://arxiv.org/abs/2503.04398v1 | ✅ |
| D8 | Opus (arXiv) | https://arxiv.org/abs/2602.12521 | ✅ |
| D9 | Switching Efficiency (arXiv) | https://arxiv.org/abs/2604.14690 | ✅ |
| D10 | MegaScale-Infer (ACM) | https://dl.acm.org/doi/abs/10.1145/3718958.3750506 | ✅（paywall） |
| D11 | MegaScale-Infer (arXiv) | https://arxiv.org/abs/2504.02263 | ✅ |
