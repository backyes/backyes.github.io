# Sparse Clos 调研 - 来源清单

> 所有调研访问过的网站地址（含成功/失败），按主题分类。状态截至 2026-07-12。

## A. Clos 基础理论与 seminal 论文

| # | 标题 | URL | 来源质量 | 状态 |
|---|------|-----|---------|------|
| 1 | Clos 1953 原始论文 (Bell System Technical Journal) | https://ieeexplore.ieee.org/document/1106836 (需补充) | primary | 待访问 |
| 2 | Jupiter Rising: A Decade of Clos Topologies (SIGCOMM 2015) | https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p183.pdf | primary | ✅ |
| 3 | Jupiter Rising (ACM DL) | https://dl.acm.org/doi/10.1145/2785956.2787508 | primary | ✅ |
| 4 | Hedera: Dynamic Flow Scheduling (NSDI 2010) | https://raghavan.usc.edu/papers/hedera-nsdi10.pdf | primary | ✅ |
| 5 | Per-packet Load-balanced Routing for Clos (CoNEXT 2013) | https://conferences.sigcomm.org/co-next/2013/program/p49.pdf | primary | ✅ |
| 6 | DRILL: Micro Load Balancing for Clos (SIGCOMM 2017) | https://pbg.cs.illinois.edu/papers/ghorbani17drill.pdf | primary | ✅ |
| 7 | FatPaths: Routing in Supercomputers/DC/Clos (ETH SPCL) | https://spcl.inf.ethz.ch/Publications/.pdf/fat-paths.pdf | primary | ✅ |
| 8 | On the data path performance of leaf-spine fabrics (IEEE HotI 2013) | https://ieeexplore.ieee.org/abstract/document/6627738/ | primary | ✅ |
| 9 | Minimal rewiring: Efficient live expansion for Clos DCN (NSDI 2019) | https://www.usenix.org/conference/nsdi19/presentation/zhao | primary | ✅ |

## B. Sparse Clos / 光纤 DCN / Spineless 设计

| # | 标题 | URL | 来源质量 | 状态 |
|---|------|-----|---------|------|
| 10 | Threshold-based routing-topology co-design for optical DCN (TROD, IEEE/ACM ToN 2023) | https://ieeexplore.ieee.org/abstract/document/10102400/ | primary | ✅ |
| 11 | Spineless data centers (HotNets 2020) | https://dl.acm.org/doi/abs/10.1145/3422604.3425945 | primary | ✅ |
| 12 | OpenOptics: Open Research Framework for Optical DCN (NSDI 2026) | https://arxiv.org/abs/2411.18319 | primary | ✅ |
| 13 | Optimal Oblivious Load-Balancing for Sparse Traffic (INFOCOM 2026) | https://arxiv.org/abs/2601.02537 | primary | ✅ |
| 14 | RNG: Flat Datacenter Networks at Scale (arXiv 2026) | https://arxiv.org/abs/2604.15261 | primary | ✅ |

## C. 工业界 / Hyperscaler 实践

| # | 标题 | URL | 来源质量 | 状态 |
|---|------|-----|---------|------|
| 15 | The evolution of Google's Jupiter (Google Cloud Blog 2022) | https://cloud.google.com/blog/topics/systems/the-evolution-of-googles-jupiter-data-center-network | primary | ✅ |
| 16 | Jupiter Evolving (SIGCOMM 2022, Google Research) | https://research.google/pubs/jupiter-evolving-transforming-googles-datacenter-network-via-optical-circuit-switches-and-software-defined-networking/ | primary | ✅ |
| 17 | Jupiter Evolving (ACM DL) | https://dl.acm.org/doi/abs/10.1145/3544216.3544265 | primary | ✅ |
| 18 | How flat is replacing fat in AWS DC networks (Amazon Science 2026) | https://www.amazon.science/blog/how-flat-is-replacing-fat-in-aws-data-center-networks | primary | ✅ |
| 19 | AWS RNG (James Hamilton blog) | https://perspectives.mvdirona.com/ | primary | 待访问 |
| 20 | Disaggregated Scheduled Fabric: Scaling Meta's AI Journey (Meta 2025) | https://engineering.fb.com/2025/10/20/data-center-engineering/disaggregated-scheduled-fabric-scaling-metas-ai-journey/ | primary | ✅ |
| 21 | Collective Communication for 100k+ GPUs (Meta Llama4, arXiv 2510.20171) | https://arxiv.org/abs/2510.20171 | primary | ✅ |
| 22 | This AI Network Has No Spine (The Next Platform 2024) | https://www.nextplatform.com/connect/2024/08/23/this-ai-network-has-no-spine-and-thats-a-good-thing/1640017 | secondary | ✅ |
| 23 | Rail-Only architecture (Hot Interconnects 2024, CSAIL) | (via Next Platform) | secondary | ✅ |
| 24 | Alibaba C4P / 3-Tier Clos (arXiv 2024) | https://arxiv.org/html/2406.04594v1 (相关) | primary | ✅ |
| 25 | Doubling all2all Performance with NCCL 2.12 (PXN, NVIDIA) | https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/ | primary | ✅ |
| 26 | RailS: Load Balancing for All-to-All (arXiv 2510.19262, 2025) | https://arxiv.org/html/2510.19262v1 | primary | ✅ |
| 27 | C4: Boosting Large-scale Parallel Training Efficiency (arXiv 2406.04594) | https://arxiv.org/html/2406.04594v1 | primary | ✅ |

## D. 拓扑对比 / 替代方案

| # | 标题 | URL | 来源质量 | 状态 |
|---|------|-----|---------|------|
| 28 | Jellyfish: Networking Data Centers Randomly (NSDI 2012) | https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final82.pdf | primary | ✅ |
| 29 | Jellyfish (USENIX HotCloud 2011 版) | https://www.usenix.org/event/hotcloud11/tech/final_files/Singla.pdf | primary | ✅ |
| 30 | Technology-Driven Dragonfly Topology (ISCA 2008) | https://www.researchgate.net/publication/4349973_Technology-Driven_Highly-Scalable_Dragonfly_Topology | primary | ✅ |
| 31 | Slim Fly: Cost Effective Low-Diameter Topology (SC16/arXiv) | https://arxiv.org/pdf/1912.08968 | primary | ✅ |
| 32 | Network topologies for large-scale compute centers (ETH HotI 2016) | https://htor.inf.ethz.ch/publications/img/HotI16-Topologies-SlimFly.pdf | primary | ✅ |
| 33 | Analyzing Cost-Performance Tradeoffs of HPC Network Designs (SIGSIM PADS 2019) | https://dl.acm.org/doi/10.1145/3316480.3325516 | primary | ✅ |
| 34 | Efficient Direct-Connect Topologies for Collective Communication (NSDI 2025) | https://www.usenix.org/system/files/nsdi25-zhao-liangyu.pdf | primary | ✅ |
| 35 | Survey on topology of high-performance interconnection networks (2026) | https://www.sciopen.com/article/10.11887/j.issn.1001-2486.25110046 | primary | ✅ |
| 36 | Topology Designs for Data Centers (MDPI Encyclopedia 2023) | https://encyclopedia.pub/entry/46512 | secondary | ✅ |
| 37 | GPU Cluster Network Topology Design guide (2026) | https://introl.com/blog/gpu-cluster-network-topology-fat-tree-dragonfly-rail-optimized-2025 | blog | ✅ |
| 38 | Network Topology Optimization for AI Workloads (Signal65 2025) | https://signal65.com/wp-content/uploads/2025/10/Signal65-Insights_Network-Topology-Optimization-for-AI-Workloads.pdf | industry | ✅ |

## E. 深化调研新增来源

| # | 标题 | URL | 来源质量 | 状态 |
|---|------|-----|---------|------|
| 39 | RNG: Flat DC at Scale (arXiv 全文) | https://arxiv.org/abs/2604.15261 | primary | ✅ |
| 40 | James Hamilton blog: Flat DC at Scale | https://perspectives.mvdirona.com/2026/06/flat-datacenter-networks-at-scale/ | primary | ✅ |
| 41 | Meta DSF 工程博客 | https://engineering.fb.com/2025/10/20/data-center-engineering/disaggregated-scheduled-fabric-scaling-metas-ai-journey/ | primary | ✅ |
| 42 | Meta NCCLX Llama4 (arXiv HTML 全文) | https://arxiv.org/html/2510.20171v4 | primary | ✅ |
| 43 | Meta NCCLX Llama4 (arXiv 摘要) | https://arxiv.org/abs/2510.20171 | primary | ✅ |
| 44 | Landing Optical Circuit Switching at DC Scale (arXiv) | https://arxiv.org/pdf/2208.10041 | primary | ✅ |
| 45 | OCP OCS White Paper (April 2026) | https://www.opencompute.org/documents/ocp-ocs-white-paper-april-2026-final-pdf | primary | ✅ |
| 46 | SemiAnalysis: Google Apollo >$3B | https://newsletter.semianalysis.com/p/google-apollo-the-3-billion-game | secondary | ✅ |
| 47 | 2026 综述 (国防科大) | https://www.sciopen.com/article/10.11887/j.issn.1001-2486.25110046 | primary | ✅ |
| 48 | Zcube SIGCOMM 2025 DOI | https://doi.org/10.1145/3718958.3750503 | primary | ✅ |
| 49 | Ethernet Switches for AI (Introl) | https://introl.com/blog/ethernet-switches-ai-tomahawk-spectrum-x-51-2t-2025 | blog | ✅ |
| 50 | OCS expanders 关联论文 A | https://dl.acm.org/doi/10.1145/2834050.2834059 | primary | ✅（Hamilton 引） |
| 51 | OCS expanders 关联论文 B | https://dl.acm.org/doi/10.1145/2999572.2999580 | primary | ✅（Hamilton 引） |
| 52 | Broadcom TH5 Bailly CPO 新闻稿 | https://investors.broadcom.com/news-releases/news-release-details/broadcom-delivers-industrys-first-512-tbps-co-packaged-optics | primary | ✅ |
| 53 | Google Jupiter OCS arXiv | https://arxiv.org/pdf/2208.10041 | primary | ✅ |
| 54 | Clos 1953 原始论文 (Wikipadia 引用) | https://en.wikipedia.org/wiki/Clos_network | primary (ref) | ✅ |
| 55 | Clos 1953 全文 (archive.org) | https://archive.org/details/bstj32-2-406 | primary | ✅ |
| 56 | Clos 1953 DOI | https://doi.org/10.1002/j.1538-7305.1953.tb01433.x | primary | ✅ |
| 57 | 百度 万卡至十万卡智算网络进化 (腾讯云社区) | https://cloud.tencent.com/developer/article/2503872 | blog | ✅（搜索结果摘要） |
| 58 | 火山引擎 火鸿AI-HPC Fat-Tree | https://developer.volcengine.com/articles/ | blog | ✅（搜索结果摘要） |
| 59 | H3C AI Spine-Leaf 网络技术 | https://www.h3c.com/ | blog | ✅（搜索结果摘要） |
| 60 | 智能算网 AI Fabric 2.0 (华为) | https://e.huawei.com/ | primary | ✅（搜索结果摘要） |

## F. 仍待补充

- AWS RNG arXiv PDF 全文本地提取（需 pdftotext 工具）
- Alibaba HPN7.0 论文全文
- 国产 hyperscaler（华为/字节等）更详细网络拓扑公开资料
- NANOG Meta DSF keynote 视频/PPT
