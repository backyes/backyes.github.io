# AWS RNG 深化调研发现（2026-07-12）

## 来源
1. arXiv 论文：https://arxiv.org/abs/2604.15261 "RNG: Flat Datacenter Networks at Scale"
   - 作者：Giacomo Bernardi, Ratul Mahajan, C. Seshadhri, Enrico Carlesso, Chinchu Merine Joseph, Saurabh Kumar, Pavan Manikonda, Luiza Popa, Randy Ram, Steven Robinson, Elizabeth Tennent
2. James Hamilton blog：https://perspectives.mvdirona.com/2026/06/flat-datacenter-networks-at-scale/ （2026-06-08）
3. Amazon Science blog：https://www.amazon.science/blog/how-flat-is-replacing-fat-in-aws-data-center-networks

## 关键发现

### 1. arXiv 摘要（权威原文）
> "We design and deploy in production the first flat datacenter networks. Our design, called RNG, is based on quasi-random graphs. While the cost and fault-tolerance benefits of such topologies have been long known, their practical realization has been hampered by a lack of scalable routing and cabling approaches. RNG has a new distributed routing protocol that exploits the properties of random graphs to find a large number of edge disjoint paths between pairs of endpoints. It uses a novel passive optical device that internally shuffles cables, which makes its cabling complexity similar to that of fat trees. We show that RNG matches or exceeds the performance of fat trees for a range of traffic patterns, despite being up to 45% cheaper. RNG is now the default datacenter network for most workloads at Amazon."

要点：
- **首个生产部署的扁平数据中心网络**
- 基于准随机图（quasi-random graphs）
- 新型分布式路由协议，利用随机图性质找到大量**边不相交路径**（edge disjoint paths）
- 无源光器件内部打乱线缆，使布线复杂度与 fat-tree 相当
- 较 fat-tree 便宜**最多 45%**，性能持平或超越
- 现为 Amazon 大多数工作负载的默认 DC 网络

### 2. James Hamilton 叙事：完整历史谱系
**理论根源（1970s expanders）**：
- 1976 Leslie Valiant 最早讨论 expander 图
- Alon-Boppana 工作：理解"最优可能"expander
- Lubotzky, Phillips, Sarnak 构造最优 expander（高级数论，仅特定规模/度数可用）
- **1991 Friedman 证明：随机布线网络以高概率近似最优 expander**
- 2023 新数学结果：随机图确实匹配该下界
- 启示："if you want an optimal network for routing, you could simply wire it at random"

**产业路径（fat-tree 主导）**：
- 1980s 中期起，受 Clos 启发，通信网建在 fat-tree（folded Clos）上
- 2009 VL2（Greenberg 等 9 人）：fat-tree + flat addressing + Valiant Load Balancing。2019 获 SIGCOMM 时间检验奖
- VL2 证明：即便在结构化拓扑内，**流量随机化**也能提升性能。但底层网络仍"层次化、刚性、布线复杂"

**Jellyfish 2012**：将随机图与 DC 网络连接。但基于简单理论模型与仿真，留下三大硬问题未解：
1. 路由（随机图路径太多样，难）
2. 布线（端点随机选择，难）
3. 运维（不可预测，难）
"Building random networks at scale remained an elusive target: routing, cabling, and operations were the three unsolved challenges."

### 3. RNG 起源故事
- 2023 Giacomo Bernardi（AWS principal scientist）研究 Penrose tiling（彭罗斯密铺，形状无重复铺满）排列 DC 路由器
- Ratul Mahajan（Amazon Scholar，UW 教授）加入，数月仿真推进
- 2024 年中碰壁：Penrose tiling 仿真不可靠，效率增益不达标
- **"用随机替代结构"后效果大幅提升**，内部笑话："just be random!"
- 但缺理论支撑 Amazon 规模，需新模型预测性能、保证弹性、可运维
- Slack 招募："any random graph experts here?" -> Seshadhri Comandur（Amazon Scholar，理论计算机教授）加入

### 4. 三大难题的破解
| 难题 | 解决方案 | 机制 |
|------|---------|------|
| 路由 | **Spraypoint** | 转发方案，利用图的 expansion 性质分发流量，**不让转发状态压垮路由器内存** |
| 布线 | **ShuffleBox** | 无源光器件，内部走线 + 随机化 ShuffleBox 间布线 = 表现如真随机图的"准随机"图 |
| 运维 | **复用现有硬件** | RNG 用与 fat-tree DC 完全相同的路由器与光模块，软件定义 |

### 5. 关键设计哲学（Hamilton 总结）
> "RNG uses 69% few routers than the current common Fat Tree and, again, randomization is a key tool being used to avoid hot spots but this time it's **randomization at the link hardware layer rather than in the protocol layer above**."

> "Both approaches are fairly effective at avoiding hot spots but the RNG approach is much more hardware efficient. Perhaps the lesson is to **pull optimizations down to the lowest level possible** to avoid redundancy not fundamentally required by the approach."

**核心洞察**：VL2 把随机化放在协议层（VLB），RNG 把随机化下沉到链路硬件层。两者都避免热点，但 RNG 硬件效率更高。**"把优化压到尽可能低的层级"**是设计教训。

### 6. VL2 对比（Hamilton）
> "The VL2 paper proposed replacing super expensive chassis switches with custom routers based upon commodity ASICs in a Fat Tree topology. To make it work as well as the big chassis switches and avoid hot spots, VL2 proposed Valiant Load Balancing. The price reduction of making this move was absolutely staggering at the time and that has been our design choice for a decade and a half."

### 7. 历史 reference 补充
Jellyfish 是早期随机图 DC 拓扑提案，但**显式将 expander graph 与 DC 设计关联**的是：
- https://dl.acm.org/doi/10.1145/2834050.2834059
- https://dl.acm.org/doi/10.1145/2999572.2999580

### 8. 园区规划维度（前 AWS 房产开发者评论）
> "the most interesting implication of RNG isn't the networking architecture. It's the flexibility it creates for campus planning. Continuous scaling changes how we think about land utilization, phasing, capital deployment, and ultimately monetization."

RNG 的增量扩展能力改变了超大规模数据中心的**土地/分期/资本部署/变现**逻辑--不仅是网络架构创新。

## 待补充
- arXiv 2604.15261 全文 PDF（量化的拓扑参数、度数、直径、对比实验数据）
- Spraypoint 路由算法的形式化描述
