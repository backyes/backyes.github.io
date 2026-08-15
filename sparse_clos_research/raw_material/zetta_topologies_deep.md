# Zetta 级拓扑与 Clos 1953 深化调研（2026-07-12）

## 来源
- 综述论文：https://www.sciopen.com/article/10.11887/j.issn.1001-2486.25110046
- Zcube 原文：YAN Z H, LI D, CHEN L, et al. "From ATOP to ZCube: automated topology optimization pipeline and a highly cost-effective network topology for large model training" (SIGCOMM 2025, pp 861-881) DOI: 10.1145/3718958.3750503
- 国产（国防科大）期刊，英文出版

## 关键发现

### 拓扑三分类体系（综述 2026）
1. **DC 数据中心拓扑**：Fat-tree 及变体、DCell、BCube、Jellyfish — 高带宽、可扩展、弹性
2. **HPC 并行计算拓扑**：高维 Mesh/Torus（低 radix）、Flattened Butterfly、Dragonfly/Dragonfly+、Slim Fly、Galaxyfly 等 — 低成本小二分带宽，但高度依赖自适应路由
3. **智能计算拓扑**：HammingMesh、Google TPU、Rail-Only、HPN7.0、Zcube、Zettafly — 从目标应用通信模式推导的定制结构设计

### 三规模最优推荐
> Based on publicly available pricing information, this study models each topology's per-endpoint copper/optical cable count, port count, cost, and power consumption, ultimately recommending the use of Zcube, Fat-tree, and Zettafly for building small-, medium-, and large-scale interconnection systems, respectively.

| 规模 | 推荐拓扑 | 备注 |
|------|---------|------|
| 小规模 | **Zcube** | 自动拓扑优化流水线，SIGCOMM 2025 |
| 中规模 | **Fat-tree (folded Clos)** | 传统方案，性能和可靠性经过验证 |
| 大规模 | **Zettafly** | 定制的低直径拓扑，适合 10 万+加速器 |

### Zcube（SIGCOMM 2025）
- SIGCOMM 2025 正会议论文（861-881 页）
- 核心思想：自动化拓扑优化流水线（ATOP → ZCube）
- 面向大模型训练的高度成本效益网络拓扑
- 成本建模：端点线缆数、端口数、成本、功耗

### Zettafly
- 面向 Zetta 级规模（10 万+加速器 vs fat-tree 数十亿美元互联成本）
- 具体结构未详述，综述指出属于"定制结构设计，从目标应用通信模式推导"
- 与 Dragonfly/Galaxyfly 类似属低直径高 radix 体系

### HammingMesh / HPN7.0
- **HammingMesh**：面向 DL 推理的拓扑，利用汉明距离的通信模式
- **HPN7.0**：Alibaba/国内的智能计算拓扑（"High-Performance Network"）
- 这些均属于"从应用通信模式推导的定制拓扑"，即 workload-aware 设计

### 拓扑设计的不变结论
> Network topology design is an engineering art of balance and compromise. It requires finding optimal solutions among multiple interdependent constraints, such as construction cost, power capacity, router port resource limitations, virtual channel constraints, efficient adaptive routing, and fault tolerance. Neither the lowest-cost topology nor the highest-performance, high-cost topology necessarily represents the optimal choice.

> The ideal topology should be highly aligned with the characteristics of its running applications, combining cost efficiency with sound design principles while maintaining simplicity in understanding, packaging, and deployment.

> when domestic router chips have limited port counts, constructing a topology that achieves high bandwidth, good scalability, and cost control presents a significant challenge. (国内芯片视角)

### Clos 1953 原始论文
Clos 1953 原文仍待 IEEE 全文补充，但其核心结论已通过综述和引用充分传达：
- C(n,m,r) 三级结构
- 严格无阻塞条件：m ≥ 2n-1
- 可重排无阻塞条件：m ≥ n
- 三级 Clos 的流路由多项式时间可解，但五级（fat-tree）无已知多项式算法
