# 芯片层（交换 ASIC + OCS）深化调研（2026-07-12）

## 来源
- Google 搜索 "Broadcom Tomahawk 5 51.2T switch ASIC specifications"
- Google 搜索 "Google Jupiter OCS MEMS optical circuit switch specifications port count switching time"
- SemiAnalysis: "Google OCS Apollo: The >$3 Billion Game-Changer" https://newsletter.semianalysis.com/p/google-apollo-the-3-billion-game
- arXiv 2208.10041: "Landing Optical Circuit Switching at Datacenter Scale"
- OCP OCS White Paper (April 2026): https://www.opencompute.org/documents/ocp-ocs-white-paper-april-2026-final-pdf  
- Google Cloud Blog Jupiter 演进：https://cloud.google.com/blog/topics/systems/the-evolution-of-googles-jupiter-data-center-network
- Introl Blog: "Ethernet Switches for AI: The 51.2Tbps Platforms"

## 交换 ASIC 代际演进

### Broadcom Tomahawk 系列
| 代 | 带宽 | 工艺 | 端口×速率 | SerDes | 关键特性 | 年份 |
|---|------|------|----------|--------|---------|------|
| TH4 | 25.6 Tbps | 7nm | 64×400G | 256×56G PAM4 | — | ~2020 |
| TH5 | 51.2 Tbps | 5nm | 64×800G | 512×112G PAM4 | ~250ns 转发时延；**Bailly** CPO 变体（八个 6.4T 硅光引擎，比可插拔降 70% 光互连功耗） | 2024 |
| TH6 (下一代) | ~102.4 Tbps | — | — | — | 2026 被提及 "next-generation silicon approaches 102.4Tbps" | ~2026-2027? |
| Jericho3-AI | — | — | — | — | Broadcom 面向 AI 的交换+路由 ASIC | — |

### 竞争格局
- **NVIDIA Spectrum-4 / Spectrum-X**：AI 原生 Ethernet，与 InfiniBand 竞争
- **Cisco Silicon One**：统一/模块化 AI 平台
- **Marvell Teralynx**：51.2T 级交换芯片
- **Arista 7800R4** 系列：模块化 AI 交换平台

### 高 radix 对 Clos 稀疏化的意义
> "for a given number of endpoints and a specified network topology, the greater the switch radix, the smaller the network diameter" (Alibaba C4P 论文)

高 radix 交换芯片推高后，fat-tree/Clos 所需的层级数和交换机总数减少——这本身就是一种"稀疏化"。51.2T (TH5) → 102.4T (TH6) → 204.8T (未来) 正使每台交换机连接更多端点，理论上降低网络层数。

## OCS 光交换器件

### Google Jupiter OCS 规格
- **架构**：3D MEMS（微电机系统），创建任意透明光路径
- **端口数**：**136×136**（来自 arXiv 论文 "Landing OCS at Datacenter Scale"）
- **切换时延**：**毫秒级**（典型 1.5ms-3ms；100 端口 600μm 镜面直径 MEMS OCS 达 1.5ms）
- **关键特性**：
  - 镜面稳定后保持状态，**零额外功耗**、**零排队延迟**
  - 消除 O-E-O（光电光）转换，**大幅节省功耗与延迟**
- **规模化**：OCS 层支持增量扩展（相对于 Clos 的全量重布线）

### 为何 OCS 是动态稀疏化的物理基础
> "The OCS platform allows the Jupiter architecture to dynamically restripe links between data center aggregation blocks based on changing traffic patterns." (Google)

OCS 使拓扑从"静态布线绑定"变为"**软件定义拓扑**"：
- 拓扑随流量模式变化，无需物理重布线
- 支持不同速率（40G→100G→200G→400G）的光模块增量升级
- "3x faster fabric reconfiguration compared to pre-evolution Clos fabrics that used a patch panel based interconnect"

### CPO（共封装光学）
- Broadcom **Bailly**：将 Tomahawk 5 与 8×6.4Tbps 硅光引擎封装在一起
- **降 70% 光互连功耗** vs 可插拔光模块
- 未来 CPO + OCS 可能彻底消灭传统电气 spine

### 半分析（SemiAnalysis）估计
- Google Apollo/OCS 已为其节省 **>30 亿美元**（网络成本 + 运营成本）
- 这是 OCS 动态稀疏化在 hyperscaler 尺度上的直接经济效益量化

## 趋势研判（芯片研究者视角）
1. **交换 ASIC radix 持续翻倍**：51.2T→102.4T→204.8T，直接减少网络层级，使"spine 消亡"更容易
2. **CPO 降低光互连功耗**：硅光引擎 + 交换芯片共封装是降功耗的关键使能技术
3. **OCS 从 Google 独家走向行业**：OCP 已成立 OCS 子项目（2026 白皮书），标准化将加速
4. **光电混合交换**：电交换（per-packet, ~ns 粒）+ 光交换（拓扑重构, ~ms 粒）是明确趋势
