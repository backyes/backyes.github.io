# HBM / CXL / Memory 市场调研中间数据
## 调研日期: 2026-07-05
## 数据来源: Google Search via Playwright, 深度文章阅读

---

## 1. HBM4 量产时间线与供应商状态

### SK Hynix
- 状态: HBM4 已实现量产，正在供应首批 HBM4 分配量
- 技术路线: 与 TSMC 合作，使用 3nm 代工工艺制造基础芯片 (Base Die)
- DRAM 节点: 1c DRAM node
- 封装: 自有 MR-MUF 封装技术
- HBM4E: 提前推进样品出货，预计 2026年6-7月向关键客户送样
- 1c DRAM 产能扩张: 从月产 ~20K 片扩大到 160K-190K 片 300mm 晶圆 (8-9倍增长)
- 市场定位: 为 NVIDIA Vera Rubin 架构提供早期产能，保持多数市场份额
- 基础设施投资: 超过此前公布金额的 4 倍
- 来源: Silicon Analysts (2026-07-02), TrendForce (2026-06-15), Reuters (2026-06-17)

### Samsung
- 状态: 2026年2月开始量产并出货 HBM4
- HBM4 性能: 11.7 Gbps (稳定), 最高 13 Gbps, 比 HBM3E 提升 22%
- 技术路线: 10nm 级 (1c) DRAM 核心, 10nm 级逻辑基础芯片, TC-NCF 键合
- 产能扩张: 2026年 HBM 产能增加 ~50%
- HBM4E: 计划 2026 下半年送样
- 主要客户: Google TPU, NVIDIA Vera Rubin
- 良率挑战: 2025年底遭遇 HBM4 良率问题导致延迟
- 来源: Reuters (2026-02-12, 2026-05-28, 2026-01-25), Silicon Analysts (2026-07-02)

### Micron
- 状态: 2026年初进入 HBM4 高量产，需求超过供给
- 2026 全年 HBM4 供应已售罄，只能满足一半需求
- 技术路线: 1β (第5代 10nm 级) DRAM 工艺
- HBM4E: 推进 2027 年底开发
- 定位: 专业化定制 AI 内存供应商
- 来源: LinkedIn/Mark Hirsch (2026-03-17), TrendForce (2025-11-13)

### NVIDIA Vera Rubin 认证
- 2026年6月: NVIDIA 认证 Samsung, SK Hynix, Micron 三家均为 Vera Rubin HBM4 供应商
- Vera Rubin 系统 Q3 2026 开始出货
- 来源: Yahoo Finance (2026-06-05)

---

## 2. HBM 在 AI 加速器中的成本占比

| 平台 | HBM 代 | HBM 容量 | 估算 HBM 成本 | 估算总制造成本 | HBM 占比 |
|------|--------|----------|--------------|---------------|---------|
| NVIDIA H100 SXM5 | HBM3 | 80 GB | ~$1,350 | ~$3,320 | ~41% |
| NVIDIA H200 SXM5 | HBM3e | 141 GB | ~$1,500 | ~$4,250 | ~35% |
| NVIDIA B200 | HBM3e | 192 GB | ~$2,900 | ~$6,400 | ~45% |
| NVIDIA GB200 | HBM3e | 384 GB | ~$5,800 | ~$13,500 | ~43% |
| AMD MI300X | HBM3 | 192 GB | ~$2,900 | ~$5,300 | ~55% |

来源: Silicon Analysts 成本模型 (2026-07-02)

---

## 3. HBM / DRAM 市场规模预测

### TrendForce 数据 (2026年5月)
- 2026年 DRAM 市场预测: $618.7B (年增 303%)
- 2027年 全球内存市场预计突破 $1.28T
- HBM 在 2026 年蚕食 23% 的 DRAM 晶圆产能
- 2026年 HBM 出货量预计超过 300亿 Gb
- HBM4 在 2026 下半年成为主流

### HBM 合同价格
- 预计 2027年 HBM 合同价格将成倍增长
- AI 基础设施部署加速持续推动 HBM 需求

来源: TrendForce (2026-05-29, 2026-01-22, 2026-06-03)

---

## 4. CXL 市场关键数据

### 市场规模
- CXL 4.0 Memory Fabric Switch 市场: 2025年 $1.8B, 预计 2034年 $9.7B (CAGR 20.6%)
- CXL Memory Expansion 市场: 2025年 $1.3B, 预计 2034年 $11.8B (CAGR 28.7%)
- CXL 内存设备市场: 2025年 $1.69B, 2026年 $2.25B (CAGR ~33%)
- CXL Switch 市场 (2027年预测): $2.2B-$2.5B, Expanders ~$3.5B
- Data Center Memory 市场: 2026年 $28.8B, 预计 2031年 $66.9B (CAGR 18.36%)

来源: DataIntelo, MarketIntelo, TBRC, X/@TheValueist (2026-06-23), Mordor Intelligence

### CXL 生态关键厂商
- CXL Switch: Marvell (XConn), Astera Labs, Broadcom, Microchip
- CXL 控制器 IP: Rambus, 澜起科技 (Montage), Astera Labs
- CXL 内存模组: Samsung, SK Hynix, Micron
- PCIe Switch: Broadcom 占 85-90% 份额, Microchip/Astera 等瓜分其余
- Marvell: 领先量产 CXL 压缩芯片 (Structera X 和 Structera A)

### CXL 技术进展
- CXL 3.0: 支持 coherent memory sharing, 多主机共享内存池
- CXL 4.0: 下一代 fabric switch, 更高带宽
- CXL Type 3 模组: 2026年基本由 hyperscaler 分配

来源: OFC Conference (2026-03-17), Penguin Solutions, Cosolvic (2026-06-11)

---

## 5. 关键行业动态

### 韩国双雄投资计划
- Samsung 和 SK Hynix: 十年期 $870B 半导体产能扩展计划
- 涵盖 memory, storage, logic
- 与 OpenAI 签署意向书 (2025年10月)

### SK Hynix 市场地位
- 2026年6月: 短暂超越 Samsung 成为韩国市值最高公司
- HBM 市场持续领先

### 地缘政治
- 出口管制持续影响先进内存技术流向
- 供应链区域化重组

来源: TechPowerUp (2026-07-02), Fortune (2026-06-23)
