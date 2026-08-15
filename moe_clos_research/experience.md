# MoE × Sparse CLOS 调研经验教训

> 完成日期：2026-07-12
> 调研类型：学术论文 + 系统研究
> 项目目录：moe_clos_research/

## 本次任务总结

聚焦"MoE 稀疏性带来的 CLOS 网络效率问题"这一细分方向，深入调研了 Rail-only、Spine-free、MixNet、Opus、Switchless Topologies、UBEP 等 2024-2026 年最新研究成果。核心发现：**MoE 的稀疏激活模式使得传统 CLOS 网络的全二分带宽成为过度配置，削减 spine 层/改用 switchless 拓扑可节省 20-77% 网络成本且性能损失极小。**

## 经验教训与改进

### 1. 关键词精确定位节省大量 Token ✅
- **好做法**：从上一个调研（sparse Clos 作为数据中心网络拓扑）精确过渡到本任务的 MoE 稀疏推理场景，没有走弯路
- 7 轮 Playwright 搜索即定位到 8 篇核心论文，效率远高于派发并行 Agent
- **节省**：相比上一个 32-agent 调研（~100K tokens），本次仅用 ~20K tokens 完成搜索阶段

### 2. Playwright 直接读 arXiv 抽象页 ✅ → ✅✅
- **arXiv 抽象页（/abs/）** 比 HTML 全文页（/html/）节省 5-10× token，包含足够的关键信息（title + abstract + authors）
- 核心论文只读抽象页，用 `browser_evaluate` 提取摘要，不读全文
- 报告需要的深度信息（定量数据、方法、结论）通过摘要已足够

### 3. 论文正文的提取技巧 ✅
- 仅对最关键的论文（Rethinking Topologies for MoE Serving）浏览了 HTML 全文版，提取 figure captions、section headings、关键结果段落
- 其他 7 篇仅提取 abstract + metadata
- 论文 PDF 全部通过 curl 下载到本地（不经过模型），供后续人工阅读

### 4. 并行下载节省时间 ✅
- 7 篇论文 PDF 通过 Shell 脚本并行 curl 下载，总耗时 <30 秒
- 如果逐篇通过模型读取 PDF 会消耗 ~50-100K tokens 且速度慢

### 5. 本次调研的核心结论提炼 ✅
- **学术共识正在形成**：多篇独立论文（MIT, UC Berkeley, HKUST, Microsoft, 华为）从不同角度指向同一结论——MoE 的稀疏通信不需要传统 CLOS 网络
- **收益量化**：网络成本节省 20-77%，网络功耗降低 23-75%，性能损失仅 0-5.6%
- 细分方向可分为：拓扑简化（Rail-only, Switchless）、可重构光网络（MixNet, Opus）、软件通信优化（UBEP, Speculative MoE, Semantic Parallelism）、分析框架（Switching Efficiency）

## 改进清单
| 项 | 操作 |
|----|------|
| 论文下载 | 全部用 curl 并行下载到本地，不通过模型 |
| 摘要提取 | 只读 arXiv /abs/ 页面，不读全文 |
| 关键词定位 | 从上一个调研精确过渡，不做无用搜索 |
| 搜索结果页 | Google Scholar 有反爬，用 Playwright 截图即可，不用 curl 存 HTML |
