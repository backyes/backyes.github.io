# MTP算法原理深度解读 — 调研更新笔记 (2026-07-07)

## 本次调研新增内容

### 1. 数学基础梳理
- 从Meta FAIR论文 (arXiv:2404.19737) 提取了完整的MTP数学公式
- 从DeepSeek-V3论文 (arXiv:2412.19437) 提取了序列化MTP的完整推导
- 对比了两大范式的数学差异

### 2. 最新学术进展 (2025-2026)
- "How Transformers Learn to Plan via MTP" — 理论证明MTP改善推理
- "Your LLM Knows the Future" — 发现NTP训练的LLM已具备隐式MTP能力
- "L-MTP: Leap Multi-Token Prediction" — 跳跃式MTP (NeurIPS 2025)
- "Multi-token prediction needs registers" — MTP内部机制 (NeurIPS 2025)
- Google MTP Retrofit — 冻结模型注入MTP (2026.06)
- MTP自我蒸馏 — 3x加速

### 3. 行业采用全景
- DeepSeek-V3/V4: 序列化MTP (D=2)
- GLM-5: DeepSeek风格MTP
- Gemma 4: MTP推测解码
- 华为openPangu: 3-head MTP

### 4. DSpark深度分析
- 半自回归生成 (DFlash并行 + Markov序列头)
- 置信度调度验证 (Sequential Temperature Scaling)
- 硬件感知前缀调度器
- 性能: 60-85% per-user加速, 51-400% throughput提升

### 5. 关键洞察
- MTP已成为LLM训练的事实标准
- 训练-推理统一架构是核心趋势
- 从memory-bound到compute-bound的转变影响硬件设计
- MTP在不同规模模型上的效果差异显著

## 主要来源
- Meta FAIR: Better & Faster LLMs via Multi-token Prediction (arXiv:2404.19737)
- DeepSeek-V3 Technical Report (arXiv:2412.19437)
- DSpark: Confidence-Scheduled Speculative Decoding (alphaXiv:2026.dspark)
- X/Twitter 全球讨论 (via Google cache + Playwright)
- Google Scholar 最新论文检索