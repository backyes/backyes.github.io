---
title: "DeepSeek-V4-Flash-0731 vs Kimi-K3: Benchmark 全景对比"
date: 2026-08-01
tags: ["DeepSeek", "Kimi", "Benchmark", "Model-Comparison", "MoE", "Agent", "Coding"]
excerpt: "DeepSeek-V4-Flash-0731 与 Kimi-K3 的 benchmark 全景对比分析。原始数据直呈 + 深度对比：5 项共有 benchmark Kimi 全面领先，但两者定位截然不同 — Flash 高效推理 vs 旗舰全能模型。"
---

# DeepSeek-V4-Flash-0731 vs Kimi-K3: Benchmark 全景对比

## 前言

2026 年 7 月底，两个来自中国的旗舰模型几乎同期发布 benchmark 数据：

- **DeepSeek-V4-Flash-0731** — DeepSeek 家族的高效推理版本（推测解码增强）
- **Kimi-K3** — Moonshot 的 MoE 旗舰（2.8T 总参数 / 104B 激活）

本文分两大部分：
1. **原始数据直呈** — 不加工、不解读，直接展示两模型 HuggingFace 卡片上的 benchmark 数据
2. **深度对比分析** — 共有 benchmark 直接对比、间接锚点对比、模型定位差异、评测方法论差异

> 数据来源: [DeepSeek-V4-Flash-0731 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) · [Kimi-K3 Model Card](https://huggingface.co/moonshotai/Kimi-K3) (2026-08-01 采集)

---

## Part 1: 原始数据直呈

### 1.1 DeepSeek-V4-Flash-0731

**模型定位**: 高效推理 Flash 版本，配备推测解码模块 (DeepSeek-V4-Flash-DSpark)

**数据来源**: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731

| Benchmark | DeepSeek-V4-Flash-0731 | DeepSeek-V4-Flash (Preview) | DeepSeek-V4-Pro (Preview) | GLM-5.2 | Opus-4.8 |
|---|---|---|---|---|---|
| Terminal Bench 2.1 | 82.7 | 61.8 | 72.1 | 81.0 | 85.0 |
| NL2Repo | 54.2 | 39.4 | 38.5 | 48.9 | 69.7 |
| Cybergym | 76.7 | 38.7 | 52.7 | - | 83.1 |
| DeepSWE | 54.4 | 7.3 | 12.8 | 46.2 | 58.0 |
| Toolathlon-Verified | 70.3 | 49.7 | 55.9 | 59.9 | 76.2 |
| Agents' Last Exam | 25.2 | 15.8 | 16.5 | 23.8 | 25.7 |
| AutomationBench Public | 25.1 | 10.8 | 12.8 | 12.9 | 27.2 |
| DSBench-FullStack † | 68.7 | 37.0 | 41.8 | 61.8 | 71.6 |
| DSBench-Hard † | 59.6 | 25.8 | 31.1 | 54.5 | 71.7 |

> † 表示 DeepSeek 自建评测集

---

### 1.2 Kimi-K3

**模型定位**: MoE 旗舰模型

**架构规格**:

| 规格项 | 数值 |
|---|---|
| 总参数 | 2.8T |
| 激活参数 | 104B |
| 层数 | 93 (1 Dense + 69 KDA + 24 Gated MLA) |
| 专家数 | 896 (top-16 + 2 shared) |
| 上下文长度 | 1M tokens |

**数据来源**: https://huggingface.co/moonshotai/Kimi-K3

#### Reasoning & Knowledge

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|---|---|---|---|---|---|---|
| GPQA Diamond | 93.5 | 92.6 | 94.1 | 91.0 | 93.5 | 91.2 |
| CritPt | 23.4 | 28.6 | 32.3 | 20.9 | 27.1 | 20.9 |
| AA-LCR | 74.7 | 70.0 | 73.7 | 67.7 | 74.3 | 71.3 |
| HLE-Full | 43.5 / 56.0 | 53.3 / 63.0 | 44.5 / 58.0 | 49.8 / 57.9 | 41.4 / 52.2 | — |

#### Coding

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|---|---|---|---|---|---|---|
| DeepSWE | 67.5 | 70.0 | 73.0 | 59.0 | 67.0 | 46.2 |
| ProgramBench | 77.8 | 76.8 | 77.6 | 71.9 | 70.8 | 63.7 |
| Terminal-Bench 2.1 | 88.3 | 88.0 | 88.8 | 84.6 | 83.4 | 82.7 |
| FrontierSWE | 81.2 | 86.6 | 71.3 | 66.7 | 64.9 | 67.3 |
| SWE-Marathon | 42.0 | 35.0 | 39.0 | 40.0 | 14.0 | 13.0 |
| PostTrainBench | 36.6 | 41.4 | 34.6 | 34.1 | 28.4 | 34.3 |
| MLS-Bench-Lite | 48.3 | 49.9 | 46.2 | 42.8 | 35.5 | 40.4 |
| SciCode | 58.7 | 60.2 | 56.1 | 53.5 | 56.1 | 50.5 |
| Kimi Code Bench 2.0 | 72.9 | 76.9 | 64.8 | 71.7 | 69.0 | 64.2 |

#### Agentic

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|---|---|---|---|---|---|---|
| BrowseComp | 91.2 | 88.0 | 90.4 | 84.3 | 84.4 | — |
| DeepSearchQA (F1) | 95.0 | 94.2 | — | 93.1 | — | — |
| ResearchRubrics | 76.2 | — | 73.8 | 73.5 | 64.0 | 71.1 |
| GDPval-AA v2 (Elo) | 1686 | 1747 | 1736 | 1593 | 1491 | 1510 |
| Toolathlon-Verified | 76.5 | 77.9 | 74.9 | 76.2 | 73.5 | 59.9 |
| MCPMark-Verified | 94.5 | 87.4 | 92.9 | 76.4 | 92.9 | — |
| MCP-Atlas | 84.2 | 84.7 | 83.6 | 83.6 | 82.8 | 82.6 |
| AutomationBench | 30.8 | 29.1 | 29.7 | 27.2 | 22.7 | 12.9 |
| JobBench | 54.3 | 57.4 | 45.4 | 48.4 | 38.3 | 43.4 |
| AA-Briefcase (Elo) | 1548 | 1583 | 1495 | 1354 | 1158 | 1260 |
| Agents' Last Exam | 28.3 | 25.7† | 29.6 | 27.0 | 26.6 | 20.4 |
| APEX-Agents | 41.0 | 43.3 | 39.9 | 39.4 | 38.5 | 35.6 |
| OfficeQA Pro | 63.3 | 69.9 | 63.2 | 63.9 | 60.9 | 41.4 |
| SpreadsheetBench 2 | 34.8 | 34.7 | 32.4 | 31.6 | 29.1 | 28.1 |
| OSWorld-Verified | 84.8 | 85.0 | 83.0 | 83.4 | 79.0 | — |
| OSWorld 2.0 | 58.3 | 66.1 | 62.6 | 55.7 | 49.5 | — |
| SaaS-Bench | 60.1 | — | 61.4 | 56.1 | 43.8 | — |
| τ³-Banking | 33.4 | 26.8 | 33.0 | 27.6 | 31.3 | 26.8 |
| Harvey Lab-AA | 94.6 | 93.6 | 87.2 | 91.1 | 86.3 | 91.0 |
| CorpFin v2 | 71.6 | 71.8 | 64.4 | 66.7 | 68.4 | 66.1 |
| Finance Agent v2 | 54.4 | 56.3 | 53.8 | 53.9 | 51.8 | 49.7 |
| Legal Research Bench | 44.2 | 49.5 | 48.1 | 43.8 | 40.4 | 31.3 |

#### Vision

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|---|---|---|---|---|---|---|
| WorldVQA ForceAnswer | 51.0 | 56.7 | 41.8 | 39.1 | 38.5 | — |
| OmniDocBench | 91.1 | 89.8 | 85.8 | 87.9 | 89.4 | — |
| PerceptionBench | 58.5 | 57.2 | 59.7 | 47.2 | 55.8 | — |
| Video-MME (w. sub) | 90.0 | — | 89.5 | 86.0 | 89.3 | — |
| MMVU | 82.1 | — | 81.2 | 79.2 | 81.7 | — |
| BabyVision w/ python | 85.7 | 90.5 | 88.9 | 81.2 | 83.6 | — |
| MMMU-Pro | 81.6 / 83.4 | 81.2 / 86.5 | 83.0 / 84.6 | 78.9 / 82.7 | 81.2 / 83.2 | — |
| CharXiv (RQ) | 84.8 / 91.3 | 88.9 / 93.5 | 84.6 / 89.1 | 80.5 / 89.9 | 84.1 / 89.0 | — |
| MathVision | 94.3 / 97.8 | 94.8 / 98.6 | 95.8 / 97.8 | 86.7 / 97.1 | 92.2 / 96.8 | — |
| ZeroBench (pass@5) | 23.0 / 41.0 | 23.0 / 46.0 | 17.0 / 35.0 | 17.0 / 34.0 | 22.0 / 41.0 | — |

---

## Part 2: 深度对比分析

### 2.1 共有 Benchmark 直接对比

两模型在 5 个 benchmark 上有直接可比的评测结果。==**Kimi-K3 在全部 5 项上领先 DeepSeek-V4-Flash-0731"**==。

| Benchmark | DeepSeek-V4-Flash-0731 | Kimi-K3 (max) | 差值 | 胜出方 |
|---|---|---|---|---|
| Terminal Bench 2.1 | 82.7 | $88.3$ | ==+5.6== | **Kimi-K3** |
| DeepSWE | 54.4 | $67.5$ | ==+13.1 🏆** | **Kimi-K3** |
| Toolathlon-Verified | 70.3 | $76.5$ | ==+6.2== | **Kimi-K3** |
| Agents' Last Exam | 25.2 | $28.3$ | +3.1 | **Kimi-K3** |
| AutomationBench | 25.1 (Public) | $30.8$ | ==+5.7== | **Kimi-K3** |

> 🏆 = 差距最大的 benchmark

**差距分层**:

| 差距级别 | Benchmark | 差值 | 解读 |
|---|---|---|---|
| 🔴 **大幅领先** | DeepSWE | **+13.1** | Kimi 在 SWE 代码推理上优势最为显著 |
| 🟡 **中等领先** | Terminal Bench | **+5.6** | 终端操作 Kimi 明显更强 |
| 🟡 **中等领先** | Toolathlon-Verified | **+6.2** | 工具使用 Kimi 占优 |
| 🟡 **中等领先** | AutomationBench | **+5.7** | 自动化任务 Kimi 占优 |
| 🟢 **小幅领先** | Agents' Last Exam | +3.1 | 最小差距，但仍 Kimi 领先 |

---

### 2.2 通过 GLM-5.2 的间接对比锚点

两模型都评测了 GLM-5.2，可作为间接性能参照。注意两模型卡片中 GLM-5.2 的评测数值不完全一致（评测配置可能不同），仅供参考。

| Benchmark | DeepSeek-V4-Flash | Kimi-K3 | ==胜出== | GLM-5.2 锚点 (DeepSeek表) | GLM-5.2 锚点 (Kimi表) |
|---|---|---|---|---|---|
| Terminal Bench 2.1 | 82.7 | $88.3$ | **Kimi +5.6** | 81.0 | 82.7 |
| DeepSWE | 54.4 | $67.5$ | **Kimi +13.1** | 46.2 | 46.2 |
| Toolathlon-Verified | 70.3 | $76.5$ | **Kimi +6.2** | 59.9 | 59.9 |
| Agents' Last Exam | 25.2 | $28.3$ | **Kimi +3.1** | 23.8 | 20.4 |
| AutomationBench | 25.1 (Public) | $30.8$ | **Kimi +5.7** | 12.9 | 12.9 |

**关键发现**:

- 在 DeepSeek 评测表中，==DeepSeek-V4-Flash-0731 在 Terminal Bench 2.1 ($82.7$) 上几乎与 GLM-5.2 ($81.0$) 持平==，而 Kimi-K3 ($88.3$) 远超两者
- 在 DeepSWE 上，两模型均远超 GLM-5.2 ($46.2$)，但 Kimi ($67.5$) 比 DeepSeek ($54.4$) 高出 ==13.1 分==
- Agents' Last Exam 存在数据不一致：DeepSeek 表 GLM-5.2 = 23.8，Kimi 表 GLM-5.2 = 20.4，可能评测配置不同

---

### 2.3 模型定位与架构差异

这是理解 benchmark 差异的核心背景 — **两者并非同量级竞品**:

| 维度 | DeepSeek-V4-Flash-0731 | Kimi-K3 |
|---|---|---|
| **定位** | 高效推理 Flash 版本 | 旗舰全能模型 |
| **架构** | 推测解码增强 (DSpark) | MoE (1 Dense + 69 KDA + 24 Gated MLA) |
| **总参数** | 未公开（Flash 版本，强调小激活） | 2.8T |
| **激活参数** | 远小于 Pro 版本 | 104B |
| **专家数** | — | 896 (top-16 + 2 shared) |
| **上下文** | — | 1M tokens |
| **多模态** | 未在 benchmark 中体现 | 支持视觉 (Vision 类 benchmark) |
| **对比基线** | V4-Pro Preview, GLM-5.2, Opus-4.8 | Claude Fable 5, GPT-5.6 Sol, Opus-4.8, GPT-5.5 |

**核心差异**: DeepSeek-V4-Flash 是 "小而精" 的高效版，追求性价比；Kimi-K3 是 "大而全" 的旗舰版，覆盖推理/编码/Agent/视觉全场景。

---

### 2.4 DeepSeek-V4-Flash-0731 独有 Benchmark 分析

DeepSeek 在 Code Agent 任务上评测了更多独有 benchmark:

| Benchmark | Flash-0731 | Flash (Preview) | V4-Pro (Preview) | GLM-5.2 | Opus-4.8 | ==Flash-0731 vs Opus== |
|---|---|---|---|---|---|---|
| NL2Repo | $54.2$ | 39.4 | 38.5 | 48.9 | $69.7$ | **−15.5** |
| Cybergym | $76.7$ | 38.7 | 52.7 | — | $83.1$ | **−6.4** |
| DSBench-FullStack † | $68.7$ | 37.0 | 41.8 | 61.8 | $71.6$ | **−2.9** |
| DSBench-Hard † | $59.6$ | 25.8 | 31.1 | 54.5 | $71.7$ | **−12.1** |

**关键发现**:

- **Flash-0731 相对 Preview 版本的飞跃**: NL2Repo +14.8, Cybergym +38.0, DeepSWE +37.1 — 推测解码模块 (DSpark) 带来的提升非常显著
- **但仍逊于 Opus-4.8**: 在 NL2Repo (54.2 vs 69.7)、Cybergym (76.7 vs 83.1)、DSBench (68.7 vs 71.6 / 59.6 vs 71.7) 上，Flash-0731 仍落后于 Opus-4.8
- ==**DSBench-Hard 差距最大**: Flash-0731 (59.6) vs Opus-4.8 (71.7)，差距 **12.1 分**==

---

### 2.5 Kimi-K3 独有 Benchmark 亮点

Kimi-K3 的评测覆盖远超 DeepSeek，涵盖推理、编码、Agent、视觉四大类。以下是 Kimi-K3 表现最突出的领域:

#### 推理 & 知识

| Benchmark | Kimi-K3 | 最强对手 | 对手得分 | 结果 |
|---|---|---|---|---|
| GPQA Diamond | $93.5$ | GPT-5.6 Sol | 94.1 | 🥈 惜败 0.6 |
| CritPt | 23.4 | GPT-5.6 Sol | $32.3$ | 🥉 差距 8.9 |
| AA-LCR | $74.7$ | Kimi-K3 胜出 | — | 🏆 **冠军** |
| HLE-Full | 43.5 / 56.0 | Claude Fable 5 | $53.3 / 63.0$ | 🥈 惜败 9.8 |

#### 编码

| Benchmark | Kimi-K3 | 最强对手 | 对手得分 | 结果 |
|---|---|---|---|---|
| DeepSWE | $67.5$ | GPT-5.6 Sol | 73.0 | 🥈 差距 5.5 |
| ProgramBench | $77.8$ | Kimi-K3 胜出 | — | 🏆 **冠军** |
| Terminal-Bench 2.1 | $88.3$ | GPT-5.6 Sol | 88.8 | 🥈 惜败 0.5 |
| FrontierSWE | $81.2$ | Claude Fable 5 | 86.6 | 🥈 差距 5.4 |
| SWE-Marathon | $42.0$ | Kimi-K3 胜出 | — | 🏆 **冠军** (GPT-5.5 仅 14.0) |

#### Agent

| Benchmark | Kimi-K3 | 最强对手 | 对手得分 | 结果 |
|---|---|---|---|---|
| BrowseComp | $91.2$ | Kimi-K3 胜出 | — | 🏆 **冠军** |
| MCPMark-Verified | $94.5$ | Kimi-K3 / GPT-5.6 Sol | 92.9 | 🏆 **冠军** |
| DeepSearchQA F1 | $95.0$ | Claude Fable 5 | 94.2 | 🏆 **冠军** |
| Harvey Lab-AA | $94.6$ | Kimi-K3 胜出 | — | 🏆 **冠军** |
| GDPval-AA v2 (Elo) | 1686 | Claude Fable 5 | $1747$ | 🥈 惜败 61 |

#### 视觉

| Benchmark | Kimi-K3 | 最强对手 | 对手得分 | 结果 |
|---|---|---|---|---|
| OmniDocBench | $91.1$ | Kimi-K3 胜出 | — | 🏆 **冠军** |
| Video-MME (w. sub) | $90.0$ | GPT-5.6 Sol | 89.5 | 🏆 **冠军** |
| MathVision | $94.3 / 97.8$ | GPT-5.6 Sol | 95.8 / 97.8 | 🥈 惜败 1.5 |
| MMVU | $82.1$ | GPT-5.5 | 81.7 | 🏆 **冠军** |

---

### 2.6 评测方法论差异（重要）

> ⚠️ 两模型评测配置可能不一致，直接对比需谨慎。

| 维度 | DeepSeek-V4-Flash-0731 | Kimi-K3 |
|---|---|---|
| **推理 effort** | max (temperature=1.0, top_p=0.95) | max |
| **Agent 框架** | DeepSeek Harness 最小模式 | 未公开具体细节 |
| **AutomationBench 版本** | Public 子集 | 完整版 |
| **评测集数量** | 9 项 | 40+ 项 |

**注意**: AutomationBench 的 DeepSeek 标注为 "Public" 子集，而 Kimi 为完整版，这可能导致两者在该项上的差距被高估。

---

### 2.7 总结

| 维度 | 结论 | 数据支撑 |
|---|---|---|
| **共有 benchmark** | Kimi-K3 5/5 全面领先 | DeepSWE +13.1 · Terminal +5.6 · Toolathlon +6.2 |
| **模型量级** | 非对称对比 | Kimi 2.8T/104B vs Flash 未公开（强调小激活） |
| **性价比** | ==Flash 以小博大== | Flash 在 Terminal Bench (82.7) ≈ GLM-5.2 (81.0) |
| **全能性** | Kimi 覆盖推理+编码+Agent+视觉 | 40+ benchmark，15+ 项 🏆 冠军 |
| **代码能力** | Kimi 优势最大 | DeepSWE $67.5$ vs $54.4$ (**+13.1**) |
| **Agent 能力** | Kimi 工具使用更强 | Toolathlon +6.2 · AutomationBench +5.7 |

**一句话**: Kimi-K3 是当今最全面的多模态旗舰之一；DeepSeek-V4-Flash-0731 则证明了 ==**DSpark 推测解码模块能以极小激活量逼近旗舰水平的 Code Agent 能力**==。

---

## 数据来源

| 来源 | URL | 采集时间 |
|---|---|---|
| DeepSeek-V4-Flash-0731 Model Card | [https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 2026-08-01 |
| Kimi-K3 Model Card | [https://huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 2026-08-01 |

---

> ⚠️ **免责声明**: 本文数据直接来自模型发布方 HuggingFace 卡片的 self-reported benchmark。两模型评测配置可能存在差异（Agent 框架、temperature、评测子集等），对比结论需结合具体评测条件审慎参考。建议读者点击上方链接查看原始 Model Card 获取完整评测细节。
