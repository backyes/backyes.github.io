---
title: "DeepSeek-V4-Flash-0731 vs Kimi-K3: Full Benchmark Comparison"
date: 2026-08-01
tags: ["DeepSeek", "Kimi", "Benchmark", "Model-Comparison", "MoE", "Agent", "Coding"]
excerpt: "DeepSeek-V4-Flash-0731 vs Kimi-K3 benchmark comparison. Kimi leads on all 5 shared benchmarks, but the two models target different segments — Flash efficiency vs flagship all-rounder."
---

# DeepSeek-V4-Flash-0731 vs Kimi-K3: Full Benchmark Comparison

## Introduction

In late July 2026, two flagship Chinese models released benchmark data almost simultaneously:

- **DeepSeek-V4-Flash-0731** — DeepSeek's efficiency inference variant (speculative decoding enhanced)
- **Kimi-K3** — Moonshot's MoE flagship (2.8T total params / 114B activated)

> Data sources: [DeepSeek-V4-Flash-0731 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) · [Kimi-K3 Model Card](https://huggingface.co/moonshotai/Kimi-K3) (collected 2026-08-01)

---

## Part 1: In-Depth Comparison

### 1.1 Model Positioning & Architecture Differences

This is the essential context — **these are not same-class competitors**:

| Dimension | DeepSeek-V4-Flash-0731 | Kimi-K3 |
|---|---|---|
| **Positioning** | Efficient inference Flash variant | Flagship all-rounder |
| **Architecture** | Speculative decoding enhanced (DSpark) | MoE (1 Dense + 69 KDA + 24 Gated MLA) |
| **Total params** | Undisclosed (Flash variant, emphasizes small activation) | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">2.8T</span> |
| **Activated params** | Far smaller than Pro variant | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">104B</span> |
| **Experts** | — | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">896 (top-16 + 2 shared)</span> |
| **Context** | — | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">1M tokens</span> |
| **Multimodal** | Not reflected in benchmarks | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">Vision benchmarks included</span> |
| **Baselines** | V4-Pro Preview, GLM-5.2, Opus-4.8 | Claude Fable 5, GPT-5.6 Sol, Opus-4.8, GPT-5.5 |

**Core difference**: DeepSeek-V4-Flash is the "small and refined" efficiency variant pursuing cost-performance; Kimi-K3 is the "large and comprehensive" flagship covering reasoning/coding/agent/vision.

---

### 1.2 Shared Benchmark Head-to-Head

Both models have results on 5 shared benchmarks. ==**Kimi-K3 leads on all 5**==.

| Benchmark | DeepSeek-V4-Flash-0731 | Kimi-K3 (max) | Delta | Winner |
|---|---|---|---|---|
| Terminal Bench 2.1 | 82.7 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">88.3</span> | +5.6 | **Kimi-K3** |
| DeepSWE | 54.4 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">67.5</span> | ==+13.1 🏆== | **Kimi-K3** |
| Toolathlon-Verified | 70.3 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">76.5</span> | +6.2 | **Kimi-K3** |
| Agents' Last Exam | 25.2 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">28.3</span> | +3.1 | **Kimi-K3** |
| AutomationBench | 25.1 (Public) | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">30.8</span> | +5.7 | **Kimi-K3** |

> 🏆 = largest gap

**Gap breakdown**:

| Level | Benchmark | Delta | Interpretation |
|---|---|---|---|
| 🔴 **Large lead** | DeepSWE | **+13.1** | Kimi dominates SWE code reasoning |
| 🟡 **Medium lead** | Terminal Bench | **+5.6** | Kimi stronger at terminal operations |
| 🟡 **Medium lead** | Toolathlon | **+6.2** | Kimi better at tool use |
| 🟡 **Medium lead** | AutomationBench | **+5.7** | Kimi better at automation |
| 🟢 **Small lead** | Agents' Last Exam | +3.1 | Smallest gap, still Kimi |

---

### 1.3 Indirect Comparison via GLM-5.2 Anchor

Both models benchmarked GLM-5.2 as an indirect reference. Note that GLM-5.2 scores differ between cards (evaluation configs may vary).

| Benchmark | DeepSeek-V4-Flash | Kimi-K3 | Winner | GLM-5.2 (DeepSeek card) | GLM-5.2 (Kimi card) |
|---|---|---|---|---|---|
| Terminal Bench 2.1 | 82.7 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">88.3</span> | **Kimi +5.6** | 81.0 | 82.7 |
| DeepSWE | 54.4 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">67.5</span> | **Kimi +13.1** | 46.2 | 46.2 |
| Toolathlon-Verified | 70.3 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">76.5</span> | **Kimi +6.2** | 59.9 | 59.9 |
| Agents' Last Exam | 25.2 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">28.3</span> | **Kimi +3.1** | 23.8 | 20.4 |
| AutomationBench | 25.1 (Public) | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">30.8</span> | **Kimi +5.7** | 12.9 | 12.9 |

**Key findings**:

- In DeepSeek's card, ==Flash-0731 (82.7) roughly ties GLM-5.2 (81.0) on Terminal Bench==, while Kimi (88.3) far exceeds both
- On DeepSWE, both crush GLM-5.2 (46.2), but Kimi (67.5) beats DeepSeek (54.4) by ==13.1 points==
- Agents' Last Exam has inconsistent GLM-5.2 scores across cards (23.8 vs 20.4)

---

### 1.4 DeepSeek-V4-Flash-0731 Exclusive Benchmarks

DeepSeek evaluated additional Code Agent benchmarks:

| Benchmark | Flash-0731 | Flash (Preview) | V4-Pro (Preview) | GLM-5.2 | Opus-4.8 | Flash vs Opus |
|---|---|---|---|---|---|---|
| NL2Repo | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">54.2</span> | 39.4 | 38.5 | 48.9 | 69.7 | **−15.5** |
| Cybergym | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">76.7</span> | 38.7 | 52.7 | — | 83.1 | **−6.4** |
| DSBench-FullStack † | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">68.7</span> | 37.0 | 41.8 | 61.8 | 71.6 | **−2.9** |
| DSBench-Hard † | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">59.6</span> | 25.8 | 31.1 | 54.5 | 71.7 | **−12.1** |

> Flash-0731 leads its own Preview on all 4 exclusive benchmarks, but trails Opus-4.8 on all of them

---

### 1.5 Kimi-K3 Exclusive Benchmark Highlights

Kimi-K3 covers far more benchmarks across reasoning, coding, agent, and vision.

### Reasoning & Knowledge

> **Kimi strengths**: AA-LCR · **Kimi weaknesses**: CritPt (gap 8.9)

| Benchmark | Kimi-K3 | Top Competitor | Competitor Score | Result |
|---|---|---|---|---|
| GPQA Diamond | 93.5 | GPT-5.6 Sol | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">94.1</span> | 🥈 Lost by 0.6 |
| CritPt | 23.4 | GPT-5.6 Sol | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">32.3</span> | 🥉 Gap 8.9 |
| AA-LCR | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">74.7</span> | Kimi wins | — | 🏆 **Champion** |
| HLE-Full | 43.5 / 56.0 | Claude Fable 5 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">53.3 / 63.0</span> | 🥈 Lost by 9.8 |

### Coding

> **Kimi strengths**: ProgramBench / SWE-Marathon · **Kimi weaknesses**: DeepSWE / FrontierSWE

| Benchmark | Kimi-K3 | Top Competitor | Competitor Score | Result |
|---|---|---|---|---|
| DeepSWE | 67.5 | GPT-5.6 Sol | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">73.0</span> | 🥈 Gap 5.5 |
| ProgramBench | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">77.8</span> | Kimi wins | — | 🏆 **Champion** |
| Terminal-Bench 2.1 | 88.3 | GPT-5.6 Sol | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">88.8</span> | 🥈 Lost by 0.5 |
| FrontierSWE | 81.2 | Claude Fable 5 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">86.6</span> | 🥈 Gap 5.4 |
| SWE-Marathon | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">42.0</span> | Kimi wins | — | 🏆 **Champion** (GPT-5.5 only 14.0) |

### Agent

> **Kimi strengths**: BrowseComp / MCPMark / DeepSearchQA / Harvey · **Kimi weaknesses**: GDPval

| Benchmark | Kimi-K3 | Top Competitor | Competitor Score | Result |
|---|---|---|---|---|
| BrowseComp | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">91.2</span> | Kimi wins | — | 🏆 **Champion** |
| MCPMark-Verified | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">94.5</span> | Kimi wins | — | 🏆 **Champion** |
| DeepSearchQA F1 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">95.0</span> | Kimi wins | — | 🏆 **Champion** |
| Harvey Lab-AA | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">94.6</span> | Kimi wins | — | 🏆 **Champion** |
| GDPval-AA v2 (Elo) | 1686 | Claude Fable 5 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">1747</span> | 🥈 Lost by 61 |

### Vision

> **Kimi strengths**: OmniDocBench / Video-MME / MMVU · **Kimi weaknesses**: WorldVQA / BabyVision

| Benchmark | Kimi-K3 | Top Competitor | Competitor Score | Result |
|---|---|---|---|---|
| OmniDocBench | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">91.1</span> | Kimi wins | — | 🏆 **Champion** |
| Video-MME (w. sub) | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">90.0</span> | Kimi wins | — | 🏆 **Champion** |
| MathVision | 94.3 / 97.8 | GPT-5.6 Sol | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">95.8</span> / 97.8 | 🥈 Lost by 1.5 |
| MMVU | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">82.1</span> | Kimi wins | — | 🏆 **Champion** |
| WorldVQA ForceAnswer | 51.0 | Claude Fable 5 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">56.7</span> | 🥈 Gap 5.7 |
| BabyVision w/ python | 85.7 | Claude Fable 5 | <span style="background:#e8f5e9;font-weight:700;padding:2px 6px;border-radius:3px;">90.5</span> | 🥈 Gap 4.8 |

---

### 1.6 Evaluation Methodology Differences (Important)

> ⚠️ Evaluation configs may differ between models — direct comparison requires caution.

| Dimension | DeepSeek-V4-Flash-0731 | Kimi-K3 |
|---|---|---|
| **Reasoning effort** | max (temperature=1.0, top_p=0.95) | max |
| **Agent framework** | DeepSeek Harness minimal mode | Undisclosed |
| **AutomationBench version** | Public subset | Full version |
| **Benchmark count** | 9 | 40+ |

---

### 1.7 Summary

| Dimension | Conclusion | Evidence |
|---|---|---|
| **Shared benchmarks** | Kimi-K3 leads 5/5 | DeepSWE +13.1 · Terminal +5.6 · Toolathlon +6.2 |
| **Model scale** | Asymmetric comparison | Kimi 2.8T/104B vs Flash undisclosed (small activation) |
| **Cost-performance** | ==Flash punches above its weight== | Flash Terminal Bench (82.7) ≈ GLM-5.2 (81.0) |
| **Versatility** | Kimi covers reasoning+coding+agent+vision | 40+ benchmarks, 15+ 🏆 champions |
| **Coding** | Kimi leads by largest margin | DeepSWE 67.5 vs 54.4 (**+13.1**) |
| **Agent** | Kimi stronger at tool use | Toolathlon +6.2 · AutomationBench +5.7 |

**One-liner**: Kimi-K3 is today's most versatile multimodal flagship; DeepSeek-V4-Flash-0731 proves that ==**DSpark speculative decoding can approach flagship-level Code Agent capability with minimal activation**==.

---

## Part 2: Raw Benchmark Data

Below are the unprocessed benchmark figures from each model's HuggingFace card.

### 2.1 DeepSeek-V4-Flash-0731

**Positioning**: Efficient inference Flash variant with speculative decoding module (DeepSeek-V4-Flash-DSpark)

**Source**: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731

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

> † DeepSeek internal benchmark

---

### 2.2 Kimi-K3

**Positioning**: MoE flagship model

**Architecture Specs**:

| Spec | Value |
|---|---|
| Total params | 2.8T |
| Activated params | 104B |
| Layers | 93 (1 Dense + 69 KDA + 24 Gated MLA) |
| Experts | 896 (top-16 + 2 shared) |
| Context length | 1M tokens |

**Source**: https://huggingface.co/moonshotai/Kimi-K3

### Reasoning & Knowledge

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max) | GPT-5.6 Sol (max) | Claude Opus 4.8 (max) | GPT-5.5 (xhigh) | GLM-5.2 (max) |
|---|---|---|---|---|---|---|
| GPQA Diamond | 93.5 | 92.6 | 94.1 | 91.0 | 93.5 | 91.2 |
| CritPt | 23.4 | 28.6 | 32.3 | 20.9 | 27.1 | 20.9 |
| AA-LCR | 74.7 | 70.0 | 73.7 | 67.7 | 74.3 | 71.3 |
| HLE-Full | 43.5 / 56.0 | 53.3 / 63.0 | 44.5 / 58.0 | 49.8 / 57.9 | 41.4 / 52.2 | — |

### Coding

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

### Agentic

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

### Vision

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

## Data Sources

| Source | URL | Collected |
|---|---|---|
| DeepSeek-V4-Flash-0731 Model Card | [https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 2026-08-01 |
| Kimi-K3 Model Card | [https://huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 2026-08-01 |

---

> ⚠️ **Disclaimer**: All data is self-reported from the models' HuggingFace cards. Evaluation configs may differ (agent framework, temperature, benchmark subsets). Interpret comparisons with caution. Click the links above for full model card details.
