# References — Long-Context Challenges under PD Disaggregation (调研附件)

> 调研时间: 2026-07-15
> 调研目标: 工业界头部公司/团队关于"超长序列 LLM 推理挑战"的博客/论文/技术分享，特别关注 PD 分离视角

## 访问过的 URL 清单（含成功/失败）

| # | URL | 状态 | 用途 |
|---|-----|------|------|
| 1 | https://www.lmsys.org/blog/ | ✅ 200 | SGLANG blog index |
| 2 | https://docs.sglang.ai/ (→ docs.sglang.io) | ✅ 200 | SGLang docs 首页 |
| 3 | https://blog.vllm.ai/ (→ vllm.ai/blog) | ✅ 200 | vLLM blog index |
| 4 | https://blog.vllm.ai/blog/2026-06-12-minimax-m3-vllm | ✅ 200 | MiniMax M3 1M-token serving |
| 5 | https://github.com/MoonshotAI/Mooncake | ❌ 404 (可能仓库已删除/重命名) | Mooncake README |
| 6 | https://github.com/LMCache/LMCache | ✅ 200 | LMCache README |
| 7 | https://pytorch.org/blog/ | ✅ 200 | Meta/PyTorch blog index |
| 8 | https://pytorch.org/blog/disaggregated-inference-with-vllm/ | ❌ 404 | 推测已删除的旧 post |
| 9 | https://www.google.com/search?q=%22long+context%22+%22PD+disaggregation%22+challenge+2025+blog | ✅ | Google 搜索 |
| 10 | https://www.google.com/search?q=%22million+token+context%22+LLM+serving+inference+systems+2025 | ✅ | Google 搜索 |
| 11 | https://www.google.com/search?q=KV+cache+compression+%22disaggregated%22+%22long+context%22+paper | ✅ | Google 搜索 |
| 12 | https://www.google.com/search?q=%22split+serve%22+OR+%22distserve%22+long+context+attention+memory+challenge | ✅ | Google 搜索 |
| 13 | https://haoailab.com/distserve-retro | ❌ 404 (推测旧链接已删除) | "Disaggregated Inference: 18 Months Later" |
| 14 | https://haoailab.com/ | ✅ 200 | Hao AI Lab 首页 |
| 15 | https://haoailab.com/blogs/distserve | ✅ 200 | DistServe 原始博客 (2024.03.17) |
| 16 | https://blog.lmcache.ai/2025/04/29/bringing-state-of-the-art-pd-speed-to-vllm-v1-with-lmcache | ✅ 200 | LMCache + vLLM v1 PD 博客 |
| 17 | https://z.ai/blog/zcube | ✅ 200 | Z.ai ZCube 网络架构博客 (2026.05.20) |
| 18 | https://www.lmsys.org/blog/2025-07-20-deploying-kimi-k2-pd | ❌ 404 | 猜测路径错误 |
| 19 | https://www.lmsys.org/blog/2025-07-20-k2-large-scale-ep | ✅ 200 | Kimi K2 PD 博客 |
| 20 | https://www.lmsys.org/blog/2026-02-19-gb300-longctx | ✅ 200 | GB300 long-context 推理博客 |
| 21 | https://www.lmsys.org/blog/2025-05-05-large-scale-ep | ✅ (已知) | DeepSeek R1 PD + EP 博客 |
| 22 | https://www.lmsys.org/blog/2026-01-12-epd | ✅ (已知) | EPD 分离博客 |
| 23 | https://arxiv.org/abs/2407.00079 | ✅ 200 | Mooncake 论文 (arXiv:2407.00079) |

## 识别出的关键博客/论文/新闻报道（按团队分组）

### SGLang / LMSYS
- 2026-02-19: "Deploying DeepSeek on GB300 NVL72: Big Wins in Long-Context Inference"
  - URL: https://www.lmsys.org/blog/2026-02-19-gb300-longctx
- 2025-07-20: "Deploying Kimi K2 with PD Disaggregation and Large-Scale Expert Parallelism on 128 H200 GPUs"
  - URL: https://www.lmsys.org/blog/2025-07-20-k2-large-scale-ep
- 2025-05-05: "Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs"
  - URL: https://www.lmsys.org/blog/2025-05-05-large-scale-ep
- 2026-01-12: "EPD Disaggregation: Elastic Encoder Scaling for Vision-Language Models in SGLang"
  - URL: https://www.lmsys.org/blog/2026-01-12-epd
- 2026-06-01: "Heterogeneous CPU + GPU EPD Disaggregation to Boost VLM Serving"
  - URL: https://www.lmsys.org/blog/2026-06-01-hetero-epd

### vLLM
- 2026-06-12: "MiniMax M3 in vLLM: Day-0 Serving for 1M-Token Multimodal Reasoning"
  - URL: https://vllm.ai/blog/2026-06-12-minimax-m3-vllm
- 2025-06-30: "MiniMax-M1 Hybrid Architecture Meets vLLM: Long Context, Fast Inference"
  - URL: https://vllm.ai/blog/2025-06-30-minimax-m1
- 2026-04-24: "DeepSeek V4 in vLLM: Efficient Long-context Attention"
  - URL: https://vllm.ai/blog/2026-04-24-deepseek-v4
- 2025-09-29: "DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention"
  - URL: https://vllm.ai/blog/2025-09-29-deepseek-v32

### LMCache
- 2025-04-29: "Bringing State-Of-The-Art PD Speed to vLLM v1 with LMCache"
  - URL: https://blog.lmcache.ai/2025/04/29/bringing-state-of-the-art-pd-speed-to-vllm-v1-with-lmcache
- 2025-10: LMCache 论文 arXiv:2510.09665
  - 标题: "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference"

### Moonshot AI / Mooncake
- 2024-06 (v4 2025-09): Mooncake 论文 arXiv:2407.00079
  - 标题: "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving"

### Hao AI Lab (UCSD)
- 2024-03-17: "Throughput is Not All You Need: Maximizing Goodput in LLM Serving using Prefill-Decode Disaggregation" (DistServe 原始博客)
  - URL: https://haoailab.com/blogs/distserve
- 2025-11-03: "Disaggregated Inference: 18 Months Later"
  - URL (已被删除/迁移): https://haoailab.com/distserve-retro [404]

### Z.ai (智谱)
- 2026-05-20: "Next-generation LLM Inference Network: How ZCube Alleviates Network Bottlenecks?"
  - URL: https://z.ai/blog/zcube

### 学术文献 (通过 Google 搜索识别)
- SplitZip: "Ultra Fast Lossless KV Compression for PD Disaggregated LLM Serving" (2026)
  - URL: https://arxiv.org/html/...
- KVServe: "Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving" (2026)
  - URL: https://www.researchgate.net/...
- HACK: "Homomorphic Acceleration of Quantization" (2025)
- DistServe 论文: arXiv:2406.03117 (2024-06-06)
- PPD (Prefill-Predictive-Decode): OpenReview 文档
- DOPD: "A Dynamic PD-Disaggregation Architecture" (IEEE, 2026-02)
- CXL-SpecKV (ACM DL, 2026)
- SAC: "Disaggregated KV Cache System for Sparse Attention Models" (Semantic Scholar)
- OrbitFlow: "SLO-Aware Long-Context LLM Serving" (VLDB)
- LoongServe: "Efficiently Serving Long-Context Large Language Models" (ACM DL, 2024)
- Medha: "Efficient LLM Inference on Multi-Million Context" (arXiv, 2025-06)
- Hybe: "GPU-NPU Hybrid System for Efficient LLM Inference with Million-Token Context" (ACM DL, 2025-06)
- RocketKV: "Training-free KV cache compression strategy for long-context LLM" (OpenReview)
- ShadowKV: "KV cache in shadows for high-throughput long-context LLM inference" (2025)
- RetAttention / DuoAttention / MosaicKV 等

## 关键搜索片段（Google AI 摘要）

### 搜索: "long context" "PD disaggregation" challenge 2025 blog
Google AI 摘要原文 (越南语界面):
> Prefill-Decode (PD) disaggregation splits an AI model's prompt reading (prefill) and token generation (decode) into different processes. While it helps process millions of context tokens, the "long context" and "PD" challenge stems from **GPU memory starvation** and **massive KV cache (memory map) transfers** between servers.
>
> The main challenges in modern LLM infrastructure include:
> - **Bandwidth Bottlenecks**: Long-context prompts create massive amounts of memory state (KV Cache). Moving this cache from Prefill nodes to Decode nodes across the network creates severe traffic congestion and latency.
> - **Resource Mismatch**: Prefill is highly compute-bound, while decoding is memory-bandwidth-bound. Balancing these two workloads without idling GPUs is difficult.
> - **Head-of-Line Blocking**: Massive long-context requests take a long time to read, blocking smaller requests behind them in the queue and causing system timeouts.

### 搜索: "million token context" LLM serving
Google AI 摘要:
> Serving multi-million token contexts requires solving the **Prefill Bottleneck** — processing the massive initial prompt — and managing the **KV Cache**, which is the memory required to store the model's "attention" memory for every single token.
>
> A 1-million-token input can require **tens of gigabytes of expensive GPU memory per user**. Without proper serving infrastructure, memory runs out quickly, or the system experiences a long freeze (latency) before generating a response.

### 搜索: KV cache compression "disaggregated" "long context"
Google AI 摘要:
> KV cache compression in disaggregated long-context LLMs solves a major data bottleneck. In modern systems, generation (decoding) runs on different hardware than initial reading (prefilling). Passing huge amounts of historical data (the KV cache) between them slows everything down.

## 过程记录

1. 主线程 Playwright 导航 + 主线程 Bash 抓取（GitHub raw 被 block）
2. MoonshotAI/Mooncake GitHub 仓库已删除（404）
3. Hao AI Lab 的 distserve-retro 链接已失效 (404)，但 DistServe 原始博客 (haoailab.com/blogs/distserve) 仍在线
4. raw.githubusercontent.com 在 Playwright/Bash 两侧都 timeout —— 典型网络限制
5. Google 搜索页的 body.innerText 稳定可用（但越南语 UI），6000 字足以覆盖 AI 摘要 + 前 8-10 条链接
