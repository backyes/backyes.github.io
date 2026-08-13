# vLLM vs SGLang 第三方评价调研 - 关键洞察

## A. Benchmark 与性能对比（已采集）

### 核心数字（H100 80GB, Llama 3.1 8B 或 3.3 70B）

| 来源 | 日期 | 模型/Setup | SGLang tok/s | vLLM tok/s | 差异 |
|------|------|-----------|-------------|-----------|------|
| PremAI/LocalAIMaster | 2026-Q1 | Llama 3.1 8B, 单卡 | ~16,200 | ~12,500 | SGLang +29% |
| AIMultiple | 2026-04 | Llama 3.1 8B, 10K ops | 16,215 | 12,553 | SGLang +29% |
| Spheron 3-engine | 2026-03 | Llama 3.3 70B FP8, c=50 | 1,920 | 1,850 | ≈持平 (独特prompt) |
| Spheron vLLM vs SGLang | 2026-06 | Llama 3.3 70B, c=100 (prefix-heavy) | 2,460 | 2,400 | ≈持平 |
| Jarvis Labs (ITL) | 2026-05 | H100, decode latency | ~61-63ms (SGLang) | ~70ms (vLLM) | SGLang低~10% |

### 关键定性洞察

1. **GMI (Google AI Overview)**: 在多轮/RAG中 SGLang 吞吐高出 15-30%，First-Token Latency 低达 40%。但 vLLM 在独特prompt + 极限并发下延迟更稳定。

2. **Particula (Sebastian Mondragon)**: "The 29% throughput gap... can shrink to nearly zero on unique-prompt batch jobs, or balloon to 6x on prefix-heavy RAG pipelines." —— 工作负载形态决定胜负。

3. **AIMultiple 最深洞察**: "Even when vLLM is optimized with the exact same kernels (FlashInfer) as SGLang, it still significantly trails... the bottleneck is no longer the mathematical kernel, but the engine's internal orchestration overhead." 瓶颈不在 kernel，在引擎编排开销。

4. **Runpod benchmark**: 多轮对话高并发下，SGLang 稳定 ~30-31 tok/s/req，vLLM 从 22 跌到 16 tok/s。

5. **Spheron 量化**: 前缀重叠率须 >60% 才有 SGLang 显著收益；独特 prompt 下两引擎 <5% 差异在误差范围。

6. **Jarvis Labs (May 2026)** - 最系统三方对比:
   - TPOT/ITL: SGLang 61-63ms vs vLLM ~70ms vs TRT-LLM 更高
   - 使用 dense + MoE 模型, ShareGPT + RULER 16K datasets

7. **Techsy (Jul 14)** - 最新（2026 年 7 月）:
   - 70B scale: delta 仅 3-5%
   - 8B scale: SGLang +29%
   - TGI 已于 2025-12 进入 maintenance mode

8. **Inference Engineering**: 最新(2026-06) Interactive benchmark on DeepSeek-R1-Distill-Llama-8B, 2xH100, TP=2
   - medium_chat c=100: SGLang 4,587 TPS vs vLLM 4,432 TPS
   - prefix_cache_warm 场景数据最有价值

9. **Medium (Sebastian Buzdugan, 6天前)**: "SGLang is 29% faster than vLLM on one benchmark and dead even on another."

10. **DeepSeek V4 officially endorsing SGLang** (来源: Particula)

### 共识结论 (A维度)
- **SGLang 在 prefix-heavy / 多轮 / RAG / 结构化输出场景领先 15%-6x**
- **vLLM 在独特 prompt _batch + 极限并发稳定性 + 硬件广度上占优**
- **并非 SGLang always wins** —— 胜率和幅度完全依赖 prefix overlap ratio

---

## B. 行业与媒体
(待采集)

## C. 学术
(待采集)

## D. 社交媒体/社区
(待采集)
