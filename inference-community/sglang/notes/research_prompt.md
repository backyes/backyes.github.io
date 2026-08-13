# SGLang 社区调研 Prompt（2025.7 至今）

## 调研目标
为推理系统架构师和芯片设计者提供 SGLang 近1年关键演进的系统性洞察。

## 核心研究问题
1. **设计哲学**：SGLang 为什么选择不做 LLM 框架而是做 serving system？RadixAttention 的核心洞察是什么？与 vLLM 的分叉点在哪？
2. **架构演化路径**：从 v0.2→v0.5 版本，serving stack 每一层（scheduler / attention backend / KV cache / quantization / routing）如何重构？
3. **KV cache 为中心**：RadixAttention 把 KV cache 视为可复用的"记忆"而非一次性上下文，这一范式如何驱动 prefix caching / multi-turn / multi-layer sharing？
4. **分布式推理**：TP/DP/EP disaggregation 的工程路径？与 NVIDIA Dynamo、Mooncake、LMCache 的关系？
5. **Attention backend 策略**：FlashInfer vs TRT-LLM vs vkernel vs triton 的演进逻辑？
6. **量化部署**：FP8/INT4/W4A8 的支持路线？与 GPTQ/AWQ/GGUF 生态的集成？
7. **Reasoning 模型**：DeepSeek-R1/V3 带来的部署挑战（长输出、thinking token），SGLang 如何优化？
8. **芯片设计启示**：SGLang 的内存管理、调度策略对下一代加速器（NPU/ASIC）的启示？

## 与 vLLM 差异化判断框架
- "KV radix tree as the first-class abstraction" vs "block-based KV cache as fixed page"
- "dynamic tree-shaped reuse" vs "fixed-size block reuse"
- 调度策略：continuity-aware vs longest-prefix-match
- 多模态 / speculative 支持节奏差异
