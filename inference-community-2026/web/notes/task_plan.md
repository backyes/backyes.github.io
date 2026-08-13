# vLLM vs SGLang 推理引擎社区调研 (2025.7~2026.7)

## 调研Prompt设计
- 第一轮prompt（本次）：roadmap + 关键PR/issue + 代码架构演进 + 第三方benchmark + 学术视角
- 计划在第一轮结果上再写第二轮深化prompt（针对架构设计争论焦点和代码级细节）
- 目标受众：软件架构师、推理系统架构师、超节点设计者、体系结构研究者

## 调研方法论
- rule1: 坚持使用playwright做联网搜索，不用内置WebSearch
- rule2: 优先CLI工具（gh api等）获取结构化数据，大模型仅做洞察分析
- rule17: 所有原始材料本地持久化，最终输出可点击溯源的HTML报告
- rule6/8/9/10: 要求深度洞察，不流于表面

## 并行agent任务分配
1. agent A (vLLM): roadmap + PR/issue + 代码架构
2. agent B (SGLang): roadmap + PR/issue + 代码架构 + reasoning支持
3. agent C (第三方): benchmark对比 + 行业媒体 + 学术引用 + 社区讨论

## 关键调研问题（待回答）
- 两个引擎在disaggregated prefill架构上的路线差异？
- KV cache管理策略的根本分歧（PagedAttention vs RadixAttention）？
- 多机并行策略的演进（TP/EP/DP/PP）？
- 调度器设计哲学差异？
- 量化/硬件支持差异？
- 社区治理模式差异（vLLM基金会 vs LMSYS主导）？
