# vLLM Key Insights (调研过程积累)
Date: 2026-07-16

## A. Roadmap & Design Vision (已采集)

### 版本节奏 (重大发现)
- vLLM 已加速到 **双周发布节奏** (biweekly), 从 v0.16(2026-02) → v0.25(2026-07), 半年内10个大版本
- 对比旧路线图的 "v0.4→v0.5→v0.6" 预期，实际远超预期——架构迭代极快

### Q3 2026 Roadmap (#48168, simon-mo) — 核心方向
**Agentic Workload 成为主题**: Q2 达到 TensorRT 级性能, Q3 聚焦生产级 Agent负载和高交互性premium tokens
- **Flat Model Migration**: 完成 top-20 模型架构迁移, 新模型 day-0 仅 Flat Model
- **Model Runner V2**: 完成迁移, MRV1 废弃
- **Scheduler 重构** + **KV Cache Manager 重新设计**
- **Rust Frontend → production ready** + refactored tool-calling
- **降低冷启动时间** (#48193)

### SIG 结构 (Special Interest Groups)
1. **SIG Core**: Scheduler/KV Cache Manager/Distributed/Model Runner/KV Connector
2. **SIG Large Scale Serving**: 聚焦 disagg/wide EP/elastic serving on GB200/B200/H200, 对接 llm-d/Dynamo/AMD
   - Agent-oriented Prefix Caching (Session-ID/Correlation-ID)
   - KV Events for distributed KV caches (Mooncake P2P, tiered offloading)
   - Elastic EP: async scale-up/down, fault-tolerance recovery
   - AMD Parity: RCCL/RDMA path for Disagg+KV offload+Elastic EP+Mooncake
3. **SIG Spec Decode**: acceptance length >5, target 1000 TPS, DFlash/DFlare/DSpark draft模型, Dynamic Speculative Decoding
4. **SIG Quantization**: KV-cache compression → production-grade (FP8/NVFP4/INT2-4/TurboQuant/HIGGS), unified QuantKey dispatch, manual quant fusion + Flat Model
5. **vLLM-Omni**: 实时全双工模型(JoyVL/MiniCPM-o), 视频生成/FastVideo集成
6. **SIG RL**: sleep/wake/drain state transitions, weight update, training-inference consistency

### Q2 2026 Roadmap (#39749) — 上季度对照
- MRV2 hardening + default (完成), KV cache manager rethink(完成), scheduler issues(完成)
- PD disagg: 继续 roadmap #33702
- Zero-cost async EPLB (完成), experimental fault-tolerant EP (完成)

### PD Disaggregation (#33702) — 已支持特性清单极长
- NIXL 核心框架, 完全异步KV传输, 多传输后端(UCX/LIBFABRIC/RIXL)
- 异构TP (P和D不同TP大小), 异构block_size, 异构KV layout (HND↔NHD)
- CPU host buffer transfer (D2H→H2D, 用于TPU/XPU)
- 兼容性hash验证, KV转移失败处理, NIXL遥测/指标
- Bidirectional KV transfer (#43097), KV block Lease机制 (#43099)
- Speculative decoding + PD disagg 集成

### Rust Frontend (#44280)
- 已落地: /v1/chat+completions streaming, tool calling, 多引擎LB, admin routes
- 待完成: Elastic EP支持, Anthropic/Responses API, Pooling/embedding APIs, STT/翻译
- 原则: 不做1:1 parity, 跳过不常用/纯Python workaround/应重新设计的feature

### KV Offloading (#33689)
- 已支持: CPU-GPU offloading (NVIDIA+AMD), 自定义block size, 完全异步, LRU+ARC eviction, cross layer blocks, HMA, KV Events
- 新增: tiering (CPU→pluggable backend), 对象存储二级tier (#41968)

## B. Release Notes 关键特性演进 (v0.16→v0.25)

### 架构级里程碑
| 版本 | 关键架构变更 |
|------|-------------|
| v0.16 | Async scheduling + PP (30.8% throughput↑), Realtime API(WebSocket), RLHF weight sync |
| v0.17 | PyTorch 2.10, FlashAttention 4, MRV2 成熟(PP+DP+spec decode), --performance-mode, Elastic EP M2 |
| v0.18 | gRPC serving, GPU-less Render, NGram GPU spec decode, CPU KV offloading通用化, Responses API tool calling |
| v0.19 | Gemma 4, zero-bubble async scheduling+spec decode, ViT full CUDA Graphs, DBO generalization |
| v0.20 | **DeepSeek V4** 首发, CUDA 13.0, PyTorch 2.11, FA4 default MLA prefill, TurboQuant 2-bit KV, vLLM IR首发 |
| v0.21 | Transformers v4 deprecation, C++20 required, KV Offload + HMA集成, spec decode + thinking budget |
| v0.22 | **DeepSeek V4 成熟**, MRv2 default for Qwen3, **Rust frontend 首落地**(#40848), multi-tier KV offload |
| v0.23 | MRv2 default for Llama+Mistral, breakable CUDA graphs, PP bubble elimination, Transformers v5目标 |
| v0.24 | **MiniMax-M3**, DeepSeek-V4 持续优化, MRv2 quantized default, **DeepEP v2**, Rust frontend成熟 |
| v0.25 | **MRV2 default ALL dense models**, **PagedAttention 删除**, Transformers backend=原生速度, Streaming Parser统一 |

### 架构趋势洞察
1. **PagedAttention 被删除** (#47361): 标志性事件。V1/MRv2后端已成为绝对标准，老impl完全退出
2. **MRV2 快速替代 MRV1**: v0.21(Qwen3) → v0.23(Llama/Mistral) → v0.25(ALL dense), 约2个版本推进一个模型族
3. **Transformers backend 比肩原生**: 第三方模型day-0支持的重要路径
4. **MRV2设计哲学**: Flat Model + Model Runner分离 + 明确backend选择 → 解决MRV1的"编译驱动"不可控问题

---
(后续blog/PR/issue洞察继续补充)
