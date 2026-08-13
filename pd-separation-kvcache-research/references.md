# References Index — P/D Disaggregation KV Cache Survey

> 调研周期：2026-07-14 ~ 2026-07-15
> Status 说明：✅ 成功访问 · ⚠️ 部分访问 · ❌ 失败/受限（如 503 / 速率限制）

---

## 1. SGLang 社区

| # | URL | Status | 类型 |
|---|---|---|---|
| 1 | https://www.lmsys.org/blog/ | ✅ | Blog index |
| 2 | https://www.lmsys.org/blog/2025-05-05-large-scale-ep | ✅ | Blog: DeepSeek R1 + PD + 96 H100 |
| 3 | https://www.lmsys.org/blog/2025-06-16-gb200-part-1 | ✅ | Blog: DeepSeek + GB200 (2.7×) |
| 4 | https://www.lmsys.org/blog/2025-07-20-k2-large-scale-ep | ✅ | Blog: Kimi K2 + 128 H200 |
| 5 | https://www.lmsys.org/blog/2025-07-17-mtp | ⚠️ | Blog: MTP + PD |
| 6 | https://www.lmsys.org/blog/2026-01-12-epd | ⚠️ | Blog: EPD Disaggregation |
| 7 | https://www.lmsys.org/blog/2026-06-01-hetero-epd | ⚠️ | Blog: Heterogeneous CPU+GPU EPD |
| 8 | https://www.lmsys.org/blog/2026-05-28-mori | ⚠️ | Blog: AMD MI355X + MoRI |
| 9 | https://github.com/sgl-project/sglang | ✅ | GitHub main |
| 10 | https://docs.sglang.ai/ | ⚠️ | Docs root |

## 2. vLLM 社区

| # | URL | Status | 类型 |
|---|---|---|---|
| 11 | https://docs.vllm.ai/en/latest/features/disagg_prefill.html | ✅ | Feature doc |
| 12 | https://docs.vllm.ai/en/latest/examples/disaggregated/disaggregated_serving/ | ✅ | Example guide |
| 13 | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/nixl_kv_push_connector.md | ✅ | Design doc |
| 14 | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/nixl_kv_cache_lease.md | ✅ | Design doc |
| 15 | https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/arch_overview.md | ⚠️ | (only small grep hit) |
| 16 | https://github.com/vllm-project/vllm | ✅ | GitHub main |

## 3. 第三方系统

| # | URL | Status | 类型 |
|---|---|---|---|
| 17 | https://github.com/ai-dynamo/dynamo | ✅ | GitHub main |
| 18 | https://github.com/MoonshotAI/Mooncake | ✅ | GitHub main |
| 19 | https://github.com/LMCache/LMCache | ✅ | GitHub main |
| 20 | https://docs.lmcache.ai/ | ✅ | Docs |
| 21 | https://github.com/LMCache/LMCache/blob/main/docs/design/ | ⚠️ | Doc placeholder |

## 4. 论文（按年）

| Year | arXiv ID | URL | Status |
|---|---|---|---|
| 2024 | 2401.09670 | https://arxiv.org/abs/2401.09670 | ✅ (full abstract) |
| 2024 | 2407.00079 | https://arxiv.org/abs/2407.00079 | ✅ (full abstract) |
| 2024 | 2412.12488 | https://arxiv.org/abs/2412.12488 | ✅ (listed) |
| 2025 | 2501.14743 | https://arxiv.org/abs/2501.14743 (KVDirect) | ✅ (abstract) |
| 2025 | 2510.09665 | https://arxiv.org/abs/2510.09665 (LMCache) | ✅ (abstract) |
| 2025 | 2510.13223 | https://arxiv.org/abs/2510.13223 (BanaServe) | ✅ (abstract) |
| 2025 | 2511.20982 | https://arxiv.org/abs/2511.20982 (DOPD) | ✅ (abstract) |
| 2025 | 2512.03416 | https://arxiv.org/abs/2512.03416 (TokenScale) | ✅ (abstract) |
| 2025 | 2512.18194 | https://arxiv.org/abs/2512.18194 (TraCT) | ✅ (abstract) |
| 2026 | 2601.11822 | https://arxiv.org/abs/2601.11822 (RAPID) | ✅ (abstract) |
| 2026 | 2602.18755 | https://arxiv.org/abs/2602.18755 (DualScale) | ✅ (abstract) |
| 2026 | 2602.21548 | https://arxiv.org/abs/2602.21548 (DualPath) | ✅ (abstract) |
| 2026 | 2603.13358 | https://arxiv.org/abs/2603.13358 (PPD multi-turn) | ✅ (abstract) |
| 2026 | 2603.17456 | https://arxiv.org/abs/2603.17456 (Multi-stage Flow) | ✅ (abstract) |
| 2026 | 2605.01708 | https://arxiv.org/abs/2605.01708 (SplitZip) | ✅ (abstract) |
| 2026 | 2605.16637 | https://arxiv.org/abs/2605.16637 (HexAGenT) | ✅ (abstract) |
| 2026 | 2605.22850 | https://arxiv.org/abs/2605.22850 (ObjectCache) | ✅ (abstract) |
| 2026 | 2606.01839 | https://arxiv.org/abs/2606.01839 (Observ, not Pred) | ✅ (abstract) |
| 2026 | 2606.03910 | https://arxiv.org/abs/2606.03910 (NetKV) | ✅ (abstract) |
| 2026 | 2606.07684 | https://arxiv.org/abs/2606.07684 (SCD) | ✅ (abstract) |
| 2026 | 2606.08635 | https://arxiv.org/abs/2606.08635 (SpectrumKV) | ✅ (abstract) |
| 2026 | 2606.24506 | https://arxiv.org/abs/2606.24506 (CrossPool) | ✅ (abstract) |
| 2026 | 2606.29986 | https://arxiv.org/abs/2606.29986 (HBM Not All You Need) | ✅ (abstract) |
| 2026 | 2607.01617 | https://arxiv.org/abs/2607.01617 (3DLS) | ✅ (full abstract) |
| 2026 | 2607.01831 | https://arxiv.org/abs/2607.01831 (Lynx) | ✅ (listed) |
| 2026 | 2607.02043 | https://arxiv.org/abs/2607.02043 (Prefill Deflection) | ✅ (abstract) |
| 2026 | 2604.15039 | https://arxiv.org/abs/2604.15039 (PaaS) | ✅ (abstract) |

## 5. 中文社区 & 头部公司

| # | URL | Status | 来源 |
|---|---|---|---|
| 22 | https://github.com/deepseek-ai/DeepEP | ✅ | DeepSeek |
| 23 | https://github.com/ModelTC/LightLLM | ✅ | 网易 | 
| 24 | https://github.com/flashinfer-project/flashinfer | ✅ | FlashInfer team |
| 25 | https://www.nvidia.com/gtc/ | ⚠️ | GTC 2026 (NIM) |

## CXL 专项（详见 research_cxl.md）

| Keyword | Sources |
|---|---|
| CXL 3.0 规范 | https://computeexpresslink.org/ |
| SK Hynix CMM | ✅ 间接引用 |
| Beluga CXL Switch | ✅ 引用 |
| PNM-CXL | ✅ 引用 |
| TraCT / CXL Shared KV | https://arxiv.org/abs/2512.18194 |

---

## 6. PD 分离请求路由深度调研追加来源（2026-07-15）

> 详细报告见 [routing-report/report.html](routing-report/report.html)

### 6.1 头部公司 / 云服务

| Company | Source URL | Remarks |
|---|---|---|
| Meta | https://pytorch.org/blog | Disaggregated Inference at Scale with PyTorch & vLLM (2025-09-12) |
| Meta Engineering | https://engineering.fb.com | Scaling LLM Inference (2025-10) |
| Alibaba Cloud PAI | https://help.aliyun.com/zh/pai | PD 分离部署（静态/动态/共置三种模式） |
| Alibaba Cloud Blog | https://www.alibabacloud.com | EAS 发布 PD 分离 (2025-08-06) |
| Volcengine | https://www.volcengine.com/docs | 推理模式（PD 共置 + PD 分离） |
| AWS | https://docs.aws.amazon.com/bedrock | Dynamo + vLLM + HyperPod 集成 |
| NVIDIA Dynamo | https://developer.nvidia.com/blog | KV-aware Smart Routing |
| GitHub | https://github.com/ai-dynamo/dynamo | Apache-2.0, 被 AWS/Azure/GCP 集成 |

### 6.2 路由相关论文（路由/分流/调度维度）

| arXiv ID | Title | Key Contribution |
|---|---|---|
| 2401.09670 | DistServe (OSDI'24) | P/D placement 按集群带宽联合优化 |
| 2407.00079 | Mooncake (FAST'25) | Conductor 全局调度 + early rejection |
| 2501.14743 | KVDirect | 分布式 PD，pull-based KV |
| 2510.13223 | BanaServe | KV 与路由解耦（layer/attn migration） |
| 2512.03416 | TokenScale | Token Velocity + Convertible Decoder |
| 2601.11589 | LAPS | prompt-length-aware 独立调度 |
| 2602.21548 | DualPath | storage→decode 新路径 + RDMA 送 |
| 2603.13358 | PPD MultiTurn | Turn 2+ append-prefill 在 D 节点 |
| 2603.21354 | WRP Architecture | vLLM Semantic Router 3 维框架 |
| 2604.08075 | Dual-Pool Token-Budget | token budget 路由分拣 |
| 2604.09562 | StreamServe | metric-aware + speculation |
| 2605.16637 | HexAGenT | Agentic workflow DAG 调度 |
| 2606.01839 | ConServe | 对话级调度，p95 First ↓51% |
| 2606.03910 | NetKV | 网络拓扑 aware decode 选择 |
| 2606.17081 | Price of Anarchy | 博弈论 + 自适应路由控制器 |
| 2606.24506 | CrossPool | FFN weight pool vs KV pool 分离路由 |
| 2607.00466 | ELDR | MoE expert signature routing |
| 2607.02043 | Load-Aware Prefill Deflection | 解码节点在 P 过载时替跑 prefill |
| 2504.09285 | DynaServe | micro-request 任意 token 切分 PD |
| 2504.18154 | EcoServe | 部分 PD + 滚动激活 |
| 2511.20982 | DOPD | 动态调整 P/D 比例 |
| 2602.18755 | DualScale | Phase-aware placement + DVFS |
| 2603.17456 | MFS | Multi-stage flow Least-Laxity-First |
| 2604.15039 | Prefill-as-a-Service | 跨 DC commodity Ethernet KV 传输 |
| 2606.29986 | HMA-Serve | GDDR-P + HBM-D 异构 KV 流通 |

### 7. 访问失败/降级

| URL | Issue |
|---|---|
| https://blog.vllm.ai/ | 抓取空白 |
| https://export.arxiv.org | Rate limit 触发 (301 redirect + 503)，须 30~60s 间隔 |
| vLLM 中文部署文档 | 未找到独立中文站 |
| 字节跳动 MLV Engine blog | 无法直接访问 |
| Google Vertex AI PD 相关工程博客 | 未找到公开材料 |
| OpenAI inference 架构博客 | 未公开（仅招聘/采购信息） |
| Anthropic 推理栈 | 未公开 PD 路由细节 |
| JetBrains LLM Internals 中文站 | 需要读者访问 —— 参考但未深抓 |
