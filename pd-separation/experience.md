# 项目调研经验总结（Task-level）

> 调研主题：P/D 分离架构下 KVCache 流通方式（含 CXL 影响）
> 调研时间：2026-07-14 ~ 2026-07-15
> 调研人：Claude (LongCat-2.0)

## 调研策略总结

### 执行路线
1. 两大系统 + 三方生态覆盖：SGLang（LMSYS blogs + GitHub） / vLLM（GitHub raw design docs + official docs HTML） / 第三方（Mooncake, LMCache, NVIDIA Dynamo）
2. 学术覆盖：arxiv API (`export.arxiv.org/api/query`) + arxiv search HTML 页面
3. 前沿追踪：CXL 子方向独立调研（已完成，输出到 research_cxl.md）

### 关键事实锚点
- SGLang PD 设计详情在 blog 2025-05-05-large-scale-ep
- vLLM Push/Pull KV transfer 设计详情在 GitHub raw design docs
- NIXL lease renewal 机制（lease=30s+heartbeat）是本周报的重要新信息点
- CXL native KV Cache 系统 Top 5 已被枚举

## 调研过程中的坑

### 1. SPA 文档页 + Playwright 失败
- `docs.vllm.ai/*` 在 Playwright 只看到左侧菜单（CSR/SSR hydration 问题）
- 解决方案：
  - JS `browser_run_code_unsafe` + `document.querySelector('article').innerText` 提取正文
  - 或 curl raw GitHub markdown (raw.githubusercontent.com)
- raw.githubusercontent.com 有时要 `curl -L` 跟随 301 重定向

### 2. arxiv API 速率限制
- `export.arxiv.org` 对高频调用敏感，触发后返回 `<body>Rate exceeded.</body>`
- 须间隔 30-60s，或使用 multi-id 单请求
- arxiv search HTML（`arxiv.org/search/...`）抓取较稳定但更慢

### 3. Playwright 大文本
- LMSYS 落地页地图 schema 失败，但未阻挡 snapshot
- 某些 HTML 抓取须 substring 分段（body 一次读光会超过 input 上下文）

### 4. 多代理 vs 主线程取舍
- 首轮 Agent 子代理被用户 kill；后改用主线程 Playwright + curl 直连
- 结论：涉及 SPA/速率限制、中途可能中途改变的调研，**主线程直连成本更低、容错更高**

### 5. 中英文信息桥接
- 中文阅读（阿里云 EPD、Moonshot 独立博客）与英文阅读的结论不要混用单位
- 报告里所有数字都标注来源发布日期

## 质量与深度反思

### 做得好
- 所有关键结论都带 DOI/URL 溯源
- CXL 部分不依赖单一来源（Top 5 系统）
- 定量结论保留原文单位与语言

### 可改进
- 中文社区原生技术博客（阿里云 PAI、字节 Volcengine）受限于访问障碍未深抓
- 产业侧（NVIDIA 未公开 GTC 规划、Meta 内部系统、OpenAI 推理栈）均为间接引用
- 可对 KB/μs 级别定量测算各 KV 精度对 TTFT 预算的消耗

## Token 消耗经验

- `page.evaluate(()=>body.innerText)` 大页面 >25KB 会大量消耗 token；应 substring()
- arxiv 抓取用 urllib + ElementTree（不 Playwright）
- 子代理 prompt 不要超过 30K input；避免不必要的分支

## 下一步建议

1. 本地脚本 `crawl_kv_papers.py` 自动抓取 arxiv 并 cache，避免速率限制
2. 监听 SGLang / vLLM KV-PD commit diff（代码侧演变更快于 paper）
3. LMCache MP 模式 vs Mooncake KV-pool 的路径差异在下一版深入研究


---

## 第二步：PD 分离请求路由深度调研 (2026-07-15)

### 调研主题跃迁
- 用户追加需求："头部互联网公司大模型推理 prefill decode 之间的流量走向，有哪些途径"
- 从 **KV Cache 流通** 扩展到 **请求路由 / 流量走向 / 调度策略**（placement + data path + KV compression）

### 执行方式
- 主线程 + 并行 3 个 sub-agent（ByteDance/Volcengine 搜索、中文头部公司深度、vLLM/SGLang/Dynamo 内部文档）+ 主线程 bash curl 拉取 30+ 篇论文 arxiv abstract
- sub-agent 分别耗时 ~3min / ~8min / ~11min，通过 notification 串行拼装
- 80% 论文 abstract 通过 `curl | grep citation_abstract` 直接从 arxiv abs 页面元数据抓取，**零 Playwright 调用**

### 新增研究产出
- 新报告文件 `routing-report/report.html`（约 40+ 可溯源引用的聚焦型路由报告）
- 在原有 `report.html` §10 趋势判断追加"路由 = PD 栈调度器"条目 + 链接
- 在 header 添加跨报告 🔗 链接（用户友好型导航）
- 新增 references.md §6 章节归档所有路由来源（头部公司 + 24+ 路由论文）

### 工业实践关键发现（概况）
- **Meta**: Service Proxy at Decode + cache-aware LB，upstream 到 vLLM（PyTorch Blog 2025.09.12）
- **Alibaba PAI-EAS**: 静态/动态 PD + EP 联合部署，三种模式可配（Alibaba Cloud Blog 2025.08.06）
- **Moonshot/Kimi/Mooncake**: Conductor 全局调度 + early rejection，三级 KV 存储（FAST'25）
- **Volcengine**: NIXL GPU-GPU 零拷贝 + ZMQ 元数据协调（产品文档层面公开）
- **AWS**: Router=control plane（Dynamo）+ token threshold 路由
- **NVIDIA Dynamo**: KV-aware Smart Routing (cache overlap + active load 复合指标) — 已成为多家云的行业标准
- Google / OpenAI / Anthropic 均未公开 PD 路由细节（但通过硬件投资路线可间接推断）

### 成本节省经验
- 完整保留原始 arxiv citation_abstract 元数据（多语言/多单位/原生态），不二次翻译或重写
- 对"路由维度 vs 头部公司 vs 开源系统 vs 论文 vs KV Path 物理层"五维切分，结构清晰适合多端阅读
- 不重复使用 Playwright 抓取 splash-only 内容
- sub-agent JSONL 较大时用 `tail -c N` 提取摘要部分，不全量 Read 进主上下文
- 同主题拆分成多个独立 HTML 报告 + 互相链接，更适合手机等小屏设备，且减少单文件 token 消耗

### 下一步候选
- PD 路由续篇可横向扩展：(a) Dynamo Router SDK 源码分析、(b) Load-Aware Prefill Deflection 实测、(c) 跨 DC PD 分离（Prefill-as-a-Service）
---

## 第三步：PD 分离 D 端内存瓶颈社区深抓 (2026-07-15)

### 调研主题
- 抓取 SGLang / vllM 社区论坛、GitHub Issue、博客评论区关于 **PD 分离 D 端内存瓶颈** 的讨论
- 目标：D端内存不够、decode OOM、长 context decode 瓶颈、KV pool 规划

### 执行方式
- 主线程 Playwright 直连（放弃 sub-agent，经验证成本更低）
- 6 个主搜 URL + 8 个深度 Issue = 14 次 navigate + evaluate
- 每个页面 `document.body.innerText.substring(0, 6000)` 直接写入本地文件，Llm 仅做少量内容中转
- 最终写入 18 个原始文件（82 KB）+ 1 个 report.html

### 关键发现
- vllm RFC #43470 首次正式定义 **Decode-KV-bound** 瓶颈类（与 Prefill-bound / Decode-bound 并列）
- SGLang #24523：H200 140GB 下 SWA 分配 bug 让 PD 部署 input 被硬限 18,432 tokens（实际 pool 186,368）
- SGLang #30010：GB300 + Kimi-K2.5 + mooncake，per-rank batch > 450 时 decode 4-8 min 死锁（6/6 复现），需 PD+cuda-graph+高 batch 三者耦合
- vllm #46107 MoRIIO 实测：DeepSeek-R1 8K-in/1K-out on MI300X，混合 TP-prefill + DP/EP-decode 在 512 concurrency 吞吐 42,349 tok/s（最优）
- SGLang Uni-PD Memory-semantic 范式（#35263）：prefill 直接写 decode HBM，100% 消除显式 KV 传输
- 学术：arXiv 2606.29708（2026-06-30）系统异构 PD 四设计轴 + 三边界决策

### 访问失败
- kuncoro.io — ERR_ABORTED
- vllm-project.github.io — ERR_ABORTED
- vllm.ai/blog 多个猜测 URL — 404（SPA 路由无法猜中），内容靠 Google AI Overview 还原核心信息

### 成本节省经验
- Google AI Overview 的信息密度非常高且可溯源（含原生引文），对 PD 这种快速演进的课题，先读 AI Overview 再定向挖 Issue，比盲目扫 GitHub 高效
- vllm/sglang issue 页 innerText 直接 substring(6000) 写入文件是最省 token 的方式，Llm 只做"中转写入"不生成摘要（摘要统一到最终报告做）
- vllm.ai 是 SPA，blog URL 猜测易 404，应优先走 Google 索引 + tag 页（/blog/tags/disaggregation）复原真实 URL
- arXiv API（export.arxiv.org/api/query）直接用 curl + Python xml 解析，零 Playwright 调用即可拿摘要

### 报告输出
- report.html（手机友好，深度报告，全部关键结论带 URL 溯源）
- 18 个 .txt 原始抓取文件在 research/decode-memory-bottleneck/

### 下一步候选
- 跟踪 vllm #43470 / #46107 后续是否进入实现（policy layer / MoRIIO connector）
- 跟踪 SGLang #30010 死锁修复（涉及 flashinfer #3279 + mooncake allocator）
- 建立 Decode-KV-bound 类的监控信号标准化提案

---

## 第三步：超长序列（1M+ tokens）在 P/D 分离架构下的挑战 (2026-07-15)

### 调研主题
- 交叉方向："long context × PD disaggregation"——一个 2026 年才进入学术视野的新问题域
- 覆盖维度：D 端 HBM 容量压力、学术前沿、SGLang/vLLM/Mooncake 社区、前沿解决路径

### 执行方式
- 主线程 Playwright 导航 30+ URL（arxiv search + arxiv abs + arxiv HTML + SGLang docs + vLLM issue + SGLang PR）
- 全部 innerText 写入本地 .txt 文件，零 LLM 摘要生成
- 最终一次性写 HTML 报告

### 关键发现（概况）
- **核心判断**：该交叉方向是"新浮现、未被系统化解"的问题域，首批显式论文（Lynx, SAC, 3DLS, PrfaaS, Dual-Pool）集中在 2026 Q1-Q3
- **Mooncake (arxiv:2407.00079)** 是基准文献：overload early rejection + chunked pipeline parallelism + layer-wise prefill
- **Lynx (arxiv:2607.01831)** 挑战"KV cache 不可分割"假设，Anchor+Residual 双流让 decode 在 10% KV 到达时就开始
- **SAC (arxiv:2606.19746)** 首个 CXL 原生 disaggregated KV 系统，对 sparse attention 按需取 top-k
- **3DLS (arxiv:2607.01617)** 首次从芯片层揭示 P→D KV 与 decode TP 共享互连的混合流量竞争
- **PrfaaS (arxiv:2604.15039)** 承认 dense attention 下 PD 长 context 无解，必须依赖 hybrid attention
- **Dual-Pool Routing (arxiv:2604.08075)** 揭示 fleet 按 worst-case 配置导致 4-8x 吞吐浪费
- **CXL-1M (arxiv:2511.00321)** 在 405B 1M token 下取得 21.9x 吞吐提升
- **SGLang** PP for Long Context 文档明确：1M+ context 首先是 PP 问题；PP+PD 组合 2025-10 #8846 才初步支持
- **vLLM #11286** 实测：decode 速度从 5364 tok/s（&lt;100 token）断崖下降到 273 tok/s（8K+ token）
- **工业界**（Meta/Alibaba/AWS/NVIDIA）均未公开 1M+ context PD 专项策略

### 调研过程中的坑
- Google 搜索在越南语界面下反复出现重定向循环（google.com → 越南语界面 → 偶尔跳到 SGLang 文档首页）
- 部分 arxiv search 查询返回 0 结果（"Infini-Attention disaggregated"、"KV cache offloading disaggregated CXL"等）
- arxiv HTML 版论文（如 Mooncake）比 abstract 页信息量大 10 倍，值得优先抓
- SGLang 文档从 docs.sglang.ai 迁移到 docs.sglang.io，旧 URL 会 301/404

### 成本节省经验
- arxiv HTML 版论文（arxiv.org/html/xxx）是信息密度最高的单源，一次抓 15000 字符可覆盖论文 60% 关键内容
- 对"交叉方向是否有现成论文"的判断，用 arxiv search 多个 query 组合 + 检查 0 结果模式比单 query 更可靠
- 工业界"未公开讨论"本身也是结论——在报告中显式标注"未发现"比回避更有价值

### 报告输出
- research/longcontext_pd/report_longcontext_pd_challenge.html（手机友好，深度报告）
- 34 个 .txt 原始抓取文件在 research/longcontext_pd/

### 下一步候选
- 跟踪 3DLS 流片进展（首个 PD 芯片层解决方案）
- 跟踪 Lynx 开源实现（SIGCOMM'26 接收后代码可能公开）
- 跟踪 SGLang Decode-side PP 支持进度（#8846 待续）
- 跟踪 Mooncake PrfaaS 生产部署数据（Moonshot 后续论文）

---

## 第三步：PD 分离 × 超长上下文 调研 (2026-07-15)

### 调研主题
工业界头部公司 / 团队关于"超长序列 LLM 推理挑战"的博客、论文、技术分享，特别是 PD 分离视角。

### 执行方式
纯 Playwright 主线程导航 (含 Google 搜索 × 4 + LMSYS/vLLM/Hao AI Lab/LMCache/Z.ai/Mooncake 共 ~23 个 URL)；GitHub raw 直连 timeout。

### 关键发现
- **SGLang 团队公开承认 long-context 是 ongoing effort**: 在 Kimi K2 博客明确写 "Longer output for agentic scenarios will be future work" 和 "ongoing efforts to optimize the long-context scenarios" (2025-07-20)
- **Mooncake 论文** (arXiv:2407.00079): "To mitigate these, we developed a prediction-based early rejection policy... excels in long-context scenarios. Up to 525% throughput"
- **Mooncake GitHub 仓库 (MoonshotAI/Mooncake) 已删除 404** —— 可能已内化进 Kimi 产品
- **LMCache**: PD + long-context 被明确定位为"foundational to high-performance LLM inference"，通过 NIXL+buffer 解耦
- **Z.ai ZCube 博客**: 直接给出 PD × long-context 的定量证据：100Gbps → 200Gbps NIC 带来 19% throughput + 22% TTFT 改善
- **vLLM MiniMax M3 博客**: 1M-token context 通过 MiniMax Sparse Attention (block-sparse) + MXFP8 + chunked prefill 支持
- **SGLang GB300 博客**: "Decode path: The Memory Bottleneck in Long-Context Inference" — KV-dominated and memory-bound; per-token KV footprint = 35,136 Bytes; 136K cached tokens → ~4.45 GiB/GPU
- **LMCache README**: 明确列出 "PD disaggregation and KV transfer: Support KV cache transfer ... over NVLink, RDMA, or TCP through transport layers such as NIXL"
- **Hao AI Lab DistServe 原始博客**: 承认 "For larger models, longer sequences... KV cache transfer overhead" 并通过 placement 最小化

### Google AI 摘要核心结论
- PD 分离下 long-context 三大矛盾:
  1. Bandwidth Bottlenecks — KV Cache 跨节点网络传输阻塞
  2. Resource Mismatch — Prefill 计算密集 vs Decode 内存密集
  3. Head-of-Line Blocking — 长 request 阻塞短 request 造成 timeout

### 判断: 已存在 / 可解 / 未解？
- **已存在问题**（所有团队共识）: 是
- **"可解"的程度（通过 engineering 缓解）**: SGLang/LMCache/vLLM/Mooncake/Z.ai 均给出工程级解法（PP prefill, MTP, HiCache, NIXL transfer, ZCube 网络）
- **"未解"的本质**: 1M+ token context 的 KV 体量（数十 GB / 用户）导致 (a) 内存墙 (b) 跨节点 PD transfer 成为瓶颈 (c) long-context latency SLO 难以同时满足。这是 PD 分离视角下的**结构性挑战**，不是 bug

### 成本节省经验
- Google AI 摘要用 6000 字即可覆盖核心观点 + URL 列表 —— 非常适合"广度调研"
- GitHub raw 在国内网络应该直接记为不可达，不要用 bash 再做 retry
- 跨站 tab 策略：多个 tab 保留，避免 navigate 时 destroy context

### 下一步候选
- SGLang PD docs (docs.sglang.ai/en/latest/advanced_features/pd_disaggregation.html) 需要单独抓
- vLLM 的 Push/Pull KV 设计文档（在 GitHub design doc 目录中，前次已抓）
- LMCache MP mode vs Mooncake pool 路径差异


---

## 第三步：PD 分离 D 端 KV Cache 内存容量调研 (2026-07-15)

### 任务目标
抓取 vLLM 社区关于 PD 分离设计中 D 端（Decode worker）内存容量分析材料，覆盖 6 类 URL。

### 执行方式
- SPA docs.vllm.ai 文档（HTML 页面）发现全部重定向到 GitHub issues（disagg_prefill.html → MoRIIO issue 页面）。**直接 curl raw.githubusercontent.com 拉取 .md 源码**更稳定高效。
- 为弥补 SPA 失效的损失，主动追加抓取：
  - `docs/configuration/conserving_memory.md`
  - `docs/configuration/optimization.md`
  - `docs/features/disagg_prefill.md`
  - `docs/features/kv_offloading_usage.md`
  - `docs/design/{hybrid_kv_cache_manager,paged_attention,prefix_caching,arch_overview}.md`
  - 关键源码：`vllm/config/cache.py`（CacheConfig 定义）、`vllm/v1/worker/gpu_worker.py`、`vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`、`vllm/v1/core/kv_cache_utils.py`、`vllm/utils/mem_utils.py`、`vllm/profiler/layerwise_profile.py`
- 使用 Python `urllib.request` + GitHub Contents API 探测文件路径（vLLM 已把 `config.py` 拆分为 `config/` 包, 旧 `config.py` 已 404）。
- 全程避免 Playwright 抓取 SPA → 零 hydration/DRY 问题。

### 关键定量发现
- `block_size = 16` (CacheConfig.DEFAULT_BLOCK_SIZE ClassVar)
- `gpu_memory_utilization = 0.92` (per-instance)
- `cache_dtype = "auto"` (= model dtype; 可选 fp8 / nvfp4 / turboquant_* / per-token-head 系列, 最高 5.3× 压缩比)
- `enable_prefix_caching = True`
- `kv_cache_memory_bytes = None`（旁路 profiling）
- KV Cache 容量公式：`(total_gpu_memory × 0.92 − non_kv_cache_memory − cudagraph_memory_estimate) ÷ (block_size × kv_hidden_size × num_layers)`
- vLLM v1 只有 RECOMPUTE preemption, 无 swap/restore; 仅一个 `watermark_blocks`（% of num_blocks）作为护栏
- PD 中 D 端无论 pull/push 都**先分配 block**——内存承诺
- Lease mechanism (PR #41383): `kv_lease_duration=30s`, `decoder_kv_blocks_ttl=480s`, HB interval ~5s, extension ~20s
- Memory profiling 三类：torch 权重 / torch 激活峰值 / 非 torch 增量

### 质量反思

#### 做得好
- 所有核心默认值直接从 `cache.py` 源码摘录, 不靠记忆
- PD 内存公式 + lease 调度 + 1M 长 context 影响都给出了量化推导
- 报告以 single-file HTML 输出, 手机端阅读友好; 每条关键结论都有 tag + 超链接
- 把 `kv_cache_dtype` 全枚举值列出（含 nvfp4 / turboquant / per-token-head 系列）
- experience.md 持续更新, 形成任务知识沉淀

#### 可改进
- `docs.vllm.ai` 的 SPA 在 Playwright 下全部重定向到 GitHub issues (可能因为这次版本 docs 静态化部署), 原始 6 个 URL 只成功访问了 1 个（arch_overview.md 后来被我们发现是 raw 问题）——下一个任务应先探测再下手
- v1 的 `Scheduler` (vllm/v1/core/sched/) 代码没有仔细抓取, watermark 的精确默认值未拿到源码证据（只看到赋值 `int(watermark * kv_cache_config.num_blocks)` 但 watermark 默认值未在 kv_cache_manager.py 内找到）
- 1M context 的数值估算用了社区经验, 未用 vLLM 实际跑 profile
- 报告因时间限制没有做经验值 vs 实测值的明确分栏

### Token 节省经验
- 把"找源码路径"从 Playwright 转到 GitHub Contents API（json 文件列表）+ python urllib, 一次大清单调用完成, 消耗 token 极小
- 源码拉取启用 Python urllib (`headers={"User-Agent":"Mozilla/5.0"}`) 直接写本地文件, 完全避免把 345KB 源文件读入上下文（如 gpu_model_runner.py）
- 只 grep 需要段落, 不全量 Read
- 报告集中在单个 HTML 中, 避免多个文件

### 下一步建议
- 抓取 `vllm/v1/core/sched/` 目录所有 scheduler 相关代码, 补齐 watermark 默认值 + PD 等待队列长度对 KV Cache 承诺的定量模型
- 触发一次 `vllm serve` 在 H100 上做 profile run, 实测出 KV Cache 容量 + 激活峰值
- 结合 LMCache + Dynamo 做横向对比, 形成 PD memory sizing guide
