---
title: "[Draft] Google Rapid Storage 不是为 KV Cache 而生 — 解读 Google AI Hypercomputer 的存储层级设计"
date: 2026-07-22
tags: ["Google-Cloud", "Rapid-Storage", "KV-Cache", "TPU-8i", "Managed-Lustre", "AI-Inference", "Storage-Hierarchy"]
excerpt: "Google Rapid Storage 发布后，业界普遍认为它是为 AI 大模型 KV Cache 而生。但深入调研后发现：这是一个误解。Google 为 KV Cache 外部化准备的方案是 Managed Lustre 和一个未公开的专用子系统，而 Rapid Storage 解决的是另一个问题——让数据不成为加速器的瓶颈。"
---

# [Draft] Google Rapid Storage 不是为 KV Cache 而生

## 一个误解的诞生

2026 年 4 月 Google Cloud Next 大会上，Google 发布了 Cloud Storage Rapid（含 Rapid Bucket 和 Rapid Cache），同时发布的还有 TPU 8i——后者将片上 SRAM 提升了 3 倍至 384 MB，官方明确表示是为了"host massive KV Caches entirely on silicon"。

这两件事同时发生，加上整个行业对 KV Cache 外部化的高度关注，一个自然的推论诞生了：**Rapid Storage 也是为 KV Cache 而生的**。

但这个推论是错的。

> ==Rapid Storage 不是为 KV Cache 而生。==它解决的是"如何让数据快速到达加速器"，而 KV Cache 外部化需要的是"推理状态如何在节点间共享和快速恢复"。

要理解这个区别，需要看懂 Google 为 AI Hypercomputer 设计的完整存储层级。

---

## 一、Google 的存储层级：四层架构

Google Cloud 官方文档 [1] 明确给出了 AI 工作负载的存储推荐方案：

| 使用场景 | 首选方案 | 为什么 |
|---|---|---|
| **训练数据加载** | Rapid Bucket | 15 TB/s 带宽 + EB 级容量 + 成本效率 |
| **Checkpoint 写入/恢复** | Managed Lustre | 亚毫秒延迟 + 并行数据访问 |
| **推理模型权重加载** | Rapid Cache / Rapid Bucket | 消除推理冷启动 |
| ==**KV Cache 外部化**== | ==**Managed Lustre**== | ==亚毫秒延迟 + 跨节点拉取== |

关键发现：**KV cache offloading 的首选方案是 Managed Lustre，不是 Rapid Storage。**

Google 官方博客的原话：

> "We believe that Google Cloud Managed Lustre should be your primary storage solution for external KV Cache." [2]

---

## 二、Rapid Storage 的真正能力

Rapid Storage 基于 Google 内部的 Colossus 分布式存储系统（与 Gemini、YouTube 共享基础设施），包含两个产品：

### Rapid Bucket

| 指标 | 数值 | 对比传统对象存储 |
|---|---|---|
| 带宽 | ==15 TB/s== | 7-15x 提升 |
| 请求速率 | ==20M ops/s== | 200x 提升 |
| Checkpoint 写入 | — | ==3.2x== 加速 |
| Checkpoint 恢复 | — | ==5x== 加速 |
| GPU Blocked Time | — | ==50%== 减少 |

### Rapid Cache

| 指标 | 数值 |
|---|---|
| 聚合读取吞吐 | ==2.5 TB/s== |
| 适用场景 | 突发工作负载（推理模型加载） |
| 代码变更 | 零代码变更 |

这些指标指向一个明确的方向：**训练吞吐量和加速器利用率**，而非推理时的 KV Cache 存取。

---

## 三、为什么 Rapid Storage 不适合 KV Cache？

核心原因是架构层面的不匹配：

| 需求 | KV Cache 外部化 | Rapid Storage（对象存储） |
|---|---|---|
| **访问粒度** | ~KB 级随机读取 | ~MB 级块读写 |
| **一致性** | 强一致性（状态不能丢） | 最终一致性 |
| **延迟** | 亚毫秒（直接影响 TTFT） | 亚毫秒（仅打开文件） |
| **访问模式** | 按 token 索引随机存取 | 大文件顺序读写 |
| **共享性** | 多 worker 同时读取 | 有限 |

相比之下，Managed Lustre 作为 POSIX 并行文件系统，天然支持 KV Cache 外部化的所有需求：细粒度随机访问、跨节点共享、RDMA 直达。

Managed Lustre 的 KV Cache 性能数据 [2]：
- TTFT 降低 ==40%+==（对比仅用主机内存）
- 推理吞吐提升 ==75%==
- 基于 DDN EXAScaler，支持 TPUDirect RDMA

---

## 四、Google 的 KV Cache 策略："尽可能不上外部存储"

从 TPU 8i 的架构设计可以推导出 Google 的 KV Cache 策略：

| 层级 | 存储 | 容量 | KV Cache 角色 |
|---|---|---|---|
| **L0** | 片上 SRAM | ==384 MB/chip== | ==最优==：KV Cache 首选驻留 |
| **L1** | HBM | ==288 GB/chip== | ==次优==：主要容量层 |
| **L2** | Managed Lustre | ==80 PB== | ==外部化首选==：跨节点共享 |
| **L3** | Rapid Storage | ==EB 级== | ==不适合==：训练数据/权重 |

TPU 8i 的 Pod 级 SRAM 总量：==384 MB × 1,152 chips = 432 GB==。这是 Google 为 KV Cache 设计的第一道防线——让 KV Cache 尽可能驻留在最快的存储层级上。

Google 的逻辑很清晰：**与其让 KV Cache 频繁进出外部存储，不如把它留在芯片上。**只有当 KV Cache 大到连 HBM 都装不下时，才需要 Managed Lustre 作为 L2 外部化层。Rapid Storage 在这个链条中根本不承担 KV Cache 的角色。

---

## 五、误解从何而来？

这个误解有四个来源：

**1. 同时发布，语境混同**
Rapid Storage 和 TPU 8i 的 3x SRAM 扩容在同一场大会发布，都被包装在"AI Hypercomputer for the agentic era"叙事中。但 TPU 8i 的 SRAM 是为 KV Cache，Rapid Storage 不是。

**2. "Agentic Era"叙事中的存储重定义**
Google 说"During AI inference, storage is the access layer that makes it responsive" [3]。这里的"context"被误解为 KV Cache，实际上指的是 Agent 运行时的完整上下文数据。

**3. 命名暗示**
"Rapid"（快速）暗示低延迟，容易联想到 KV Cache 外部化的延迟需求。但 Rapid 指的是对象存储层级的吞吐量和 IOPS 提升。

**4. 行业趋势的投射**
2026 年，KV Cache 外部化是行业热点（LMCache、Mooncake、NVIDIA KV offload）。任何新发布的"AI 存储"产品都容易被默认解读为"为 KV Cache 而生"。

---

## 六、那 "Dedicated KV Cache scalable storage subsystem" 是什么？

在 Next '26 的公告中 [4]，Google 还宣布了一个未公开细节的组件：

> "Dedicated KV Cache scalable storage subsystem"

这个组件被单独列出，与 TPU 8t/8i、Virgo Network、Managed Lustre 并列。截至本文写作时（2026-07），Google 尚未发布其技术细节。

它很可能是专门用于 KV Cache 外部化的独立存储层。如果存在，那才是 Google 为 KV Cache 设计的"专用存储"——但它不是 Rapid Storage。

---

## 七、对模型设计者的启示

从模型架构设计的角度看 Google 的存储层级：

**如果你设计长上下文模型：**
TPU 8i 的 384 MB SRAM 意味着你可以将模型的 KV Cache 完全驻留在片上。这是比 GPU 方案（HBM 有限，需要 offload）更具优势的架构。对于百万 token 级别的上下文窗口，TPU 8i 的存储层级设计使得 KV Cache 的存取延迟远低于 GPU 方案。

**如果你设计 Agent 系统：**
Managed Lustre 的 KV Cache 共享能力意味着多 Agent 可以共享上下文状态，减少重复计算。这是 Google 在 Agentic AI 时代的核心存储优势。

**Rapid Storage 对你的直接影响最小：**
它更多影响训练效率和推理冷启动时间，而非推理过程中的 KV Cache 管理。

---

## 八、一句话总结

> ==Google Rapid Storage 是 AI 数据管道的"高速公路"，让数据快速到达加速器；而 KV Cache 外部化需要的是"保险箱"——Managed Lustre 提供的共享、持久、细粒度状态存储。两者互补，但不可互相替代。==

Rapid Storage 解决的是供给端的问题（如何让 TPU/GPU 持续满负荷运转），KV Cache 外部化解决的是状态管理的问题（如何让推理状态在节点间高效流转）。在 Google 的 AI Hypercomputer 架构中，两者各司其职，不存在谁为谁而生的关系。

---

## 参考来源

[1] [Google Cloud Documentation — About storage services for AI and ML workloads](https://docs.cloud.google.com/ai-hypercomputer/docs/storage) — 官方存储层级推荐，KV cache offloading 首选 Managed Lustre

[2] [Blocks & Files — Google's cloud storage gets faster and smarter for AI](https://www.blocksandfiles.com/public-cloud/2026/04/22/googles-cloud-storage-gets-faster-and-smarter-for-ai/5218551) — Managed Lustre 作为共享 KV Cache 方案的 40% TTFT 改善

[3] [Google Cloud Blog — Storage innovations at Next '26](https://cloud.google.com/blog/products/storage-data-transfer/next26-storage-announcements) — Rapid Storage 官方定位和性能指标

[4] [Google Cloud Blog — AI infrastructure at Next '26](https://cloud.google.com/blog/products/compute/ai-infrastructure-at-next26) — Dedicated KV Cache scalable storage subsystem 公告

[5] [Google Cloud Blog — TPU 8t/8i technical deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) — TPU 8i 384 MB SRAM 为 KV Cache 设计

[6] [IO Fund — Google TPU v8 vs Nvidia](https://io-fund.com/ai-stocks/google-tpu-v8-vs-nvidia-inference-rewrites-ai-market) — 分析师视角的 TPU 8i 推理优势

[7] [StorageNews — Rapid cloud storage fixes AI training bottlenecks](https://storagenews.top/posts/rapid-cloud-storage-cuts-gpu-idle-time-by-half/) — Managed Lustre 用于 KV Cache，Smart Storage 用于数据选择

---

*本报告基于的完整调研（含所有访问成功/失败的来源清单）见：[Google Rapid Storage 深度调研报告](/posts/research/google-rapid-storage/research_report.html)*
