---
title: "MTP 论 — 多符预测对算力、芯片、互连之结构性影响"
date: 2026-07-31
tags: ["DeepSeek", "MTP", "文言文", "芯片", "算力", "互连"]
excerpt: "以文言译英文原文，论多符预测（MTP）如何移推理之负载从访存密集转向计算密集，及其对芯片、互连、超节点之结构性冲击。"
---

# MTP 论

## 要旨

**多符预测（Multi-Prediction Prediction, MTP）者，非止推理之微调也，乃重构计算-存储-通信三角之根本术也。** 传统自回归解码，每符计算甚微，而 KV-Cache 访存极繁，此访存密集之典范也。MTP 反是：以同一 KV-Cache 摊派于 k 符之预测，遂使算力强度（FLOPs/Byte）随 k 线性增长。

> DeepSeek 者，高端之选手也，有定力，守高性价比，击成本之最高突破口，秉普惠人类之术。

---

## 一、算力芯片：计算与 KV-Cache 之比反转

传统解码，每符一算，KV 一访。MTP 行，则 k 符共享一 KV，算力 k 倍而访存不变。

| 指标 | 传统解码 | MTP（k 步） |
|---|---|---|
| 每符算力 | 1× | ≈k× |
| KV-Cache 访存 | 1× | 1×（共享） |
| 算力强度 | 低 | ==随 k 线性增长== |

**然则 k× 乃理论之上限也**，实得与否，系于**接受率**（acceptance rate）。DSpark（DeepSeek 与北大，2026）实证：纯堆叠之深层预测，接受率随深度急速衰减（"suffix decay"）。此正所以 DeepSeek 弃"更深 MTP"之径，改行半自回归草拟 + 置信度调度验证之故。生产部署（DeepSeek-V3）仅用 1-2 层辅助预测，loss scaling 0.1，**k=5+ 之"大 MTP"至今乃理论推测，非生产验证也**。

**业界脉络**：NVIDIA GPU 迭代，带宽增长落后于算力增长——此 MTP 所利用之结构缺口也。

| GPU | 架构 | 显存 | 带宽 | FP8 稠密 | 带宽/算力 | 出处 |
|---|---|---|---|---|---|---|
| H100 SXM | Hopper | 80 GB HBM3 | 3.35 TB/s | 989 TFLOPS | 3.4 | [NVIDIA](https://www.nvidia.com/en-us/data-center/h100/) |
| H200 SXM | Hopper | 141 GB HBM3e | 4.8 TB/s | 989 TFLOPS | 4.9 | [NVIDIA](https://www.nvidia.com/en-us/data-center/h200/) |
| B200 | Blackwell | 192 GB HBM3e | 8 TB/s | 2.25 PFLOPS | 3.6 | [NVIDIA](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) |
| B300 (NVL72) | Blackwell | 288 GB HBM3e | 16 TB/s | ~4.5 PFLOPS | 3.6 | [NVIDIA](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) |
| Rubin (R100) | Rubin | HBM4 (待定) | ~36 TB/s (估) | 待定 | 待定 | [NVIDIA](https://www.nvidia.com/en-us/data-center/technologies/rubin/) |

> ⚠️ 规格待 review — 建议读者点出处链接人工核实。

**分歧昭然**：H100→B200，算力增约 2.3×，带宽仅增约 2.4×。然算术强度上限（每 Byte 之 FLOPs）方为访存密集推理之真正约束。MTP 直击此约束：以同一 KV-Cache 摊 k 符计算，等效于将 compute-per-byte 比率乘以 k，**遂使推理负载从访存密集推入计算密集之域**。

---

## 二、互连：低延迟 vs 高带宽之分叉

MTP 于互连之影响，非一端也，随规模而分：

| MTP 规模 | 生产状态 | 利 | 害 |
|---|---|---|---|
| **小 MTP**（k=1-2） | 已部署（DeepSeek-V3） | 大超节点、低延迟语义（如 LPX 类） | — |
| **深推测**（DSpark 式） | 已部署（2026.06） | 半自回归草拟 + 置信度验证 | 纯深层堆叠 |
| **"大 MTP"**（k=5+ 纯堆叠） | ==仅理论== — 接受率衰减使不经济 | 高带宽结构（若得解） | 低延迟机制 |

**析之**：
- **小 MTP（k=1-2）** 乃当前生产现实。此规模下，低延迟互连仍有价值——LPX 类 SRAM 中心芯片于延迟敏感之解码场景依旧占优。
- **深推测（DSpark）** 乃 DeepSeek 对接受率衰减之回答：不堆更深 MTP，改行半自回归草拟 + 置信度调度验证，得 60-85% 提速。此乃"k>2 增益"之实际生产机制，非 MTP 之简单堆叠也。
- **"大 MTP"（k=5+ 纯堆叠）** 至今纯理论，盖接受率随深度急速衰减故也。k× 算力之利，仅于接受率得解时方为现实——此正所以 DSpark 弃此径也。

> **拐点何在？取决于 MTP 之两级部署规模**——小 MTP 保留低延迟价值，DSpark 式深推测则将价值转向验证带宽。

**业界对比**：NVIDIA Groq 3 LPX 以片上 SRAM 得 150 TB/s 每 LPU——约为 Rubin HBM4 22 TB/s 之 6.8×。然此优势**仅于带宽为瓶颈时方有意义**。MTP 转向计算密集，削弱 HBM 带宽天花板之间接惩罚，遂弱化 SRAM 中心架构之价值主张。

---

## 三、超节点域：从"摊销 Weight"到"通信瓶颈"

**推导链**：

```
MTP ↑ → 算力强度 ↑ → 单节点 HBM 压力 ↓
→ 大 EP 之"摊销 Weight"收益被抑制
→ 瓶颈转向通信（大块传输）
→ 利小超节点域之高带宽
```

**大 EP 之本质**：将专家居于多节点，每节点仅载部分 Weight，以摊销 HBM 带宽。MTP 瓦解之——若 HBM 带宽已非约束，则大 EP 之通信开销遂为不偿失。

**启示**：瓶颈转至通信，然乃**大块通信**——利紧凑域（机柜级）内之高带宽，非广域互连也。

---

## 四、专用芯片：SRAM 独立架构之挤压

**无 GPU 伙伴之纯 SRAM 架构，市场窗口收窄——然 LPX 乃例外，非通则也。**

| 芯片 | 架构 | SRAM | 带宽 | GPU 伙伴？ | MTP 影响 |
|---|---|---|---|---|---|
| Cerebras WSE-3 | 晶圆级 SRAM | 44 GB | ~21 PB/s | ✗ | 显存优势被稀释 |
| Groq Trillium | 确定性数据流 | 大量片上 | 80 TB/s | ✗ | SRAM 溢价更难支撑 |
| Etched Sohu | 硬接线 Transformer ASIC | 片上权重 | 极高 | ✗ | 计算密集友好 |
| **NVIDIA LPX** (Groq 3) | SRAM 中心 LPU | 500 MB/LPU | 150 TB/s/LPU | ==✓ (Rubin NVL72)== | ==价值保留==——利低延迟分支 |
| NVIDIA B200 | HBM 平衡 GPU | 极少 | 8 TB/s | — | ==更优定位==——算力优先 |

**LPX 之例外**：LPX 乃 Rubin NVL72 之解码伙伴——LPU 管低延迟解码，GPU 管 prefill/attention。此正对应 MTP 之小 MTP/低延迟分支（第二节），故 LPX 之价值主张**被 MTP 强化而非侵蚀**。其所号称 35× 每瓦吞吐量，正因其瞄准 SRAM 优势依旧有效之延迟敏感域。

**受挤压者**：无 GPU 伙伴之纯 SRAM 架构——Cerebras WSE-3（~21 PB/s 晶圆带宽）、Groq Trillium（确定性数据流，80 TB/s）——面临更严苛之算计。其设计假设曰：消除片外存储即最优解。MTP 使计算为瓶颈时，SRAM 带宽优势被稀释，而其面积/功耗劣势依旧。Cerebras 基准测试示其超 Groq 6× 以上（晶圆级），然此优势以访存密集负载衡量——随负载转移，差距收窄。

> **纯 SRAM 假设——消除片外存储即足够——正在瓦解。** 无 GPU 伙伴之架构（Cerebras、Trillium）窗口收窄。有 GPU 伙伴者（LPX）则占据 MTP 所保留之低延迟分支。

---

## 五、片上介质：群雄并起之机

MTP 转计算密集，对芯片供应链有==去高端化==之效：

- **利国产/替代存储介质**：降低对 HBM 极致带宽之依赖，国产 HBM3E、CXL 挂载存储、乃至先进 DDR 配置皆得以上桌。
- **利中端工艺节点**：计算密集负载少赖 HBM 带宽（高端工艺之差异化处），多赖原始 FLOPs（4-5nm 可达）。
- **害高端 HBM 带宽**：HBM3E（SK Hynix 市占 62%）之垄断溢价削弱，盖带宽已非约束故也。

> **人人皆得入席。** 此结构性转向，侵蚀 HBM 生态之定价权——数年之趋势逆转也。

**数据**：HBM3E 单堆叠 1.2 TB/s；中端 DDR5 单通道约 50 GB/s。24 倍带宽差距，于计算密集负载意义大减。全球 HBM 市场 2026 年预计 $58B——MTP 不缩此市场，然将压缩其**溢价**。

---

## 六、算力密度：加速堆叠，再逢新壁

**两阶段动态**：

**阶段一（1-2 年）**：访存墙暂缓，算力密度加速堆叠。Chiplet、3D 封装、晶圆级集成所面临之每计算单元带宽约束减少。

**阶段二（2-3 年）**：算力翻倍 → 存储再成瓶颈 → 触发 HBM 新一轮增长周期。

```
访存墙破 → 密度堆叠 → 算力翻倍
→ 新存储压力 → HBM 入新增长周期
**

**业界平行**：此犹如 2022-2024 周期——HBM3（819 GB/s）→ HBM3E（1.2 TB/s）→ HBM4（2.0 TB/s/堆叠）之采用加速，正因 GPU 算力超前存储带宽。MTP 压缩此周期。

---

## 七、"存储非瓶颈矣"——常见之谬

谓 MTP 消除存储压力者，**谬也**。

- MTP 暂减存储压力（同 KV-Cache，更多计算）。
- 然算力密度增长更速——应用层迅速吸纳释放之算力。
- **1-2 年后，存储需求复强**——非 MTP 失效，乃算力增长超前 HBM 带宽增长故也。

> MTP 不毁访存墙——**将其推迟至更高算力基线**。

---

## 八、片外介质：DDR 之反直觉利

一结构性可能：**DDR 或反受其利**。

逻辑：MTP 降 HBM 带宽依赖 → 然模型容量需求日增（更长序列、更大模型）。若 HBM **容量**成瓶颈，DDR 作为容量层遂得更重要。推理系统或从"HBM 独大"转向"HBM + DDR"分层存储。

此非定论，乃**结构性开口**——MTP 改变存储层级之间边际替代率。

---

## 九、Token 成本与序列长度：加速演进

| 维度 | 当前（2026 中） | 预测（2027） |
|---|---|---|
| 推理 token 成本 | 基准 | ==2-4× 下降== |
| 主流序列长度 | 256K | ==2M+== |

MTP 以更高算力利用率直接压低成本。DSpark（2026.06）称较 MTP-1 基线快 60-85%，SGLang 基准示 MTP 推测解码得 1.4× 吞吐量。

**长期趋势不改，唯节奏加快**：
- UB 低延迟 ✓
- 高带宽结构 ✓
- 大容量 HBM ✓
- 超低延迟互连 ✓
- 超长序列 ✓

> **此等方向不变——唯更快尔。**

---

## 十、路线之争：唯稀疏可扩展

通向百万序列者，两路焉：

| 路线 | 代表 | 存储成本 | 计算成本 | 多层介质友好 |
|---|---|---|---|---|
| **线性注意力** | Kimi3 (KDA) | 线性（然指数衰减） | 线性 + Full Attn | ✗ |
| **稀疏注意力 (DSA)** | DeepSeek V4, GLM52 | 稀疏可控 | 稀疏可控 | ✓ |

**Kimi3 之根本问题**：线性 + Full Attention 混合，仍意味著==存储与计算成本之不良扩展==。当前遗忘门机制下，线性注意力之有效存储容量随序列长度指数衰减（遗忘门信息损失累积）——"线性"之断言仅适用于短上下文质量，非百万 token 之有效保留。Kimi3 保留之 25% Full Attention 于大序列下仍属平方级成本。

> **于当前遗忘门范式下，唯稀疏注意力可亚线性扩展存储/成本。** 线性注意力中信息衰减之累积乃门控递归之数学性质，非工程差距——可缓解（如更优门设计、混合比例），然无法在不实质上转为稀疏之情况下根除。

Kimi 去岁末之技术报告亦承认此点：彼等认可后续需融进稀疏路线。Kimi3 之产品化径择线性注意力先行而已。

**预测**：至 2027 下半年，多数领先模型将汇于 DeepSeek 稀疏 + MTP 路线。

---

## 总结矩阵

| 维度 | 近期（1 年） | 中期（2-3 年） |
|---|---|---|
| **算力芯片** | 计算/KV 比上升 | 强度天花板逼近/超越训练 |
| **互连** | 分叉：小 MTP→延迟 / 深推测→带宽 | 拐点取决于 MTP 规模部署 |
| **超节点** | 大 EP 收益受抑 | 小域高带宽成核心资产 |
| **专用芯片** | 流芯片（LPX）窗口收窄 | SRAM 中心假设过时 |
| **片上介质** | 国产/中端上桌 | HBM 垄断溢价侵蚀 |
| **算力密度** | 加速堆叠 | 新 HBM 增长周期触发 |
| **访存墙** | 压力缓解（非消除） | 算力翻倍后复强 |
| **片外介质** | DDR 觅得分层角色 | 混合存储架构 |
| **Token 成本** | 2-4× 下降 | 2M+ 序列成标配 |

---

## 收尾

DeepSeek 之策略非偶然也——乃**系统设计方法论**焉：识成本结构中最贵之环，以算法创新破之，让市场跟随。较诸 Kimi3"堆精度"之路线，DeepSeek 之系统思维高出一境。

> **至明年或 2027 下半年，多数领先模型将汇于 DeepSeek 之谱。**

---

## 参考文献

<a id="ref-1"></a>**[1]** DeepSeek-V3 技术报告 — MTP 机制：1-2 层预测深度，序列因果链，0.1 loss scaling。[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

<a id="ref-2"></a>**[2]** NVIDIA Groq 3 LPX 架构 — 500 MB SRAM/LPU, 150 TB/s, 128 GB/机架, 35× 每瓦吞吐量。[NVIDIA Blog](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/)

<a id="ref-3"></a>**[3]** HBM 市场数据 — 2026 年预计 $58B, SK Hynix 市占 62%。[Introl](https://introl.com/blog/hbm-evolution-hbm3-hbm3e-hbm4-memory-ai-gpu-2025)

<a id="ref-4"></a>**[4]** HBM 代际 — HBM3: 819 GB/s → HBM3E: 1.2 TB/s → HBM4: 2.0 TB/s 每堆叠。GPU: H100 3.35 TB/s → B200 ~8 TB/s → Rubin ~22 TB/s。[Wikipedia](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)

<a id="ref-5"></a>**[5]** DeepSeek DSpark — 60-85% 推测解码提速（2026.06）。半自回归草拟 + 置信度调度验证；明确解决深层 MTP 堆叠之接受率衰减。[arXiv:2607.05147](https://arxiv.org/abs/2607.05147)

<a id="ref-6"></a>**[6]** SGLang 推测解码 — DeepSeek 模型 MTP 得 1.4× 吞吐量。[HPC-AI](https://company.hpc-ai.com/blog/sglang-speculative-decoding-tutorial)

### 相关阅读（本站）

- [百万序列：存储 vs 计算，谁是真瓶颈？](million-seq-storage-vs-compute.html)
- [Kimi3 架构分析：线性注意力、稀疏注意力与百万 Token 级的架构战争](kimi3-architecture-analysis.html)
- [Kimi3 成本效率：为何线性路线无法 Scaling Cost](kimi3-cost-efficiency.html)
