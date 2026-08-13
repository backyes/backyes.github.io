# 论文分析报告 ·《SkipKV: Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models》

> 文件：`/Users/backyes/Library/Mobile Documents/com~apple~CloudDocs/paper/mlsys2026/mlsys2026_papers/0EsV9SIm8p.pdf`
> OpenReview：https://openreview.net/forum?id=0EsV9SIm8p
> 会议：MLSys 2026 (9th MLSys Conference, Bellevue, WA)
> 代码：https://github.com/TTTTTTris/SkipKV

---

## 0. 元数据

- **标题**：SkipKV: Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models
- **作者团队**（page 1）：
  - Jiayi Tian (UCSB / Intel Labs，工作主体在 Intel 实习期间完成)
  - Seyedarmin Azizi、Erfan Baghaei Potraghloo、Massoud Pedram (USC)
  - Yequan Zhao、Zhengyang Wang、Zheng Zhang (UCSB)
  - Sean McPherson、Sharath Nittur Sridhar、Souvik Kundu (Intel Labs，通讯作者)
- **资助**：Intel Strategic Research Sectors (Systems Integration SRS & Devices SRS)（page 11）
- **页数**：19 页（正文 11 页 + 附录 8 页）
- **代码**：开源仓库 `TTTTTTris/SkipKV`（page 1 摘要末）
- **方法定位**：免训练 (training-free) 的 KV 缓存压缩 + 生成长度抑制框架，专注 Large Reasoning Models (LRMs) 的长 CoT 推理场景
- **测试模型**：DeepSeek-R1-Distill-Qwen-7B / 14B / Llama-8B（page 8）
- **测试数据集**：MATH-500、AIME-24、GSM8K、LiveCodeBench（page 8）
- **硬件**：单块 NVIDIA A100 (40GB) GPU（page 8）
- **关键数字**（摘要 + page 1）：
  - 与 SoTA (R-KV) 相比，准确率提升 **最高 26.7%**
  - 生成长度缩短 **最高 1.6×**
  - 吞吐量提升 **最高 1.7×**（同 batch size 比 R-KV）
  - 相对 FullKV 在多 batch 场景下吞吐量提升 **最高 9.6×**

---

## 1. TL;DR

SkipKV 是一篇瞄准 **大型推理模型 (LRM, 如 DeepSeek-R1, o1, QwQ)** 长链式推理 (CoT) 场景下 KV 缓存爆炸问题的 MLSys 2026 论文。它的核心信条是：**在 reasoning 模型时代，token 级 KV 驱逐 (eviction) 已经走到瓶颈——必须把"哪些 token 不写入 KV 缓存"和"哪些思考干脆不要生成"这两件事一起做。**

具体做法分三层：

1. **句子粒度 (sentence-level) 的 KV 驱逐**：用 LRM 自身最后一层 hidden state 的均值作为句子嵌入 (避免外接 sentence-BERT 推理成本)，计算 Pairwise Sentence Similarity (PSS)，对 PSS ≥ 0.95 的高度冗余句子整段驱逐；
2. **Adaptive Steering 抑制冗余生成**：维护一个 non-execution 思考计数器 N_o，把 steering 强度从固定 α₀ 改为 α_t = α₀ + γ·N_o，"思考越啰嗦、转向越狠"，从源头不让 LRM 生成冗余 token——这等价于"跳过 KV 生成"；
3. **Batch Grouping 修复多 batch 退化**：作者发现现有 KV eviction 方法（H2O, R-KV, SnapKV）在 multi-batch 设定下准确率塌陷，根源是 padding token 吃掉了"有效 KV budget"。SkipKV 按 prefill 长度排序后再分 batch，把 padding 几乎清零。

一句话：**SkipKV = 句子级 selective storage skip + 句子级 selective generation skip + batch grouping 三件套**，用计算和存储的"双省"代替传统"用计算换存储"，专门治 LRM 长 CoT 的过度思考 (overthinking)。

---

## 2. 问题背景

### 2.1 LRM 时代 KV cache 的双重压力

论文开篇 (page 1) 指出 LRM 与传统 LLM 的本质差异：**输出长度比理解任务高一个数量级**。具体数据：
- DeepSeek-R1-Distill-Llama-8B 解一道复杂数学题平均输出 **超过 32K tokens**（page 1）；
- batch size = 10 时，KV cache 占用约为 **模型权重的 2.5×**（page 1），KV cache 而不是模型权重成为了显存瓶颈；
- KV cache 在 decoding 阶段是 memory-bound 的，直接决定 throughput（page 1）。

这与传统的 long-context prefill 场景（KV 主要由输入 prompt 主导）有根本不同——LRM 是**长 decode** 主导。

### 2.2 现有 KV 优化方法在 CoT 场景的失败模式

论文系统梳理了三类前置工作 (page 2，§2 Related Works)：

**(a) Eviction（永久删除冗余 token）**：
- **H2O** (Zhang et al., 2023, NeurIPS)：基于累计注意力分数 (Heavy-Hitter)，保留 attention magnitude 高的 KV，**忽略语义连贯性**；
- **SnapKV** (Li et al., 2024)：在 prefill 阶段一次性压缩 KV，long-context 场景表现好，但 **CoT reasoning 准确率显著下降**；
- **ChunkKV** (Liu et al., 2025c)：把 token 分块再驱逐，提升 memory locality，但仍依赖 attention 分数；
- **R-KV** (Cai et al., 2025, NeurIPS)：是 SoTA，提出"redundancy-aware token scoring"，单 batch 下能压缩 80% KV、准确率几乎无损，**但论文发现它在多 batch 场景下崩盘**，且 **token-level 粒度破坏了高层语义结构、反而让生成更长**（page 2）。

**(b) Selection-based eviction（保留 FullKV 但每步只取片段）**：
- **Quest** (Tang et al.)：global memory 仍保留 FullKV，每个 query 取相关 chunk。**问题**：KV 总量随序列线性增长，throughput 受限、batch 不可扩展、长 CoT 直接 OOM（page 3）。

**(c) Quantization (KIVI, GEAR 等)**：
- 与 eviction 正交，但对 LRM 的 CoT 性能损失显著（Liu et al., 2025b 指出 quantization hurts reasoning，page 3）；本文不与之直接竞争，而是聚焦 eviction。

**(d) 其他降低 KV 内存的路径**：
- **KVsharer** (Yang et al., 2024)：层间 KV 共享，但 CoT 任务准确率掉很多；
- **DEER** (Yang et al., 2026)：early exit；
- **SEAL / Activation steering** (Chen et al., 2025a; Azizi et al., 2025)：在 latent 空间引导生成更精炼。**问题**：不能满足固定 KV budget，batch 扩展性也是 open problem（page 3）。

### 2.3 论文挖出的三个关键 Observation

§3 Motivational Case Studies (page 3-4) 是全文的"为什么 R-KV 不够"的三条证据：

**Observation 1（page 3）**：KV eviction 在 multi-batch 下准确率掉。  
- R1-Llama-8B + MATH-500，bs=1 vs bs=10，H2O 和 R-KV 在低 KV budget 下都显著掉分；
- 根因：variable-length prefill → padding 增加 → "有效 KV budget"被 padding 吃掉，且 padding 还扭曲 attention 分布。
- 例证：MATH-500 batch (bs=10) 内最长最短 prefill 差距能超过 400 tokens（Fig. 3 center）。

**Observation 2（page 4）**：KV budget 越紧，**生成越长**（不是越短！）。  
- R1-Qwen-7B + MATH-500：R-KV 在所有 KV budget 下生成的 token 数都比 FullKV 多（Fig. 3 right）；
- 根因：valid context 被驱逐后，模型"补偿性"地重复推理、再次验证。

**Observation 3（page 4）**：Token 级驱逐造成"碎片化"，引发 overthink。  
- Fig. 4 是一个非常直观的 case study：让模型算 $(1+2i)\cdot 6 - 3i$，正确答案 $6+9i$。R-KV 留下了答案的碎片 ($(6,9) \to (,9)$，$+12i-3i \to +2i$)，导致模型 **重复 self-validation 8 次、共生成 1517 tokens**。
- 根因：token-level redundancy 不感知语义边界，会从关键计算步骤里抠掉数字、留下答案碎片，反过来让模型反复"再确认一遍"。

这三个 observation 直接指向：**KV 驱逐必须粒度更粗——粗到句子级。**

---

## 3. 核心思想 / 方法

### 3.1 设计哲学：从"删 token"到"删句子 + 不让生成"

SkipKV 的核心两件事 (page 4 末)：
- **Skipping KV Storage**（§4.1）：把"删什么"从 token 提到 sentence；
- **Skipping KV Generation**（§4.2）：用 adaptive steering 让模型从源头少生成 non-execution 思考。

外加一个工程 fix：
- **Batch Grouping**（§4.3）：按 prefill 长度排序分 batch，消灭 padding。

### 3.2 句子级冗余的统计学动机（Observations 4 & 5, page 5）

在做方法前，作者补充两个核心观察：

**Observation 4**：不论答对答错，LRM 都会生成大量"高相似度句子"，但 **错答样本相似句占比是对答的 1.7×**。
- R1-Qwen-7B + AIME24：correct 51.3% vs incorrect 78.8%；
- R1-Llama-8B + AIME24：correct 44.3% vs incorrect 76.5%。

**Observation 5**：错答样本的 non-execution thoughts (反思/转折/元评论这种"非真正在算"的句子) 占比，是对答的 1.8×~2.6×（Fig. 5 bottom，page 5）。
- "Non-execution thoughts" 来自 SEAL (Chen et al., 2025a) 的定义，对应 "Wait", "Alternatively", "again" 这类关键词触发的句子（page 14 附录 A.1）。

**结论**：失败的推理 = 高比例重复 + 高比例 meta-思考。压住这两类句子，就是优化 LRM 推理的最大杠杆。

### 3.3 §4.1 Skipping KV Storage with Sentence-level Scoring（page 5-7）

#### 3.3.1 句子嵌入：用最后一层 hidden state，免外挂 sentence-BERT

朴素做法是用 sentence-BERT (Reimers & Gurevych, 2019) 编码每句，但这样在 decoding 阶段引入额外推理开销。SkipKV 的工程取巧：

> "Instead, we leverage the last-layer hidden state, denoted as $H \in \mathbb{R}^{bs \times N \times d}$, as latent contextual representations of the sentence segments."（page 5）

句子 i 的嵌入 = 该句 token 范围内 hidden state 的 mean：
$$v_i = \text{mean}(H[k]_{b_i:e_i}) \quad (page\ 6,\ Eq.\ 2)$$

#### 3.3.2 句子级 PSS (Pairwise Sentence Similarity)

定义见 page 4 末尾：
$$\text{PSS}(v_i, v_j) = v_i^\top v_j \quad (page\ 4,\ Eq.\ 1)$$

冗余集合：
$$P = \{i : \lambda_{i,j} > \tau, i \le j\}, \quad \lambda_{i,j} = v_i^\top v_j \quad (page\ 6,\ Eq.\ 3)$$

阈值 τ 取 0.95~0.99（附录 A.2，page 15）。当一对 (i, j) 相似度过高时，**保留较晚的 j、驱逐较早的 i**——这与 R-KV 删较晚 token 的策略相反，理由是较早句子更可能是"刚才说过的废话"，较晚句子可能是"刚收敛到的有效结论"。

#### 3.3.3 三合一的 cumulative eviction score（Eq. 6, page 6）

SkipKV 不是单纯句子驱逐，而是把句子相似度 λ_{i,j}、token importance I_α^{h_k}、token redundancy R^{h_k}（继承自 R-KV）合成最终分数：

$$
I_{\text{final}} =
\begin{cases}
\sigma I_\alpha^{h_k} - (1-\sigma) R^{h_k} - \lambda_{i,j}, & \text{if } i \in P \\
\sigma I_\alpha^{h_k} - (1-\sigma) R^{h_k}, & \text{otherwise}
\end{cases}
$$

其中：
- $I_\alpha^{h_k}$：观察窗口 α 上的 token attention importance（page 6, Eq. 4），处理 GQA 时要 maxpool query head 到 KV head；
- $R^{h_k}$：基于 normalized key 的 cosine similarity 求得的 token 冗余分数（page 6, Eq. 5），mask 掉 padding；
- σ 是权衡因子（附录 A.2 设为 0.1）。

**关键工程点**（page 6）："since similarity scores for redundant sentences (≥ 0.95) are typically an order of magnitude higher than token-level scores (~0.1), Eq. (6) ensures that highly redundant sentences are removed before token level eviction." 也就是说量纲上保证"句子级冗余"先被惩罚到底，token 级只在句子无冗余时起作用。

#### 3.3.4 KV Cache Sentence Range Monitoring（page 6-7）

这是工程实现的核心难点：**KV cache 一直在被驱逐，但句子边界是在原始 generation space 上记录的**。论文用一个映射函数 Φ 把 generation space (gs) 上的句子 span (b_i^{gs}, e_i^{gs}) 映射到 cache space (cs) 上的 (b_i^{cs}, e_i^{cs})，并维护一个 lookup table：

$$T \leftarrow \{(b_i^{gs}, e_i^{gs}) \mapsto \lambda_{i,j}\}, \forall i \in P \quad (page\ 6,\ Eq.\ 7)$$

每次驱逐前后都要更新 cs 中的 sentence range：
- **驱逐前 case (1)**：第一次压缩，cs 与 gs 等同（Eq. 9, page 7）；
- **驱逐前 case (2)**：之后每次有新句子加入，按公式 $b_i^{cs} \leftarrow e_{i-1}^{cs}+1, e_i^{cs} \leftarrow l_t^{cs} - \Delta$ 更新（Eq. 10, page 7）；
- **驱逐后 (Eq. 11, page 7)**：把 surviving token 的索引集合 P = {p_1, ..., p_B} 算出新的 (b_i^{cs}, e_i^{cs})——`b_i ← min{k | p_k ≥ b_i^{cs}_old}`、`e_i ← max{k | p_k ≤ e_i^{cs}_old}`，整段被清空就丢弃。

伪代码见 Algorithm 1（page 14）。这一段是 SkipKV 落地最 nontrivial 的部分，本质是在 **流式 paged attention** 风格的环境下，多维护一张"句子-缓存范围映射表"。

### 3.4 §4.2 Skipping KV Generation with Adaptive Steering（page 7-8）

#### 3.4.1 Steering vector 来源

继承自 SEAL (Chen et al., 2025a)：用 1000 个 MATH 训练样本（Hendrycks et al.），把 LRM 隐状态分成 execution / non-execution 两类，steering vector 取均值差：
$$v = H_E - H_O \quad (page\ 7)$$

Execution thoughts = "真在算"的 token；non-execution thoughts = "Wait, let me double-check..." 这种反思/重申。

#### 3.4.2 Adaptive 强度——本文新意

SEAL 用固定强度 α₀，而 SkipKV 维护一个 non-execution 计数器 N_o，自适应：
$$\alpha_t \leftarrow \alpha_0 + \gamma \cdot N_o \quad (page\ 7)$$

实现细节（附录 A.2，page 15）：
- 仅在遇到 newline delimiter 集合 D = {"\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"} 时注入 steering vector；
- 仅在某个特定 layer L_s 注入：R1-Qwen-7B/Llama-8B 用第 20 层，R1-Qwen-14B 用第 35 层；
- α₀ ∈ {1, 1.25}，γ = 0.02，结果 α_t ∈ [1, 2]，落在 SEAL 安全区间内；
- non-execution 关键词集合 N = {"Wait", "Alternatively", "again"}（page 14-15）。

效果（page 16，附录 A.4）：
- SkipKV 的 non-execution thoughts 数量从 FullKV 的 ~1000 降到 ~50（**约 4×~8× 减少**）；
- 比 SEAL 进一步少 1.3×~1.8×；
- 关键的是 **方差也大幅缩小**——sample-wise adaptive 让"难题转向更狠、易题转向更轻"。

> 设计哲学很有意思（page 7）："we treat this distinction as **statistical rather than causal**, and softly bias the model toward patterns more frequently associated with successful reasoning." 不是消灭 non-execution thoughts (它们其实有用)，而是统计学意义上向"对答样本的分布"靠近。

### 3.5 §4.3 Batch Grouping（page 8）

这是回应 Observation 1 的工程修复。Valid KV budget：
$$B' = B - \Delta_{\text{pad}}, \quad \Delta_{\text{pad}} = N_p^{\max} - N_p \quad (Eq.\ 12,\ page\ 8)$$

策略简单粗暴：
1. 把所有样本按 prefill 长度升序排；
2. 顺序切成 batch（每 batch 大小 bs）。

效果（page 11，Table 3）：MATH-500 + R1-Qwen-7B + B=768：
- 不分组：valid budget = 630/768 = 21%（理论 26%）
- 分组：valid budget = 759/768 = 25%（接近理论 26%）
- 准确率 +5.6%

这个方法在 vLLM 这种 continuous batching 框架下不需要（vLLM 没有 explicit padding，page 16），所以 vLLM 实验里只评估 sentence eviction 主体。

---

## 4. 实现 / 工程细节

### 4.1 训练-推理范式

- **训练-free**：不改模型权重，只在 decode 时干预；
- **基础 stack**：先在 HuggingFace + PyTorch + FlashAttention-2 上实现（page 9 throughput 实验明确"with FlashAttention-2 enabled for all methods"）；
- **vLLM v1 集成**：附录 A.3 (page 15-16) 介绍把 SkipKV 接入 vLLM v1（基于 R-KV 的 vLLM 实现扩展），适配 paged KV cache + continuous batching。

### 4.2 关键 hyperparameters（附录 A.2，page 15）

| 参数 | 值 | 说明 |
|---|---|---|
| 压缩间隔 ∆t | 128 decoding steps | 每 128 步做一次 sentence-level eviction |
| σ | 0.1 | token importance vs redundancy 权衡 |
| τ | 0.95 ~ 0.99 | 句子相似度阈值 |
| α₀ | 1 或 1.25 | steering 初始强度 |
| γ | 0.02 | steering 增量因子 |
| Steering layer | R1-Qwen-7B/Llama-8B = 20，R1-Qwen-14B = 35 | 单层注入 |
| Max gen length | GSM8K/MATH-500 = 8192, LiveCodeBench = 10000, AIME-24 = 16384 | （page 8） |

### 4.3 GQA 与 multi-head 支持

GQA 处理（page 6, Eq. 4）：每个 query head i 映射到 KV head g(i) = ⌊i/n⌋，attention importance 在 query head 上 max-pool 后才到 KV head 粒度。这一步保证 SkipKV 的 token importance 在 modern LRM (用 GQA) 上正确生效。

### 4.4 Padding mask M

Token redundancy R 中显式引入 attention mask：
$$\bar{K}^{h_k} = \frac{K^{h_k} \odot M}{\|K^{h_k} \odot M\|_2 + \epsilon} \quad (Eq.\ 5,\ page\ 6)$$
避免 padding token 污染 cosine similarity。

### 4.5 Sentence segmentation

完全用规则做：标点 + 换行（"\n\n" 等）。这是 LRM 自然换行的副产物。两个细节（page 14-15）：
- **合并连续 delimiter** 避免过度碎片化；
- **排除 syntax-critical delimiter**（如 ":\n"），防止 LiveCodeBench 上把 if/for 块切碎。

### 4.6 Algorithm 2（完整 SkipKV pipeline，page 14）

外层 while t < N：
1. 扫描当前 X，遇到 delimiter 就划句子 → 标 Non-exe / Execution；
2. 跑一步 forward → KV cache 增长；
3. 每 ∆t 步：调用 Algorithm 1 更新 KV cache 与句子范围；
4. 在指定层 L_s 注入 α_t · v 到 hidden state（仅 delimiter 处）；
5. 算句子级 PSS、刷新 redundant set T；
6. 数 non-execution 数 N_o，更新 α_t；
7. 自回归继续。

---

## 5. 评测

### 5.1 实验配置

- **模型**：R1-Qwen-7B / 14B / Llama-8B（DeepSeek-R1 蒸馏版）
- **数据**：MATH-500、AIME-24、GSM8K、LiveCodeBench
- **硬件**：A100 40GB 单卡
- **batch size**：R1-Qwen-7B/Llama-8B = 10；R1-Qwen-14B 取最大可装下的 batch
- **基线**：FullKV、H2O、R-KV（SoTA token eviction）、SEAL（SoTA steering）

### 5.2 主结果 1：准确率 (Fig. 9, page 8 + Fig. 13 附录, page 15)

跨 R1-Qwen-7B/14B 与三个 reasoning 数据集：

- **AIME-24（最难）**：
  - R1-Qwen-14B 在仅 15% KV budget 下，SkipKV 还能维持 FullKV 准确率；
  - 同 KV budget 下，SkipKV 比 R-KV 高 **最高 26.7%**（与摘要一致）；
  - R1-Qwen-14B + 41% KV budget，SkipKV 比 FullKV 高 **+10%**。
- **LiveCodeBench**：
  - R1-Qwen-7B 用 2× 更少 KV memory 时，准确率比 FullKV 高 **+5.2%**；
- **MATH-500**：
  - SkipKV 在所有 KV budget 下都超过 H2O / R-KV，且与 FullKV 持平甚至更高。

**结论**（page 9 末）：SkipKV 在 R1-Qwen-14B + AIME-24 上能用 **6.7× 更少 KV memory** 达到 FullKV 精度，所有模型/数据集组合中至少 **4× KV memory 节省**。

### 5.3 主结果 2：生成长度 (Fig. 10, page 9)

- 现有 token eviction (H2O, R-KV) 全部生成 **比 FullKV 还长** 的 token 序列（实锤 Observation 2）；
- SkipKV 比 FullKV 短 **最高 28%**；
- 比 R-KV 短 **最高 32% (R1-Qwen-7B), 39% (14B), 48% (Llama-8B)**——直接转化为 1.5×~2× 解码延迟下降。

### 5.4 主结果 3：吞吐量 (Table 1, page 9)

GSM8K + R1-Qwen-7B + A100 40GB + KV budget=512：

| Method | bs=10 latency↓ | bs=10 throughput↑ | bs=100 throughput↑ | bs=140 throughput↑ |
|---|---|---|---|---|
| FullKV | 324 min | 4.07 | OOM | OOM |
| SEAL | 178 min | 7.41 | OOM | OOM |
| R-KV | 227 min | 5.81 | 20.0 | 18.8 |
| **SkipKV** | **136 min** | **9.70** | **25.4** | **19.4** |

- SkipKV 比 FullKV 总加速 **9.6×**；
- 同 batch size 下比 R-KV 快 **最高 1.7×**（来自更短生成）；
- FullKV / SEAL 由于无固定 KV budget，bs=100 直接 OOM。

### 5.5 与 SEAL 对比 (Fig. 11, page 10)

KV memory 占用 vs 准确率（AIME-24）：

| Model | FullKV | SEAL | SkipKV |
|---|---|---|---|
| R1-Qwen-7B | 5.9 GB / 46.7% | 5.2 GB / 43.3% | **2.2 GB (2.7×) / 53.3%** |
| R1-Qwen-14B | 18.5 GB / 60.0% | 18.0 GB / 56.7% | **2.8 GB (6.6×) / 60.0%**, 7.5GB / 70.0% |
| R1-Llama-8B | 13.6 GB / 40.0% | 12.0 GB / 46.7% | **3.3 GB (4.1×) / 46.7%** |

**结论**：SEAL 只削生成长度 (~10%)，KV memory 节省有限；SkipKV 同时省存储 + 省生成，最高比 SEAL **6.6× 更小 KV memory + 13.3% 更高准确率**。

### 5.6 vLLM 系统集成实验 (Table 4, page 15)

GSM8K + KV budget = 512 + vLLM v1 + 不同 batch size：

| Metric | bs=64 | bs=32 | bs=16 | bs=8 |
|---|---|---|---|---|
| R-KV latency | 51 | 55 | 72 | 117 |
| SkipKV latency | **44** | **52** | **68** | **112** |
| R-KV acc | 73.8 | 81.7 | 85.3 | 87.1 |
| SkipKV acc | **84.4** | **86.8** | **88.2** | **88.6** |

R-KV 在大 batch 下准确率明显塌陷 (87.1 → 73.8)；SkipKV 稳定在 84.4+。说明 sentence-level 选择本身（不靠 batch grouping）已经天然鲁棒。

### 5.7 消融 (Table 2, page 10)

R1-Qwen-7B + AIME24，从 R-KV baseline 渐进加：

| 方案 | bs=20% acc | bs=37% acc | bs=20% length | bs=37% length |
|---|---|---|---|---|
| R-KV | 36.7 | 36.7 | 12000 | 11403 |
| + Sentence Scoring | 40.0 (+3.3) | 40.0 (+3.3) | 11332 (-6%) | 11819 (+4%) |
| + Adaptive Steering | 40.0 | 40.0 | 8860 (-26%) | 10041 (-12%) |
| + Batch Grouping (= SkipKV) | **50.0 (+13.3)** | **50.0 (+13.3)** | **9228 (-23%)** | **7988 (-30%)** |

- Sentence Scoring：温和提精度 (+3.3)；
- Adaptive Steering：长度暴跌（最高 -26%），但精度保持；
- Batch Grouping：精度大跳 +20，长度持续优化。

清晰的"局部语义筛选 → 全局 decoding 协调"的 hierarchical design。

### 5.8 Batch Grouping 单项验证 (Table 3, page 10)

R1-Qwen-7B + MATH-500，对比 valid budget 与精度：
- B=768: avg valid 21% → 25%, acc 77.8 → 83.4 (+5.6)；
- B=1024: 30% → 34%, acc 85.2 → 86.4；
- B=1536: 47% → 51%, acc 86.0 → 88.0 (+2.0)。

KV budget 越紧，分组收益越大——印证"padding 浪费"在低 budget 下放大。

### 5.9 句子级行为分析 (Fig. 14, page 16)

R1-Qwen-7B 三个数据集上的 non-execution thought ratio 与 high-similarity sentence ratio 箱线图：
- SkipKV 比 FullKV 的 non-execution 减少 **4× (correct), 8× (incorrect)**；
- 比 SEAL 进一步少 **1.8×, 1.3×**；
- 相似句减少：R-KV 只削 11%，SkipKV 削 **2× 以上**。

### 5.10 定性案例 (Fig. 15-16, page 17-19)

同一 MATH-500 题目（展开 $x(x(1+x)+2x)-3(x^2-x+2)$）：
- R-KV：5 次 re-validate，2125 tokens；
- SkipKV：3 次 re-validate，1742 tokens（**-18%**）。

可视化显示：R-KV 倾向驱逐"碎片 token"和"答案区域 token"，触发 re-validation；SkipKV 整段移除冗余句、保住答案完整。

---

## 6. 思想精读 / 启示

### 6.1 LRM 时代的 KV 革新：从 prefill-bound 到 decode-bound

传统 long-context 优化（H2O / SnapKV / Quest）瞄准的是 **prefill 阶段**的长 prompt——KV 主要由输入决定。LRM 颠覆了这个假设：
- 输入 ≤ 1K token，输出 16K~32K token；
- KV 增长 = decoding 增长，是**流式增长**，每一步都要重新决定保留谁。

SkipKV 的设计——周期性 (∆t=128 steps) 重压缩 + cache range monitoring + adaptive steering——本质上是为这个"decode-dominant"模式重新设计的。可以预见 MLSys 26-27 这条线会有更多工作（ThinKV (Ramachandran et al., 2026)、DEER (Yang et al., 2026)、ChunkKV、RoCKETKV 都是同期）。

### 6.2 "用计算换存储" vs "两个都省"

| 范式 | 代表 | 计算 | 存储 |
|---|---|---|---|
| 用计算换存储 | KIVI (量化), KVsharer (跨层共享), Quest (selective load) | ↑ | ↓ |
| 用存储换计算 | FullKV cache | ↑↑ | ↑↑ |
| 用准确率换两者 | H2O / SnapKV / R-KV | ↓ | ↓ |
| **两者都省 + 准确率↑** | **SkipKV** | ↓ (生成短) | ↓ (KV 小) | ↑ (语义连贯) |

SkipKV 突破了"trade-off triangle"，关键在于它**承认并利用了 LRM 的 overthinking 倾向**——很多 token 本来就不该生成、本来就不该存。这是 reasoning 模型时代独特的 free lunch。

### 6.3 与 SHIP / SuperInfer / LEANN 等的存储维度对照

把 MLSys 26 数据存储相关工作摆一起看：

| 工作 | 存储对象 | 优化粒度 | reasoning 友好度 |
|---|---|---|---|
| **SkipKV** | KV cache (decode-time) | 句子级 | 原生 |
| ThinKV (ICLR 26) | KV cache | "thought" 级 | 原生 |
| SHIP / SuperInfer | weight offload | layer/expert | 不针对 |
| LEANN | embedding store | vector chunk | 不针对 |
| ChunkKV | KV cache (prefill) | chunk | 部分 |

SkipKV 的差异化定位：它是 **decode-time** + **语义粒度** + **生成抑制联动** 三者交集——而其他存储层工作多在 prefill 或权重层面。

### 6.4 句子粒度的语言学合理性

为什么句子级有效？因为 reasoning chain 的最小**自包含语义单元**是句子（或者说 reasoning step）：
- 数学推导一步 = 一句
- 代码注释一行 = 一句
- "Wait, let me double-check" = 一句完整 meta-thought

token 级别既感知不到语义边界，又会破坏数字串、答案括号这种紧密结构。SkipKV 用 LRM 自身的换行习惯做天然分词，这是 LRM 时代独有的便利：训练得越好，换行越规整。

### 6.5 Adaptive 强度：从静态干预到反馈控制

SEAL 用固定 α，把 steering 当 open-loop 控制；SkipKV 用 N_o 计数器组成 closed-loop——"模型越啰嗦，干预越强"。这是从控制论视角的升级，**也是 inference-time scaling 时代的一个范式**：用运行时观测信号反过来调推理过程，而不是事先设死所有 hyperparameter。

### 6.6 对系统层的启示

1. **多 batch 是真问题**：以前 KV eviction paper 普遍只报 bs=1 准确率，论文揭示 bs>1 是普遍痛点。这给所有 KV 优化工作立了新 benchmark；
2. **vLLM 友好**：sentence eviction 与 paged attention 兼容（page 16 已验证）；
3. **句子映射表是新的 metadata 开销**：未来如果要做硬件加速，这张表本身的更新逻辑（Eq. 9-11）可能是新的瓶颈点。

---

## 7. 局限与开放问题

### 7.1 论文自陈 (§7 Discussion, page 11)

- **Batch grouping 在 prompt 长度高度异质 / 极长 context 下退化**：仍有 padding overhead；
- 作者承认 "dynamic grouping strategies that adapt to prompt-length variability" 留作 future work。

### 7.2 我的补充观察

1. **句子分割完全靠规则**：换行/标点为锚。对中文 / 多语种 / 数学公式跨行 / 代码风格剧烈变化的场景可能不鲁棒。论文虽提到 LiveCodeBench 排除 ":\n"，但没系统评估非英文。

2. **τ ∈ [0.95, 0.99] 的人工调参敏感度未充分披露**：每个 dataset/model 是否要重新 tune？文中数据是否都是同 τ？

3. **Steering layer 的选择 (20 层 / 35 层) 也是 hand-pick**：迁移到非 DeepSeek-R1 蒸馏模型 (e.g., QwQ, o1, Mistral Reasoning) 时这个 layer 怎么定？

4. **Adaptive γ = 0.02 是经验值**：N_o 大到一定程度后 α_t 会饱和到 ~2，超过 SEAL "safe regime"会崩。论文 (page 16) 也明确 α_t ∈ [1,2] 才安全。

5. **对 KV quantization 是否真的正交**：论文断言 quantization 与 eviction 正交，但量化+ eviction 的累积误差未实证。

6. **句子嵌入 = last-layer hidden state mean**：这是工程取巧，但对极长句子（数学公式整段），mean pooling 是否过度平滑？没有 ablation 比 sentence-BERT。

7. **代码生成 (LiveCodeBench) 的语义边界是否真适合"句子"粒度**：代码块本身有强结构，按 "\n" 切可能切碎逻辑块。论文只提了排除 ":\n"，但 if/else 内部其他换行未必无害。

8. **多 batch 中 batch grouping 的"延迟"问题**：sort 整个 dataset 在 online serving 是不可能的，需要 dynamic grouping 算法（论文已承认）。在 vLLM 这种 continuous batching 框架下完全不用——这倒是个洗脱口实。

9. **Steering 与 RLHF / DPO 训练目标的兼容性**：注入 v 改变了 hidden state，对下游 alignment 是否有副作用？未涉及。

10. **从 R-KV 借的 token redundancy R 是否仍必要**：消融表显示 sentence scoring 单独只 +3.3%，主要贡献来自 adaptive steering。如果 R 移除，纯句子级是否够用？没做。

### 7.3 更长远的开放问题

- **Sentence-aware 是否能扩展到 reasoning 之外**？比如 agentic 长对话、多轮 tool-use？
- **能否把 SkipKV 训进模型？** 训练时奖励"句子级稀疏化"，而非推理时硬切。
- **与 speculative decoding / MoE inference 的协同**：当 decode 本身被 speculate，SkipKV 的句子边界检测如何配合？

---

## 8. 关键术语速查表

| 术语 | 中文 | 含义 | 来源页 |
|---|---|---|---|
| **KV cache** | KV 缓存 | autoregressive decoding 中存储历史 key/value 的张量，长度随 token 线性增长 | page 1 |
| **Paged Attention** | 分页注意力 | vLLM 提出，把 KV cache 切成定长 page，便于动态分配 | page 12 (Kwon et al. 2023) |
| **Token Eviction** | Token 驱逐 | 永久从 KV cache 删除某些 token，释放显存 | page 2 |
| **Heavy-Hitter (H2O)** | 高频命中 | 累计 attention score 高的 token 优先保留 | page 1 |
| **Selection-based Eviction (Quest)** | 选择式驱逐 | KV 留全集，每次 query 只 load 相关 chunk 到 local memory | page 3 |
| **R-KV** | 冗余感知 KV 压缩 | 用 token 间余弦相似度检测冗余 | page 2 |
| **Sparse Attention** | 稀疏注意力 | 对 query-key 矩阵稀疏化，可与 KV 压缩正交 | - |
| **CoT (Chain-of-Thought)** | 思维链 | 让模型显式生成中间推理步骤 | page 1 |
| **LRM (Large Reasoning Model)** | 大推理模型 | 输出长 CoT 的 LLM (R1, o1, QwQ) | page 1 |
| **PSS (Pairwise Sentence Similarity)** | 句对相似度 | 两句嵌入向量内积 | page 4-5 |
| **Execution Thought** | 执行型思考 | 实际进行计算/推导的句子 | page 5 |
| **Non-execution Thought** | 非执行型思考 | 反思/转折/自我验证句 ("Wait...","Alternatively...") | page 5 |
| **Steering Vector** | 引导向量 | 在 hidden state 上加的方向向量，引导模型生成偏好分布 | page 7 |
| **Adaptive Steering** | 自适应引导 | 强度随 non-execution 计数动态变化 | page 7-8 |
| **GQA (Group-Query Attention)** | 分组查询注意力 | 多个 query head 共享一个 KV head，reduce KV 内存 | page 6 |
| **Effective KV Budget** | 有效 KV 预算 | 总 budget 减去 padding 占用 | page 3, 8 |
| **Batch Grouping** | Batch 分组 | 按 prefill 长度排序后分 batch，最小化 padding | page 8 |
| **Cache Range Monitoring** | 缓存范围监控 | 跨驱逐步骤维护 generation-space 与 cache-space 句子边界映射 | page 6-7 |
| **Continuous Batching** | 连续批处理 | vLLM 的动态请求合批，无 explicit padding | page 16 |
| **Overthinking** | 过度思考 | LRM 在 KV 受限时反复 re-validate，反而生成更长 | page 1, 4 |

---

## 9. 关键页码索引

| 主题 | 页码 |
|---|---|
| Abstract / 总览数据 (26.7%, 1.6×, 1.7×) | page 1 |
| 摘要图 Fig. 1：SkipKV 在 R1-Qwen-14B AIME-24 上的 acc-memory 帕累托 | page 1 |
| 与 H2O / SnapKV / R-KV 的图示对比 (Fig. 2) | page 2 |
| 文献综述：H2O / SnapKV / ChunkKV / Quest / KIVI / GEAR / KVsharer / DEER / SEAL | page 2-3 |
| Observation 1：multi-batch 准确率塌陷 (Fig. 3 left) | page 3 |
| Observation 2：低 KV budget 反而生成更长 (Fig. 3 right) | page 4 |
| Observation 3：token 碎片化引发 re-validation (Fig. 4 case) | page 4 |
| PSS 定义 (Eq. 1) | page 4 |
| Observation 4: 错答样本相似句多 (Fig. 5 top) | page 5 |
| Observation 5: 错答样本 non-exec thought 多 (Fig. 5 bottom) | page 5 |
| SkipKV 架构图 (Fig. 6) | page 5 |
| 句子嵌入 = mean(hidden state)，Eq. 2 | page 6 |
| 冗余集合 P 定义 (Eq. 3) | page 6 |
| Token importance Eq. 4 (含 GQA 处理) | page 6 |
| Token redundancy Eq. 5 (含 padding mask) | page 6 |
| Cumulative eviction score Eq. 6 (核心打分) | page 6 |
| Cache range mapping Φ + lookup table T (Eq. 7-8) | page 6 |
| Cache range monitoring 流程图 (Fig. 7) | page 6 |
| Pre-eviction 更新 Eq. 9-10 | page 7 |
| Post-eviction 更新 Eq. 11 | page 7 |
| Steering vector v = H_E - H_O | page 7 |
| Adaptive 强度 α_t = α_0 + γ·N_o | page 7 |
| Batch grouping 示意 (Fig. 8) + Eq. 12 | page 8 |
| 主要 accuracy 实验 (Fig. 9) | page 8 |
| 主要 length 实验 (Fig. 10) | page 9 |
| Throughput 表 (Table 1) | page 9 |
| 与 SEAL KV memory 对比 (Fig. 11) | page 10 |
| Ablation 表 (Table 2) | page 10 |
| Batch grouping 效果可视化 (Fig. 12) | page 10 |
| Batch grouping 数值表 (Table 3) | page 10 |
| Conclusion §6 | page 11 |
| Discussion §7 (局限) | page 11 |
| Algorithm 1 (Storage skip 伪代码) | page 14 |
| Algorithm 2 (完整 SkipKV 伪代码) | page 14 |
| Delimiter set D / non-exec keyword set N | page 14-15 |
| 超参数表 (附录 A.2) | page 15 |
| R1-Llama-8B 补充实验 (Fig. 13) | page 15 |
| vLLM 集成实验 (Table 4) | page 15-16 |
| 句子级行为分析 (Fig. 14) | page 16 |
| MATH-500 R-KV vs SkipKV case study (Fig. 15-16) | page 17-19 |

---

## 10. 一句话点评

**SkipKV 给"reasoning 模型推理优化"贡献了一个三件套：句子级 KV 驱逐 + adaptive steering 抑制冗余生成 + batch grouping 修复 multi-batch 退化——它正确地识别出 LRM 时代的 KV 优化是"decode-bound 而非 prefill-bound"，并用语义粒度替代 token 粒度，使"省存储"和"省计算"在 overthinking 这个共同根因上同时被治愈，是 MLSys 2026 reasoning 推理系统方向的一篇代表性工作，值得作为后续 ThinKV / RaaS / DEER 等同方向工作的对照基线。**
