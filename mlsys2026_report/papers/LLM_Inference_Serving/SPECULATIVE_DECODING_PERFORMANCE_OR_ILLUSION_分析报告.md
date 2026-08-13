# 论文分析报告 ·《Speculative Decoding: Performance or Illusion?》

> "SD 在 batch size = 1 上的 2× speedup，到了 production batch size = 128 的 vLLM 上还剩多少？这篇 MLSys 2026 论文用一句话回答：**剩下的远比社区想象的少**。"

---

## 0. 元数据

- **论文标题**：Speculative Decoding: Performance or Illusion?
- **作者**：Xiaoxuan Liu\*, Jiaxiang Yu\*, Jongseok Park, Ion Stoica, Alvin Cheung（UC Berkeley，Sky Computing Lab）
- **会议**：MLSys 2026（Proceedings of the 9th MLSys Conference, Bellevue, WA）
- **OpenReview**：https://openreview.net/forum?id=fzkqtezFEi
- **代码与数据**：https://github.com/orgs/SpecDecode-Bench/repositories（包含 vLLM profiling suite + 独立 simulator，pre-profiled traces 已开源）
- **资助**：NSF IIS-1955488 / IIS-2027575；DOE DE-SC0016260 / AC02-05CH11231；DARPA HR00112590131；NVIDIA DGX gift。
- **类别**：LLM inference / serving / speculative decoding 的 critical empirical study
- **页数**：23 页（含 Appendix 与 Artifact Appendix）
- **关键词**：speculative decoding (SD), vLLM, EAGLE/EAGLE-3, n-gram (prompt lookup), draft-model based SD, MTP, acceptance length, batch size scaling, theoretical upper bound, oracle proposer, reasoning workload, code-editing, BLEU-n
- **学术定位**：
  - 是首篇在 production-grade engine（vLLM v0.10.1.1）上系统化、跨 batch size、跨模型规模、跨 SD 流派的 SD 评测论文（page 1, page 2, page 16）。
  - 与同类论文（Medusa, EAGLE 系列, SpecInfer, Lookahead, REST 等）"宣告 2–3× speedup"的乐观结论形成强烈对照——这是**反主流（contrarian）立场**的一篇 critical study。

---

## 1. TL;DR — SD 真的快？还是 illusion？

**论文核心立场**：SD **几乎从不真的"打回原形"**（每个 SD 变体在每个 workload 上都能跑赢 no-SD 基线，page 1 第二段），但其常被宣传的"2–3× speedup"在 production 部署中是**部分 illusion**：

1. **"Illusion 来自 batch size = 1 的研究原型"**（page 2，"Problems of Existing Benchmarks"）。当 batch size 从 1 提升到 production 常见的 128，speedup 可以从 1.96× 退化到 1.72×（Llama3-70B + EAGLE，page 4），更大模型退化更剧烈。
2. **Verification 才是 SD 的主导成本**（page 7）。Drafting 在 n-gram 下 < 2%，在 EAGLE 下 < 20%，但 verification 在大 batch 下吃掉 42%–95% 的执行时间——所以 SD 的整体收益受制于 target model 的 forward pass，而不是 draft model 有多好。
3. **观测到的 speedup 与理论上界差距巨大**（page 11–12，Section 8）。在 InstructCoder + Llama3.1-8B + n-gram + bs=1 上，oracle 上界是 2.75×，最佳固定 k 只有 2.1×，adaptive 也只到 2.3×。
4. **不同 SD 变体互补**（page 11–12，Section 8.2）：把 EAGLE-3 与 n-gram 在 token 级别 oracle-combine，可以再额外拿到 1.6× headroom，端到端最高 4.9×。
5. **Reasoning workload 是 SD 的少数甜点**（page 5–6）：在 GPQA-Main / AIME 上，由于生成长、局部重复多，n-gram 与 EAGLE-3 都能拿到 1.5×–1.8×。

> **一句话答案**：SD 不是骗局，但社区里"2× speedup 是免费午餐"的认知是 **production illusion**——真实生产环境下，speedup 强烈依赖 batch size、target model 大小、workload 的 prompt-output 重叠度，及 SD 变体与 model pair 的匹配度。

这种"带数据的怀疑论"风格在 MLSys 社区很罕见，论文并不主张废弃 SD，而是要求**fair comparison + production setting + 公平 metric（throughput, not bs=1 latency）**。

---

## 2. 问题背景

### 2.1 SD 流派井喷与"实验数据通胀"

自 2023 年 Leviathan et al. 与 Chen et al. 同时提出 SD 后（page 2），三年间 SD 文献井喷：

| 流派 | 代表 | 特点 |
| --- | --- | --- |
| Draft-model-based | SpecInfer (Miao 2023), Online SD (Liu 2023), DistillSpec (Zhou 2023), Sequoia (Chen 2024) | 用一个独立的小 LLM 作为 draft |
| Draft-model-free / heads | Medusa (Cai 2024), EAGLE (Li 2024b), EAGLE-2 (Li 2024c), EAGLE-3 (Li 2025), BiTA (Lin 2024) | 在 target model 顶部加 auxiliary heads |
| Co-trained | MTP / Multi-Token Prediction（DeepSeek-V3 Liu 2025；GLM-4.5 Zeng 2025） | head 与 main model 联合训练 |
| Tree-based | Medusa, EAGLE, SpecInfer, Lookahead (Fu 2024) | 用 token tree 提高 verify 利用率 |
| Non-LLM proposer | n-gram / Prompt Lookup (Saxena 2023; PLD+ Somasundaram 2024), REST (He 2024) | training-free，从 prompt / corpus 中检索 |

每篇论文都报告 1.5×–4× speedup，但**评测条件高度不一**（page 2, page 3）：
- 大量论文在 batch size = 1 上做评测（Cai 2024, Leviathan 2023, Li 2024b/c/2025, Xia 2024）；
- 缺乏 CUDA Graph、continuous batching、chunked prefill 等 production 优化；
- 仅报告 latency 或 dataset-level acceptance rate，没有 time/memory breakdown 与 position-level acceptance；
- 跨变体之间用各自的实现栈，几乎不可能 apple-to-apple。

### 2.2 学术报告 vs production 体感落差

工业界的 SD 部署体感与论文宣传脱节（page 1）：
- vLLM、SGLang 默认开启 KV cache 管理 / continuous batching / CUDA Graph，prototype 级 SD 集成进去就立刻退化；
- 大 batch 下系统 compute-bound，剩余 compute 不再"免费"，被 SD 拒绝的 token 等于在烧钱；
- draft-model-based SD 在小 target model 上，draft 的相对成本甚至高达 target forward 的 37.5%（Qwen3-0.6B vs Qwen3-8B，page 5）；
- 不同 workload 的 acceptance 行为差异比论文给出的均值大得多。

### 2.3 为什么需要一篇"批判性"论文

作者明确指出 SD 评测三大顽疾（page 3）：

1. **Prototype gap**：原型实现没有 CUDA Graph 等优化（page 3）。
2. **Batch-size gap**：bs=1 评测不代表真实部署的 bs=64–128。
3. **Metric gap**：仅有平均 latency / 平均 acceptance rate，没有进入 SD 内部的时间分解与 acceptance 的位置分布。

因此，论文提出的研究问题是：**"SD 在 vLLM-级 production 系统中、在多种 batch size 下、在多个 SD 变体上的真实表现是什么？理论上界是多少？我们离上界还有多远？"**

这与 SambaNova 等厂商高调宣传的"Dataflow Is All You Need：SN40 上 SD 6× 加速"形成鲜明对照——后者本质上属于 dataflow 架构上"同步税"较低带来的有利场景，论文则在 GPU 上揭示 SD 的"同步税"成本。我们将在 Section 6 中进一步讨论这一对照。

---

## 3. 核心思想 / 方法

论文不是提出新算法，而是建立一套**"production-grade SD 评测框架 + 上界分析框架"**。其核心思想可归纳为三条主线：

### 3.1 "Fair comparison" 评测原则

(page 3, Sec 3.1; page 21, Tab 5；page 21, Tab 6)

- **共同 inference engine**：所有 SD 变体集成进 **vLLM v0.10.1.1**（少数 tree-style 实验用 SGLang v0.5.9，page 16）。
- **共同硬件**：8B 模型在 1×H100-80GB；70B / 106B 在 4×H100 with TP=4。
- **统一 SD 提议长度**：所有 SD 变体都 propose 3 tokens/step（n-gram 额外测 5 tokens 以研究 k 的影响）。
- **温度 = 0、top-p=1、top-k=-1**（贪心解码）；max gen length = 8192（reasoning 任务为 32768）。
- **统一 fair workload**：6 个数据集（CNN/DailyMail、ShareGPT、InstructCoder、GSM8K、AIME22-24、GPQA-Main）覆盖 summarization、chat、code-edit、math、scientific reasoning。
- **统一 metric**：**throughput（tokens/sec）而不是 latency**——这是论文最关键的方法论选择之一（见 §3.3）。

### 3.2 三阶段时间分解 + 内存分解

(page 7, Sec 5；Tab 2)

论文把 SD 单步分为四个阶段：
1. **Drafting**：n-gram lookup 或 EAGLE 的 autoregressive head 推理；
2. **Verification**：target model 在 k+1 个候选上的 forward pass；
3. **Rejection sampling**：根据 verify logits 决定接受多少；
4. **Other overheads**：vLLM 调度、KV 维护等。

memory 维度则区分：
- **Static memory**（额外参数，如 EAGLE head 或独立 draft model 权重）；
- **Per-token KV cache**（额外的 attention layer 带来的 KV）；
- **不计 activation tensors**（保守估计）。

### 3.3 公平 metric：为什么是 throughput 而不是 latency

(page 4，Tab 1)

作者发现：**即使在贪心解码、相同 prompt 集合下，no-SD 与 SD 生成的 token 数量也并不相等**——例如 ShareGPT + Llama3.1-8B：
- bs=1：no-SD = 737±1048，n-gram = 750±1100；
- bs=64：no-SD = 687±852，EAGLE = 709±924；
- bs=128：no-SD = 714±930，EAGLE3 = 709±924。

这种"非确定性"来自 He (2025, "Defeating nondeterminism in LLM inference") 揭示的 kernel-level / FP nondeterminism。如果用 latency 作为 metric，会被 token 数差异污染。所以论文一律用：

```
speedup = throughput_with_SD / throughput_no_SD
```

**这是论文方法论上少有的"反直觉但 production-correct"的设计**——绝大部分 SD 论文用 latency。

### 3.4 端到端 speedup 公式（来自 Leviathan）

(page 6, Eq.(1))

$$E(\text{speedup}) = \frac{1 - \alpha^{k+1}}{(1 - \alpha)(kc + 1)}$$

- $\alpha$：token acceptance rate；
- $k$：每步 propose 的 token 数；
- $c$：drafting 与 target forward 的执行时间比。

论文借此把 SD 的 speedup 拆成两个**独立 axis**：执行效率（$c$、verification 占比）+ acceptance 行为（$\alpha$、长度分布）。后续每个实验都可以映射到这两个 axis 上。

> 重要警告（page 6, footnote 4）：该公式的前提是 "k+1 个 token 的 verification ≈ 单 token forward 的成本"，**只有在 memory-bound 区间（小 batch）下成立**；compute-bound 时该公式高估 SD 收益——这是论文揭示 SD illusion 的理论根基。

### 3.5 理论上界：Oracle Proposer + Oracle Combiner

(page 11–12, Sec 8)

论文构造两层 oracle：

1. **Oracle proposed length**：假设每一步都"先知"地知道实际能被接受的 token 数 $L^*$，把 $k$ 设为 $L^*$，从而**消除 verification 上的浪费**。给出 SD 单变体的上界。
2. **Oracle Combine**：在每个 token 位置，挑选 EAGLE-3 与 n-gram 中接受长度更长的那一个，再用 oracle proposed length。给出**多变体融合**的上界。

为了估算这两条上界，作者构造了一个**simulator**（基于真实 profiling traces，独立于 vLLM，在普通 laptop 上即可重放，page 22 Artifact Appendix）。这是论文输出的"second class of artifacts"，也是后续社区可以接续的研究基座。

---

## 4. 实现 / 工程细节

### 4.1 测试平台

(page 3, Sec 3.1；page 21, Tab 5)

- **Inference engine**：vLLM v0.10.1.1 默认开启 KV cache management、continuous batching、chunked prefill、CUDA Graphs；少数 tree-style 实验用 SGLang v0.5.9（page 16）。
- **GPU**：NVIDIA H100-80GB；8B 单卡，70B / 106B 用 4 卡 TP=4。
- **精度**：FP16。
- **采样**：temperature=0, top_p=1, top_k=-1（贪心，方便对比）。
- **batch size**：1 / 16 / 32 / 64 / 128（reasoning 由于 KV 压力降到 1–16 或 1–4）。

### 4.2 SD 变体实现

(page 3 第二栏；page 21 Tab 6)

- **Draft-model-based**：
  - Llama3-70B-Instruct + Llama3.2-1B-Instruct（c≈0.125）；
  - Qwen3-8B + Qwen3-0.6B（c≈0.375）；
  - 注：Llama3.1-8B 缺乏小 vocabulary 兼容的小 draft，所以未做 draft-model 对比。
- **EAGLE / EAGLE-3**：使用官方权重 yuhuili/EAGLE-LLaMA3.1-Instruct-8B、yuhuili/EAGLE3-LLaMA3.1-Instruct-8B、yuhuili/EAGLE-LLaMA3-Instruct-70B、AngelSlim/Qwen3-8B_eagle3。EAGLE-3 在 Llama-3-70B 上无官方权重，故缺测；EAGLE 在 Qwen3-8B 上无官方权重；GLM-4.5-Air 上没有 EAGLE/EAGLE-3。
- **MTP**：用 GLM-4.5-Air-106B 自带的 MTP head（vLLM v0.11.1rc1），Section A.2.1 显示 MTP 只发布了第一层 head 因而被自递归复用，限制了 acceptance（page 16）。
- **n-gram**：prompt lookup max=7、min=3，3 tokens/step，附加 5 tokens/step 实验。
- **Tree-style verify**：SGLang 中 k=3 (chain), k=6 (depth=3, branch=2), k=21 (depth=3, branch=4，对应 EAGLE-style tree)；FlashAttention-3 chain，FlashInfer tree。

### 4.3 内存核算（page 17, A.3）

论文给出 EAGLE / EAGLE-3 / draft-model 静态内存与 per-token KV 的解析公式：

$$M_{\text{static}} = (P_{\text{target}} + P_{\text{draft}}) \cdot 2 / 2^{30}\ \text{GiB}$$

$$M_{\text{KV/token}} = L_h \cdot 2 \cdot n_{kv} \cdot d_{\text{head}} \cdot 2 / 2^{10}\ \text{KiB}$$

例：Llama3-70B + Llama3.2-1B → 静态 133.7 GiB，per-token 352 KiB（vs 320 KiB 无 SD，+10%）；Qwen3-8B + Qwen3-0.6B → per-token 144→256 KiB（+77%！）——揭示了**小模型 draft-pair 的 KV cache 代价被严重低估**。

### 4.4 Profiling 与 Simulator 工程

(page 22 B.1, B.7)

- **vLLM 分支**：每个 SD 变体一个独立分支与独立 conda env，`rebuild_env.sh` 自动构建；
- **Profiling 流程**：warmup → 多 dataset × 多方法 → 自动绘图，单 8B 模型完整跑一次 24–36 小时；
- **Simulator**：保留 acceptance traces 离线，30–40 分钟可重放 oracle 上界与 combined-proposer 实验，无需 GPU——这对社区复现非常友好。

### 4.5 Acceptance 测量协议

(page 8, Sec 6)

为衡量"位置可接受最大长度"：
- 每步 propose 上限 20 tokens（远超 production 默认的 3）；
- 记录被 verify 接受的 token 数；
- **每步只前进一个 token，丢弃多余的**——把"多步 acceptance 累积"解耦为"位置级最大可接受长度"，便于跨变体公平比较。

> 注意（page 8 footnote 5）：实际生成 token = accepted draft + 1 bonus token，所以 chain-style SD 即便所有 draft 全被 reject，也能至少前进 1 个 token。

---

## 5. 评测

### 5.1 端到端 speedup（非 reasoning）

(page 5, Fig 1)

| 模型 | 数据集 | bs=1 最佳 | bs=128 最佳 | 衰减 |
| --- | --- | --- | --- | --- |
| Llama3.1-8B | GSM8K | EAGLE 1.73× | EAGLE 1.21× | 大 |
| Llama3-70B | ShareGPT | EAGLE 1.96× | EAGLE 1.72× | 中 |
| Llama3.1-8B | InstructCoder | n-gram-5 ~2.3× | n-gram ~1.3× | 大 |
| Llama3-70B | InstructCoder | Draft-Model ~2.6× | n-gram ~1.5× | 中 |
| Qwen3-8B | GSM8K | EAGLE-3 ~2.0× | ~1.2× | 大 |

**关键发现**：
- 每个 (model, dataset, bs) 配置下，每个 SD 变体都比 no-SD 快（speedup ≥ 1）；
- 但 batch size 从 1→128，speedup 系统性下降；
- 大模型衰减更"缓"（Llama3-70B 增 bs 到 32 只掉 14%，Llama3.1-8B 只掉 4.3%——但前者绝对值更高，意味着 70B 在 4 卡 TP 已经 compute-bound 因此 SD 提升空间更小）。

**SD 变体性能对比**（page 4, page 5）：
- **Draft-model-based**：在 Llama3-70B 上几乎全场最佳，因为 c≈0.125 低（pre-trained 1B Llama 的 forward 远小于 70B）；但在 Qwen3-8B 上反而不如 EAGLE-3 甚至 n-gram，因为 c≈0.375 大幅吞噬增益。
- **EAGLE / EAGLE-3**：稳定中等，acceptance 分布最紧凑，median 2–4 tokens（page 9 Fig 6）。
- **n-gram**：均值最低、方差最大、长尾最重；**在 InstructCoder 上反超 EAGLE / EAGLE-3**，因为 code-editing 有大量 prompt-output 重叠（详见 §5.4）。
- **n-gram-5（k=5）**：在高 BLEU bucket 上提升更显著，最高比 EAGLE-3 快 100%（page 18, Fig 16）——揭示了 k 应该自适应。

### 5.2 端到端 speedup（reasoning）

(page 6, Fig 2)

- Qwen3-8B-Thinking + GPQA-Main：EAGLE-3 1.64–1.80×，n-gram 1.50–1.58×；
- Qwen3-8B-Thinking + AIME22-24：EAGLE-3 1.6–1.8×。
- 因 KV cache 易爆 + request preemption 风险，reasoning 评测局限在 medium batch（≤16）。
- 论文特别选择"避免 preemption"的 batch range，否则 reported speedup 会被内存压力扭曲。

**洞察**：reasoning workload 上 n-gram 增长比 EAGLE-3 快得多（page 6, page 8 Fig 5），因为 chain-of-thought 中变量名、公式、中间步骤反复出现，形成 n-gram 福利。在 >13K token 的长生成里，n-gram 平均接受长度可达 2.7–5。

### 5.3 时间与内存分解

(page 7–8, Fig 3, Tab 2)

**时间**：
- **Verification 占 42%–95%**，并随 batch / model size 递增。
- Drafting：n-gram <2%；EAGLE / EAGLE-3 在 bs=1 时 12–20%、bs=512 时 3–7%；Draft-model 在 Llama3-70B + 1B 下 21%→3%、Qwen3-8B + 0.6B 下 47%→16%（小模型 draft 比例**惊人**）。
- Sampling <1.7%；vLLM overhead 3–12%，随 batch 摊薄。

**内存**：
- n-gram 零额外 GPU 显存（draft 来自 CPU history）；
- EAGLE static +3.1%（8B）/+1.4%（70B）/+5.3%（EAGLE-3 8B）/+4.9%（EAGLE-3 Qwen3-8B）；
- Draft-model：Llama3-70B + 1B → +1.8% static、+10% per-token KV；Qwen3-8B + 0.6B → +7.3% static、**+77.8% per-token KV**。
- 在 reasoning 长序列下，per-token KV 是显存压舱石——这意味着 Qwen3-8B + 0.6B draft 在 long-context reasoning 下会显著缩短可服务 batch / context length。

**关键 implication**（page 7）：因为 verification 主导，"减小 verify 成本"才是 SD 优化的真正杠杆——为后续 oracle 上界铺垫。

### 5.4 Case Study：n-gram 在 InstructCoder 上的反超

(page 10–11, Sec 7, Fig 7)

- 用 BLEU-4 衡量 prompt-output 重叠度，分 5 桶：[0–0.2), [0.2–0.4), [0.4–0.6), [0.6–0.8), [0.8–1.0]；
- 在 BLEU-4 ≥ 0.6 时，n-gram (k=3) 比 EAGLE 快 18–53%；k=5 时最高比 EAGLE 快 100%（page 18, Fig 16）；
- 反之 BLEU-4 ≤ 0.4 时，n-gram 慢 EAGLE 10–33%。

这是论文最 sharp 的一个发现：**training-free 的 n-gram 在合适的 workload 下能完胜重训练的 EAGLE 系列**。

### 5.5 Tree-style verification

(page 16–17, A.2.2, Fig 13)

- bs=1 时，tree (k=21) 比 chain 快 ~10–12%（Qwen3-8B GSM8K 1.65→1.85；Llama3-70B ShareGPT 1.81→2.03）；
- bs=64 时，tree 全部 < 1× speedup（即比 no-SD 还慢）；
- 原因：tree 在 verify 时跑了大量被 reject 的 branch，而 verify 本身就是瓶颈。
- **Acceptance rate 数据**：Qwen3-8B GSM8K chain 0.415 → tree-21 0.095（虽然 accepted length 从 2.25→2.92 提高，但 rate 暴跌）。
- **结论**：tree-style 在 production 大 batch 下基本无收益，是 prototype-only 的优化。

### 5.6 MTP

(page 16, A.2.1, Fig 12)

- GLM-4.5-Air-106B：1.3×–1.8× on GPQA-Main；
- 但因为开源版只放了 1 个 MTP head 并被自回归复用，position-wise acceptance 从 0.91→0.67→0.38 呈陡降，限制了 effective k；
- 这条线侧面证明：MTP 的 production 表现强烈依赖 "released MTP heads 的数量与训练策略"，不能简单按照 paper 数字外推。

### 5.7 Oracle 上界

(page 11–12, Sec 8, Fig 8, Fig 10)

- **Oracle proposed length** vs fixed-k vs adaptive：在 InstructCoder + Llama3.1-8B + n-gram + bs=1 下，oracle 2.75× vs fixed-5 2.1× vs adaptive 2.3×。差距随 batch size 上升而**扩大**（fixed/adaptive 衰减得更快，oracle 衰减得慢）。
- **Oracle Combine (EAGLE-3 + n-gram)**：相对单变体 oracle 再增 1.6×，端到端最高 4.9×（vs no-SD）。
- 在 InstructCoder 上 headroom 最大（红蓝交替最频繁）；GSM8K 上 headroom 接近 0（n-gram 极少有长 acceptance）。
- 这给出了**"自适应 SD method selector"**的明确 ROI——这是论文最 actionable 的研究方向输出。

### 5.8 何处 SD 是 illusion

综合 §5.1–§5.7，SD 在以下场景**接近 illusion**：
- **大 batch（≥64）+ 大模型 + compute-bound** 情形：speedup 普遍跌至 1.1×–1.3×；
- **Tree-style + 大 batch**：速度甚至比 no-SD 还慢；
- **小 target model + 大 c 的 draft pair**（如 Qwen3-8B + 0.6B）：draft 占执行 30–47%，吞噬大量增益；
- **prompt-output 低重叠 workload + n-gram**：n-gram 显著拖累；
- **MTP 的 head 复用模式**：position-wise acceptance 急剧衰减。

而在以下场景，SD 提供 1.5–4.9× **真实**收益：
- **Small/medium batch + memory-bound** 服务（< 32–64）；
- **Reasoning workload + EAGLE-3**（GPQA, AIME）；
- **InstructCoder + n-gram-5**（高 BLEU 桶）；
- **Llama3-70B + 1B Llama draft**（c 极低）；
- **EAGLE-3 + n-gram oracle-combined**（理论上界）。

---

## 6. 思想精读 / 启示

### 6.1 SD 的"GPU 同步税"——为什么 SD 在 GPU 上比 dataflow 难

SambaNova 在《Dataflow Is All You Need》中宣称在 SN40 上 SD 能拿到 6× 端到端加速，这与本文揭示的 vLLM/H100 上 1.2×–2× 形成强烈反差。如何理解？

- **GPU 上每一步 SD 内部都有 host/device 同步、调度、KV-cache 维护**：n-gram lookup → small forward → verify → reject sampling → 下一步——这条链路里每一段都要 launch CUDA kernel、对齐 stream，存在 launch latency 与 graph capture/replay 边界。论文显示 vLLM overhead 在 bs=1 时 12%，在 bs=64 时 ~3%（page 7）——但这只是"命名 overhead"，实际跨阶段的同步税混入了 verification 时间里。
- **Dataflow 架构（如 SN40）天然支持流水线 fusing**：propose-verify 路径可以以 dataflow graph 的形式编译为单一长 pipeline，draft 与 verify 重叠，rejection 可由片上 control plane 解决，"同步税"接近 0。所以同样的 acceptance rate 在 dataflow 上转化率更高。
- **GPU 上 verify 是顺序串行 batch matmul**：本文 Fig 3 显示 verify 在大 batch 下吃掉 95% 时间——这是 GPU 的 attention/MLP kernel 本身的 compute-bound 边界，光改 SD 算法救不了。

**启示**：评估"SD 是否值得"必须分平台讨论。对 GPU + vLLM/SGLang 部署，要严肃对待 batch=64–128 的低收益；对 dataflow 加速器，SD 的杠杆系数本就大于 GPU。

### 6.2 batch size 的"分水岭"

论文反复揭示一个核心机制：**SD 是把 idle compute 转成 throughput 的交换**（page 4 倒数第二段）。

- bs 小时 → memory-bound → idle compute 富余 → SD 几乎免费 → 全部 verify 失败也只多花点剩余算力；
- bs 大时 → compute-bound → 没有 idle compute → 每个 reject 的 token 都直接拖慢吞吐 → SD 退化。

公式 (1) 的有效区间被 bs 严格限制（page 6 footnote 4）。这意味着 production 团队在 SD A/B 测试时，必须**沿着 batch size 轴扫描**，而不是单点比对。

**衍生洞察**：
- 当 batch size 大时，应该 **降低 k 而不是升高**（fixed-k 在大 batch 下衰减最快，page 11 Fig 8）；
- adaptive k 简单启发式（Joao Gante 2023）已经能拿到一部分 oracle 收益，但远未到位；
- 真正的"自适应 SD scheduler"需要根据 system load + per-position acceptance prediction 动态调 k，是论文留下的最重要 open problem。

### 6.3 Verification 主导 → 真正的优化方向

(page 7 implication; page 11 Sec 8 motivation)

verification 占 42–95% → 即便把 drafting 优化到 0，端到端 speedup 也撑死再提升 20%。所以 SD 优化的真正杠杆是：
1. **减少 verify 失败**：oracle proposer / 高 acceptance head；
2. **跳过 verify**：对极高置信度的 token 直接接受（论文未展开，但是 8.1 暗示的方向）；
3. **复用 verify 计算**：这正是 EAGLE 的 hidden-state reuse 思路；
4. **跨变体融合**：Oracle Combine 给出 1.6× headroom。

### 6.4 Acceptance 的三层异质性

(page 8 Sec 6 Summary)

| 维度 | 表现 | 启示 |
| --- | --- | --- |
| 单 request 内 | EAGLE 平稳；n-gram bursty（reasoning 末尾会跌） | 应做 within-request adaptive k |
| 跨 request | 同一 dataset 下 std 巨大，n-gram 是 EAGLE 的 2–5× | 不能用 dataset-level 平均报告 SD 性能 |
| 跨 dataset | 6 个 dataset 上 n-gram 排名波动巨大 | SD 必须按 workload 选 variant |

这是对 prior SD 论文"avg acceptance rate ≈ X%"宣传方式的釜底抽薪——单一 metric 抹平了所有可利用的异质性。

### 6.5 与"Dataflow Is All You Need"的对照（再补一刀）

| 维度 | SN40 (Dataflow) 上 SD | H100 (vLLM) 上 SD |
| --- | --- | --- |
| 架构同步税 | 极低（编译入 pipeline） | 高（CUDA kernel chain + KV 维护） |
| Batch size sensitivity | 可能更弱 | 极强（本文重点） |
| 端到端 speedup | 报告 ~6× | 1.2×–2.6× 主流，oracle combine 4.9× |
| Verify 主导 | 部分被 fuse | 95% 顶死 |
| Production 标准 | 厂商特化 | vLLM/SGLang 标准 |

> **一句话点评**：SambaNova 的 6× 不是错的，但只在 dataflow + 厂商专属 stack 上成立；将其外推到 GPU/vLLM 是错误的。本文给出的 1.2–2.6× 是 GPU/production 真值，**4.9× 是 GPU 上的理论天花板（且需 oracle）**——两者并不矛盾，但社区不应混为一谈。

### 6.6 何时 SD 真的有用

综合：
- **Reasoning workload + 长生成 + EAGLE-3**：GPQA / AIME 上 1.6–1.8× 是稳定的；
- **Code-editing + n-gram-5（高 BLEU）**：可超 EAGLE-3 100%；
- **Memory-bound 部署 (small batch, large model)**：典型如 70B + bs ≤ 32；
- **Draft-model c < 0.15** 的 model pair（70B + 1B 是范例）；
- **未来方向**：SD method ensemble + adaptive k + position-aware proposer。

---

## 7. 局限与开放问题

### 7.1 论文承认的局限

1. **Verification 内部未细分**（page 7 倒数第二段）：作者把 verify 当作单一 bucket，未来可拆 attention / MLP / layernorm。
2. **Activation memory 未计**（page 7, A.3）：仅算 weights + KV cache，实际 activation tensors 在长序列下也很显著。
3. **MTP 实验受限于 open-source weights**（page 16, A.2.1）：GLM-4.5-Air 只发布了 1 个 MTP head，被自递归复用，未达 MTP 真实潜力。
4. **EAGLE-3 在 Llama3-70B / GLM-4.5-Air 上无官方权重**（page 3 footnote 2），跨模型对比有缺口。
5. **BLEU 仅与完整 prompt 比对**（page 11 倒数第二段）：n-gram 实际可命中"prompt + 已生成上下文"，因而真实 reuse 比 BLEU-prompt 衡量到的更高。
6. **Tree-style 评测仅 SGLang**：vLLM 当前 tree 路径未充分优化，作者明示。
7. **Greedy only**：所有实验 temp=0；high-temp 采样下 acceptance 更低，SD 表现可能进一步下降，但本文未覆盖。
8. **Batch size 是 steady-load 等长 batch**（page 4，Tab 1 上方解释）：真实 serving 是 mixed-length + preemption，论文有意避开但承认这是简化。

### 7.2 开放问题（论文挖出的研究坑）

1. **Adaptive k 学习器**：能否训练一个轻量预测器，对每个 (model, workload, position) 给出最佳 k？
2. **Position-aware ensemble**：oracle combine 给 1.6× headroom，但实际 selector 怎么做？需要 < proposing overhead 的开销才能净赚。
3. **Verify-skip mechanism**：高置信 token 是否可以直接接受、跳过 verify？这突破了 SD 的"零分布偏差"前提，需要新的统计保证。
4. **MTP 多 head 联合训练**：开源 MTP 仅放 1 head 是社区瓶颈，需要 community-scale 的 MTP head zoo。
5. **跨平台公平性**：如何在 GPU、TPU、SambaNova、Cerebras 上做 SD 公平评测？本文是 GPU 单平台。
6. **Production serving 下的 mixed batch SD**：当前 vLLM continuous batching 下，每个 request 处于不同 batch position（prefill/decode 混合），SD 选择策略有待研究。
7. **Long-context + 长 KV 下 draft 的退化**：reasoning 已暴露 MTP head 性能衰减，draft-model 的 c 可能也会随 context length 漂移，未充分测量。
8. **rejection sampling 的 fairness**：page 4 Tab 1 显示 SD vs no-SD 输出长度本身就有差异（kernel/FP nondeterminism），现实中要不要更严格的 distribution match？

### 7.3 我的额外质疑（论文未说）

- **Throughput metric 的副作用**：throughput 抹平了 latency 异质性，但 production SLA 通常关心 P99 latency 与 TPOT（time-per-output-token）。SD 是否会改善/恶化 P99？论文未给。
- **Draft-model 重新训练成本**：draft-model-based 方法的"训练成本 + 维护成本"（每个 target 升级都要重训 draft）没有计入 TCO。
- **仅 H100 + FP16**：FP8 / Hopper Transformer Engine / Blackwell 的 SD 行为可能完全不同——SD 的 verify 占比在 FP8 下会进一步上升。

---

## 8. 关键术语速查表

| 术语 | 定义 | 位置 |
| --- | --- | --- |
| **Speculative Decoding (SD)** | 用一个轻量 proposer 生成 k 个候选 token，让 target LLM 一次 forward 同时验证，加速生成 | page 2 |
| **Target Model** | 最终决定输出分布的大 LLM；SD 不改变其分布 | page 2 |
| **Draft Model** | 生成 k 个候选 token 的小模型；其分布只需"接近"target | page 2 |
| **Acceptance Rate (α)** | 每个 propose 的 token 被 target 接受的概率 | page 6 |
| **Proposed Length (k)** | 每步 propose 的 token 数 | page 6 |
| **Drafting / Proposing** | SD 第一阶段：生成候选 token | page 7 |
| **Verification** | SD 第二阶段：target 对 k+1 候选并行 forward | page 7 |
| **Rejection Sampling** | 第三阶段：根据 verify logits 决定接受 | page 7 |
| **EAGLE** | Auxiliary Transformer head + hidden state reuse 的 SD 变体 | page 2; Li et al. 2024b |
| **EAGLE-3** | EAGLE 的训练时 test-time scaling 变体 | page 2; Li et al. 2025 |
| **MTP (Multi-Token Prediction)** | head 与 main model **联合训练**，开箱即用的 SD | page 3; Liu et al. 2025 (DeepSeek-V3) |
| **n-gram / Prompt Lookup** | 从已生成上下文中检索 n-gram 作为 proposal，training-free | page 3; Saxena 2023 |
| **Tree-style verification** | 用 token tree 同时验证多个分支（如 Medusa, EAGLE） | page 16 |
| **Chain-style verification** | 单链 propose-verify | page 16 |
| **Oracle Proposer** | 假设知道每步实际可接受长度，把 k 设为实际值 | page 11 |
| **Oracle Combine** | 跨变体（如 EAGLE-3 + n-gram）按 token 位置取最优 | page 11 |
| **Adaptive k Heuristic** | Joao Gante 2023：从 5 起，全接受+2，否则-1，min 1 | page 11 |
| **c (drafting/target ratio)** | draft 单 forward / target 单 forward 时间比 | page 6 |
| **Bonus Token** | verify 阶段除接受 draft 外，还会从 target 自身分布采样 1 个额外 token | page 8 footnote 5 |
| **vLLM** | UC Berkeley/Sky Computing Lab 的 production-grade serving engine | page 3 |
| **CUDA Graphs** | NVIDIA 的 launch-overhead 优化，capture-replay 静态计算图 | page 3 |
| **continuous batching** | vLLM 的 request 级动态 batching | page 3 |
| **chunked prefill** | 将长 prompt 切片以与 decode 流水交错 | page 3 |
| **InstructCoder** | code-editing instruction tuning dataset | page 3; Li et al. 2024a |
| **GPQA-Main / AIME22-24** | reasoning benchmarks | page 4 |
| **BLEU-n** | n-gram 重叠度量；本文用于度量 prompt-output reuse | page 10 |
| **Memory-bound / Compute-bound** | 内存带宽/计算瓶颈，决定 SD 收益高低的关键状态机 | page 4 |
| **Token Throughput** | tokens/sec，本文公平 metric | page 4 |
| **Steady-load Setting** | 论文用复制 request 至等长 batch 模拟稳态 | page 4 |

---

## 9. 关键页码索引

| 主题 | 起始页 |
| --- | --- |
| Abstract / 论文立场 | page 1 |
| 流派综述 + benchmark 三大顽疾 | page 2–3 |
| 实验设置（engine, hardware, models, datasets） | page 3–4 |
| Generation length nondeterminism + throughput metric 选择 | page 4, Tab 1 |
| **Fig 1 端到端 speedup（非 reasoning）** | **page 5** |
| Batch size scaling 现象 | page 4–5 |
| n-gram 在 InstructCoder 反超 | page 5 |
| Reasoning workload 表现（Fig 2） | page 6 |
| **公式 (1) E(speedup) = (1−α^(k+1)) / [(1−α)(kc+1)]** | **page 6** |
| Time breakdown (Fig 3) | page 7–8 |
| Memory breakdown (Tab 2) | page 7 |
| **Acceptance 三层异质性** | **page 8–10** |
| 单 request：长生成下 n-gram vs EAGLE-3 (Fig 5) | page 9 |
| 跨 request / 跨 dataset (Fig 6, Tab 4) | page 9, page 18 |
| InstructCoder × BLEU heatmap (Fig 7) | page 10 |
| **Oracle 上界 (Fig 8)** | **page 11** |
| 多变体 oracle combine (Fig 9, Fig 10) | page 11–12 |
| Conclusion + 致谢 | page 12 |
| References | page 12–15 |
| 数据集统计 (Fig 11) | page 16 |
| MTP appendix (Fig 12) | page 16 |
| Tree-style appendix (Fig 13) | page 16–17 |
| Memory 解析公式 + Tab 3 模型 spec | page 17–18 |
| Reasoning request-level 分布 (Fig 14, Tab 4) | page 18 |
| n-gram-5 BLEU heatmap (Fig 16) | page 18 |
| 全数据集 oracle 对比 (Fig 17, 18, 19) | page 19 |
| Latency 形式的端到端结果 (Fig 20, Fig 21) | page 20 |
| 实验配置 Tab 5 + Tab 6 | page 21 |
| **Artifact Appendix（复现指南）** | **page 22–23** |

---

## 10. 一句话点评

> **"SD 不是骗局，但 production-grade 评测显示，社区把 batch=1 的研究原型 speedup 当成产品 SLA 是赤裸裸的 illusion——这篇论文用 vLLM、6 个 workload、3 个模型规模、4 个 SD 流派 + Oracle 上界，第一次把 SD 钉在了'verification-bound、batch-size-fragile、workload-specific'的真实坐标系里；同时它也指出真正的 4.9× 上界仍然存在，但需要 adaptive k 与多变体融合——这是一篇'反主流但建设性'的批判式 systems 论文，应当成为今后所有 SD 论文的评测 baseline。"**

---

## 附录 A：与 prior SD 论文的"立场对照"

| 论文 / 系统 | 报告 speedup | 评测 batch | 评测 engine | 与本文对照 |
| --- | --- | --- | --- | --- |
| Leviathan 2023 (vanilla SD) | 2–3× | 1 | prototype | bs scaling 缺失 |
| Medusa (Cai 2024) | 2.2–3.6× | 小 | prototype | 大 batch 退化未测 |
| EAGLE / EAGLE-3 (Li 2024b/2025) | 3–5× | 主要 1 | prototype | 本文在 vLLM 上重测，1.5–1.96× |
| SpecInfer (Miao 2023) | 1.9–2.8× | 小 | own engine | tree verify 大 batch 失效 |
| Lookahead (Fu 2024) | 1.5–2.3× | 1 | prototype | 同上 |
| REST (He 2024) | 1.8× | 小 | prototype | 同 family（n-gram-like） |
| Liu 2024 (Goodput-aware SD) | – | 大 | own | 与本文最一致：揭示大 batch SD 退化 |
| **本文 (Liu 2026, MLSys)** | 1.2–2.6× 实测 / 4.9× oracle | **1–128** | **vLLM v0.10.1.1** | **production-fair** |

## 附录 B：可立刻复现的实验设计

(page 22, B.5)

```bash
# (1) vLLM profiling
export SHAREGPT_PATH=/path/to/sharegpt.json
ENV_DIR=/path/to/envs bash scripts/rebuild_env.sh
conda activate /path/to/envs
bash scripts/run-l3-8b.sh   # 1× H100, 24-36 小时

# (2) Simulator (无需 GPU)
cd simulator/
conda create -n specbench_simulator python=3.10 -y
conda activate specbench_simulator
pip install -r requirements.txt
./simulate_and_plot.sh        # 30-40 分钟
```

预期输出：复现 Fig 1 (a)–(d)、Fig 8、Fig 10、Fig 17、Fig 19；smoke test：`num_reqs=5, batch_sizes="1 16"` < 10 分钟。

## 附录 C：给 production team 的检查清单

1. **测前先沿 batch size 扫描**（1, 16, 32, 64, 128），不可单点报数。
2. **用 throughput 而非 latency**，并用 He 2025 的去 nondeterminism 工具控制变量。
3. **按 workload 选 SD**：code-edit → n-gram-5；general chat → EAGLE/EAGLE-3；reasoning → EAGLE-3；70B+ → 1B draft。
4. **量化 c 与 α**：拿到具体的 c < 0.2 才考虑 draft-model SD。
5. **跑时间分解**：若 verify 占比 > 80%，停止优化 drafting，转向 ensemble / oracle direction。
6. **per-token KV 增量预算**：reasoning 长序列下 draft pair 的 +77% KV 直接卡死可服务 batch，要事先算。
7. **不要在 GPU 上推外推 SambaNova 6×**：架构差异巨大，自家平台跑数。
8. **adaptive k 已是必备**：fixed-k 在大 batch 几乎线性垮，至少先上 Joao Gante 启发式。
9. **MTP 当心 head 复用陷阱**：开源 MTP 模型先跑 position-wise α 衰减再决定。
10. **Tree-style 仅在 bs ≤ 8 用**：bs ≥ 32 几乎必跌穿 1×。

---

*报告完*。所有引用页码与原 PDF 对齐（23 页正文 + appendix），引用论文的 OpenReview ID 为 **fzkqtezFEi**。

