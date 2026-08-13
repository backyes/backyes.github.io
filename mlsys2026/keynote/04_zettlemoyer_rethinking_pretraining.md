# 04 — Luke Zettlemoyer · *Rethinking Pretraining: Data and Architecture*

## 元信息

- **会议**：MLSys 2026，Bellevue, WA，2026/05/18–22
- **场次**：Keynote Talk · 周四 10:30–11:30 PDT · Grand Ballroom 1（含 Grand Ballroom 2 overflow）
- **官方页**：https://mlsys.org/virtual/2026/invited-talk/3706
- **讲者**：Luke Zettlemoyer — University of Washington Paul G. Allen School 教授、Meta Senior Research Director。研究方向 NLP × ML × 不确定性下决策；近年聚焦 text 与多模态语言模型的训练科学。荣誉：2025 Schmidt AI 2050 Senior Fellow、2024 ACL 主席当选、2021 ACL Fellow、PECASE 2016 等。

## 官方摘要（mlsys.org，逐字）

> Large language model training follows a standard pipeline: tokenization, pretraining, possibly mid-training, and post training or alignment. Despite its wild success, we understand relatively little about this recipe and are almost certainly missing many opportunities to improve it. In this talk, I will focus on three such cases.
>
> I'll describe our work on **data efficient post training** (e.g. **LIMA**, **ALMA**, and **s1**) where we argue that nearly all advanced model capabilities ultimately come from the pretraining data, even if effective alignment is still essential for controlling model behavior.
>
> I will also describe **new methods for extracting more signal from the pretraining data**, including **new hierarchical architectures for byte-level language models (e.g. BLT)** that are both tokenizer-free and scale better than traditional BPE-based methods, especially in the long tail.
>
> Finally, I will discuss **decentralized, modular training algorithms (e.g. BTM)** that better isolate and control the influence of specific data on specific model components and behaviors.
>
> Together, these methods promise to simplify training and improve scaling, by centering and amplifying the influence of data in architecture design.

## 关键论点

1. **能力来源主张**：先进模型能力几乎全部来自 **pretraining data**；alignment 控制行为，但不创造新能力。
2. **数据高效后训练**：LIMA / ALMA / s1 等工作证明，少量高质量 post-train 数据已经足够"解锁"预训练里的能力。
3. **抛弃 BPE，回到 byte-level**：BLT（Byte Latent Transformer）等新型层次化字节级架构是 tokenizer-free 的，长尾 scaling 更好。
4. **去中心化、模块化训练**：BTM（Branch-Train-Merge 系列）让特定数据→特定模块的影响可隔离、可控制。
5. **整合主张**：把"数据"放回架构设计的中心，而非把 architecture 与 data 视作两条独立轴——这能同时简化训练流程并改善 scaling。

## 相关工作 / 背景资料

- **LIMA**（Less Is More for Alignment, Meta 2023）
- **ALMA**（Advanced Language Model with Multilingual Augmentation）
- **s1**（Stanford/UW 2025 简单 test-time scaling，数据高效推理时缩放）
- **BLT**（Byte Latent Transformer, Meta 2024–2025，tokenizer-free 字节级模型）
- **Branch-Train-Merge / Cluster-BTM / BTX**（Meta 模块化训练系列）

## 视频

- **SlidesLive 嵌入**：talk 页有 SlidesLive embed。
- **MLSys 官方录播**：预计 2026/06/22 前后上线 https://mlsys.org/virtual/2026/invited-talk/3706。
- **可能在 UW / Allen School 频道二次上架**：之前 Luke 的几次 keynote 在 https://www.youtube.com/@AllenSchool 也能找到。

## 社交媒体讨论

> 本环境受限，无法直连 reddit / x.com；待手动补充。

预期热点：
- r/MachineLearning：BPE vs byte-level 的长期口水战会因 BLT 重燃
- X 上 NLP 圈（@srush_nlp、@_jasonwei、@ylecun、@sleepinyourhat 等）会有 live-tweet
- 数据高效 post-training（s1 风格）与 RLHF/PPO 派的方法论之争

## 我的总结

这是 5 场 keynote 中 **学术含量最纯**、对 ML 研究者最直接相关 的一场。Luke 把一个看似已被定型的 LLM pipeline（tokenize → pretrain → mid-train → post-train）逐段解构，用三组 Meta/UW 近年自己的工作把每一段都"再问一次"。最有冲击力的是核心主张——**模型能力几乎完全来自预训练数据，alignment 只是行为控制器**——这是对 RLHF 时代"对齐工作主导能力上限"叙事的有力反驳，也呼应了 LIMA/s1 等"很少高质量样本就够了"的现象级结果。第二个主张更偏架构：BLT 把 tokenizer 砍掉，让模型直接在字节流上学层次表征，长尾上的 scaling 比 BPE 更好——这是对当前几乎所有 LLM 都默认 BPE tokenizer 这一"黑盒前置"的根本质疑。第三块 BTM 是把"哪些数据训练哪些参数"做模块化隔离，使得数据影响可定位、可剥离。整体看，他在把 **"data 才是真正的架构"** 提升为下一个十年 LLM 训练科学的中心命题——这与早年间 Sutton《Bitter Lesson》同类，但更建设性：它给出可操作的工程路线（去 tokenizer、模块化合并、数据高效 post-train），而不是仅仅哲学口号。
