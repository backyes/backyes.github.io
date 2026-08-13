# 02 — Lidong Zhou · *The Next Horizon of Systems: From MLSys to System Intelligence*

## 元信息

- **会议**：MLSys 2026，Bellevue, WA，2026/05/18–22
- **场次**：Keynote Talk · 周二 10:30–11:30 PDT · Grand Ballroom 1（含 Grand Ballroom 2 overflow）
- **官方页**：https://mlsys.org/virtual/2026/invited-talk/3665
- **讲者**：Dr. Lidong Zhou — Microsoft Corporate VP，Microsoft Asia Pacific R&D Group 首席科学家，Microsoft Research Asia 院长；研究方向为可扩展、可靠、可信的分布式系统；SOSP / OSDI / USENIX ATC 多次最佳论文；主导过 Microsoft 搜索、大数据、云和 AI 基础设施的多个大型系统设计。

## 官方摘要（mlsys.org，逐字）

> MLSys showed how systems can accelerate AI. The next shift is broader: AI is beginning to reshape the practice of systems itself. This emerging paradigm, which we call **system intelligence**, goes beyond automating programming tasks. It enables new forms of reasoning, design, validation, and evolution for complex systems while preserving rigor.
>
> In this talk, I will argue that system intelligence changes not only what systems we can build, but also how we understand systems as a discipline. It pushes us to rethink systems principles and methodology, shifting attention from code-level complexity to greater rigor in specification, design, and validation. Through our experiences with system verification, I will discuss how this shift may help give systems a stronger scientific foundation.

## 关键论点

1. **范式跃迁**：MLSys 上半场是"系统加速 AI"，下半场是"AI 重塑系统学科本身"。
2. **"System Intelligence"** 不是把 LLM 套到编码助手上——它要覆盖 **reasoning / design / validation / evolution** 全生命周期。
3. **方法论位移**：注意力从"代码级复杂度"向 **规约（specification）、设计、验证** 的更高严谨度迁移。
4. **形式化验证 + AI**：他将以 MSR Asia 在系统验证（system verification）方面的实践作为论据，主张这条路径能给 systems 这门工程学科一个更坚实的"科学基础"。
5. **隐含立场**：systems 长期以来缺少自然科学意义上的"理论体系"；AI 介入后，规约/证明/演化变得可大规模化，"工程学科"有机会升格为"科学学科"。

## 相关工作 / 背景资料

- MSR 在分布式系统验证方面的多年积累（Ironclad、IronFleet、Verus 风格的工具链）。
- MSR Asia 的 AI4Systems / Systems4AI 双向研究路线图。
- Lidong 此前关于 Bing/Cosmos/AzureML 等大规模生产系统的论文与演讲。

## 视频

- **SlidesLive 嵌入**：talk 页内嵌 SlidesLive 播放器；anonymous 抓不到 presentation ID。
- **MLSys 官方录播**：预计 2026/06/22 前后免费上线 https://mlsys.org/virtual/2026/。
- **可能的二次发布**：MSR 经常把同一 keynote 在 https://www.microsoft.com/en-us/research/video/ 重新上架，带专门页面与字幕。

## 社交媒体讨论

> 调度环境无法访问 reddit / x.com / 微信 / 知乎；待手动补充。`social_media_queries.md` 含 query 模板。

预期热点：
- 学术圈（systems 教授、PhD 群体）对"是否真的有 system intelligence 范式"的概念性争论
- 中文圈（知乎、X 上的华人 AI 系统圈）会有较多解读，因为讲者在中国系统社区影响力大
- 与 Dawn Song / Tianqi Chen 等 SOSP-OSDI 圈子的互动

## 我的总结

Lidong Zhou 的 keynote 是 5 场里 **野心最大、最哲学化** 的一场——它不谈某个技术点的提速，而是直接提出一个新范式名词：**System Intelligence**。这是很典型的 MSR Asia 风格：站在十年尺度上重新定义学科边界。如果说 Saroufim 在讲"AI 写 kernel 的工程化"，Lidong 则把同一件事推到极限——AI 不只是写代码，而是参与 specification、reasoning、validation、evolution 全过程，目标是把 systems 从一门"经验工艺"升格为有 **形式化骨架** 的科学学科。这一论断有重要操作含义：未来 MLSys 论文的"竞争前沿"可能不在更快的 attention，而在 **可验证的系统抽象** + AI 辅助的设计/演化回路。他用 MSR 多年系统验证经验为论据，避免了"概念漂浮"——这一点值得整个社区认真对待，尤其当国内的 AI Infra 团队正大量投入"AI-for-systems"和"自我演化系统"时，这场 keynote 是最权威的方向锚点之一。
