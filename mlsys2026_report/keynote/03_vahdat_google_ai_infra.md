# 03 — Amin Vahdat · *Keynote: SVP and Chief Technologist, AI & Infrastructure*

## 元信息

- **会议**：MLSys 2026，Bellevue, WA，2026/05/18–22
- **场次**：Keynote Talk · 周三 10:30–11:30 PDT · Grand Ballroom 1（含 Grand Ballroom 2 overflow）
- **官方页**：https://mlsys.org/virtual/2026/invited-talk/3684
- **讲者**：Amin Vahdat — Google SVP 兼 Chief Technologist, AI & Infrastructure；负责 Alphabet/Google 的定制硅、数据中心、网络、供应链与运营。2019 年前为 Google Networking 副总裁与技术负责人。

## 官方摘要

> ⚠️ MLSys 官方页 **未发布该 keynote 的 abstract**——只列了讲者信息与时间地点。这是 5 场 keynote 中唯一一场无公开摘要的，原因可能是 Google 内部审稿/披露策略。

## 关键论点（基于讲者职责与 Google 近期公开演讲推断，**非演讲实录**）

> 以下推断标记为「⚠ 推测」，仅作为方向性预读，核对请等录像上线。

1. ⚠ **AI 基础设施是一个完整 stack 协同问题**：从 TPU/GPU 定制硅，到液冷/光网络数据中心，到调度器与服务系统，再到 Gemini 这一级模型——只优化一层无法继续 scale。
2. ⚠ **网络是新瓶颈**：随着模型从单数据中心训练扩展到多 region，光交换/optical circuit switch、AI-aware 网络拥塞控制可能是重点。
3. ⚠ **能效/单位功耗算力**：Google 长期强调 PUE 与每瓦推理性能；2026 年大概率重申"摩尔定律之外"靠系统协同获得的 5–10× 收益。
4. ⚠ **从训练到推理的重心迁移**：行业整体推理算力消耗即将（或已）超过训练，定制芯片与服务架构需要相应再设计。

## 相关工作 / 背景资料

- Vahdat 在 OCP / Hot Chips / Google I/O 关于 TPU、Jupiter network、Andromeda、Borg/Omega 等的多年公开演讲。
- Google 近年披露：TPU v5e/v5p、Trillium、Ironwood（截至 2026）。
- Google MLSys 2024/2025 历次工业 talk 的同源主题。

## 视频

- **SlidesLive 嵌入**：talk 页同样有 SlidesLive embed。
- **MLSys 官方录播**：预计 2026/06/22 前后上线 https://mlsys.org/virtual/2026/invited-talk/3684。
- **Google 官方频道**：Vahdat 的 keynote 经常会在 https://www.youtube.com/@Google 或 Google Cloud 频道二次上架（带 transcript）。

## 社交媒体讨论

> 本次环境无法访问外网；待补。

预期热点：
- 工业界（OpenAI、Anthropic、Meta、字节、阿里）关于 "Google 比同行领先/落后多少"的横向比较
- TPU 路线图猜测（继任 Ironwood 的产品）
- 网络/光交换方向的硬件讨论（Hot Chips 圈）

## 我的总结

Vahdat 这一场是 5 场里 **可预测度最高**、**惊喜度最低** 但 **信息密度最高** 的一场——因为他代表 Google 全栈 AI 基础设施的负责人身份，演讲基本等价于 Google 对外发布的"AI Infra 白皮书"。在没有公开 abstract 的前提下，最值得期待的是三类新增信息：(1) Ironwood 之后下一代 TPU/网络的若干指标，(2) 多数据中心训练（multi-region / multi-DC）的最新工程实践，(3) 推理工作负载占比超过训练的官方数字。从 MLSys 社区视角，他这一场往往是设定"现实工程上限"的——告诉学术界"已经做到的 baseline 在哪"，间接框定了哪些研究问题已被工业界吃掉、哪些还开放。⚠️ **本档案不收录任何演讲细节，请以 2026/06/22 后上线的官方录像为准。**
