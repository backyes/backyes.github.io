# 05 — Christos Kozyrakis · *The Path to Inference Efficiency*

> 注：MLSys 日程页标题拼写为 "In**fe**rence" 时漏一字母（"In**ference**"），属官方页 typo。

## 元信息

- **会议**：MLSys 2026，Bellevue, WA，2026/05/18–22
- **场次**：Keynote Talk · 周五 09:45–10:45 PDT · Grand Ballroom 1
- **官方页**：https://mlsys.org/virtual/2026/invited-talk/3723
- **讲者**：Christos Kozyrakis — NVIDIA 计算机体系结构研究员、Stanford Leonard Bosack and Sandy K Lerner Engineering Professor。研究方向：AI 软硬件基础设施 + 用 AI 做硬件/软件设计。荣誉：ACM/IEEE Fellow、IEEE Harry H. Goode Award、ACM SIGARCH Maurice Wilkes Award、ISCA & ASPLOS Influential Paper、HPCA & SoCC Test of Time、Okawa Foundation Research Grant 等。

## 官方摘要（mlsys.org，逐字）

> Agentic AI is moving out of demos and into daily use, creating enormous demand for efficient inference: higher throughput, lower latency, and better efficiency in both dollars and joules. Meeting these targets requires rethinking the full inference stack, from the specialized silicon that runs the models, to the system software that compiles, schedules, and serves them at scale, to the model architectures that determine what must be computed in the first place.
>
> In this talk, we will examine these layers with an eye toward the next major advances in hardware architecture, and how systems and algorithms can be co-designed to fully exploit them. Large gains in inference efficiency will come not from isolated improvements, but from treating hardware, systems, and models as an integrated stack.

## 关键论点

1. **驱动力**：Agentic AI 已从 demo 走入日常使用，推理 throughput / latency / $-per-token / J-per-token 同步成为一线指标。
2. **三层栈**：(a) 专用硅，(b) 编译/调度/服务的系统软件，(c) 决定"该算什么"的模型架构——必须协同设计。
3. **硬件下一波**：以新型硬件架构进展（精度、内存层次、互联、近存计算等）为核心切入点。
4. **核心命题**：**收益来自整合，而非各层局部优化之和**——这是体系结构 + 系统 + 算法 co-design 的经典主张，但被放到 agentic 推理的新工作负载下重新校准。

## 相关工作 / 背景资料

- Christos 在 Stanford 多年系统/体系结构论文（数据中心计算、内存系统、近数据处理 NDP、Resource Central 等）。
- NVIDIA 同期产品线：B200/B300、Rubin、NVL 互联、FlashInfer、TensorRT-LLM。
- ISCA/ASPLOS/HPCA 近年关于 LLM inference 的高被引文章（Sarathi、PagedAttention/vLLM、SplitWise、DistServe、Mooncake、Loongserve 等的延伸）。
- MLSys 2026 同期的 NVIDIA FlashInfer AI Kernel Generation Contest（与本场 keynote 同一个 stack-co-design 主题）。

## 视频

- **SlidesLive 嵌入**：talk 页有 SlidesLive embed。
- **MLSys 官方录播**：预计 2026/06/22 前后上线 https://mlsys.org/virtual/2026/invited-talk/3723。
- **可能的二次发布**：NVIDIA GTC、Stanford EE / SystemX、Stanford Online 频道经常会复用同一份 keynote。

## 社交媒体讨论

> 本环境无法访问外网；待补。

预期热点：
- 推理服务系统圈（vLLM / SGLang / TensorRT-LLM 团队）的同行评价
- 近存/HBM / memory bandwidth wall 的讨论
- 体系结构 PhD 圈对"硬件/系统/算法 co-design 还能继续打多少代"的争论
- 行业话题：agentic 推理负载（多步、长上下文、多工具调用）对当前 prefill/decode 服务架构的冲击

## 我的总结

Kozyrakis 这一场是 5 场 keynote 中 **最系统工程视角** 的一场，定位与 Vahdat 互补：Vahdat 谈 hyperscaler 已经做到的工程上限，Kozyrakis 谈学界/NVIDIA 视角下"下一代 stack 该怎么搭"。摘要里把 inference 拆成"silicon — system software — model architecture"三层并强调 **co-design 才有大收益**，这是 ISCA/ASPLOS 派对 ML inference 一贯的方法论。在 agentic AI 真正落地的 2026 年，这个观点的现实压力更大：传统 prefill/decode 流水化只针对单轮请求做了优化，但 agentic workload（多轮、长上下文、工具反复调用、KV cache 跨请求复用）正逼近现有服务架构的天花板——硬件层（更便宜的高带宽内存/对称互联）、系统层（KV cache 全局调度、推测解码、speculative + serving 联合优化）、算法层（更稀疏注意力、混合架构、small-and-many MoE）必须同时迭代。这场 keynote 的"最大潜在干货"不在某个新数字，而在他作为 NVIDIA 同时拥有学术身份的双栖角色，会披露 **"下一代硬件假设 + 系统接口"**——这往往会成为之后两到三年 MLSys 论文的 baseline 假设。和 Vahdat（工业落地上限）+ Saroufim（AI 写 kernel 的可验证子问题）+ Lidong（系统学科范式）+ Zettlemoyer（数据中心化的训练科学）合起来，五场 keynote 形成一个非常完整的 2026 年 MLSys 立体图。
