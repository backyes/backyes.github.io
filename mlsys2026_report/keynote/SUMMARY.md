# MLSys 2026 Keynote · 跨主题综合总结

> 5 场 keynote 拼起来，是 2026 年这个时间点对 ML × Systems 学科最完整的一张全景图。  
> 本文以摘要 + 讲者既往工作为唯一依据综合得出，不含外部社交媒体或视频内容（待录像上架与社交检索后另议）。

---

## 1. 一行口号

| Keynote | 一行核心 |
|---------|----------|
| Saroufim · *When AI Starts Writing Systems Code* | AI 正在写 GPU kernel，但能否 production-ready，瓶颈在评估基础设施而不是模型 |
| Lidong Zhou · *System Intelligence* | 上半场系统加速 AI；下半场 AI 重塑系统学科——从规约到验证全链路被 AI 渗透 |
| Vahdat · *Google AI & Infra* | hyperscaler 视角下 AI 基础设施的真实工程上限（硅 / DC / 网络 / 软件全栈） |
| Zettlemoyer · *Rethinking Pretraining* | 模型能力几乎只来自预训练数据；BPE 该退场、模块化训练该上场 |
| Kozyrakis · *Path to Inference Efficiency* | Agentic AI 推理负载逼近天花板；硅 + 系统 + 模型必须 co-design 才有大收益 |

---

## 2. 三条主轴

### 主轴 A：Stack Co-design（Vahdat × Kozyrakis）

两位讲者从 **工业落地** 与 **学术 + 体系结构** 两侧撞出同一结论：单层优化的边际收益快速衰减，**收益必须从硬件 / 系统 / 模型联合设计中获得**。

- Vahdat：Google 已经把 TPU + Jupiter 网络 + Borg 调度 + Gemini 模型当一个整体迭代。
- Kozyrakis：从硅、编译/调度/服务、到模型架构必须 **被当作一个整体 stack** 来设计，特别是 agentic AI 的多轮长上下文负载已经把传统 prefill/decode 流水设计逼到极限。

**对从业者的含义**：单层（如纯算子库、纯调度器）的论文和工程窗口在收窄；具备跨层视野的 stack-level 工作在 2026–2028 是 MLSys 主要增长点。

### 主轴 B：Data 与 Code 的"中心化"反向运动（Zettlemoyer × Saroufim）

两人各占一头，但都把**"中心"**从架构/算法重新拉回到原料层。

- **Zettlemoyer**（训练侧）：先进模型能力几乎完全来自预训练数据，alignment 只是行为控制器；BPE 该被字节级架构（BLT）替代；BTM 把"哪份数据训哪些参数"做成可隔离的模块。结论：**data 才是真正的架构**。
- **Saroufim**（系统代码侧）：写代码的成本趋零，价值上移到 **可验证的基准、可复现的反馈回路**。研究人员的杠杆点不再是写更多代码，而是 **建评估和数据飞轮**。

放一起看，这两场对训练科学和系统工程同时给出了相同的方法论判断：**当生成成本变低时，验证和数据成为新的稀缺资源**。

### 主轴 C：System Intelligence（Lidong Zhou，单点突出）

Lidong 这一场野心最大，单独成轴：把 AI 从"被加速的对象"提升为"系统设计/演化的参与者"——它要重写 specification → design → validation → evolution 全生命周期。

这条主轴是对前两轴的 **元层** 总结：如果主轴 A 说"硬件/系统/模型联合迭代"、主轴 B 说"数据是新的中心"，那么主轴 C 说 **系统这门工艺学科本身正在被 AI 改造，未来会更接近一门"具有形式化骨架的科学"**。

---

## 3. 五场 keynote 之间的呼应与冲突

```
              Saroufim ─── (AI 写 kernel) ───┐
                                              │
   Zettlemoyer ─── (Data-centric) ────────────┤
                                              ├──→  Lidong Zhou (System Intelligence)
   Kozyrakis  ─── (Co-design) ────────────────┤
                                              │
              Vahdat ─── (Stack on production) ┘
```

- **Vahdat ↔ Kozyrakis**：同主题，前者偏现状披露，后者偏未来预言；互相校准。
- **Saroufim ↔ Vahdat**：一上一下——Saroufim 谈 GPU kernel 这种最底层算子的 AI 自动化；Vahdat 谈最顶层 datacenter / 网络的全栈协同。
- **Zettlemoyer ↔ Lidong**：表面上一个偏 ML（训练科学）、一个偏 Systems，但都强调"形式化/规约/可控隔离"——Zettlemoyer 的 BTM 是数据维度的隔离，Lidong 的 system intelligence 是规约维度的严谨化。
- **没有明显冲突**：5 位讲者方向互补，2026 年 MLSys 社区似乎已经在大方向（co-design + data-centric + AI-for-systems）上达成稳定共识；分歧将出现在战术层（哪种 byte-level 架构、哪种 inference 调度、哪种规约语言）。

---

## 4. 对中国 AI Infra/系统/训练社区的实操启示

1. **不要再投只优化单层的论文**：单点 attention 加速、单点 LLM serving 优化的窗口在快速关闭；建议组队覆盖 silicon ↔ system software ↔ model 至少两层。
2. **GPU kernel 自动生成是少有的"可自驱评估"赛道**：Saroufim + NVIDIA FlashInfer Contest + AWS Trainium2/3 MoE Kernel Challenge + Google Graph Scheduling Competition——MLSys 2026 三个 Competition Track 全部围绕"AI 写硬件友好代码"，是值得长期押注的方向。
3. **重新审视 tokenizer 与训练数据 pipeline**：BLT / byte-level / BTM 这条线极可能成为下一代 pretrain 的事实标准；中文/多语种长尾尤其受益。
4. **System verification + AI 是被低估的赛道**：Lidong Zhou 说出"system intelligence"这种概念名时，意味着 MSR 已经做了几年准备，国内还没有同等级别的工作产出，缺口明显。
5. **Agentic AI 推理压力是 2026–2027 服务系统主要矛盾**：长上下文、多轮工具调用、KV cache 跨请求复用——传统 vLLM/SGLang 风格的请求隔离架构正被 agentic 工作负载冲击，需要从硬件假设到调度器一起重做。

---

## 5. 待补完的内容（录像上架后回来回填）

- [ ] 5 场 SlidesLive / MLSys 官方录像直链（约 2026/06/22 起）
- [ ] Vahdat 一场的实际演讲内容（官方未发 abstract，必须以录像为准，以替换当前 03 文件中的「⚠推测」段）
- [ ] Reddit、X、HackerNews、知乎、微信公众号上的高质量讨论摘要
- [ ] Saroufim / Zettlemoyer / Kozyrakis 是否在 keynote 里发布了配套论文或开源仓库（待录像核实）

---

*— Claude · 2026/06/18 调度归档*
