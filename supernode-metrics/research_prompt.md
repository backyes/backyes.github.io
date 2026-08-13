# 超节点行业指标定义 — 研究 Prompt 框架

> 调研日期: 2026-07-16
> 目标: 从学术、头部互联网(超大规模云商)、头部制造商三个视角，回答一个问题：
> **行业如何定义"超节点"(Super-Node / Super Node)，并用哪些指标(指标体系)衡量它？**

---

## 1. 核心术语界定（调研之前置共识）

"超节点"不是标准学术术语，中文语境下它被不同参与者赋予不同含义：
- **制造商语境**（华为昇腾、NVIDIA DGX）：rack-scale 一体化算力单元，跨GPU统一内存/高速互联构成"一台计算机"
- **超大规模云商语境**（Google TPU Pod、Meta、MSFT）：最小可调度/可故障隔离的"逻辑超级计算机"
- **学术语境**：scale-up domain、rack-scale computer、tightly-coupled GPU ensemble 等
- **运营商/智算中心语境**：智算中心交付单元（一个"超节点"= 一个交付/计费/部署单元）

⚠️ 注意区分：超节点(super-node) vs 超级计算机(supercomputer) vs 智算集群(cluster) vs 节点(node)。

---

## 2. 研究子问题 (Sub-questions)

### Q1 定义维度：什么构成一个"超节点"？
- 硬件边界：一个机架？一个 NVLink domain？一个 pod slice？
- 内存语义：统一共享内存(UVA/SVM) vs 分布式内存——这是否是超节点的必要条件？
- 拓扑约束：all-to-all / torus / dragonfly / fat-tree——哪些拓扑被认定为"超节点"？
- 管理边界：单一故障域？单一调度域？单一租户域？
- 规模阈值：GPU/加速器数量 >= N？互联带宽 >= X TB/s？

### Q2 指标体系：行业用哪些指标衡量超节点？
- **规模指标**：#GPUs, #CPUs, 总算力 (PFLOPS/EFLOPS), 总内存 (TB), 总存储
- **互联指标**：per-GPU bandwidth (TB/s), aggregate bisection bandwidth (TB/s), 拓扑直径, 对分带宽比
- **算力指标**：FP8/FP16/BF16/FP32 峰值 & 实测 (MFU/HBM-bandwidth-utilization)
- **能效指标**：PUE(机架级), kW/rack, FLOPS/W, 冷却方式
- **可靠性指标**：MTBF, 故障域大小, check-point 粒度
- **调度指标**：最小分配粒度, 弹性伸缩边界, 队列调度单位
- **经济指标**：$, TFLOPS/$, TCO/Token

### Q3 制造商视角（NVIDIA / AMD / 华为昇腾 / Intel Gaudi）
- 各家的产品命名如何隐含"超节点"定义？（NVL72/NVL144, DGX SuperPOD, CloudMatrix 384, AMD Helios）
- 官方技术规格中哪些指标被作为"超节点"的 core spec？
- 各家如何划分 scale-up (节点内) vs scale-out (节点间) 边界？

### Q4 头部互联网/超大规模云商视角
- Google: TPU Pod / TPU slice 的定义与指标
- Meta: MTIA / "Grand Teton" rack 的指标公开表述
- Microsoft/OpenAI: "AI flagship" 集群的披露指标
- Amazon: Trainium2 Trn2 超节点定义
- ByteDance/阿里/腾讯: 公开披露的超节点规格（若有）
- 关键：云商如何看待"超节点"的故障域、调度域、容量规划单位？

### Q5 学术视角
- 关键论文对"rack-scale computing""scale-up domain""GPU ensemble"的定义性表述
- 顶会（ISCA/HPCA/MICRO/SC/OSDI/SOSP/SoCC/MLSys/HPDC）近5年相关文章
- "Super-node" 在拓扑/路由/调度文献中的使用
- 定义性/综述性论文、教科书表述

### Q6 行业标准与智算中心交付
- OCP（Open Compute Project）对 rack/node 的规范
- 中国智算中心的"超节点"采购/交付标准
- UALink Consortium 的 scale-up domain 定义（≤1024 加速器）
- 运营商/IDC 对"超节点"的商业定义

---

## 3. 调研方法论

1. **搜索关键词** (中英双语):
   - "super-node" "AI" "definition" "scale-up domain" "rack-scale computer"
   - "GPU super node" "cluster architecture" metrics benchmark
   - "超节点" "智算" "集群" "指标体系" "规模" "互联"
   - NVIDIA NVL72/NVL144 spec, Google TPU v5p/v6e Pod spec, Huawei CloudMatrix
   - "MFU" "model flops utilization" "super-node efficiency"

2. **执行约束** (遵循项目规则):
   - 用 Playwright 做联网搜索 (rule#1)
   - 所有访问过的 URL 落盘 (sources/*.md) (rule#13/#17)
   - 关键原始信息保留原文语言 (rule#12)
   - 尽量本地脚本批量处理，减少大模型 API 交互 (rule#11)
   - 最终输出手机友好的 HTML 报告，含可溯源超链接 (rule#5)
   - 受众：AI 算法/系统/芯片研究人员 (rule#6)

---

## 4. 预期输出结构

- `supernode_metrics_report.html` — 主报告
- `sources/manufacturer/*.md` — 制造商原始素材
- `sources/hyperscaler/*.md` — 云商原始素材
- `sources/academia/*.md` — 学术原始素材
- `sources/media/*.md` — 行业媒体/分析机构素材
- `all_urls.md` — 所有访问过的 URL（含失败/成功）
- `process_log.md` — 调研关键过程记录
