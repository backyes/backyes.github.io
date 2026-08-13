# DeepSeek MTP & DSpark 调研中间笔记

## 调研时间：2026-07-05

---

## 一、技术原理

### 1.1 MTP (Multi-Token Prediction)
- **来源**: DeepSeek-V3 论文 (arXiv:2412.19437), DeepSeek-V4 论文 (arXiv:2606.19348)
- **核心思想**: 不同于传统自回归模型只预测下一个token，MTP在同一个模型主干上使用辅助预测头，一次性预测多个未来token（T+1, T+2, T+3...）
- **效果**: 
  - 训练信号更密集，收敛更快
  - 迫使模型捕获更深层的上下文依赖关系
  - 显著提升样本效率
- **在V4中的应用**: DeepSeek-V4系列同样设置了MTP模块和目标函数

### 1.2 DSpark (Speculative Decoding Framework)
- **发布时间**: 2026年6月27日
- **作者**: DeepSeek-AI + 北京大学 (含梁文锋)
- **论文**: "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
- **开源**: MIT licensed (DeepSpec训练代码 + DSpark checkpoints)

**核心技术架构**:
1. **半自回归生成 (Semi-Autoregressive Generation)**:
   - 并行骨干网络(DFlash) + 轻量级序列头(Markov Head, rank-256)
   - 解决并行drafter的"多模态碰撞"和suffix acceptance decay问题
   - 继承并行骨干的高首token准确率，序列头保持深层block的接受率稳定

2. **置信度调度验证 (Confidence-Scheduled Verification)**:
   - 置信度头输出每个draft位置的存活概率估计
   - Sequential Temperature Scaling进行后校准（将校准误差从3-8%降至约1%）

3. **硬件感知前缀调度器**:
   - GPU空闲时验证更多token，繁忙时验证更少
   - 基于预测量的吞吐量曲线SPS(B)
   - 使用早停规则保持无损

**性能数据**:
- 离线测试: accepted length比Eagle3提升26-31%, 比DFlash提升16-18%
- 生产环境(DeepSeek-V4): 
  - V4-Flash: 60-85% per-user generation加速
  - V4-Pro: 57-78% per-user generation加速
- @danielhanchen: throughput提升51%到400%

---

## 二、投资与算力市场影响

### 2.1 Motley Fool分析 (2026.07.01)
**标题**: "DeepSeek's DSpark Just Made Nvidia's Most Important New Bet Harder to Close"
- DSpark威胁NVIDIA的利润池
- NVIDIA正在尝试在GPU之上销售专门的解码硬件/机架
- DSpark作为免费开源方案，原生解决了解码延迟问题
- 使NVIDIA推销高端解码机架变得更困难
- 超大规模云厂商正在自建替代方案（AWS Trainium + Cerebras CS-3）

### 2.2 36Kr分析 (2026.07)
**标题**: "400 billion DeepSeek, how to spend the 50 billion raised?"
- DeepSeek完成510亿元首轮外部融资，估值近4000亿元
- 打破梁文锋"不融资、不上市、不商业化"原则
- 原因：人才留存压力（竞对如智谱市值近1万亿港元，MiniMax超1300亿港元）
- 资金用途：数据中心基础设施（IDC团队在乌兰察布）、人才招聘（33个岗位7大类）
- 行业专家潘和林："行业已进入重资本阶段"

### 2.3 BigGo财经
- DeepSeek V4上市首日即获华为Ascend NPU抢先支持
- 跨平台开源核心：NVIDIA GPU和华为Ascend NPU均带来1.5-1.73倍原始速度提升
- 永久性75%降价 + 高速迭代

### 2.4 鉅亨網
**标题**: "DeepSeek V4 如何以「淪為二流」的風險換取中國AI 算力主權？"
- DeepSeek V4参数规模1.6万亿，含V4-Pro和V4-Flash两个版本
- 原生支持百万token超长上下文
- 以"可能沦为二流"的风险换取中国AI算力主权

### 2.5 科技產業資訊室 (台湾)
**标题**: "DeepSeek V4觸發的全球AI雙軌格局與Nvidia的生存策略"
- 全球AI双轨格局：中国AI生态 vs 西方AI生态
- NVIDIA全面优化全球物流与供应链
- 面临关税压力或更高级别禁运

---

## 三、社交媒体讨论

### 3.1 X/Twitter
- **@danielhanchen** (3560+赞): DSpark boosting throughput 51-400%
- **@teortaxesTex** (290+赞): "Naively serving Transformers is not going to cut it in 2026"
- **@echo_vic**: "DSpark重点不是新模型，而是推理系统能力"
- **@AlphaSignalAI**: 详细技术解读
- **@Marktechpost** (30+赞): 开源框架报道
- **@lianyanshe**: "AI算力困局与破局：从算力优化到推理加速"
- **@thePandaily**: 梁文锋融资后首篇论文

### 3.2 Reddit r/LocalLLaMA
- 160+评论的热门讨论
- 核心观点: "No hyping and overselling GPU compute as exotic intelligence at 95-98% profit margin"

---

## 四、关键数据汇总

| 指标 | 数值 |
|------|------|
| DSpark推理加速 | 60-85% (per-user) |
| Throughput提升 | 51-400% |
| 接受长度提升(vs Eagle3) | 26-31% |
| 接受长度提升(vs DFlash) | 16-18% |
| DeepSeek融资额 | 510亿元 |
| DeepSeek估值 | ~4000亿元 |
| V4参数规模 | 1.6万亿 |
| V4训练token | 14.8万亿 |
| V3训练GPU时 | 278.8万 H800 GPU时 |
| 跨平台加速(NVIDIA+华为) | 1.5-1.73x |
| API降价 | 永久性75% |
