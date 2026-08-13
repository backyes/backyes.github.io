# 超节点行业指标定义调研 — 经验教训

## 2026-07-16

### 任务
调研"超节点"(Super-Node) 的行业指标定义，从学术、头部互联网(云商)、制造商三视角展开。

### 成功经验

1. **"超节点=中文行业术语"这一判断节省了巨大搜索成本**：
   - 早期 Google Scholar 搜索 "super-node"+"scale-up domain" 几乎无匹配
   - 一旦确认"超节点"主要是中文产业术语，学术对应是 "rack-scale computer/scale-up domain"，立刻把搜索重心转向：
     - 中文源（华为白皮书、上海AI实验室、共熵、ZTE）
     - 英文官方产品文档（NVIDIA Enterprise RA, Google Cloud TPU docs, UALink Consortium）
     - 学术只做验证性搜索（确认术语 gap 即可）

2. **先写 research_prompt.md 框架的价值**：
   - 把研究分解为 Q1(定义)-Q2(指标)-Q3(制造商)-Q4(云商)-Q5(学术)-Q6(标准) 六个子问题
   - 调研过程就不容易跑偏，每个搜索结果都可以归类到某个 Q
   - 最终报告的目录直接对应这六个 Q

3. **华为白皮书 PDF 是最单枪匹马的高价值来源**：
   - 31 页 PDF 给出了行业<strong>唯一的量化超节点定义</strong>："32+ AI芯片, ≥400GB/s chip-to-switch, &lt;500ns, 内存统一编址"
   - 联合编写方（中国电子技术标准化研究院、GCC、国家信息中心）赋予其半官方权威
   - 用 curl 直接下载 PDF + PyMuPDF 提取文本，完全不通过 LLM 解析 PDF 内容，省 token

4. **NVIDIA 官方参考架构文档结构清晰**：
   - abstract → terminology → components 三层递进，正好分别回答"超节点是什么/术语定义/具体规格"
   - 一个 terminology appendix 就给出了 NVLink Domain / Block / Partition 三个精确定义
   - 这种"官方文档结构引导调研路径"值得复制到其他调研任务

5. **Google Cloud TPU docs 的规格表格式可直接搬运**：
   - cloud.google.com/tpu/docs/v6e 的规格表（256 chips, 234.9 PFLOPS, 800 GBps ICI）
   - 此类结构化数据用 evaluate 一次性提取，避免多次 snapshot 翻页

### 踩过的坑

1. **Google Scholar 在 Playwright 中不太稳定**：
   - 搜索 "super-node"+"scale-up domain" 只返回 1 条结果（语义太窄）
   - 需要多组关键词组合才能覆盖，但 scholarly 搜索对精确短语匹配要求高
   - 教训：学术术语检索要先确认实际用词，避免用中文思维构造英文短语

2. **知乎/新浪文章是"二次报道"，需追溯到原始白皮书**：
   - 智东西和新浪财经的文章都是对白皮书的摘要
   - 真正的量化定义在原始 PDF 中
   - 识别到这个差异后直接用 curl 下载 PDF 替代（而不是在文章中寻找细节）

3. **NVIDIA 文档在 Playwright evaluate 中比 snapshot 高效得多**：
   - nvidia 文档的 snapshot 会产生巨大的 accessibility tree（导航栏+正文总计数千行）
   - 改用 `document.querySelector('main').innerText` 一次提取，写入本地 .txt 文件
   - 后续直接从文件读取，避免多次 snapshot

4. **术语"超节点"在三家制造商处的粒度不同**：
   - NVIDIA: NVLink Domain = "全套节点通过单一 NVLink fabric 连接"
   - 华为: 超节点 = "32+ AI芯片 + ≥400GB/s + 内存统一编址"
   - Google: TPU Pod = "massive physical cluster over specialized network"
   - 三者<strong>定义方式</strong>（功能性 vs 量化 vs 描述性）差异大，报告的"层次化定义"框架解决了这个矛盾

### Token 消耗优化

- 用 PyMuPDF + bash 直接从 PDF 提取目标页（pages 7-21, 31页PDF只取~15页），不解析全文
- 用 evaluate 一次提取整页文本写入文件，而非多次 snapshot/interact
- Google Cloud 和 NVIDIA 都用 evaluate 一次性获取大块内容
- 报告采用<strong>一次 Write</strong> 写整个 HTML（~700行），而非多次 Edit
- 不启动子 agent — 单一主题的调研任务（聚焦定义）上下文连贯，拆分反而浪费

### 与前序调研的衔接关系

- ai_supernode_bus_research/ (2026.7.5) — 侧重"总线技术"（NVLink/UALink/CPO/SerDes 代际参数）
- 本次调研 — 侧重"定义与指标体系"（行业怎么定义超节点、衡量超节点）
- 两者互补：总线是超节点的"实现手段"，本次是超节点的"抽象定义"
