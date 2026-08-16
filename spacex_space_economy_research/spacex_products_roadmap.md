# SpaceX第二轮深度调研笔记：前沿产品、规划、政府合同与财务花费

> 调研日期：2026-07-11
> 调研方法：Playwright Google搜索4组关键词，提取AI Overview + 权威来源
> 注：本轮因Agent卡住，改为主研究员直接Playwright搜索

---

## 一、前沿研究与产品矩阵

### 1.1 Starlink V3（下一代卫星）
- 每颗约4,400磅（~2吨），容量**1 Tbps/卫星**
- 多千兆对称速率、超低延迟，面向消费者和AI设备
- Starship V3一次可发射**60颗V3卫星**（单次发射新增60 Tbps容量）
- 2026下半年开始部署
- 来源：Data Center Dynamics、Space.com、Ars Technica、satnews.com

### 1.2 星座规模
- 截至2026.6约**10,413颗**在轨（10,397颗运行）— Wikipedia/Space.com
- 将现有卫星移至更低480km轨道改善空间安全
- FCC申请目标**100,000+总卫星规模**
- 之前报告的42,000颗是早期规划，已扩展

### 1.3 Starship V3
- 2026年首飞（瞄准4月窗口）
- 全复用LEO运力**100+吨**
- 使Starlink V3大规模部署成为可能
- 来源：KeepTrack、satnews.com、Sacra

### 1.4 轨道数据中心（重磅）
- **SpaceX收购xAI后**，向FCC申请发射多达**100万颗卫星**作为轨道数据中心
- 2026年开始在Starlink V3硬件上试点在轨计算节点
- 利用V3组件量产，在LEO托管空间AI计算载荷
- 通过空间激光链路直接路由企业和AI推理工作负载
- 与Google、Anthropic合作（多头分析师观点）
- 来源：introl.com (2026.2.3)、Sacra、Payload Space

### 1.5 Starshield（军事业务）
- SpaceX政府/军事/国家安全专用网络
- 直接承接轨道数据中心和通信能力
- 来源：sacra.com

---

## 二、几年规划（2026-2030路线图）

### 2.1 战略重心转移：月球优先
- Musk将短期重点转向**2030年代初建立自给自足月球基地**
- 同时瞄准2026/2027地火转移窗口进行无人货运
- Musk论月球：月球短的可重复发射节奏（约每10天一次任务，2天往返）允许更快进展
- 来源：Instagram (spacex_spacenews, 2026.2.9)、Mashable

### 2.2 月球路线图（Artemis HLS）
- **2027年初**：无人Starship HLS演示飞行（NASA+SpaceX目标）
- **2027年6月**：无人Starship登月（报道）
- **2028年9月**：首次载人月球任务
- NASA要求载人前进行两艘Starship之间长时间在轨推进剂转移测试
- 在轨加注使Starship可向月面运送100吨
- 来源：SpaceX Updates (spacex.com/updates, 2026.5.12)、Facebook Space FrontPage、Space.com

### 2.3 发射节奏目标
- 瞄准**8天一次发射节奏**（支持月球运营）
- 需多个发射台：Starbase德州 + 佛州SLC-37规划塔
- 最多**15次加油飞行**才能在轨加满HLS Starship
- Gwynne Shotwell概述路线图，Starship发射计划日益雄心勃勃
- 来源：YouTube NASASpaceflight (125K+ views, 3周前)

### 2.4 火星时间表
- **2026年底**地火轨道对齐窗口：5艘无人Starship发射（Aerospace America）
- 携带物资和类人机器人（Tesla Optimus）准备地面基础设施
- 火星任务前提：Starship V3 + LEO无缝转移超冷推进剂
- 来源：Mashable、Yahoo、Aerospace America (aerospaceamerica.aiaa.org/aiaa-spacex/)

### 2.5 Starship试飞状态
- 截至2026.5.27：12次试飞（7成功，5失败）
- Block 2上面级2025年前四次试飞全部失败
- 来源：Wikipedia SpaceX Starship

---

## 三、美国政府合同与工作报告

### 3.1 累计政府合同
- **累计联邦合同约$22 billion**（NASA、Space Force、NRO、SDA）
- 来源：fed-spend.com (2026.3.17)

### 3.2 近期重大国防合同
- **$4.16 billion** Space Force合同（威胁探测卫星星座，2026.5.29）— SpaceNews/Reuters
- **$2.29 billion** Space Force合同（军事空间数据网络，固定价格，2026.5.27）— Morningstar/Yahoo Finance
- NASA增加6次载人任务

### 3.3 政府收入占比（S-1披露）
- 2025年$18.7B营收中，**约1/5（~$3.7B）来自美国政府合同**
- 来源：Washington Technology (2026.5.20)、SEC S-1

### 3.4 S-1政府依赖表述
- "We derive significant revenue from U.S. government contracts that are subject to competitive bidding, funding approvals and other..."
- 来源：SEC.gov S-1

---

## 四、财务花费与资本配置（S-1深度）

### 4.1 亏损结构
- 2025年净亏损**$4.9B**（GAAP $4.94B），营收$18.67B
- **Q1 2026单季净亏损$4.28B**
- 2024年净亏损$4.6B（Reddit披露）
- 累计亏损$41.3B
- 亏损驱动：Starship R&D + 新收购xAI基础设施快速整合
- 来源：CNBC、Morningstar、Moomoo、Evest

### 4.2 资本开支（CapEx）— 重磅
- 2025年CapEx **$20.7B**
- **Q1 2026单季CapEx $10.1B**（年化~$40B）
- 其中约**$7.7B直接用于AI**
- CapEx严重倾向AI计算和基础设施
- 现金从2025年底$24.7B下降
- 来源：Morningstar、LinkedIn Thomas Li、Market Research Reports

### 4.3 R&D开支
- 核心航空和Starship R&D约**$3B/年**
- 对近期自由现金流造成重大压力
- 来源：Leverage Shares

### 4.4 Starlink盈利能力（现金引擎）
- 2025营收$11.4B
- 分部运营收入$4.4B
- **EBITDA利润率63%**
- 用户超10.3M（Q1 2026）
- **ARPU下降至约$66/月**（容量需求上升、用户稀释导致）
- 来源：Market Research Reports、LinkedIn Thomas Li

### 4.5 xAI收购与AI烧钱
- SpaceX收购xAI
- xAI部门**Q1 2026单季烧约$2.5B**
- AI基础设施投入是亏损主因
- 来源：Morningstar、Payload Space

---

## 五、投资多空分析

### 5.1 多头逻辑
- 分部加总优势
- Starlink + 轨道数据中心（与Google、Anthropic合作）
- 目标2026年底**$50B季度营收run rate**
- 来源：Payload Space

### 5.2 空头逻辑
- 极端资本密集度
- xAI部门现金消耗大（Q1 2026单季$2.5B）
- 近期自由现金流堪忧
- 估值溢价相比传统航天/连接同行过高
- S&P 500要求连续4季度GAAP盈利，但Q1 2026单季亏$4.28B
- 来源：Morningstar、Moomoo

### 5.3 关键财务指标汇总
| 指标 | 数值 |
|------|------|
| 2025营收 | $18.67B (+33%) |
| 2025 GAAP净亏损 | $4.94B |
| Q1 2026净亏损 | $4.28B |
| 2025调整后EBITDA | $6.58B |
| 2025 CapEx | $20.7B |
| Q1 2026 CapEx | $10.1B (其中AI $7.7B) |
| Starlink 2025营收 | $11.4B |
| Starlink EBITDA利润率 | 63% |
| Starlink运营利润 | $4.4B |
| Starlink用户 | 10.3M |
| Starlink ARPU | ~$66/月 |
| 累计亏损 | $41.3B |
| 现金(2025底) | $24.7B |
| 政府收入占比 | ~1/5 (~$3.7B) |

---

## 六、数据来源URL清单

### 成功访问
1. Google搜索 - Starship路线图: https://www.google.com/search?q=SpaceX+Starship+2026+2027+roadmap...
2. Mashable - Musk火星更新: https://mashable.com/article/elon-musk-mars-update-key-takeaways-spacex-starship-2026
3. SpaceX Updates: https://www.spacex.com/updates
4. Space.com - 2028登月: https://www.space.com/space-exploration/artemis/nasa-wants-to-land-astronauts-on-the-moon-in-2028-will-spacexs-starship-or-blue-origins-blue-moon-lander-be-ready-in-time
5. Aerospace America - SpaceX火星计划: https://aerospaceamerica.aiaa.org/aiaa-spacex/
6. Wikipedia Starship: https://en.wikipedia.org/wiki/SpaceX_Starship
7. fed-spend.com - $22B政府合同: https://fed-spend.com/Blog
8. SpaceNews - $4.16B合同: https://spacenews.com/space-force-awards-spacex-4-1...
9. Reuters - $4.16B合同: https://www.reuters.com/science/us-space-force-awa...
10. Morningstar - $2.29B合同: https://www.morningstar.com/news/20260526349/s...
11. Washington Technology - S-1政府分析: https://www.washingtontechnology.com/2026/05/sp...
12. SEC S-1: https://www.sec.gov/Archives/edgar/data/spacee...
13. Introl - 轨道数据中心: https://introl.com/Blog
14. Wikipedia Starlink: https://en.wikipedia.org/wiki/Starlink
15. Sacra - SpaceX分析: https://sacra.com/research/spacex-at-15b-yr-growing-...
16. 36Kr - Musk X发布: https://eu.36kr.com/...
17. KeepTrack - Starship V3: https://keeptrack.space/...
18. CNBC - IPO直播: https://www.cnbc.com/2026/06/12/spacex-ipo-spcx-...
19. Morningstar - IPO分析: https://global.morningstar.com/en-nd/stocks/spacex...
20. Market Research Reports - SPCX $2T报告
21. Moomoo - IPO分析: https://www.moomoo.com/community/feed/spcx-in...
22. BitMEX - IPO指南: https://www.bitmex.com/.../Trading Guides
23. Evest - IPO: https://www.evest.com/trading-blog/spacex-ipo
24. LinkedIn Thomas Li - IPO分析
25. YouTube NASASpaceflight - Starship计划
26. Reddit r/ArtemisProgram - 财务分析

### 调研执行动作
1. 搜索1 (Starship 2026-2027 roadmap Mars Artemis HLS): 提取AI Overview完整内容—月球优先战略、8天发射节奏、15次加油、Artemis时间表
2. 搜索2 (NASA DoD Space Force contract NSSL Artemis Starshield): 提取$22B累计合同、$4.16B+$2.29B新合同、S-1政府收入1/5
3. 搜索3 (Starlink V3 42000 Starshield Starship V3 orbital data center): 提取V3 1Tbps/卫星、60颗/发射、100万颗轨道数据中心、xAI收购、Starshield
4. 搜索4 (S-1 IPO CapEx R&D net loss Starlink profitability bull bear): 提取CapEx $20.7B/Q1 $10.1B、xAI烧$2.5B、Starlink EBITDA 63%、多空观点
