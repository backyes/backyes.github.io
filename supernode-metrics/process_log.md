# 调研关键过程记录

## 调研日期: 2026-07-16

## 第一轮：全景搜索
- Google 搜索 "超节点 AI 智算 定义 指标 体系" → 发现上海AI实验室白皮书、华为白皮书、共熵系列、H3C、ZTE 等核心来源
- 确认"超节点"是中文行业术语，学术对应概念为 rack-scale computer / scale-up domain / super-node (interconnect topology)

## 第二轮：核心来源采集
1. **上海AI实验室《超节点技术体系白皮书》** (知乎/新浪财经, 2026.3.31)
   - 关键信息：超节点 = 超高带宽低延迟芯片间互联，将数十至上百颗计算芯片构建成逻辑高度协同的"超级计算单元"
   - 产业共识：算力竞争从芯片级转向系统级（互联+软件+整机+RAS）
   - 缺少行业共认的评价框架 → 白皮书解决"认知"第一步

2. **华为云《超节点发展报告》** (PDF, 31页, 2026)
   - 联合编写：中国电子技术标准化研究院、GCC全球计算联盟、国家信息中心
   - 序言：郑纬民院士、杨超斌、魏亮
   - **最核心的量化定义**：
     - "超节点是AI计算节点通过高速互联协议组成更大内存空间的AI系统"
     - "超节点可以支持32及以上AI芯片，AI芯片到交换芯片带宽不小于400GB/s，交换设备时延小于500ns"
     - "超节点域内AI芯片支持内存统一编址，AI芯片使用内存语义可直接访问其他AI芯片的内存"
   - 技术特征分层：基础特征（大带宽低时延、内存统一编址）+ 扩展特征（多级缓存池化、资源灵活配比）
   - 系统特征：超大规模、超高可靠、灵活切分
   - 产品：昇腾384超节点 → 下一代8192

3. **共熵(产业与标准创新)服务中心** (2026.3.18)
   - 综合效能基准指标体系：算力、网络、存储、稳定性与能效、运维 五大维度
   - 千卡超节点成为新基准，万卡超节点竞赛开启
   - 五大趋势：协议收敛、规模突破、互联革新、生态开放、软件适配

## 第三轮：制造商官方定义
4. **NVIDIA NVL72 AI Factory 官方文档** (docs.nvidia.com)
   - **Scalable Unit (SU)** = 18 compute nodes + 72 GPUs（超节点构建块）
   - **NVLink Domain** = 通过单一多节点 NVLink fabric 连接的全套节点
   - **NVLink Block** = domain 内分配给同一作业的节点组
   - **NVLink Partition** = domain 内节点互相访问 GPU 内存的隔离组
   - GB300 NVL72 规格：9 NVSwitch trays, 130 TB/s 聚合带宽, 142 kW/rack, 液冷
   - 每 GPU 1800 GB/s NVLink, 400-800 GB/s 以太网计算网络
   - 规模：2 Racks/144 GPUs → 4 Racks/288 → 8 Racks/576

5. **Google Cloud TPU v6e 官方文档** (docs.cloud.google.com)
   - **TPU Pod** = a massive physical cluster of TPU chips connected over specialized high-speed network
   - TPU v6e Pod = 256 chips, BF16 234.9 PFLOPS/Pod, all-reduce 102.4 TB/s, bisection 3.2 TB/s
   - 每 chip: 918 TFLOPS (bf16), 32 GB HBM, 1638 GBps HBM带宽, 800 GBps ICI (4 ports)
   - 拓扑：2D torus
   - TPU v5p Pod = 8960 chips; TPU v5e Pod = 256 chips, all-reduce 51.2 TB/s

6. **UALink Consortium 官方** (ualinkconsortium.org)
   - **scale-up domain** = single-hop rack-scale domain up to 1,024 accelerators per pod
   - UALink 1.0: 200 Gbps/lane, 信令 212.5 GT/s, <1μs RTT
   - 2026.1 白皮书更新版

## 第四轮：学术视角
7. Google Scholar 搜索确认 "super-node" + "scale-up domain" 学术匹配极少
   - 对应学术概念：rack-scale computer, scale-up domain, super-node (interconnect topology layer), GPU ensemble
   - IEEE 论文 "Research on the Scale-Up Network for AI" (2025) 提到 super-node cabinets
   - MDPI survey 将 super node 作为拓扑第一层
   - 结论：超节点主要是中文行业/产业术语，学术界用 rack-scale / scale-up domain / scale-up network

## 关键发现
- "超节点"定义存在三个层次：制造商产品定义、行业标准定义、学术概念定义
- 各家量化指标差异大，缺少统一评价框架（上海AI实验室白皮书正在解决）
- 核心共识指标：加速器数量、per-GPU带宽、聚合带宽、内存统一编址、时延、算力(PFLOPS)、能效(kW/rack)
