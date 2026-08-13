# AI超节点总线技术市场调研 - 中间数据
## 调研日期: 2026-07-05

---

## 1. NVIDIA NVLink + NVSwitch

### NVLink代际演进
| 代际 | GPU | 年份 | 每GPU链路数 | 每链路带宽 | 每GPU总带宽(双向) |
|------|-----|------|-----------|----------|----------------|
| NVLink 1.0 | Pascal P100 | 2016 | 4 | 20 GB/s | 160 GB/s |
| NVLink 2.0 | Volta V100 | 2017 | 6 | 25 GB/s | 300 GB/s |
| NVLink 3.0 | Ampere A100 | 2020 | 12 | 25 GB/s | 600 GB/s |
| NVLink 4.0 | Hopper H100/H200 | 2022 | 18 | 25 GB/s | 900 GB/s |
| NVLink 5.0 | Blackwell B200/B300 | 2024 | 18 | 50 GB/s | 1.8 TB/s |
| NVLink 6.0 | Rubin R100 | 2026 | 18 | ~66 GB/s | 3.6 TB/s (实测) |

### NVLink 6 (Rubin) 关键参数
- 每GPU 3.6 TB/s 双向带宽 (为NVLink 5.0的2倍)
- NVL72机架: 72 GPU + 36 Vera CPU, 总带宽260 TB/s
- NVL144: 144 GPU + 72 CPU, 跨两机架all-to-all mesh
- NVLink-C2C: 1.8 TB/s CPU-GPU一致性互联
- NVSwitch 6芯片: 28.8T带宽, 端口数减半但速率翻倍
- 使用400G SerDes链路
- 搭配ConnectX-9 SuperNIC (1.6 TB/s RDMA/GPU) 和 Spectrum-6 Ethernet
- Rubin Ultra NVL576 进一步扩展

### 关键来源
- NVIDIA NVLink官方页面: https://www.nvidia.com/en-us/data-center/nvlink/
- NVIDIA Developer Blog (2026.1.5): https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/
- Spheron UALink vs NVLink对比 (2026.6.24): https://www.spheron.network/blog/ualink-vs-nvlink-open-gpu-interconnect-2026/

---

## 2. UALink (Ultra Accelerator Link)

### UALink 1.0 核心规格
- 每通道200 Gbps (~25 GB/s), 信令速率212.5 GT/s
- 端口聚合: 最多4通道/端口 → 800 Gbps (100 GB/s) 单向
- 集群规模: 单scale-up域最多1,024加速器
- 延迟: <1微秒往返延迟 (机架内<4米)
- 协议: 基于IEEE开放标准, 修改的以太网物理传输层
- 支持点对点和交换拓扑
- UALoE (UALink over Ethernet): AMD Helios使用该变体

### 联盟成员 (Promoter Group, 2024年5月成立)
AMD, Broadcom, Cisco, Google, HPE, Intel, Meta, Microsoft

### AMD MI400 Helios系统
- 72 GPU per rack, UALoE互联
- AMD CES 2026.1披露: MI455X 每加速器 ~3.6 TB/s scale-up带宽
- 机架级总带宽 ~260 TB/s (与NVLink 6 NVL72相当)
- H2 2026首次量产硅, 超大规模客户认证中
- Broadcom开发UALink Switch ASIC

### UALink 2.0路线图
- 目标400 Gbps (~50 GB/s) 每通道
- 无确认发布时间, 预计2027+

### 关键来源
- UALink Consortium: https://ualinkconsortium.org/specification/
- UALink白皮书 (2025.4): https://ualinkconsortium.org/wp-content/uploads/2025/04/UALink-1.0-White_Paper_FINAL_UPDATED.pdf
- Spheron对比 (2026.6.24): https://www.spheron.network/blog/ualink-vs-nvlink-open-gpu-interconnect-2026/
- iThome中文分析: https://www.ithome.com.tw/tech/173039

---

## 3. NVIDIA NVL72/NVL144 超节点拓扑

### NVL72 (Vera Rubin)
- 72 Rubin GPU + 36 Vera CPU per rack
- All-to-all non-blocking mesh via NVSwitch 6
- 总带宽: 260 TB/s rack-level
- 每GPU: 3.6 TB/s all-to-all
- 液冷: 先进液体冷却 (高kW密度)
- Compute Tray拓扑类似GB200/GB300

### NVL144
- 144 GPU + 72 CPU, 两机架统一内存空间
- 多层NVSwitch逻辑跨机架扩展NVLink fabric

### Scale-Out互联
- Spectrum-X Ethernet 或 InfiniBand 连接多机架
- CPO技术: 100+ TB/s per switch for east-west traffic

### 关键来源
- NVIDIA NVL72 AI Factory文档: https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/
- SemiAnalysis深度分析: https://newsletter.semianalysis.com/p/vera-rubin-extreme-co-design-an-evolution

---

## 4. Ultra Ethernet Consortium (UEC) vs InfiniBand

### InfiniBand XDR (2026)
- 每端口800 Gb/s (4x200G通道)
- 聚合: 最高1.6 Tb/s per link (SHARP)
- 延迟: 0.6-1.2μs (确定性, credit-based flow control)
- 适合: 紧耦合GPU集群, 数千节点

### Ultra Ethernet (UEC 1.0/1.0.2)
- 基于IEEE 802.3dj (200G/lane)
- 每端口: 800 Gb/s, 早期1.6 Tb/s部署中
- 延迟: 亚微秒级 (随规模变化)
- 控制: Packet Delivery Sublayer (PDS) + 动态负载均衡
- 适合: 百万级端点, 多租户云环境
- 成员: AMD, Broadcom, Cisco, Google, HPE, Intel, Meta, Microsoft等

### 关键对比
- InfiniBand性能领先15%, 但成本2.3x
- UEC开放生态 vs InfiniBand专有生态
- NVIDIA Spectrum-X作为专有以太网替代方案

### 关键来源
- Introl Blog对比: https://introl.com/blog/infiniband-vs-ethernet-gpu-clusters-800g-architecture
- Spheron网络指南: https://www.spheron.network/blog/gpu-networking-infiniband-roce-spectrum-x-guide/
- Network World (2026.1.7): https://www.networkworld.com/article/4113364/ethernet-groups-keep-2026-focus-on-higher-bandwidth-ai-demands.html

---

## 5. CPO (光电共封装) 与硅光子

### 市场现状 (2026)
- CPO进入全面量产阶段, 2026年被视为CPO元年
- 市场规模估计: ~200亿美元 (labo-llm.fr估算)
- 功耗效率比铜缆提升5-10倍

### 关键玩家
- **TSMC COUPE**: 2026年4月量产, SoIC-X 3D堆叠电子/光子IC, 200 Gb/s光信号调制
- **NVIDIA Spectrum-X Photonics**: CPO交换机已出货, NVLink Fusion扩展光互联
- **Ayar Labs**: TeraPHY光引擎 (UCIe兼容), 每引擎8 Tb/s, 5亿美元E轮融资(估值37.5亿)
- **Lightmatter**: 光子计算+互联, 进入量产
- **Broadcom**: CPO产品2025-2026发布

### 关键来源
- 三井物产报告: https://www.mitsui.com/mgssi/en/report/detail/__icsFiles/afieldfile/2026/04/01/2601bt_tsuji_e.pdf
- NextWave Insight: https://nextwavesinsight.com/photonic-compute-production-lightmatter-ayar-labs/
- 北美智权中文分析: https://naipnews.naipo.com/42316/

---

## 6. 技术对比矩阵

| 维度 | NVLink 6 | UALink 1.0 | Infinity Fabric (AMD) |
|------|---------|-----------|---------------------|
| 类型 | 专有 | 开放标准 | 专有(通过UALoE开放) |
| 每通道带宽 | ~66 GB/s | 200 Gbps (~25 GB/s) | - |
| 每GPU总带宽 | 3.6 TB/s | ~3.6 TB/s (MI455X) | ~3.6 TB/s (MI455X) |
| 最大GPU数 | 144 (NVL144) | 1,024 | 72 (Helios rack) |
| 延迟 | ~1-2μs (NVLink 4实测) | <1μs (规格目标) | 取决于UALoE实现 |
| 交换机 | NVSwitch (专有ASIC) | 开放规格(Broadcom等) | 开放规格 |
| 量产状态 | 2026 Rubin | H2 2026 | H2 2026 |
| 跨厂商 | 否 | 是 | 是(通过UALink) |

---

## 7. 待补充方向 (第二轮搜索)
- CXL 3.0/3.1在AI超节点中的角色
- UCIe die-to-die互联标准
- 华为昇腾HCCS互联方案
- 国产GPU互联方案(寒武纪/壁仞/燧原)
- AI互联芯片市场规模预测
- SerDes 224G/448G技术进展
