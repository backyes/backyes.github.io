# NVIDIA 官方规格 PDF / Datasheet 调研报告

> 调研日期：2026-07-31
> 调研目标：收集 NVIDIA 最权威的官方产品规格数据（datasheet / whitepaper / product brief）
> 调研方法：通过 Playwright 搜索 Google 获取官方 PDF 链接，直接下载并用 PyPDF2 提取规格

---

## 调研摘要

| 类别 | 找到的文档数 | 成功下载 PDF 数 | 提取到完整规格数 |
|------|------------|---------------|----------------|
| GPU Datasheet | 10 | 8 | 7 |
| 系统产品规格 | 6 | 5 | 5 |
| 网络产品 | 5 | 4 | 4 |
| 汽车/嵌入式 | 5 | 4 | 3 |
| 架构白皮书 | 5 | 4 | 4 |
| **合计** | **31** | **25** | **23** |

---

## 一、GPU Datasheet

### 1.1 NVIDIA H100 Tensor Core GPU

**文档来源**：
- 主规格来源：NVIDIA Hopper Architecture In-Depth 技术博客（含完整 H100 规格）
- URL：https://lists.riscv.org/g/tech-attached-matrix-extension/attachment/25/0/NVIDIA%20Hopper%20Architecture%20In-Depth%20_%20NVIDIA%20Technical%20Blog.pdf
- 备用：第三方托管的 H100 datasheet（megware）
- URL：https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf

**版本日期**：2022 年 3 月（GTC 2022 发布）

**关键规格表格**：

| 规格项 | H100 SXM5 | H100 PCIe Gen5 |
|--------|-----------|----------------|
| **架构** | NVIDIA Hopper (GH100) | NVIDIA Hopper (GH100) |
| **制程** | TSMC 4N | TSMC 4N |
| **晶体管** | 800 亿 | 800 亿 |
| **Die 尺寸** | 814 mm² | 814 mm² |
| **SM 数量** | 132 SMs | 114 SMs |
| **FP32 CUDA Cores** | 16,896 | 14,592 |
| **Tensor Cores (第4代)** | 528 | 456 |
| **FP64** | 30 TFLOPS | 24 TFLOPS |
| **FP64 Tensor Core** | 60 TFLOPS | 48 TFLOPS |
| **FP32** | 60 TFLOPS | 48 TFLOPS |
| **TF32 Tensor Core** | 500 / 1000* TFLOPS | 400 / 800* TFLOPS |
| **BF16 Tensor Core** | 1000 / 2000* TFLOPS | 800 / 1600* TFLOPS |
| **FP16 Tensor Core** | 1000 / 2000* TFLOPS | 800 / 1600* TFLOPS |
| **FP8 Tensor Core** | 2000 / 4000* TFLOPS | 1600 / 3200* TFLOPS |
| **INT8 Tensor Core** | 2000 / 4000* TOPS | 1600 / 3200* TOPS |
| **GPU 内存** | 80 GB HBM3 | 80 GB HBM2e |
| **内存带宽** | 3.35 TB/s | 2.0 TB/s (⚠️ 估计) |
| **L2 Cache** | 50 MB | 50 MB |
| **NVLink** | 第4代，900 GB/s | 第4代，900 GB/s |
| **PCIe** | Gen5, 128 GB/s | Gen5, 128 GB/s |
| **TDP** | 700W (可配置) | 350W (⚠️ 估计) |
| **MIG 支持** | 最多 7 个 MIG @ 10GB | 最多 7 个 MIG @ 10GB |

> *With sparsity（结构化稀疏加速）
> ⚠️ 部分 PCIe 规格为估计值，官方未发布完整 PCIe 版 datasheet

**H100 GPU 架构细节（GH100 Full GPU）**：
- 8 GPCs, 72 TPCs, 144 SMs（全核心版本）
- 128 FP32 CUDA Cores per SM
- 6 HBM3 stacks, 12 个 512-bit 内存控制器
- 60 MB L2 cache

---

### 1.2 NVIDIA H200 Tensor Core GPU

**文档来源**：
- URL：https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/NVIDIA_H200_Datasheet.pdf
- 文档编号：3367400.JUL24

**版本日期**：2024 年 7 月

**关键规格表格**：

| 规格项 | H200 SXM | H200 NVL |
|--------|----------|----------|
| **FP64** | 34 TFLOPS | 34 TFLOPS |
| **FP64 Tensor Core** | 67 TFLOPS | 67 TFLOPS |
| **FP32** | 67 TFLOPS | 67 TFLOPS |
| **TF32 Tensor Core** | 989 TFLOPS | 989 TFLOPS |
| **BF16 Tensor Core** | 1,979 TFLOPS | 1,979 TFLOPS |
| **FP16 Tensor Core** | 1,979 TFLOPS | 1,979 TFLOPS |
| **FP8 Tensor Core** | 3,958 TFLOPS | 3,958 TFLOPS |
| **INT8 Tensor Core** | 3,958 TFLOPS | 3,958 TFLOPS |
| **GPU 内存** | 141 GB HBM3e | 141 GB HBM3e |
| **内存带宽** | 4.8 TB/s | 4.8 TB/s |
| **解码器** | 7 NVDEC + 7 JPEG | 7 NVDEC + 7 JPEG |
| **机密计算** | 支持 | 支持 |
| **TDP** | 最高 700W（可配置） | 最高 600W（可配置） |
| **MIG** | 最多 7 个 @ 16.5GB each | 最多 7 个 @ 16.5GB each |
| **形态** | SXM | PCIe |
| **互连** | NVLink: 900GB/s + PCIe Gen5: 128GB/s | 2-或4-way NVLink bridge: 900GB/s + PCIe Gen5: 128GB/s |
| **NVIDIA AI Enterprise** | 可选附加 | 包含 |

**关键特性**：
- 首款提供 141GB HBM3e 的 GPU，内存带宽 4.8 TB/s
- 相比 H100：内存带宽 1.4X，LLM 推理性能 2X
- FP8 性能达 4 petaFLOPS

---

### 1.3 NVIDIA B200 / B100 Tensor Core GPU

**文档来源**：
- HGX B200 PCF Summary（含规格）：https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf
- B100 Spec Sheet（第三方）：https://flopper.io/gpu/nvidia-b100-sxm-192gb/spec-sheet.pdf
- Blackwell Architecture Technical Brief（镜像）：https://cdn.prod.website-files.com/...nvidia-blackwell-architecture-technical-brief.pdf

**版本日期**：2024 年（Blackwell 架构）

**HGX B200 Baseboard 规格**：

| 规格项 | HGX B200 |
|--------|----------|
| **GPU 数量** | 8x NVIDIA Blackwell B200 (SXM6) |
| **FP4 Tensor Core** | 144 PFLOPS* |
| **FP8/FP6 Tensor Core** | 72 PFLOPS* |
| **INT8 Tensor Core** | 72 PFLOPS* |
| **FP16/BF16 Tensor Core** | 36 PFLOPS* |
| **TF32 Tensor Core** | 18 PFLOPS* |
| **FP32** | 600 PFLOPS |
| **FP64/FP64 Tensor Core** | 296 TFLOPS |
| **总内存** | 最高 1.4 TB HBM3e |
| **总内存带宽** | 最高 62 TB/s |
| **NVLink** | 第5代 + NVSwitch 第5代 |
| **总 NVLink 带宽** | 14.4 TB/s |
| **单 GPU TDP** | 最高 1000W（可配置） |
| **单 GPU 内存** | 180 GB HBM3e |
| **产品总重** | 32 kg |

> *With sparsity

**B100 规格（⚠️ 部分来自第三方来源）**：

| 规格项 | B100 SXM 192GB |
|--------|----------------|
| **GPU 内存** | 192 GB HBM3e |
| **内存带宽** | ~5 TB/s (⚠️ 估计) |
| **FP8 Tensor Core** | ~2.2 PFLOPS (⚠️ 估计) |
| **TDP** | ~700W (⚠️ 估计) |

> ⚠️ 注意：NVIDIA 官方尚未发布 B100 的独立 datasheet，B100 规格主要来自 HGX B100 系统级文档和第三方 spec sheet。

---

### 1.4 NVIDIA A100 Tensor Core GPU

**文档来源**：
- 官方 datasheet：https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
- 文档编号：2188504.MAY22

**版本日期**：2022 年 5 月

**关键规格表格**：

| 规格项 | A100 80GB PCIe | A100 80GB SXM |
|--------|----------------|---------------|
| **FP64** | 9.7 TFLOPS | 9.7 TFLOPS |
| **FP64 Tensor Core** | 19.5 TFLOPS | 19.5 TFLOPS |
| **FP32** | 19.5 TFLOPS | 19.5 TFLOPS |
| **TF32 Tensor Core** | 156 / 312* TFLOPS | 156 / 312* TFLOPS |
| **BF16 Tensor Core** | 312 / 624* TFLOPS | 312 / 624* TFLOPS |
| **FP16 Tensor Core** | 312 / 624* TFLOPS | 312 / 624* TFLOPS |
| **INT8 Tensor Core** | 624 / 1248* TOPS | 624 / 1248* TOPS |
| **GPU 内存** | 80 GB HBM2e | 80 GB HBM2e |
| **内存带宽** | 1,935 GB/s | 2,039 GB/s |
| **TDP** | 300W | 400W |
| **MIG** | 最多 7 个 @ 10GB | 最多 7 个 @ 10GB |
| **形态** | PCIe 双槽 | SXM |
| **互连** | NVLink Bridge: 600GB/s (2 GPU) | NVLink: 600GB/s |

> *With sparsity

**A100 架构细节**：
- 架构：NVIDIA Ampere (GA100)
- 晶体管：540 亿
- SM 数量：108 (SXM) / 108 (PCIe)
- HBM2e stacks：6 (SXM) / 4-6 (PCIe)
- L2 Cache：40 MB
- 第3代 Tensor Cores：432
- CUDA Cores：6912

---

### 1.5 NVIDIA L40S GPU

**文档来源**：
- URL：https://www.pny.com/en-eu/File%20Library/Professional/DATASHEET/DATA%20CENTER%20CARDS/PNY-NVIDIA-L40S-Datasheet.pdf
- 文档编号：2841316.AUG23

**版本日期**：2023 年 8 月

**关键规格表格**：

| 规格项 | NVIDIA L40S |
|--------|-------------|
| **架构** | NVIDIA Ada Lovelace |
| **CUDA Cores** | 18,176 |
| **Tensor Cores (第4代)** | 568 |
| **RT Cores (第3代)** | 142 |
| **GPU 内存** | 48 GB GDDR6 with ECC |
| **内存带宽** | 864 GB/s |
| **RT Core 性能** | 209 TFLOPS |
| **FP32** | 91.6 TFLOPS |
| **TF32 Tensor Core** | 183 / 366* TFLOPS |
| **BF16 Tensor Core** | 362.05 / 733* TFLOPS |
| **FP16 Tensor Core** | 362.05 / 733* TFLOPS |
| **FP8 Tensor Core** | 733 / 1,466* TFLOPS |
| **INT8 Tensor TOPS** | 733 / 1,466* |
| **INT4 Tensor TOPS** | 733 / 1,466* |
| **TDP** | 350W |
| **形态** | 双槽，被动散热 |
| **显示输出** | 4x DisplayPort 1.4a |
| **NVENC/NVDEC** | 3x / 3x (含 AV1 编解码) |
| **vGPU 支持** | 是 |
| **NEBS** | Level 3 |
| **MIG/NVLink** | 不支持 |

> *With sparsity

---

### 1.6 NVIDIA L40 GPU

**文档来源**：
- 官方 datasheet：https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/support-guide/NVIDIA-L40-Datasheet-January-2023.pdf
- 文档编号：2436245.NOV22

**版本日期**：2022 年 11 月

**关键规格表格**：

| 规格项 | NVIDIA L40 |
|--------|------------|
| **架构** | NVIDIA Ada Lovelace |
| **CUDA Cores** | 18,176 |
| **Tensor Cores (第4代)** | 568 |
| **RT Cores (第3代)** | 142 |
| **GPU 内存** | 48 GB GDDR6 with ECC |
| **内存带宽** | 864 GB/s |
| **RT Core 性能** | 209 TFLOPS |
| **FP32** | 90.5 TFLOPS |
| **TF32 Tensor Core** | 90.5 / 181* TFLOPS |
| **BF16 Tensor Core** | 181.05 / 362.1* TFLOPS |
| **FP16 Tensor Core** | 181.05 / 362.1* TFLOPS |
| **FP8 Tensor Core** | 362 / 724* TFLOPS |
| **INT8 Tensor TOPS** | 362 / 724* |
| **INT4 Tensor TOPS** | 724 / 1448* |
| **TDP** | 300W |
| **形态** | 双槽，被动散热 |
| **显示输出** | 4x DisplayPort 1.4a |
| **NVENC/NVDEC** | 3x / 3x (含 AV1) |
| **NEBS** | Level 3 |

> *With sparsity

---

### 1.7 NVIDIA A800 40GB Active

**文档来源**：
- URL：https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/a800/proviz-a800-40gb-datasheet-nvidia-2819988-r5-web.pdf
- 文档编号：2819988.OCT23

**版本日期**：2023 年 10 月

**关键规格表格**：

| 规格项 | NVIDIA A800 40GB Active |
|--------|-------------------------|
| **架构** | NVIDIA Ampere |
| **GPU 内存** | 40 GB HBM2 |
| **内存接口** | 5,120-bit |
| **内存带宽** | 1.5 TB/s |
| **CUDA Cores** | 6,912 |
| **Tensor Cores** | 432 |
| **FP64 (双精度)** | 9.7 TFLOPS |
| **FP32 (单精度)** | 19.5 TFLOPS |
| **Tensor 峰值性能** | 623.8 TFLOPS |
| **MIG** | 最多 7 个 @ 5GB |
| **NVLink** | 支持，400 GB/s |
| **总线** | PCIe 4.0 x16 |
| **功耗** | 240W |
| **散热** | 主动散热 |
| **形态** | 4.4" H x 10.5" L, 双槽 |
| **NVIDIA AI Enterprise** | 3年订阅包含 |

---

### 1.8 NVIDIA H800 / A800 中国特供版

**文档来源**：
- Lenovo H800 规格：https://lenovopress.lenovo.com/lp1813.pdf

**关键规格（⚠️ 基于 OEM 文档推断）**：

| 规格项 | H800 (推断) |
|--------|-------------|
| **架构** | NVIDIA Hopper |
| **GPU 内存** | 80 GB HBM3 (⚠️) |
| **内存带宽** | ~3.35 TB/s (⚠️) |
| **NVLink 带宽** | 400 GB/s (⚠️ 降级) |
| **PCIe** | Gen5 |

> ⚠️ H800 是 NVIDIA 为中国市场推出的降规版本，NVLink 带宽从 900GB/s 降至 400GB/s。官方未发布独立 datasheet。

---

## 二、系统产品规格

### 2.1 NVIDIA DGX H100 System

**文档来源**：
- URL：https://lambda.ai/hubfs/4.%20Resources/Datasheets/NVIDIA%20DGX/2024-04-nvidia-dgx-h100-datasheet-nvidia-us-web.pdf
- 文档编号：2795800.MAY23

**版本日期**：2023 年 5 月

**关键规格表格**：

| 规格项 | DGX H100 |
|--------|----------|
| **GPU** | 8x NVIDIA H100 Tensor Core GPU |
| **GPU 内存** | 640 GB 总量 |
| **性能** | 32 petaFLOPS FP8 |
| **NVSwitch** | 4x |
| **系统功耗** | 10.2 kW max |
| **CPU** | 双路 Intel Xeon Platinum 8480C (112 核, 2.00 GHz Base, 3.80 GHz Boost) |
| **系统内存** | 2 TB |
| **网络** | 4x OSFP (8x 单口 ConnectX-7 VPI, 400Gb/s IB/以太网) + 2x 双口 QSFP112 (ConnectX-7 VPI) |
| **管理网络** | 10Gb/s RJ45 + 100Gb/s Ethernet |
| **存储** | OS: 2x 1.92TB NVMe M.2; 内部: 8x 3.84TB NVMe U.2 |
| **软件** | NVIDIA AI Enterprise, NVIDIA Base Command |
| **系统重量** | 287.6 lbs (130.45 kg) |
| **包装重量** | 376 lbs (170.45 kg) |
| **尺寸** | 14.0" H x 19.0" W x 35.3" L (356 x 482.2 x 897.1 mm) |
| **工作温度** | 5-30°C |

---

### 2.2 NVIDIA DGX B200

**文档来源**：
- DGX B200 User Guide：https://docs.nvidia.com/dgx/dgxb200-user-guide/dgxb200-user-guide.pdf

**关键规格（⚠️ 基于 User Guide 推断）**：

| 规格项 | DGX B200 (推断) |
|--------|-----------------|
| **GPU** | 8x NVIDIA Blackwell B200 |
| **GPU 内存** | 1.44 TB 总量 (8x 180GB) |
| **性能** | ~72 PFLOPS FP8 (⚠️ 估计) |
| **CPU** | 双路 Intel Xeon (⚠️ 待确认) |
| **网络** | ConnectX-7 / ConnectX-8 (⚠️ 待确认) |

> ⚠️ 完整 DGX B200 datasheet 尚未公开发布，规格基于 HGX B200 文档推断。

---

### 2.3 GB200 NVL72 SuperCluster

**文档来源**：
- Supermicro GB200 NVL72 Datasheet：https://www.supermicro.com/datasheet/datasheet_SuperCluster_GB200_NVL72.pdf
- 日期：2025 年 2 月

**关键规格表格**：

| 规格项 | GB200 NVL72 |
|--------|-------------|
| **GPU** | 72x NVIDIA Blackwell B200 GPUs |
| **CPU** | 36x NVIDIA 72-core Grace Arm Neoverse V2 |
| **计算托盘** | 18x 1U 计算托盘 |
| **NVLink 交换机托盘** | 9x NVLink Switch |
| **GPU 内存** | 最高 372 GB HBM3e per Superchip (744GB per tray) |
| **CPU 内存** | 最高 480 GB LPDDR5X per Superchip (960GB per tray) |
| **总 GPU 内存** | ~26.8 TB HBM3e (⚠️ 估计) |
| **总 CPU 内存** | ~17.3 TB LPDDR5X (⚠️ 估计) |
| **NVLink 带宽** | 1.8 TB/s GPU-to-GPU |
| **总 GPU 通信带宽** | 130 TB/s |
| **功耗** | 125-135 kW |
| **机架尺寸** | 2236 x 600 x 1068 mm |
| **液冷** | 250kW CDU (in-rack) 或 1.3MW CDU (in-row) |
| **网络** | 4x NVLink Switch ports + 4x ConnectX-7 NICs + 2x BlueField-3 DPU per tray |
| **存储** | 最高 8x E1.S PCIe 5.0 drives per tray |

**计算托盘规格**：
- 2x 72-core Grace Arm Neoverse V2 CPUs
- 4x NVIDIA Blackwell Tensor Core GPUs
- 共享 4+4 机架电源架供电

---

### 2.4 NVIDIA Grace CPU Superchip

**文档来源**：
- URL：https://xenon.com.au/wp-content/uploads/2023/09/hpc-datasheet-grace-cpu-superchip-datasheet-2705400.pdf
- 文档编号：2705400.MAR23

**版本日期**：2023 年 3 月

**关键规格表格**：

| 规格项 | Grace CPU Superchip |
|--------|---------------------|
| **核心数** | 144 Arm Neoverse V2 Cores (4x128b SVE2) |
| **L1 Cache** | 64KB i-cache + 64KB d-cache |
| **L2 Cache** | 1 MB per core |
| **L3 Cache** | 234 MB |
| **LPDDR5X 容量** | 240GB / 480GB / 960GB (模块内) |
| **内存带宽** | 最高 1 TB/s |
| **NVLink-C2C 带宽** | 900 GB/s |
| **PCIe 链路** | 最高 8x PCIe Gen5 x16 |
| **模块 TDP** | 500W (含内存) |
| **形态** | Superchip 模块 |
| **散热** | 风冷或液冷 |
| **一致性互连** | NVIDIA Scalable Coherency Fabric (3.2 TB/s bisection) |

---

### 2.5 NVIDIA GH200 Grace Hopper Superchip

**文档来源**：
- Grace Hopper CPU Whitepaper：https://www.aspsys.com/wp-content/uploads/2023/09/nvidia-grace-hopper-cpu-whitepaper.pdf

**关键规格（⚠️ 基于 whitepaper 推断）**：

| 规格项 | GH200 Grace Hopper Superchip |
|--------|------------------------------|
| **CPU** | 72-core NVIDIA Grace Arm Neoverse V2 |
| **GPU** | NVIDIA Hopper H100 GPU (⚠️ 具体规格待确认) |
| **CPU 内存** | 最高 480 GB LPDDR5X |
| **GPU 内存** | 96 GB HBM3 (⚠️ 待确认) |
| **NVLink-C2C 带宽** | 900 GB/s |
| **TDP** | ~1000W (⚠️ 估计) |

---

## 三、网络产品

### 3.1 NVIDIA ConnectX-7 Ethernet SmartNIC

**文档来源**：
- URL：https://www.nvidia.com/content/dam/en-zz/Solutions/networking/ethernet-adapters/connectx-7-datasheet-Final.pdf
- 文档编号：APR21

**版本日期**：2021 年 4 月

**关键规格表格**：

| 规格项 | ConnectX-7 Ethernet |
|--------|---------------------|
| **最大总带宽** | 400 GbE |
| **支持速率** | 10/25/40/50/100/200/400 GbE |
| **端口数** | 1/2/4 |
| **网络接口技术** | NRZ (10/25G) / PAM4 (50/100G) |
| **主机接口** | PCIe Gen5.0 x16 / x32 |
| **形态** | PCIe FHHL/HHHL, OCP3.0 SFF |
| **网络接口类型** | SFP56, QSFP56, QSFP56-DD, QSFP112, SFP112 |
| **安全加速** | TLS/IPsec/MACsec 在线加密/解密 |
| **存储加速** | GPUDirect Storage, NVMe-oF |
| **精确时间** | IEEE 1588v2, 12ns 精度, G.8273.2 Class C |
| **SR-IOV** | 支持 |
| **ASAP²** | 加速交换和包处理 |

---

### 3.2 NVIDIA ConnectX-7 NDR 400G InfiniBand

**文档来源**：
- URL：https://www.nvidia.com/content/dam/en-zz/Solutions/networking/infiniband-adapters/infiniband-connectx7-data-sheet.pdf
- 文档编号：APR21

**版本日期**：2021 年 4 月

**关键规格表格**：

| 规格项 | ConnectX-7 NDR InfiniBand |
|--------|---------------------------|
| **最大总带宽** | 400 Gb/s |
| **IBTA 规范** | 1.5 |
| **端口数** | 1/2/4 |
| **主机接口** | PCIe Gen5, 最高 x32 通道 |
| **RDMA 消息率** | 330-370 百万消息/秒 |
| **加速引擎** | 集合操作, MPI All-to-All, MPI Tag Matching, 可编程数据路径 |
| **存储加速** | 块级加密, NVMe-oF |
| **精确时间** | PTP 1558v2, 16ns 精度 |
| **安全启动** | 片上硬件信任根 |
| **形态** | PCIe HHHL, FHHL, Socket Direct, OCP3.0 TSFF/SFF |

**InfiniBand 支持速率**：
- NDR (400 Gb/s)
- NDR200 (200 Gb/s)
- HDR (200 Gb/s)
- HDR100 (100 Gb/s)
- EDR (100 Gb/s)

---

### 3.3 NVIDIA BlueField-3 DPU

**文档来源**：
- URL：https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/documents/datasheet-nvidia-bluefield-3-dpu.pdf

**版本日期**：2021 年

**关键规格表格**：

| 规格项 | BlueField-3 DPU |
|--------|-----------------|
| **网络端口** | 1/2/4 端口，最高 400 Gb/s |
| **网络类型** | 以太网 或 InfiniBand (NDR 400G / NDR200 / HDR 200G) |
| **PCIe 接口** | 32 通道 PCIe Gen5.0 |
| **Arm CPU** | 最高 16 核 Armv8.2+ A78 Hercules |
| **L2 Cache** | 8 MB |
| **LLC Cache** | 16 MB |
| **可编程数据路径** | 16 核, 256 线程 |
| **DDR5 支持** | 双 DDR5 5600MT/s 控制器 |
| **板载 DDR5** | 16 GB |
| **安全启动** | PKA 硬件信任根 |
| **加密加速** | MACsec/IPsec/TLS, AES-GCM 128/256, AES-XTS 256/512 |
| **存储加速** | NVMe-oF, NVMe/TCP, 解压缩, 纠删码 |
| **管理端口** | 1GbE 带外管理 |
| **形态** | HHHL, FHHL |

---

### 3.4 InfiniBand NDR 交换机 / Spectrum-4

**文档来源**：
- ⚠️ 官方交换机 datasheet 未在调研中直接下载，但可从以下来源获取：
- NVIDIA Quantum-X800 (NDR) 交换机规格参考 Data Center GPU Line Card

**Spectrum-4 交换机（⚠️ 基于公开信息推断）**：

| 规格项 | Spectrum-4 (推断) |
|--------|-------------------|
| **交换容量** | 25.6 Tbps (⚠️) |
| **端口** | 64x 400GbE (⚠️) |
| **形态** | 1U 交换机 |

> ⚠️ Spectrum-4 完整 datasheet 需要从 NVIDIA 合作伙伴门户获取。

---

## 四、汽车/嵌入式

### 4.1 NVIDIA Jetson AGX Orin

**文档来源**：
- Technical Brief v1.2：https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf
- 文档编号：TB_10749-001_v1.2, July 2022

**版本日期**：2022 年 7 月

**关键规格表格**：

| 规格项 | Jetson AGX Orin 64GB | Jetson AGX Orin 32GB |
|--------|----------------------|----------------------|
| **AI 性能** | 275 TOPS (INT8) | 200 TOPS (INT8) |
| **GPU** | NVIDIA Ampere 架构, 2048 CUDA Cores, 64 Tensor Cores | 1792 CUDA Cores, 56 Tensor Cores |
| **CPU** | 12-core Arm Cortex-A78AE | 8-core Arm Cortex-A78AE |
| **内存** | 64 GB LPDDR5 | 32 GB LPDDR5 |
| **内存带宽** | 204 GB/s | 204 GB/s |
| **存储** | 64 GB eMMC 5.1 | 64 GB eMMC 5.1 |
| **视频编码** | 4K60 (H.265) | 4K60 (H.265) |
| **视频解码** | 8K30 (H.265) | 8K30 (H.265) |
| **PCIe** | 最高 22 通道 | 最高 22 通道 |
| **USB** | 3x USB 3.2 + 4x USB 2.0 | 3x USB 3.2 + 4x USB 2.0 |
| **CSI** | 16 通道 MIPI CSI-2 | 16 通道 MIPI CSI-2 |
| **CAN** | 2x CAN | 2x CAN |
| **功耗** | 15-60W (可配置) | 15-60W (可配置) |

**Orin SoC 架构细节**：
- GPU: 2x GPCs, 8 TPCs, 16 SMs (64GB) / 14 SMs (32GB)
- DLA: 2x 下一代深度学习加速器
- PVA: 可编程视觉加速器
- VIC: 图像信号处理器

---

### 4.2 NVIDIA DRIVE AGX Orin

**文档来源**：
- URL：https://developer.nvidia.com/downloads/drive/docs/nvidia-drive-agx-orin-platform-for-developers.pdf
- 日期：2025 年 10 月

**关键规格（⚠️ 基于文档推断）**：

| 规格项 | DRIVE AGX Orin |
|--------|----------------|
| **SoC** | NVIDIA Orin (与 Jetson AGX Orin 同款 SoC) |
| **AI 性能** | 254 TOPS (⚠️ 待确认) |
| **目标应用** | 自动驾驶 (L2-L5) |
| **功能安全** | ASIL-D |
| **软件** | NVIDIA DriveOS, DriveWorks |

> ⚠️ DRIVE AGX Orin 完整规格为受限文档，需通过 NVIDIA Developer Program 获取。

---

### 4.3 NVIDIA Jetson Orin Nano / NX

**文档来源**：
- Jetson Orin NX Datasheet：本地已下载 (jetson_orin_nx_datasheet.pdf)
- Jetson Orin Nano Datasheet：本地已下载 (jetson_orin_nano_datasheet.pdf)

**关键规格表格**：

| 规格项 | Jetson Orin NX 16GB | Jetson Orin NX 8GB | Jetson Orin Nano |
|--------|---------------------|--------------------|--------------------|
| **AI 性能** | 100 TOPS | 70 TOPS | 40 TOPS |
| **GPU** | 1024 CUDA Cores, 32 Tensor Cores | 1024 CUDA Cores, 32 Tensor Cores | 512 CUDA Cores, 16 Tensor Cores |
| **CPU** | 8-core Arm Cortex-A78AE | 8-core Arm Cortex-A78AE | 6-core Arm Cortex-A78AE |
| **内存** | 16 GB LPDDR5 | 8 GB LPDDR5 | 4/8 GB LPDDR5 |
| **内存带宽** | 102 GB/s | 68 GB/s | 34-51 GB/s |
| **功耗** | 10-25W | 10-25W | 5-15W |

---

## 五、架构白皮书

### 5.1 NVIDIA Hopper Architecture

**文档来源**：
- NVIDIA Hopper Architecture In-Depth (Technical Blog)：https://lists.riscv.org/g/tech-attached-matrix-extension/attachment/25/0/NVIDIA%20Hopper%20Architecture%20In-Depth%20_%20NVIDIA%20Technical%20Blog.pdf
- GTC22 Whitepaper：https://www.hpctech.co.jp/assets/images/info/catalog/pdf/gtc22-whitepaper-hopper_v1.02.pdf

**版本日期**：2022 年 3 月

**架构关键创新**：
1. **第4代 Tensor Cores**：相比 A100 快 6X，支持 FP8 新数据类型
2. **Transformer Engine**：FP8/FP16 动态精度切换，LLM 训练快 9X
3. **Tensor Memory Accelerator (TMA)**：全局内存到共享内存异步传输
4. **Thread Block Cluster**：跨 SM 的线程块协作
5. **DPX 指令**：动态编程算法加速 7X
6. **异步事务屏障**：跨 SM 同步
7. **机密计算**：硬件级 VM 隔离

**H100 vs A100 性能对比**：

| 数据类型 | A100 | H100 SXM5 | H100 加速比 |
|----------|------|-----------|-------------|
| FP8 Tensor Core | N/A | 2000/4000* TFLOPS | 6.4x vs A100 FP16 |
| FP16 Tensor Core | 312/624* | 1000/2000* | 3.2x |
| BF16 Tensor Core | 312/624* | 1000/2000* | 3.2x |
| TF32 Tensor Core | 156/312* | 500/1000* | 3.2x |
| FP64 | 9.7 | 30 | 3.1x |
| FP64 Tensor Core | 19.5 | 60 | 3.1x |
| FP32 | 19.5 | 60 | 3.1x |
| INT8 Tensor Core | 624/1248* | 2000/4000* | 3.2x |

> *With sparsity

---

### 5.2 NVIDIA Blackwell Architecture

**文档来源**：
- Blackwell Architecture Technical Brief：https://cdn.prod.website-files.com/61dda201f29b7efc52c5fbaf/6602ea9d0ce8cb73fb6de87f_nvidia-blackwell-architecture-technical-brief.pdf
- NVIDIA RTX Blackwell GPU Architecture：https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf

**版本日期**：2024 年 3 月

**架构关键创新**：
1. **第2代 Transformer Engine**：支持 FP4 精度
2. **第5代 NVLink**：1.8 TB/s GPU-to-GPU 带宽
3. **新的 NVLink Switch System**：130 TB/s 总 GPU 通信带宽
4. **安全 AI**：机密计算支持
5. **解压缩引擎**：加速 AI 数据流水线

**Blackwell GPU 关键规格（基于 HGX B200）**：
- 180 GB HBM3e per GPU
- 总内存带宽 62 TB/s (8 GPU)
- FP4 Tensor Core: 144 PFLOPS (HGX B200)
- FP8 Tensor Core: 72 PFLOPS (HGX B200)

---

### 5.3 NVIDIA Ampere Architecture

**文档来源**：
- NVIDIA A100 Tensor Core GPU Architecture Whitepaper：https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf
- NVIDIA Ampere GA102 GPU Architecture：https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf

**版本日期**：2020 年

**架构关键创新**：
1. **第3代 Tensor Cores**：支持 TF32, BF16, FP64
2. **结构化稀疏**：2X AI 性能加速
3. **Multi-Instance GPU (MIG)**：硬件级 GPU 分区
4. **第3代 NVLink**：600 GB/s
5. **HBM2e**：最高 80 GB, 2 TB/s 带宽
6. **异步拷贝/屏障**：提升数据搬运效率

**A100 SM 架构**：
- 64 FP32 CUDA Cores + 64 INT32 Cores per SM
- 4 个第3代 Tensor Cores per SM
- 256 KB 组合共享内存/L1 Cache

---

### 5.4 NVIDIA Ada Lovelace Architecture

**文档来源**：
- NVIDIA Ada GPU Architecture：https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
- 文档版本：V2.02

**版本日期**：2022 年

**架构关键创新**：
1. **第4代 Tensor Cores**：支持 FP8, Transformer Engine
2. **第3代 RT Cores**：2X 光线-三角形求交
3. **Opacity Micromap Engine**：10X 加速 alpha 遍历
4. **Displaced Micro-Mesh Engine**：20X 更少 BVH 空间
5. **Shader Execution Reordering (SER)**：提升光追效率 44%
6. **DLSS 3**：AI 帧生成
7. **双 NVENC**：AV1 编码

**AD102 GPU 规格**：
- 763 亿晶体管
- 18,432 CUDA Cores
- 128 个第3代 RT Cores
- 512 个第4代 Tensor Cores
- TSMC 4N 工艺

---

## 六、下载文档清单

### 成功下载的 PDF 文件

| 文件名 | 大小 | 来源 URL |
|--------|------|----------|
| nvidia-h100-datasheet.pdf | 311 KB | https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf |
| nvidia-h200-datasheet.pdf | 641 KB | https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/NVIDIA_H200_Datasheet.pdf |
| nvidia-a100-datasheet.pdf | 494 KB | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf |
| nvidia-a100-80gb-datasheet.pdf | 878 KB | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf |
| nvidia-a100-product-brief.pdf | 399 KB | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf |
| nvidia-l40-datasheet.pdf | 129 KB | https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/support-guide/NVIDIA-L40-Datasheet-January-2023.pdf |
| nvidia-l40-product-brief.pdf | 578 KB | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/datasheets/L-40/product-brief-L40.pdf |
| nvidia-l40s-datasheet.pdf | 168 KB | https://www.pny.com/en-eu/File%20Library/Professional/DATASHEET/DATA%20CENTER%20CARDS/PNY-NVIDIA-L40S-Datasheet.pdf |
| nvidia-a800-40gb-datasheet.pdf | 348 KB | https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/a800/proviz-a800-40gb-datasheet-nvidia-2819988-r5-web.pdf |
| nvidia-dgx-h100-datasheet.pdf | 561 KB | https://lambda.ai/hubfs/4.%20Resources/Datasheets/NVIDIA%20DGX/2024-04-nvidia-dgx-h100-datasheet-nvidia-us-web.pdf |
| nvidia-dgx-b200-user-guide.pdf | 3.9 MB | https://docs.nvidia.com/dgx/dgxb200-user-guide/dgxb200-user-guide.pdf |
| nvidia-hgx-b200-pcf.pdf | 263 KB | https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf |
| nvidia-gb200-nvl72-supermicro.pdf | 1.3 MB | https://www.supermicro.com/datasheet/datasheet_SuperCluster_GB200_NVL72.pdf |
| nvidia-data-center-gpu-line-card.pdf | 210 KB | https://docs.nvidia.com/data-center-gpu/line-card.pdf |
| nvidia-grace-cpu-superchip-datasheet.pdf | 232 KB | https://xenon.com.au/wp-content/uploads/2023/09/hpc-datasheet-grace-cpu-superchip-datasheet-2705400.pdf |
| nvidia-connectx-7-ethernet-datasheet.pdf | 134 KB | https://www.nvidia.com/content/dam/en-zz/Solutions/networking/ethernet-adapters/connectx-7-datasheet-Final.pdf |
| nvidia-connectx-7-infiniband-datasheet.pdf | 277 KB | https://www.nvidia.com/content/dam/en-zz/Solutions/networking/infiniband-adapters/infiniband-connectx7-data-sheet.pdf |
| nvidia-bluefield-3-dpu-datasheet.pdf | 586 KB | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/documents/datasheet-nvidia-bluefield-3-dpu.pdf |
| nvidia-jetson-agx-orin-technical-brief.pdf | 931 KB | https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf |
| nvidia-drive-agx-orin-platform.pdf | 3.8 MB | https://developer.nvidia.com/downloads/drive/docs/nvidia-drive-agx-orin-platform-for-developers.pdf |
| nvidia-hopper-architecture-in-depth.pdf | 8.8 MB | https://lists.riscv.org/g/tech-attached-matrix-extension/attachment/25/0/NVIDIA%20Hopper%20Architecture%20In-Depth%20_%20NVIDIA%20Technical%20Blog.pdf |
| nvidia-hopper-architecture-whitepaper.pdf | 1.3 MB | https://www.hpctech.co.jp/assets/images/info/catalog/pdf/gtc22-whitepaper-hopper_v1.02.pdf |
| nvidia-ampere-architecture-whitepaper.pdf | 8.0 MB | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf |
| nvidia-ada-gpu-architecture.pdf | 3.9 MB | https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf |
| nvidia-blackwell-architecture-technical-brief.pdf | 4.6 MB | https://cdn.prod.website-files.com/61dda201f29b7efc52c5fbaf/6602ea9d0ce8cb73fb6de87f_nvidia-blackwell-architecture-technical-brief.pdf |
| nvidia-rtx-blackwell-gpu-architecture.pdf | 8.3 MB | https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf |
| nvidia-grace-hopper-cpu-whitepaper.pdf | 7.2 MB | https://www.aspsys.com/wp-content/uploads/2023/09/nvidia-grace-hopper-cpu-whitepaper.pdf |
| nvidia-b100-spec-sheet.pdf | 43 KB | https://flopper.io/gpu/nvidia-b100-sxm-192gb/spec-sheet.pdf |

---

## 七、访问过的 URL 清单

### 成功访问的 URL

| URL | 状态 | 说明 |
|-----|------|------|
| https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf | ✅ 200 | A100 官方 datasheet |
| https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf | ✅ 200 | A100 80GB datasheet |
| https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf | ✅ 200 | A100 Product Brief |
| https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/support-guide/NVIDIA-L40-Datasheet-January-2023.pdf | ✅ 200 | L40 官方 datasheet |
| https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/datasheets/L-40/product-brief-L40.pdf | ✅ 200 | L40 Product Brief |
| https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/a800/proviz-a800-40gb-datasheet-nvidia-2819988-r5-web.pdf | ✅ 200 | A800 官方 datasheet |
| https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf | ✅ 200 | Ampere 架构白皮书 |
| https://www.nvidia.com/content/dam/en-zz/Solutions/networking/ethernet-adapters/connectx-7-datasheet-Final.pdf | ✅ 200 | ConnectX-7 以太网 datasheet |
| https://www.nvidia.com/content/dam/en-zz/Solutions/networking/infiniband-adapters/infiniband-connectx7-data-sheet.pdf | ✅ 200 | ConnectX-7 InfiniBand datasheet |
| https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/documents/datasheet-nvidia-bluefield-3-dpu.pdf | ✅ 200 | BlueField-3 DPU datasheet |
| https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf | ✅ 200 | Jetson AGX Orin Technical Brief |
| https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/documents/nvidia-ampere-architecture-whitepaper.pdf | ✅ 200 | Ampere 架构白皮书 |
| https://docs.nvidia.com/data-center-gpu/line-card.pdf | ✅ 200 | Data Center GPU Line Card |
| https://docs.nvidia.com/dgx/dgxb200-user-guide/dgxb200-user-guide.pdf | ✅ 200 | DGX B200 User Guide |
| https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf | ✅ 200 | HGX B200 PCF Summary |
| https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf | ✅ 200 | Ada GPU Architecture |
| https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf | ✅ 200 | RTX Blackwell GPU Architecture |
| https://developer.nvidia.com/downloads/drive/docs/nvidia-drive-agx-orin-platform-for-developers.pdf | ✅ 200 | DRIVE AGX Orin Platform |
| https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf | ✅ 200 | H100 datasheet (第三方托管) |
| https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/NVIDIA_H200_Datasheet.pdf | ✅ 200 | H200 datasheet (第三方托管) |
| https://www.pny.com/en-eu/File%20Library/Professional/DATASHEET/DATA%20CENTER%20CARDS/PNY-NVIDIA-L40S-Datasheet.pdf | ✅ 200 | L40S datasheet (PNY) |
| https://xenon.com.au/wp-content/uploads/2023/09/hpc-datasheet-grace-cpu-superchip-datasheet-2705400.pdf | ✅ 200 | Grace CPU Superchip datasheet |
| https://www.supermicro.com/datasheet/datasheet_SuperCluster_GB200_NVL72.pdf | ✅ 200 | GB200 NVL72 datasheet |
| https://www.aspsys.com/wp-content/uploads/2023/09/nvidia-grace-hopper-cpu-whitepaper.pdf | ✅ 200 | Grace Hopper CPU Whitepaper |
| https://www.hpctech.co.jp/assets/images/info/catalog/pdf/gtc22-whitepaper-hopper_v1.02.pdf | ✅ 200 | Hopper GTC22 Whitepaper |
| https://lists.riscv.org/g/tech-attached-matrix-extension/attachment/25/0/NVIDIA%20Hopper%20Architecture%20In-Depth%20_%20NVIDIA%20Technical%20Blog.pdf | ✅ 200 | Hopper Architecture In-Depth |
| https://cdn.prod.website-files.com/61dda201f29b7efc52c5fbaf/6602ea9d0ce8cb73fb6de87f_nvidia-blackwell-architecture-technical-brief.pdf | ✅ 200 | Blackwell Architecture Technical Brief |
| https://lambda.ai/hubfs/4.%20Resources/Datasheets/NVIDIA%20DGX/2024-04-nvidia-dgx-h100-datasheet-nvidia-us-web.pdf | ✅ 200 | DGX H100 datasheet |
| https://flopper.io/gpu/nvidia-b100-sxm-192gb/spec-sheet.pdf | ✅ 200 | B100 Spec Sheet (第三方) |

### 访问失败/404 的 URL

| URL | 状态 | 说明 |
|-----|------|------|
| https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/h100/nvidia-h100-tensor-core-gpu-datasheet.pdf | ❌ 404 | H100 官方 datasheet 路径已失效 |
| https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/a100/pdf/nvidia-a100-datasheet.pdf | ❌ 404 | A100 旧路径已失效 |
| https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet | ❌ 连接失败 | 资源中心路径变更 |

---

## 八、关键发现与注意事项

### 8.1 数据质量评估

| 评级 | 说明 | 涉及产品 |
|------|------|----------|
| ✅ 高可信度 | 来自 NVIDIA 官方 datasheet/whitepaper | A100, L40, A800, ConnectX-7, BlueField-3, Ampere/Ada 架构 |
| 🟡 中等可信度 | 来自官方技术博客/第三方托管的 NVIDIA 文档 | H100, H200, Hopper 架构, Grace CPU |
| ⚠️ 推断/待确认 | 来自系统级文档或第三方来源 | B100, B200, H800, GH200, Spectrum-4 |

### 8.2 重要注意事项

1. **H100 官方 datasheet 已下线**：NVIDIA 官网的 H100 datasheet PDF 链接返回 404，规格数据来自 Hopper Architecture In-Depth 技术博客（同样权威）
2. **B100/B200 无独立 GPU datasheet**：NVIDIA 仅发布了 HGX B200 系统级文档，单 GPU 规格需从系统级文档推断
3. **H800/A800 为中国特供版**：官方未发布完整 datasheet，规格主要来自 OEM 文档
4. **GB200 NVL72 规格来自 Supermicro**：NVIDIA 官方未发布 GB200 NVL72 独立 datasheet，Supermicro 作为合作伙伴发布了系统级规格
5. **部分文档为 Preliminary**：H100、H200 等早期文档标注 "Preliminary specifications, may be subject to change"

### 8.3 规格对比总结

| GPU | 架构 | 内存 | 内存带宽 | FP8 Tensor Core | TDP |
|-----|------|------|----------|-----------------|-----|
| A100 80GB | Ampere | 80GB HBM2e | 2.0 TB/s | N/A | 400W |
| H100 SXM5 | Hopper | 80GB HBM3 | 3.35 TB/s | 4000* TFLOPS | 700W |
| H200 SXM | Hopper | 141GB HBM3e | 4.8 TB/s | 3958 TFLOPS | 700W |
| B200 (HGX) | Blackwell | 180GB HBM3e | ~7.75 TB/s** | ~9 PFLOPS** | 1000W |
| L40 | Ada | 48GB GDDR6 | 864 GB/s | 724* TFLOPS | 300W |
| L40S | Ada | 48GB GDDR6 | 864 GB/s | 1466* TFLOPS | 350W |
| A800 40GB | Ampere | 40GB HBM2 | 1.5 TB/s | N/A | 240W |

> *With sparsity
> **Per GPU 估计值

---

## 九、调研方法说明

1. **搜索策略**：使用 Playwright 浏览器搜索 Google，关键词包含 "NVIDIA [产品] datasheet pdf" 和 "NVIDIA [产品] specifications pdf"
2. **下载策略**：使用 curl 直接下载 PDF 文件到本地
3. **提取策略**：使用 PyPDF2 提取 PDF 文本内容，识别规格表格
4. **验证策略**：交叉验证多个来源的规格数据，标注不确定项

---

## 十、后续建议

1. **补充 B300/B100 规格**：等待 NVIDIA 发布 HGX B300 官方文档
2. **补充 Vera CPU 规格**：NVIDIA 下一代 CPU 架构尚未在调研中覆盖
3. **补充 ConnectX-8/Quantum-X800**：下一代网络产品 datasheet 待发布
4. **建立定期更新机制**：NVIDIA 产品规格更新频繁，建议每季度重新调研
