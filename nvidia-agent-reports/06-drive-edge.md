# NVIDIA 边缘与嵌入式产品深度规格调研报告

> **调研日期**: 2026年7月31日  
> **调研范围**: NVIDIA Drive汽车平台、Jetson机器人平台的SoC与系统级产品规格  
> **信息来源**: NVIDIA官方Datasheet、GTC技术文档、开发者页面、技术媒体分析

---

## 摘要

本报告系统梳理NVIDIA在边缘AI计算领域的两大核心产品线：面向自动驾驶的**Drive平台**和面向机器人/边缘AI的**Jetson平台**。调研覆盖从Orin（Ampere架构）到Thor（Blackwell架构）两代产品的完整规格，包括SoC微架构、算力指标、功耗范围、内存子系统和安全特性。

**关键发现**：
- **Jetson AGX Orin** 是当前量产最强边缘AI模块，275 TOPS INT8算力，Ampere GPU + Cortex-A78AE
- **Drive AGX Thor** （2025年8月发布）采用Blackwell架构，1000 TOPS INT8，面向L4自动驾驶
- **Jetson Thor** （T5000/T4000）采用Blackwell GPU + Neoverse V3AE，FP4算力高达2070 TFLOPs
- 代际性能跃迁：Orin→Thor实现约4-8倍算力提升，同时引入FP4/FP8新精度支持

---

## 1. Jetson平台规格总览

### 1.1 代际演进路线

| 代际 | SoC架构 | GPU架构 | CPU | 发布时间 | 代表产品 |
|------|---------|---------|-----|----------|----------|
| 第1代 | Tegra K1 | Kepler (192 CUDA) | Cortex-A15 | 2014 | Jetson TK1 |
| 第2代 | Tegra X1 | Maxwell (256 CUDA) | Cortex-A57+A53 | 2015 | Jetson TX1 |
| 第3代 | Tegra X2 | Pascal (256 CUDA) | Denver2 + A57 | 2017 | Jetson TX2 |
| 第4代 | Xavier | Volta (512 CUDA) | Carmel (自研) | 2018 | Jetson AGX Xavier |
| 第5代 | Orin | Ampere (2048 CUDA) | Cortex-A78AE | 2022 | Jetson AGX Orin |
| 第6代 | Thor | Blackwell (2560 CUDA) | Neoverse V3AE | 2025 | Jetson AGX Thor |

### 1.2 Jetson Orin系列完整规格对比

| 规格 | Jetson AGX Orin 64GB | Jetson AGX Orin 32GB | Jetson AGX Orin Industrial | Jetson Orin NX 16GB | Jetson Orin NX 8GB | Jetson Orin Nano 8GB | Jetson Orin Nano 4GB |
|------|---------------------|---------------------|---------------------------|--------------------|--------------------|--------------------|--------------------|
| **AI Performance (INT8 Sparse)** | 275 TOPS | 200 TOPS | 248 TOPS | 100 TOPS | 70 TOPS | 67 TOPS | 34 TOPS |
| **GPU CUDA Cores** | 2048 | 1792 | 2048 | 1024 | 1024 | 1024 | 512 |
| **Tensor Cores** | 64 | 56 | 64 | 32 | 32 | 32 | 16 |
| **GPU Max Frequency** | 1.3 GHz | 939 MHz | 1.185 GHz | 918 MHz | 765 MHz | 1020 MHz | 1020 MHz |
| **CPU** | 12-core A78AE | 8-core A78AE | 12-core A78AE | 8-core A78AE | 6-core A78AE | 6-core A78AE | 6-core A78AE |
| **CPU Max Frequency** | 2.2 GHz | 2.2 GHz | 1.971 GHz | - | - | 1.7 GHz | 1.7 GHz |
| **DLA** | 2x NVDLA 2.0 | 2x NVDLA 2.0 | 2x NVDLA 2.0 | 2x NVDLA | 1x NVDLA | - | - |
| **Memory** | 64GB LPDDR5 | 32GB LPDDR5 | 64GB LPDDR5 | 16GB LPDDR5 | 8GB LPDDR5 | 8GB LPDDR5 | 4GB LPDDR5 |
| **Memory Bus** | 256-bit | 256-bit | 256-bit | 128-bit | 128-bit | 128-bit | 64-bit |
| **Memory Bandwidth** | 204.8 GB/s | 102.4 GB/s | 204.8 GB/s | - | - | 102 GB/s | 51 GB/s |
| **Storage** | 64GB eMMC 5.1 | 64GB eMMC 5.1 | 64GB eMMC 5.1 | - | - | - | - |
| **Power范围** | 15W-60W | 15W-40W | 15W-75W | 10W-40W | 10W-25W | 7W-25W | 7W-25W |
| **PCIe** | 2x8 + 1x4 + 2x1 (Gen4) | 同左 | 同左 | 1x4 + 3x1 (Gen4) | 同左 | 1x4 + 3x1 (Gen3) | 同左 |
| **USB** | 3x USB 3.2 + 4x USB 2.0 | 同左 | 同左 | 3x USB 3.2 + 3x USB 2.0 | 同左 | 3x USB 3.2 + 3x USB 2.0 | 同左 |
| **以太网** | 1x GbE + 1x 10GbE | 同左 | 同左 | 1x GbE | 1x GbE | 1x GbE | 1x GbE |
| **MIPI CSI** | 16 lanes D-PHY | 同左 | 同左 | 8 lanes | 8 lanes | 8 lanes | 8 lanes |
| **尺寸** | 100mm x 87mm | 同左 | 同左 | 69.6mm x 45mm | 同左 | 69.6mm x 45mm | 同左 |
| **连接器** | 699-pin B2B | 同左 | 同左 | 260-pin SO-DIMM | 同左 | 260-pin SO-DIMM | 同左 |

**来源**: [Jetson AGX Orin Datasheet (DS-10662-001v1.8)](https://static.generation-robots.com/media/Jetson-AGX-Orin-Data-Sheet.pdf), [Jetson Orin NX Datasheet (DS-10712-001)](https://developer.nvidia.com/downloads/jetson-orin-nx-series-data-sheet), [Jetson Orin Nano Datasheet](https://static.generation-robots.com/media/jetson-orin-datasheet-nano-modules.pdf), [NVIDIA Jetson Orin产品页](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

### 1.3 Jetson AGX Orin Developer Kit规格

| 规格 | 详情 |
|------|------|
| **SoC** | Orin SoC (同64GB配置) |
| **AI Performance** | 275 TOPS (INT8 Sparse) |
| **GPU** | 2048-core Ampere, 64 Tensor Cores, 1.3 GHz |
| **CPU** | 12-core Cortex-A78AE @ 2.2 GHz |
| **内存** | 64GB 256-bit LPDDR5 (204.8 GB/s) |
| **存储** | 64GB eMMC 5.1 |
| **视频编码** | 4K60 H.265/H.264/AV1 |
| **视频解码** | 8K30 H.265 / 4K60 H.265 (多流) |
| **接口** | 1x 10GbE, 1x GbE, 3x USB 3.2, 1x USB-C, PCIe Gen4 x8 |
| **价格** | $1,999 (开发套件) |

**来源**: [NVIDIA Jetson AGX Orin Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

### 1.4 Jetson Orin Nano Super Developer Kit规格

| 规格 | 详情 |
|------|------|
| **AI Performance** | 67 TOPS (INT8) |
| **GPU** | 1024-core Ampere, 32 Tensor Cores, 1020 MHz |
| **CPU** | 6-core Cortex-A78AE @ 1.7 GHz |
| **内存** | 8GB 128-bit LPDDR5 (102 GB/s) |
| **存储** | 支持外部NVMe |
| **视频编码** | 1080p30 (CPU) |
| **视频解码** | 4K60 H.265 |
| **接口** | 1x GbE, 3x USB 3.2, 1x USB-C (PCIe Gen3 x4) |
| **价格** | $249 |

**来源**: [Seeed Studio Jetson Orin Nano DevKit Datasheet](https://files.seeedstudio.com/wiki/Jetson-Orin-Nano-DevKit/jetson-orin-nano-developer-kit-datasheet.pdf)

---

## 2. Jetson Thor系列（Blackwell架构，2025年8月发布）

### 2.1 Jetson Thor模块规格

| 规格 | Jetson AGX Thor (T5000) | Jetson Thor (T4000) |
|------|------------------------|---------------------|
| **AI Performance (FP4 Sparse, MAXN)** | 2070 TFLOPs | 1200 TFLOPs |
| **AI Performance (FP8 Dense, MAXN)** | 517 TFLOPs (≈517 TOPS INT8) | 300 TFLOPs (≈300 TOPS INT8) |
| **GPU CUDA Cores** | 2560 | 1536 |
| **Tensor Cores** | 96 (5th Gen) | 96 (5th Gen) |
| **GPU架构** | Blackwell (3 GPC, 10 TPC) | Blackwell (2 GPC, 6 TPC) |
| **GPU MAXN Frequency** | 1.575 GHz | 1.53 GHz |
| **GPU 120W/70W Frequency** | 1.386 GHz | 1.53 GHz |
| **FP32 TFLOPs (MAXN)** | 8.064 | 4.700 |
| **CPU** | 14-core Neoverse V3AE | 12-core Neoverse V3AE |
| **CPU Max Frequency** | 2.6 GHz | 2.6 GHz |
| **L3 Cache** | 16MB shared | 16MB shared |
| **PVA** | 1x PVA 3.0 @ 1.215 GHz | 同左 |
| **内存** | 128GB LPDDR5X (256-bit) | 64GB LPDDR5X (256-bit) |
| **内存频率** | 4,266 MHz | 4,266 MHz |
| **内存带宽** | 273 GB/s | 273 GB/s |
| **NVDEC** | 2x (1.56 GHz @120W) | 1x (1.56 GHz @70W) |
| **NVENC** | 2x (1.56 GHz @120W) | 1x (1.56 GHz @70W) |
| **以太网** | 4x 25GbE MGBE | 3x 25GbE MGBE |
| **PCIe** | Gen5 x8 (11 lanes total) | 同左 |
| **USB** | 3x USB 3.2 + 4x USB 2.0 | 同左 |
| **CAN** | 4x CAN | - |
| **显示** | 4x HDMI 2.1 / DP 1.4a | 同左 |
| **MIPI CSI** | 16 lanes (D-PHY 2.1 + C-PHY 2.1) | 同左 |
| **功耗模式** | 70W / 90W / 120W / MAXN | 70W / MAXN |
| **最大模块功耗** | 130W | 90W |
| **尺寸** | 87mm x 100mm x 15.29mm | 同左 |
| **连接器** | 699-pin B2B | 同左 |
| **工作温度** | -25°C ~ 115°C (Tj) | 同左 |

**来源**: [Jetson Thor Datasheet (DS-11945-001_v1.5)](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/thor/docs/jetson-thor-series-modules-datasheet_ds-11945-001.pdf), [RidgeRun Jetson Thor分析](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Thor:_Powering_the_Future_of_Physical_AI), [VideoCardz](https://videocardz.com/newz/nvidia-jetson-thor-with-blackwell-gpu-architecture-launched-costs-3499)

### 2.2 Jetson AGX Thor Developer Kit

| 规格 | 详情 |
|------|------|
| **模块** | Jetson T5000 |
| **AI Performance** | 2070 TFLOPs FP4 / 1035 TFLOPs FP8 (MAXN) |
| **GPU** | Blackwell 2560 CUDA cores, 96 Tensor Cores |
| **CPU** | 14-core Arm Neoverse V3AE @ 2.6 GHz |
| **内存** | 128GB LPDDR5X (273 GB/s) |
| **价格** | $3,499 |
| **状态** | 2025年8月上市 |

**来源**: [VideoCardz](https://videocardz.com/newz/nvidia-jetson-thor-with-blackwell-gpu-architecture-launched-costs-3499), [NVIDIA Developer Blog](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)

---

## 3. Drive平台规格

### 3.1 Drive AGX Orin

Drive AGX Orin是NVIDIA面向L2+到L4自动驾驶的量产计算平台，基于Orin SoC。

| 规格 | Drive AGX Orin |
|------|---------------|
| **SoC** | Orin SoC (与Jetson Orin同架构) |
| **AI Performance** | 254-275 TOPS (INT8) |
| **GPU** | Ampere架构, 2048 CUDA cores, 64 Tensor Cores |
| **GPU Frequency** | 1.3 GHz |
| **CPU** | 12-core Arm Cortex-A78AE v8.2 (另有8核选项) |
| **CPU频率** | 2.2 GHz |
| **DLA** | 2x NVDLA v2.0 |
| **PVA** | 1x Programmable Vision Accelerator |
| **内存** | 32GB 或 64GB 256-bit LPDDR5 (204.8 GB/s) |
| **存储** | 64GB eMMC 5.1 |
| **功耗** | 15W-40W+ (可配置) |
| **功能安全** | ASIL-D (系统级) |
| **视频** | 多通道8K/4K编解码 |

**来源**: [NVIDIA DRIVE AGX开发者页面](https://developer.nvidia.com/drive/agx), [NVIDIA Jetson AGX Orin Technical Brief](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)

### 3.2 Drive AGX Thor

Drive AGX Thor是NVIDIA新一代自动驾驶计算平台，基于Blackwell架构，面向L4级自动驾驶。

| 规格 | Drive AGX Thor |
|------|---------------|
| **SoC** | Thor SoC (Blackwell架构) |
| **AI Performance** | 1,000 TOPS (INT8) / 2,000 TFLOPs (FP4) |
| **GPU** | Blackwell集成GPU, 第5代Tensor Cores |
| **CPU** | 14-core Arm Neoverse V3AE |
| **内存** | 64GB-128GB LPDDR5X |
| **内存带宽** | 273 GB/s |
| **功耗** | 75W-130W (可配置) |
| **功能安全** | ASIL-D |
| **精度支持** | FP32, FP16, FP8, FP4, INT8 |
| **Transformer Engine** | 原生FP4/FP8 Transformer加速 |
| **目标应用** | 自动驾驶、主动安全、数字座舱统一计算 |
| **部署时间** | 2025-2026年量产爬坡 |

**来源**: [NVIDIA DRIVE AGX Thor Platform for Developers](https://developer.nvidia.com/downloads/drive/docs/nvidia-drive-agx-thor-platform-for-developers.pdf), [Edge AI and Vision Alliance](https://www.edge-ai-vision.com/2025/09/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/), [NVIDIA Developer Blog](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)

### 3.3 Drive Hyperion平台

Drive Hyperion是NVIDIA的传感器+计算参考架构平台：

| 规格 | Hyperion 8 (Orin) | Hyperion 9 (Thor) |
|------|-------------------|-------------------|
| **计算平台** | Drive AGX Orin | Drive AGX Thor (双芯片冗余) |
| **摄像头** | 11个 | 14个 |
| **毫米波雷达** | 6个 | 9个 |
| **激光雷达** | 可选 | 3个 |
| **超声波** | 12个 | 20个 |
| **舱内传感** | 驾驶员监控 | 摄像头+麦克风阵列 |
| **功能安全** | ASIL-B/D | ASIL-D |
| **目标级别** | L2+ | L4 |

**来源**: [NVIDIA Blog - Drive Hyperion 9](https://blogs.nvidia.com/blog/drive-hyperion-9-thor/), [IoT Automotive News](https://iot-automotive.news/introducing-nvidia-drive-hyperion-9-next-generation-platform-for-software-defined-autonomous-vehicle-fleets/), [Forbes](https://www.forbes.com/sites/patrickmoorhead/2022/04/07/nvidia-announces-next-gen-automotive-drive-hyperion-9-and-new-drive-map-platform-at-gtc-2022/)

### 3.4 Drive OS软件栈

| 组件 | 说明 |
|------|------|
| **Drive OS** | 基于QNX/Linux的车规级操作系统 |
| **DriveWorks** | 中间件框架，传感器抽象层 |
| **DRIVE AV** | 自动驾驶软件栈（感知、规划、控制） |
| **DRIVE IX** | 座舱体验（可视化、AI功能） |
| **CUDA/cuDNN/TensorRT** | 标准NVIDIA AI库的车规版本 |
| **功能安全** | ASIL-D认证（ISO 26262） |

---

## 4. 前代Jetson平台对比基线

### 4.1 Jetson AGX Xavier系列

| 规格 | Jetson AGX Xavier 64GB | Jetson AGX Xavier 32GB | Jetson Xavier NX |
|------|----------------------|----------------------|-----------------|
| **AI Performance** | 32 TOPS (INT8) | 32 TOPS (INT8) | 21 TOPS |
| **GPU** | 512-core Volta, 64 Tensor Cores | 同左 | 384-core Volta |
| **GPU频率** | 1.21 GHz | 1.21 GHz | 1.1 GHz |
| **CPU** | 8-core Carmel (自研) | 同左 | 6-core Carmel |
| **DLA** | 2x NVDLA | 2x NVDLA | 2x NVDLA |
| **内存** | 32/64GB LPDDR4x (256-bit) | 同左 | 8GB LPDDR4x (128-bit) |
| **内存带宽** | 137 GB/s | 137 GB/s | 51.2 GB/s |
| **FP16 TFLOPs** | 11 TFLOPs | 11 TFLOPs | - |
| **功耗** | 10W/15W/30W | 同左 | 10W/15W |
| **尺寸** | 100mm x 87mm | 同左 | 69.6mm x 45mm |

**来源**: [NVIDIA Jetson AGX Xavier产品页](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/), [NVIDIA Developer Forum](https://forums.developer.nvidia.com/t/how-the-32-tops-of-jetson-agx-xavier-is-calculated/108078), [Macnica](https://www.macnica.co.jp/en/business/semiconductor/manufacturers/nvidia/products/134046/)

### 4.2 Jetson TX2系列

| 规格 | Jetson TX2 NX | Jetson TX2 4GB | Jetson TX2i |
|------|-------------|---------------|------------|
| **GPU** | 256-core Pascal | 256-core Pascal | 256-core Pascal |
| **CPU** | 6-core A57 + Denver2 | 同左 | 同左 |
| **内存** | 4GB LPDDR4 | 8GB LPDDR4 | 8GB LPDDR4 |
| **AI Performance** | 1.33 TFLOPs (FP16) | 1.33 TFLOPs | 1.33 TFLOPs |
| **功耗** | 7.5W/15W | 7.5W/15W | 同左 |

### 4.3 Jetson Nano

| 规格 | Jetson Nano |
|------|------------|
| **GPU** | 128-core Maxwell |
| **CPU** | 4-core Cortex-A57 |
| **内存** | 4GB LPDDR4 |
| **AI Performance** | 0.472 TFLOPs (FP16) |
| **功耗** | 5W/10W |

---

## 5. Orin SoC微架构深度分析

### 5.1 Ampere GPU架构

| 组件 | Orin SoC (AGX Orin 64GB) | Orin SoC (AGX Orin 32GB) | Orin SoC (Orin NX 16GB) | Orin SoC (Orin Nano 8GB) |
|------|--------------------------|--------------------------|-------------------------|--------------------------|
| **GPC (Graphics Processing Cluster)** | 2 | 2 | 1 | 1 |
| **TPC (Texture Processing Cluster)** | 8 | 7 | 4 | 4 |
| **SM (Streaming Multiprocessor)** | 16 | 14 | 8 | 8 |
| **CUDA Cores** | 2048 | 1792 | 1024 | 1024 |
| **Tensor Cores (3rd Gen)** | 64 | 56 | 32 | 32 |
| **RT Cores** | 有 | 有 | 有 | 有 |
| **L2 Cache** | 32MB (共享) | 32MB | 4MB | 4MB |
| **内存控制器** | 256-bit LPDDR5 | 256-bit LPDDR5 | 128-bit LPDDR5 | 128-bit LPDDR5 |

### 5.2 DLA (Deep Learning Accelerator)

| 规格 | Orin SoC |
|------|---------|
| **NVDLA版本** | NVDLA 2.0 |
| **实例数** | 2x (AGX Orin) / 1x (Orin Nano/NX 8GB) |
| **频率** | 1.6 GHz (AGX Orin 64GB) / 1.4 GHz (32GB) |
| **算力** | 52.5 TOPS each (Sparse INT8, 64GB) |
| **支持格式** | INT8, INT16, FP16 |
| **特性** | 硬件调度、内存压缩、卷积加速 |

### 5.3 PVA (Programmable Vision Accelerator)

| 规格 | Orin SoC |
|------|---------|
| **版本** | PVA v2.0 |
| **实例数** | 1x |
| **特性** | VLIW架构，支持计算机视觉原语 |
| **应用** | 光流、立体视觉、图像处理 |

### 5.4 PISP (Image Signal Processor)

| 规格 | Orin SoC |
|------|---------|
| **MIPI CSI lanes** | 16x D-PHY v2.1 (40 Gbps) |
| **C-PHY** | 16x trio links (164 Gbps) |
| **处理能力** | 多摄像头并发处理 |
| **特性** | HDR、降噪、去马赛克、色彩校正 |

### 5.5 内存子系统

| 组件 | 规格 |
|------|------|
| **DRAM类型** | LPDDR5 (Orin) / LPDDR5X (Thor) |
| **ECC支持** | AGX Orin Industrial / Thor T5000 (Alt-link ECC) |
| **加密** | 128-bit AES (Thor) |
| **TrustZone** | 安全内存访问保护 |
| **System MMU** | CPU/GPU统一地址空间 |

### 5.6 安全特性

| 特性 | 说明 |
|------|------|
| **功能安全** | ASIL-D (Drive平台系统级) |
| **安全启动** | 硬件root of trust |
| **密钥存储** | 8MB NOR Secure Key Flash |
| **内存保护** | TrustZone, System MMU |
| **HDCP** | 支持（需激活） |

---

## 6. Thor SoC微架构深度分析

### 6.1 Blackwell GPU架构

| 组件 | Thor T5000 | Thor T4000 |
|------|-----------|-----------|
| **GPC** | 3 | 2 |
| **TPC** | 10 | 6 |
| **CUDA Cores** | 2560 | 1536 |
| **Tensor Cores (5th Gen)** | 96 | 96 |
| **MIG支持** | 是 (2个实例) | 是 (2个实例) |
| **L2 Cache** | 32MB | 32MB |
| **System Cache** | 16MB | 16MB |
| **FP32 TFLOPs** | 8.064 | 4.700 |
| **FP4 TFLOPs (Sparse)** | 2070 | 1200 |
| **FP8 TFLOPs (Dense)** | 517 | 300 |

### 6.2 Neoverse V3AE CPU

| 规格 | 详情 |
|------|------|
| **架构** | Arm Neoverse V3AE (64-bit) |
| **核心数** | 14 (T5000) / 12 (T4000) |
| **频率** | 2.6 GHz |
| **L1 Cache** | 64KB I-cache + 64KB D-cache per core |
| **L2 Cache** | 1MB per core |
| **L3 Cache** | 16MB shared |
| **SPECint@2017** | 6.6 (单核) / 80 (全核, T5000) |
| **SMT** | 不支持 (1 cluster = 1 core) |

### 6.3 关键新特性

| 特性 | 说明 |
|------|------|
| **FP4支持** | 原生FP4量化推理，E2M1编码 |
| **FP8支持** | E4M3/E5M2格式，Transformer Engine加速 |
| **MIG** | Multi-Instance GPU，GPC级硬件隔离 |
| **Transformer Engine** | 专用Transformer推理加速 |
| **结构化稀疏** | 2x稀疏推理加速 |
| **Compute Data Compression** | L2压缩，4x DRAM带宽提升 |
| **PCIe Gen5** | 最高x8 lanes |

---

## 7. 性能与功耗对比

### 7.1 各SKU的TOPS对比

| 产品 | INT8 TOPS | FP16 TFLOPs | FP32 TFLOPs | 功耗范围 | 能效 (TOPS/W) |
|------|-----------|-------------|-------------|----------|--------------|
| Jetson Nano | 0.472 (FP16) | 0.472 | - | 5-10W | ~0.09 |
| Jetson TX2 | 1.33 (FP16) | 1.33 | - | 7.5-15W | ~0.18 |
| Jetson Xavier NX | 21 | 6.8 | - | 10-15W | ~1.4 |
| Jetson AGX Xavier | 32 | 11 | - | 10-30W | ~1.6 |
| Jetson Orin Nano | 67 | 26.8 | - | 7-25W | ~2.7 |
| Jetson Orin NX 16GB | 100 | 40 | - | 10-40W | ~2.5 |
| Jetson AGX Orin 64GB | 275 | 110 | - | 15-60W | ~4.6 |
| Jetson AGX Thor T5000 | ~1035 (FP8) | - | 8.064 | 70-130W | ~8.0 |
| Jetson Thor T4000 | ~600 (FP8) | - | 4.700 | 70-90W | ~6.7 |
| Drive AGX Orin | 254-275 | 110 | - | 15-40W+ | ~6.4 |
| Drive AGX Thor | 1000 | - | - | 75-130W | ~7.7 |

### 7.2 代际性能跃迁

| 代际跃迁 | 算力提升 | 能效提升 | 关键架构变化 |
|----------|---------|---------|-------------|
| Xavier → Orin | ~8x | ~2.5x | Volta→Ampere, DLA升级 |
| Orin → Thor | ~4x | ~1.7x | Ampere→Blackwell, 引入FP4/FP8 |
| TX2 → Orin | ~200x | ~25x | Pascal→Ampere, CPU大幅升级 |

---

## 8. 产品选型指南

### 8.1 按应用场景推荐

| 应用场景 | 推荐产品 | 理由 |
|----------|---------|------|
| 入门级边缘AI/教育 | Jetson Orin Nano Super DevKit ($249) | 性价比高，67 TOPS |
| 机器人/无人机 | Jetson Orin NX 16GB | 100 TOPS，小尺寸 |
| 高端机器人/边缘服务器 | Jetson AGX Orin 64GB | 275 TOPS，全功能 |
| 生成式AI/大模型边缘推理 | Jetson AGX Thor T5000 | 128GB内存，FP4支持 |
| L2+自动驾驶 | Drive AGX Orin | 车规级，ASIL-D |
| L4自动驾驶/Robotaxi | Drive AGX Thor | 1000 TOPS，Hyperion 9 |

### 8.2 按算力需求推荐

| 算力需求 | 产品选择 |
|----------|---------|
| < 10 TOPS | Jetson Nano / Orin Nano 4GB |
| 10-50 TOPS | Jetson Orin Nano 8GB / Xavier NX |
| 50-150 TOPS | Jetson Orin NX 16GB |
| 150-300 TOPS | Jetson AGX Orin 32/64GB |
| 300-1000 TOPS | Jetson Thor T4000 |
| > 1000 TOPS | Jetson Thor T5000 / Drive AGX Thor |

---

## 9. 调研来源清单

### 9.1 官方Datasheet（已下载）

| 文件 | 来源URL | 状态 |
|------|---------|------|
| Jetson AGX Orin Datasheet (DS-10662-001v1.8) | https://static.generation-robots.com/media/Jetson-AGX-Orin-Data-Sheet.pdf | ✅ 已下载 (854KB) |
| Jetson Orin NX Datasheet (DS-10712-001) | https://developer.nvidia.com/downloads/jetson-orin-nx-series-data-sheet | ✅ 已下载 (998KB) |
| Jetson Orin Nano Datasheet | https://static.generation-robots.com/media/jetson-orin-datasheet-nano-modules.pdf | ✅ 已下载 (174KB) |
| Jetson Orin Nano DevKit Datasheet | https://files.seeedstudio.com/wiki/Jetson-Orin-Nano-DevKit/jetson-orin-nano-developer-kit-datasheet.pdf | ✅ 已下载 (188KB) |
| Jetson Thor Datasheet (DS-11945-001_v1.5) | https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/thor/docs/jetson-thor-series-modules-datasheet_ds-11945-001.pdf | ✅ 已下载 (803KB) |
| Jetson AGX Orin Technical Brief | https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf | ✅ 已下载 (930KB) |

### 9.2 官方产品页面

| 页面 | URL |
|------|-----|
| Jetson Orin产品页 | https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/ |
| Jetson Thor产品页 | https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/ |
| Jetson AGX Xavier产品页 | https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/ |
| DRIVE AGX开发者页面 | https://developer.nvidia.com/drive/agx |
| DRIVE Hyperion产品页 | https://www.nvidia.com/en-us/solutions/autonomous-vehicles/drive-hyperion/ |
| Jetson下载中心 | https://developer.nvidia.com/embedded/downloads |

### 9.3 技术分析与新闻

| 来源 | URL | 内容 |
|------|-----|------|
| RidgeRun | https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Thor:_Powering_the_Future_of_Physical_AI | Jetson Thor SoC详细分析 |
| VideoCardz | https://videocardz.com/newz/nvidia-jetson-thor-with-blackwell-gpu-architecture-launched-costs-3499 | Jetson Thor发布新闻 |
| Edge AI and Vision Alliance | https://www.edge-ai-vision.com/2025/09/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/ | Drive AGX Thor开发者套件 |
| IoT Automotive News | https://iot-automotive.news/introducing-nvidia-drive-hyperion-9-next-generation-platform-for-software-defined-autonomous-vehicle-fleets/ | Hyperion 9传感器配置 |
| NVIDIA Blog | https://blogs.nvidia.com/blog/drive-hyperion-9-thor/ | Drive Hyperion 9发布 |
| NVIDIA Developer Blog | https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/ | Jetson Thor介绍 |
| Forbes | https://www.forbes.com/sites/patrickmoorhead/2022/04/07/nvidia-announces-next-gen-automotive-drive-hyperion-9-and-new-drive-map-platform-at-gtc-2022/ | Hyperion 9分析 |
| NVIDIA Developer Forum | https://forums.developer.nvidia.com/t/how-the-32-tops-of-jetson-agx-xavier-is-calculated/108078 | Xavier TOPS计算方式 |
| Macnica | https://www.macnica.co.jp/en/business/semiconductor/manufacturers/nvidia/products/139794/ | AGX Orin规格 |
| Connect Tech | https://connecttech.com/pdf/Jetson_AGX_Xavier_Industrial_Datasheet.pdf | AGX Xavier Industrial |
| generation-robots.com | https://static.generation-robots.com/media/jetson-agx-xavier-module-datasheet.pdf | AGX Xavier Datasheet |
| Mouser | https://www.mouser.com/pdfDocs/Jetson_Orin_NX_Series_and_Orin_Nano_Series_Design_Guide_DG-10931-001_v11.pdf | Orin NX/Nano设计指南 |
| Seeed Studio | https://www.seeedstudio.com/blog/2022/03/16/everything-you-want-to-know-before-getting-nvidia-agx-orin-dev-kit-on-hands/ | AGX Orin DevKit介绍 |
| e-con Systems | https://www.e-consystems.com/blog/camera/technology/what-is-the-nvidia-orin-series-what-are-the-building-blocks-of-nvidia-orin/ | Orin系列架构分析 |
| Reddit r/hardware | https://www.reddit.com/r/hardware/comments/1jmwn7v/nvidia_reveals_jetson_thor_specs_during_gtc_2025/ | Jetson Thor规格讨论 |
| twowin technology | https://twowintech.com/nvidia-jetson-agx-thor-full-analysis/ | AGX Thor完整分析 |
| 株式会社ネクスティ | https://www.nexty-ele.com/en/news/detail/news20250825/ | Drive AGX Thor新闻 |
| Qiita | https://qiita.com/yukoba/items/10d0ba3fb1d19a6ab6a5 | NVIDIA GPU规格总结 |

### 9.4 访问失败/重定向的URL

| URL | 状态 | 说明 |
|-----|------|------|
| https://www.nvidia.com/en-us/automotive/ | 404 | 页面不存在 |
| https://www.nvidia.com/en-us/self-driving-cars/drive-platform/hardware/ | ERR_ABORTED | 被重定向 |
| https://www.nvidia.com/en-us/self-driving-cars/drive-agx/ | ERR_ABORTED | 被重定向 |
| https://developer.nvidia.com/embedded/jetson-agx-orin-datasheet | ERR_ABORTED | 被重定向 |
| https://www.nvidia.com/content/dam/en-zz/Solutions/dgx/nvidia-jetson-agx-orin-datasheet.pdf | 返回HTML | 地域重定向 |
| https://developer.nvidia.com/embedded/downloads | 被劫持 | 浏览器劫持 |

---

## 10. 调研过程记录

### 10.1 关键执行步骤

1. **Jetson Orin产品页抓取** → 成功获取规格总览表
2. **Datasheet PDF下载** → 成功下载6个官方PDF文件
3. **PDF文本提取** → 使用PyMuPDF提取关键规格数据
4. **Google搜索** → 搜索Drive AGX Orin/Thor、Jetson Xavier等规格
5. **技术媒体分析** → 获取RidgeRun、VideoCardz等第三方分析

### 10.2 遇到的问题与解决

| 问题 | 解决方案 |
|------|---------|
| NVIDIA官网地域重定向 | 使用generation-robots.com等第三方镜像下载PDF |
| 浏览器被劫持重定向 | 使用Google搜索+新标签页方式获取信息 |
| WebFetch无法访问 | 改用playwright浏览器+curl直接下载 |
| 部分PDF需要认证 | 寻找公开镜像或使用搜索摘要 |

### 10.3 Token消耗优化

- 使用PyMuPDF本地提取PDF内容，避免将PDF内容发送给LLM
- 使用curl直接下载文件，减少浏览器交互
- 使用Google搜索摘要获取关键信息，减少完整页面加载

---

## 附录：术语表

| 缩写 | 全称 | 说明 |
|------|------|------|
| TOPS | Tera Operations Per Second | 万亿次操作/秒，AI算力单位 |
| GPC | Graphics Processing Cluster | GPU处理集群 |
| TPC | Texture Processing Cluster | 纹理处理集群 |
| SM | Streaming Multiprocessor | 流式多处理器 |
| DLA | Deep Learning Accelerator | 深度学习加速器 |
| PVA | Programmable Vision Accelerator | 可编程视觉加速器 |
| PISP | Image Signal Processor | 图像信号处理器 |
| NVDEC | NVIDIA Video Decoder | 视频解码器 |
| NVENC | NVIDIA Video Encoder | 视频编码器 |
| NVJPEG | NVIDIA JPEG Processor | JPEG处理器 |
| VIC | Video Image Compositor | 视频图像合成器 |
| MGBE | Multi-Gigabit Ethernet | 多千兆以太网 |
| MIG | Multi-Instance GPU | 多实例GPU |
| ASIL | Automotive Safety Integrity Level | 汽车安全完整性等级 |
| TTP | Thermal Transfer Plate | 散热传输板 |
| LPDDR | Low Power Double Data Rate | 低功耗双倍数据速率内存 |

---

*本报告由Claude Code调研代理自动生成，数据截至2026年7月31日。所有规格以NVIDIA官方最新datasheet为准。*
