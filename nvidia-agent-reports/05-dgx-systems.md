# NVIDIA 系统级产品规格深度调研报告

> **调研日期**: 2026-07-31  
> **调研目标**: DGX系统、Superchip、Grace CPU、超算、整机柜产品  
> **信息来源**: NVIDIA官方产品页、Wikipedia、GTC资料、技术白皮书  

---

## 目录

1. [DGX 系统系列](#1-dgx-系统系列)
2. [GB200 NVL72 机架级系统](#2-gb200-nvl72-机架级系统)
3. [Grace CPU](#3-grace-cpu)
4. [Grace Blackwell Superchip / Ultra](#4-grace-blackwell-superchip--ultra)
5. [HGX 平台](#5-hgx-平台)
6. [超算系统](#6-超算系统)
7. [整机柜/机架级产品](#7-整机柜机架级产品)
8. [规格对比总览](#8-规格对比总览)
9. [参考来源](#9-参考来源)

---

## 1. DGX 系统系列

### 1.1 DGX H100

NVIDIA DGX H100 是第四代DGX服务器，基于Hopper架构。

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **GPU** | 8× NVIDIA H100 SXM (Hopper架构) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **GPU内存** | 640 GB HBM3 (8×80GB) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **内存带宽** | 3 TB/s (每GPU) | [NVIDIA H100 page](https://www.nvidia.com/en-us/data-center/h100/) |
| **FP8 AI算力** | 32 PFLOPs | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **CPU** | 2× Intel Xeon Platinum 8480C (Sapphire Rapids) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **系统内存** | 2 TB DDR5 | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **NVLink** | 4th Gen NVLink, 900 GB/s 双向带宽 | [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) |
| **网络** | 2× BlueField-3 DPU, ConnectX-7 400Gb/s IB | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **存储** | 2× 1.92TB SSD (OS), 30.72TB SSD (数据) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **机架尺寸** | 8U | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **功耗** | ~10.2 kW (H100 SXM TDP 700W×8) | 推算 |
| **发布价格** | ~US$482,000 (£379,000) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |

**DGX H100 NVLink 拓扑**:
```
┌─────────────────────── DGX H100 (8U) ───────────────────────┐
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│  │H100 │─│H100 │─│H100 │─│H100 │─│H100 │─│H100 │─│H100 │─│H100 │ │
│  │GPU0 │ │GPU1 │ │GPU2 │ │GPU3 │ │GPU4 │ │GPU5 │ │GPU6 │ │GPU7 │ │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ │
│     │       │       │       │       │       │       │       │      │
│  ┌──┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴──┐  │
│  │              2× 3rd Gen NVSwitch (全互联)                  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────┐  │
│  │Xeon 8480C #1 │──│Xeon 8480C #2 │──│BlueField-3 │──│ConnectX│  │
│  └──────────────┘  └──────────────┘  │   DPU ×2   │──│-7 ×2  │  │
│                                      └────────────┘  └────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 1.2 DGX B200

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **GPU** | 8× NVIDIA Blackwell GPUs | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **GPU内存** | 1,440 GB HBM3e (每GPU 180GB) | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **HBM带宽** | 64 TB/s (聚合) | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **FP4 Tensor Core** | 144 PFLOPS (sparse) \| 72 PFLOPS (dense) | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **FP8 Tensor Core** | 72 PFLOPS (sparse) | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **NVLink** | 2× NVLink Switch (5th Gen), 14.4 TB/s 聚合带宽 | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **CPU** | 2× Intel Xeon Platinum 8570 (112 cores, 2.1GHz Base/4GHz Boost) | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **系统内存** | 2 TB (可配置至4 TB) | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **网络** | 4× OSFP (8× ConnectX-7 VPI, 400Gb/s IB/以太网) | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **DPU** | 2× BlueField-3 DPU (双端口QSFP112, 400Gb/s) | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **存储** | 2× 1.9TB NVMe M.2 (OS), 8× 3.84TB NVMe U.2 | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **功耗** | ~14.3 kW (最大) | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **机架尺寸** | 10U | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **尺寸** | 17.5" H × 19" W × 35.3" L | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| **软件** | NVIDIA AI Enterprise, NVIDIA Mission Control, DGX OS/Ubuntu | [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |

**性能对比 (vs DGX H100)**:
- 训练性能: **3X** (同规模集群)
- 推理性能: **15X** (8×DGX H100 vs 1×DGX B200, per GPU)
- 推理成本: $0.02/百万tokens (GPT-OSS-120B, SemiAnalysis验证)

---

### 1.3 DGX GB200

DGX GB200 基于 GB200 NVL72 机架级系统，详见 [第2节](#2-gb200-nvl72-机架级系统)。

---

### 1.4 DGX Station (GB300)

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **Superchip** | GB300 Grace Blackwell Ultra Desktop Superchip | [NVIDIA DGX Platform](https://www.nvidia.com/en-us/data-center/dgx-platform/) |
| **CPU** | Grace CPU (72 Neoverse V2 ARM cores) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **GPU** | Blackwell GPU | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **统一内存** | 748 GB 相干内存 | [NVIDIA DGX Platform](https://www.nvidia.com/en-us/data-center/dgx-platform/) |
| **HBM3e内存** | 252 GB (连接GPU, 7.1 TB/s带宽) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **支持模型规模** | 最高1万亿参数 (量化) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **交付时间** | 2026年8月起 | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **价格** | $100,000 - $120,000 | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |

---

### 1.5 DGX Spark (Project Digits)

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **Superchip** | GB10 Grace Blackwell Superchip | [NVIDIA DGX Platform](https://www.nvidia.com/en-us/data-center/dgx-platform/) |
| **CPU** | Grace CPU (ARM cores) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **GPU** | 集成Blackwell iGPU | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **统一内存** | 128 GB | [NVIDIA DGX Platform](https://www.nvidia.com/en-us/data-center/dgx-platform/) |
| **支持模型规模** | 最高200亿参数 (量化) | [NVIDIA DGX Platform](https://www.nvidia.com/en-us/data-center/dgx-platform/) |
| **网络** | ConnectX-7 NIC | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **发布时间** | 2025年3月 (GTC 2025) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |

---

## 2. GB200 NVL72 机架级系统

### 2.1 系统规格

GB200 NVL72 是NVIDIA的机架级液冷AI超算系统。

| 规格项 | GB200 NVL72 | GB200 Grace Blackwell Superchip | 来源 |
|--------|-------------|--------------------------------|------|
| **GPU配置** | 72× Blackwell GPUs | 2× Blackwell GPUs | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **CPU配置** | 36× Grace CPUs | 1× Grace CPU | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **NVFP4 Tensor Core** | 1,440 \| 720 PFLOPS | 40 \| 20 PFLOPS | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **FP8/FP6 Tensor Core** | 720 PFLOPS | 20 PFLOPS | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **INT8 Tensor Core** | 720 POPS | 20 POPS | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **FP16/BF16 Tensor Core** | 360 PFLOPS | 10 PFLOPS | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **TF32 Tensor Core** | 180 PFLOPS | 5 PFLOPS | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **FP32** | 5,760 TFLOPS | 160 TFLOPS | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **FP64** | 2,880 TFLOPS | 80 TFLOPS | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **GPU内存** | 13.4 TB HBM3E | 372 GB HBM3E | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **HBM带宽** | 576 TB/s | 16 TB/s | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **NVLink带宽** | 130 TB/s (全机架) | 3.6 TB/s | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **CPU核心** | 2,592 Arm Neoverse V2 cores | 72 Arm Neoverse V2 cores | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **CPU内存** | 17 TB LPDDR5X | Up to 480 GB LPDDR5X | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **CPU内存带宽** | 14 TB/s | Up to 512 GB/s | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **冷却方式** | 液冷 (Liquid Cooled) | 液冷 | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |

### 2.2 NVLink 拓扑架构

```
┌──────────────────── GB200 NVL72 机架 ────────────────────┐
│                                                           │
│  ┌─── NVLink Switch System (5th Gen) ───────────────────┐ │
│  │     18× NVLink Switch Chips (每switch连接4 GPU)      │ │
│  │     总带宽: 130 TB/s                                 │ │
│  │     连接方式: 全互联 (Full Mesh)                     │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─── 36× GB200 Superchip ──────────────────────────────┐ │
│  │                                                      │ │
│  │  ┌──────────────┐  ┌──────────────┐     (×18行)     │ │
│  │  │ Grace CPU    │  │ Grace CPU    │                  │ │
│  │  │ (72 cores)   │  │ (72 cores)   │                  │ │
│  │  │     │        │  │     │        │                  │ │
│  │  │  NVLink-C2C  │  │  NVLink-C2C  │                  │ │
│  │  │  (900GB/s)   │  │  (900GB/s)   │                  │ │
│  │  │     │        │  │     │        │                  │ │
│  │  │ Blackwell GPU│  │ Blackwell GPU│                  │ │
│  │  │  (GPU Pair)  │  │  (GPU Pair)  │                  │ │
│  │  └──────────────┘  └──────────────┘                  │ │
│  │        Superchip #1       Superchip #2               │ │
│  │              ...                 ...                  │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─── 网络互联 ─────────────────────────────────────────┐ │
│  │  Quantum-X80 InfiniBand / Spectrum-X80 Ethernet     │ │
│  │  ConnectX-8 SuperNICs                               │ │
│  └──────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

### 2.3 性能声明

| 工作负载 | 对比H100 | 来源 |
|----------|---------|------|
| **LLM推理 (万亿参数)** | 30X 更快 | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **MoE架构推理** | 10X 性能提升 | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **LLM训练** | 4X 更快 | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **能效** | 25X (相同功耗) | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **数据处理** | 18X vs CPU | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |

---

## 3. Grace CPU

### 3.1 Grace CPU 微架构

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **架构** | 基于ARM Neoverse V2 定制核心 | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **核心数** | 72 cores (单CPU) | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **内存** | LPDDR5X (带ECC) | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **内存带宽** | Up to 1 TB/s | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **与GPU互联** | NVLink-C2C (900 GB/s 双向) | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **功耗效率** | 2X 于同功耗传统CPU | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |

### 3.2 Grace CPU Superchip

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **配置** | 2× Grace CPU | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **总核心数** | 144 Arm Neoverse V2 cores | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **CPU间互联** | NVLink-C2C (900 GB/s) | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **内存带宽** | Up to 1 TB/s | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |

### 3.3 NVIDIA Vera CPU (新一代)

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **架构** | 定制ARM Olympus核心 (Arm兼容) | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **核心数** | 88 cores (HGX Vera Rubin NVL8配置) | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) |
| **线程支持** | SMT (Spatial Multithreading) | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) |
| **内存** | LPDDR5X | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **内存带宽** | Up to 1.2 TB/s | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **CPU-GPU互联** | 2nd Gen NVLink-C2C (1.8 TB/s 双向) | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **缓存一致性** | NVIDIA Scalable Coherency Fabric (SCF) Gen2 | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |

---

## 4. Grace Blackwell Superchip / Ultra

### 4.1 GH200 Grace Hopper Superchip

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **CPU** | 1× Grace CPU (72 Neoverse V2 cores) | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **GPU** | 1× Hopper H100 GPU | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **CPU-GPU互联** | NVLink-C2C (900 GB/s) | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |
| **统一内存** | 支持统一内存空间 | [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) |

### 4.2 GB200 Grace Blackwell Superchip

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **CPU** | 1× Grace CPU (72 Neoverse V2 cores) | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **GPU** | 2× Blackwell GPUs | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **CPU-GPU互联** | NVLink-C2C (900 GB/s) | [NVIDIA GB200 NVL72](httpshttps://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **GPU内存** | 372 GB HBM3E | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **GPU内存带宽** | 16 TB/s | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **NVLink带宽** | 3.6 TB/s (Superchip内部) | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |

### 4.3 GB300 NVL72 (Blackwell Ultra)

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **GPU** | 72× Blackwell Ultra GPUs | [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) |
| **CPU** | 36× Grace CPUs | [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) |
| **冷却** | 全液冷 | [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) |
| **AI工厂输出性能** | 50X vs Hopper平台 | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| **部署状态** | Microsoft, CoreWeave, OCI规模化部署 | [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) |

---

## 5. HGX 平台

### 5.1 HGX H100 规格

| 规格项 | HGX H100 4-GPU | HGX H100 8-GPU | 来源 |
|--------|----------------|----------------|------|
| **GPU** | 4× H100 | 8× H100 | [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) |
| **NVSwitch** | 3rd Gen NVSwitch | 4× 3rd Gen NVSwitch | 推断 |
| **NVLink** | 4th Gen (900 GB/s/GPU) | 4th Gen (900 GB/s/GPU) | [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) |
| **CPU** | 双路x86 | 双路x86 | 推断 |

### 5.2 HGX B200 / B300

| 规格项 | HGX B200 | HGX B300 | 来源 |
|--------|----------|----------|------|
| **GPU** | 8× Blackwell | 8× Blackwell Ultra | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) |
| **NVLink** | 5th Gen | 5th Gen | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) |

### 5.3 HGX Rubin NVL8 (下一代)

| 规格项 | HGX Vera Rubin NVL8 | HGX Rubin NVL8 | 来源 |
|--------|---------------------|----------------|------|
| **GPU** | 8× Rubin SXM | 8× Rubin SXM | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) |
| **CPU** | Single Socket Vera CPU (88 Olympus cores) | x86 CPU | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) |
| **CPU内存** | 1.5TB LPDDR5X (1.2 TB/s) | x86 CPU | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) |
| **NVLink** | 6th Gen (28.8 TB/s) | 6th Gen (28.8 TB/s) | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) |
| **Token工厂吞吐** | 10X vs HGX B200 | 10X vs HGX B200 | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) |
| **训练GPU数量** | 4X fewer vs HGX B200 | 4X fewer vs HGX B200 | [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/) |

---

## 6. 超算系统

### 6.1 Eos 超算

NVIDIA自研的AI超算系统：

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **构成** | 18× H100-based SuperPods | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **DGX系统数** | 576× DGX H100 | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **H100 GPU总数** | 4,608 (576×8) | 推算 |
| **InfiniBand交换机** | 500× Quantum-2 | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **NVLink交换机** | 360× | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **FP8算力** | 18 EFLOPs | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **FP16算力** | 9 EFLOPs | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **TOP500排名** | 第5名 (2023年11月) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |

### 6.2 DGX GH200

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **H100 GPU** | 256× H100 (32 Superchips) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **Grace CPU** | 32× (72-core Neoverse V2) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **共享内存** | 19.5 TB | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **网络** | 32× ConnectX-7 VPI (400Gb/s IB), 16× BlueField-3 (200Gb/s) | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **NVLink Switch** | 支持256 GPU互联 | [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) |
| **全对全带宽** | 57.6 TB/s | [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) |

### 6.3 DGX Helios

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **构成** | 4× DGX GH200 | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **H100 GPU总数** | 1,024 | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **网络** | Quantum-2 InfiniBand | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |

### 6.4 DGX SuperPOD (Hopper世代)

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **规模** | 32× DGX H100 nodes | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **H100 GPU** | 256× | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **CPU** | 64× x86 CPUs | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **HBM3内存** | 20 TB | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **双切带宽** | 70.4 TB/s | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |
| **FP8算力** | 1 ExaFLOP | [NVIDIA DGX Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) |

---

## 7. 整机柜/机架级产品

### 7.1 OVX 系统

| 规格项 | 参数 | 来源 |
|--------|------|------|
| **定位** | 数字孪生、物理AI、工业数字化 | [NVIDIA OVX](https://www.nvidia.com/en-us/data-center/products/ovx/) |
| **GPU** | L40S GPUs | [NVIDIA OVX](https://www.nvidia.com/en-us/data-center/products/ovx/) |
| **网络** | ConnectX-7, BlueField-3 DPU | [NVIDIA OVX](https://www.nvidia.com/en-us/data-center/products/ovx/) |
| **规模** | 支持100+ OVX服务器 | [NVIDIA OVX](https://www.nvidia.com/en-us/data-center/products/ovx/) |
| **软件** | NVIDIA AI Enterprise, Omniverse | [NVIDIA OVX](https://www.nvidia.com/en-us/data-center/products/ovx/) |

### 7.2 OCP 贡献

NVIDIA 对 OCP (Open Compute Project) 的贡献包括：
- HGX 基板设计开放给OCP成员
- MGX 参考架构 (模块化服务器设计)
- OVX 系统规范

---

## 8. 规格对比总览

### 8.1 DGX 系统世代对比

| 规格 | DGX H100 | DGX B200 | DGX Station | DGX Spark |
|------|----------|----------|-------------|-----------|
| **架构** | Hopper | Blackwell | Blackwell Ultra | Blackwell |
| **GPU** | 8× H100 | 8× Blackwell | GB300 Superchip | GB10 Superchip |
| **GPU内存** | 640GB HBM3 | 1,440GB HBM3e | 252GB HBM3e | 共享128GB |
| **CPU** | 2× Xeon 8480C | 2× Xeon 8570 | Grace (72c) | Grace |
| **系统内存** | 2TB | 2-4TB | 748GB统一 | 128GB统一 |
| **NVLink** | 4th Gen | 5th Gen | NVLink-C2C | NVLink-C2C |
| **NVLink带宽** | 900GB/s/GPU | 14.4TB/s聚合 | - | - |
| **网络** | 400Gb IB | 400Gb IB | - | ConnectX-7 |
| **机架尺寸** | 8U | 10U | 台式机 | 台式机 |
| **功耗** | ~10.2kW | ~14.3kW | - | - |

### 8.2 NVLink 代际演进

| 代际 | 产品 | 单链路带宽 | 聚合带宽 | 最大GPU数 |
|------|------|-----------|----------|-----------|
| **3rd Gen** | A100 | 600 GB/s | - | 8 |
| **4th Gen** | H100 | 900 GB/s | - | 256 (NVLink Switch) |
| **5th Gen** | B200/GB200 | 1.8 TB/s | 130 TB/s (NVL72) | 576 |
| **6th Gen** | Rubin | - | 28.8 TB/s (8-GPU) | - |

### 8.3 HBM 规格演进

| 代际 | 产品 | 每Stack容量 | 每Stack带宽 | 每GPU总带宽 |
|------|------|-------------|-------------|-------------|
| **HBM2e** | A100 80GB | 8 GB | 307 GB/s | 2,039 GB/s |
| **HBM3** | H100 80GB | 10 GB | 819 GB/s | 3,350 GB/s |
| **HBM3e** | H200/B200 | 24-36 GB | 1,229 GB/s | 8,000+ GB/s |

---

## 9. 参考来源

### 9.1 NVIDIA官方页面

1. [NVIDIA DGX Platform](https://www.nvidia.com/en-us/data-center/dgx-platform/) - DGX平台总览
2. [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) - DGX B200规格
3. [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) - GB200 NVL72规格
4. [NVIDIA Grace CPU](https://www.nvidia.com/en-us/data-center/grace-cpu/) - Grace CPU规格
5. [NVIDIA HGX Platform](https://www.nvidia.com/en-us/data-center/hgx/) - HGX平台规格
6. [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) - Blackwell架构
7. [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) - Hopper架构
8. [NVIDIA H100 GPU](https://www.nvidia.com/en-us/data-center/h100/) - H100规格
9. [NVIDIA OVX Systems](https://www.nvidia.com/en-us/data-center/products/ovx/) - OVX系统

### 9.2 技术文档

10. [NVIDIA H100 GPU Whitepaper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) - H100白皮书(71页)
11. [NVIDIA H100 GPU Datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet) - H100数据手册

### 9.3 Wikipedia参考

12. [Nvidia DGX - Wikipedia](https://en.wikipedia.org/wiki/Nvidia_DGX) - DGX系统完整历史
13. [High Bandwidth Memory - Wikipedia](https://en.wikipedia.org/wiki/High_Bandwidth_Memory) - HBM规格

### 9.4 其他来源

14. [NVIDIA GTC 2024 Keynote](https://resources.nvidia.com/en-us-dgx-systems/gtc-2024-next-gen-dgx-architecture) - 下一代DGX架构
15. [NVIDIA Grace CPU Superchip Datasheet](https://resources.nvidia.com/en-us-dgx-gh200/nvidia-grace-hopper-superchip-datasheet) - GH200数据手册
16. [NVIDIA GB200 NVL4 Datasheet](https://resources.nvidia.com/en-us-dgx-gh200/nvidia-dgx-gh200-datasheet-web-us) - GB200 NVL4数据手册

---

## 附录：调研过程记录

### A. 关键发现

1. **NVLink带宽跳跃**: 从H100的900GB/s到B200/GB200的1.8TB/s (2X提升)，NVLink Switch聚合带宽达到130TB/s
2. **统一内存架构**: Grace CPU通过NVLink-C2C (900GB/s) 与GPU连接，实现统一内存空间
3. **机架级系统崛起**: GB200 NVL72将72个GPU通过NVLink Switch连接成单一GPU域
4. **液冷成为标配**: B200/GB200/B300系统全面采用液冷设计
5. **桌面AI超算**: DGX Spark (128GB, 200B参数) 和 DGX Station (748GB, 1T参数) 将AI带到桌面

### B. 调研限制

- 部分NVIDIA产品页面因地区限制或页面更新导致访问异常
- 部分规格为官方声明值，实际性能可能因工作负载而异
- GB300/B300详细规格仍在披露中

---

*报告完成时间: 2026-07-31*
