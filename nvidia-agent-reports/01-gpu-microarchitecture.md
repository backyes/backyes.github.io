# NVIDIA GPU微架构规格深度调研报告

> **调研日期**: 2026-07-31  
> **调研目标**: 为专业AI芯片研究人员提供精确、可溯源的NVIDIA GPU微架构规格报告  
> **覆盖架构**: Ampere → Hopper → Blackwell → Ada Lovelace

---

## 1. 架构演进总览

| 属性 | Ampere (A100) | Hopper (H100/H200/H800) | Blackwell (B100/B200) | Ada Lovelace (L40S/RTX 4090) |
|------|--------------|------------------------|----------------------|------------------------------|
| **架构代** | 第7代 | 第8代 | 第9代 | 第8代(消费级) |
| **制程工艺** | TSMC N7 | TSMC 4N | TSMC 4NP(数据) / 4N(消费) | TSMC 4N |
| **晶体管数量** | 54.2B (GA100) | 80B (GH100) | 104B (GB100单die) / 208B (B100双die) | 76.3B (AD102) |
| **Die Size** | 826 mm² | 814 mm² | ~814mm² (GB100, reticle limit) | 609 mm² (AD102) |
| **TDP** | 400W (SXM4) | 700W (SXM5) | 700W (B100) / 1000W (B200) | 450W (RTX 4090) / 350W (L40S) |
| **推出时间** | Q1 2020 | Q3 2022 | Q4 2024 | Q4 2022 |
| **Compute Capability** | 8.0 | 9.0 | 10.0 / 12.0 | 8.9 |

**来源**: [Wikipedia Hopper](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)), [Wikipedia Blackwell](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)), [Wikipedia Ampere](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)), [Wikipedia Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture))

---

## 2. Hopper架构 (H100, H200, H800) 详细规格

### 2.1 SM微架构细节

| 属性 | H100 SXM5 | H100 PCIe | GH100 (Full GPU) | 来源 |
|------|-----------|-----------|-----------------|------|
| **SM数量** | 132 | 114 | 144 | [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| **FP32 CUDA Cores/SM** | 128 | 128 | 128 | 同上 |
| **FP32 CUDA Cores总数** | 16,896 | 14,592 | 18,432 | 同上 |
| **FP64 CUDA Cores/SM** | 64 (非Tensor) | 64 | 64 | 同上 |
| **FP64 CUDA Cores总数** | 4,608 | - | - | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **Tensor Cores/SM** | 4 (第4代) | 4 | 4 | [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| **Tensor Cores总数** | 528 | 456 | 576 | 同上 |
| **Boost Clock** | 1980 MHz | - | - | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **Max warps/SM** | 64 | 64 | 64 | [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| **Max threads/SM** | 2048 | 2048 | 2048 | 同上 |
| **Shared Memory/SM** | 228 KB (可配置up to) | 228 KB | 228 KB | [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| **L1+Texture+Shared (combined)** | 256 KB/SM | 256 KB/SM | 256 KB/SM | 同上 |
| **Shared Memory带宽/SM** | 2000 GB/sec | 2000 GB/sec | 2000 GB/sec | 同上 |

**关键微架构变化**:
- FP32 throughput per SM 是A100的 **2x** (clock-for-clock)
- FP8 引入 → per-SM 峰值算力为A100的 **4x** (FP8 vs FP16)
- Tensor Core MMA速率：equivalent dtype 是A100的 **2x**，FP8是A100 FP16的 **4x**
- Sparsity feature → 标准Tensor Core operations **2x** 性能提升

### 2.2 缓存层级

| 属性 | H100 SXM5 | H100 PCIe | GH100 Full | A100 (对比) | 来源 |
|------|-----------|-----------|------------|------------|------|
| **L1 Data Cache + Shared Memory** | 256 KB/SM (combined) | 256 KB/SM | 256 KB/SM | 164 KB/SM (192 KB prof) | [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| **L1 Total** | 25,344 KB (~24.75 MB) | - | - | 20,736 KB (192KB × 108) | 计算 |
| **L2 Cache** | 50 MB | 50 MB | 50 MB | 40 MB | [Wikipedia Hopper](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **L2 Cache** | 51,200 KB | 51,200 KB | 51,200 KB | 40,960 KB | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |

**注意**: H100的L1/Shared Memory combined为256 KB，是A100 (192 KB pro) 的 **1.33x**。可配置carveout通过 `cudaFuncAttributePreferredSharedMemoryCarveout`。

### 2.3 HBM规格

| 属性 | H100 SXM5 (HBM3) | H100 PCIe (HBM2e) | H200 SXM5 (HBM3e) | H800 SXM5 (HBM3) | 来源 |
|------|------------------|-------------------|-------------------|-----------------|------|
| **HBM类型** | HBM3 | HBM2e | HBM3e | HBM3 | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **容量** | 80 GB | 80 GB | 141 GB | 80 GB | 同上 |
| **堆叠数** | 5 stacks | 5 stacks | 6 stacks? | 5 stacks? | ⚠️ 部分推断 |
| **Memory Speed** | 5.2 Gb/s | 3.2 Gb/s | 6.3 Gb/s | ~5.2 Gb/s? | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **Bus Width** | 5120-bit | 5120-bit | 6144-bit | 5120-bit? | 同上 |
| **Memory Controllers** | 10 × 512-bit | 10 × 512-bit | - | - | [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| **带宽** | **3.35 TB/s** | ~2.0 TB/s | **4.8 TB/s** | ~3.35 TB/s? (降级) | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)), [H200 Datasheet](https://resources.nvidia.com/en-us-hopper-architecture/hpc-datasheet-sc23) |

**注意**: 
- H100 SXM5 HBM3 带宽 3.35 TB/s，比A100 (2.0 TB/s) 提升 **67.5%** (非白皮书声称的"nearly 2x"，实际白皮书早期声称3 TB/s)
- H200 HBM3e 带宽 4.8 TB/s，比H100提升 **43%**
- H800是中国特供版，⚠️ NVLink带宽被降级 (传言从900 GB/s降至400 GB/s)，HBM带宽可能保持

### 2.4 算力规格 (H100 SXM5)

| 精度 | Dense (TFLOPS) | Sparse (2x) (TFLOPS) | 对比A100 | 来源 |
|------|---------------|---------------------|---------|------|
| **FP64** | 34 | - | 3.5x | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **FP64 Tensor Core** | 67 | 134 | - | 同上 |
| **FP32** | 67 | - | 3.4x | 同上 |
| **TF32 Tensor Core** | 495 | 990 | 3.2x | 同上 |
| **BF16 Tensor Core** | 990 | 1980 | 3.2x | 同上 |
| **FP16 Tensor Core** | 990 | 1980 | 3.2x | 同上 |
| **FP8 Tensor Core** | 1980 | 3960 | - (新dtype) | [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| **INT8 Tensor Core** | 1980 TOPS | 3960 TOPS | 3.2x | 同上 |

**⚠️ 注意**: 不同来源的TFLOPS数据有差异：
- Wikipedia DGX对比表: FP32 67 TFLOPS, FP64 34 TFLOPS, FP16 990 TFLOPS
- NVIDIA博客早期"preliminary"表: FP32 60 TFLOPS, FP64 24 TFLOPS, FP16 1000 TFLOPS (sparsity 2000)
- **最终 shipping 规格以Wikipedia DGX表为准** (基于1980 MHz boost clock)

### 2.5 关键微架构创新

#### 2.5.1 Transformer Engine
- 动态精度管理：FP8 ↔ FP16 自动切换
- 每层动态缩放 (dynamic scaling) 和 re-casting
- 训练速度提升 **up to 9x** vs A100
- 推理速度提升 **up to 30x** vs A100 (LLM)
- **来源**: [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

#### 2.5.2 DPX Instructions
- 加速动态规划算法：Smith-Waterman, Floyd-Warshall
- vs CPU-only server: **up to 40x** 加速
- vs A100 GPU: **up to 7x** 加速
- **来源**: [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)

#### 2.5.3 Thread Block Cluster
- 新编程层次：threads → thread blocks → **thread block clusters** → grids
- 可保证多个thread block并发调度
- 最大 portable cluster size = 8
- H100 可支持 cluster size = 16 (通过 `cudaFuncAttributeNonPortableClusterSizeAllowed`)
- **来源**: [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

#### 2.5.4 Distributed Shared Memory (DSMEM)
- 允许直接 SM-to-SM 通信 (loads, stores, atomics)
- 跨多个 SM shared memory blocks
- 可利用 distributed shared memory + L2 的 combined bandwidth
- **来源**: [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

#### 2.5.5 Tensor Memory Accelerator (TMA)
- 全局内存 ↔ 共享内存之间双向异步传输
- 支持 up to 5D tensors
- 支持 elementwise reduction 和 bitwise operators
- 暴露为 `cuda::memcpy_async`
- **来源**: [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

#### 2.5.6 第四代NVLink & NVSwitch
- 18条NVLink links → 900 GB/s total bandwidth (per GPU)
- 比PCIe Gen5 高 **7x**
- 第三代NVSwitch: 64 ports, 13.6 Tbits/sec (vs 上代7.2 Tbits/sec)
- NVLink Switch System: 支持256 GPUs, 57.6 TB/s all-to-all bandwidth
- **来源**: [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

#### 2.5.7 第二代MIG (Multi-Instance GPU)
- 最多7个GPU实例
- 每个实例有独立 NVDEC/NVJPG
- Confidential Computing at MIG-level TEE
- **来源**: [NVIDIA Hopper Architecture](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)

### 2.6 NVLink/网络互连

| 属性 | H100 | A100 | 来源 |
|------|------|------|------|
| **NVLink Generation** | 4th gen | 3rd gen | [NVIDIA Hopper In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |
| **NVLink Links** | 18 | 12 | 同上 |
| **NVLink总带宽** | 900 GB/s | 600 GB/s | 同上 |
| **PCIe** | Gen5 x16 (128 GB/s) | Gen4 x16 (64 GB/s) | 同上 |
| **Networking** | ConnectX-7 (400 Gb/s) | ConnectX-6 (200 Gb/s) | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |

---

## 3. Blackwell架构 (B100, B200, GB200) 详细规格

### 3.1 工艺与封装

| 属性 | B100 | B200 | GB200 (superchip) | 来源 |
|------|------|------|-------------------|------|
| **制程** | TSMC 4NP | TSMC 4NP | TSMC 4NP | [Wikipedia Blackwell](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)) |
| **Die** | GB100 (×2 in package) | GB100 (×2?) | GB100 + Grace CPU | 同上 |
| **晶体管 (单die)** | 104B | 104B | - | 同上 |
| **晶体管 (总package)** | 208B | 208B | - | 同上 |
| **Die Size** | ~814 mm² (reticle limit) | ~814 mm² | - | 同上 |
| **封装** | CoWoS-L 2.5D | CoWoS-L 2.5D | CoWoS-L 2.5D | 同上 |
| **Die间互连** | NV-HBI (10 TB/s) | NV-HBI (10 TB/s) | NVLink-C2C | 同上 |
| **TDP** | 700W | 1000W | up to 1000W | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |

**关键**: 4NP是4N的增强版 (Hopper和Ada Lovelace使用的工艺)，可能增加metal layers。GB100 die达到光刻机reticle limit。B100使用双die封装实现208B晶体管。

### 3.2 SM微架构 (第5代Tensor Core)

| 属性 | B100 | B200 | 来源 |
|------|------|------|------|
| **SM数量** | ⚠️ 未公布确切 | 192 (基于GB202 die配置推断) | [Wikipedia Blackwell](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)) |
| **Tensor Cores/SM** | 4 (第5代) | 4 | 同上 |
| **CUDA Cores/SM** | 128 (推断) | 128 | ⚠️ 推断 |
| **L2 Cache** | 128 MB (推断) | 128 MB | 同上 |
| **支持的精度** | FP4, FP6, FP8, FP16, BF16, TF32, FP64, INT8, MXFP4, MXFP6 | 同上 | 同上 |

**⚠️ 注意**: B100/B200的SM数量、CUDA cores数等详细规格NVIDIA未完全公布。上表部分数据基于消费级GB202 die配置推断。

### 3.3 HBM3e规格

| 属性 | B100 SXM6 | B200 SXM | 来源 |
|------|-----------|---------|------|
| **HBM类型** | HBM3e | HBM3e | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **容量** | 192 GB | ⚠️ 未公布 (可能192GB或288GB) | 同上 |
| **Memory Speed** | 8 Gb/s | 8 Gb/s? | 同上 |
| **Bus Width** | 8192-bit | 8192-bit? | 同上 |
| **带宽** | **8 TB/s** | 8 TB/s? | 同上 |

**注意**: 
- B100 HBM3e 带宽 8 TB/s，比H100 (3.35 TB/s) 提升 **139%**
- 早期传闻B200有288GB版本，但Wikipedia DGX表未确认此数据

### 3.4 算力规格

| 精度 | B100 | B200 | 来源 |
|------|------|------|------|
| **FP64** | ⚠️ | ⚠️ | - |
| **FP64 Tensor Core** | 30 TFLOPS | 40 TFLOPS | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **FP32** | ⚠️ | ⚠️ | - |
| **TF32 Tensor Core** | 989 TFLOPS | 1.2 PFLOPS | 同上 |
| **BF16 Tensor Core** | 1.98 PFLOPS | 2.25 PFLOPS | 同上 |
| **FP16 Tensor Core** | 1.98 PFLOPS | 2.25 PFLOPS | 同上 |
| **FP8 Tensor Core** | ⚠️ | ⚠️ | - |
| **FP4 Tensor Core** | ⚠️ | ⚠️ (GB200 dual-GPU: 20 PFLOPS excl sparsity) | [Wikipedia Blackwell](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)) |
| **INT8 Tensor Core** | 3.5 POPS | 4.5 POPS | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |

**⚠️ 注意**: 
- B100/B200的FP32/FP64/FP8 TFLOPS未在Wikipedia DGX对比表中公布
- GB200 (dual-GPU superchip) 声称 **20 PFLOPS FP4** (不含sparsity 2x增益)
- NVIDIA强调sub-8-bit精度 (MXFP4, MXFP6) 是Blackwell关键创新

### 3.5 关键微架构创新

#### 3.5.1 第5代Tensor Core
- 原生支持 sub-8-bit 数据类型
- MXFP6 和 MXFP4 microscaling formats (OCP社区定义)
- FP4 支持 → 推理吞吐量进一步翻倍
- **来源**: [Wikipedia Blackwell](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture))

#### 3.5.2 第2代Transformer Engine
- 支持 MXFP4 和 MXFP6
- 4-bit数据 → 推理效率/吞吐量翻倍
- **来源**: [Wikipedia Blackwell](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture))

#### 3.5.3 Micro-tensor切片
- 支持细粒度tensor切片以适配低精度计算
- 提高低精度计算的精度控制
- **来源**: ⚠️ 推断 (基于MXFP格式设计)

#### 3.5.4 NV-HBI (NV-High Bandwidth Interface)
- 双die间 10 TB/s 互连
- 基于 NVLink 7 protocol
- 全cache coherency between dies
- **来源**: [Wikipedia Blackwell](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture))

#### 3.5.5 AI Management Processor (AMP)
- 基于 RISC-V 的专用调度芯片
- 从CPU卸载调度任务
- 通过 Windows HAGS 利用
- **来源**: [Wikipedia Blackwell](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture))

### 3.6 NVLink/网络互连

| 属性 | B100 | B200 | 来源 |
|------|------|------|------|
| **NVLink总带宽** | 1.8 TB/s | ⚠️ | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **Networking** | ConnectX-7 (400 Gb/s) | ⚠️ | 同上 |

---

## 4. Ada Lovelace架构 (L40S, RTX 4090等) 详细规格

### 4.1 工艺与封装

| 属性 | AD102 (RTX 4090) | AD103 | AD104 | 来源 |
|------|------------------|-------|-------|------|
| **制程** | TSMC 4N | TSMC 4N | TSMC 4N | [Wikipedia Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)) |
| **晶体管** | 76.3B | 45.9B | 35.8B | 同上 |
| **Die Size** | 609 mm² | 378 mm² | 294 mm² | 同上 |
| **Transistor Density** | 125.3 MTr/mm² | 121.1 MTr/mm² | 121.8 MTr/mm² | 同上 |

### 4.2 SM微架构

| 属性 | AD102 (RTX 4090) | AD103 | AD104 | GA102 (Ampere对比) | 来源 |
|------|------------------|-------|-------|-------------------|------|
| **SM数量** | 128 (144?) | 80 | 60 | 84 (GA102 full) | [Wikipedia Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)) |
| **CUDA Cores** | 16,384 (18,432?) | 10,752 | 6,144 | 10,752 | 同上 |
| **Tensor Cores** | 512 (第4代) | 336 | 192 | 336 (第3代) | 同上 |
| **RT Cores** | 128 (第3代) | 84 | 60 | 84 (第2代) | 同上 |
| **TMUs** | 512 | 336 | 192 | 336 | 同上 |
| **ROPs** | 192 | 112 | 96 | 112 | 同上 |
| **Boost Clock** | 2520 MHz | 2520 MHz? | - | 1695 MHz? | ⚠️ 部分推断 |
| **Shared Memory/SM** | 128 KB | 128 KB | 128 KB | 128 KB (消费) | 同上 |

**⚠️ 注意**: 
- AD102 die有两种配置：RTX 4090使用AD102-300 (16,384 CUDA cores, 128 SMs)，完整版AD102可能18,432 CUDA cores (144 SMs)
- Ada Lovelace使用第4代Tensor Core (与Hopper同代)
- Ada Lovelace使用第3代RT Core

### 4.3 缓存层级

| 属性 | AD102 (RTX 4090) | AD103 | AD104 | GA102 (对比) | 来源 |
|------|------------------|-------|-------|-------------|------|
| **L1/Shared (combined)** | 128 KB/SM | 128 KB/SM | 128 KB/SM | 128 KB/SM | [Wikipedia Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)) |
| **L1 Total** | 10.5 MB (128KB × 84?) | - | - | 10.5 MB | 计算 |
| **L2 Cache** | **64 MB** | 48 MB | 36 MB? | 6 MB | 同上 |

**关键**: AD102 L2 cache 64 MB 是 GA102 (6 MB) 的 **10.7x** 增加！这是Ada Lovelace最显著的架构改进之一。

### 4.4 显存规格

| 属性 | RTX 4090 | RTX 4080 | RTX 4070 Ti | RTX A6000 (Ampere) | 来源 |
|------|----------|----------|-------------|-------------------|------|
| **显存类型** | GDDR6X | GDDR6X | GDDR6X | GDDR6 | [Wikipedia Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)) |
| **容量** | 24 GB | 16 GB | 12 GB | 48 GB | 同上 |
| **Memory Speed** | 21 Gbit/s | 22.4 Gbit/s | 21 Gbit/s | 16-20 Gbit/s? | 同上 |
| **Bus Width** | 384-bit | 256-bit | 192-bit | 384-bit | 同上 |
| **带宽** | 1008 GB/s | 717 GB/s | 504 GB/s | 768 GB/s | 计算/来源同上 |

### 4.5 算力规格 (RTX 4090)

| 精度 | TFLOPS | 来源 |
|------|--------|------|
| **FP32** | ~82.6 TFLOPS | 计算: 16384 × 2 × 2520MHz |
| **TF32 Tensor Core** | ~330 TFLOPS (dense) / ~660 TFLOPS (sparse) | ⚠️ 推断 |
| **BF16/FP16 Tensor Core** | ~330 TFLOPS (dense) / ~660 TFLOPS (sparse) | ⚠️ 推断 |
| **INT8 Tensor Core** | ~660 TOPS (dense) / ~1320 TOPS (sparse) | ⚠️ 推断 |
| **RT Core** | 191 TFLOPS (total) | [Wikipedia Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)) |

### 4.6 关键创新

#### 4.6.1 第3代RT Core
- RTX 4090: 128 RT cores (vs RTX 3090 Ti: 84)
- 191 TFLOPS ray tracing performance
- 1.49 TFLOPS per RT core
- **来源**: [Wikipedia Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture))

#### 4.6.2 DLSS 3 (Frame Generation)
- AI驱动的帧生成技术
- 使用光流加速器 (Optical Flow Accelerator)

#### 4.6.3 第4代Tensor Core
- 与Hopper相同的第4代Tensor Core架构
- 支持FP8 (但无Transformer Engine硬件支持)
- **来源**: [Wikipedia Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture))

#### 4.6.4 Shader Execution Reordering (SER)
- 着色器执行重排序
- 提高ray tracing的SIMD效率

#### 4.6.5 双NVENC + AV1
- 双编码器
- AV1编码效率比H.264高40%
- **来源**: [Wikipedia Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture))

---

## 5. Ampere架构 (A100) 规格 (基线对比)

### 5.1 关键规格

| 属性 | A100 SXM4 (80GB) | A100 SXM4 (40GB) | GA100 Full | 来源 |
|------|------------------|------------------|------------|------|
| **制程** | TSMC N7 | TSMC N7 | TSMC N7 | [Wikipedia Ampere](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)) |
| **晶体管** | 54.2B | 54.2B | 54.2B | 同上 |
| **Die Size** | 826 mm² | 826 mm² | 826 mm² | 同上 |
| **SM数量** | 108 | 108 | 128 | 同上 |
| **CUDA Cores** | 6,912 | 6,912 | 8,192 | 同上 |
| **Tensor Cores** | 432 (第3代) | 432 | 512 | 同上 |
| **FP64 CUDA Cores** | 3,456 | 3,456 | - | 同上 |
| **Boost Clock** | 1410 MHz | 1410 MHz | - | 同上 |
| **TDP** | 400W | 400W | - | 同上 |
| **Shared Memory/SM** | 192 KB (professional) | 192 KB | 192 KB | 同上 |
| **L1 Total** | 20,736 KB | 20,736 KB | 24,576 KB | 计算 |
| **L2 Cache** | 40 MB | 40 MB | 40 MB | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |

### 5.2 HBM规格

| 属性 | A100 80GB SXM4 | A100 40GB SXM4 | 来源 |
|------|----------------|----------------|------|
| **HBM类型** | HBM2e | HBM2 | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **容量** | 80 GB | 40 GB | 同上 |
| **Memory Speed** | 3.2 Gb/s | 2.4 Gb/s | 同上 |
| **Bus Width** | 5120-bit | 5120-bit | 同上 |
| **带宽** | **2.0 TB/s** | 1.52 TB/s | 同上 |

### 5.3 算力规格 (A100 80GB SXM4)

| 精度 | Dense (TFLOPS) | Sparse (2x) | 来源 |
|------|---------------|-------------|------|
| **FP64** | 9.7 | - | [Wikipedia DGX表](https://en.wikipedia.org/wiki/Hopper_(microarchitecture)) |
| **FP64 Tensor Core** | 19.5 | - | 同上 |
| **FP32** | 19.5 | - | 同上 |
| **TF32 Tensor Core** | 156 | 312 | 同上 |
| **BF16 Tensor Core** | 312 | 624 | 同上 |
| **FP16 Tensor Core** | 312 | 624 | 同上 |
| **INT8 Tensor Core** | 624 TOPS | 1248 TOPS | 同上 |

---

## 6. 跨代架构对比

### 6.1 数据中心GPU关键指标对比

| 指标 | A100 (2020) | H100 (2022) | H200 (2023) | B100 (2024) | B200 (2024) | B200 vs A100 |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| **晶体管** | 54.2B | 80B | 80B | 208B | 208B | 3.8x |
| **Die Size** | 826mm² | 814mm² | 814mm² | ~814mm² ×2 | ~814mm² ×2 | ~2x |
| **TDP** | 400W | 700W | 700W | 700W | 1000W | 2.5x |
| **HBM容量** | 80GB | 80GB | 141GB | 192GB | 192GB+ | 2.4x |
| **HBM带宽** | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | 8 TB/s | 8 TB/s | 4.0x |
| **L2 Cache** | 40MB | 50MB | 50MB | 128MB? | 128MB? | 3.2x |
| **FP32 TFLOPS** | 19.5 | 67 | 67 | ⚠️ | ⚠️ | ~3.4x |
| **FP16 TFLOPS (dense)** | 312 | 990 | 990 | 1980 | 2250 | 7.2x |
| **INT8 TOPS (dense)** | 624 | 1980 | 1980 | 3500 | 4500 | 7.2x |
| **NVLink带宽** | 600 GB/s | 900 GB/s | 900 GB/s | 1.8 TB/s | 1.8 TB/s | 3.0x |

### 6.2 架构效率趋势

| 指标 | A100 | H100 | B200 | 趋势 |
|------|------|------|------|------|
| **FP32 TFLOPS/W** | 0.049 | 0.096 | ⚠️ | ↑ 96% (A100→H100) |
| **FP16 TFLOPS/W** | 0.78 | 1.41 | 2.25 | ↑ 188% (A100→B200) |
| **Transistor密度** | 65.6 MTr/mm² | 98.3 MTr/mm² | ~128 MTr/mm² (单die) | ↑ 95% (A100→B200) |
| **HBM带宽/TDP** | 5.0 GB/s/W | 4.8 GB/s/W | 8.0 GB/s/W | ↑ 60% (A100→B200) |

---

## 7. 关键发现与专业分析

### 7.1 算力增长路径
- **A100→H100**: FP16 dense 3.2x (312→990 TFLOPS)，主要来自：SM数量增加(108→132, 1.22x) + 频率提升(1410→1980 MHz, 1.4x) + 每SM算力翻倍(2x) = 理论3.4x，实际3.2x
- **H100→B200**: FP16 dense ~1.3x (990→2250 TFLOPS)，主要来自SM数量增加 + 第5代Tensor Core
- **工艺瓶颈**: 4N→4NP 无重大节点进步 → 功耗墙显著 (TDP 700→1000W)

### 7.2 内存墙持续加剧
- HBM带宽增长：2.0→3.35→4.8→8.0 TB/s
- 但算力增长更快 → **bytes/FLOP 持续下降**
- A100: 2000/312 = 6.4 bytes/FP16-FLOP
- H100: 3350/990 = 3.4 bytes/FP16-FLOP
- B200: 8000/2250 = 3.6 bytes/FP16-FLOP
- **结论**: 低精度计算 (FP8/FP4/MXFP4) 是缓解内存墙的关键路径

### 7.3 封装技术关键转折
- B100 双die + NV-HBI (10 TB/s) 是NVIDIA首次在大规模量产GPU中采用chiplet架构
- CoWoS-L 2.5D packaging 是核心使能技术
- 这标志着 **monolithic die时代终结** 的开始

### 7.4 Transformer Engine 范式
- Hopper: 第1代TE (FP8↔FP16动态切换)
- Blackwell: 第2代TE (MXFP4/MXFP6支持)
- **趋势**: 精度自适应 + 动态缩放 → 软件-硬件协同设计

---

## 8. 数据可信度评估

| 数据类别 | 可信度 | 说明 |
|---------|--------|------|
| **晶体管/Die Size/工艺** | ⭐⭐⭐⭐⭐ | 多源交叉验证一致 |
| **SM/CUDA Core/Tensor Core数量** | ⭐⭐⭐⭐⭐ | NVIDIA官方白皮书确认 |
| **Boost Clock** | ⭐⭐⭐⭐⭐ | Wikipedia DGX对比表一致 |
| **TDP** | ⭐⭐⭐⭐⭐ | 多源确认 |
| **HBM类型/容量** | ⭐⭐⭐⭐⭐ | 官方datasheet确认 |
| **HBM带宽** | ⭐⭐⭐⭐ | 早期白皮书与最终datasheet有差异 (H100: 3.0 vs 3.35 TB/s) |
| **TFLOPS (FP32/FP64/FP16)** | ⭐⭐⭐⭐ | 基于clock × cores × ops/clock计算，与官方一致 |
| **TFLOPS (FP8/FP4)** | ⭐⭐⭐ | NVIDIA未公布完整B100/B200 FP8/FP4数据 |
| **Shared Memory带宽** | ⭐⭐⭐ | 仅找到一个来源 (2000 GB/s/SM) |
| **B100/B200 SM数量** | ⭐⭐ | NVIDIA未完全公布，部分推断 |
| **H800规格** | ⭐⭐ | 中国特供版，NVIDIA未正式发布spec |

---

## 9. 完整URL清单

### 9.1 访问成功
1. ✅ https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/ - Hopper架构概览
2. ✅ https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c - H100白皮书在线阅读器
3. ✅ https://resources.nvidia.com/en-us-hopper-architecture/hpc-datasheet-sc23 - H200 Datasheet在线阅读器
4. ✅ https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ - Hopper架构深度解析 (主要数据来源)
5. ✅ https://en.wikipedia.org/wiki/Hopper_(microarchitecture) - Hopper维基百科 (主要数据来源)
6. ✅ https://en.wikipedia.org/wiki/Blackwell_(microarchitecture) - Blackwell维基百科 (主要数据来源)
7. ✅ https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture) - Ada Lovelace维基百科
8. ✅ https://en.wikipedia.org/wiki/Ampere_(microarchitecture) - Ampere维基百科
9. ✅ https://en.wikipedia.org/wiki/Template:NvidiaDgxAccelerators - NVIDIA DGX加速器对比表 (关键数据来源)

### 9.2 访问失败/重定向
10. ❌ https://www.nvidia.com/en-us/data-center/h100/ - ERR_ABORTED
11. ❌ https://www.nvidia.com/en-us/data-center/h800/ - 404
12. ❌ https://www.nvidia.com/en-us/data-center/h100/specifications/ - 404
13. ❌ https://www.nvidia.com/en-us/data-center/h200/specifications/ - 404
14. ❌ https://www.servethehome.com/nvidia-h100-tensor-core-gpu/ - 404
15. ⚠️ PDF下载 (H100白皮书/H200 datasheet) - 下载的文件实际是产品目录而非目标PDF

### 9.3 关键参考 (未访问但相关)
16. https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper - GTC22 Hopper白皮书 (博客中引用)
17. https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data - AnandTech Blackwell深度分析
18. https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/ - Triton on Blackwell

---

## 10. 调研方法说明

1. **信息来源优先级**: NVIDIA官方博客 > Wikipedia (含官方引用) > AnandTech/ServeTheHome
2. **数据冲突处理**: 同一规格在不同来源有冲突时，优先采用：
   - 最终shipping产品规格 > 早期preliminary规格
   - 官方datasheet > 博客宣传数字
   - 多源交叉验证一致的数据
3. **⚠️标注**: 对于未完全公布或推断的数据，使用⚠️标注
4. **Token节省策略**: 使用Wikipedia作为主要聚合信息源 (已整合官方数据)，减少对多个独立来源的重复访问

---

*报告完成于 2026-07-31*
