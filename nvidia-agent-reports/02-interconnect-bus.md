# NVIDIA 互联总线与片间/系统间互联技术规格深度调研报告

> 调研日期：2026-07-31
> 调研目标：为专业研究人员提供精确、可溯源的NVIDIA互联技术规格报告

---

## 1. NVLink 各代规格

### 1.1 NVLink 代际演进总览

| 代际 | 发布年份 | 配套架构 | 每链路带宽 (双向) | 每GPU链路数 | 每GPU总带宽 (双向) | 信号技术 | 主要应用 |
|------|----------|----------|-------------------|-------------|---------------------|----------|----------|
| NVLink 1.0 | 2016 | Pascal (GP100) | 20 GB/s + 20 GB/s | 4 sub-links | 160 GB/s | NRZ 20 GT/s | Tesla P100, POWER8+ |
| NVLink 2.0 | 2017 | Volta (GV100) | 25 GB/s + 25 GB/s | 6 sub-links | 300 GB/s | NRZ 25 GT/s | Tesla V100, POWER9 |
| NVLink 3.0 | 2020 | Ampere (GA100) | 25 GB/s + 25 GB/s | 12 sub-links | 600 GB/s | NRZ 50 GT/s | A100 |
| NVLink 4.0 | 2022 | Hopper (H100) | 25 GB/s + 25 GB/s | 18 sub-links | 900 GB/s | PAM4 50 GT/s | H100, Grace CPU |
| NVLink 5.0 | 2024 | Blackwell (B200) | 50 GB/s + 50 GB/s | 18 sub-links | 1,800 GB/s | PAM4 100 GT/s | B200, Grace CPU |
| NVLink 6.0 | 2026 | Rubin (R200) | 50 GB/s + 50 GB/s | 36 sub-links | 3,600 GB/s | PAM4 100 GT/s | Rubin GPU, Vera CPU |

**来源**: [NVIDIA NVLink官方产品页](https://www.nvidia.com/en-us/data-center/nvlink/), [Wikipedia NVLink](https://en.wikipedia.org/wiki/NVLink)

### 1.2 NVLink 物理层详细规格

| 参数 | NVLink 1.0 | NVLink 2.0 | NVLink 3.0 | NVLink 4.0 | NVLink 5.0 | NVLink 6.0 |
|------|------------|------------|------------|------------|------------|------------|
| 每差分对速率 | 20 Gbit/s | 25 Gbit/s | 50 Gbit/s | 50 Gbit/s | 100 Gbit/s | 100 Gbit/s |
| 调制方式 | NRZ | NRZ | NRZ | PAM4 | PAM4 | PAM4 |
| 每sub-link差分对数 | 8+8 | 8+8 | 4+4 | 4+4 | 4+4 | 4+4 |
| 每sub-link单向带宽 | 20 GB/s | 25 GB/s | 25 GB/s | 25 GB/s | 50 GB/s | 50 GB/s |
| 每sub-link双向带宽 | 40 GB/s | 50 GB/s | 50 GB/s | 50 GB/s | 100 GB/s | 100 GB/s |
| 每GPU sub-link数 | 4 | 6 | 12 | 18 | 18 | 36 |
| 每GPU总双向带宽 | 160 GB/s | 300 GB/s | 600 GB/s | 900 GB/s | 1,800 GB/s | 3,600 GB/s |

**关键架构演进**:
- **NVLink 1.0→2.0**: 信号速率从20 GT/s提升到25 GT/s，sub-link数从4增加到6
- **NVLink 2.0→3.0**: 信号速率从25 GT/s提升到50 GT/s，但每sub-link差分对数从8减少到4（保持相同带宽），sub-link数翻倍到12
- **NVLink 3.0→4.0**: 引入PAM4调制，保持50 GT/s但提升频谱效率，sub-link数增加到18
- **NVLink 4.0→5.0**: 信号速率翻倍到100 GT/s，每链路带宽翻倍到50 GB/s
- **NVLink 5.0→6.0**: 链路数翻倍到36，总带宽翻倍到3.6 TB/s

**来源**: [Wikipedia NVLink - Performance表](https://en.wikipedia.org/wiki/NVLink), [NVIDIA Hopper架构页](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)

### 1.3 NVLink 链路结构

NVLink采用全双工sub-link结构：
- **NVLink 1.0/2.0**: 每个sub-link = 8个差分对（发送）+ 8个差分对（接收），形成16对全双工
- **NVLink 3.0+**: 每个sub-link = 4个差分对（发送）+ 4个差分对（接收），形成8对全双工
- 两个同向sub-link组成一个"link"（NVLink 1.0/2.0定义）

### 1.4 NVLink 拓扑支持

| 拓扑类型 | 支持代际 | 描述 |
|----------|----------|------|
| 直连Mesh | 1.0-6.0 | 少量GPU间直接全互联 |
| Ring | 1.0-6.0 | GPU环形连接 |
| Crossbar via NVSwitch | 2.0-6.0 | 通过NVSwitch实现全互联 |
| 大规模NVLink域 | 4.0-6.0 | 通过NVLink Switch System实现576+ GPU全互联 |

---

## 2. NVSwitch 规格

### 2.1 NVSwitch 代际规格

| 参数 | NVSwitch Gen1 (Volta) | NVSwitch Gen2 (Ampere) | NVSwitch Gen3 (Hopper) | NVLink 5 Switch (Blackwell) | NVLink 6 Switch (Rubin) |
|------|----------------------|----------------------|----------------------|---------------------------|------------------------|
| 发布年份 | 2017 | 2020 | 2022 | 2024 | 2026 |
| 配套架构 | Volta | Ampere | Hopper | Blackwell | Rubin |
| NVLink代际 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 |
| 端口数 | 8 | 8 | 8 | 8 (可扩展至72 GPU域) | 8 (可扩展至72 GPU域) |
| 每端口带宽 | 50 GB/s (双向) | 50 GB/s (双向) | 50 GB/s (双向) | 100 GB/s (双向) | 100 GB/s (双向) |
| 总聚合带宽 | 7.2 TB/s | 7.2 TB/s | 7.2 TB/s | 130 TB/s (NVL72) | 260 TB/s (NVL72) |
| SHARP支持 | 否 | 是 (v2) | 是 (v3) | 是 (v4, FP8) | 是 (v4, FP8) |
| 每芯片NVLink数 | - | - | 18 | 18 | 36 |

**来源**: [NVIDIA NVLink官方产品页 - 规格表](https://www.nvidia.com/en-us/data-center/nvlink/), [Wikipedia NVLink - NVSwitch for Hopper](https://en.wikipedia.org/wiki/NVLink)

### 2.2 NVSwitch Gen3 (Hopper) 详细规格

| 参数 | 规格 |
|------|------|
| 芯片名称 | NVSwitch for Hopper |
| NVLink版本 | 4.0 |
| 端口数 | 8 (连接8个GPU) |
| 每端口带宽 | 900 GB/s (双向) |
| 每芯片NVLink数 | 18 |
| 每差分对速率 | 106.25 GT/s |
| 每sub-link差分对数 | 9+9 |
| 每sub-link带宽 | 450 Gbit/s |
| 总聚合带宽 | 7,200 GB/s (双向) |
| 内部结构 | 全连接64端口交换 |
| SHARP引擎 | 支持 (in-network reduction) |

**来源**: [Wikipedia NVSwitch for Hopper表](https://en.wikipedia.org/wiki/NVLink), [Hot Chips 34 - NVIDIA NVLink4 NVSwitch](https://www.servethehome.com/nvidia-nvlink4-nvswitch-at-hot-chips-34/)

### 2.3 NVSwitch 在系统中的角色

```
HGX H100 8-GPU板级拓扑:
┌─────────────────────────────────────────────────────────────┐
│                      HGX H100 Baseboard                      │
│                                                              │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                    │
│  │GPU 0 │──│GPU 1 │──│GPU 2 │──│GPU 3 │                    │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                    │
│     │         │         │         │                          │
│  ┌──┴─────────┴─────────┴─────────┴──┐                      │
│  │         NVSwitch Gen3 x4          │                      │
│  │    (每GPU 18 NVLink → 900 GB/s)   │                      │
│  └───────────────────────────────────┘                      │
│     │         │         │         │                          │
│  ┌──┴───┐  ┌──┴───┐  ┌──┴───┐  ┌──┴───┐                    │
│  │GPU 4 │──│GPU 5 │──│GPU 6 │──│GPU 7 │                    │
│  └──────┘  └──────┘  └──────┘  └──────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. NVLink-C2C (Chip-to-Chip) 接口

### 3.1 NVLink-C2C 规格

| 参数 | 规格 |
|------|------|
| 名称 | NVIDIA NVLink-C2C |
| 类型 | 芯片级互联 (die-to-die) |
| 带宽 | 900 GB/s (双向，Hopper时代) |
| 能效 | 比PCIe Gen 6 PHY高6x |
| 面积效率 | 比PCIe Gen 6 PHY高3.5x |
| 支持协议 | Arm AMBA CHI / CXL |
| 封装支持 | PCB级、MCM、硅中介层、晶圆级 |
| 原子操作 | 支持 (处理器与加速器间) |

**来源**: [NVIDIA NVLink-C2C官方产品页](https://www.nvidia.com/en-us/data-center/nvlink-c2c/)

### 3.2 NVLink-C2C 应用场景

| 产品 | 配置 | NVLink-C2C用途 |
|------|------|----------------|
| GH200 Superchip | 1 Grace CPU + 1 Hopper GPU | CPU-GPU互联 (900 GB/s) |
| GB200 Superchip | 1 Grace CPU + 2 Blackwell GPU | CPU-GPU互联 (900 GB/s) |
| GB300 NVL72 | 36 Grace CPU + 72 Blackwell Ultra GPU | 封装内CPU-GPU互联 |
| Grace CPU Superchip | 2x Grace CPU (144核) | CPU-CPU互联 (1 TB/s内存带宽) |
| Vera Rubin NVL72 | 2x Rubin GPU + 1 Vera CPU | 封装内CPU-GPU互联 |
| DGX Spark | GB10 Grace Blackwell Superchip | 封装内CPU-GPU互联 |

**来源**: [NVIDIA NVLink-C2C产品页](https://www.nvidia.com/en-us/data-center/nvlink-c2c/), [GB200 NVL72产品页](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)

### 3.3 Grace Blackwell Ultra 封装内NVLink-C2C

GB200/GB300 Superchip内部结构：
```
┌─────────────────────────────────────────────────────┐
│           GB200 Grace Blackwell Superchip            │
│                                                      │
│  ┌──────────────────┐      ┌──────────────────┐    │
│  │  Blackwell GPU 0  │◄────►│  Blackwell GPU 1  │    │
│  │   (B200 Die)     │NVLink│   (B200 Die)     │    │
│  │                  │ C2C  │                  │    │
│  └────────┬─────────┘      └─────────┬────────┘    │
│           │                          │              │
│           │ NVLink-C2C               │ NVLink-C2C   │
│           │ 900 GB/s                 │ 900 GB/s     │
│           │                          │              │
│           └────────────┬─────────────┘              │
│                        │                            │
│                 ┌──────┴──────┐                     │
│                 │  Grace CPU   │                     │
│                 │  (Arm v2)   │                     │
│                 └─────────────┘                     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 4. PCIe 规格

### 4.1 NVIDIA GPU支持的PCIe代际

| PCIe代际 | 每lane速率 | x16总带宽 (双向) | 配套GPU架构 | 发布年份 |
|----------|------------|------------------|-------------|----------|
| PCIe 3.0 | 8 GT/s | 32 GB/s | Pascal, Volta, Turing | 2016 |
| PCIe 4.0 | 16 GT/s | 64 GB/s | Volta (Xavier), Ampere, POWER9 | 2020 |
| PCIe 5.0 | 32 GT/s | 128 GB/s | Hopper (H100) | 2022 |
| PCIe 6.0 | 64 GT/s (PAM4) | 242 GB/s | Blackwell (B200) | 2024 |

**来源**: [Wikipedia NVLink - Performance表](https://en.wikipedia.org/wiki/NVLink), [NVIDIA H100产品页](https://www.nvidia.com/en-us/data-center/h100/)

### 4.2 PCIe与NVLink带宽对比

| GPU | PCIe带宽 | NVLink带宽 | NVLink/PCIe倍数 |
|-----|----------|------------|-----------------|
| P100 | 32 GB/s (PCIe 3.0) | 160 GB/s | 5x |
| V100 | 32 GB/s (PCIe 3.0) | 300 GB/s | 9.4x |
| A100 | 64 GB/s (PCIe 4.0) | 600 GB/s | 9.4x |
| H100 | 128 GB/s (PCIe 5.0) | 900 GB/s | 7x |
| B200 | 242 GB/s (PCIe 6.0) | 1,800 GB/s | 7.4x |
| R200 (Rubin) | 242 GB/s (PCIe 6.0) | 3,600 GB/s | 14.9x |

### 4.3 CXL支持情况

| CXL版本 | NVIDIA支持状态 | 备注 |
|---------|---------------|------|
| CXL 1.1 | 有限支持 | 通过NVLink-C2C兼容 |
| CXL 2.0 | 通过NVLink-C2C支持 | NVLink-C2C支持CXL协议 |
| CXL 3.0 | 未明确 | NVLink-C2C设计可扩展 |

**注**: NVIDIA主要通过NVLink-C2C提供CXL协议兼容性，而非原生CXL接口。NVLink-C2C支持Arm AMBA CHI或CXL行业标准协议。

**来源**: [NVIDIA NVLink-C2C产品页](https://www.nvidia.com/en-us/data-center/nvlink-c2c/)

---

## 5. 系统级互联拓扑

### 5.1 HGX H100 8-GPU板级拓扑

```
NVIDIA HGX H100 8-GPU 系统拓扑:
┌─────────────────────────────────────────────────────────────────────┐
│                        HGX H100 Baseboard                           │
│                                                                     │
│   ┌─────────┐                                                       │
│   │GPU 0    │◄════════════════════════════════════►│GPU 1    │     │
│   │H100 SXM │  NVLink 4.0 (通过NVSwitch)           │H100 SXM │     │
│   │900 GB/s │                                      │900 GB/s │     │
│   └────┬────┘                                      └────┬────┘     │
│        │                                                │           │
│   ┌────┴────┐                                      ┌────┴────┐     │
│   │GPU 2    │◄════════════════════════════════════►│GPU 3    │     │
│   │H100 SXM │                                      │H100 SXM │     │
│   │900 GB/s │                                      │900 GB/s │     │
│   └────┬────┘                                      └────┬────┘     │
│        │                                                │           │
│   ┌────┴────────────────────────────────────────────────┴────┐     │
│   │              NVSwitch Gen3 × 4 芯片                      │     │
│   │      (每NVSwitch 8端口, 全连接crossbar)                  │     │
│   │      (每GPU 18 NVLink → 900 GB/s 双向)                   │     │
│   └────┬────────────────────────────────────────────────┬────┘     │
│        │                                                │           │
│   ┌────┴────┐                                      ┌────┴────┐     │
│   │GPU 4    │◄════════════════════════════════════►│GPU 5    │     │
│   │H100 SXM │                                      │H100 SXM │     │
│   │900 GB/s │                                      │900 GB/s │     │
│   └────┬────┘                                      └────┬────┘     │
│        │                                                │           │
│   ┌────┴────┐                                      ┌────┴────┐     │
│   │GPU 6    │◄════════════════════════════════════►│GPU 7    │     │
│   │H100 SXM │                                      │H100 SXM │     │
│   │900 GB/s │                                      │900 GB/s │     │
│   └─────────┘                                      └─────────┘     │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  CPU (x86) ──PCIe Gen5──► GPU集群                           │   │
│   │  (Host CPU通过PCIe Gen5 x16 = 128 GB/s 访问GPU)              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**HGX H100 规格**:
- 8x H100 SXM GPU
- 每GPU NVLink带宽: 900 GB/s (双向)
- 板级总NVLink带宽: 7.2 TB/s
- 4x NVSwitch Gen3芯片
- PCIe Gen5 x16 (CPU到GPU): 128 GB/s

**来源**: [NVIDIA H100产品页](https://www.nvidia.com/en-us/data-center/h100/), [NVIDIA Hopper架构页](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)

### 5.2 DGX H100 系统级NVLink拓扑

```
NVIDIA DGX H100 系统拓扑:
┌─────────────────────────────────────────────────────────────────────┐
│                         DGX H100 Server                             │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────┐              │
│  │   CPU Complex 0      │    │   CPU Complex 1      │              │
│  │  (Intel Xeon)        │    │  (Intel Xeon)        │              │
│  │  PCIe Gen5           │    │  PCIe Gen5           │              │
│  └──────────┬───────────┘    └───────────┬──────────┘              │
│             │                            │                          │
│  ┌──────────┴────────────────────────────┴──────────┐              │
│  │              NVLink Backplane                      │              │
│  │                                                   │              │
│  │  ┌──────────────────────────────────────────┐    │              │
│  │  │         HGX H100 Baseboard 0              │    │              │
│  │  │  GPU0─GPU1─GPU2─GPU3 (NVSwitch全互联)     │    │              │
│  │  │  GPU4─GPU5─GPU6─GPU7 (NVSwitch全互联)     │    │              │
│  │  │  每GPU: 900 GB/s NVLink 4.0               │    │              │
│  │  └──────────────────────────────────────────┘    │              │
│  │                                                   │              │
│  └───────────────────────────────────────────────────┘              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  NDR InfiniBand (400 Gb/s) × 8 ──► 跨节点互联               │   │
│  │  (Scale-out网络, 用于多DGX H100集群)                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**DGX H100 规格**:
- 8x H100 SXM5 GPU
- 总GPU内存: 640 GB HBM3
- 总GPU内存带宽: 26.8 TB/s
- NVLink带宽: 900 GB/s per GPU (双向)
- NDR InfiniBand: 8x 400 Gb/s
- CPU: 2x Intel Xeon (Sapphire Rapids)

**来源**: [NVIDIA H100产品页](https://www.nvidia.com/en-us/data-center/h100/)

### 5.3 GB200 NVL72 NVLink 5.0 拓扑 (72 GPU全互联)

```
NVIDIA GB200 NVL72 机架级拓扑:
┌─────────────────────────────────────────────────────────────────────┐
│                      GB200 NVL72 Rack                               │
│                      (液冷机架级架构)                                │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    NVLink Switch System                       │  │
│  │                   (5x NVLink 5 Switch芯片)                    │  │
│  │                                                              │  │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐      │  │
│  │   │NVSwitch │ │NVSwitch │ │NVSwitch │ ... │NVSwitch │      │  │
│  │   │  #1     │ │  #2     │ │  #3     │     │  #5     │      │  │
│  │   └────┬────┘ └────┬────┘ └────┬────┘     └────┬────┘      │  │
│  │        │           │           │               │            │  │
│  └────────┼───────────┼───────────┼───────────────┼────────────┘  │
│           │           │           │               │               │
│  ┌────────┴───────────┴───────────┴───────────────┴────────────┐  │
│  │                                                              │  │
│  │   9x GB200 Compute Tray (每Tray 2x Superchip)                │  │
│  │                                                              │  │
│  │   ┌──────────────┐  ┌──────────────┐       ┌──────────────┐ │  │
│  │   │ Superchip #1  │  │ Superchip #2  │  ...  │ Superchip #18 │ │  │
│  │   │ ┌────┬────┐ │  │ ┌────┬────┐ │       │ ┌────┬────┐ │ │  │
│  │   │ │GPU0│GPU1│ │  │ │GPU0│GPU1│ │       │ │GPU0│GPU1│ │ │  │
│  │   │ │B200│B200│ │  │ │B200│B200│ │       │ │B200│B200│ │ │  │
│  │   │ └──┬─┴─┬──┘ │  │ └──┬─┴─┬──┘ │       │ └──┬─┴─┬──┘ │ │  │
│  │   │    │C2C│    │  │    │C2C│    │       │    │C2C│    │ │  │
│  │   │  ┌─┴──┴─┐  │  │  ┌─┴──┴─┐  │       │  ┌─┴──┴─┐  │ │  │
│  │   │  │Grace │  │  │  │Grace │  │       │  │Grace │  │ │  │
│  │   │  │CPU   │  │  │  │CPU   │  │       │  │CPU   │  │ │  │
│  │   │  └──────┘  │  │  └──────┘  │       │  └──────┘  │ │  │
│  │   └──────────────┘  └──────────────┘       └──────────────┘ │  │
│  │                                                              │  │
│  │   总计: 36x Grace CPU + 72x Blackwell GPU                     │  │
│  │   每GPU NVLink 5.0: 1.8 TB/s (双向)                           │  │
│  │   机架总NVLink带宽: 130 TB/s                                  │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ConnectX-8 SuperNIC I/O Module                              │  │
│  │  (每GPU 800 Gb/s 网络连接)                                    │  │
│  │  Quantum-X800 InfiniBand / Spectrum-X Ethernet               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**GB200 NVL72 规格**:
- 配置: 36x Grace CPU + 72x Blackwell GPU
- NVLink代际: 5.0
- 每GPU NVLink带宽: 1.8 TB/s (双向)
- 机架总NVLink带宽: 130 TB/s
- GPU内存: 13.4 TB HBM3E (总)
- GPU内存带宽: 576 TB/s (总)
- CPU内存: 17 TB LPDDR5X
- 网络: ConnectX-8 SuperNIC (800 Gb/s per GPU)

**来源**: [GB200 NVL72产品页](https://www.nvidia.com/en-us/data-center/gb200-nvl72/), [NVIDIA NVLink官方产品页](https://www.nvidia.com/en-us/data-center/nvlink/)

### 5.4 GB300 NVL72 规格

| 参数 | GB300 NVL72 | GB200 NVL72 |
|------|-------------|-------------|
| GPU | 72x Blackwell Ultra | 72x Blackwell |
| CPU | 36x Grace (Arm) | 36x Grace (Arm) |
| NVLink代际 | 5.0 | 5.0 |
| 每GPU NVLink带宽 | 1.8 TB/s | 1.8 TB/s |
| 机架总NVLink带宽 | 130 TB/s | 130 TB/s |
| GPU内存 | 20 TB HBM3E | 13.4 TB HBM3E |
| GPU内存带宽 | 576 TB/s | 576 TB/s |
| NVFP4 Tensor Core | 1,440 PFLOPS | 720 PFLOPS |
| 相对H100性能 | 50x (AI工厂输出) | 30x (LLM推理) |

**来源**: [GB300 NVL72产品页](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)

### 5.5 Vera Rubin NVL72 (NVLink 6.0) 拓扑

```
NVIDIA Vera Rubin NVL72 机架级拓扑:
┌─────────────────────────────────────────────────────────────────────┐
│                    Vera Rubin NVL72 Rack                            │
│                     (NVLink 6.0, 2026)                              │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   NVLink 6 Switch System                      │  │
│  │                                                              │  │
│  │   72x Rubin GPU 全互联 (all-to-all)                          │  │
│  │   每GPU NVLink 6.0: 3.6 TB/s (双向)                           │  │
│  │   机架总NVLink带宽: 260 TB/s                                  │  │
│  │                                                              │  │
│  │   ┌─────────────────────────────────────────────────────┐   │  │
│  │   │         36x Vera Rubin Superchip                     │   │  │
│  │   │                                                     │   │  │
│  │   │  每Superchip: 2x Rubin GPU + 1x Vera CPU            │   │  │
│  │   │  GPU-GPU: NVLink-C2C (封装内)                        │   │  │
│  │   │  CPU-GPU: NVLink-C2C                                 │   │  │
│  │   │                                                     │   │  │
│  │   │  Rubin GPU: 288 GB HBM4, 22 TB/s 内存带宽            │   │  │
│  │   │  Vera CPU: 88x Olympus cores, 1.5TB LPDDR5X         │   │  │
│  │   │                                                     │   │  │
│  │   └─────────────────────────────────────────────────────┘   │  │
│  │                                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   关键新特性:                                                       │
│   - 控制平面弹性 (Control Plane Resilience)                         │
│   - 部分填充机架运行 (Partially Populated Rack)                     │
│   - 交换机托盘热插拔 (Hot-swapping of Switch Trays)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Vera Rubin NVL72 规格**:
- 配置: 36x Vera CPU + 72x Rubin GPU
- NVLink代际: 6.0
- 每GPU NVLink带宽: 3.6 TB/s (双向)
- 机架总NVLink带宽: 260 TB/s
- GPU内存: 2.3 TB HBM4 (总)
- GPU内存带宽: 176 TB/s (总)
- 每GPU: 288 GB HBM4, 22 TB/s
- AI计算: 3.6 exaFLOPS (整机架)

**来源**: [NVIDIA NVLink官方产品页](https://www.nvidia.com/en-us/data-center/nvlink/), [NVIDIA HGX平台页](https://www.nvidia.com/en-us/data-center/hgx/)

### 5.6 NVLink Switch System 规格

| 参数 | NVLink 4 Switch (Hopper) | NVLink 5 Switch (Blackwell) | NVLink 6 Switch (Rubin) |
|------|-------------------------|---------------------------|------------------------|
| GPU域大小 | 8 (单机) / 256 (多机) | 8 (单机) / 72 (NVL72) | 8 (单机) / 72 (NVL72) |
| GPU-to-GPU带宽 | 900 GB/s | 1,800 GB/s | 3,600 GB/s |
| 总聚合带宽 | 7.2 TB/s (8-GPU) / 57.6 TB/s (256-GPU) | 130 TB/s (NVL72) | 260 TB/s (NVL72) |
| SHARP支持 | v3 | v4 (FP8) | v4 (FP8) |
| 带宽效率 | 标准 | 4x (SHARP FP8) | 4x (SHARP FP8) |

**来源**: [NVIDIA NVLink官方产品页 - 规格表](https://www.nvidia.com/en-us/data-center/nvlink/)

---

## 6. 封装级互联

### 6.1 多芯片封装中的Die-to-Die互联

| 产品 | 封装技术 | Die-to-Die互联 | 带宽 |
|------|----------|----------------|------|
| B200 (Blackwell) | 2-die MCM | 10 TB/s chip-to-chip | 10 TB/s |
| GB200 Superchip | 3-die (2 GPU + 1 CPU) | NVLink-C2C | 900 GB/s (CPU-GPU) |
| GB300 Superchip | 3-die (2 GPU + 1 CPU) | NVLink-C2C | 900 GB/s (CPU-GPU) |
| Grace CPU Superchip | 2-die (2 CPU) | NVLink-C2C | 1 TB/s (内存带宽) |
| Vera Rubin Superchip | 3-die (2 GPU + 1 CPU) | NVLink-C2C | 900+ GB/s (CPU-GPU) |

**来源**: [NVIDIA Blackwell架构页](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/), [NVIDIA NVLink-C2C产品页](https://www.nvidia.com/en-us/data-center/nvlink-c2c/)

### 6.2 Blackwell 封装内互联

Blackwell B200 GPU采用双die MCM封装：
```
┌─────────────────────────────────────────────────────┐
│              NVIDIA B200 GPU Package                 │
│                                                      │
│  ┌────────────────────┐  ┌────────────────────┐    │
│  │   Blackwell Die 0   │  │   Blackwell Die 1   │    │
│  │   (Reticle-limited) │  │   (Reticle-limited) │    │
│  │                     │  │                     │    │
│  │   104B transistors  │  │   104B transistors  │    │
│  │   (每die)           │  │   (每die)           │    │
│  └──────────┬──────────┘  └──────────┬──────────┘    │
│             │                        │               │
│             └────────┬───────────────┘               │
│                      │                               │
│              ┌───────┴───────┐                       │
│              │ 10 TB/s       │                       │
│              │ chip-to-chip  │                       │
│              │ interconnect  │                       │
│              └───────────────┘                       │
│                                                      │
│  总晶体管: 2080亿 (2x 1040亿)                        │
│  工艺: TSMC 4NP                                      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**来源**: [NVIDIA Blackwell架构页](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

### 6.3 Grace Blackwell Ultra 封装内NVLink-C2C带宽

| 参数 | 规格 |
|------|------|
| 封装内互联技术 | NVLink-C2C |
| CPU-GPU带宽 | 900 GB/s (双向) |
| GPU-GPU带宽 | 通过NVLink 5.0 (1.8 TB/s per GPU) |
| 能效优势 | 比PCIe Gen6 PHY高6x |
| 面积效率优势 | 比PCIe Gen6 PHY高3.5x |
| 支持协议 | AMBA CHI / CXL |

**来源**: [NVIDIA NVLink-C2C产品页](https://www.nvidia.com/en-us/data-center/nvlink-c2c/)

---

## 7. NVLink Fusion

### 7.1 NVLink Fusion 概述

NVLink Fusion是NVIDIA开放的半定制互联技术许可，允许第三方ASIC/CPU通过NVLink技术集成到NVIDIA基础设施中。

| 参数 | 规格 |
|------|------|
| 技术 | NVLink Fusion |
| NVLink代际 | 6.0 |
| 每XPU带宽 | 3.6 TB/s (全互联) |
| 域大小 | 72 (NVL72) / 可扩展至1,152 |
| 总带宽 | 260 TB/s (NVL72) |
| SHARP支持 | FP8 |
| 许可伙伴 | ARM, SiFive, AWS (Trainium4), Intel, Fujitsu |

**来源**: [NVIDIA NVLink Fusion产品页](https://www.nvidia.com/en-us/data-center/nvlink-fusion/)

### 7.2 NVLink Fusion 生态系统

| 伙伴类型 | 合作伙伴 | 用途 |
|----------|----------|------|
| CPU伙伴 | ARM, Intel, Fujitsu, SiFive | 定制CPU集成 |
| 定制硅伙伴 | AWS (Trainium4) | Trainium4加速器 |
| IP供应商 | ARM, SiFive | NVLink IP许可 |

---

## 8. 互联带宽演进趋势

### 8.1 历代NVLink带宽演进

```
NVLink 每GPU带宽演进 (双向, GB/s):
                                                            
3600 ┤                                              ┌─── NVLink 6.0 (Rubin)
     │                                              │
3000 ┤                                              │
     │                                              │
2400 ┤                                              │
     │                                              │
1800 ┤                              ┌───────────────┤─── NVLink 5.0 (Blackwell)
     │                              │               │
1200 ┤                              │               │
     │                              │               │
 900 ┤              ┌───────────────┤─── NVLink 4.0 (Hopper)
     │              │               │
 600 ┤              │               │
     │              │               │
 300 ┤              │               │
     │              │               │
 160 ┤──────────────┤─── NVLink 1.0 (Pascal)
     │              │
   0 ┼──────────────┴───────────────┴───────────────┴───
     2016           2020           2022           2024    2026
```

### 8.2 NVLink与PCIe带宽对比趋势

| 年份 | NVLink带宽/GPU | PCIe带宽/x16 | NVLink/PCIe倍数 |
|------|----------------|--------------|-----------------|
| 2016 | 160 GB/s | 32 GB/s (Gen3) | 5x |
| 2017 | 300 GB/s | 32 GB/s (Gen3) | 9.4x |
| 2020 | 600 GB/s | 64 GB/s (Gen4) | 9.4x |
| 2022 | 900 GB/s | 128 GB/s (Gen5) | 7x |
| 2024 | 1,800 GB/s | 242 GB/s (Gen6) | 7.4x |
| 2026 | 3,600 GB/s | 242 GB/s (Gen6) | 14.9x |

---

## 9. 关键发现总结

### 9.1 技术演进规律

1. **信号速率**: NVLink信号速率从20 GT/s (NVLink 1.0) 提升到100 GT/s (NVLink 5.0/6.0)，5年翻倍一次
2. **调制方式**: 从NRZ (NVLink 1.0-3.0) 演进到PAM4 (NVLink 4.0+)，提升频谱效率
3. **链路数扩展**: 从4条 (NVLink 1.0) 扩展到36条 (NVLink 6.0)
4. **带宽增长**: 每GPU带宽从160 GB/s增长到3.6 TB/s，6年增长22.5x

### 9.2 架构设计洞察

1. **Scale-up优先**: NVLink持续扩大机架级GPU域，从8 GPU (HGX) → 72 GPU (NVL72) → 可扩展至576 GPU
2. **NVSwitch角色**: 从板级crossbar交换 (Gen1-3) 演进到机架级全互联交换 (Gen5-6)
3. **SHARP引擎**: 在网计算能力从InfiniBand下移到NVLink，支持FP8 reduction
4. **封装内互联**: NVLink-C2C提供比PCIe高6x能效和3.5x面积效率

### 9.3 系统级影响

1. **全互联拓扑**: NVL72实现72 GPU all-to-all全互联，任意GPU对通信带宽一致
2. **内存语义**: NVLink-C2C支持缓存一致性，CPU和GPU共享统一内存空间
3. **机架即计算机**: NVLink Switch System将整个机架转变为单一巨型GPU

---

## 10. 访问过的URL列表

### 10.1 NVIDIA官方页面

| URL | 状态 | 内容 |
|-----|------|------|
| https://www.nvidia.com/en-us/data-center/nvlink/ | ✅ 成功 | NVLink官方产品页，规格表 |
| https://www.nvidia.com/en-us/data-center/nvlink-c2c/ | ✅ 成功 | NVLink-C2C产品页 |
| https://www.nvidia.com/en-us/data-center/nvlink-fusion/ | ✅ 成功 | NVLink Fusion产品页 |
| https://www.nvidia.com/en-us/data-center/gb200-nvl72/ | ✅ 成功 | GB200 NVL72规格 |
| https://www.nvidia.com/en-us/data-center/gb300-nvl72/ | ✅ 成功 | GB300 NVL72规格 |
| https://www.nvidia.com/en-us/data-center/h100/ | ✅ 成功 | H100 GPU规格 |
| https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/ | ✅ 成功 | Hopper架构页 |
| https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/ | ✅ 成功 | Blackwell架构页 |
| https://www.nvidia.com/en-us/data-center/hgx/ | ✅ 成功 | HGX平台规格表 |
| https://www.nvidia.com/en-us/data-center/dgx-platform/ | ✅ 成功 | DGX平台页 |
| https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/ | ✅ 成功 | Grace CPU Superchip |

### 10.2 第三方信息源

| URL | 状态 | 内容 |
|-----|------|------|
| https://en.wikipedia.org/wiki/NVLink | ✅ 成功 | NVLink Wikipedia，详细规格表 |
| https://www.servethehome.com/nvidia-nvlink4-nvswitch-at-hot-chips-34/ | ❌ 访问失败 | Hot Chips 34 NVSwitch论文 |
| https://www.servethehome.com/nvidia-gb200-nvl72/ | ❌ 404 | GB200 NVL72分析 |
| https://www.servethehome.com/nvidia-blackwell/ | ⚠️ 重定向 | 重定向到Wikipedia HBM页 |

### 10.3 技术文档

| URL | 状态 | 内容 |
|-----|------|------|
| https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703 | ✅ 可用 | Blackwell Datasheet |
| https://nvdam.widen.net/s/7hztspzswk/gpu-architecture-datasheet-vera-rubin-nvidia-us-5198950-web | ✅ 可用 | Vera Rubin Datasheet |
| https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c | ✅ 可用 | H100白皮书 |

---

## 11. 数据来源说明

本报告所有规格数据均来自以下类型的来源：

1. **NVIDIA官方产品页**: 主要规格数据来源
2. **NVIDIA官方规格表**: NVLink/NVSwitch代际规格
3. **Wikipedia NVLink**: 历史规格和对比数据
4. **NVIDIA技术博客**: 架构细节和路线图
5. **Hot Chips论文**: NVSwitch技术细节 (通过Wikipedia引用)

**数据可信度**:
- ✅ 高可信度: NVIDIA官方产品页和规格表
- ✅ 中高可信度: Wikipedia (引用官方来源)
- ⚠️ 初步规格: Rubin/Vera Rubin部分数据标注为"Preliminary"

---

*报告完成时间: 2026-07-31*
*调研方法: 通过Playwright浏览器工具访问NVIDIA官方页面和第三方信息源*
