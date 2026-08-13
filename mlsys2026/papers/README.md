# MLSys 2026 — 19 篇深度解读合集

本目录是从 `iCloud Drive/paper/mlsys2026/` 中挑选的 **19 篇 MLSys 2026 重点论文**，每篇配套：

- 📄 **原文 PDF**（按论文标题命名，便于阅读）
- 📝 **中文深度分析报告**（~500 行 markdown，含 TL;DR / 背景 / 方法 / 实现 / 评测 / 启示 / 局限 / 术语 / 页码索引 / 一句话点评）

合计：19 PDF + 19 报告 = 38 文件 / ~68 MB / ~9400 行中文分析。

---

## 📚 论文清单（按主题分组）

### 🔧 ML 编译器 / 内核（4 篇）

| 论文 | 单位 | 报告 | PDF |
|---|---|---|---|
| **Dataflow Is All You Need** | SambaNova | [报告](ML_Compilers_Kernels/DATAFLOW_IS_ALL_YOU_NEED_分析报告.md) | [PDF](ML_Compilers_Kernels/DATAFLOW_IS_ALL_YOU_NEED.pdf) |
| **FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling** | Tri Dao 团队 | [报告](ML_Compilers_Kernels/FLASHATTENTION-4_分析报告.md) | [PDF](ML_Compilers_Kernels/FLASHATTENTION-4.pdf) |
| **HipKittens: Fast and Furious AMD Kernels** | Stanford Hazy Research | [报告](ML_Compilers_Kernels/HIPKITTENS_分析报告.md) | [PDF](ML_Compilers_Kernels/HIPKITTENS.pdf) |
| **ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels** | Stanford Hazy Research | [报告](ML_Compilers_Kernels/PARALLELKITTENS_分析报告.md) | [PDF](ML_Compilers_Kernels/PARALLELKITTENS.pdf) |

### 🏋️ LLM 训练与微调（2 篇）

| 论文 | 单位 | 报告 | PDF |
|---|---|---|---|
| **AXLearn: Modular, Hardware-Agnostic Large Model Training** | Apple | [报告](LLM_Training_Fine-tuning/AXLEARN_分析报告.md) | [PDF](LLM_Training_Fine-tuning/AXLEARN.pdf) |
| **veScale-FSDP: Flexible and High-Performance FSDP at Scale** | ByteDance Seed | [报告](LLM_Training_Fine-tuning/VESCALE-FSDP_分析报告.md) | [PDF](LLM_Training_Fine-tuning/VESCALE-FSDP.pdf) |

### 🚀 LLM 推理与服务（5 篇）

| 论文 | 单位 | 报告 | PDF |
|---|---|---|---|
| **TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference** | Microsoft 等 | [报告](LLM_Inference_Serving/TOKENWEAVE_分析报告.md) | [PDF](LLM_Inference_Serving/TOKENWEAVE.pdf) |
| **SAKURAONE: An Open Ethernet–Based AI HPC System and Its Observed Workload Dynamics** | KONISHI 等 | [报告](LLM_Inference_Serving/SAKURAONE_分析报告.md) | [PDF](LLM_Inference_Serving/SAKURAONE.pdf) |
| **Beyond the Buzz: A Pragmatic Take on Inference Disaggregation** ⭐NEW | NVIDIA | [报告](LLM_Inference_Serving/BEYOND_THE_BUZZ_分析报告.md) | [PDF](LLM_Inference_Serving/BEYOND_THE_BUZZ.pdf) |
| **Demystifying the Mixture of Experts Serving Tax** ⭐NEW | Patel 等 | [报告](LLM_Inference_Serving/MOE_SERVING_TAX_分析报告.md) | [PDF](LLM_Inference_Serving/MOE_SERVING_TAX.pdf) |
| **Speculative Decoding: Performance or Illusion?** ⭐NEW | UC Berkeley | [报告](LLM_Inference_Serving/SPECULATIVE_DECODING_PERFORMANCE_OR_ILLUSION_分析报告.md) | [PDF](LLM_Inference_Serving/SPECULATIVE_DECODING_PERFORMANCE_OR_ILLUSION.pdf) |

### 🏗️ 硬件与加速器（3 篇）

| 论文 | 单位 | 报告 | PDF |
|---|---|---|---|
| **A Lightweight High-Throughput Collective-Capable NoC for Large-Scale ML Accelerators** | Colagrande 等 | [报告](Hardware_Accelerators/NOC_COLLECTIVE_CAPABLE_分析报告.md) | [PDF](Hardware_Accelerators/NOC_COLLECTIVE_CAPABLE.pdf) |
| **SHIP: SRAM-Based Huge Inference Pipelines for Fast LLM Serving** | NVIDIA / Groq 路线 | [报告](Hardware_Accelerators/SHIP_分析报告.md) | [PDF](Hardware_Accelerators/SHIP.pdf) |
| **SuperInfer: SLO-Aware Rotary Scheduling and Memory Management for LLM Inference on Superchips** | Yu 等 | [报告](Hardware_Accelerators/SUPERINFER_分析报告.md) | [PDF](Hardware_Accelerators/SUPERINFER.pdf) |

### 🌐 分布式与超节点总线（1 篇）

| 论文 | 单位 | 报告 | PDF |
|---|---|---|---|
| **fabric-lib: RDMA Point-to-Point Communication for LLM Systems** | Licker 等 | [报告](Distributed_Federated_ML/FABRIC-LIB_分析报告.md) | [PDF](Distributed_Federated_ML/FABRIC-LIB.pdf) |

### 📱 端侧推理（1 篇）

| 论文 | 单位 | 报告 | PDF |
|---|---|---|---|
| **ExecuTorch — A Unified PyTorch Solution to Run ML Models On-Device** | Meta | [报告](Edge_Mobile_Embedded/EXECUTORCH_分析报告.md) | [PDF](Edge_Mobile_Embedded/EXECUTORCH.pdf) |

### 🔍 数据 / 存储 / 检索（3 篇）

| 论文 | 单位 | 报告 | PDF |
|---|---|---|---|
| **LEANN: A Low-Storage Overhead Vector Index** | Wang 等 | [报告](Data_Storage_Retrieval/LEANN_分析报告.md) | [PDF](Data_Storage_Retrieval/LEANN.pdf) |
| **GriNNder: Breaking the Memory Capacity Wall in Full-Graph GNN Training with Storage Offloading** ⭐NEW | Song 等 | [报告](Data_Storage_Retrieval/GRINNDER_分析报告.md) | [PDF](Data_Storage_Retrieval/GRINNDER.pdf) |
| **SkipKV: Selective Skipping of KV Generation and Storage for Efficient Inference with Large Reasoning Models** ⭐NEW | Tian 等 | [报告](Data_Storage_Retrieval/SKIPKV_分析报告.md) | [PDF](Data_Storage_Retrieval/SKIPKV.pdf) |

---

## 🔑 19 篇论文核心主线

### 主线 ① 集合通信硬件下沉（fabric → switch → on-chip 三层光谱）

| 层级 | 论文 | 关键贡献 |
|---|---|---|
| **fabric 层** | **SAKURAONE** | 开放以太网 + RoCEv2 + DCQCN 替代闭源 IB 生态 |
| **fabric 层** | **fabric-lib** | RDMA P2P 抽象（IMMCOUNTER 解决完成语义） |
| **片上** | **Lightweight Collective NoC** | multi-address mask + DCA，16.5% router 面积换 2.9× 加速 |
| **多卡** | **ParallelKittens** | GPU 上 collective tile primitives DSL |
| **跨卡** | **TokenWeave** | NVLink4 SymmetricMemory + Multimem PTX 通信–计算重叠 |
| **MoE** | **MoE Serving Tax** ⭐ | 量化 6 类税：AllToAll 4.6× 网络流量、p95 2.7× tail |

### 主线 ② 「同步税」与 dataflow 哲学

- **Dataflow Is All You Need**（SambaNova）—— 用专用硬件解决 decode 的同步税
- **SHIP**（Groq LPU TSP）—— 同步执行 + 静态调度 + SRAM 主导
- **TokenWeave**（GPU）—— 在 NV 体系上模拟 dataflow 思想
- **FlashAttention-4** —— Hopper 非对称硬件下的 warp specialization 极致工程
- **Speculative Decoding: Illusion?** ⭐ —— 反证：SD 在 GPU 上的"同步税"被低估，bs=1→128 EAGLE 加速从 1.73× 跌到 1.21×

### 主线 ③ 存储层级革命（HBM-bound → 多级存储联动）

- **SHIP**：SRAM 69.7 GB / DDR 72 TB（Groq LPU），weight stationary
- **SuperInfer**：GH200 NVLink-C2C 900 GB/s，把 480GB CPU DRAM 当 KV cache 二级存储
- **LEANN**：向量索引 50× 存储压缩（HNSW 188GB → 4GB），用计算换存储
- **GriNNder** ⭐：full-graph GNN 训练用 NVMe SSD，单卡 A5000 达到 16-GPU IB 集群吞吐
- **SkipKV** ⭐：reasoning 模型 KV cache 选择性跳过，存储+算力双省，accuracy +26.7% vs R-KV
- **Dataflow / SambaNova**：520MB SRAM + 64GB HBM + 1.5TB DDR

### 主线 ④ Superchip / 超节点对系统设计的范式冲击

- **SuperInfer**：三个 Insight + RotaSched + DuplexKV，证明 superchip 不是 PCIe 的「更快版本」
- **fabric-lib**：disaggregated serving 让 P2P 取代 collective 成为一等公民
- **SAKURAONE**：单租户大集群网络栈观测，rail-optimized 拓扑
- **Beyond the Buzz** ⭐：NVIDIA 给 PD 分离泼冷水，给出"何时该用 / 何时反而是幻觉"的 6 条充分条件 + 5 反例

### 主线 ⑤ 训练框架两强对比

- **AXLearn**（Apple，JAX）vs **veScale-FSDP**（字节，PyTorch DTensor）
- 同一时间点，两家工业巨头在不同基座上走出迥异路径

### 主线 ⑥ 端侧 RAG 的两块基础设施

- **ExecuTorch**（端侧统一 runtime）+ **LEANN**（端侧低存储向量索引）

### 主线 ⑦ 批判性视角（Critical / Pragmatic）⭐NEW

MLSys 2026 罕见地出现两篇明确「反主流」的论文，配合 Demystifying MoE 的量化分析，构成完整的"工业 reality check"：

- **Beyond the Buzz** —— 给 PD 分离泼冷水（NVIDIA 视角）
- **Speculative Decoding: Performance or Illusion?** —— 给 SD 泼冷水（UC Berkeley 视角，bs=1 prototype 的 production illusion）
- **Demystifying MoE Serving Tax** —— 把 MoE 推理开销做 6 类拆解，破除"sparse MoE 一定比 dense 便宜"的简单叙事

---

## 📂 目录结构

```
mlsys2026_paper_session/
├── README.md                        # 本文件
├── ML_Compilers_Kernels/           # 4 篇 (kernel/编译)
├── LLM_Training_Fine-tuning/       # 2 篇 (训练框架)
├── LLM_Inference_Serving/          # 5 篇 (推理服务)
├── Hardware_Accelerators/          # 3 篇 (硬件/体系结构)
├── Distributed_Federated_ML/       # 1 篇 (RDMA P2P)
├── Edge_Mobile_Embedded/           # 1 篇 (端侧)
└── Data_Storage_Retrieval/         # 3 篇 (向量索引 + GNN + KV)
```

每个子目录下放：
- `<论文简称>.pdf` — 原文
- `<论文简称>_分析报告.md` — 中文深度解读

---

## 📊 统计

| 项目 | 数值 |
|---|---|
| 论文数 | 19 |
| 报告总行数 | ~9400 行 |
| 报告总大小 | ~615 KB markdown |
| PDF 总大小 | ~68 MB |
| 覆盖类别 | 7 / 13 |

---

*最近更新：2026-06-18*
*原始论文 + 索引：`/Users/backyes/Library/Mobile Documents/com~apple~CloudDocs/paper/mlsys2026/`*
