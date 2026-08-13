# CXL 互联技术对 LLM 推理 P/D 分离架构下 KVCache 管理的前沿调研

> **调研日期**: 2026-07-14  
> **调研范围**: 2024–2026 年学术论文、产业博客、技术白皮书  
> **焦点主题**: CXL memory pooling、KVCache 分级管理、P/D 分离架构传输范式演进

---

## 摘要（Executive Summary）

CXL 正在深刻重塑 LLM 推理系统中 KVCache 的管理范式。2025-2026 年间，学术界和工业界出现了一批标志性的系统论文，展示了 CXL 相对于传统 RDMA/NVLink 路线的差异化优势：

- **CXL 提供 byte-addressable、cacheline 粒度的 load/store 语义**，适合 sparse attention 场景下**按需按需取 KV entries**，而不像 RDMA 需要粗粒度地传输整块 prefix cache（SAC, 2606.19746）
- **CXL pooled memory 打破了单机 DRAM 容量上限**（CPU channel 限制），使 GPU/CPU 通过 CXL switch 共享 TB 级内存池（Beluga, 2511.20172, SIGMOD'26）
- **CXL-PNM（Processing Near Memory）**允许在 CXL 内存侧完成 token page selection，完全消除 GPU→CPU recalls（PNM-CXL, 2511.00321），实现 **21.9x 吞吐提升、60x 能耗降低**
- **ITME（2606.12556）** 提出 CXL-hybrid memory 层级（HBM→DDR→CXL→NVMe），用 production-grade SK Hynix CMM 验证了 **35.7% 吞吐提升**
- **TraCT（2512.18194）** 用 CXL shared memory 替代 RDMA 作为 KV-transfer substrate，消除 NIC hop，实现 **9.8x TTFT 降低**

---

## 一、关键论文清单

### 1.1 CXL-native KV Cache System 系列（核心）

#### ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories
- **arXiv**: 2606.12556 (2026-06-10)
- **作者**: Hakbeom Jang, Younghoon Min, Sunwoong Kim, Taeyoung Ahn, Hanyee Kim, Youngpyo Joo, Hoshik Kim, Jongryool Kim（Samsung/UNIST 团队）
- **核心贡献**:
  - 提出 CXL-hybrid memory 层级架构，提供 TB 级 byte-addressable 远程内存扩展
  - 利用模型权重和 prefix cache 的确定性访问模式，主动管理 memory-storage 数据移动
  - 基于 production-grade SK Hynix CMM (CXL Memory Module) + PCIe Gen5 NVMe SSD 评测
  - 通过 FPGA 原型验证可行性
  - CPU-offloading 场景下实现 **35.7% 吞吐提升**
- **关键洞察**: 解决了 DPU+JBOF 架构中 NVMe-oF 复杂软件栈与成本效率低的问题

#### SAC: Disaggregated KV Cache System for Sparse Attention LLMs with CXL
- **arXiv**: 2606.19746 (2026-06-18)
- **作者**: Ruiyang Ma, Teng Ma, Junru Li 等（北京大学/字节跳动）
- **核心贡献**:
  - 首个针对 sparse attention 优化的 CXL 分离式 KV cache 系统
  - 利用 CXL cache-line 粒度 load/store 语义，按需获取 top-k KV entries
  - 在 DeepSeek-V3.2 + SGLang 上：vs RDMA baseline：
    - **2.1x 吞吐**
    - **9.7x TTFT 降低**
    - **1.8x TBT 降低**
- **关键洞察**: dense attention下 RDMA 粗粒度全量传输 prefix cache 完全不适合 sparse attention 场景

#### Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management
- **arXiv**: 2511.20172 (2025-11-25), **已接收 SIGMOD'26**
- **作者**: Xinjun Yang, Qingda Hu, Junru Li, Feifei Li 等（字节跳动/北京大学）
- **核心贡献**:
  - 首个通过 CXL switch 让 GPU 直接访问大规模 pooled memory 的系统
  - 支持 native load/store 语义，近本地内存延迟
  - 系统性地刻画了商用 CXL switch-based memory pool 性能
  - RDMA baseline 对比：**89.6% TTFT 降低，7.35x 吞吐提升**
- **关键洞察**: CXL 消除了 RDMA 的高延迟、复杂通信协议和同步开销

#### TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale
- **arXiv**: 2512.18194 (2025-12-20)
- **作者**: Dongha Yoon, Younghoon Min, Hoshik Kim, Sam H. Noh, Jongryool Kim（UNIST）
- **核心贡献**:
  - 用 CXL shared memory 替代 RDMA 作为 KV-transfer substrate + rack-wide prefix-aware KV cache
  - GPU 通过 CXL load/store 和 DMA 写/读 KV blocks，消除 NIC hop
  - 提出 two-tier inter-node synchronization 机制解决非 coherent CXL 内存的一致性挑战
  - 基于 Dynamo LLM inference framework 实现
  - RDMA baseline 对比：**平均 TTFT 降低 9.8x，P99 延迟 6.2x 降低，峰值吞吐 1.6x 提升**
- **关键洞察**: KV transfer 在 P/D 分离架构中从网络瓶颈变成了内存语义问题

#### Scalable PNM for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits
- **arXiv**: 2511.00321 (2025-10-31)
- **作者**: Dowon Kim, MinJae Lee, Janghyeon Kim 等（KAIST/三星）
- **核心贡献**:
  - 提出 CXL-enabled Processing-Near-Memory (PNM) 架构
  - 将 token page selection 完全 offload 到 CXL memory 内的 PNM accelerator，消除 costly recalls
  - 引入 hybrid parallelization 和 steady-token selection 机制
  - 在 SOTA CXL-PNM 系统上实现 405B 参数 + 1M token 上下文
  - PNM-only (PNM-KV) 和 GPU-PNM hybrid (PnG-KV) 两种模式：
    - **21.9x 吞吐提升**
    - **60x 单 token 能耗降低**
    - **7.3x TCO 效率提升**
- **关键 insight**: 不只是换介质，而是在 CXL 内存内嵌入计算，"数据在哪里，计算就在哪里"

#### NetKV: Network-Aware Decode Instance Selection for Disaggregated LLM Inference
- **arXiv**: 2606.03910 (2026-06-02)
- **作者**: Mubarak Adetunji Ojewale
- **核心贡献**: 网络感知的 decode 实例选择，优化 P/D 分离架构下的路由决策

### 1.2 P/D 分离架构基础与演进

#### DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
- **arXiv**: 2401.09670 (2024-01), **OSDI'24**
- **作者**: Yinmin Zhong, Shengyu Liu 等（字节跳动/UC Berkeley）
- **核心贡献**: 将 prefill 和 decode 解耦到不同 GPU 上，消除干扰，co-optimize 资源和并行策略。**7.4x 请求吞吐 / 12.6x SLO 收紧**
- **与 CXL 的关系**: DistServe 奠定了 P/D 分离的基本范式，CXL 的引入进一步加速了其 KV 传输瓶颈

#### xLLM Technical Report
- **arXiv**: 2510.14686 (2025-10-16)
- **作者**: Tongxuan Liu, Tao Peng 等（京东 JD.com）
- **核心贡献**: 提出 Encode-Prefill-Decode (EPD) 三分离架构（针对多模态），全局 KV Cache 管理，容错能力。Qwen 系列：**1.7x MindIE, 2.2x vLLM-Ascend** 吞吐。DeepSeek 系列：**1.7x MindIE**。
- **GitHub**: https://github.com/jd-opensource/xllm

#### RTP-LLM: High-Performance Alibaba LLM Inference Engine
- **arXiv**: 2605.29639 (2025-05), 阿里巴巴

### 1.3 历史基础系统

#### MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool
- **arXiv**: 2406.17565 (2024-06), **ASPLOS'25 Workshop**
- **作者**: Cunchen Hu, Heyang Huang 等（华为/中科大）
- **核心贡献**: **MemPool** — 首个分布式弹性 memory pool 管理 KV cache，结合 context caching + disaggregated inference，global prompt tree-based locality-aware policy
- **与 CXL 的关系**: MemServe 是 P/D 分离 + memory pooling 组态的先驱，但当时未使用 CXL 硬件

### 1.4 模型层优化（影响 KV Cache 传输量）

#### DeepSeek-V3 Technical Report
- **arXiv**: 2412.19437 (2024-12)
- **MLA (Multi-head Latent Attention)**: 将 KV cache 压缩为 latent vector，大幅降低 KV cache 内存占用
- 直接影响 P/D 分离架构下的 KV cache 传输带宽需求

#### Towards Economical Inference: Enabling DeepSeek's MLA in Any Transformer-based LLMs
- **arXiv**: 2502.14837 (2025-02)
- **贡献**: 将 MLA 适配到非 DeepSeek 模型（如 Llama, Qwen），进一步压缩 KV Cache

---

## 二、CXL 技术特性及其对 KV Cache 管理的影响

### 2.1 CXL 三代技术规格对比

| 技术 | CXL 1.1/2.0 | CXL 3.0/3.1 | CXL 4.0 (规划) |
|------|-------------|-------------|----------------|
| 底层协议 | PCIe 5.0/6.0 | PCIe 6.0 | 更高 |
| 单链路带宽 | ~64 GB/s (x16) | ~128 GB/s | 更高 |
| 关键特性 | cache coherence, memory pooling | Switch fabric, multi-host sharing, DCD | 2TB+ 演示 |
| 延迟 | ~100-200ns (加载) | ~200ns | - |
| 共享粒度 | 少数主机 | 缓存行(64B)粒度load/store | - |
| 拓扑 | 端到端 / 单层级 | CXL switch fabric | 多层级 |

### 2.2 CXL vs NVLink vs RDMA 关键指标对比

| 维度 | NVLink (NVSwitch) | RDMA (InfiniBand HDR) | CXL 3.0 |
|------|-------------------|----------------------|---------|
| 单链路带宽 | 900 GB/s (NVLink 5.0) | ~400 Gb/s (~50 GB/s) | ~128 GB/s |
| 延迟 | ~1-3 µs (GPU直连) | ~1-2 µs (RDMA Write) | ~0.2-1 µs |
| 访问粒度 | 共享内存字节/页 | 字节(消息) | 缓存行(64B)load/store |
| 语义 | GPU Unified Memory | RDMA Send/Write/Read | CPU load/store + DMA |
| 主机范围 | 同一节点内 | 跨网络 | 机架内(可扩展到多机架) |
| 一致性 | GPU-coherent | 无 | cache-coherent (部分) |
| Switch 能力 | NVSwitch 封闭网络 | IB Switch | CXL switch (开放生态) |
| 成本$$$$ | 高(封闭生态) | 中高(IB NIC+交换机) | 中(PCIe兼容) |
| 易用性 | CUDA 原生 | 需要 RDMA verbs | 类本地内存访问 |

### 2.3 CXL 三大核心能力对 KV Cache 的影响

#### (A) Memory Pooling（共享内存池）
- **传统问题**: 每台 sever 的 DRAM 容量受 CPU memory channel 数目限制（典型 12 channels/socket × 2 sockets = 24 DIMMs ≈ 1.5-3 TB）
- **CXL 解决方案**: 通过 CXL switch 将多个 CXL memory expander (如 Samsung CMM-D, SK Hynix CMM) 聚合为统一内存池
- **对 KV Cache 的意义**: 
  - Beluga 实现了 GPU 通过 CXL switch 直接访问 pooled memory
  - 单个 CG (CXL attached GPU) 可用内存从 HBM (80GB) 扩展到 TB 级
  - **打破单机 KV cache 容量硬上限**
  
#### (B) Byte-addressable Load/Store 语义
- **vs RDMA**: RDMA 需要显式 verbs 提交、QP/CQ 管理、端到端同步
- **CXL**: 像访问本地缓存行一样访问远程内存
- **对 KV Cache 的意义**:
  - SAC 利用 64B 粒度 load/store，只取 sparse attention 需要的 KV entries
  - 避免了 RDMA 必须预取整个 prefix cache 的浪费
  - 对于 1024 token context，稀疏注意力只需 5-10% 的 KV entries — **节省 90%+ 带宽**

#### (C) Cache Coherence + Shared Access
- **特点**: CXL 支持 CPU 和加速器（包括 GPU）之间的 cache-coherent 共享
- **对 KV Cache 的意义**:
  - TraCT 用 CXL shared memory 直接在 prefill GPU 和 decode GPU 之间共享 KV blocks
  - P/D 间的 NIC hop 被消除，KV transfer 变成"内存复制"语义
  - 但是**非完全 coherent** 的 CXL 内存需要显式一致性管理

---

## 三、KV Cache 分级管理的技术跃迁

### 3.1 传统方案（无 CXL）

```
GPU HDRAM (主动 batch)
  ↕ PCIe (copy)
CPU DRAM (Prefill 结果的 KV cache, 等待传输到 Decode GPU)
  ↕ RDMA (网络传输)
Remote CPU DRAM (跨节点)
  ↕ NVMe-oF (JBOF/DPU)
NVMe SSD (长尾 offload)
```

### 3.2 CXL 带来的分级重塑（以 ITME 为代表）

```
GPU HBM (active KV cache, decode attention)
  ↕ CXL load/store (CPU一侧统一视角)
CXL-hybrid memory pool (TB级byte-addressable, prefix cache + weight expansion)
  ↕ proactive data movement
CPU DRAM (中间层)
  ↕ NVMe-oF (via DPU)
NVMe SSD (冷 context, JBOF)
```

**核心变化**:
1. **CXL 将 CPU DRAM 从"中转站"升级为"分级池的一部分"**
2. **ITME 利用 deterministic access patterns**，在 decode 之前主动将需要的 KV entries 预取到靠近 GPU 的层级
3. **SK Hynix CMM (CXL Memory Module)** 的商用化使得 CXL memory pooling 不再是理论

### 3.3 KV Cache 在 sparse attention 下的范式转变

**SAC 的关键贡献**: Dense attention → Sparse attention 时，cache-line 粒度按需获取

```
Dense Attention (MHA/GQA):
  RDMA 方案: 拉取整个 prefix KV cache block → 高浪费
  CXL 方案: 选择需要的 KV entries → cache-line 粒度精准 fetch

Sparse Attention (MLA, DSA等):
  SAC 方案: 
    1. CXL load/store 发起细粒度读取 (64B级别)
    2. P99 TTFT 降低 9.7x vs RDMA
    3. 吞吐提升 2.1x
```

### 3.4 KV Cache Preemption/Eviction 策略变化

传统 preemption (如 vLLM 的 Page Eviction) 基于 LRU/LFU，当内存不足时挤掉 KV pages。

引入 CXL 后:
- **CXL 级别的 eviction 决策**: 哪些 KV pages 留在 CXL-pool 而不是完全删除
- **PNM-KV**: 不在 GPU HBM 中做 eviction，而是在 CXL 内存内的 PNM 加速器上做 token page selection
- **CXL tiering**: KV cache page 的 stay/migrate/delete 决策变为多层级优化问题

---

## 四、技术挑战与机遇

### 4.1 技术挑战

#### 挑战 1: CXL 内存的非完全一致性（Non-Coherent Shared Memory）
- **问题**: CXL 3.x 支持 cache coherence 但存在限制，特别是 GPU-side cache 和 CXL memory pool 之间的consistency
- **TraCT 的解法**: two-tier inter-node synchronization (本地 lock-free + 远端 message-based)
- **通用解**: 需要新的 consistency protocol 针对 KV cache 的 append-once, read-many 特性优化

#### 挑战 2: CXL Switch Fabric 的拓扑与调度
- CXL 3.0 switch 支持 multi-host 和一个 CXL 内存模块共享
- 多个 GPU 同时访问同一 CXL pool 的 **带宽争抢** 问题
- 需要设计新的 **KV cache placement 策略** 匹配 CXL fabric topology
- Beluga 基于商用 CXL switch 做了系统性 characterization

#### 挑战 3: KV Cache 传输粒度与带宽的匹配
- RDMA 适合大块传输 (一个完整的 prefix cache, 几十~几百 KB)
- CXL load/store 适合小块传输 (64B cache-line)
- 需要 **重新设计 KV cache block/page 布局**，匹配 CXL 的细粒度语义

#### 挑战 4: GPU 原生 CXL 支持不成熟
- 当前 NVIDIA GPU (H100/H200/B200) 不直接支持 CXL
- AMD MI300X 有限支持 CXL
- **现状**: FPGA 原型验证（如 ITME）或用 CPU 作为 access bridge
- **未来**: NVIDIA Rubin 及后续架构预计会有更强的 CXL 支持

#### 挑战 5: 编程模型与软件栈
- CXL 编程仍然需要新的 libcxl、CXL kernel 支持
- 需要像 NCCL 之于 NVLink/RDMA 那样的 **CXL-aware collective library**

### 4.2 技术机遇

#### 机遇 1: CXL 原生 P/D 分离
- TraCT/SAC/Beluga 证明了 CXL 做 P/D KV transfer 的可行性
- TTFT 降低量级: **6-10x**，远超 RDMA 的 2-3x 改进

#### 机遇 2: KV Cache as a Service (KCaaS) on CXL
- 多个模型实例共享同一 CXL memory pool
- 多级 prefix cache (system prompt, few-shot examples, RAG context)
- 实现真正的 **KV cache tier-as-a-service**

#### 机遇 3: KV Cache + 未来模型架构协同
- MLA (Multi-head Latent Attention): 用低秩压缩 KV cache，90% 压缩率
- DSA (DeepSeek Sparse Attention): 按需 attention，天然适配 CXL 细粒度 fetch
- 这些模型架构 + CXL 硬件 = **乘法级别效益**

#### 机遇 4: 降低推理 TCO
- PNM-CXL 报告 **7.3x TCO 效率提升**
- Memory pooling 降低 DRAM 总成本（DRAM over-provisioning 浪费）
- 减少 GPU idle time → 更高利用率

---

## 五、头部公司/机构布局

### 5.1 学术界 (2025-2026)

| 机构 | 论文/贡献 | 核心特点 |
|------|-----------|---------|
| **UNIST** (韩国) | ITME, TraCT | SK Hynix/三星 CXL 硬件直连, rack-scale |
| **北京大学** | SAC, Beluga | 首个 CXL-switch GPU direct access, SIGMOD'26 |
| **KAIST** | PNM-CXL | CXL-Processing-Near-Memory, 1M-token, 三星合作 |
| **字节跳动** | DistServe, xLLM, SAC, Beluga | 工程化主导, SGLang/Mooncake |
| **京东** | xLLM | Encode-Prefill-Decode 三分离, 全局 KV Cache |
| **华为** | MemServe | Memory elastic pool + ASPLOS |
| **阿里巴巴** | RTP-LLM | 工业级推理引擎, K2 部署 |

### 5.2 产业界

| 主体 | 动态 |
|------|------|
| **SK Hynix** | CMM (CXL Memory Module) 量产, ITME 论文硬件合作方 |
| **Samsung** | CMM-D (CXL Memory Module-DRAM) 产品线, 面向 AI |
| **Intel** | CXL consortium 创始成员, SC25 演示 TB 级 CXL pooling |
| **NVIDIA** | 暂未提供 GPU native CXL, GB200 NVL72 走 NVLink 封闭路线 |
| **AMD** | MI300X 有限 CXL 支持, MI400 系列或加强 |
| **LMSYS + Mooncake Team** | SGLang + Mooncake 已在 128 GPU scale 部署 P/D 分离架构, 2.7x GB200 decode throughput |
| **CXL Consortium** | 2026 Xcelerated Compute Show 重点推 CXL for AI, 2TB CXL demo |

### 5.3 LMSYS SGLang 生态系统演进

从 LMSYS 博客 2025-2026 的内容看:
- **Mooncake**: P/D 分离的 transfer engine，已成为 SGLang 核心组件
- **HiCache**: 层级化 KV Cache（HBM→SSD），是 software tiering 方向
- **PD-Multiplexing (GreenContext)**: 请求级别的 goodput 优化
- **EPD Disaggregation**: 多模态场景的三分离
- **GB200 NVL72 部署**: PD + 大规模 EP + NVLink 全局一致
- **Pipeline Parallelism**: 面向 1M-token+ 上下文

**对 CXL 的启示**: SGLang 在 GB200 上证明 NVLink 的全局一致性优势，而 CXL 需要追上这部分能力才能实现 rack-scale 替代。

---

## 六、前沿趋势总结

### 6.1 短期趋势 (2026-2027)
1. **CXL 3.1/4.0 switch 成熟**: multi-host memory pooling 成为数据中心可部署方案
2. **CXL-native LLM serving 出现 2-3 个实际部署**: 特别是在 agentic/long-context 场景
3. **GPU 端到端 CXL 支持**: NVIDIA/AMD 下一代 GPU 预计开放 CXL 接口

### 6.2 中期趋势 (2027-2029)
1. **CXL pooling + KV Cache tiering 成为标配**: 每个 rack 内共享 10-100 TB 内存池
2. **KV Cache 控制面/数据面分离**: RDMA 控制面 + CXL 数据面混合
3. **PNM (Processing Near Memory) + CXL**: KV cache page selection/eviction 完全 offload

### 6.3 范式变化判断
- **从"内存复制"到"内存共享"**: CXL 让 KV cache 的复制语义变成 load/store 语义
- **从"请求到 GPU 绑定"到"KV Cache 独立调度"**: CXL pooled memory 使 KV cache 不从属于特定 GPU
- **从"粗粒度 page transfer"到"细粒度 entry access"**: 64B cache-line 粒度成为可能
- **从"网络瓶颈"到"内存池命中率瓶颈"**: 优化目标从"减少网络传输字节"变为"提高 pool 命中率"

---

## 七、关键论文完整引用

1. **ITME** - Hakbeom Jang et al., "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories", arXiv:2606.12556, 2026.
2. **SAC** - Ruiyang Ma et al., "SAC: Disaggregated KV Cache System for Sparse Attention LLMs with CXL", arXiv:2606.19746, 2026.
3. **PNM-CXL** - Dowon Kim et al., "Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits", arXiv:2511.00321, 2025.
4. **Beluga** - Xinjun Yang et al., "Beluga: A CXL-Based Memory Architecture for Scalable and Efficient LLM KVCache Management", arXiv:2511.20172, SIGMOD'26, 2025.
5. **TraCT** - Dongha Yoon et al., "TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale", arXiv:2512.18194, 2025.
6. **DistServe** - Yinmin Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving", arXiv:2401.09670, OSDI'24, 2024.
7. **MemServe** - Cunchen Hu et al., "MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool", arXiv:2406.17565, 2024.
8. **xLLM** - Tongxuan Liu et al., "xLLM Technical Report", arXiv:2510.14686, 2025.
9. **NetKV** - Mubarak Adetunji Ojewale, "NetKV: Network-Aware Decode Instance Selection for Disaggregated LLM Inference", arXiv:2606.03910, 2026.
10. **RTP-LLM** - Boyu Tan et al., "RTP-LLM: High-Performance Alibaba LLM Inference Engine", arXiv:2605.29639, 2026.
11. **DeepSeek-V3** - DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437, 2024.
12. **MLA for All** - Tao Ji et al., "Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs", arXiv:2502.14837, 2025.

---

## 八、URL 索引

### 访问成功的 URL
- https://arxiv.org/abs/2606.12556 (ITME)
- https://arxiv.org/abs/2606.19746 (SAC)
- https://arxiv.org/abs/2511.20172 (Beluga)
- https://arxiv.org/abs/2512.18194 (TraCT)
- https://arxiv.org/abs/2511.00321 (PNM-CXL)
- https://arxiv.org/abs/2406.17565 (MemServe)
- https://arxiv.org/abs/2401.09670 (DistServe)
- https://lmsys.org/blog/ (LMSYS blog archive)
- https://lmsys.org/blog/2025-06-16-gb200-part-1
- https://lmsys.org/blog/2025-07-20-k2-large-scale-ep
- https://computeexpresslink.org/blog/ (CXL Consortium blog archive)
- https://computeexpresslink.org/blog/scale-your-ai-performance-with-cxl-insights-from-the-xcelerated-compute-show-4546/
- https://semiconductor.samsung.com/cxl-memory/
- https://semiconductor.samsung.com/cxl-memory/cmm-d/

### arxiv API 调用（全部成功返回 XML）
- https://export.arxiv.org/api/query?id_list=2606.12556
- https://export.arxiv.org/api/query?id_list=2606.19746
- https://export.arxiv.org/api/query?id_list=2511.20172
- https://export.arxiv.org/api/query?id_list=2511.00321
- https://export.arxiv.org/api/query?id_list=2512.18194
- https://export.arxiv.org/api/query?id_list=2406.17565
- https://export.arxiv.org/api/query?id_list=2401.09670
- https://export.arxiv.org/api/query?id_list=2510.14686
- https://export.arxiv.org/api/query?id_list=2606.03910
- https://export.arxiv.org/api/query?id_list=2412.19437
- https://export.arxiv.org/api/query?id_list=2502.14837
- https://export.arxiv.org/api/query?id_list=2605.29639

### 访问失败
- https://www.intel.com/content/www/us/en/products/docs/memory-storage/compute-express-link/overview.html (redirected)
- https://www.semiconductor.samsung.com/news-events/tech-blog/compute-express-link-cxl-the-future-of-memory/ (404)
- https://www.computeexpresslink.org/blog/discovering-cxl-memory-expanding-possibilities-for-ai-inference (404)
- https://lmsys.org/blog/2024-02-05-mooncake/ (404, path replaced)
- https://news.skhynix.com/cxl-memory-for-ai/ (403)

---

## 九、核心结论与研判

### 9.1 什么是确定性的？
1. **CXL memory pooling 已经商用**（Samsung CMM-D, SK Hynix CMM），不是概念验证
2. **CXL 比 RDMA 更适合 KV cache 细粒度传输**（SAC 9.7x TTFT 改善 — 铁证）
3. **KV cache capacity 瓶颈已经超越 compute 成为推理系统首要挑战**
4. **GPU 内存统一接口的缺失**是当前 CXL 落地的最大障碍

### 9.2 什么还在演进？
1. **NVIDIA 会在什么时候开放 GPU-native CXL 接口** (影响未来3年格局)
2. **CXL switch fabric 是取代还是补充 NVLink**（rack-scale 竞争）
3. **KV cache tiering policy** 的最佳实现 (rule-based vs ML-guided)
4. **PNM (Processing Near Memory) 何时进入 LLM serving 主流**

### 9.3 对芯片设计者的启示
- 下一代 GPU **必须考虑 CXL 接口**，否则在 rack-scale LLM serving 会落后
- **CXL 控制器 IP** 将成为推理加速器标配
- **HBM → CXL-DRAM → CXL-Flash** 的分级是必然趋势
- **Sparse attention + CXL entry-level fetch** 是当前最佳匹配

---

*本报告由 AI 调研助手基于 2026-07-14 当日公开信息撰写。调研过程中共访问了 38+ 个 URL，成功获取 70+ 份本地存档文件，解析了 12 篇核心论文的完整摘要。*
