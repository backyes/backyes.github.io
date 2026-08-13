# Blogs Analysis Summary: KV Cache Management & Disaggregated Inference

> 涵盖 5 篇技术博客的核心洞察：SGLang HiCache、Multi-Token Prediction (MTP)、Kimi K2 PD 部署、P2P 权重传输、GB200 NVL72 部署。

---

## 1. SGLang HiCache — 分层 KV Cache 架构

**来源**: "SGLang HiCache: Fast Hierarchical KV Caching with Your Favorite Storage Backends" (Sept 10, 2025)

### 核心设计理念
- **HiRadixTree**: 在 RadixAttention 之上构建的分层扩展，类似"页表"机制，统一管理 GPU ↔ CPU ↔ 外部存储的 KV Cache 引用
- **三层存储**: GPU HBM → CPU DRAM → 磁盘/远端存储（3FS、Mooncake、NIXL）
- **Controller 自动管理**: 数据备份/加载/驱逐由中央 Cache Controller 调度

### 数据面优化 (Optimized Data Plane)
- **GPU-assisted I/O kernels**: 自研 CPU↔GPU 传输内核，吞吐提升 **3x**
- **Page-first layout**: CPU 存储层使用 "page-first" 布局（GPU 仍用 "layer-first"），单事务传输量更大；配合 **zero-copy** 机制，典型部署吞吐提升 **2x**
- **Layer-wise overlap**: GPU miss 但 CPU hit 时，layer N 执行的同时加载 layer N+1 的 KV Cache，掩盖传输延迟

### 控制面策略 (Versatile Control Plane)
- **存储层 prefetch**: 命中存储层时从存储→host 预取；支持 best-effort、terminate-on-schedule、aggressive-staging 等模式
- **写策略灵活**: write-through（带宽富余时最强缓存效果）、write-through-selective（仅备份 hot spot，减少 I/O）、write-back（缓解慢层容量压力）
- 与 PD Disaggregation **协同设计**（ongoing）

### 性能指标
| 场景 | 关键指标 |
|------|---------|
| Qwen3-Coder-480B (coding agent) + 3FS | TTFT **-56%**，吞吐量 **2x**，cache hit rate 40%→**80%** |
| DeepSeek-R1-671B PD 部署 + Mooncake | cache hit 时 TTFT **-84%**（对比全量重计算） |
| 通用长上下文 benchmark | 吞吐最高 **6x** 提升，TTFT 最高 **-80%** |

### 已集成存储后端
**3FS**（阿里 TairKVCache）、**Mooncake**、**NIXL**（NVIDIA Dynamo 生态），另有 HiFile 参考实现。接入仅需实现 `get(key)` / `exist(key)` / `set(key, value)` 三个接口。

---

## 2. Multi-Token Prediction (MTP) — 加速 Decode

**来源**: "Accelerating SGLang with Multiple Token Prediction" (July 17, 2025)

### 核心机制
- **轻量级 draft model** 预测 n 个 token → **单次并行验证** 于 target model
- 属于 **Speculative Decoding** 范畴，用并行验证替代 n 个串行 decode step

### 部署性能（DeepSeek V3）
| 配置 | 吞吐 (tokens/sec/rank) | Acceptance Length | 相对基线 |
|------|------------------------|-------------------|----------|
| 无 overlap, 无 MTP | 51.0 | — | baseline |
| overlap only | 60.4 | — | +20.4% |
| 3-token MTP (topk=1) | **81.5** | 2.18 | **+59.8%** |
| 4-token MTP (topk=1) | **82.0** | 2.44 | **+60.8%** |

- 大规模集群 (128×H200, 4P+12D): MTP 仍有 **+14.2%** 吞吐提升
- 与 EP、PD Disaggregation、CUDA Graph、Two Batch Overlap 完全集成

### 工程实践建议
- 默认 `draft_token_num=2` 是低风险高回报选择
- GPU headroom 充足可尝试 4+（接受率下降时回调）
- **与 overlap scheduling 尚未融合**（未来方向，仍有额外空间）

---

## 3. Kimi K2 PD 部署 — 万亿参数 MoE 的 PD 分离实战

**来源**: "Deploying Kimi K2 with PD Disaggregation and Large-Scale EP on 128 H200 GPUs" (July 20, 2025)

### 模型特征
- 1T 总参数, **32B 激活参数/token**, **384 experts** 动态路由, MLA 长上下文支持

### PD 分离策略
| 阶段 | 角色 | 优化方向 |
|------|------|---------|
| Prefill | 大 prompt 摄入（~2000 tokens） | compute-bound, 大 batch 并行 |
| Decode | 自回归生成（~100 tokens） | latency-sensitive, 高吞吐 |

- 4 Prefill 节点 + 12 Decode 节点（优先扩大 decode 节点 KV cache 池）
- NUMA-aware GPU 分组优化 NVLink/PCIe 利用
- Decode 侧 **96 冗余 experts** 平衡 MoE 路由 + EPLB 负载均衡

### 集群性能 (128×H200)
| 指标 | 数值 |
|------|------|
| Prefill Throughput | 224k tokens/sec (4 P nodes) |
| Decode Throughput | 288k tokens/sec (12 D nodes) |
| 每百万 output tokens 成本 | **~$0.21** (H200 $2.3/hr) |
| Decode Batch Size | 480 |

对比 R1@96×H100 (22.3k tokens/sec/node)，K2@128×H200 单节点 decode 达 24k tokens/sec，**单位 GPU 吞吐更高**。

### OME（Open Model Engine）
- Kubernetes Operator 式声明式部署，抽象所有分布式细节
- SGLang Router: 动态服务发现 + least-privilege routing + decode 独立扩缩容
- **RDMA 加速 KV Cache 传输** 是 PD 分离的核心网络方案

### 关键洞察
- 1T MoE 的 EP=32, PP=8 大规模场景下，PD 分离仍然保持良好单位 GPU 效率
- Agent 场景输入可达 **30k–50k tokens**（长上下文后续优化方向）

---

## 4. P2P 权重传输 — 分布式 RL 的关键路径优化

**来源**: "Updating 1T parameters in seconds — P2P weight transfer in Large Scale Distributed RL" (April 29, 2026)

### 核心差异
NCCL Broadcast 在 RL 权重同步中的痛点：
- **冗余传输**: 相同数据多次发送
- **资源闲置**: 仅少数 rank 参与 broadcast，其余 idle
- **组固定**: NCCL group 一旦定义不能动态扩展

### RDMA P2P 方案
- **Source-side CPU engine replica**: 训练侧 CPU 内存创建模型副本，不占 GPU VRAM
- **P2P mapping**: 每个 trainer rank 将专属 shard 直传目标，消除冗余
- **Zero-copy**: TransferEngine 启动时一次性注册内存，跳过 CUDA IPC 序列化

### 传输性能对比
| 模型 | NCCL (ms) | RDMA P2P (ms) | 加速比 |
|------|-----------|---------------|--------|
| Qwen3-235B-A22B | 10,753.6 | 3,162.0 | **3.40x** |
| GLM-5 (744B) | 58,301.5 | 8,479.7 | **6.88x** |
| **Kimi-K2-fp8 (1T)** | **53,279.1** | **7,227.3** | **7.37x（53s → 7.2s）** |

### 内存权衡公式
```
NCCL:   参与 source ranks = pp,  每 rank 接收 ep*P params,  target 侧 buffer = K
RDMA P2: 参与 source ranks = M,  每 rank 接收 P params,    target 侧 buffer = 0, source 侧 buffer = K + P
```

### 工程要点
- CPU 侧 replica 复用同一物理内存，通过 threadpool 串行化多目标传输
- bucketed all-gather 的 shard 收集等待通过 buffer > 1 的设计解决
- 代价: 每个训练 rank 额外 32GB CPU 副本

### 与 KV Cache 的关联
权重更新期间**整个 RL 训练停滞**，P2P 的 7.37x 提速直接缩短推理引擎等待窗口。PD 分离推理侧依赖 RDMA 高速网络，侧证了 RDMA 在 inference 基础设施中的普适价值。

---

## 5. GB200 NVL72 — PD + 低精度 MoE 部署

**来源**: "Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II)" (Sept 25, 2025)

### 核心技术创新
| 技术 | 收益 |
|------|------|
| **FP8 Attention (KV cache)** | Decode attention kernel 速度 **1.8x**；KV cache 容量翻倍，batch size 更大 |
| **NVFP4 GEMM** | GEMM kernel 最高 **1.9x**；内存带宽减半；权重显存减半 |
| **DeepEP NVFP4 fusion** | token dispatch 网络流量 **减半** |
| **Offloading to CPU** | GB200 CPU↔GPU 900GB/s 双向带宽，weight offload + prefetch |
| **Fine-grained overlap** | combine 通信与 down GEMM + shared expert 重叠；atomic release + cp.async.bulk.wait |

### 集群性能（DeepSeek V3/R1, ISL=2000, OSL=100）
| 精度 | Prefill (tokens/sec/GPU) | Decode (tokens/sec/GPU) | 相对 H100 |
|------|--------------------------|-------------------------|-----------|
| BF16 Attn + FP8 MoE | 18,471 | 9,087 | — |
| **FP8 Attn + NVFP4 MoE** | **26,156** | **13,386** | **Prefill 3.8x / Decode 4.8x** |

### 量化精度验证
- NVFP4（block size=16, FP8 scale 因子）精度损失可忽略
- 与 NVIDIA 官方 NVFP4 checkpoint 精度一致

### 与 KV Cache 的关联洞察
1. **FP8 KV Cache 是 PD 分离的核心放大器**: KV cache 容量翻倍 → batch size 从 768→1408 (2k ISL) → 系统效率提升 ~10%
2. **权重 offload 释放显存给 KV cache**: EP 降配 + CPU offload，更多显存给 KV pool
3. **Decode 注意力属于 memory-bound**: FP8 减半显存带宽压力，attention decode 速度质变
4. **DeepGEMM Blackwell 统一 prefill/decode kernel**: 使 attention kernel 在 PD 两头都高效

---

## 跨博客综合洞察

### KV Cache 层次演进路线
1. **单层 GPU**: RadixAttention — in-GPU 复用
2. **双层 (HiCache)**: GPU ↔ CPU + Page-first layout + GPU-assisted IO kernel
3. **三层 (HiCache)**: GPU ↔ CPU ← RDMA → 远端分布式存储 (3FS / Mooncake)
4. **存储级 KV Cache**: NIXL 对接 GPU-direct storage / cloud object store

### PD 分离正在融合的技术栈
- **Prefill**: 大 batch 并行 + BF16/FP8 attention + DeepGEMM/CUTLASS GEMM + NVFP4 offload
- **Decode**: FP8 attention（KV cache 容量翻倍）+ 大 batch + 低精度 MoE + EP load balance
- **KV 传输**: RDMA（P2P 侧证其价值）、3FS/Mooncake 远端存储、HiCache 透明分层
- **调度**: SGLang Router 独立扩缩容 + least-privilege routing

### 精度-效率关系
- **FP8 KV Cache**: 已验证精度无损，decode attention 速度 1.8x，batch size 扩容 ~1.83x → 总收益乘数效应
- **NVFP4 MoE**: block-wise 量化（block=16, FP8 scale），精度损失可忽略，GEMM 速度 1.9x + 网络流量减半
- PTQ 量化对推理引擎友好，但需 per-block 细粒度 + FP8 scale 保留动态范围

### 稀缺资源再分配趋势
- 算力增长（Blackwell FP4）赶不上模型增长（1T MoE） → 显存成为瓶颈
- 策略: weight offload/quantize → 把显存让给 KV cache → 大 batch → 系统效率
- **KV Cache 容量正在成为 throughput 的第一性指标**

---

## 关键原始链接索引
- HiCache: SGLang blog, Sept 2025 (Zhiqiang Xie)
- MTP: SGLang blog, July 2025 (Eigen AI Team)
- Kimi K2 PD: Mooncake Team blog, July 2025
- P2P Weight Transfer: miles/sglang-miles blog, April 2026 (Jiadong Guo et al.)
- GB200 Part II: SGLang Team blog, Sept 2025
