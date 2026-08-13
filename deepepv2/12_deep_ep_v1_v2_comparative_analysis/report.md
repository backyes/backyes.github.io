# DeepEP V1 vs V2 深度对比分析：统一 ElasticBuffer、路由缓存、Engram 与 SM-Free 通信

> **分析范围**：DeepEP V1 (Legacy Buffer, NVSHMEM) vs V2 (ElasticBuffer, NCCL Gin) 的架构差异、路由缓存机制、Engram 远端内存访问、以及 Normal/Low-Latency 模式统一
>
> **核心源码文件**：
> - `deep_ep/buffers/legacy.py` (714 行) — V1 Legacy Buffer
> - `deep_ep/buffers/elastic.py` (1108 行) — V2 ElasticBuffer
> - `csrc/elastic/buffer.hpp` (1382 行) — ElasticBuffer C++ 核心
> - `csrc/kernels/elastic/engram.hpp` — Engram JIT runtime
> - `deep_ep/include/deep_ep/impls/engram_fetch.cuh` — Engram fetch kernel
> - `csrc/kernels/legacy/internode_ll.cu` — V1 low-latency kernel
> - `tests/elastic/test_ep.py` — V2 EP 测试（含 cached dispatch）
> - `tests/elastic/test_engram.py` — Engram 测试
> - `DeepEP/docs/legacy.md` — V1 官方文档
> - `DeepEP/README.md` — V2 官方文档

---

## 目录

1. [V1 vs V2 核心差异总览](#1-v1-vs-v2-核心差异总览)
2. [统一 ElasticBuffer：V2 的核心设计哲学](#2-统一-elasticbufferv2-的核心设计哲学)
3. [EPHandle 路由缓存机制](#3-ephandle-路由缓存机制)
4. [为什么 Decode 用 Low-Latency 而 Prefill 用 Normal？](#4-为什么-decode-用-low-latency-而-prefill-用-normal)
5. [SM-Free 通信：RDMA Hook 机制](#5-sm-free-通信rdma-hook-机制)
6. [Normal vs Low-Latency 模式的算法差异](#6-normal-vs-low-latency-模式的算法差异)
7. [Engram：远端内存访问原语](#7-engram远端内存访问原语)
8. [DeepSeek MoE 路由与两种模式的配合](#8-deepseek-moe-路由与两种模式的配合)
9. [关键引用与源码索引](#9-关键引用与源码索引)

---

## 1. V1 vs V2 核心差异总览

| 维度 | V1 (Legacy) | V2 (ElasticBuffer) |
|------|-------------|---------------------|
| **通信后端** | NVSHMEM | NCCL Gin (GPU-Initiated Networking) |
| **编译方式** | 预编译 NVSHMEM kernel | Fully JIT (运行时编译) |
| **API 设计** | `Buffer` 类，Normal/Low-Latency 两套独立 API | `ElasticBuffer` 类，统一 API |
| **Buffer 模型** | 分离的 NVL buffer + RDMA buffer | 统一 symmetric memory |
| **SM 控制** | 静态变量 `Buffer.num_sms = 24` | 解析式自动计算 `get_theoretical_num_sms` |
| **QP 控制** | 手动指定 `num_qps_per_rank` | 解析式自动计算 `get_theoretical_num_qps` |
| **扩展规模** | EP ≤ 128 | EP ≤ 2048 |
| **新增功能** | 无 | Engram, PP, CP, AGRS |
| **性能** | 基准 | 1.3× 峰值带宽，4× SM 节省 |

**来源**: `DeepEP/README.md` L1-39, `DeepEP/docs/legacy.md` L1-9

---

## 2. 统一 ElasticBuffer：V2 的核心设计哲学

### 2.1 V1 时代的割裂

在 DeepEP V1 中，Normal 和 Low-Latency 是**两套完全独立的实现**：

```python
# V1 Normal: 3-phase forwarding
num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = \
    buffer.get_dispatch_layout(topk_idx, num_experts)
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
    buffer.dispatch(x, topk_idx=topk_idx, ...)

# V1 Low-Latency: 完全不同的 API
recv_hidden_states, recv_expert_count, handle, event, hook = \
    buffer.low_latency_dispatch(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts)
```

**关键问题**：
- 两个 API 使用不同的 buffer 分配策略
- 不同的 kernel 文件 (`internode.cu` vs `internode_ll.cu`)
- 不同的 handle 结构 (tuple of tensors)
- Low-Latency 模式无法与 Normal 模式共享 buffer

### 22 V2 的统一设计

V2 的核心洞察：**Normal 和 Low-Latency 的本质差异不是"通信协议"，而是"资源分配策略"**。

```python
# V2 统一 API
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_experts=num_experts,
    num_sms=num_sms,  # 控制 SM 数量：多=Normal, 少=Low-Latency
    ...
)
```

**统一的维度**：
1. **同一个 Buffer**：所有通信模式共享
2. **同一个 dispatch/combine API**：参数控制行为
3. **SM 数量**是主要区分因子（Normal 64-160 SMs, Low-Latency 4-8 SMs）

**来源**: `docs/11_deep_ep_v2_deepdive_normal_lowlatency_sm_estimation.md` L22-60

---

## 3. EPHandle 路由缓存机制

### 3.1 EPHandle 的定义

`EPHandle` 是 `dispatch` 返回的**路由元数据对象**，包含所有用于后续 combine 的信息：

```python
class EPHandle:
    do_expand: bool
    num_experts: int
    expert_alignment: int
    num_max_tokens_per_rank: int
    num_sms: int  # dispatch 时使用的 SM 数, combine 时复用
    topk_idx: Tensor  # [num_tokens, num_topk], clone 的 expert indices
    psum_num_recv_tokens_per_scaleup_rank: Tensor  # [num_scaleup_ranks]
    psum_num_recv_tokens_per_expert: Tensor  # [num_local_experts]
    num_unaligned_recv_tokens_per_expert: Tensor  # [num_local_experts]
    num_recv_tokens_per_expert_list: list  # CPU-side per-expert counts
    recv_src_metadata: Tensor  # 源 token 元数据
    dst_buffer_slot_idx: Tensor  # 目标 buffer slot 索引
    token_metadata_at_forward: Optional[Tensor]  # Hybrid 模式
    channel_linked_list: Optional[Tensor]  # Hybrid 模式
```

**来源**: `deep_ep/buffers/elastic.py` L25-98

### 3.2 Cached Dispatch 机制

当 `handle` 参数被传入 `dispatch` 时，进入 **cached mode**：

```python
# elastic.py L937-944
if handle is not None:
    assert topk_idx is None  # 不再接受新的 topk_idx
    assert do_cpu_sync is None or not do_cpu_sync  # 禁止 CPU sync
    topk_idx = handle.topk_idx  # 复用缓存的 topk_idx
    num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, handle.num_max_tokens_per_rank)
    num_experts = value_or(num_experts, handle.num_experts)
    do_cpu_sync = False  # 关键：跳过 CPU sync
```

**Cached mode 的关键行为**：
1. **跳过 layout recomputation**：`cached_num_recv_tokens`, `cached_psum_*` 直接传入 C++ kernel
2. **跳过 CPU sync**：不需要等待 GPU→CPU 的信号
3. **跳过 topk_idx clone**：`do_handle_copy` 仍会执行
4. **C++ 层行为**：`cached_mode = cached_num_recv_tokens.has_value()`（`buffer.hpp` L734）

### 3.3 为什么需要路由缓存？

**Decode 场景的特殊性**：
- Batch size 小（128 tokens）
- Routing decisions 在连续 decode step 之间**可能不变**（如果 gating 结果一致）
- 每次重新计算 layout 有 CPU overhead
- CPU sync 会引入 latency

**使用方式**（`README.md` L271-329）：
```python
def decode_dispatch(x, topk_idx, topk_weights, ...):
    if cached_handle is not None:
        # Reuse cached handle: skip layout recomputation and CPU sync
        recv_x, _, _, handle, event = _buffer.dispatch(
            x, handle=cached_handle, ...)
        return recv_x, cached_handle.topk_idx, None, handle, event
    
    # 首次：完整计算
    recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(...)
    return recv_x, recv_topk_idx, recv_topk_weights, handle, event
```

**来源**: `DeepEP/README.md` L271-329, `deep_ep/buffers/elastic.py` L937-944

---

## 4. 为什么 Decode 用 Low-Latency 而 Prefill 用 Normal？

### 4.1 工作负载特征对比

| 维度 | Prefill (Normal) | Decode (Low-Latency) |
|------|------------------|----------------------|
| **Batch size** | 4096-8192 tokens | 64-256 tokens |
| **优化目标** | 吞吐量 (GB/s) | 延迟 (μs) |
| **并行度** | 高（大量数据可并行） | 低（少量数据） |
| **Latency hiding** | 容易（大计算量） | 困难（小计算量） |
| **通信模式** | 3-phase forwarding | Direct RDMA get |

### 4.2 官方文档的说明

**Normal kernels**（`docs/legacy.md` L9-10）：
> "These kernels deliver high throughput, making them suitable for both training and inference prefilling tasks."

**Low-latency kernels**（`docs/legacy.md` L11）：
> "For latency-sensitive inference decoding, DeepEP V1 includes a set of low-latency kernels with pure RDMA to minimize delays."

### 4.3 量化分析

**V1 Normal 性能**（`docs/legacy.md` L17-26）：
- EP=16 internode: 43 GB/s, 4096 tokens
- EP=64 internode: 51 GB/s

**V1 Low-Latency 性能**（`docs/legacy.md` L29-39）：
- EP=8: 77 μs latency, 98 GB/s
- EP=64: 173 μs latency, 43 GB/s
- EP=256: 194 μs latency, 39 GB/s

**关键洞察**：Low-Latency 在小 EP 时延迟极低（77μs），但随 EP 增大延迟上升。Normal 的吞吐量稳定在 43-58 GB/s。

### 4.4 V2 的统一解释

在 V2 中，"Low-Latency" 不是独立模式，而是 Normal 在**特定参数配置下的特例**：

| 维度 | Normal 场景 | Low-Latency 场景 |
|------|-------------|------------------|
| **SM 数** | 多（64-160） | 少（4-8） |
| **QP 数** | 多（SM×16+1） | 少（SM+1） |
| **prefer_overlap** | False | True |
| **Buffer** | 相同 | 相同 |

**来源**: `docs/11_deep_ep_v2_deepdive_normal_lowlatency_sm_estimation.md` L136-150, `DeepEP/docs/legacy.md` L9-11

---

## 5. SM-Free 通信：RDMA Hook 机制

### 5.1 V1 的 Hook-Based Overlap

V1 Low-Latency 模式最引人注目的特性：**0 SM occupation** 通信。

```python
# docs/legacy.md L264-272
recv_hidden_states, recv_expert_count, handle, event, hook = \
    buffer.low_latency_dispatch(x, topk_idx, ...,
                                return_recv_hook=True)

# NOTES: the actual tensor will not be received only if you call `hook()`,
# it is useful for double-batch overlapping, but **without any SM occupation**
```

**机制**：
1. `return_recv_hook=True` 时，kernel 只**发起 RDMA 请求**，不等待完成
2. 返回的 `hook` 是一个 callable，调用时阻塞等待数据到达
3. RDMA 传输在**网络硬件**中进行，**不占用 GPU SMs**

### 5.2 双 Batch Overlap

`docs/legacy.md` L288-290 描述的 overlap 模式：

```
Batch 1: [Attention] → [Dispatch] → [MoE Compute] → [Combine]
Batch 2:          [Attention] → [Dispatch] → [MoE Compute] → [Combine]
                   ↑ 此时 Batch 1 的 RDMA 传输在后台进行，不占 SM
```

**4 个阶段的 overlap**：
1. Attention（当前 batch）
2. Dispatch RDMA（后台，SM-free）
3. MoE GEMM（当前 batch）
4. Combine RDMA（后台，SM-free）

### 5.3 V2 的变化

**重要**: V2 README 明确说明：

> "**Notes**: ... 0 SM RDMA low-latency EP is no longer supported"

V2 不再支持纯 0 SM 的 low-latency EP，但保留了：
- **0 SM Engram**（RDMA one-sided get）
- **0 SM PP**（RDMA send/recv）
- **0 SM CP**（Copy Engine）

V2 的通信-computation overlap 通过 `EventOverlap` + `async_with_compute_stream=True` 实现，仍然占用少量 SM。

**来源**: `DeepEP/README.md` L28-29, `DeepEP/docs/legacy.md` L11, L264-290

---

## 6. Normal vs Low-Latency 模式的算法差异

### 6.1 Normal 模式：3-Phase Forwarding

```
Phase 1: Push (NVLink)
  Local Rank → Send Buffer (via NVLink)
  
Phase 2: Forward (RDMA)
  Scaleup Warps → Scaleout RDMA → Remote Node
  
Phase 3: Reduce (Local)
  Received tokens → Expert computation
```

**特点**：
- 使用 **NVLink 域内聚合** + **RDMA 域间转发**
- 适合 DeepSeek-V3 的 group-limited gating（top-4 groups, top-8 experts）
- 需要 **24+ SMs** 处理 forwarding 逻辑
- Buffer 模型复杂（5 层 buffer）

### 6.2 Low-Latency 模式：Direct RDMA Get

```
Direct: Local Token → RDMA Get → Remote Expert Buffer
```

**特点**：
- **纯 RDMA**（IBGDA — InfiniBand GPU Direct Async）
- GPU SM 直接操作 NIC QP，无需 CPU 介入
- 每个 expert 对应一个 QP（`num_qps_per_rank = num_local_experts`）
- **无 forwarding**：直接 get 到目标位置
- 使用 **2 个 buffer** 交替（ping-pong）
- 需要 **clean_low_latency_buffer** 在 dirty 后重新 zero-init

### 6.3 Buffer 布局差异

**V1 Normal Buffer**：
```
[NVL Send Buffer] [NVL Recv Buffer] [RDMA Send Buffer] [RDMA Recv Buffer] [Expert Buffer]
```

**V1 Low-Latency Buffer**：
```
[RDMA Buffer 0] [RDMA Buffer 1]  # Double buffering
形状: [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]
```

**V2 Unified Buffer**：
```
[Workspace] [GPU Buffer] [CPU Buffer (for Engram)]
```

### 6.4 Kernel 对比

| 维度 | Normal (V1) | Low-Latency (V1) | V2 Unified |
|------|-------------|------------------|------------|
| Kernel 文件 | `internode.cu` | `internode_ll.cu` | `dispatch.cuh` / `hybrid_dispatch.cuh` |
| SM 角色 | Dispatch + Notify + Forward | Dispatch only | Dispatch + Notify (configurable) |
| 同步 | 多阶段 barrier | 简单 barrier | Gin barrier |
| 数据布局 | 5 层 buffer | 2 层 buffer | 统一 token layout |

**来源**: `DeepEP/docs/legacy.md` L9-11, L528-550, `docs/11_deep_ep_v2_deepdive_normal_lowlatency_sm_estimation.md` L22-60

---

## 7. Engram：远端内存访问原语

### 7.1 Engram 的定义

Engram 是 DeepEP 弹性模式的**远端内存访问原语**——通过 NCCL Gin（GPU-initiated RDMA）直接读取远端 GPU 的存储窗口（`ncclWindow`），无需 CPU 介入。

| API | 语义 |
|-----|------|
| `engram_write(storage, sf)` | 将本地数据注册到 NCCL window，供远端读取 |
| `engram_fetch(indices)()` | 异步发起 RDMA get，返回 callable `hook`；调用 `hook()` 阻塞等待完成 |

**来源**: `docs/08_test_barrier_engram_gate_analysis.md` L87-93

### 7.2 Engram 与 KV Cache 的关系

Engram 的设计目标是**跨节点参数/激活拉取**，典型用例包括：

1. **Remote KV Cache Fetching**：
   - 在 disaggregated inference 中，KV cache 可能存储在远程节点
   - Engram 允许直接通过 RDMA 读取远端 KV cache entries
   - 无需 CPU 介入，保持 GPU-centric

2. **Parameter Server 模式**：
   - 大模型参数存储在 CPU memory 或远程 GPU
   - Engram 支持"拉取-计算"模式

**存储布局**（`elastic.py` L409-435）：
```python
@staticmethod
def get_engram_storage_size_hint(num_entries, hidden, num_max_tokens_per_rank, dtype):
    # GPU buffer: 用于接收 fetched 数据
    num_gpu_bytes = align(hidden * dtype.itemsize * num_max_tokens_per_rank, 2MB)
    # CPU buffer: 用于本地 storage（远端可读取）
    num_cpu_bytes = align(hidden * dtype.itemsize * num_entries, 2MB)
    return num_gpu_bytes, num_cpu_bytes
```

**内存布局**（`buffer.hpp` L229-236）：
```
Hybrid Elastic Symmetric Memory:
[ GPU VRAM | CPU rank0 | CPU rank1 | ... | CPU rank(N-1) ]
                       ↑ Engram storage 写入这里
```

### 7.3 Engram 的 0 SM 特性

Engram 是 V2 中少数保留的 **0 SM** 功能：

```cpp
// buffer.hpp L292-309
launch_engram_fetch(
    nccl_context->dev_comm, nccl_context->window,
    math::advance_ptr(buffer, num_gpu_buffer_bytes),  // CPU segment
    fetched.data_ptr(),
    indices.data_ptr<int>(),
    ...
);
```

**机制**：
1. `engram_write` 将数据写入 CPU segment（`cudaMemcpyDeviceToDevice`）
2. `engram_fetch` 发起 **Gin RDMA get** 请求
3. 返回的 callable 在调用时等待完成
4. 数据传输由 **NIC hardware** 完成，**不占用 GPU SMs**

### 7.4 Engram 的 FP8 支持

```python
# elastic.py L579-602
def engram_write(self, storage, sf=None):
    """
    storage: [num_entries, hidden], torch.bfloat16 or torch.float8_e4m3fn
    sf: [num_total_entries, num_sf_packs], FP8 scaling factors
    """
```

**关键**：FP8 scaling factors 是**全局冗余**的（每个 rank 持有全部 SF），只有 main data 通过 RDMA 传输。

**来源**: `csrc/kernels/elastic/engram.hpp`, `csrc/elastic/buffer.hpp` L210-325, `tests/elastic/test_engram.py`

---

## 8. DeepSeek MoE 路由与两种模式的配合

### 8.1 DeepSeek-V3 的 Group-Limited Gating

DeepSeek-V3 采用 **group-limited gating** 策略：
- **256 experts** 分为 **32 groups**（每 group 8 experts）
- 每个 token 选择 **top-4 groups**
- 在每个选中 group 内选择 **top-2 experts**
- 最终 **top-8 experts** per token

**意义**：限制每个 token 只能访问 4 个 group → 减少跨节点通信量。

### 8.2 Normal 模式的优化

DeepEP V1 Normal 内核针对 group-limited gating 做了专门优化：

> "To align with the group-limited gating algorithm proposed in the DeepSeek-V3 paper, DeepEP V1 offers a set of kernels optimized for asymmetric-domain bandwidth forwarding, such as forwarding data from NVLink domain to RDMA domain."

**`docs/legacy.md` L9**

### 8.3 Decode 场景的路由缓存

**问题**：在 decode 阶段，如果连续 batch 的 routing decisions 相同，每次重新计算 layout 浪费。

**V2 解决方案**：`EPHandle` 缓存

```python
# 首次 decode step
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights, ...)

# 后续 step（routing 不变）
recv_x, _, _, handle, event = buffer.dispatch(x, handle=handle)
```

**缓存的内容**：
- `psum_num_recv_tokens_per_scaleup_rank`：每个 scaleup rank 接收的 token 数前缀和
- `psum_num_recv_tokens_per_expert`：每个 expert 接收的 token 数前缀和
- `recv_src_metadata`：源 token 元数据
- `dst_buffer_slot_idx`：目标 buffer slot 索引

### 8.4 Deterministic Sort

DeepEP 支持确定性路由（`deterministic=True`），通过 `EPHandle.deterministic_sort()` 对接收到的 token 排序，确保跨 rank 的一致性。

```python
# elastic.py L100-193
def deterministic_sort(self, do_cpu_sync, is_cached_dispatch,
                       recv_x, recv_sf, recv_topk_idx, recv_topk_weights,
                       channel_linked_list):
    # Non-expand mode: sort recv_x, recv_sf, recv_topk_weights, recv_topk_idx
    # Expand mode: only sort expanded arrays
```

**来源**: `DeepEP/docs/legacy.md` L9, `deep_ep/buffers/elastic.py` L100-193, `DeepEP/README.md` L271-329

---

## 9. 关键引用与源码索引

### 9.1 官方文档
- `DeepEP/README.md` — V2 官方文档，统一 API 示例
- `DeepEP/docs/legacy.md` — V1 官方文档，Normal/Low-Latency 分离 API
- `DeepEP/docs/nvshmem.md` — NVSHMEM 安装指南

### 9.2 Python 源码
- `deep_ep/__init__.py` — 包初始化，导出 `Buffer`, `ElasticBuffer`, `EPHandle`
- `deep_ep/buffers/legacy.py` — V1 Buffer 类（714 行）
- `deep_ep/buffers/elastic.py` — V2 ElasticBuffer 类（1108 行）
  - L25-98: EPHandle 定义
  - L100-193: deterministic_sort
  - L195-368: ElasticBuffer 类定义
  - L409-435: Engram storage size hint
  - L569-604: engram_write/engram_fetch
  - L728-834: get_theoretical_num_sms
  - L855-1033: dispatch 方法
  - L1046-1107: combine 方法

### 9.3 C++ 源码
- `csrc/elastic/buffer.hpp` — ElasticBuffer C++ 核心（1382 行）
  - L62-66: engram 状态
  - L210-240: engram_write
  - L242-325: engram_fetch
  - L586-686: buffer size 计算
  - L734: cached_mode 检测
- `csrc/kernels/elastic/dispatch.hpp` — Dispatch JIT runtime
- `csrc/kernels/elastic/engram.hpp` — Engram JIT runtime
- `csrc/kernels/backend/nccl.cu` — NCCL Gin 后端

### 9.4 Kernel 实现
- `deep_ep/include/deep_ep/impls/dispatch.cuh` — Direct dispatch kernel
- `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` — Hybrid dispatch kernel
- `deep_ep/include/deep_ep/impls/engram_fetch.cuh` — Engram fetch kernel
- `deep_ep/include/deep_ep/impls/combine.cuh` — Combine kernel

### 9.5 测试代码
- `tests/elastic/test_ep.py` — V2 EP 测试（含 cached dispatch）
- `tests/elastic/test_engram.py` — Engram 测试
- `tests/legacy/test_low_latency.py` — V1 low-latency 测试

### 9.6 已有研究报告
- `docs/10_deep_ep_v2_elastic_architecture.md` — V2 架构全面分析
- `docs/11_deep_ep_v2_deepdive_normal_lowlatency_sm_estimation.md` — SM 估算与统一接口
- `docs/08_test_barrier_engram_gate_analysis.md` — Engram 测试分析
- `docs/07_deep_ep_03_normal_vs_lowlatency.md` — Normal vs Low-Latency 源码分析
