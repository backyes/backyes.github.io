# DeepEP V2 深度深潜：统一接口、SM 估算本质与 TopK 复制机制

> **分析范围**：ElasticBuffer 统一 Normal/Low-Latency 的量化因素、SM 自动估算的数学本质、dispatch 内 TopK 复制过程
> 
> **核心源码**：
> - `deep_ep/buffers/elastic.py:728-834` — `get_theoretical_num_sms` 解析式 SM 计算
> - `deep_ep/include/deep_ep/impls/dispatch.cuh` — Direct dispatch kernel
> - `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` — Hybrid dispatch kernel
> - `deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh` — Copy epilogue
> - `csrc/elastic/buffer.hpp:980-1003` — dispatch 主流程

---

## 目录

1. [ElasticBuffer 为什么能统一 Normal 和 Low-Latency？](#1-elasticbuffer-为什么能统一-normal-和-low-latency)
2. [SM 自动估算的本质是什么？](#2-sm-自动估算的本质是什么)
3. [dispatch 内的 TopK 复制机制](#3-dispatch-内的-topk-复制机制)

---

## 1. ElasticBuffer 为什么能统一 Normal 和 Low-Latency？

### 1.1 V1 时代 Normal vs Low-Latency 的割裂

在 DeepEP V1 中，Normal 和 Low-Latency 是 **两套独立的实现**：

| 维度 | Normal | Low-Latency |
|------|--------|-------------|
| Kernel | `internode.cu` | `internode_ll.cu` |
| 通信路径 | 3-phase (Push → Forward → Reduce) | 直接 RDMA get |
| SM 使用 | 24+ | 较少 |
| Buffer 模型 | 5 层 buffer | 不同设计 |
| API | 独立调用 | 独立调用 |

### 1.2 V2 的统一设计哲学

V2 的核心洞察：**Normal 和 Low-Latency 的本质差异不是"通信协议"，而是"资源分配策略"**。

#### 统一的关键——Buffer 是一等公民

```python
# elastic.py:195-226
class ElasticBuffer:
    """
    The elastic communication buffer, which supports:
        - high-throughput expert-parallel all-to-all (dispatch and combine)
        - Engram (remote KV cache fetch)
        - pipeline-parallel send/recv (PP)
        - all-gather reduce-scatter (AGRS)
    "Elastic" refers to the flexibility of underlying memory
    """
```

**所有通信模式共享同一个 Buffer**，区别仅在于：
- **SM 数量**（Normal 多，Low-Latency 少）
- **QP 数量**（Normal 多，Low-Latency 少）
- **Warp 角色分配**（Normal 用 notify + dispatch warps，Low-Latency 可能只用 dispatch warps）

### 1.3 量化的系统影响因素

#### 关键因子 1：Token 复制量（Replicated Token Volume）

每个 token 需要被复制到多少个远端 rank？这是 **topk 和 expert 分布** 的函数：

```python
# elastic.py:770-772
def get_expected_topk(num_groups: int) -> float:
    """计算单个 token 期望去多少个不同的 rank（去重后）"""
    return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk) / math.comb(num_experts, num_topk))
```

**推导**：
- 总共有 `num_groups` 个 rank 组
- 每组有 `num_experts // num_groups` 个 expert
- 一个 token 选 `num_topk` 个 expert
- 期望去重的 rank 数 = `num_groups × P(至少一个 expert 在该组)`

**示例**：EP=256, experts=256, topk=8
- 每个 rank 1 个 expert
- 期望去重 rank 数 ≈ 8（因为 topk=8 很少碰撞）

**示例**：EP=256, experts=2048, topk=8
- 每个 rank 8 个 expert
- 期望去重 rank 数 ≈ 8 × (1 - C(2040,8)/C(2048,8)) ≈ 8 × 0.031 ≈ 0.25
- 大部分 token 都在本地 rank

#### 关键因子 2：HBM 读写带宽（SM 侧）

```python
# elastic.py:781-806
# Read tokens
sm_read += 1 / num_expected_topk  # 每个 token 只需要读一次（无论去多少个 rank）

if self.num_scaleout_ranks > 1:
    # Hybrid 模式
    sm_write += 1 / num_expected_topk  # Scaleup warps 写 send buffer
    sm_write += (1 / num_expected_topk) * (num_expected_scaleout_topk / self.num_scaleout_ranks)  # Local bypass
    sm_read += num_expected_scaleout_topk / num_expected_topk  # Forward warps 读
    sm_write += 1  # Forward warps 写 scaleup
else:
    # Direct 模式
    if self.num_rdma_ranks > 1:
        sm_write += 1 / num_expected_topk  # 写 send buffer
    sm_write += self.num_nvlink_ranks / self.num_ranks  # 发起 NVLink
```

**单位**：每 token 的 HBM 读写量（以 token 数据量为单位 1.0）

#### 关键因子 3：NVLink/RDMA 通信带宽（网络侧）

```python
# Direct 模式
nvlink_traffic += self.num_nvlink_ranks / self.num_ranks * (1 - 1 / self.num_nvlink_ranks)
rdma_traffic += (self.num_ranks - self.num_nvlink_ranks) / self.num_ranks

# Hybrid 模式
rdma_traffic += (1 / num_expected_topk) * (num_expected_scaleout_topk * (1 - 1 / self.num_scaleout_ranks))
nvlink_traffic += 1 - (1 / self.num_scaleup_ranks)
```

**单位**：每 token 的网络通信量（以 token 数据量为单位 1.0）

#### 关键因子 4：瓶颈识别

```python
# elastic.py:809-812
if self.num_scaleout_ranks > 1 and (rdma_traffic / rdma_gbs) > (nvlink_traffic / nvlink_gbs):
    bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
else:
    bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs
```

**瓶颈 = max(流量/带宽)**，找到时间上最受限的资源。

### 1.4 Normal vs Low-Latency 的统一解释

| 维度 | Normal 场景 | Low-Latency 场景 | 统一机制 |
|------|-------------|------------------|----------|
| **SM 数** | 多（64-160） | 少（4-8） | `num_sms` 参数控制 |
| **QP 数** | 多（SM×16） | 少（SM+1） | `get_theoretical_num_qps` 自动 |
| **prefer_overlap** | False（不重叠） | True（重叠计算） | `prefer_overlap_with_compute` |
| **瓶颈** | 通常 RDMA | 通常 NVLink | 自动识别 |
| **Buffer** | 相同 | 相同 | 同一 `ElasticBuffer` |

**在 V2 中，"Low-Latency" 不是独立模式，而是 Normal 在特定参数配置下的特例**：
- SM 数少（4-8）
- QP 数少
- 优先与计算重叠

---

## 2. SM 自动估算的本质是什么？

### 2.1 核心公式推导

```python
# elastic.py:818-825
num_sms = max(
    bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
    bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
)
num_sms = align(max(4, math.ceil(num_sms * 1.25)), 2)
```

#### 物理含义

```
num_sms = max(
    (网络瓶颈带宽 / 每token网络流量) × (每token HBM读量 / 每SM HBM读带宽),
    (网络瓶颈带宽 / 每token网络流量) × (每token HBM写量 / 每SM HBM写带宽)
)
```

**第一项**：为了满足网络读取需求，需要多少 SM 来提供 HBM 读数据
**第二项**：为了满足网络写入需求，需要多少 SM 来消费 HBM 写数据

#### 为什么是这个公式？

**Roofline 模型**：

```
时间 = max(网络时间, HBM时间)
     = max(流量/带宽, 数据量/(SM数 × 每SM带宽))
```

令 网络时间 = HBM时间（最优状态）：

```
流量 / 网络带宽 = 数据量 / (SM数 × 每SM带宽)
→ SM数 = (网络带宽 / 流量) × (数据量 / 每SM带宽)
```

### 2.2 为什么用 num_experts 和 num_topk，而不是卡数？

#### 关键洞察：SM 处理的是 Buffer，Buffer 是 Expert 粒度的

**用户的推测是正确的**：

> SM 要处理到所有专家/卡的连接，推测 deepep buffer 是一等公民，而 buffer 是 expert 专家粒度，所以要读写内存位置（buffer）数量是 expert 的函数，而不仅仅是卡数。

**源码验证**：

```python
# elastic.py:770-772
def get_expected_topk(num_groups: int) -> float:
    assert num_experts % num_groups == 0
    return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk) / math.comb(num_experts, num_topk))
```

**`num_expected_topk` 是 `num_experts` 和 `num_topk` 的函数**，而不是直接依赖卡数。

#### 为什么 Expert 数比卡数更本质？

**原因 1：Token 复制量取决于 Expert 分布，而非卡数**

考虑两种配置：
- 配置 A：EP=64, experts=64, topk=8 → 每卡 1 expert
- 配置 B：EP=64, experts=512, topk=8 → 每卡 8 expert

虽然卡数相同，但配置 A 每个 token 期望去 8 个不同 rank（去重后），配置 B 可能只有 1-2 个 rank。

**原因 2：Buffer 寻址是 Expert 粒度的**

```cpp
// dispatch.cuh:318-320
const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
```

**TopK idx 直接给出 expert index**，然后才转换为 rank index。SM 的负载取决于 **需要访问多少个不同的 expert → 多少个不同的 buffer slot**。

**原因 3：卡数的影响已隐式包含**

```python
# elastic.py:778
num_expected_topk = get_expected_topk(self.num_ranks)  # num_ranks = num_scaleout_ranks × num_scaleup_ranks
```

`num_ranks` 就是卡数（总 EP 数）。但 **卡数通过与 expert 分布的交互来影响流量**，而非独立变量。

### 2.3 TopK 的作用：Buffer 间数据搬移量

用户的第二个推测：

> SM 处理逻辑里，要针对 token level 的处理，token level 有一个 topk 的复制数量，采用了 topk 粒度的发送控制逻辑，也可能 buffer 粒度没有 topk 概念，但是 buffer 间的数据搬移量跟 topk 相关。

**源码验证**：

```cpp
// dispatch.cuh:315-328
EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
int stored_dst_rank_idx = -1;
if (lane_idx < kNumTopk) {
    const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
    const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
    stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
    tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
    if (topk_weights != nullptr)
        tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
}
```

**TopK 直接影响**：
1. **每个 token 需要发送多少次**：topk 个 expert → 去重后 N 个 rank → N 次发送
2. **Warp 内并行度**：topk ≤ 32（一个 warp 的 lane 数），每个 lane 处理一个 expert 选择
3. **Buffer slot 分配**：每个 topk 选择对应一个独立的 buffer slot

### 2.4 为什么不直接考虑卡数？

用户的第三个问题：

> 每卡处理负载应该与卡数有关，每卡上的专家数量跟卡数有关，为什么不考虑卡数这个概念，来估算 sms？

**回答**：卡数 **确实被考虑了**，但它是通过 `num_expected_topk` 间接体现的：

```python
# elastic.py:778
num_expected_topk = get_expected_topk(self.num_ranks)
```

**数学关系**：
- 固定 experts 和 topk 时，增加卡数 → 每卡 expert 数减少 → 去重后 rank 数增加 → 通信量增加 → 需要更多 SM
- 固定卡数和 topk 时，增加 experts → 每卡 expert 数增加 → 去重后 rank 数减少 → 通信量减少 → 需要更少 SM

**卡数本身不是独立变量**，因为：
- 如果 expert 数和卡数同比增加（保持每卡 expert 数不变），通信模式不变
- 真正决定通信量的是 **token 需要跨越多少边界**，这由 expert 分布决定

### 2.5 SM 估算的完整流程图

```
输入: num_experts, num_topk, num_ranks (卡数)
         ↓
    get_expected_topk(num_ranks)
         ↓
    num_expected_topk = num_ranks × (1 - C(E-E/R, k) / C(E, k))
         ↓
    ┌──────────────────────────────────────┐
    │  sm_read = 1 / num_expected_topk     │  (每 token HBM 读)
    │  sm_write = f(num_expected_topk, 卡数) │  (每 token HBM 写)
    │  rdma_traffic = g(卡数, num_expected_topk) │
    │  nvlink_traffic = h(卡数, num_expected_topk) │
    └──────────────────────────────────────┘
         ↓
    bottleneck = max(rdma_traffic/rdma_gbs, nvlink_traffic/nvlink_gbs)
         ↓
    num_sms = bottleneck × max(sm_read/sm_read_gbs, sm_write/sm_write_gbs)
         ↓
    num_sms = align(max(4, ceil(num_sms × 1.25)), 2)
```

---

## 3. dispatch 内的 TopK 复制机制

### 3.1 用户的三个问题

1. `dispatch_forward` 函数是否包含了 topk buffer 的生成？
2. Token 的复制过程，是否发生在 `buffer.dispatch` 内？
3. 是否利用了 `topk_idx` 函数？

### 3.2 答案：是的，全部在 buffer.dispatch 内完成

`buffer.dispatch()` 是一个 **多阶段 kernel pipeline**，包含：

| 阶段 | Kernel | 作用 |
|------|--------|------|
| 1. Notify | `dispatch_impl` (notify warps) | 统计 expert → rank 映射 |
| 2. Dispatch | `dispatch_impl` (dispatch warps) | 根据 topk_idx 复制 token 到 send buffer |
| 3. Barrier | Gin barrier | 等待所有数据到达 |
| 4. Epilogue | `dispatch_copy_epilogue_impl` | 从 buffer 搬到 recv tensor |

### 3.3 阶段详解

#### 阶段 1：Notify Warps 统计（`dispatch.cuh:79-258`）

```cpp
// dispatch.cuh:95-107
const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;
for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
    // 每个 token 的 topk 选择
    const auto dst_expert_idx = lane_idx < kNumTopk ?
        static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
    if (dst_expert_idx >= 0)
        atomicAdd_block(expert_count + dst_expert_idx, 1);  // Expert 维度计数

    // Rank 维度需要去重
    const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
    if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
        atomicAdd_block(rank_count + dst_rank_idx, 1);  // Rank 维度去重计数
}
```

**关键**：`ptx::deduplicate` 确保同一个 rank 只计数一次，即使多个 expert 都在该 rank。

#### 阶段 2：Dispatch Warps 复制 Token（`dispatch.cuh:259-395`）

```cpp
// dispatch.cuh:278-394
const auto token_start = dispatch_warp_idx * kNumSMs + sm_idx;
const auto token_stride = kNumDispatchWarps * kNumSMs;
for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {
    // 1. TMA load hidden data from global to shared memory
    ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), 
                     math::advance_ptr(x, token_i64_idx * kNumHiddenBytes), ...);
    
    // 2. Load top-k indices and weights
    if (lane_idx < kNumTopk) {
        const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
        const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
        stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
        tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;  // ← TopK idx 写入 smem buffer
        if (topk_weights != nullptr)
            tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
        if (copied_topk_idx != nullptr)
            copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;  // ← 复制 topk_idx
    }
    
    // 3. Deduplicate ranks and assign slots
    if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
        stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);
    
    // 4. Write dst_buffer_slot_idx (for later combine)
    if (lane_idx < kNumTopk) {
        const auto value = stored_dst_slot_idx >= 0 ?
            rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
        dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = value;  // ← 生成 slot idx
    }
    
    // 5. TMA store to send buffer (NVLink)
    const auto dst_ptr = stored_dst_slot_idx >= 0 ?
        gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), stored_dst_rank_idx) :
        nullptr;
    if (dst_ptr != nullptr)
        ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
    
    // 6. RDMA put to remote ranks
    if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
        gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
    }
}
```

**TopK 复制的核心机制**：

```
for each token:
    for each topk selection (lane_idx < kNumTopk):
        expert_idx = topk_idx[token_idx][lane_idx]
        rank_idx = expert_idx / experts_per_rank
        slot_idx = atomicAdd(counter[rank_idx], 1)  # 原子分配 slot
        
        # 写入 smem TMA buffer
        tma_buffer.topk_idx[lane_idx] = expert_idx
        tma_buffer.topk_weights[lane_idx] = topk_weights[token_idx][lane_idx]
        
        # 发送数据到远端
        if rank_idx is NVLink-accessible:
            TMA store to symmetric buffer
        else:
            RDMA put to remote buffer
```

#### 阶段 3：Copy Epilogue（`dispatch_copy_epilogue.cuh`）

```cpp
// dispatch_copy_epilogue.cuh:70-81
for (int i = global_warp_idx; i < num_recv_tokens; i += kNumWarps * kNumSMs) {
    // 确定当前 token 来自哪个 rank
    while (i >= current_rank_end) {
        current_rank_idx += 1;
        current_rank_start = current_rank_end;
        current_rank_end = psum_num_recv_tokens_per_scaleup_rank[current_rank_idx];
    }
    const auto buffer_token = scaleup_buffer.get_rank_buffer(current_rank_idx).get_token_buffer(i - current_rank_start);
    
    // TMA load from buffer to smem
    ptx::tma_load_1d(tma_buffer.get_base_ptr(), buffer_token.get_base_ptr(), ...);
    
    // 确定目标位置
    if (kDoExpand) {
        // Expand mode: 按 expert 顺序排列
        dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);
    } else {
        // Non-expand mode: 保持到达顺序
        dst_tensor_idx = i;
    }
    
    // TMA store to recv tensor
    ptx::tma_store_1d(math::advance_ptr(recv_x, dst_tensor_idx * kNumHiddenBytes),
                      tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
}
```

### 3.4 TopK Buffer 的生成时机

**TopK buffer 在 dispatch kernel 内部生成**：

```cpp
// dispatch.cuh:321-325
tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
if (topk_weights != nullptr)
    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
if (copied_topk_idx != nullptr)
    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
```

**TMA buffer 的内存布局（TokenLayout）**：

```
[hidden (TMA-aligned)] [sf (TMA-aligned)] [metadata (TMA-aligned)] [mbarrier]
                                    ↓
                             metadata = topk_idx (num_topk * 4) 
                                      + topk_weights (num_topk * 4)
                                      + src_token_global_idx (4) 
                                      + linked_list_idx (num_topk * 4)
```

### 3.5 完整数据流图

```
input: x[num_tokens, hidden], topk_idx[num_tokens, topk], topk_weights[num_tokens, topk]
                                    ↓
                    ┌───────────────────────────────────────┐
                    │         dispatch_impl kernel          │
                    │                                       │
                    │  Notify Warps:                        │
                    │    for each token:                    │
                    │      for each topk (lane):            │
                    │        expert = topk_idx[token][lane] │
                    │        rank = expert / E_per_rank     │
                    │        atomicAdd(expert_count, 1)     │
                    │        atomicAdd(rank_count, 1)  // dedup │
                    │    → prefix sum → psum_num_recv_tokens │
                    │                                       │
                    │  Dispatch Warps:                      │
                    │    for each token:                    │
                    │      TMA load hidden → smem          │
                    │      load topk_idx → smem            │  ← TopK buffer 生成
                    │      load topk_weights → smem        │
                    │      for each topk (lane):            │
                    │        slot = atomicAdd(counter, 1)   │  ← slot 分配
                    │        if NVLink: TMA store           │  ← Token 复制
                    │        else: RDMA put                 │  ← Token 复制
                    │    → trigger PDL (Programmatic Launch)│
                    └───────────────────────────────────────┘
                                    ↓
                    ┌───────────────────────────────────────┐
                    │    dispatch_copy_epilogue_impl kernel │
                    │                                       │
                    │    for each received token:           │
                    │      TMA load from buffer → smem     │
                    │      determine dst_tensor_idx:        │
                    │        if expand: atomicAdd(psum)     │
                    │        else: dst = arrival order      │
                    │      TMA store to recv_x[dst]        │
                    │      store recv_topk_idx[dst]        │
                    │      store recv_topk_weights[dst]    │
                    └───────────────────────────────────────┘
                                    ↓
output: recv_x[num_recv_tokens, hidden], recv_topk_idx, recv_topk_weights, handle
```

### 3.6 Hybrid Dispatch 的特殊性

在 Hybrid 模式下，token 复制分为两步：

1. **Scaleout Warps**：将 token 从本地 GPU → RDMA → 远端 scaleout rank 的 scaleout_recv_buffer
2. **Forward Warps**：将 token 从 scaleout_recv_buffer → NVLink → 目标 scaleup rank 的 scaleup_buffer

```cpp
// hybrid_dispatch.cuh:447-455 — Scaleout warps
if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx != scaleout_rank_idx) {
    gin.put<ncclTeamTagRail>(  // RDMA put to remote scaleout rank
        scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
        scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
        tma_buffer.get_num_bytes<false>(),
        stored_dst_scaleout_rank_idx,
        ncclGinOptFlagsAggregateRequests);
}

// hybrid_dispatch.cuh:593-598 — Forward warps
if (stored_dst_slot_idx >= 0) {
    const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(  // NVLink TMA store
        scaleup_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
        stored_dst_scaleup_rank_idx);
    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
}
```

---

## 总结

### Q1: ElasticBuffer 统一 Normal/Low-Latency 的量化因素

| 因子 | 公式 | 含义 |
|------|------|------|
| `num_expected_topk` | `num_ranks × (1 - C(E-E/R, k)/C(E, k))` | 每 token 期望去重 rank 数 |
| `sm_read` | `1 / num_expected_topk` | 每 token HBM 读量 |
| `sm_write` | `f(num_expected_topk, 卡数)` | 每 token HBM 写量 |
| `rdma_traffic` | `g(卡数, num_expected_topk)` | 每 token RDMA 流量 |
| `nvlink_traffic` | `h(卡数, num_expected_topk)` | 每 token NVLink 流量 |
| **瓶颈** | `max(rdma_traffic/rdma_gbs, nvlink_traffic/nvlink_gbs)` | 时间瓶颈 |

### Q2: SM 估算本质

```
num_sms = bottleneck_ratio × max(sm_read_per_token/sm_read_bw, sm_write_per_token/sm_write_bw)
```

**为什么用 experts/topk 而非卡数**：
- Buffer 寻址是 expert 粒度的
- Token 复制量取决于 expert 分布（topk 选择）
- 卡数通过 `num_expected_topk` 间接体现

### Q3: TopK 复制机制

- **是的**，`dispatch_forward`（即 `dispatch_impl`）包含 TopK buffer 生成
- **是的**，token 复制发生在 `buffer.dispatch` 内
- **是的**，利用 `topk_idx` 进行：
  1. 目标 rank 计算：`rank = expert_idx / experts_per_rank`
  2. Slot 分配：`atomicAdd(counter[rank], 1)`
  3. 数据路由：NVLink TMA store 或 RDMA put

---

> **分析方法论**：本报告基于源码实证，关键结论均有代码行号标注。核心洞察是：V2 通过将 Buffer 提升为一等公民、SM/QP 解析式计算、以及统一的 Kernel Pipeline，实现了 Normal 和 Low-Latency 的统一——差异仅在于参数配置，而非架构设计。
