# 05.10 Counting Sort / Bucketization 在 DeepGEMM Mega MoE 中的映射

> 分析日期: 2026-07-30
> 目标: 验证博客中 "Count → Prefix Sum → Scatter" 概念在 Mega MoE 中的具体实现位置与形态

---

## 1. 核心结论

**是的，Mega MoE 中存在完整的 Count → Prefix Sum → Scatter 流程**，但它以 **GPU 内联（inline）** 方式实现，而非 CPU 预处理。整个流程发生在 kernel 内部的 **dispatch warp** 中，通过 **两阶段原子操作** 完成从稀疏路由到连续 layout 的转换。

Mega MoE 的设计哲学与 DeepEP 的博客描述一致：
- **不是传统排序**，而是 Counting Sort / Bucketization
- **目标：产生连续 layout**，而非按大小排序
- **核心数据结构**：`smem_expert_count`（计数 + prefix sum）、`src_token_topk_idx`（scatter 索引）

---

## 2. 代码定位：Count → Prefix Sum → Scatter 在哪里？

整个流程位于 `sm100_fp8_fp4_mega_moe.cuh` 的 **dispatch warp 阶段**（`warp_idx < kNumDispatchWarps`），分为三步：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Dispatch Warp (warp_idx < kNumDispatchWarps)              │
│                                                                             │
│  Step 1: COUNT          Step 2: PREFIX SUM         Step 3: SCATTER          │
│  ┌──────────────┐       ┌───────────────────┐      ┌──────────────────┐     │
│  │ atomicAdd to │  ──▶  │ atomicAdd to global│ ──▶ │ atomicAdd to get │     │
│  │ smem_expert  │       │ expert_send_count  │      │ dst slot, write  │     │
│  │ _count[expert]│      │ → get global offset │      │ src_token_topk_idx│    │
│  └──────────────┘       └───────────────────┘      └──────────────────┘     │
│                                                                             │
│  Result: 每个 token-topk pair 被写入目标 rank 的 scatter 索引表              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Step 1: COUNT — 每个 SM 本地计数

**文件**: `sm100_fp8_fp4_mega_moe.cuh`，约 line 382-386

```cpp
// Count experts' tokens
read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
   atomicAdd_block(smem_expert_count + expert_idx, 1);
});
ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
```

**机制**：
- `smem_expert_count` 是一个 smem 数组，大小为 `kNumExperts`
- 每个 SM 的 dispatch warp 遍历本地 token 的 topk_idx
- 对每个 `(token, expert)` pair，执行 `atomicAdd_block(smem_expert_count + expert_idx, 1)`
- 结果：`smem_expert_count[i]` = 本 SM 发往 expert i 的 token 数量

**为什么用 smem 而非 global？**
- 减少 global memory 原子操作的竞争
- 每个 SM 先本地聚合，再与全局同步

### 2.2 Step 2: PREFIX SUM — 跨 SM 聚合得到全局偏移

**文件**: `sm100_fp8_fp4_mega_moe.cuh`，约 line 388-395

```cpp
// Get SM offset (~6.5 us)
#pragma unroll
for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
    const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
    smem_expert_count[i] = static_cast<uint32_t>(
        ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
}
ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);
```

**机制**：
- `expert_send_count` 是一个 global 数组（在 workspace 中），每个 entry 是 64-bit
- 高 32-bit 存储 "完成信号"（SM count），低 32-bit 存储 "累计 token 数"
- `atomic_add` 返回值 = 该 SM 在全局序列中的 **起始偏移**
- 结果写回 `smem_expert_count[i]`，此时它不再是计数，而是 **本 SM 在全局 scatter 中的起始 offset**

**这是经典的 parallel prefix sum 的两级实现**：
1. Level 1: SM 内部原子加（smem）
2. Level 2: SM 之间原子加（global）

### 2.3 Step 3: SCATTER — 写入源 token-topk 索引

**文件**: `sm100_fp8_fp4_mega_moe.cuh`，约 line 397-404

```cpp
// Write source indices (~2 us with 512 tokens)
read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
    const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
    const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
    const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
        expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
    *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
});
```

**机制**：
- 再次遍历本地 token 的 topk_idx
- `atomicAdd_block(smem_expert_count + expert_idx, 1)` 返回当前可用的 slot index
- 将 `token_topk_idx`（编码了 src_token_idx + src_topk_idx）写入目标 rank 的 `src_token_topk_idx` 表
- 写入通过 NVLink（`sym_buffer.map`）直达目标 rank 的对称内存

**Scatter 索引表结构**：
```
src_token_topk_idx[local_expert][src_rank][slot] = token_topk_idx
                                                    └── (src_token_idx * num_topk + src_topk_idx)
```

---

## 3. GPU vs CPU？Preprocessing vs Inline？

| 维度 | DeepEP (博客描述) | Mega MoE (实际实现) |
|------|------------------|---------------------|
| **执行位置** | GPU kernel 内 | GPU kernel 内（dispatch warp） |
| **时机** | Inline（与通信/计算融合） | Inline（与 pull GEMM 融合） |
| **预处理** | 无 CPU 预处理 | 无 CPU 预处理 |
| **排序算法** | Counting Sort | Counting Sort（两级原子） |
| **输出** | Expert Buffer（连续 layout） | Pool Buffer（连续 layout） |

**关键区别**：
- DeepEP 的 Counting Sort 产物是 **Expert GEMM 的输入 Buffer**
- Mega MoE 的 Counting Sort 产物是 **scatter 索引表**（`src_token_topk_idx`），后续的 "连续 layout" 由 **pull 阶段** 完成

---

## 4. 内核如何知道 token 去哪个 expert？

Mega MoE 的输入包含：

```python
# Python 侧 (tests/test_mega_moe.py)
buffer.topk_idx[:num_tokens].copy_(topk_idx)        # [num_tokens, num_topk]
buffer.topk_weights[:num_tokens].copy_(topk_weights) # [num_tokens, num_topk]
```

**数据流**：
```
Router 输出
    │
    ▼
topk_idx[token] = [expert_0, expert_1, ..., expert_k]   ← 每个 token 选择的 expert ID
topk_weights[token] = [w_0, w_1, ..., w_k]              ← 对应的 gate 权重
    │
    ▼
Kernel 内 dispatch warp 读取 topk_idx
    │
    ▼
Counting Sort → scatter 索引表写入目标 rank
    │
    ▼
Pull 阶段根据 scatter 索引表拉取 token
```

**关键洞察**：Kernel 不需要 "排序"，只需要知道每个 `(token, topk_slot)` 对的目标 expert。`topk_idx` 就是路由决策本身。

---

## 5. Bucketization：Token 分配到 Expert Buckets

**Bucketization 发生在两个层面**：

### 5.1 全局 Bucketization（跨 Rank）

```
                    Rank 0                          Rank 1
            ┌─────────────────┐             ┌─────────────────┐
            │ Expert 0: 5 tok │             │ Expert 2: 3 tok │
            │ Expert 1: 3 tok │             │ Expert 3: 7 tok │
            └─────────────────┘             └─────────────────┘
                     │                               │
                     └───────────┬───────────────────┘
                                 ▼
                     src_token_topk_idx[expert][rank][slot]
                                 │
                                 ▼
                    按 expert 聚合的全局 scatter 表
```

### 5.2 本地 Bucketization（Pool Buffer）

**文件**: `sm100_fp8_fp4_mega_moe.cuh`，约 line 572

```cpp
// Store token data
const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
```

**Pool Buffer 布局**：
```
┌────────────────────────────────────────────────────────────────┐
│                     l1_token_buffer (Pool)                      │
├──────────────────┬──────────────────┬──────────────────────────┤
│   Expert 0       │   Expert 1       │   Expert 2               │
│   [0..n0-1]      │   [n0..n0+n1-1]  │   [n0+n1..n0+n1+n2-1]   │
│                  │                  │                          │
│  expert_pool_    │  expert_pool_    │  expert_pool_            │
│  block_offset=0  │  block_offset=n0 │  block_offset=n0+n1      │
└──────────────────┴──────────────────┴──────────────────────────┘
```

**`expert_pool_block_offset` 的计算**（scheduler/mega_moe.cuh）：

```cpp
CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
    uint32_t num_blocks = 0;
    #pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
        if (i * 32 + ptx::get_lane_idx() < expert_idx)
            num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
    }
    return __reduce_add_sync(0xffffffff, num_blocks);
}
```

**这就是 Prefix Sum！** 对每个 expert 之前的 token 数做累加，得到该 expert 在 pool 中的起始 block offset。

---

## 6. 连续 Layout 的产生：从稀疏路由到连续访问

### 6.1 完整数据流

```
                        Rank 0 (Local)
                        ┌─────────────────────────────────────┐
                        │  x[token_idx] = fp8 token data      │
                        │  topk_idx[token] = [e2, e5, e1]     │
                        └──────────────┬──────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              expert=2            expert=5            expert=1
           ┌────────────┐    ┌────────────┐    ┌────────────┐
           │ Count += 1 │    │ Count += 1 │    │ Count += 1 │
           └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
                 │                 │                 │
                 ▼                 ▼                 ▼
           ┌─────────────────────────────────────────────────┐
           │         Prefix Sum (跨 SM 聚合)                  │
           │  smem_expert_count[i] → global offset           │
           └─────────────────────────────────────────────────┘
                                   │
                                   ▼
           ┌─────────────────────────────────────────────────┐
           │         Scatter (写入目标 rank)                  │
           │  src_token_topk_idx[expert][rank][slot] =       │
           │      token_topk_idx                             │
           └─────────────────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
        Rank 2 (Remote)                           Rank 0 (Local)
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │ Pull 阶段读取 scatter 表 │             │ Pull 阶段读取 scatter 表 │
    │ 按 expert 聚合 token     │             │ 按 expert 聚合 token     │
    │ 写入 pool buffer         │             │ 写入 pool buffer         │
    └─────────────────────────┘             └─────────────────────────┘
              │                                         │
              ▼                                         ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │ Pool Buffer:            │             │ Pool Buffer:            │
    │ Expert 0 | Expert 1 |.. │             │ Expert 0 | Expert 1 |.. │
    │ (连续 layout)            │             │ (连续 layout)            │
    └─────────────────────────┘             └─────────────────────────┘
              │                                         │
              └─────────────────┬───────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │  GEMM: 按 expert 切 block │
                    │  每个 expert 的 token 连续  │
                    │  访问 → Tensor Core 高效    │
                    └───────────────────────┘
```

### 6.2 Pull 阶段的连续化

**文件**: `sm100_fp8_fp4_mega_moe.cuh`，约 line 544-598

```cpp
// Read source token-topk index (written by remote dispatch via NVLink)
const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
    current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

// TMA load token from remote rank into shared memory
ptx::tma_load_1d(
    pull_buffer.get_base_ptr(),
    sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                   current_rank_in_expert_idx),
    pull_mbarrier, kHidden);

// Store token to local L1 buffer via TMA
const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
ptx::tma_store_1d(
    l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
    pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());
```

**这就是 "contiguous layout" 的最终产物**：
- 每个 expert 的 token 在 pool buffer 中连续排列
- `expert_pool_block_offset` 确保不同 expert 的 token 不重叠
- `token_idx_in_expert` 确保同一 expert 内的 token 紧凑排列

---

## 7. Preprocessing vs Inline Routing？

### 7.1 Mega MoE 是纯 Inline Routing

**证据 1**：Python 侧只准备 `topk_idx` 和 `topk_weights`：

```python
# tests/test_mega_moe.py, line 82-83
scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
```

没有任何预处理排序操作。`topk_idx` 直接拷贝进 symmetric buffer。

**证据 2**：Kernel 入口不做任何排序：

```cpp
// sm100_fp8_fp4_mega_moe.cuh, line 363-380
const auto read_topk_idx = [&](const auto& process) {
    for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
         i < num_tokens;
         i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
        if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
            expert_idx = static_cast<int>(
                __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
            if (expert_idx >= 0)
                process(i * kNumTopk + lane_idx, expert_idx);
        }
        __syncwarp();
    }
};
```

Kernel 直接读取原始 `topk_idx`，然后在 GPU 上完成 Counting Sort。

### 7.2 为什么不用 CPU 预处理？

1. **Data Locality**：`topk_idx` 已经在 GPU global memory 中
2. **避免 CPU-GPU 同步**：CPU 预处理需要 sync，破坏 pipeline
3. **与 NVLink 通信融合**：scatter 写入目标 rank 的对称内存，直接利用 NVLink
4. **SM 利用率**：dispatch warp 专门做 routing，与其他 warp 角色并行

---

## 8. 关键数据结构

### 8.1 Prefix Sum 数据结构

| 名称 | 位置 | 用途 |
|------|------|------|
| `smem_expert_count[kNumExperts]` | smem | 本地计数 → 全局偏移（复用） |
| `expert_send_count[kNumExperts]` | global workspace | 跨 SM 聚合（64-bit，高32位是 SM 计数） |
| `expert_recv_count[src_rank][local_expert]` | global workspace | 每个 rank 发给本 rank 某 expert 的 token 数 |
| `expert_recv_count_sum[local_expert]` | global workspace | 所有 rank 发给本 rank 某 expert 的总数 |

**`expert_send_count` 的 64-bit 编码**：
```
┌─────────────────────┬─────────────────────┐
│ 高 32-bit           │ 低 32-bit           │
│ SM 完成计数          │ 累计 token 数        │
│ (到达 kNumSMs*kNumRanks │ (prefix sum 结果)    │
│  表示全部完成)        │                     │
└─────────────────────┴─────────────────────┘
```

### 8.2 Scatter 数据结构

| 名称 | 位置 | 用途 |
|------|------|------|
| `src_token_topk_idx[local_expert][src_rank][slot]` | global workspace | 目标 rank 读取源 rank token 的索引表 |
| `token_src_metadata[pool_token_idx]` | global workspace | 记录 pool 中每个 token 的来源（用于 combine） |

**`src_token_topk_idx` 的值编码**：
```
token_topk_idx = src_token_idx * num_topk + src_topk_idx
                 │                │
                 │                └── 该 token 的第几个 topk 选择
                 └── 源 rank 中的 token 位置
```

**`TokenSrcMetadata` 结构**（layout/mega_moe.cuh）：
```cpp
struct TokenSrcMetadata {
    uint32_t rank_idx;   // 来源 rank
    uint32_t token_idx;  // 来源 token
    uint32_t topk_idx;   // 来源 topk slot
};
```

---

## 9. Scheduler 中的 Prefix Sum

**文件**: `scheduler/mega_moe.cuh`

Scheduler 是 "连续 layout" 的消费者。它通过读取 `expert_recv_count_sum` 来获取每个 expert 的 token 数，然后计算 pool block offset：

```cpp
CUTLASS_DEVICE void fetch_expert_recv_count() {
    // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
    #pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
        const auto expert_idx = i * 32 + ptx::get_lane_idx();
        uint64_t value = 0;
        if (expert_idx < kNumExpertsPerRank) {
            do {
                value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
            } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
        }
        stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
    }
    __syncwarp();
}
```

**注意等待条件**：`(value >> 32) != kNumSMs * kNumRanks`
- 高 32-bit 是 SM 完成计数
- 当所有 SM 都完成 dispatch 后，才开始调度 GEMM

### 9.1 Pool Block Offset 计算（Prefix Sum）

```cpp
CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
    uint32_t num_blocks = 0;
    #pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
        if (i * 32 + ptx::get_lane_idx() < expert_idx)
            num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
    }
    return __reduce_add_sync(0xffffffff, num_blocks);
}
```

**这是对 `stored_num_tokens_per_expert` 的 prefix sum**：
- 输入：每个 expert 的 token 数
- 输出：该 expert 在 pool 中的起始 block 位置
- 用途：GEMM 计算时确定 token 在 pool buffer 中的位置

---

## 10. 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Mega MoE Kernel                                     │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    Dispatch Warp (Counting Sort)                          │   │
│  │                                                                          │   │
│  │   topk_idx[token]                                                        │   │
│  │        │                                                                 │   │
│  │        ▼                                                                 │   │
│  │   ┌─────────┐    ┌─────────────┐    ┌──────────────────┐                │   │
│  │   │ COUNT   │──▶│ PREFIX SUM  │──▶│     SCATTER      │                │   │
│  │   │ (smem   │    │ (global     │    │ (写入目标 rank    │                │   │
│  │   │  atomic)│    │  atomic)    │    │  对称内存)        │                │   │
│  │   └─────────┘    └─────────────┘    └──────────────────┘                │   │
│  │        │                                      │                          │   │
│  │        ▼                                      ▼                          │   │
│  │   smem_expert_count[i]              src_token_topk_idx                   │   │
│  │   = 本 SM 发给 expert i 的 token 数   [expert][rank][slot]              │   │
│  │                                             │                            │   │
│  └─────────────────────────────────────────────┼────────────────────────────┘   │
│                                                │                                │
│                                                ▼                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    Pull Warp (连续化)                                     │   │
│  │                                                                          │   │
│  │   读取 scatter 表 → TMA pull token → 写入 pool buffer                    │   │
│  │                                                                          │   │
│  │   ┌────────────────────────────────────────────────────────────┐         │   │
│  │   │              Pool Buffer (连续 layout)                      │         │   │
│  │   ├────────────┬────────────┬────────────┬─────────────────────┤         │   │
│  │   │  Expert 0  │  Expert 1  │  Expert 2  │  ...                │         │   │
│  │   │  token0..n │  token0..m │  token0..k │                     │         │   │
│  │   └────────────┴────────────┴────────────┴─────────────────────┘         │   │
│  │                         ▲                                                │   │
│  │                         │ expert_pool_block_offset (prefix sum)          │   │
│  └─────────────────────────┼────────────────────────────────────────────────┘   │
│                            │                                                    │
│                            ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    GEMM Scheduler + MMA Warp                              │   │
│  │                                                                          │   │
│  │   按 expert 切 block，每个 expert 的 token 连续访问 → Tensor Core          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. 与 DeepEP 博客的对比

| 维度 | DeepEP 博客描述 | Mega MoE 实际实现 |
|------|----------------|-------------------|
| **Count** | `Count[expert]++` | `atomicAdd_block(smem_expert_count + expert_idx, 1)` |
| **Prefix Sum** | `prefix[expert] = sum(count[0..expert-1])` | 两级：SM 内 atomic + SM 间 atomic to global |
| **Scatter** | `dst = prefix[expert]++` | `atomicAdd_block(smem_expert_count + expert_idx, 1)` 返回 slot |
| **连续 layout** | Expert Buffer (Expert-major) | Pool Buffer (Expert-major, per-expert 连续) |
| **通信融合** | 分离的 Dispatch/Combine | 与 NVLink pull/store 融合 |
| **计算融合** | 分离的 Expert GEMM | 与 GEMM 在同一 kernel |

**关键演化**：
- DeepEP 的 Counting Sort 是 **通信准备**（为 NVLink/RDMA 准备数据）
- Mega MoE 的 Counting Sort 是 **计算准备**（为 Tensor Core 准备连续 layout）
- Mega MoE 的 Counting Sort 产物（scatter 索引表）**同时服务于通信和计算**

---

## 12. 核心发现总结

1. **Count → Prefix Sum → Scatter 存在于 Mega MoE**，位于 dispatch warp 的前三步
2. **完全在 GPU 上 inline 执行**，无 CPU 预处理，无 CPU-GPU 同步
3. **两级 Prefix Sum**：SM 内 smem atomic → SM 间 global atomic
4. **Bucketization 通过 scatter 索引表实现**：每个 `(token, topk)` pair 被写入目标 rank 的 expert-specific slot
5. **连续 layout 由 pull 阶段产生**：根据 scatter 索引表拉取 token 到 pool buffer，按 expert 连续排列
6. **Scheduler 也使用 prefix sum**：计算 `expert_pool_block_offset`，让 GEMM 直接访问连续 layout
7. **数据结构与博客一致**：`smem_expert_count`（计数+prefix sum）、`src_token_topk_idx`（scatter 索引）

---

## 附录：关键代码位置索引

| 代码位置 | 行号 | 功能 |
|----------|------|------|
| `sm100_fp8_fp4_mega_moe.cuh` | 382-386 | COUNT: `atomicAdd_block(smem_expert_count + expert_idx, 1)` |
| `sm100_fp8_fp4_mega_moe.cuh` | 388-395 | PREFIX SUM: `atomic_add(workspace.get_expert_send_count_ptr(i), send_value)` |
| `sm100_fp8_fp4_mega_moe.cuh` | 397-404 | SCATTER: `atomicAdd_block(smem_expert_count + expert_idx, 1)` 返回 slot |
| `sm100_fp8_fp4_mega_moe.cuh` | 572 | Pool layout: `pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert` |
| `sm100_fp8_fp4_mega_moe.cuh` | 539-542 | 读取 scatter 索引表: `get_src_token_topk_idx_ptr` |
| `scheduler/mega_moe.cuh` | 80-88 | `get_pool_block_offset`: prefix sum 计算 expert 起始 block |
| `scheduler/mega_moe.cuh` | 183-197 | `fetch_expert_recv_count`: 等待所有 SM 完成 dispatch |
| `layout/mega_moe.cuh` | 26-30 | `TokenSrcMetadata` 结构定义 |
| `layout/mega_moe.cuh` | 153-160 | `get_src_token_topk_idx_ptr` scatter 表访问 |
| `tests/test_mega_moe.py` | 82-83 | Router 输出 `topk_idx` 直接作为 kernel 输入 |
