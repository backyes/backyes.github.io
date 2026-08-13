# 05_05_Metadata System: Blog 概念到 Mega MoE 实现的映射

> 分析目标：将博客中提出的 **Layout Metadata (Where?)** 和 **Identity Metadata (Who?)** 两个概念模型，映射到 DeepGEMM Mega MoE 的实际实现中。

---

## 1. 概念回顾：Blog 中的 Metadata 模型

博客提出了两个**概念性**抽象（非 DeepEP 官方术语）：

| 类型 | 核心问题 | 包含信息 | 作用 |
|------|---------|---------|------|
| **Layout Metadata** | Where? | count, prefix sum, offset | 动态映射 → 连续地址 |
| **Identity Metadata** | Who? | token id, expert id, gate weight, top-k slot | Combine 时恢复语义 |

Blog 的经典流程：**Count → Prefix Sum → Scatter**：
```
Expert2: 3 tokens → offset=0
Expert3: 5 tokens → offset=3
dst = prefix[expert]++  // 完成连续写入
```

---

## 2. Mega MoE 的 Metadata 实现全景

Mega MoE 的实现与 Blog 概念模型**高度同构**，但采用了更适合 GPU 硬件的变体。核心区别：

| Blog 概念 | Mega MoE 实现 | 关键差异 |
|-----------|--------------|---------|
| Count | `atomicAdd_block(smem_expert_count + expert_idx, 1)` | 分 SM 局部计数 |
| Prefix Sum | `ptx::atomic_add(expert_send_count_ptr, packed_value)` | 全局原子加获取偏移 |
| Scatter | `atomicAdd_block(smem_expert_count + expert_idx, 1)` 获取 slot | 两阶段而非传统 prefix[expert]++ |
| Identity | `TokenSrcMetadata {rank_idx, token_idx, topk_idx}` | 显式结构体存储 |

---

## 3. Layout Metadata 的实现

### 3.1 核心数据结构

Mega MoE 的 Layout Metadata 分布在 workspace 中：

```cpp
// layout/mega_moe.cuh: Workspace 结构
struct Workspace {
    // Expert send count: 每个 expert 发送的 token 数（全局）
    uint64_t* get_expert_send_count_ptr(uint32_t expert_idx);

    // Expert recv count: 每个 rank 接收到的每个 expert 的 token 数
    uint64_t* get_expert_recv_count_ptr(uint32_t rank_idx, uint32_t expert_idx);

    // Expert recv count sum: 跨 rank 的 recv count 总和
    uint64_t* get_expert_recv_count_sum_ptr(uint32_t expert_idx);

    // L1 arrival count: 每个 pool block 的 token 到达计数
    uint32_t* get_l1_arrival_count_ptr(uint32_t pool_block_idx);

    // L2 arrival mask: 每个 pool block 的 K-block 到达位掩码
    uint64_t* get_l2_arrival_mask_ptr(uint32_t pool_block_idx);

    // Dispatch pulling: 源 token-topk 索引
    uint32_t* get_src_token_topk_idx_ptr(uint32_t expert_idx, uint32_t rank_idx, uint32_t token_idx);
};
```

### 3.2 Count: 按 Expert 计数 Token

每个 SM 的 dispatch warp 首先在 **shared memory** 中局部计数：

```cpp
// sm100_fp8_fp4_mega_moe.cuh: Count experts' tokens
read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
    atomicAdd_block(smem_expert_count + expert_idx, 1);
});
```

这里 `smem_expert_count` 是一个共享内存数组，大小为 `kNumExperts`。每个 SM 独立计数自己负责的 tokens。

### 3.3 Prefix Sum: 跨 SM 全局偏移

接下来，每个 SM 将自己的计数原子加到全局 `expert_send_count`，获取全局偏移：

```cpp
// sm100_fp8_fp4_mega_moe.cuh: Get SM offset (~6.5 us)
for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
    const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
    // 关键：atomic_add 返回旧值，相当于 prefix sum
    smem_expert_count[i] = static_cast<uint32_t>(
        ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
}
```

**关键洞察**：`send_value` 的高 32 位是 `1`（表示 count 加 1），低 32 位是 token 数量。`atomic_add` 返回旧值，这样：
- 每个 SM 获得一个**全局唯一的偏移区间**
- `smem_expert_count[i]` 被覆写为该 SM 在 expert `i` 中的起始偏移

### 3.4 Scatter: 写入源 Token-Topk 索引

然后再次使用 `atomicAdd_block` 获取 slot，写入 `src_token_topk_idx`：

```cpp
// sm100_fp8_fp4_mega_moe.cuh: Write source indices (~2 us with 512 tokens)
read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
    const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
    const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
    const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
        expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
    *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
});
```

**与传统 prefix[expert]++ 的区别**：
- Blog 模型：全局 prefix[expert]++（单阶段）
- Mega MoE：**两阶段**——先原子加获取全局偏移（跨 SM），再原子加获取局部 slot（SM 内）
- 这是因为多个 SM 同时写入，需要先协调全局区间，再在 SM 内部分配

### 3.5 Pool Block Offset: Expert 的连续地址空间

每个 expert 在 token pool 中有连续的区域：

```cpp
// 在 dispatch pulling 循环中
expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
```

Token 在 pool 中的地址：
```cpp
const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
```

这就是 Blog 中 **"dynamic mapping → contiguous addresses"** 的实现：
- **动态映射**：Router 输出的 `(token, expert)` 对是稀疏的
- **连续地址**：通过 `expert_pool_block_offset` 将同一 expert 的 tokens 连续存放

---

## 4. Identity Metadata 的实现

### 4.1 输入层 Identity: topk_idx 和 topk_weights

Python 层传入：
```python
# tests/test_mega_moe.py: 创建输入
scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
```

C++ 层从 symmetric buffer 切片：
```cpp
// csrc/apis/mega.hpp: 创建 tensor views
auto topk_idx = torch::from_blob(
    math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_idx_buffer.base)),
    {num_max_tokens_per_rank, num_topk},
    torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
auto topk_weights = torch::from_blob(
    math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_weights_buffer.base)),
    {num_max_tokens_per_rank, num_topk},
    torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
```

### 4.2 TokenSrcMetadata: 用于 Combine 写回

```cpp
// layout/mega_moe.cuh: 每个 token 的源 metadata
struct TokenSrcMetadata {
    uint32_t rank_idx;   // 源 rank
    uint32_t token_idx;  // 源 token 索引
    uint32_t topk_idx;   // 源 top-k 槽位
};
```

这个结构体在 **dispatch pulling** 阶段写入：
```cpp
// sm100_fp8_fp4_mega_moe.cuh: 写入源 metadata
*workspace.get_token_src_metadata_ptr(pool_token_idx) =
    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
```

在 **L2 epilogue** 阶段读出，用于将结果写回远程 rank：
```cpp
// sm100_fp8_fp4_mega_moe.cuh: 读取源 metadata 进行 combine
const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
const uint32_t dst_rank_idx = src_metadata.rank_idx;
const uint32_t dst_token_idx = src_metadata.token_idx;
const uint32_t dst_topk_idx = src_metadata.topk_idx;

// 写入远程 combine buffer
const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                       .get_data_buffer(dst_token_idx);
*sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
```

**这就是 Blog 中 Identity Metadata 的精确实现**：
- `rank_idx` + `token_idx` + `topk_idx` = "Who is this data?"
- Combine 时知道该把结果送回哪里

### 4.3 权重在 Identity 中的角色

权重 `topk_weights` 的处理分为两个阶段：

**Dispatch 阶段**：权重从远程 rank 拉取到本地 L1 buffer
```cpp
const auto weight = *sym_buffer.map(
    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
    current_rank_in_expert_idx);
*l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;
```

**L1 Epilogue 阶段**：权重在 SwiGLU 之后应用
```cpp
// L1 epilogue: 应用权重到 SwiGLU 输出
swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
```

**注意**：Mega MoE 将权重**提前应用到 L1 输出**，这样 Combine 阶段只需要做**累加**（sum），不需要再做加权求和。

---

## 5. topk_idx 如何决定 Expert 分配

### 5.1 Dispatch 阶段：读取 topk_idx

```cpp
// sm100_fp8_fp4_mega_moe.cuh: read_topk_idx lambda
const auto read_topk_idx = [&](const auto& process) {
    for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
         i < num_tokens;
         i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
        int expert_idx = -1;
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

**工作流程**：
1. 每个 warp 负责一批 tokens（跨 SM 交错分配）
2. 每个 lane 读取一个 `(token, topk_slot)` 对应的 `expert_idx`
3. 调用 `process(token_topk_idx, expert_idx)` 进行计数/scatter

### 5.2 无效 Expert 处理

测试代码中支持 mask 掉一些 expert 选择：
```python
# tests/test_mega_moe.py
if args.masked_ratio > 0:
    rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
    topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
    topk_weights.masked_fill_(topk_idx < 0, 0)
```

Kernel 中检查 `expert_idx >= 0` 跳过无效选择：
```cpp
if (expert_idx >= 0)
    process(i * kNumTopk + lane_idx, expert_idx);
```

---

## 6. topk_weights 如何用于 Combine

### 6.1 Mega MoE 的 Combine 策略

**关键设计决策**：Mega MoE 将权重**提前到 L1 epilogue 应用**，使得 Combine 阶段简化为纯累加。

**L1 Epilogue（权重应用）**：
```cpp
// L1 epilogue: silu(gate) * up * weights
swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
```

**Combine 阶段（纯累加）**：
```cpp
// Combine: 累加所有 top-k 贡献
while (do_reduce) {
    do_reduce = move_mask_and_load(load_stage_idx ^ 1);
    combine_load_barriers[load_stage_idx]->wait(combine_phase);
    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
    }
    combine_phase ^= load_stage_idx;
    load_stage_idx ^= 1;
}
```

**与 Blog 模型的对比**：
- Blog：`Token17 = 0.73 × Expert2 + 0.27 × Expert7`（Combine 时加权）
- Mega MoE：权重在 L1 epilogue 已应用，Combine 时直接 `sum(weighted_outputs)`

**优势**：Combine 阶段不需要再访问 topk_weights，减少内存带宽。

---

## 7. "prefix[expert]++" 风格 Scatter 的实现

### 7.1 没有传统的全局 prefix[expert]++

Mega MoE **没有**使用单一的全局 `prefix[expert]++`，而是采用了**两阶段**策略：

```
阶段 1: 局部计数 (SM 内)
    atomicAdd_block(smem_expert_count + expert_idx, 1)

阶段 2: 全局偏移 (跨 SM)
    ptx::atomic_add(expert_send_count_ptr, packed_value)
    → 返回旧值作为全局偏移

阶段 3: 局部 scatter (SM 内)
    atomicAdd_block(smem_expert_count + expert_idx, 1)
    → 获取 slot，写入 src_token_topk_idx
```

### 7.2 为什么不用单一 prefix[expert]++？

**原因**：多个 SM 同时写入全局计数器会产生严重的 contention：
- 80+ SM 同时 atomicAdd 同一个全局数组 → 序列化
- 两阶段方案让 SM 先在共享内存中局部竞争，再全局协调一次

**等价性**：数学上等价于全局 prefix sum，但**并行度更高**。

---

## 8. "Dynamic Mapping → Contiguous Addresses" 的实现

### 8.1 问题定义

Router 输出：稀疏的 `(token, expert)` 对
Expert GEMM 需要：连续的 `[M, K]` 矩阵（按 expert 组织）

### 8.2 Mega MoE 的解决方案

**Token Pool 架构**：
- 所有 local experts 共享一个连续的 token pool
- 每个 expert 占据 pool 中的一个连续子区域
- 偏移由 `expert_pool_block_offset` 确定

```
Token Pool Layout:
┌──────────────────────┬──────────────────────┬──────────────────────┐
│ Expert 0 Tokens      │ Expert 1 Tokens      │ Expert 2 Tokens      │
│ [0, e0_count)        │ [e0_count, e0+e1)    │ [e0+e1, e0+e1+e2)    │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

**地址计算**：
```cpp
// Pool token index = expert 起始偏移 + expert 内 token 索引
const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
```

### 8.3 与 DeepEP 的对比

| 方面 | DeepEP | Mega MoE |
|------|--------|---------|
| 中间 buffer | Dispatch Buffer → Receive Buffer → Expert Buffer | 单一 Token Pool |
| 布局变换 | Token-major → Dest-major → Expert-major | 直接在 pool 中按 expert 组织 |
| GEMM 输入 | 从 Expert Buffer 读取 | 从 Token Pool 直接 TMA load |

Mega MoE 的 pool 架构**消除了中间 buffer**，减少了内存占用和数据搬运。

---

## 9. Count → Prefix Sum → Scatter 的完整流程

### 9.1 流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Stage 1: Count (SM 内局部)                          │
│  for each (token, topk_slot):                                          │
│    expert_idx = topk_idx[token][slot]                                  │
│    atomicAdd_block(&smem_expert_count[expert_idx], 1)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    Stage 2: Prefix Sum (跨 SM 全局)                    │
│  for each expert i:                                                    │
│    packed = (1ull << 32) | smem_expert_count[i]  // 高32位=count增量   │
│    old = atomic_add(&global_send_count[i], packed)                     │
│    smem_expert_count[i] = old & 0xffffffff  // SM 的起始偏移            │
└─────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    Stage 3: Scatter (SM 内局部)                        │
│  for each (token, topk_slot):                                          │
│    expert_idx = topk_idx[token][slot]                                  │
│    slot_idx = atomicAdd_block(&smem_expert_count[expert_idx], 1)        │
│    dst = &src_token_topk_idx[expert_idx][rank][slot_idx]               │
│    *dst = token_topk_idx                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 代码对应

```cpp
// Stage 1: Count
read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
    atomicAdd_block(smem_expert_count + expert_idx, 1);
});

// Stage 2: Prefix Sum (跨 SM)
for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
    const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
    smem_expert_count[i] = static_cast<uint32_t>(
        ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
}

// Stage 3: Scatter
read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
    const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
    const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1);
    const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
        expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
    *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
});
```

---

## 10. Metadata 流：Python → C++ → CUDA Kernel

### 10.1 完整数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Python 层                                       │
│  topk_idx: [num_tokens, num_topk] int64                                 │
│  topk_weights: [num_tokens, num_topk] float32                           │
│                                                                          │
│  buffer.topk_idx[:num_tokens].copy_(topk_idx)                           │
│  buffer.topk_weights[:num_tokens].copy_(topk_weights)                   │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                          C++ API 层 (mega.hpp)                           │
│  slice_input_buffers(buffer) → (x, x_sf, topk_idx, topk_weights, ...)   │
│                                                                          │
│  fp8_fp4_mega_moe(y, l1_w, l2_w, sym_buffer, ...)                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                          CUDA Kernel 入口                                 │
│  sm100_fp8_fp4_mega_moe_impl<...>(                                      │
│      y, num_tokens, sym_buffer,                                         │
│      tensor_map_l1_acts, tensor_map_l1_acts_sf, ...)                    │
│                                                                          │
│  从 sym_buffer 计算各 buffer 基址:                                       │
│    input_topk_idx_buffer = sym_buffer + offset                           │
│    input_topk_weights_buffer = sym_buffer + offset                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                          Dispatch Warps                                  │
│  - 读取 topk_idx 进行计数和 scatter                                      │
│  - 通过 NVLink 拉取远程 token 和 weight                                  │
│  - 写入 TokenSrcMetadata                                                 │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                          Epilogue Warps                                  │
│  - L1: 应用 topk_weights 到 SwiGLU 输出                                 │
│  - L2: 读取 TokenSrcMetadata，写回远程 combine buffer                    │
│  - Combine: 累加 top-k 贡献到最终输出                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Buffer 布局计算

```cpp
// csrc/apis/mega.hpp: Buffer 布局
const auto input_token_buffer = layout::Buffer(
    fp8_token_layout, 1, num_max_tokens_per_rank, workspace.get_end_ptr());
const auto input_sf_buffer = layout::Buffer(
    fp8_sf_layout, 1, num_max_tokens_per_rank, input_token_buffer.get_end_ptr());
const auto input_topk_idx_buffer = layout::Buffer(
    input_topk_idx_layout, 1, num_max_tokens_per_rank, input_sf_buffer.get_end_ptr());
const auto input_topk_weights_buffer = layout::Buffer(
    input_topk_weights_layout, 1, num_max_tokens_per_rank, input_topk_idx_buffer.get_end_ptr());
```

### 10.3 Python → C++ 的桥接

```python
# deep_gemm/mega/__init__.py
def fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, ...):
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        sym_buffer.buffer,                    # 原始 buffer
        sym_buffer.handle.buffer_ptrs,        # 跨 rank 指针
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe, activation, activation_clamp, fast_math
    )
```

---

## 11. 关键洞察总结

### 11.1 概念映射表

| Blog 概念 | Mega MoE 实现 | 代码位置 |
|-----------|--------------|---------|
| **Layout Metadata** | `Workspace` 中的 count/recv_count/src_token_topk_idx | `layout/mega_moe.cuh` |
| Count | `atomicAdd_block(smem_expert_count + expert_idx, 1)` | dispatch warps L382-385 |
| Prefix Sum | `ptx::atomic_add(expert_send_count_ptr, packed_value)` | dispatch warps L389-394 |
| Scatter | `atomicAdd_block(smem_expert_count + expert_idx, 1)` + 写入 | dispatch warps L398-404 |
| **Identity Metadata** | `TokenSrcMetadata {rank_idx, token_idx, topk_idx}` | `layout/mega_moe.cuh:26-30` |
| Token id | `src_token_idx = src_token_topk_idx / kNumTopk` | dispatch pulling L541 |
| Expert id | `expert_idx` (from topk_idx) | dispatch warps L373 |
| Gate weight | `topk_weights_buffer` → `l1_topk_weights_buffer` | dispatch pulling L574-578 |
| Top-k slot | `src_topk_idx = src_token_topk_idx % kNumTopk` | dispatch pulling L542 |

### 11.2 架构创新点

1. **两阶段 Count-Scatter**：避免全局 contention
2. **Token Pool 架构**：消除中间 buffer，减少内存占用
3. **权重前移**：Combine 简化为纯累加
4. **显式 TokenSrcMetadata**：将 Identity Metadata 结构化存储

### 11.3 与 DeepEP 的演进

| 方面 | DeepEP | Mega MoE |
|------|--------|---------|
| 角色 | Data Movement Runtime | Communication + Compute Fusion |
| Metadata 存储 | 分散在多个 buffer | 集中在 Workspace + TokenSrcMetadata |
| 布局变换 | 多阶段、多 buffer | 单 pool、原地变换 |
| Combine 语义 | 加权求和 | 权重已应用的纯累加 |

---

## 12. 源码索引

| 文件 | 关键内容 |
|------|---------|
| `deep_gemm/mega/__init__.py` | Python 入口，topk_idx/topk_weights 传入 |
| `csrc/apis/mega.hpp` | C++ API，buffer 切片，metadata 布局计算 |
| `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | JIT 编译入口，TMA descriptor 创建 |
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` | 核心 kernel，metadata 读写逻辑 |
| `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` | Workspace 定义，TokenSrcMetadata 结构 |
| `tests/test_mega_moe.py` | 测试代码，metadata 生成和传递 |

---

## 13. 附录：关键代码片段

### A. TokenSrcMetadata 定义
```cpp
// layout/mega_moe.cuh
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};
```

### B. Dispatch 写入 Identity Metadata
```cpp
// sm100_fp8_fp4_mega_moe.cuh:590-591
*workspace.get_token_src_metadata_ptr(pool_token_idx) =
    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};
```

### C. Combine 读取 Identity Metadata
```cpp
// sm100_fp8_fp4_mega_moe.cuh:1183-1186
const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
const uint32_t dst_rank_idx = src_metadata.rank_idx;
const uint32_t dst_token_idx = src_metadata.token_idx;
const uint32_t dst_topk_idx = src_metadata.topk_idx;
```

### D. 权重在 L1 Epilogue 的应用
```cpp
// sm100_fp8_fp4_mega_moe.cuh:1011
swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights);
```

### E. Combine 纯累加
```cpp
// sm100_fp8_fp4_mega_moe.cuh:1315-1321
for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
    ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
```

---

*分析基于 DeepGEMM Mega MoE 源码，与博客概念模型对照。*
