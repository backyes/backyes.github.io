# Chunk Streaming 概念在 DeepGEMM Mega MoE 中的映射分析

> 分析日期: 2026-07-30
> 分析目标: 对比 DeepEP 博客中描述的 "Chunk Streaming" 概念与 Mega MoE 实际实现的差异
> 源码版本: DeepGEMM (Blackwell SM100)

---

## 1. 核心结论摘要

**Mega MoE 不使用 DeepEP 式的 "Chunk" 概念，而是通过 Symmetric Memory 实现了 "Token 即通信粒度" 的零拷贝直接访问。**

| 维度 | DeepEP (Normal Kernel) | Mega MoE |
|------|------------------------|----------|
| 调度粒度 (Scheduling Granularity) | Token | Token |
| 通信粒度 (Communication Granularity) | Chunk (多 Token 聚合) | **Token (单 Token 直接访问)** |
| 聚合机制 | Token Stream → Chunk → Network | Symmetric Memory 直接读写 |
| 流式行为 | Chunk Streaming (整 Chunk 流动) | Token Streaming (单 Token 按需拉取) |
| 等待策略 | 等待 Chunk 填满 | 等待 BLOCK_M 个 Token 到达即计算 |

---

## 2. DeepEP 的 Chunk Streaming 模型回顾

### 2.1 核心概念

根据博客描述，DeepEP Normal Kernel 的数据路径为：

```
Token Buffer → Dispatch Buffer → Chunk Buffer → NVLink/RDMA Pipeline → Receive Buffer → Expert Buffer
```

关键定义：
- **Token 是调度粒度** (scheduling granularity): Router 的输出单位
- **Chunk 是通信粒度** (communication granularity): 网络传输的单位
- **Chunk Streaming**: Chunk 流过网络，无需等待整个 Batch

### 2.2 为什么需要 Chunk？

> "The network is unsuitable for single-Token sends — produces small packets, high startup overhead, low bandwidth utilization. Tokens are aggregated: Token Stream → Chunk → Network Transfer"

核心原因：
1. **小包开销**: 单 Token 发送产生小 packet，startup overhead 高
2. **带宽利用率**: 小 packet 无法充分利用网络带宽
3. **聚合收益**: 将多个 Token 聚合成 Chunk 后发送，摊薄 startup 开销

---

## 3. Mega MoE 的通信模型

### 3.1 Symmetric Memory: 消除 Chunk 需求的基石

Mega MoE 使用 `torch.distributed._symmetric_memory` 创建跨 GPU 可直接访问的对称内存缓冲区：

```python
# deep_gemm/mega/__init__.py
class SymmBuffer:
    def __init__(self, group, num_experts, num_max_tokens_per_rank, ...):
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
```

**Symmetric Memory 的本质**: 它不是传统的 "发送-接收" 模型，而是让所有 GPU 通过 NVLink 直接读写同一块逻辑内存。这意味着：
- **无需 packet 聚合**: 每次读写可以直接以 Token 为单位
- **无 startup overhead**: NVLink 直接内存访问没有 per-packet 启动开销
- **带宽由 TMA 硬件保证**: Tensor Memory Access 硬件控制器处理数据传输

### 3.2 Token 级直接拉取 (Token-Level Pull)

Mega MoE 的 dispatch 阶段以 **单个 Token** 为粒度从远端拉取数据：

```cpp
// sm100_fp8_fp4_mega_moe.cuh - Dispatch Warp 主循环
constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;
for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; 
     ; token_idx += kNumGlobalWarps) {
    // ... 确定当前 token 属于哪个 expert、来自哪个 rank ...
    
    // TMA load token from remote rank into shared memory
    if (cute::elect_one_sync()) {
        ptx::tma_load_1d(
            pull_buffer.get_base_ptr(),
            sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                           current_rank_in_expert_idx),
            pull_mbarrier, kHidden);  // 一次拉取一个 Token (kHidden bytes)
    }
    
    // Store token to local L1 buffer via TMA
    ptx::tma_store_1d(
        l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
        pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());
}
```

**关键发现**: 循环变量 `token_idx` 每次递增 1，每次 TMA 操作传输 `kHidden` 字节（一个 Token）。这是 **Token 级通信粒度**。

---

## 4. 调度粒度 vs 通信粒度

### 4.1 DeepEP 的分离模型

```
┌─────────────────────────────────────────────────────────────┐
│ DeepEP Normal Kernel                                        │
│                                                             │
│  Token Stream ──[聚合]──> Chunk ──[网络]──> Chunk Buffer    │
│  (调度粒度)      聚合      (通信粒度)    传输                │
│                                                             │
│  • Token: 调度器决策单位                                    │
│  • Chunk: 网络传输单位 (多 Token 打包)                      │
│  • 需要 Chunk Buffer 做聚合/解聚                            │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Mega MoE 的统一模型

```
┌─────────────────────────────────────────────────────────────┐
│ Mega MoE                                                    │
│                                                             │
│  Token ──[Symmetric Memory 直接读写]──> Local Pool Buffer   │
│  (调度粒度 = 通信粒度)                                      │
│                                                             │
│  • Token: 既是调度单位，也是通信单位                        │
│  • 无需聚合/解聚缓冲区                                      │
│  • TMA 硬件保证传输效率                                     │
└─────────────────────────────────────────────────────────────┘
```

**核心差异**: Symmetric Memory 消除了 "聚合" 的必要性。NVLink 的 TMA 引擎可以高效处理小至 128B (一个 FP8 Token) 的传输，因为：
- NVLink 的 burst 传输效率不依赖 packet 大小
- TMA 硬件自动处理地址计算和传输调度
- 多个 TMA 请求可以流水线化执行

---

## 5. 可变 Token 数处理

### 5.1 每个 Expert 的 Token 数动态变化

Mega MoE 通过 `Workspace` 和 `MegaMoEScheduler` 处理每个 Expert 接收到的可变 Token 数：

```cpp
// scheduler/mega_moe.cuh - 等待所有 Expert 计数器就绪
CUTLASS_DEVICE void fetch_expert_recv_count() {
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

### 5.2 调度器按 Expert 分配 Block

```cpp
// scheduler/mega_moe.cuh - 核心调度逻辑
CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
    return math::ceil_div(current_num_tokens, BLOCK_M);
}

CUTLASS_DEVICE bool fetch_next_l1_block() {
    while (current_local_expert_idx < wave_end_expert_idx) {
        const auto num_m_blocks = get_current_num_m_blocks();
        m_block_idx = block_idx / kNumL1BlockNs;
        if (m_block_idx < num_m_blocks)
            return true;
        // 当前 Expert 已分配完，移动到下一个 Expert
        block_idx -= num_m_blocks * kNumL1BlockNs;
        advance_expert_idx();
    }
    return false;
}
```

**关键机制**:
- `stored_num_tokens_per_expert[i]`: 缓存每个 Expert 的 Token 数
- `get_current_num_m_blocks()`: 计算当前 Expert 需要多少个 M 方向 Block
- `advance_expert_idx()`: 当一个 Expert 处理完毕，移动到下一个 Expert

### 5.3 有效 M 维度处理 (非对齐 Token 数)

```cpp
// scheduler/mega_moe.cuh
template <bool kDoUMMAAligned = false>
CUTLASS_DEVICE uint32_t get_valid_m() const {
    const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
    return kDoUMMAAligned ? math::align(m, 16u) : m;
}
```

当最后一个 Block 的 Token 数不足 `BLOCK_M` 时，`get_valid_m()` 返回实际的有效 Token 数，MMA 指令会据此调整。

---

## 6. 调度单元 (Scheduling Unit)

### 6.1 Mega MoE 的三级调度

```
┌──────────────────────────────────────────────────────────────────┐
│                    Mega MoE 调度层次                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 1: Expert 级 (Wave)                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Wave = kNumExpertsPerWave 个 Expert 一组               │    │
│  │  先执行 Wave 内所有 Expert 的 L1，再执行 L2             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Level 2: Block 级 (CTA)                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  每个 CTA 处理一个 (expert, m_block, n_block) 组合      │    │
│  │  block_idx 以 kNumSMs 步长在 Block 间跳跃               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Level 3: Token 级 (Dispatch)                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Dispatch Warp 以 Token 为单位轮询各 Rank 的数据         │    │
│  │  token_idx += kNumGlobalWarps (跨 SM 协作)              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 调度器状态机

```cpp
// scheduler/mega_moe.cuh - 核心状态机
CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
    while (true) {
        if (current_local_expert_idx >= kNumExpertsPerRank)
            break;

        if (next_phase == BlockPhase::Linear1) {
            if (fetch_next_l1_block()) {
                n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                block_idx += kNumSMs;  // 步长为 SM 数量
                return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
            } else {
                // L1 完成，切换到 L2
                next_phase = BlockPhase::Linear2;
                set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
            }
        } else {
            if (fetch_next_l2_block()) {
                n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                block_idx += kNumSMs;
                return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
            } else {
                // L2 完成，切换到下一 Wave 的 L1
                next_phase = BlockPhase::Linear1;
            }
        }
    }
    return {BlockPhase::None, 0, 0, 0};
}
```

**调度单元总结**:
- **Dispatch 阶段**: Token 是基本单位 (每个 Warp 处理若干 Token)
- **Compute 阶段**: (Expert, M_Block, N_Block) 是基本单位 (每个 CTA 处理一个 Block)
- **Combine 阶段**: Token 是基本单位 (每个 Warp 处理若干 Token 的 top-k reduce)

---

## 7. "Flow Through Without Waiting" 流式行为

### 7.1 DeepEP 的 FIFO 流式

博客描述 DeepEP 通过 FIFO 实现流式：

```
Without FIFO: Send → Barrier → Forward → Barrier → Receive
With FIFO:    Send → FIFO → Forward → FIFO → Receive
```

### 7.2 Mega MoE 的 Arrival Counter 流式

Mega MoE 使用 **arrival counter** 机制实现类似的流式行为：

```cpp
// sm100_fp8_fp4_mega_moe.cuh - Dispatch 阶段写入后通知
ptx::red_add_rel(
    workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
```

```cpp
// GEMM TMA Load Warp - 等待 Token 到达后立即计算
if (block_phase == sched::BlockPhase::Linear1) {
    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
    const auto expected = scheduler.template get_valid_m<false>();
    while (ptx::ld_acq(ptr) != expected);  // 自旋等待 BLOCK_M 个 Token 到达
}
```

**流式行为对比**:

| 特性 | DeepEP | Mega MoE |
|------|--------|----------|
| 等待单位 | Chunk | BLOCK_M 个 Token |
| 通知机制 | FIFO 信号 | Arrival Counter |
| 触发条件 | Chunk 填满 | BLOCK_M 个 Token 到达 |
| 计算启动 | 收到完整 Chunk | 收到 BLOCK_M 个 Token |

### 7.3 L1 → L2 的流式衔接

```cpp
// L1 Epilogue 完成后通知 L2 可以开始
if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
    ptx::red_or_rel_gpu(
        workspace.get_l2_arrival_mask_ptr(pool_block_idx),
        1ull << n_block_idx  // 标记该 N block 的数据已就绪
    );
}

// L2 等待 L1 输出就绪
if (block_phase == sched::BlockPhase::Linear2) {
    const uint64_t needed = 3ull << (k_block_idx * 2);
    if ((cached_l2_arrival_mask & needed) != needed) {
        const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
        do {
            cached_l2_arrival_mask = ptx::ld_acq_gpu(ptr);
        } while ((cached_l2_arrival_mask & needed) != needed);
    }
}
```

**关键设计**: L2 不是等待整个 Expert 的所有 L1 输出，而是等待 **当前 K block 所需的 2 个 L1 输出 Block** (因为 SwiGLU 后 N 维度减半)。

---

## 8. Combine 阶段的 Chunk 概念

### 8.1 Mega MoE 中唯一的 "Chunk"

Mega MoE 代码中确实存在 `chunk` 概念，但仅用于 **Combine 阶段的 Hidden 维度分块**：

```cpp
// sm100_fp8_fp4_mega_moe.cuh - Combine 阶段
// 3 slots of chunk is needed: 2 load stages and 1 store
constexpr uint32_t kNumChunkSlots = 3;

// NOTES: either 1 or 2 chunks for simplicity
constexpr uint32_t kNumChunks =
    kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE 
    and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
```

### 8.2 Combine 的 Token 级处理

```cpp
// Combine: 每个 Warp 处理一个 Token 的 top-k reduce
for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
     token_idx < num_tokens;
     token_idx += kNumSMs * kNumEpilogueWarps) {
    
    // 遍历所有 chunk (Hidden 维度分块)
    for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
        // 遍历 top-k，加载并累加
        while (do_reduce) {
            do_reduce = move_mask_and_load(load_stage_idx ^ 1);  // 预取下一个
            combine_load_barriers[load_stage_idx]->wait(combine_phase);
            // 累加到寄存器
            for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
            }
        }
        // 写回
        ptx::tma_store_1d(
            math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
            combine_store_buffer, kNumChunkBytes);
    }
}
```

**这里的 Chunk 含义**: 将 Hidden 维度分成 1~2 块，以解决 Shared Memory 和 Register 的容量限制。这与 DeepEP 的 "通信 Chunk" 完全不同。

---

## 9. 架构对比图

### 9.1 DeepEP Chunk Streaming 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DeepEP Normal Kernel                                │
│                                                                         │
│  GPU 0                    GPU 1                    GPU 2                │
│  ┌─────────┐              ┌─────────┐              ┌─────────┐         │
│  │Token Buf│              │Token Buf│              │Token Buf│         │
│  │ T0 T1 T2│              │ T3 T4 T5│              │ T6 T7 T8│         │
│  └────┬────┘              └────┬────┘              └────┬────┘         │
│       │                        │                        │              │
│       ▼                        ▼                        ▼              │
│  ┌─────────┐              ┌─────────┐              ┌─────────┐         │
│  │Dispatch │              │Dispatch │              │Dispatch │         │
│  │  Buffer │              │  Buffer │              │  Buffer │         │
│  └────┬────┘              └────┬────┘              └────┬────┘         │
│       │                        │                        │              │
│       ▼                        ▼                        ▼              │
│  ╔═════════╗              ╔═════════╗              ╔═════════╗         │
│  ║ Chunk   ║              ║ Chunk   ║              ║ Chunk   ║         │
│  ║ Buffer  ║              ║ Buffer  ║              ║ Buffer  ║         │
│  ║(聚合N个 ║              ║(聚合N个 ║              ║(聚合N个 ║         │
│  ║ Token)  ║              ║ Token)  ║              ║ Token)  ║         │
│  ╚════╤════╝              ╚════╤════╝              ╚════╤════╝         │
│       │                        │                        │              │
│       ▼                        ▼                        ▼              │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │              NVLink / RDMA Network Fabric                   │       │
│  │              (Chunk 级传输)                                  │       │
│  └─────────────────────────────────────────────────────────────┘       │
│       │                        │                        │              │
│       ▼                        ▼                        ▼              │
│  ┌─────────┐              ┌─────────┐              ┌─────────┐         │
│  │Receive  │              │Receive  │              │Receive  │         │
│  │ Buffer  │              │ Buffer  ║              │ Buffer  ║         │
│  └────┬────┘              └────┬────┘              └────┬────┘         │
│       ▼                        ▼                        ▼              │
│  ┌─────────┐              ┌─────────┐              ┌─────────┐         │
│  │Expert   │              │Expert   │              │Expert   │         │
│  │ Buffer  │              │ Buffer  │              │ Buffer  │         │
│  └─────────┘              └─────────┘              └─────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Mega MoE Symmetric Memory 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Mega MoE (Blackwell SM100)                          │
│                                                                         │
│  GPU 0                    GPU 1                    GPU 2                │
│  ┌─────────┐              ┌─────────┐              ┌─────────┐         │
│  │Input    │              │Input    │              │Input    │         │
│  │Token Buf│              │Token Buf│              │Token Buf│         │
│  │ T0 T1 T2│              │ T3 T4 T5│              │ T6 T7 T8│         │
│  └────┬────┘              └────┬────┘              └────┬────┘         │
│       │                        │                        │              │
│       │    ┌───────────────────┴───────────────────┐    │              │
│       │    │                                       │    │              │
│       │    │     Symmetric Memory (NVLink)         │    │              │
│       │    │     Token 级直接访问                   │    │              │
│       │    │                                       │    │              │
│       │    │  ┌─────────────────────────────────┐  │    │              │
│       │    │  │  SymBuffer (All-Gather 视图)    │  │    │              │
│       │    │  │  Rank0 │ Rank1 │ Rank2 │ ...    │  │    │              │
│       │    │  └─────────────────────────────────┘  │    │              │
│       │    │                                       │    │              │
│       │    └───────────────────┬───────────────────┘    │              │
│       │                        │                        │              │
│       ▼                        ▼                        ▼              │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                    SM 0 (Compute)                           │       │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐              │       │
│  │  │Dispatch   │  │ GEMM      │  │ Combine   │              │       │
│  │  │Warp Group │  │ Warp Group│  │ Warp Group│              │       │
│  │  │           │  │           │  │           │              │       │
│  │  │Token级    │  │Block级    │  │Token级    │              │       │
│  │  │TMA Pull   │  │MMA Compute│  │TMA Store  │              │       │
│  │  └───────────┘  └───────────┘  └───────────┘              │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                         │
│  调度粒度 = 通信粒度 = Token                                            │
│  (Symmetric Memory 消除了聚合需求)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. 关键问题回答

### Q1: Mega MoE 是否使用 Chunking？代码中是否有 "chunk" 概念？

**答案**: 
- **通信阶段**: 不使用。Mega MoE 的 dispatch/combine 以 Token 为单位通过 Symmetric Memory 直接访问。
- **Combine 阶段**: 有 `kNumChunks` 概念，但这是将 Hidden 维度分块以适应 Shared Memory 容量限制，与 DeepEP 的通信 Chunk 完全不同。

### Q2: 内核如何处理每个 Expert 的可变 Token 数？

**答案**: 通过以下机制：
1. `fetch_expert_recv_count()`: 等待所有 Rank 的 Expert 计数器就绪
2. `stored_num_tokens_per_expert[]`: 缓存每个 Expert 的 Token 数
3. `get_current_num_m_blocks()`: 动态计算每个 Expert 需要的 M Block 数
4. `get_valid_m()`: 处理最后一个 Block 的非对齐 Token 数

### Q3: 是否存在 "调度粒度" vs "通信粒度" 的区分？

**答案**: **不存在**。这是 Mega MoE 与 DeepEP 的核心差异：
- DeepEP: Token (调度) ≠ Chunk (通信)，需要聚合
- Mega MoE: Token (调度) = Token (通信)，Symmetric Memory 允许 Token 级直接访问

### Q4: 内核如何处理 Token — 逐个还是分批/分块？

**答案**: 分阶段不同：
- **Dispatch**: 逐个 Token 处理 (token_idx 递增 1)
- **Compute**: 按 Block 处理 (BLOCK_M 个 Token 一组)
- **Combine**: 逐个 Token 处理 (token_idx 递增 1)

### Q5: Symmetric Memory 是否消除了对 Chunking 的需求？

**答案**: **是的**。这是核心洞察：
- DeepEP 需要 Chunk 是因为传统 RDMA/NVLink 网络的小包传输效率低
- Mega MoE 的 Symmetric Memory 通过 NVLink 直接内存访问，TMA 硬件可以高效处理任意大小的传输
- 因此 "Token 即通信粒度" 成为可能，无需聚合

### Q6: Mega MoE 的 "调度单元" 是什么？

**答案**: 三级调度单元：
1. **Expert Wave**: 宏观调度，决定哪些 Expert 一起处理
2. **CTA Block**: 中观调度，每个 CTA 处理一个 (expert, m_block, n_block)
3. **Token**: 微观调度，Dispatch/Combine Warp 以 Token 为单位工作

### Q7: 内核如何处理 "flow through without waiting" 流式行为？

**答案**: 通过 Arrival Counter 机制：
1. Dispatch Warp 每写入一个 Token 到 Local Pool，执行 `red_add_rel` 增加 arrival count
2. GEMM Warp 自旋等待 `arrival_count == BLOCK_M` (即一个 M Block 的 Token 全部到达)
3. L1 Epilogue 完成后通过 `red_or_rel_gpu` 设置 L2 arrival mask
4. L2 GEMM 等待所需的 L1 输出 Block 就绪后立即开始

**与 DeepEP 的区别**:
- DeepEP: 等待 Chunk 填满 → 转发 → 接收
- Mega MoE: 等待 BLOCK_M 个 Token 到达 → 开始 GEMM

---

## 11. 设计哲学对比

### 11.1 DeepEP: 通信优化范式

```
问题: 网络不适合小 packet 传输
方案: 聚合 Token → Chunk → 传输
本质: 适配网络硬件特性
```

### 11.2 Mega MoE: 计算-通信融合范式

```
问题: 如何最大化计算效率
方案: Symmetric Memory 直接访问，Token 按需拉取
本质: 让通信适配计算需求 (而非让计算适配通信)
```

### 11.3 演进路径

```
DeepEP:         Token → Chunk → Network → Chunk → Token → Compute
                (通信约束了数据流)

Mega MoE:       Token → Symmetric Memory → Token → Compute
                (通信不再约束数据流)

理想状态:       Token → Compute (通信完全透明)
```

---

## 12. 代码引用索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `deep_gemm/mega/__init__.py` | 16-48 | SymmBuffer 定义与分配 |
| `deep_gemm/mega/__init__.py` | 58-74 | Token 对齐到 block_m |
| `csrc/apis/mega.hpp` | 14-122 | Buffer 布局计算 |
| `csrc/apis/mega.hpp` | 124-206 | C++ API 入口 |
| `scheduler/mega_moe.cuh` | 30-219 | MegaMoEScheduler 完整实现 |
| `scheduler/mega_moe.cuh` | 183-197 | fetch_expert_recv_count |
| `scheduler/mega_moe.cuh` | 199-218 | for_each_block 主循环 |
| `impls/sm100_fp8_fp4_mega_moe.cuh` | 356-600 | Dispatch Warp (Token 级拉取) |
| `impls/sm100_fp8_fp4_mega_moe.cuh` | 650-719 | GEMM TMA Load Warp |
| `impls/sm100_fp8_fp4_mega_moe.cuh` | 763-872 | MMA Issue Warp |
| `impls/sm100_fp8_fp4_mega_moe.cuh` | 877-1357 | Epilogue & Combine Warp |
| `impls/sm100_fp8_fp4_mega_moe.cuh` | 1232-1247 | Combine 阶段的 Chunk 定义 |
| `tests/test_mega_moe.py` | 40-46 | Token count 处理测试 |

---

## 13. 总结

**Mega MoE 对 Chunk Streaming 概念的映射不是 "继承" 而是 "超越"**:

1. **Chunk 作为通信粒度**: 被 Symmetric Memory 消除。NVLink 直接内存访问使 Token 级通信成为可能。

2. **Chunk 作为调度粒度**: 演变为 BLOCK_M Token 的 "计算 Block"。GEMM 以 BLOCK_M 个 Token 为一组进行计算。

3. **Chunk Streaming 流式行为**: 被 Arrival Counter 机制继承。等待 BLOCK_M 个 Token 到达即开始计算，无需等待整个 Expert 或整个 Batch。

4. **Chunk 作为 Hidden 维度分块**: Combine 阶段的 `kNumChunks` 是唯一保留的 Chunk 概念，但这是资源约束下的工程优化，非通信需求。

**核心洞察**: Mega MoE 的设计哲学是 **"让通信适配计算"** 而非 **"让计算适配通信"**。Symmetric Memory 是这一哲学的硬件基础，它使通信粒度可以细化为 Token 级，从而消除了 DeepEP 中 Chunk 聚合的必要性。
