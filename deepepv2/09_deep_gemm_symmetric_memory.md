# DeepGEMM Mega MoE 对称内存（Symmetric Memory）深度分析

> 基于 DeepGEMM 源码的 Mega MoE kernel 对称内存使用全解析：从 Python API 到 CUDA kernel 的 NVLink 直接访问机制

---

## 目录

1. [概述与设计哲学](#1-概述与设计哲学)
2. [软件依赖与硬件要求](#2-软件依赖与硬件要求)
3. [SymBuffer 核心数据结构](#3-symbuffer-核心数据结构)
4. [MegaMoEBuffer 内存布局](#4-megamoebuffer-内存布局)
5. [对称内存分配与注册](#5-对称内存分配与注册)
6. [NVLink Barrier 同步机制](#6-nvlink-barrier-同步机制)
7. [Dispatch Warps：远程数据拉取](#7-dispatch-warps远程数据拉取)
8. [Epilogue Warps：Combine 写回](#8-epilogue-warpscombine-写回)
9. [Combine Reduction：Top-K 归约](#9-combinetop-k-归约)
10. [Python API 完整使用流程](#10-python-api-完整使用流程)
11. [与 DeepEP 的对比分析](#11-与-deepep-的对比分析)
12. [关键设计洞察](#12-关键设计洞察)

---

## 1. 概述与设计哲学

DeepGEMM 的 Mega MoE kernel 是 **单 kernel 融合 MoE 全流水线** 的极致实现。其核心创新在于利用 **symmetric memory（对称内存）** 将传统 MoE 的 Dispatch → GEMM → Combine 三阶段消除为单 kernel 内的 warp 协作：

```
传统 MoE:   Dispatch(kernel) → GEMM(kernel) → Combine(kernel)
Mega MoE:   Single Kernel (Dispatch Warps + MMA Warps + Epilogue/Combine Warps)
```

**对称内存的本质**：通过 NVLink 提供的跨 GPU 虚拟地址直接访问能力，让一个 GPU 可以直接读写另一个 GPU 的 HBM，无需显式数据拷贝。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Symmetric Memory 抽象模型                          │
│                                                                     │
│  GPU 0 HBM          GPU 1 HBM          GPU 2 HBM          GPU 3 HBM │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐       ┌───────┐ │
│  │SymBuffer │       │SymBuffer │       │SymBuffer │       │  ...  │ │
│  │  Rank 0  │◄──────│  Rank 1  │◄──────│  Rank 2  │◄──────│       │ │
│  └──────────┘       └──────────┘       └──────────┘       └───────┘ │
│       ▲                   ▲                   ▲                     │
│       │    NVLink Fabric (GPU-centric access)    │                     │
│       └───────────────────┴─────────────────────┘                     │
│              任何 Rank 可访问任何其他 Rank 的 SymBuffer                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 软件依赖与硬件要求

### 2.1 PyTorch Symmetric Memory API

Mega MoE 依赖 PyTorch 的 `torch.distributed._symmetric_memory` 模块：

```python
# deep_gemm/mega/__init__.py:10
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed as dist
```

**核心 API 调用**：

| API | 用途 | 代码位置 |
|-----|------|---------|
| `symm_mem.empty(num_bytes, dtype, device)` | 单 rank 分配 | `__init__.py:43` |
| `symm_mem.rendezvous(buffer, group)` | 多 rank 注册并获取跨 rank 指针 | `__init__.py:47` |
| `allocator.empty(...)` | 根据 rank 数自动选择分配器 | `__init__.py:42-43` |

### 2.2 硬件要求

| 要求 | 原因 |
|------|------|
| **SM100 (Blackwell)** | TMEM（Tensor Memory）是 Mega MoE 单 kernel 融合的关键硬件原语 |
| **NVLink** | 对称内存的跨 GPU 访问依赖 NVLink fabric |
| **2-CTA Cluster** | SM100 的 2-CTA MMA 需要 cluster 支持 |

### 2.3 为什么必须 SM100？

```cpp
// sm100_fp8_fp4_mega_moe.cuh:78
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
```

SM100 提供了三个 Mega MoE 不可或缺的能力：

1. **TMEM（Tensor Memory）**：片上存储 MMA 累加器，使计算与写回完全解耦
2. **UTCCP（UTility Copy）**：SMEM → TMEM 的异步拷贝，用于 Scale Factor 加载
3. **2-CTA UMMA**：单指令完成 256×N×K 的矩阵乘法，配合 multicast 减少 SMEM 带宽压力

```
┌──────────────────────────────────────────────────────────────┐
│              SM100 Mega MoE 硬件能力依赖                      │
│                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │  TMEM   │    │ UTCCP   │    │2-CTA    │                  │
│  │ 256 cols│    │SMEM→TMEM│    │  UMMA   │                  │
│  │ per SM  │    │ async   │    │Multicast│                  │
│  └────┬────┘    └────┬────┘    └────┬────┘                  │
│       │              │              │                        │
│       ▼              ▼              ▼                        │
│  ┌─────────────────────────────────────────┐                │
│  │    计算-写回解耦 + SF异步加载 + 大矩阵   │                │
│  └─────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. SymBuffer 核心数据结构

### 3.1 结构定义

```cpp
// sym_buffer.cuh:9-12
template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base;                           // 本地 buffer 基地址
    int64_t offsets[kNumMaxRanks];          // 各 rank 相对于 base 的偏移
    uint32_t rank_idx;                      // 当前 rank 编号
    // ...
};
```

**设计精髓**：每个 rank 的 SymBuffer 大小相同、布局相同，因此跨 rank 访问只需加上一个固定偏移：

```
Rank 0 视角:
  base = 0x1000 (本地)
  offsets[0] = 0        (本地)
  offsets[1] = 0x800000 (Rank 1 的 buffer 基地址 - Rank 0 的 buffer 基地址)
  offsets[2] = 0x1000000
  ...

Rank 1 视角:
  base = 0x1800 (本地)
  offsets[0] = -0x800000
  offsets[1] = 0
  offsets[2] = 0x800000
  ...
```

### 3.2 map() 函数：地址转换核心

```cpp
// sym_buffer.cuh:34-40
template <typename ptr_t>
CUTLASS_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
    if constexpr (kNumRanks == 1)
        return ptr;  // 单 rank 无需转换

    int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
    return *reinterpret_cast<ptr_t*>(&mapped_ptr);
}
```

**工作原理**：
- 输入：本地 rank 视角的虚拟地址 `ptr`
- 输出：可通过 NVLink 远程访问的地址（目标 rank 的 HBM 物理地址）
- 关键：`ptr` 必须是 **本地 rank 的合法虚拟地址**，map 后变成远端可访问地址

### 3.3 构造过程

```cpp
// sym_buffer.cuh:19-25
template <typename Container>
explicit SymBuffer(const Container& c, const uint32_t& rank_idx): rank_idx(rank_idx) {
    const auto size = static_cast<uint32_t>(c.size());
    base = c[rank_idx];                          // 本地基地址
    for (uint32_t i = 0; i < kNumMaxRanks; ++ i)
        offsets[i] = i < size ? (c[i] - base) : 0;  // 计算各 rank 偏移
}
```

在 JIT kernel 中的调用：

```cpp
// sm100_fp8_fp4_mega_moe.hpp:290
.sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
```

其中 `sym_buffer_ptrs` 是 `std::vector<int64_t>`，由 `symm_mem.rendezvous()` 返回。

---

## 4. MegaMoEBuffer 内存布局

### 4.1 整体布局

MegaMoEBuffer 是单一大块连续内存，通过 `base` 指针和偏移量划分为多个逻辑区域：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MegaMoEBuffer 内存布局                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Workspace (128B+)                                            │   │
│  │  - Grid sync counters (4 × uint32)                           │   │
│  │  - NVLink barrier counter + 2 phase signals                  │   │
│  │  - Schedule task counters (L1/L2/SharedL1/SharedL2)          │   │
│  │  - Expert send/recv counts (per-expert atomic)               │   │
│  │  - Ring buffer full/empty counts                             │   │
│  │  - Source token-topk indices (for dispatch)                  │   │
│  │  - Token source metadata (for combine write-back)            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Input Buffers (per-rank, 本地写入/远程读取)                   │   │
│  │  - input_token_buffer:       [num_max_tokens, hidden] FP8    │   │
│  │  - input_sf_buffer:          [num_max_tokens, hidden/128]    │   │
│  │  - input_topk_idx_buffer:    [num_max_tokens, topk] int64   │   │
│  │  - input_topk_weights_buffer:[num_max_tokens, topk] float32  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Shared Expert Buffers                                        │   │
│  │  - shared_l1_token_buffer:   (reuse input_token_buffer)      │   │
│  │  - shared_l1_sf_buffer:      [num_sf_tokens, hidden/128]    │   │
│  │  - shared_l2_token_buffer:   [num_tokens, shared_inter]     │   │
│  │  - shared_l2_sf_buffer:      [num_sf_tokens, shared_int/128]│   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Routed Expert Ring Buffers (复用池)                          │   │
│  │  - l1_token_buffer:          [num_ring_tokens, hidden]      │   │
│  │  - l1_sf_buffer:             [num_sf_ring_tokens, h/128]    │   │
│  │  - l1_topk_weights_buffer:   [num_ring_tokens, 1] float     │   │
│  │  - l2_token_buffer:          [num_ring_tokens, inter]       │   │
│  │  - l2_sf_buffer:             [num_sf_ring_tokens, int/128]  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Combine Buffer (per-topk-slot, 远程写入)                     │   │
│  │  - combine_token_buffer:     [topk(+1), num_tokens, hidden] │   │
│  │    BF16, 每个 topk slot 独立区域，用于 top-k reduce          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 关键尺寸计算

```cpp
// mega.hpp:58-65 - Ring buffer 容量计算
int num_ring_tokens = 0;
for (const auto& block_m: layout::kCandidateBlockM) {
    const auto num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
    const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
        num_pool_blocks, num_sms, hidden, intermediate_hidden);
    num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
}
num_ring_tokens = math::align(num_ring_tokens, layout::kLCMCandidateBlockM);
```

**Ring buffer 容量**取决于最坏情况下的活跃 pool blocks 数，需遍历所有候选 BLOCK_M 取最大值。

### 4.3 Workspace 详细布局

```cpp
// mega_moe.cuh:129-137 - Workspace 128B 头部布局
// [ 0..15]: 4 x uint32_t grid sync counters
// [16..20]: uint32_t NVLink barrier counter
// [20..27]: 2 x int NVLink barrier signals (phase 0 and 1)
// [28..31]: uint32_t L1 schedule task counter
// [32..35]: uint32_t L2 schedule task counter
// [36..39]: uint32_t shared L1 schedule task counter
// [40..43]: uint32_t shared L2 schedule task counter
// [44..127]: padding (隔离 hot expert counters)
```

---

## 5. 对称内存分配与注册

### 5.1 Python 侧分配

```python
# __init__.py:34-51
class SymmBuffer:
    def __init__(self, group, num_experts, num_max_tokens_per_rank, ...):
        # 1. 计算所需字节数
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts, num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, mma_type, activation, num_shared_experts
        )
        
        # 2. 选择分配器
        allocator = torch if group.size() == 1 else symm_mem
        
        # 3. 分配
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        
        # 4. 注册（多 rank 时获取跨 rank 指针）
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        
        # 5. 初始化
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()
```

### 5.2 跨 rank 指针传递

```python
# __init__.py:169-170
_C.fp8_fp4_mega_moe(
    ...,
    sym_buffer.buffer,                    # 本地 buffer tensor
    sym_buffer.handle.buffer_ptrs,        # 所有 rank 的基地址列表
    sym_buffer.group.rank(),              # 当前 rank 编号
    ...
)
```

```cpp
// mega.hpp:243-244 → sm100_fp8_fp4_mega_moe.hpp:290
const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
// ...
.sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
```

### 5.3 输入数据准备

```python
# test_mega_moe.py:170-179 - 每次 kernel 调用前拷贝输入
def copy_inputs_to_buffer():
    if is_bf16xbf16:
        buffer.x[:num_tokens].copy_(x)
    else:
        buffer.x[:num_tokens].copy_(x[0])
        buffer.x_sf[:num_tokens].copy_(x[1])
        if num_shared_experts > 0:
            _copy_fp8_sf(buffer.shared_l1_acts_sf, shared_l1_x_sf, num_tokens)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)
```

**注意**：debug 模式下 kernel 会 zero 整个 buffer，因此每次调用前必须重新拷贝输入。

---

## 6. NVLink Barrier 同步机制

### 6.1 双层同步架构

Mega MoE 使用 **grid_sync（SM 内同步）+ nvlink_barrier（跨 rank 同步）** 的双层架构：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NVLink Barrier 状态机                             │
│                                                                     │
│  Grid Sync (SM 内)          NVLink Signal            Grid Sync      │
│  ┌──────────┐    ┌──────────────────────────┐    ┌──────────┐      │
│  │ 所有 SM  │    │ SM 0 发 signal 到所有    │    │ 所有 SM  │      │
│  │ 到达屏障 │───►│ 远端 rank 并等待回应     │───►│ 继续执行 │      │
│  └──────────┘    └──────────────────────────┘    └──────────┘      │
│                                                                     │
│  Phase 0: signal = +1 → 等待 signal == num_ranks                   │
│  Phase 1: signal = -1 → 等待 signal == 0                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 grid_sync 实现

```cpp
// barrier.cuh:21-44
template <uint32_t kNumSMs, uint32_t kGridSyncIndex, typename sync_scope_t>
CUTLASS_DEVICE void grid_sync(const layout::Workspace& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        // SM 0 写入特殊值，其他 SM 写 1
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        const auto start_clock = clock64();
        do {
            new_value = ptx::ld_acq(count_ptr);
            // 超时检测 (60s)
            if (clock64() - start_clock >= kNumTimeoutCycles) {
                DG_DEVICE_ASSERT(false and "Grid sync timeout");
            }
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
    }
    sync_scope();
}
```

**关键设计**：
- 使用 XOR tag 检测完成，避免计数器溢出问题
- SM 0 写入 `0x80000000 - (kNumSMs-1)`，其他 SM 各写 1，总和达到 tag 时完成
- 支持 4 个独立的 grid sync counter（dispatch 和 epilogue 使用不同 counter 避免冲突）

### 6.3 nvlink_barrier 实现

```cpp
// barrier.cuh:46-89
template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
CUTLASS_DEVICE void nvlink_barrier(const layout::Workspace& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true) {
    // 1. Grid sync (prologue)
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

    // 2. NVLink cross-rank barrier (仅 SM 0 参与)
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // 发送信号到远端 ranks
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // 等待所有 rank 到达
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            while (ptx::ld_acq_sys(signal_ptr) != target) {
                // 超时检测
            }
        }
    }

    // 3. Grid sync (epilogue)
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}
```

**核心原语**：
- `ptx::red_add_rel_sys`：远程 atomic add（通过 NVLink 直接写远端 HBM）
- `ptx::ld_acq_sys`：带 system scope 的 acquire load（确保看到远端写入）
- 双 phase 设计避免 signal 竞争

### 6.4 Barrier 使用点

```cpp
// sm100_fp8_fp4_mega_moe.cuh:309-311 - NVLink barrier tags
constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;      // Pull 前
constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;     // Combine 前
constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;     // Workspace 清理后
```

```
时间线:
  Dispatch Count → Grid Sync → [NVLink Barrier #1] → Pull Data → 
  Clean Workspace → [NVLink Barrier #3] → ... → Combine Write-back → 
  [NVLink Barrier #2] → Combine Reduce
```

---

## 7. Dispatch Warps：远程数据拉取

### 7.1 Warp 角色分配

```cpp
// sm100_fp8_fp4_mega_moe.cuh:329
if (warp_idx < kNumDispatchWarps) {
    // Dispatch warps: 负责从远端 rank 拉取 token 数据
```

### 7.2 Dispatch 四阶段

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Dispatch Warps 执行流程                             │
│                                                                     │
│  Phase 1: Count Expert Tokens                                       │
│    ┌─────────────────────────────────────────────────────────┐     │
│    │ 读取 topk_idx → atomicAdd 到 expert_token_count         │     │
│    │ (本地 smem, 统计每个 expert 需要接收多少 token)          │     │
│    └─────────────────────────────────────────────────────────┘     │
│                              ↓                                      │
│  Phase 2: Get SM Offset                                             │
│    ┌─────────────────────────────────────────────────────────┐     │
│    │ atomicAdd 到 workspace.expert_send_count                │     │
│    │ (全局 atomic, 获取本 SM 在全局 token 中的偏移)           │     │
│    │ 64-bit 编码: [32-bit count | 32-bit sm_offset]          │     │
│    └─────────────────────────────────────────────────────────┘     │
│                              ↓                                      │
│  Phase 3: Write Source Indices (跨 rank)                            │
│    ┌─────────────────────────────────────────────────────────┐     │
│    │ sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx  │     │
│    │ (通过 NVLink 写远端 workspace 的 src_token_topk_idx)     │     │
│    └─────────────────────────────────────────────────────────┘     │
│                              ↓                                      │
│  Phase 4: Pull Token Data (跨 rank)                                 │
│    ┌─────────────────────────────────────────────────────────┐     │
│    │ TMA load 从远端 input_token_buffer → 本地 l1 ring      │     │
│    │ (通过 NVLink 直接读取远端 HBM)                           │     │
│    └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 跨 rank 写入 source indices

```cpp
// sm100_fp8_fp4_mega_moe.cuh:371-377
read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
    const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
    const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
    const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
        expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
    *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;  // ← 跨 rank 写入
});
```

**关键**：`sym_buffer.map(dst_ptr, dst_rank_idx)` 将本地虚拟地址转换为远端 rank 可访问的物理地址，然后通过普通 store 指令完成跨 rank 写入。

### 7.4 跨 rank 拉取 token 数据

```cpp
// sm100_fp8_fp4_mega_moe.cuh:533-556
const auto src_base_ptr = sym_buffer.map(
    buffer.input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(), 
    current_rank_in_expert_idx);  // ← 远端源地址
const auto dst_base_ptr = buffer.l1_token_buffer.get_data_buffer(
    pool_token_idx % kNumRingTokens).get_base_ptr();  // ← 本地目标地址

if (cute::elect_one_sync()) {
    for (uint32_t i = 0; i < kNumChunks; ++ i) {
        ptx::tma_load_1d(
            pull_buffer.get_base_ptr(),                          // 本地 smem
            math::advance_ptr(src_base_ptr, i * kNumBytesPerPull), // 远端 HBM
            pull_mbarrier, kNumBytesPerPull
        );
        ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kNumBytesPerPull);
        // ...
    }
}
```

**TMA 远程 load**：SM100 的 TMA（Tensor Memory Accelerator）支持通过 NVLink 从远端 HBM 直接加载数据到本地 smem，无需 CPU 介入。

### 7.5 Round-Robin Rank 选择

```cpp
// sm100_fp8_fp4_mega_moe.cuh:461-509
// Round-robin rank selection via iterative min-peeling
while (true) {
    // 计算活跃 rank 数和最小剩余 token 数
    uint32_t num_actives_in_lane = 0;
    uint32_t min_in_lane = 0xffffffff;
    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
        num_actives_in_lane += remaining[i] > 0;
        if (remaining[i] > 0)
            min_in_lane = cute::min(min_in_lane, remaining[i]);
    }
    const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
    const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);

    // 命中当前轮次
    const uint32_t num_round_tokens = length * num_active_ranks;
    if (slot_idx < num_round_tokens) {
        // 确定当前 token 来自哪个 rank
        // ...
        break;
    }
    // 进入下一轮
    slot_idx -= num_round_tokens;
    offset += length;
    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
        remaining[i] -= cute::min(remaining[i], length);
}
```

**min-peeling 算法**：确保来自不同 rank 的 token 被均匀交错处理，最大化 NVLink 带宽利用率。

---

## 8. Epilogue Warps：Combine 写回

### 8.1 L2 输出写回远端

```cpp
// sm100_fp8_fp4_mega_moe.cuh:1260-1299
// Write into remote buffers
const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
const uint32_t bank_group_idx = lane_idx % 8;

for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
    const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
    const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;

    if (m_idx_in_block >= valid_m)
        break;

    uint32_t dst_rank_idx, dst_token_idx, dst_topk_idx;
    if (task_info.is_shared()) {
        dst_rank_idx = sym_buffer.rank_idx;  // shared expert 写回本地
        dst_token_idx = pool_m_idx + m_idx_in_block;
        dst_topk_idx = kNumTopk;
    } else {
        const auto src_metadata = *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);
        dst_rank_idx = src_metadata.rank_idx;   // 写回源 rank
        dst_token_idx = src_metadata.token_idx; // 源 token 位置
        dst_topk_idx = src_metadata.topk_idx;   // 对应的 topk slot
    }

    // 从 smem 读取
    const auto smem_ptr = /* ... */;
    const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

    // 写入远端 combine buffer
    const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                           .get_data_buffer(dst_token_idx);
    const auto dst_ptr = math::advance_ptr<float4>(
        dst_token.get_base_ptr(),
        n_idx * sizeof(nv_bfloat16) + (lane_idx % 16) * sizeof(float4));
    *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;  // ← 跨 rank 写入
}
```

**关键**：L2 GEMM 的输出通过 `sym_buffer.map` 直接写入远端 rank 的 `combine_token_buffer`，每个 topk slot 独立区域。

### 8.2 Combine Buffer 布局

```cpp
// mega_moe.cuh:433-435
combine_token_buffer = Buffer(
    bf16_token_layout, 
    num_topk + (num_shared_experts > 0 ? 1u : 0u),  // 每个 topk + shared expert
    num_max_tokens_per_rank,
    /* ... */
);
```

```
combine_token_buffer 布局:
┌──────────────────────────────────────────────────────────────┐
│ Slot 0 (topk=0):  [token_0_hidden, token_1_hidden, ...] BF16│
│ Slot 1 (topk=1):  [token_0_hidden, token_1_hidden, ...] BF16│
│ ...                                                          │
│ Slot k-1:         [token_0_hidden, token_1_hidden, ...] BF16│
│ Slot k (shared):  [token_0_hidden, token_1_hidden, ...] BF16│  ← 仅当 shared experts > 0
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Combine Reduction：Top-K 归约

### 9.1 双缓冲流水线

```cpp
// sm100_fp8_fp4_mega_moe.cuh:1350-1354
// Per-warp buffer: 2 stage load buffers + 1 store buffer
const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
    return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
});
const auto combine_store_buffer = math::advance_ptr<uint4>(
    smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);
```

### 9.2 Top-K Reduce 核心逻辑

```cpp
// sm100_fp8_fp4_mega_moe.cuh:1367-1421
for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
     token_idx < num_tokens;
     token_idx += kNumSMs * kNumEpilogueWarps) {
    // 读取 top-k slot indices
    const int stored_topk_slot_idx = lane_idx < kNumTopk ?
        static_cast<int>(__ldg(buffer.input_topk_idx_buffer.get_base_ptr<int64_t>() 
                               + token_idx * kNumTopk + lane_idx)) :
        (kNumSharedExperts > 0 and lane_idx == kNumTopk ? static_cast<int>(kNumTopk) : -1);
    const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0);

    for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
        uint32_t mask = total_mask;
        
        // 加载第一个 topk 数据
        bool do_reduce = move_mask_and_load(load_stage_idx);
        
        float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
        while (do_reduce) {
            // 预取下一个 topk 数据（双缓冲）
            do_reduce = move_mask_and_load(load_stage_idx ^ 1);
            
            // 累加当前 topk 数据
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
        // ... cast to BF16 and TMA store to y
    }
}
```

### 9.3 Combine 数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Combine Reduction 数据流                          │
│                                                                     │
│  远端 combine_token_buffer (per-topk-slot)                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                               │
│  │ Slot 0  │ │ Slot 1  │ │ Slot k-1│                               │
│  └────┬────┘ └────┬────┘ └────┬────┘                               │
│       │           │           │                                     │
│       ▼           ▼           ▼                                     │
│  ┌─────────────────────────────────────┐                           │
│  │  TMA Load (本地, 从 combine buffer) │                           │
│  └─────────────────────────────────────┘                           │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────┐                           │
│  │  Float Accumulate (Top-K Reduce)    │                           │
│  └─────────────────────────────────────┘                           │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────┐                           │
│  │  Cast to BF16 → TMA Store → y      │                           │
│  └─────────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
```

**注意**：Combine 阶段对 `combine_token_buffer` 的访问是 **本地读取**（因为该 buffer 已被 Epilogue Warps 通过 NVLink 写回本地），不需要 `sym_buffer.map`。

---

## 10. Python API 完整使用流程

### 10.1 完整调用示例

```python
# test_mega_moe.py:93-99 - 分配对称内存
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts,
    num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden,
    num_shared_experts=num_shared_experts,
    mma_type=args.mma_type
)

# test_mega_moe.py:160-161 - 权重变换
transformed_l1_weights, transformed_l2_weights = (
    deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)
)

# test_mega_moe.py:181-198 - 执行 kernel
def run_fused():
    cumulative_local_expert_recv_stats_fused.copy_(initial_cumulative_local_expert_recv_stats_fused)
    copy_inputs_to_buffer()  # 每次调用前拷贝输入

    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    kernel_kwargs = dict(
        y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
        sym_buffer=buffer,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math))
    if num_shared_experts > 0:
        kernel_kwargs.update(
            shared_l1_weights=transformed_shared_l1_weights,
            shared_l2_weights=transformed_shared_l2_weights
        )
    (deep_gemm.bf16_mega_moe if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe)(**kernel_kwargs)
    return y, cumulative_local_expert_recv_stats_fused
```

### 10.2 权重变换

```python
# __init__.py:97-111 - gate/up 交错
def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result

# __init__.py:114-128 - SF 转置 (UTCCP 需要)
def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return result
```

### 10.3 参数传递链

```
Python:
  sym_buffer.buffer ──────────────► torch::Tensor (本地 buffer)
  sym_buffer.handle.buffer_ptrs ──► std::vector<int64_t> (所有 rank 基地址)
  sym_buffer.group.rank() ────────► int (当前 rank)

C++ API (mega.hpp):
  sym_buffer_ptrs ────────────────► layout::SymBuffer<>(sym_buffer_ptrs, rank_idx)

JIT Kernel (sm100_fp8_fp4_mega_moe.hpp):
  args.sym_buffer_ptrs ───────────► 通过 __grid_constant__ 传入 kernel

CUDA Kernel:
  sym_buffer.get_base_ptr() ──────► 本地基地址
  sym_buffer.map(ptr, rank) ──────► 跨 rank 地址转换
```

---

## 11. 与 DeepEP 的对比分析

### 11.1 架构对比

| 维度 | DeepEP | DeepGEMM Mega MoE |
|------|--------|-------------------|
| **执行模式** | 多 kernel（Dispatch + GEMM + Combine） | 单 kernel 融合 |
| **通信方式** | NCCL all-to-all / NVLink peer access | NVLink symmetric memory 直接访问 |
| **中间数据存储** | 显式 buffer（5层模型） | 对称内存中的 ring buffer |
| **同步机制** | NCCL barrier + 信号量 | 自定义 grid_sync + nvlink_barrier |
| **数据搬运** | 显式 copy kernel | TMA 远程 load/store |
| **Combine** | 独立 kernel + 跨 rank reduce | 同一 kernel 内 warp 协作 |
| **硬件要求** | SM90+ (Hopper) | SM100 (Blackwell) |

### 11.2 对称内存使用差异

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DeepEP vs DeepGEMM 对称内存使用                   │
│                                                                     │
│  DeepEP (传统模式):                                                 │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                    │
│  │ Dispatch │────►│   GEMM   │────►│ Combine  │                    │
│  │  Kernel  │     │  Kernel  │     │  Kernel  │                    │
│  └──────────┘     └──────────┘     └──────────┘                    │
│       │                 │                │                          │
│       ▼                 ▼                ▼                          │
│  NCCL all-to-all  本地 GEMM      NCCL all-to-all                   │
│  (显式数据拷贝)   (无通信)       (显式 reduce)                      │
│                                                                     │
│  DeepGEMM Mega MoE (融合模式):                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Single Kernel                              │   │
│  │  ┌──────────┐     ┌──────────┐     ┌──────────┐             │   │
│  │  │ Dispatch │────►│   GEMM   │────►│ Combine  │             │   │
│  │  │  Warps   │     │  Warps   │     │  Warps   │             │   │
│  │  └──────────┘     └──────────┘     └──────────┘             │   │
│  │       │                 │                │                    │   │
│  │       ▼                 ▼                ▼                    │   │
│  │  TMA remote load   本地 TMEM      NVLink store              │   │
│  │  (NVLink direct)   (片上累加)     (NVLink direct)            │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.3 关键差异详解

| 差异点 | DeepEP | DeepGEMM Mega MoE |
|--------|--------|-------------------|
| **数据搬运粒度** | 整块 tensor 拷贝 | 逐 token TMA load |
| **通信与计算重叠** | kernel 间流水线 | warp 间流水线（同一 kernel 内） |
| **中间 buffer** | 显式 5 层 buffer | ring buffer 复用 |
| **同步开销** | NCCL kernel launch | 单次 grid_sync (~μs 级) |
| **内存占用** | 多份完整 buffer | 单份共享 buffer + ring |
| **远程访问方式** | NCCL 集体通信 | NVLink 直接 load/store |
| **Combine reduce** | 跨 rank allreduce | 本地 top-k accumulate |

### 11.4 Mega Mo E 独特设计

1. **Ring Buffer 复用**：L1/L2 的 activation buffer 采用 ring buffer 结构，容量只需覆盖最坏情况下的活跃 blocks，而非全量 token。

2. **Warp 角色分工**：同一 kernel 内不同 warp 承担不同角色（Dispatch / TMA Load / MMA Issue / Epilogue / Combine），通过 barrier 同步协作。

3. **TMEM 计算-写回解耦**：SM100 的 TMEM 允许 MMA 计算与 epilogue 写回完全并行，通过 `tmem_full_barriers` / `tmem_empty_barriers` 同步。

4. **UTCCP SF 异步加载**：Scale Factor 通过 UTCCP 从 SMEM 异步拷贝到 TMEM，与 MMA 计算重叠。

5. **min-peeling rank 选择**：Dispatch warps 使用 min-peeling 算法实现 round-robin rank 选择，最大化 NVLink 带宽利用率。

---

## 12. 关键设计洞察

### 12.1 对称内存消除了什么？

```
传统 MoE 的 "Destination-major" 问题:
  - 每个 rank 需要为每个 expert 预分配接收 buffer
  - 需要知道 "谁要给我发多少数据" → 两次 all-to-all
  - 中间 buffer 无法复用（不同 expert 数据独立）

对称内存的 "Data Access" 范式:
  - 远端 HBM 可直接访问，无需本地暂存
  - Ring buffer 按 pool block 复用，容量 = 最坏情况活跃 blocks
  - 单 kernel 内完成全部操作，无 kernel launch 开销
```

### 12.2 64-bit 编码的巧妙运用

```cpp
// sm100_fp8_fp4_mega_moe.cuh:364-367
const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(shared_storage.expert_token_count[i]);
shared_storage.expert_token_count[i] = static_cast<uint32_t>(
    ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
```

**高 32 位**：token 数量（用于计算全局偏移）
**低 32 位**：SM 偏移（用于写入 source indices）

一次 atomic 操作同时完成 "获取偏移" 和 "预留空间"。

### 12.3 超时检测的必要性

```cpp
// barrier.cuh:12
constexpr int64_t kNumTimeoutCycles = 60ll * 2000000000ll;  // 60s at 2 GHz

// barrier.cuh:36-39
if (clock64() - start_clock >= kNumTimeoutCycles) {
    printf("DeepGEMM grid sync timeout: ...");
    DG_DEVICE_ASSERT(false and "Grid sync timeout");
}
```

跨 rank 同步必须考虑超时，否则单节点故障会导致整个集群 hang。

### 12.4 寄存器压力管理

```cpp
// sm100_fp8_fp4_mega_moe.cuh:315-322
constexpr bool kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64;
constexpr uint32_t kNumDispatchRegisters = kUseMoreEpilogueRegisters ? 48 : 96;
constexpr uint32_t kNumNonEpilogueRegisters = kUseMoreEpilogueRegisters ? 40 : 88;
constexpr uint32_t kNumEpilogueRegisters = kUseMoreEpilogueRegisters ? 208 : 160;
```

不同 warp 角色使用 `warpgroup_reg_alloc/dealloc` 动态调整寄存器分配，在 64512 的总寄存器限制内最大化利用率。

### 12.5 总结

DeepGEMM Mega MoE 的对称内存使用代表了一种 **通信-计算融合** 的极致范式：

1. **消除显式通信**：通过 NVLink 直接访问远端 HBM，将通信转化为地址映射
2. **消除中间 buffer**：ring buffer 复用 + 单 kernel 融合，内存效率最大化
3. **消除 kernel launch 开销**：warp 角色协作替代 kernel 间同步
4. **硬件-软件协同设计**：TMEM、UTCCP、2-CTA MMA 等 SM100 特性被充分利用

这种设计使 Mega MoE 在支持 shared experts 的场景下，实现了传统 MoE 无法达到的 **单 kernel 全流水线融合**。

---

## 附录 A：关键文件索引

| 文件 | 行数 | 核心内容 |
|------|------|---------|
| `deep_gemm/include/deep_gemm/layout/sym_buffer.cuh` | 44 | SymBuffer 定义与 map() |
| `deep_gemm/include/deep_gemm/comm/barrier.cuh` | 91 | grid_sync + nvlink_barrier |
| `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` | 446 | MegaMoEBuffer 内存布局 |
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` | 1461 | FP8/FP4 kernel 实现 |
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh` | 1283 | BF16 kernel 实现 |
| `csrc/apis/mega.hpp` | 406 | C++ API 入口 |
| `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | 319 | JIT 编译与 TMA desc |
| `deep_gemm/mega/__init__.py` | 203 | Python API |
| `tests/test_mega_moe.py` | 443 | 完整测试用例 |

## 附录 B：关键代码位置索引

| 功能 | 文件:行号 |
|------|----------|
| SymBuffer::map() | `sym_buffer.cuh:34-40` |
| nvlink_barrier 定义 | `barrier.cuh:46-89` |
| grid_sync 定义 | `barrier.cuh:21-44` |
| MegaMoEBuffer 构造 | `mega_moe.cuh:353-436` |
| Dispatch Pull TMA load | `sm100_fp8_fp4_mega_moe.cuh:533-556` |
| Combine 写回远端 | `sm100_fp8_fp4_mega_moe.cuh:1260-1299` |
| Combine Top-K reduce | `sm100_fp8_fp4_mega_moe.cuh:1367-1421` |
| 64-bit 编码 atomic | `sm100_fp8_fp4_mega_moe.cuh:364-367` |
| min-peeling rank 选择 | `sm100_fp8_fp4_mega_moe.cuh:461-509` |
| Python SymmBuffer 分配 | `__init__.py:34-51` |
| 权重 gate/up 交错 | `__init__.py:97-111` |
| 测试完整流程 | `test_mega_moe.py:93-198` |
