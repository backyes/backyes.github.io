# Normal vs Low-Latency: 双通信哲学在 DeepEP V1 / V2 / DeepGEMM 中的三系统比较分析

> 分析日期: 2026-07-30
> 分析目标: 从 Blog 理论出发，对比 DeepEP V1（双内核）、DeepEP V2（ElasticBuffer 统一）、DeepGEMM Mega MoE（单内核）三种实现对 Normal/Low-Latency 两种通信哲学的映射关系

---

## 1. 核心结论

| 系统 | Normal 模式 | Low-Latency 模式 | 统一机制 |
|------|------------|-----------------|---------|
| **DeepEP V1** | `internode.cu` (kLowLatencyMode=false) | `internode_ll.cu` (独立内核) | 无统一，运行时二选一 |
| **DeepEP V2** | `hybrid_dispatch_impl` (scaleout > 1) | `dispatch_impl` (scaleout = 1) | ElasticBuffer + 模板参数 `kIsScaleupNVLink` |
| **DeepGEMM** | 单一内核 | 无 | NVLink Symmetric Memory 消除模式分化 |

**关键洞察：**
- DeepEP V1 是**物理分离**：两个完全独立的 kernel 文件，编译时确定
- DeepEP V2 是**逻辑统一**：同一套 JIT 代码，通过 `num_scaleout_ranks` 和 `kIsScaleupNVLink` 模板参数在编译期分化
- DeepGEMM 是**硬件消除**：NVLink 全连接 + Symmetric Memory 使得 forwarding 和 chunk 聚合失去意义，模式分化被硬件能力抹平

---

## 2. Blog 理论：两种通信哲学

> 来源: `/tmp/deep_ep_blog_text.txt` 第 3 节

### 2.1 原文引用

Blog 对两种模式的经典描述：

| 维度 | Normal Kernel | Low-Latency Kernel |
|------|--------------|-------------------|
| **主场景** | Training / Prefill | Decode |
| **优化目标** | 最大化吞吐 (Throughput) | 最小化延迟 (Latency) |
| **核心矛盾** | 带宽 (Bandwidth) | 延迟 (Latency) |
| **Chunk** | 关键 (Critical) | 减少 (Reduced) |
| **流水线** | 深 (Deep) | 浅 (Shallow) |
| **通信路径** | NVLink + RDMA 协同 | Direct RDMA |

### 2.2 Normal Kernel: 通信流水线 (Communication Pipelining)

Blog 原文描述：

> "In multi-GPU nodes, GPU-NIC topology is not fully symmetric. Communication paths may include: **IB Sending → IB-to-NVLink Forwarding → NVLink Receiving**, forming a three-stage communication pipeline."

Normal 模式的核心特征：
- **三阶段流水线**: Source GPU → IB Sending → RDMA Network → **IB-to-NVLink Forwarding** → NVLink Receiving → Target GPU
- **Chunk 聚合**: Token 流 → Chunk → 网络传输（Token 是调度粒度，Chunk 是通信粒度）
- **深度流水线**: 多 Stage 重叠通信与计算

### 2.3 Low-Latency Kernel: 旁路转发 (Bypass Forwarding)

Blog 原文描述：

> "Decode: small batch, few Tokens, single-Token latency sensitive. Waiting for Chunk aggregation increases latency. Low-Latency Kernel reduces intermediate layers: **Token Buffer → Direct RDMA → Receive Buffer → Expert Buffer**."

Low-Latency 模式的核心特征：
- **旁路 Forwarding**: 跳过 NVLink 中继，GPU 直接通过 RDMA 发送
- **浅流水线**: 减少 Stage 以降低单 Token 延迟
- **Chunk 减少**: 不等待 Token 聚合，单 Token 直接发送

---

## 3. DeepEP V1 实现：双内核物理分离

### 3.1 架构总览

DeepEP V1 采用**物理分离**的双内核设计：

```
deep_ep/buffers/legacy.py
  ├── Buffer (Normal 模式): 使用 internode.cu + intranode.cu
  └── Buffer (Low-Latency 模式): 使用 internode_ll.cu
```

**模式选择 API：**

```python
# deep_ep/buffers/legacy.py:37-93
class Buffer:
    def __init__(self,
                 group,
                 num_nvl_bytes: int = 0,
                 num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False,  # ← 模式选择参数
                 num_qps_per_rank: int = 24,
                 allow_nvlink_for_low_latency_mode: bool = True,
                 ...):
        self.low_latency_mode = low_latency_mode
        self.runtime = _C.Buffer(self.rank, self.group_size,
                                 num_nvl_bytes, num_rdma_bytes,
                                 low_latency_mode, ...)  # ← 传入 C++ 运行时
```

### 3.2 Normal 模式: `internode.cu`

**核心特征：三阶段流水线 + Chunk 聚合 + Forwarding**

#### (1) Warp 角色分配

```cpp
// csrc/kernels/legacy/internode.cu:487-516
enum class WarpRole {
    kRDMASender,              // RDMA 发送
    kRDMASenderCoordinator,   // RDMA 发送协调
    kRDMAAndNVLForwarder,     // RDMA + NVLink 转发 ← 关键角色
    kForwarderCoordinator,    // 转发协调
    kNVLReceivers             // NVLink 接收
};

const auto role_meta = [=]() -> std::pair<WarpRole, int> {
    if (is_forwarder) {
        if (warp_id < LEGACY_NUM_MAX_NVL_PEERS) {
            return {WarpRole::kRDMAAndNVLForwarder, ...};  // 转发 warp
        } else {
            return {WarpRole::kForwarderCoordinator, ...};
        }
    } else if (warp_id < kNumDispatchRDMASenderWarps) {
        return {WarpRole::kRDMASender, -1};  // RDMA 发送 warp
    }
    // ...
};
```

**5 种 Warp 角色**形成完整的三阶段流水线：
- `kRDMASender`: 从源 GPU 读取 Dispatch Buffer，发起 RDMA 发送
- `kRDMAAndNVLForwarder`: **接收 RDMA 数据，通过 NVLink 转发到目标 GPU**
- `kNVLReceivers`: 从 NVLink 接收数据，写入 Receive Buffer

#### (2) Chunk 聚合机制

```cpp
// csrc/kernels/legacy/internode.cu:527-529
// RDMA symmetric layout: Chunk 是通信粒度
auto rdma_channel_data = SymBuffer<uint8_t>(
    rdma_buffer_ptr,
    num_max_rdma_chunked_recv_tokens * num_bytes_per_token,  // ← Chunk 大小
    kNumRDMARanks, channel_id, num_channels);
```

```cpp
// csrc/kernels/legacy/internode.cu:809-817
// 按 Chunk 粒度发送，而非单 Token
auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
```

#### (3) Forwarding 同步机制

```cpp
// csrc/kernels/legacy/internode.cu:583-585
__shared__ volatile int forward_channel_head[LEGACY_NUM_MAX_NVL_PEERS][kNumRDMARanks];
__shared__ volatile bool forward_channel_retired[LEGACY_NUM_MAX_NVL_PEERS];
auto sync_forwarder_smem = []() {
    asm volatile("barrier.sync 1, %0;" ::"r"((LEGACY_NUM_MAX_NVL_PEERS + 1) * 32));
};
```

Forwarding 是 Normal 模式的核心——解决 NIC-GPU 拓扑不对称问题：
- RDMA 数据到达后，**不是目标 GPU 直接接收**
- 而是由**中间 GPU 作为中继**，通过 NVLink 转发到目标 GPU

#### (4) 模板参数 `kLowLatencyMode` 的编译期分支

```cpp
// csrc/kernels/legacy/internode.cu:87-90
template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    // Low-Latency 模式: 每个 GPU 独立 RDMA rank
    // Normal 模式: 同一 node 的 GPU 共享 RDMA rank
    return kLowLatencyMode ? (dst_rdma_rank * LEGACY_NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

template <bool kLowLatencyMode>
__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
    // Low-Latency: 仅同步 RDMA team
    // Normal: 全局同步
    kLowLatencyMode ? void(nvshmem_sync(rdma_team)) : nvshmem_sync_all();
}
```

**关键差异：**
- Normal 模式：`nvshmem_sync_all()` — 全局 barrier，确保所有 GPU 同步
- Low-Latency 模式：`nvshmem_sync(rdma_team)` — 仅 RDMA team 同步，减少同步开销

#### (5) Config 结构：双 Buffer 设计

```cpp
// csrc/legacy/config.hpp:24-51
struct Config {
    int num_sms;
    int num_max_nvl_chunked_send_tokens;    // NVLink Chunk 发送
    int num_max_nvl_chunked_recv_tokens;    // NVLink Chunk 接收
    int num_max_rdma_chunked_send_tokens;   // RDMA Chunk 发送
    int num_max_rdma_chunked_recv_tokens;   // RDMA Chunk 接收
    // ...
    // 约束: send < recv / 2 (确保发送方总有空间)
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens / 2);
};
```

### 3.3 Low-Latency 模式: `internode_ll.cu`

**核心特征：单 Token 直接发送 + 无 Forwarding + 浅流水线**

#### (1) 无 Warp 角色分离

```cpp
// csrc/kernels/legacy/internode_ll.cu:128-155
template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch(...) {
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // 发送阶段: 所有 warp 参与 FP8 cast + IBGDA send
    if (warp_id < num_warps - 1) {
        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            // FP8 cast + 直接发送
            // ...
            if (dst_expert_idx >= 0) {
                // 直接通过 RDMA 发送，无 Forwarding
                nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, ...);
            }
        }
    }
}
```

**与 Normal 的关键区别：**
- 无 `kRDMAAndNVLForwarder` 角色
- 无 `forward_channel_head` / `forward_channel_retired` 同步
- 每个 Token 独立发送，无 Chunk 聚合等待

#### (2) 单 Token 消息格式

```cpp
// csrc/kernels/legacy/internode_ll.cu:179-182
// Message package: index at source (int), 3 reserved int fields, hidden data, FP8 scales
const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
```

每个 Token 构成一个独立消息，包含：
- 源 Token 索引 (int4)
- Hidden 数据
- FP8 scales

#### (3) 接收端：按 Expert 组织

```cpp
// csrc/kernels/legacy/internode_ll.cu:362-370
// 接收端按 Expert 组织，而非按 Rank
const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
    local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
    src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
const auto recv_x_int4 = static_cast<int4*>(packed_recv_x) +
    local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
```

#### (4) 超时与 Mask 机制

```cpp
// csrc/kernels/legacy/internode_ll.cu:383-408
// 等待 Token 到达，带超时检测
if (sub_warp_id == 1 and lane_id == 0) {
    auto start_time = clock64();
    uint64_t wait_recv_cost = 0;
    if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
        while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + ...)) == 0
               && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES)
            ;
    }
    // 超时则 mask 该 rank
    if (wait_recv_cost > LEGACY_NUM_TIMEOUT_CYCLES) {
        atomicExch(mask_buffer_ptr + src_rank, 1);
    }
}
```

### 3.4 V1 双内核对比总结

| 维度 | Normal (`internode.cu`) | Low-Latency (`internode_ll.cu`) |
|------|------------------------|--------------------------------|
| **Warp 角色** | 5 种 (Sender/Forwarder/Receiver/Coordinator) | 2 种 (发送 warp + 计数 warp) |
| **通信粒度** | Chunk (多 Token 聚合) | 单 Token |
| **Forwarding** | 有 (GPU 作为 NVLink 中继) | 无 (Direct RDMA) |
| **同步机制** | `nvshmem_sync_all()` + barrier_block | `nvshmem_sync(rdma_team)` + atomic |
| **Buffer 结构** | 双 Buffer (NVL + RDMA) | 单 RDMA Buffer (对称 odd/even) |
| **消息格式** | Token + SourceMeta + Scales | Token + Scales (更紧凑) |
| **错误处理** | 无 | Mask buffer + 超时检测 |
| **SM 使用** | 20 (默认) | 动态 (num_sms) |

---

## 4. DeepEP V2 实现：ElasticBuffer 逻辑统一

### 4.1 架构总览

DeepEP V2 通过 **JIT 编译 + 模板参数** 实现逻辑统一：

```
deep_ep/buffers/elastic.py
  └── ElasticBuffer
        ├── dispatch_impl<kIsScaleupNVLink=false>  → hybrid_dispatch.cuh (Normal: RDMA + NVLink)
        ├── dispatch_impl<kIsScaleupNVLink=true>   → dispatch.cuh (Low-Latency: 纯 NVLink)
        └── (num_scaleout_ranks == 1 时自动选择 direct)
```

### 4.2 统一机制：模板参数 `kIsScaleupNVLink`

```cpp
// csrc/kernels/elastic/dispatch.hpp:53-78
static std::string generate_impl(const Args& args) {
    std::string header_name, func_name;
    if (args.num_scaleout_ranks == 1) {
        // 单节点模式 (类似 Low-Latency 场景)
        header_name = "dispatch";
        func_name = fmt::format("dispatch_impl<{}, {}, ...>",
            args.is_scaleup_nvlink,  // ← true = NVLink only
            ...);
    } else {
        // 跨节点模式 (类似 Normal 场景)
        header_name = "hybrid_dispatch";
        func_name = fmt::format("hybrid_dispatch_impl<{}, {}, ...>",
            args.num_scaleout_warps, args.num_forward_warps,  // ← 有 forwarding warps
            ...);
    }
    return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>
...
)", header_name, func_name);
}
```

**关键模板参数：**

```cpp
// deep_ep/include/deep_ep/impls/dispatch.cuh:17-30
template <bool kIsScaleupNVLink,      // ← 核心模式开关
          bool kDoCPUSync,
          bool kReuseSlotIndices,
          int kNumSMs,
          int kNumNotifyWarps, int kNumDispatchWarps,
          int kNumRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles,
          ...>
__global__ void dispatch_impl(...)
```

### 4.3 Normal 等价：`hybrid_dispatch_impl` (scaleout > 1)

当 `num_scaleout_ranks > 1` 时，V2 使用 `hybrid_dispatch_impl`，对应 Blog 的 Normal 模式：

#### (1) 三阶段 Warp 角色

```cpp
// deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:14-32
template <..., int kNumScaleoutWarps, int kNumForwardWarps,  // ← Forward warps
          int kNumScaleoutRanks, int kNumScaleupRanks, ...>
__global__ void hybrid_dispatch_impl(...) {
    // Warp 角色:
    // 1. Notify warps: 元数据交换
    // 2. Scaleout warps: RDMA 发送
    // 3. Forward warps: NVLink 转发 ← 对应 Normal 的 Forwarding
}
```

#### (2) Forwarding 机制

```cpp
// deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:88-97
auto scaleup_buffer = layout::BufferLayout<false>(
    token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);
auto scaleout_send_buffer = layout::BufferLayout<false>(
    token_layout, 1, kNumMaxTokensPerRank, scaleup_buffer.get_buffer_end_ptr());
auto scaleout_recv_buffer = layout::BufferLayout<false>(
    token_layout, kNumScaleoutRanks, kNumChannels * kNumMaxTokensPerChannel, ...);
```

**与 V1 Normal 的对应关系：**
- `scaleout_send_buffer` → V1 的 RDMA send buffer
- `scaleup_buffer` → V1 的 NVLink receive buffer
- `Forward warps` → V1 的 `kRDMAAndNVLForwarder`

#### (3) Chunk 机制

```cpp
// deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh:24-27
int kNumChannelsPerSM = kNumScaleoutWarps,
int kNumChannels = kNumScaleoutWarps * kNumSMs,
int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
int kScaleoutUpdateInterval = 6,           // ← Chunk 更新间隔
int kNumSlotsPerForwardChunk = kScaleoutUpdateInterval,  // ← Chunk 大小
```

### 4.4 Low-Latency 等价：`dispatch_impl` (scaleout = 1)

当 `num_scaleout_ranks == 1` 时，V2 使用 `dispatch_impl`，对应 Blog 的 Low-Latency 模式：

#### (1) 无 Forwarding

```cpp
// deep_ep/include/deep_ep/impls/dispatch.cuh:362-393
// Issue TMA NVLink stores (直接写入，无转发)
const auto dst_ptr = stored_dst_slot_idx >= 0 ?
    gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), stored_dst_rank_idx) :
    nullptr;
if (dst_ptr != nullptr)
    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());

// Issue RDMA put (仅对非 NVLink 可达的 rank)
if constexpr (not kIsScaleupNVLink) {
    if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
        gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
    }
}
```

**关键差异：**
- `kIsScaleupNVLink = true`: 纯 NVLink，无 RDMA
- `kIsScaleupNVLink = false`: NVLink + RDMA，但**无 Forwarding warps**

#### (2) 单 Token 粒度

```cpp
// deep_ep/include/deep_ep/impls/dispatch.cuh:278-292
// 每个 Token 独立处理，无 Chunk 聚合
const auto token_start = dispatch_warp_idx * kNumSMs + sm_idx;
const auto token_stride = kNumDispatchWarps * kNumSMs;
for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {
    // Issue data TMA (单 Token)
    if (ptx::elect_one_sync()) {
        ptx::tma_load_1d(tma_buffer.get_hidden_ptr(),
                         math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                         mbarrier_ptr, kNumHiddenBytes);
    }
    // ...
}
```

### 4.5 SM 数量自动决策

```python
# deep_ep/buffers/elastic.py:728-834
@weak_lru(maxsize=None)
def get_theoretical_num_sms(self, num_experts, num_topk,
                            num_scaleout_topk: int = 0,
                            rdma_gbs: float = 0, nvlink_gbs: float = 0,
                            sm_read_gbs: float = 200, sm_write_gbs: float = 50) -> int:
    # 根据带宽模型自动决策 SM 数量
    # 考虑 rdma_traffic, nvlink_traffic, sm_read, sm_write
    # 选择瓶颈带宽，计算所需 SM 数
    num_sms = max(
        bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
        bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
    )
    num_sms = align(max(4, math.ceil(num_sms * 1.25)), 2)
    num_sms = min(num_sms, num_device_sms)
    return num_sms
```

**V2 的统一哲学：**
- 不再有显式的 `low_latency_mode` 参数
- 而是通过 `num_scaleout_ranks` 自动推断通信模式
- SM 数量由带宽模型动态计算，而非固定

### 4.6 V1 vs V2 对比

| 维度 | V1 Normal | V1 Low-Latency | V2 (scaleout > 1) | V2 (scaleout = 1) |
|------|-----------|----------------|-------------------|-------------------|
| **代码组织** | `internode.cu` | `internode_ll.cu` | `hybrid_dispatch.cuh` | `dispatch.cuh` |
| **后端** | NVSHMEM | NVSHMEM | NCCL Gin | NCCL Gin |
| **模式选择** | `low_latency_mode=true` | `low_latency_mode=false` | `num_scaleout_ranks > 1` | `num_scaleout_ranks == 1` |
| **Forwarding** | 有 | 无 | 有 (Forward warps) | 无 |
| **Chunk** | 有 | 无 | 有 (kScaleoutUpdateInterval) | 无 |
| **SM 使用** | 固定 20 | 动态 | 动态 (带宽模型) | 动态 (带宽模型) |
| **Buffer** | 分离 NVL + RDMA | 单一 RDMA | 统一 ElasticBuffer | 统一 ElasticBuffer |

---

## 5. DeepGEMM 实现：硬件消除模式分化

### 5.1 架构总览

DeepGEMM Mega MoE 采用**单一内核**设计，无模式区分：

```
deep_gemm/mega/__init__.py
  └── fp8_fp4_mega_moe / bf16_mega_moe
        └── csrc/apis/mega.hpp
              └── sm100_fp8_fp4_mega_moe (单一实现)
```

### 5.2 无模式区分的代码证据

#### (1) API 层无模式参数

```cpp
// csrc/apis/mega.hpp:157-286
static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math) {
    // 无 mode / kernel_type / is_low_latency 参数
    if (arch_major == 10) {
        sm100_fp8_fp4_mega_moe(y, ...);  // 单一实现
    }
}
```

#### (2) Heuristics 层无模式分支

```cpp
// csrc/jit_kernels/heuristics/mega_moe.hpp:183-248
static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, ...) {
    // 直接计算配置，无 if/else 模式分支
    const int block_n = 128;
    const int num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);
    // ...
}
```

### 5.3 为什么不需要双模式？

#### (1) 硬件基础：NVLink Symmetric Memory

```cpp
// sm100_fp8_fp4_mega_moe.cuh:545-550
// TMA load token from remote rank into shared memory
if (cute::elect_one_sync()) {
    ptx::tma_load_1d(
        pull_buffer.get_base_ptr(),
        sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                       current_rank_in_expert_idx),
        pull_mbarrier, kHidden);
}
```

**关键差异：**

| 维度 | DeepEP | DeepGEMM |
|------|--------|----------|
| **通信模式** | Push (源端发起) | Pull (目标端发起) |
| **通信硬件** | RDMA (IB) + NVLink | NVLink Symmetric Memory |
| **拓扑** | NIC-GPU 不对称 | NVLink 全连接 |
| **中继** | 需要 Forwarding | 无需中继 |

#### (2) Pull-based 通信消除 Forwarding

```
DeepEP Normal:
  Source GPU → IB Sending → RDMA Network → IB-to-NVLink Forwarding → NVLink Receiving → Target GPU
                                          ↑
                                   GPU 作为通信中继 (因 NIC-GPU 拓扑不对称)

DeepGEMM:
  Target GPU → NVLink Pull → Source GPU
  (直接读取远端 Symmetric Buffer，无需中继)
```

#### (3) 融合设计消除 Chunk 需求

```cpp
// sm100_fp8_fp4_mega_moe.cuh:671-675
// Wait the entire token arrival for linear 1
if (block_phase == sched::BlockPhase::Linear1) {
    const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
    const auto expected = scheduler.template get_valid_m<false>();
    while (ptx::ld_acq(ptr) != expected);  // ← Arrival Count 驱动
}
```

**Arrival Count 驱动 vs Chunk 聚合：**
- DeepEP Normal: 等待 Chunk 填满再发送 → 提高带宽利用率
- DeepGEMM: 单 Token 到达即可触发 GEMM → 无需等待聚合

#### (4) Block M 自适应（最接近"模式"的机制）

```cpp
// csrc/jit_kernels/heuristics/mega_moe.hpp:76-113
static std::tuple<int, int, int, int, int> get_block_config_for_mega_moe(...) {
    float num_expected_tokens_per_expert = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    if (num_expected_tokens_per_expert <= 8.5) {
        // RL long-tail rollout
        return {2, 16, 8, 256, 2};   // 最小 block_m
    } else if (num_expected_tokens_per_expert <= 16.5) {
        // Decoding, small batch
        return {2, 32, 16, 128, 2};
    } else if (num_expected_tokens_per_expert <= 32.5) {
        return {2, 64, 32, 128, 1};
    } else if (num_expected_tokens_per_expert <= 64.5) {
        return {2, 96, 16, 128, 2};
    } else if (num_expected_tokens_per_expert <= 96.5) {
        return {2, 128, 32, 128, 2};
    } else {
        // Prefill, or large EP decoding
        return {2, 192, 32, 128, 2};  // 最大 block_m
    }
}
```

**这不是"模式"切换，而是根据 Problem Size 的自动调整：**
- Decode (小 batch): block_m = 16~64
- Prefill (大 batch): block_m = 192
- 由 heuristics 自动选择，非用户指定

### 5.4 DeepGEMM 的"隐含 Decode 能力"

虽然 DeepGEMM 没有显式的 Decode 模式，但其设计**隐含地**对 Decode 场景有一定支持：

1. **Arrival Count 驱动**: 单 Token 到达后即可触发计算
2. **无 Chunk 聚合等待**: 不像 DeepEP Normal 需要等待 Chunk 填满
3. **Persistent Kernel**: Kernel 启动开销分摊到整个 Batch
4. **Block M 自适应**: 小 batch 时自动选择小 block_m

**但 Decode 场景下 Mega MoE 仍受限：**
- GEMM 的 M 维度很小（1 Token），Tensor Core 利用率低
- Combine 阶段的串行处理成为瓶颈（~3μs per token）
- 预分配的 `num_max_tokens_per_rank` 造成内存浪费

---

## 6. 三系统综合比较

### 6.1 设计哲学对比

```
DeepEP V1:
  Normal:     [Dispatch] → [Chunk] → [NVLink/RDMA + Forwarding] → [Combine]
  Low-Latency:[Dispatch] → [Direct RDMA]                          → [Combine]

DeepEP V2:
  scaleout>1: [Notify] → [Scaleout Send] → [Forward] → [Scaleup Receive] → [Copy Epilogue]
  scaleout=1: [Notify] → [Dispatch (NVLink or RDMA)]                           → [Copy Epilogue]

DeepGEMM:
  Single:     [Dispatch NVLink Pull] → [L1 GEMM + SwiGLU] → [L2 GEMM] → [Combine NVLink Write-back]
              └────────────────────── Persistent ────────────────────────────┘
```

### 6.2 核心维度对比表

| 维度 | DeepEP V1 Normal | DeepEP V1 LL | DeepEP V2 (scaleout>1) | DeepEP V2 (scaleout=1) | DeepGEMM |
|------|-----------------|-------------|----------------------|----------------------|----------|
| **场景** | Training/Prefill | Decode | Training/Prefill | 单节点 EP | Training/Prefill |
| **目标** | 最大化吞吐 | 最小化延迟 | 最大化吞吐 | 低延迟 | 最大化吞吐 |
| **Chunk** | 关键 | 减少 | 有 (kScaleoutUpdateInterval) | 无 | 无 |
| **Forwarding** | 有 | 无 | 有 (Forward warps) | 无 | 无此概念 |
| **流水线** | 深 (多 Stage) | 浅 (少 Stage) | 深 (多 Stage) | 浅 (少 Stage) | 固定深度 |
| **通信路径** | NVLink + RDMA | Direct RDMA | NVLink + RDMA | NVLink or RDMA | NVLink Symmetric |
| **通信模式** | Push | Push | Push | Push | Pull |
| **模式切换** | 运行时参数 | 运行时参数 | 编译期模板 | 编译期模板 | 无 |
| **硬件范围** | 跨节点 | 跨节点 | 跨节点 | 单节点 | 单节点 |
| **SM 使用** | 固定 20 | 动态 | 动态 (带宽模型) | 动态 (带宽模型) | 全部 SM |

### 6.3 性能数据对比

> 来源: DeepEP README.md

| 配置 | 硬件 | 带宽 | SM 使用 |
|------|------|------|---------|
| EP 8 x 2 (V2) | SM90 CX7 | 90 GB/s (RDMA) | 12 |
| EP 8 x 4 (V2) | SM90 CX7 | 61 GB/s (RDMA) | 6 |
| EP 8 x 2 (V2) | SM100 CX7 | 90 GB/s (RDMA) | 12 |
| EP 8 (V2) | SM100 | 726 GB/s (NVLink) | 64 (Max perf) |
| EP 8 (V2) | SM100 | 643 GB/s (NVLink) | 24 (Min #SM) |

**V2 vs V1 改进：**
- 峰值性能提升 1.3x
- SM 使用减少 4x (V1 用 24 SM, V2 用 4-6 SM)

### 6.4 模式统一的技术路径

```
DeepEP V1 (物理分离)
    │
    │ 问题: 两套代码维护成本高，模式切换不灵活
    ▼
DeepEP V2 (逻辑统一)
    │
    │ 方法: JIT 编译 + 模板参数 (kIsScaleupNVLink, num_scaleout_ranks)
    │ 效果: 同一份代码，编译期分化
    ▼
DeepGEMM (硬件消除)
    │
    │ 方法: NVLink Symmetric Memory + Pull-based 通信
    │ 效果: 硬件能力抹平模式分化，单内核覆盖全场景
    ▼
未来方向: ?
    - CFT (Compute Fabric Transport) 进一步降低延迟
    - 跨节点扩展 (DeepGEMM 当前仅支持单节点)
```

---

## 7. 跨引用分析

### 7.1 与 NVLink/RDMA (Agent 6) 的关系

**DeepEP V1/V2 的通信域划分：**

```
Agent 6 分析的拓扑:
  Intra-node:  NVLink 全连接 (对称)
  Inter-node:  RDMA (IB) (NIC-GPU 不对称)

Normal 模式必须处理:
  GPU → NVLink domain → NIC → RDMA → NVLink domain → GPU
         ↑
    Forwarding 解决 NIC-GPU 不对称

Low-Latency 模式:
  GPU → Direct RDMA → GPU
  (旁路 NVLink 域)
```

**DeepGEMM 的拓扑简化：**
- 仅使用 NVLink Symmetric Memory
- 无 NIC 拓扑问题
- Pull-based 通信无需中继

### 7.2 与 Chunk Streaming (Agent 9) 的关系

**DeepEP V1 Normal 的 Chunk：**

```cpp
// Token 流 → Chunk → 网络传输
// Token 是调度粒度，Chunk 是通信粒度
Token Stream → Chunk → Network Transfer
```

**DeepEP V2 的 Chunk：**

```cpp
// hybrid_dispatch.cuh 中的 Chunk
int kScaleoutUpdateInterval = 6,           // Chunk 更新间隔
int kNumSlotsPerForwardChunk = kScaleoutUpdateInterval,  // Chunk 大小
```

**DeepGEMM 的"反 Chunk"：**
- Arrival Count 驱动，单 Token 到达即可触发 GEMM
- 无需 Chunk 聚合等待
- Combine 阶段的 Hidden 维度分块 (kNumChunks = 1~2) 是**寄存器容量约束**，非通信优化

### 7.3 与 Buffer (Agent 2) 的关系

**DeepEP V1 的 Buffer 系统：**

```
Normal:
  Token Buffer → Dispatch Buffer → Chunk Buffer → NVLink / RDMA Pipeline → Receive Buffer → Expert Buffer

Low-Latency:
  Token Buffer → Direct RDMA → Receive Buffer → Expert Buffer
```

**DeepEP V2 的 ElasticBuffer：**

```python
# 统一 Buffer，支持多种后端
class ElasticBuffer:
    """
    支持:
    - high-throughput expert-parallel all-to-all (NVLink + RDMA)
    - Engram (remote KV cache fetch, RDMA)
    - pipeline-parallel send/recv (PP, NVLink)
    - all-gather reduce-scatter (AGRS, NVLink)
    """
```

**DeepGEMM 的 SymmBuffer：**

```python
# deep_gemm/mega/__init__.py:18-66
class SymmBuffer:
    """NVLink Symmetric Memory Buffer"""
    # 单一 Buffer，包含所有输入/输出视图
    # x, x_sf, topk_idx, topk_weights
    # l1_acts, l1_acts_sf, l2_acts, l2_acts_sf
```

---

## 8. 深层洞察

### 8.1 模式分化的根本原因

**Normal 和 Low-Latency 的分化源于硬件拓扑的不对称：**

```
Normal 模式存在的前提:
  1. NIC-GPU 拓扑不对称 (部分 GPU 直连 NIC)
  2. 需要 Forwarding 中继
  3. 需要 Chunk 聚合提高带宽利用率

Low-Latency 模式存在的前提:
  1. 每个 GPU 可直接访问 RDMA (或旁路 Forwarding)
  2. 单 Token 延迟敏感，不能等待 Chunk 聚合
  3. 浅流水线减少 Stage 延迟
```

### 8.2 DeepGEMM 消除模式分化的条件

DeepGEMM 能使用单内核的前提条件：

1. **NVLink 全连接**: 所有 GPU 可直接访问所有其他 GPU 的 Symmetric Memory
2. **无 NIC 拓扑问题**: 单节点内通信，无需 RDMA
3. **Pull-based 通信**: 目标端发起读取，无需源端推送 + 中继
4. **通信-计算融合**: NVLink Pull 和 GEMM 在 SM 级别交替执行

### 8.3 未来演进方向

**如果 DeepGEMM 要支持 Decode 场景：**

1. **引入 Low-Latency 变体**: 减少流水线深度，优化单 Token 路径
2. **跨节点支持**: 扩展到 RDMA 环境 (需要重新引入 Forwarding)
3. **Dynamic Shape**: 支持可变 Token 数而无需预分配
4. **KV Cache 融合**: 与 Attention 内核融合

**如果 DeepEP 要简化双模式：**

1. **硬件层面**: NVLink-RDMA 统一 fabric (如 NVLink-C2C + RDMA 融合)
2. **软件层面**: 更精细的 SM 动态分配 (已有 `get_theoretical_num_sms`)
3. **算法层面**: 自适应 Chunk 大小 (Decode 时 Chunk=1, Training 时 Chunk=N)

---

## 9. 源码索引

### DeepEP V1

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `csrc/kernels/legacy/internode.cu` | 87-90 | `kLowLatencyMode` 模板参数：RDMA rank 转换 |
| `csrc/kernels/legacy/internode.cu` | 487-516 | Warp 角色分配 (5 种角色) |
| `csrc/kernels/legacy/internode.cu` | 527-529 | Chunk Buffer 定义 |
| `csrc/kernels/legacy/internode.cu` | 583-585 | Forwarding 同步机制 |
| `csrc/kernels/legacy/internode.cu` | 809-817 | Chunk 粒度发送 |
| `csrc/kernels/legacy/internode_ll.cu` | 128-155 | Low-Latency dispatch kernel |
| `csrc/kernels/legacy/internode_ll.cu` | 179-182 | 单 Token 消息格式 |
| `csrc/kernels/legacy/internode_ll.cu` | 362-370 | 按 Expert 组织的接收端 |
| `csrc/kernels/legacy/internode_ll.cu` | 383-408 | 超时与 Mask 机制 |
| `csrc/legacy/config.hpp` | 24-51 | Config 结构 (双 Buffer) |
| `csrc/legacy/config.hpp` | 102-183 | LowLatencyLayout |
| `deep_ep/buffers/legacy.py` | 37-93 | Buffer 类 (low_latency_mode 参数) |

### DeepEP V2

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `csrc/kernels/elastic/dispatch.hpp` | 53-78 | `generate_impl` 模式选择逻辑 |
| `csrc/kernels/elastic/dispatch.hpp` | 186-196 | `num_scaleout_ranks == 1` 分支 |
| `deep_ep/include/deep_ep/impls/dispatch.cuh` | 17-30 | `kIsScaleupNVLink` 模板参数 |
| `deep_ep/include/deep_ep/impls/dispatch.cuh` | 278-292 | 单 Token 处理循环 |
| `deep_ep/include/deep_ep/impls/dispatch.cuh` | 362-393 | NVLink Store + RDMA Put |
| `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` | 14-32 | Hybrid dispatch 模板参数 |
| `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` | 88-97 | Forwarding buffer 定义 |
| `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` | 24-27 | Chunk 参数 |
| `deep_ep/buffers/elastic.py` | 728-834 | `get_theoretical_num_sms` |
| `deep_ep/buffers/elastic.py` | 836-853 | `get_theoretical_num_qps` |

### DeepGEMM

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `csrc/apis/mega.hpp` | 157-286 | C++ 单一入口，无模式分支 |
| `deep_gemm/mega/__init__.py` | 153-176 | Python 单一 API |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | 76-113 | Block M 自适应 (非模式) |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | 115-181 | Pipeline 深度由 smem 决定 |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | 183-248 | Heuristics 无模式区分 |

---

## 10. 总结

### 10.1 三种实现的设计哲学

| 系统 | 哲学 | 优势 | 劣势 |
|------|------|------|------|
| **DeepEP V1** | 物理分离 | 各模式独立优化，性能极致 | 代码冗余，维护成本高 |
| **DeepEP V2** | 逻辑统一 | 一份代码，编译期分化 | JIT 编译开销，首次调用延迟 |
| **DeepGEMM** | 硬件消除 | 单内核，零模式开销 | 仅支持单节点，无 Decode 优化 |

### 10.2 核心结论

1. **Blog 的两种通信哲学在 DeepEP V1 中精确映射**：Normal = `internode.cu` (kLowLatencyMode=false)，Low-Latency = `internode_ll.cu`

2. **DeepEP V2 通过 JIT + 模板参数实现逻辑统一**：`kIsScaleupNVLink` 和 `num_scaleout_ranks` 在编译期选择代码路径，不再需要运行时 `low_latency_mode` 参数

3. **DeepGEMM 通过硬件能力消除模式分化**：NVLink Symmetric Memory + Pull-based 通信使得 Forwarding 和 Chunk 聚合失去意义，单内核即可覆盖 Training/Prefill 场景

4. **模式分化的根本原因是硬件拓扑不对称**：NIC-GPU 不对称 → 需要 Forwarding；带宽利用率需求 → 需要 Chunk 聚合。当硬件全连接时（如 NVLink Symmetric Memory），这些需求消失

5. **DeepGEMM 不是 DeepEP 的替代品，而是特定硬件约束下的特化演进**：单节点、Training 导向、无 Decode 优化，但在其设计域内达到极致效率

---

*分析基于 DeepEP 源码、DeepGEMM 源码和 blog 文本，代码路径：*
- *DeepEP: `/Users/backyes/work/claude_workspace/deepgemm_research/`*
- *DeepGEMM: `/Users/backyes/work/claude_workspace/deepgemm_research/DeepGEMM/`*
