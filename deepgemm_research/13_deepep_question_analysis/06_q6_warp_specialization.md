# Q6: Warp Specialization — warp绑定通信阶段而非token

## 1. 问题审视

**核心论断**：
> DeepEP不是"一个warp负责一个token"，而是"warp是pipeline worker，token是流动数据"。

## 2. 源码级验证

### 2.1 WarpRole枚举

**文件**：`csrc/kernels/legacy/internode.cu` (L487)

```cpp
enum class WarpRole { 
    kRDMASender,              // RDMA发送者
    kRDMASenderCoordinator,   // RDMA发送协调者
    kRDMAAndNVLForwarder,     // RDMA接收+NVLink转发
    kForwarderCoordinator,    // 转发协调者
    kNVLReceivers             // NVLink接收者
};
```

**5种WarpRole**，对应通信pipeline的5个阶段。

### 2.2 warp角色分配

**文件**：`csrc/kernels/legacy/internode.cu` (L499-513)

```cpp
const auto role_meta = [=]() -> std::pair<WarpRole, int> {
    if (is_forwarder) {
        if (warp_id < LEGACY_NUM_MAX_NVL_PEERS) {
            return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % LEGACY_NUM_MAX_NVL_PEERS};
        } else {
            return {WarpRole::kForwarderCoordinator, warp_id - LEGACY_NUM_MAX_NVL_PEERS};
        }
    } else if (warp_id < kNumDispatchRDMASenderWarps) {
        return {WarpRole::kRDMASender, -1};
    } else if (warp_id == kNumDispatchRDMASenderWarps) {
        return {WarpRole::kRDMASenderCoordinator, -1};
    } else {
        return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % LEGACY_NUM_MAX_NVL_PEERS};
    }
}();
```

**关键**：warp角色由 `warp_id` 和 `sm_id` 决定，**与token无关**。

### 2.3 各角色的职责

#### RDMA Sender (L587-757)
```cpp
if (warp_role == WarpRole::kRDMASender) {
    // 遍历token，写入send buffer
    for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
        // 读取x → 写入rdma_channel_data.send_buffer
        UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, 
                           ld_nc_global, st_broadcast);
    }
}
```

#### RDMA Sender Coordinator (L758-848)
```cpp
if (warp_role == WarpRole::kRDMASenderCoordinator) {
    // 轮询rdma_send_channel_tail
    // 当积累足够token后，发起RDMA put
    while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
        // Issue RDMA send
        nvshmemi_ibgda_put_nbi_warp<true>(...);
    }
}
```

#### Forwarder (L849-1013)
```cpp
if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
    // 从RDMA buffer读取 → 写入NVL buffer
    while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
        // Forward tokens from RDMA buffer to NVL buffer
        tma_load_1d(tma_buffer, shifted, tma_mbarrier, num_bytes_per_token, false);
        tma_store_1d(tma_buffer, dst_shifted, num_bytes_per_token);
    }
}
```

#### NVL Receiver (L1061-1195)
```cpp
if (warp_role == WarpRole::kNVLReceivers) {
    // 从NVL buffer读取 → 写入最终的recv_x
    while (num_tokens_to_recv > 0) {
        tma_load_1d(tma_buffer, shifted, tma_mbarrier, tma_load_bytes);
        tma_store_1d(tma_buffer, recv_x + recv_token_idx * hidden_int4, hidden_bytes);
    }
}
```

### 2.4 Warp分配常量

**文件**：`csrc/kernels/legacy/internode.cu` (L1253-1254)

```cpp
constexpr int kNumDispatchRDMASenderWarps = 7;
constexpr int kNumTMABytesPerWarp = 16384;
```

总warp数：
```cpp
// L516
EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + LEGACY_NUM_MAX_NVL_PEERS);
// = 7 + 1 + 8 = 16 warps = 512 threads
```

### 2.5 与"一个warp一个token"的对比

| 模式 | 描述 | DeepEP实际 |
|------|------|-----------|
| Token-centric | 一个warp绑定一个token的全生命周期 | ❌ 不是 |
| **Pipeline-centric** | 一个warp绑定一个通信阶段，token流经所有warp | ✅ 是 |

### 2.6 Low-Latency模式的warp分配

**文件**：`csrc/kernels/legacy/internode_ll.cu` (L494-496)

```cpp
const int num_warp_groups = ceil_div(num_experts, num_device_sms);
const int num_warps_per_group = 32 / num_warp_groups;
```

Low-Latency按**expert**分配warp group：
```cpp
const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
```

## 3. 结论

**论断验证**：✅ 正确

DeepEP的warp specialization是**pipeline-centric**：
1. **Normal模式**：5种WarpRole，每个warp负责一个通信阶段
2. **Low-Latency模式**：按expert分配warp group
3. **Token流动**：token依次经过sender → forwarder → receiver

这与"一个warp绑定一个token"的模型完全不同。

