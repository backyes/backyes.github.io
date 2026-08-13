# Q3: Dispatch/Combine 为什么难

## 1. 问题审视

**核心论断**：
> Dispatch方向是 token owner → expert GPU（many-to-many）
> Combine方向是 expert output → original token（需要reduce/accumulate）

## 2. 源码级验证

### 2.1 Dispatch的复杂性

**文件**：`csrc/kernels/legacy/internode.cu` (L446-757)

Dispatch需要处理的问题：

#### (a) 多目标路由
```cpp
// L671-685: 一个token可能需要发送到多个RDMA rank
SourceMeta src_meta;
int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];
void* dst_send_buffers[kNumTopkRDMARanks];

for (int i = 0, slot_idx; i < kNumRDMARanks; ++i)
    if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
        slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
        topk_ranks[num_topk_ranks] = i;
        dst_send_buffers[num_topk_ranks++] = ...;
    }
```

#### (b) Buffer管理
```cpp
// L526-560: 复杂的buffer分配
auto rdma_channel_data = SymBuffer<uint8_t>(
    rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, 
    kNumRDMARanks, channel_id, num_channels);
auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, 
    LEGACY_NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);
```

#### (c) 流控与同步
```cpp
// L647-663: 等待远端buffer释放
while (is_token_in_rank_uint64 != 0 and 
       rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
    cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));
    // Timeout check...
}
```

### 2.2 Combine的额外复杂性

**文件**：`csrc/kernels/legacy/internode.cu` (combine kernel)

Combine比Dispatch更复杂，因为：

#### (a) 需要reduce/accumulate
```cpp
// combine kernel中：
// 同一个token可能被多个expert处理（top-k），需要加权求和
// recv_topk_weights用于加权
```

#### (b) 反向路由
Dispatch时记录了 `send_rdma_head` 和 `send_nvl_head`：
```cpp
// L667-668: 保存RDMA head用于combine
if (lane_id < kNumRDMARanks and not kCachedMode)
    send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;
```

Combine时需要**逆向**使用这些信息。

### 2.3 代码量对比

| 组件 | 代码行数 | 复杂度指标 |
|------|---------|-----------|
| `internode.cu` (dispatch+combine) | 2384行 | 5种WarpRole |
| `internode_ll.cu` (low-latency) | 1289行 | 2种模式 |
| `intranode.cu` (NVLink only) | ~500行 | 1种模式 |

## 3. 核心难点总结

### Dispatch难在：
1. **多目标**：每个token可能有top-k个目标
2. **元数据管理**：需要记录src_info供combine使用
3. **RDMA流控**：避免远端buffer溢出
4. **NVL转发**：RDMA→NVLink的forwarder协调

### Combine难在：
1. **反向路由**：需要逆向解析dispatch时的路由决策
2. **加权求和**：top-k weights的accumulate
3. **数据还原**：expert output → token order
4. **对称性**：combine kernel是dispatch kernel的"镜像"但逻辑更复杂

## 4. 结论

**论断验证**：✅ 正确

Dispatch/Combine的困难主要来自：
1. **动态稀疏路由**：目标地址由topk_idx动态决定
2. **多层级网络**：RDMA + NVLink的层级转发
3. **双向通信**：dispatch和combine需要共享路由信息
4. **异步流控**：无全局barrier下的正确性保证

