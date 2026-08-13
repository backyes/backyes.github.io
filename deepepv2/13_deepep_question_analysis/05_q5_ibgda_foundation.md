# Q5: IBGDA是DeepEP的共同底座，Normal/Low-latency区别是pipeline策略

## 1. 问题审视

**核心论断**：
> Normal和Low-latency不是"GPU通信 vs NIC通信"，而是"通信pipeline组织方式不同"。两者都使用IBGDA。

## 2. 源码级验证

### 2.1 IBGDA的定义

**文件**：`csrc/kernels/legacy/ibgda_device.cuh`

```cpp
// L1-15: 文件头部注释
// Portions derived from NVSHMEM
// Modified from original source:
//  nvshmem/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh

// L77-79: IBGDA device state
__device__ static __forceinline__ nvshmemi_ibgda_device_state_t* ibgda_get_state() {
    return &nvshmemi_ibgda_device_state_d;
}
```

**IBGDA** = InfiniBand GPU Direct Async
- 允许GPU kernel**直接**构造WQE (Work Queue Element)
- 写入NIC的doorbell buffer
- 绕过CPU proxy

### 2.2 Normal模式使用IBGDA

**文件**：`csrc/kernels/legacy/internode.cu` (L615-625)

```cpp
// Normal模式的RDMA send
if (dst_rdma_rank != rdma_rank) {
    nvshmemi_ibgda_put_nbi_warp<true>(
        reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
        reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
        sizeof(int) * (LEGACY_NUM_MAX_NVL_PEERS * 2 + 2),
        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
        channel_id, lane_id, 0);
}
```

### 2.3 Low-Latency模式使用IBGDA

**文件**：`csrc/kernels/legacy/internode_ll.cu` (L253-278)

```cpp
// Low-Latency模式的RDMA send
if (dst_p2p_ptr == 0) {
    nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, 
                                dst_rank, dst_expert_local_idx, lane_id, slot_idx);
} else {
    // NVLink P2P
    UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, 
                       ld_nc_global, st_na_global);
}
```

### 2.4 两种模式的共同点

| 维度 | Normal | Low-Latency |
|------|--------|-------------|
| RDMA发起 | GPU kernel (IBGDA) | GPU kernel (IBGDA) |
| 底层传输 | NVLink / RDMA | NVLink / RDMA |
| 编程接口 | nvshmemi_ibgda_put_nbi_warp | nvshmemi_ibgda_put_nbi_warp |
| WQE构造 | GPU直接写 | GPU直接写 |

### 2.5 两种模式的区别

| 维度 | Normal | Low-Latency |
|------|--------|-------------|
| 数据聚合 | chunk聚合后发送 | 立即发送 |
| 转发 | RDMA→NVLink forwarding | 直接RDMA |
| Buffer层级 | 多级buffer pipeline | 单级buffer |
| 同步 | 复杂的多阶段同步 | 简单的flag轮询 |
| 目标 | 最大化带宽利用率 | 最小化首token延迟 |

### 2.6 代码中的模板参数

**文件**：`csrc/kernels/legacy/internode.cu` (L446)

```cpp
template <bool kLowLatencyMode, int kNumRDMARanks, ...>
__global__ void dispatch(...)
```

`kLowLatencyMode` 是一个**编译期模板参数**，说明两种模式共享大部分代码，只是行为不同。

## 3. 结论

**论断验证**：✅ 正确

Normal和Low-latency**都使用IBGDA**作为底层通信机制：
1. **共同底座**：NVSHMEM + IBGDA + GPU-initiated RDMA
2. **区别在于策略**：
   - Normal：chunk聚合 + 多级pipeline + forwarding
   - Low-Latency：立即发送 + 直接路径 + 简单同步

这不是"不同通信技术"，而是"同一技术的不同pipeline组织"。

