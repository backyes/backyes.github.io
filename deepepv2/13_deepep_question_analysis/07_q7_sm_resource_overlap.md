# Q7: SM资源占用与Overlap机制

## 1. 问题审视

**核心论断**：
> GPU不能跨kernel共享SM执行，DeepEP overlap依靠NIC offload和通信kernel轻量化。
> "通信kernel短时间占SM，真正数据搬运由NIC完成，NIC与GPU计算并行。"

## 2. 源码级验证

### 2.1 通信kernel的SM占用

**文件**：`csrc/kernels/legacy/internode.cu` (L1253)

```cpp
// Normal dispatch kernel launch config
constexpr int kNumDispatchRDMASenderWarps = 7;
// 总warp数 = 7 + 1 + 8 = 16 warps = 512 threads per block
```

**文件**：`csrc/kernels/legacy/internode_ll.cu` (L128-129)

```cpp
template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch(...) {
    // Low-Latency: 1024 threads = 32 warps per block
    // minBlocksPerMultiprocessor = 1
```

### 2.2 通信kernel的实际计算量

**RDMA Sender的计算**（L587-757）：
```cpp
if (warp_role == WarpRole::kRDMASender) {
    for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
        // 只做数据搬运：读x → 写send buffer
        UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, 
                           ld_nc_global, st_broadcast);
    }
}
```

**关键**：
- 不使用Tensor Core
- 不使用ALU进行复杂计算
- 只有load/store操作

### 2.3 Overlap机制

#### (a) 通信kernel的"轻量"特性

```cpp
// L694-728: RDMA sender只做数据搬运
// 没有GEMM、没有复杂计算
```

#### (b) NIC offload

```cpp
// L617-624: 发起RDMA后立即返回
nvshmemi_ibgda_put_nbi_warp<true>(...);  // non-blocking
// GPU不需要等待NIC完成，可以继续执行其他warp
```

#### (c) Python层面的hook

**文件**：`deep_ep/buffers/legacy.py` (搜索 `hook` 相关代码)

```python
# DeepEP的hook-based overlap
# 在GEMM执行期间穿插通信kernel
```

### 2.4 SM占用分析

| 资源 | 通信kernel | GEMM kernel |
|------|-----------|-------------|
| Tensor Core | 不使用 | 大量使用 |
| ALU | 少量（地址计算） | 中等 |
| Load/Store | 大量 | 少量 |
| Registers | 中等 | 大量 |
| Shared Memory | 用于TMA buffer | 用于tile |

### 2.5 为什么能overlap

```
时间轴：
GEMM warp:  [====计算====][====计算====][====计算====]
通信warp:   [load][store]     [poll flag]    [load][store]
NIC:        [====RDMA传输====][====RDMA传输====]
```

1. **通信warp**：快速完成load/store，然后poll flag等待NIC
2. **NIC**：独立执行RDMA传输
3. **GEMM warp**：在通信warp等待NIC期间执行计算

### 2.6 代码中的同步原语

**文件**：`csrc/kernels/legacy/utils.cuh` (L153-157)

```cpp
// flag轮询使用volatile
__device__ __forceinline__ int ld_volatile_global(const int* ptr) {
    int ret;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}
```

**volatile** 确保每次读取都从memory获取，不会被缓存。

## 3. 结论

**论断验证**：✅ 正确

DeepEP的overlap机制：
1. **通信kernel轻量**：只做load/store，不用Tensor Core
2. **NIC offload**：RDMA传输由NIC独立完成
3. **warp级并行**：通信warp和GEMM warp分时复用SM
4. **不是零代价**：通信warp仍占用寄存器、shared memory等资源

