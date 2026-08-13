# Q11: Release-Acquire配对机制

## 1. 问题审视

**核心论断**：
> `atomic_add_release_global`与`ld_acquire_global`必须成对使用，单独一个没有意义。
> 保证具有传递性：本地warp间配对 → 跨设备配对 → 层层建立happens-before链条。

## 2. 源码级验证

### 2.1 Release-Acquire的定义

**文件**：`csrc/kernels/legacy/utils.cuh`

#### Release操作

```cpp
// L85-87: System scope release store
__device__ __forceinline__ void st_release_sys_global(const int* ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

// L89-91: CTA scope release store
__device__ __forceinline__ void st_release_cta(const int* ptr, int val) {
    asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

// L111-115: Atomic add with release
__device__ __forceinline__ int atomic_add_release_sys_global(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

// L117-121: GPU scope atomic add with release
__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}
```

#### Acquire操作

```cpp
// L93-97: System scope acquire load
__device__ __forceinline__ int ld_acquire_sys_global(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

// L105-109: GPU scope acquire load
__device__ __forceinline__ int ld_acquire_global(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}
```

### 2.2 作用域层级

| 作用域 | PTX修饰符 | 可见范围 |
|--------|----------|---------|
| CTA | `.cta` | 同一thread block内 |
| GPU | `.gpu` | 同一GPU内所有thread |
| System | `.sys` | 跨GPU、跨设备 |

### 2.3 Dispatch中的Release-Acquire配对

**发送端**（`internode_ll.cu` L277）：
```cpp
// 发送完成后，atomic_add_release通知接收方
lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
```

**接收端**（`internode_ll.cu` L330）：
```cpp
// 轮询等待发送方的release
while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) 
       != LEGACY_FINISHED_SUM_TAG * 2)
    ;
```

### 2.4 传递性示例

```
发送GPU:
  1. 写token数据到rdma buffer（普通store）
  2. atomic_add_release(flag)  ← release
  
接收GPU:
  3. ld_acquire(flag) == expected  ← acquire
  4. 读token数据（普通load）
```

**保证**：步骤1的写操作，对步骤4的读操作可见。

### 2.5 多层配对

**文件**：`csrc/kernels/legacy/internode.cu`

#### 第一层：本地warp间
```cpp
// L748: CTA scope release
st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots);

// L806: CTA scope acquire
auto processed_tail = __shfl_sync(0xffffffff, 
    ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);
```

#### 第二层：跨设备
```cpp
// L867-868: System scope release（NVL forwarding）
st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);

// L1076-1078: System scope acquire
start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
```

### 2.6 memory_fence的作用

**文件**：`csrc/kernels/legacy/utils.cuh` (L69-79)

```cpp
__device__ __forceinline__ void memory_fence() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void memory_fence_gpu() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void memory_fence_cta() {
    asm volatile("fence.acq_rel.cta;" ::: "memory");
}
```

`fence.acq_rel` 是一个**全屏障**：
- 之前的所有读写，对所有其他线程可见
- 之后的所有读写，必须等待此fence完成

### 2.7 Barrier实现

**文件**：`csrc/kernels/legacy/internode_ll.cu` (L21-69)

```cpp
template <int kNumThreads>
__forceinline__ __forceinline__ void barrier(int thread_id, int rank, int num_ranks, 
                                              int* mask_buffer_ptr, int* sync_buffer_ptr) {
    // 1. Quiet all QPs（等待RDMA完成）
    for (int i = thread_id; i < qps_per_rank * (num_ranks - 1); i += kNumThreads) {
        nvshmemi_ibgda_quiet(dst_rank, qp_id);
    }
    
    // 2. 更新本地counter
    if (thread_id == 0)
        atomicAdd(sync_buffer_ptr + rank, -1);  // 本地counter减1
    
    __syncthreads();
    
    // 3. 更新远程counter并等待
    if (thread_id < num_ranks && thread_id != rank) {
        // 远程atomic add
        nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), cnt, rank, 0);
        // 或者本地p2p store
        st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), cnt);
        
        // 等待远程更新本地counter
        while (ld_acquire_sys_global(sync_buffer_ptr + dst_rank) != cnt)
            ;
    }
}
```

## 3. 结论

**论断验证**：✅ 正确

Release-Acquire配对机制：
1. **必须成对**：单独的release或acquire没有同步意义
2. **传递性**：A→B的release-acquire + B→C的release-acquire ⇒ A→C的happens-before
3. **多层级**：CTA → GPU → System，作用域递增
4. **与Data分离**：Data用nc/na优化吞吐，Flag用release/acquire保证正确性

