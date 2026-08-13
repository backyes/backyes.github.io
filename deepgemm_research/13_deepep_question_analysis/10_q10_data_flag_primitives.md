# Q10: Data与Flag两套独立的读写原语体系

## 1. 问题审视

**核心论断**：
> DeepEP有Data和Flag两套独立的读写原语体系：
> - Data：吞吐优先，读写各一次
> - Flag：正确性优先，可能被反复轮询

## 2. 源码级验证

### 2.1 Data原语

**文件**：`csrc/kernels/legacy/utils.cuh`

#### ld_nc_global (Non-Cache load)

```cpp
// L177-181
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

// L184-188
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t* ptr) {
    auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
    return *reinterpret_cast<dtype_t*>(&ret);
}

// L199-203: int特化
template <>
__device__ __forceinline__ int ld_nc_global(const int* ptr) {
    int ret;
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}
```

**PTX指令**：`ld.global.nc.L1::no_allocate.L2::256B`
- `.nc` = non-cacheable（绕过L1）
- `.L1::no_allocate` = 不在L1分配空间
- `.L2::256B` = L2使用256B粒度预取

#### st_na_global (No-Allocate store)

```cpp
// L268-272
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

// L281-283
template <>
__device__ __forceinline__ void st_na_global(const int* ptr, const int& value) {
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}
```

**PTX指令**：`st.global.L1::no_allocate`
- `.L1::no_allocate` = 不在L1分配，直接写L2

### 2.2 Flag原语

#### ld_acquire_global / ld_acquire_sys_global

```cpp
// L105-109: GPU scope acquire
__device__ __forceinline__ int ld_acquire_global(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

// L93-97: System scope acquire
__device__ __forceinline__ int ld_acquire_sys_global(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}
```

**PTX指令**：`ld.acquire.sys.global.s32`
- `.acquire` = acquire语义，保证后续读写不会被重排到此前
- `.sys` = system scope（跨设备可见）

#### atomic_add_release_global

```cpp
// L117-121
__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}
```

**PTX指令**：`atom.add.release.gpu.global.s32`
- `.release` = release语义，保证之前的写操作对此后的acquire可见

### 2.3 两套原语的对比

| 原语 | Data | Flag |
|------|------|------|
| 读 | `ld.global.nc.L1::no_allocate` | `ld.acquire.sys.global` |
| 写 | `st.global.L1::no_allocate` | `atom.add.release.sys.global` |
| 目标 | 吞吐优化 | 正确性保证 |
| L1策略 | no_allocate（不污染L1） | 正常缓存 |
| 访问次数 | 各一次 | 可能反复轮询 |
| 作用域 | gpu（本地） | sys（跨设备） |

### 2.4 使用场景

#### Data使用

**文件**：`csrc/kernels/legacy/internode.cu` (L694)

```cpp
// 读取token数据
UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, 
                   ld_nc_global, st_broadcast);
```

#### Flag使用

**文件**：`csrc/kernels/legacy/internode_ll.cu` (L330)

```cpp
// 轮询等待发送完成
while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) 
       != LEGACY_FINISHED_SUM_TAG * 2)
    ;
```

**文件**：`csrc/kernels/legacy/internode_ll.cu` (L387)

```cpp
// 轮询等待接收完成
while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0
       && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES)
    ;
```

### 2.5 Release-Acquire配对

**发送端**（release）：
```cpp
// L277: 发送完成后atomic_add_release
lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
```

**接收端**（acquire）：
```cpp
// L330: 轮询等待release
while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) 
       != LEGACY_FINISHED_SUM_TAG * 2)
    ;
```

**语义保证**：
1. release之前的所有写操作，对acquire之后的所有读操作可见
2. 这建立了happens-before关系

## 3. 结论

**论断验证**：✅ 正确

DeepEP确实有两套独立的读写原语：

1. **Data原语**：
   - `ld_nc_global` / `st_na_global`
   - 目标：最大化吞吐
   - 策略：绕过L1、L2 256B预取

2. **Flag原语**：
   - `ld_acquire_global` / `atomic_add_release_global`
   - 目标：保证正确性
   - 策略：acquire-release语义配对

3. **分离的好处**：
   - Data不需要正确性保证（flag已经保证）
   - Flag不需要高吞吐（只传输几个int）

