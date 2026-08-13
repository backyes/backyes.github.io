# Q12: V1→V2架构演进 — NVSHMEM到NCCL GIN

## 1. 问题审视

**核心论断**：
> 架构思路（GPU发起写、symmetric memory、push+signal模型）被完整保留甚至标准化，但底层PTX级别的微观优化下沉进了NCCL自己的代码库。

## 2. 源码级验证

### 2.1 V1架构（Legacy）

**文件结构**：
```
csrc/kernels/
├── legacy/
│   ├── internode.cu        # Normal dispatch/combine (2384行)
│   ├── internode_ll.cu     # Low-Latency dispatch/combine (1289行)
│   ├── intranode.cu        # NVLink-only
│   ├── ibgda_device.cuh    # IBGDA PTX级实现 (496行)
│   ├── buffer.cuh          # Symmetric buffer定义
│   ├── utils.cuh           # PTX原语
│   └── compiled.cuh        # 编译配置
└── backend/
    └── nvshmem.cu          # NVSHMEM封装
```

**V1特点**：
1. **直接操作IBGDA**：GPU kernel直接构造WQE
2. **PTX级优化**：手工编写`ld.global.nc`、`st.release`等
3. **NVSHMEM依赖**：symmetric memory由NVSHMEM管理

### 2.2 V2架构（Elastic）

**文件结构**：
```
csrc/
├── elastic/
│   ├── buffer.hpp          # ElasticBuffer (1382行)
│   ├── utils.hpp
│   └── api.hpp
├── kernels/
│   ├── elastic/
│   │   ├── dispatch.hpp    # JIT dispatch
│   │   ├── combine.hpp     # JIT combine
│   │   ├── barrier.hpp     # GPU barrier
│   │   ├── engram.hpp      # Engram支持
│   │   └── pp_send_recv.hpp # PP支持
│   └── backend/
│       ├── nccl.cu         # NCCL GIN封装
│       ├── symmetric.hpp   # Symmetric memory
│       ├── api.cuh
│       └── cuda_driver.cu
└── jit/
    ├── compiler.hpp        # JIT编译器
    ├── kernel_runtime.hpp
    └── launch_runtime.hpp
```

**V2特点**：
1. **JIT编译**：kernel参数在运行时确定
2. **NCCL GIN**：底层RDMA由NCCL管理
3. **统一API**：`ElasticBuffer` 统一 Normal/Low-Latency
4. **扩展功能**：Engram、PP、CP支持

### 2.3 架构映射

| V1 (NVSHMEM) | V2 (NCCL GIN) | 映射关系 |
|-------------|---------------|---------|
| `nvshmemi_ibgda_put_nbi_warp` | NCCL GIN put | GPU→NIC的直接写入 |
| `nvshmem_align` | `ncclMemAlloc` | Symmetric memory分配 |
| `nvshmem_barrier_all` | NCCL window barrier | 跨GPU同步 |
| `nvshmemi_ibgda_quiet` | NCCL quiet | QP完成等待 |
| PTX WQE构造 | NCCL内部 | 下沉到NCCL |

### 2.4 保留的核心思想

#### (a) GPU发起通信

**V1**：
```cpp
// ibgda_device.cuh L128-141
__device__ static __forceinline__ void ibgda_post_send(...) {
    // GPU直接写doorbell
    ibgda_update_dbr(qp, new_prod_idx);
    ibgda_ring_db(qp, new_prod_idx);
}
```

**V2**：
```cpp
// 通过NCCL GIN API，但仍然是GPU发起
// nccl.cu L86-108: NCCL GIN context创建
reqs.ginContextCount = num_allocated_qps;
ncclDevCommCreate(comm, &reqs, ...);
```

#### (b) Push + Signal模型

**V1**：
```cpp
// internode_ll.cu L266-277
nvshmemi_ibgda_put_nbi_warp(...);  // Push数据
atomic_add_release_global(...);     // Signal完成
```

**V2**：
```cpp
// 同样的push+signal模型，但通过NCCL API
```

#### (c) Symmetric Memory

**V1**：
```cpp
// buffer.cuh L95-130
template <typename dtype_t, bool kDecoupled = true>
struct SymBuffer {
    uint8_t* send_ptr;
    uint8_t* recv_ptr;
};
```

**V2**：
```cpp
// symmetric.hpp L123-140
class GPUSymmetricMemory final : public SymmetricMemory {
    explicit GPUSymmetricMemory(const int64_t& num_bytes) {
        NCCL_CHECK(ncclMemAlloc(&ptr, num_bytes));
    }
};
```

### 2.5 下沉到NCCL的部分

**V1中DeepEP直接控制的**：
1. WQE构造（`ibgda_write_rdma_write_wqe`）
2. Doorbell ringing（`ibgda_ring_db`）
3. QP management（`ibgda_get_rc`）
4. DBREC更新（`ibgda_update_dbr`）

**V2中NCCL控制的**：
1. 以上所有细节
2. DeepEP只调用NCCL API

### 2.6 V2的新能力

#### (a) JIT编译

**文件**：`csrc/kernels/elastic/dispatch.hpp` (L51-89)

```cpp
static std::string generate_impl(const Args& args) {
    if (args.num_scaleout_ranks == 1) {
        func_name = fmt::format("dispatch_impl<{}, {}, {}, ...>",
            args.is_scaleup_nvlink,
            args.do_cpu_sync,
            args.num_notify_warps, ...);
    } else {
        func_name = fmt::format("hybrid_dispatch_impl<{}, {}, {}, ...>",
            ...);
    }
    return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
}
```

#### (b) EPHandle缓存

**文件**：`deep_ep/buffers/elastic.py` (L25-57)

```python
class EPHandle:
    """
    Communication handle returned by `ElasticBuffer.dispatch`.
    Can be reused as a cached handle in subsequent `ElasticBuffer.dispatch` calls 
    to skip layout recomputation.
    """
```

#### (c) Engram支持

**文件**：`csrc/kernels/elastic/engram.hpp`

Engram允许GPU直接访问远端内存，无需显式通信。

### 2.7 代码量对比

| 组件 | V1 | V2 |
|------|-----|-----|
| 核心通信 | ~4000行 (legacy/) | ~2000行 (elastic/) |
| Buffer管理 | 分散在kernel中 | 1382行 (buffer.hpp) |
| JIT编译 | 无 | ~1000行 (jit/) |
| 后端 | NVSHMEM | NCCL GIN |

## 3. Git历史时间线

```
2025-02-24  ebfe47e  Initial commit (V1)
2025-09-25  c9f647d  Add HybridEP
2025-10-09  85fba86  FP4 intranode ready
2025-11-21  9f2fc4b  Single Batch Overlap (SBO)
2026-04-26  b306af0  EPv2: faster EP, and Engram/PP/CP supports
2026-05-21  5616959  Fix V2 initialization
```

## 4. 结论

**论断验证**：✅ 正确

V1→V2的演进：
1. **保留**：GPU发起、symmetric memory、push+signal模型
2. **下沉**：PTX级WQE构造、doorbell管理
3. **新增**：JIT编译、EPHandle缓存、Engram/PP/CP
4. **统一**：Normal/Low-Latency → 参数化统一

这不是"推倒重来"，而是"标准化封装"。

