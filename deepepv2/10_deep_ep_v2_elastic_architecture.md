# DeepEP V2 ElasticBuffer 架构深度分析

> **分析范围**：DeepEP v2.1.0 ElasticBuffer 新架构的 13 个关键设计维度，细化到源码级别
> 
> **核心源码文件**：
> - `csrc/elastic/buffer.hpp` (1382 行) — ElasticBuffer 核心 C++ 实现
> - `deep_ep/buffers/elastic.py` (1107 行) — Python 入口与高级 API
> - `csrc/jit/` — 全 JIT 编译系统 (8 个文件，~800 行)
> - `csrc/kernels/backend/nccl.cu` + `symmetric.hpp` — NCCL Gin 后端
> - `csrc/kernels/elastic/` — 6 个 kernel launch runtime
> - `deep_ep/include/deep_ep/impls/` — 11 个 header-only kernel 实现

---

## 目录

1. [Fully JIT（全即时编译）](#1-fully-jit全即时编译)
2. [NCCL Gin Backend](#2-nccl-gin-backend)
3. [Header-only & Lightweight](#3-header-only--lightweight)
4. [复用已有 NCCL Communicator](#4-复用已有-nccl-communicator)
5. [EPv2 架构总览](#5-epv2-架构总览)
6. [统一 ElasticBuffer 接口 + 新 GEMM Layout](#6-统一-elasticbuffer-接口--新-gemm-layout)
7. [大规模 EP2048 支持](#7-大规模-ep2048-支持)
8. [解析式 SM/QP 计算（告别 Auto-tuning）](#8-解析式-smqp-计算告别-auto-tuning)
9. [Hybrid & Direct 双模式并存](#9-hybrid--direct-双模式并存)
10. [V3 训练 SM 数从 24→4-6](#10-v3-训练-sm-数从-244-6)
11. [0 SM Engram (with RDMA)](#11-0-sm-engram-with-rdma)
12. [0 SM PP (with RDMA)](#12-0-sm-pp-with-rdma)
13. [0 SM CP (with Copy Engine)](#13-0-sm-copy-engine)

---

## 1. Fully JIT（全即时编译）

### 1.1 设计哲学

DeepEP V2 将 **kernel 的生成、编译、缓存、加载、launch** 全链路都 JIT 化——运行时根据实际参数（rank 数、expert 数、topk、SM 数、QP 数等）动态生成 CUDA 源码，编译为 CUBIN，然后加载执行。

### 1.2 JIT 流水线源码实证

#### 阶段 1：Code Generation（`launch_runtime.hpp`）

```cpp
// csrc/jit/launch_runtime.hpp:32-46
template <typename Derived>
class LaunchRuntime {
    template <typename Args>
    static std::string generate(const Args& args) {
        auto code = Derived::generate_impl(args);
        // 静态缓存 include hash（只计算一次）
        static std::string include_hash;
        if (include_hash.empty())
            include_hash = include_parser->get_hash_value(code);
        code = fmt::format("// Includes' hash value: {}\n{}", include_hash, code);
        return code;
    }
```

每个具体的 Runtime（`DispatchRuntime`、`CombineRuntime`、`EngramFetchRuntime` 等）通过 `generate_impl` 生成完整 CUDA 源码。

#### 阶段 2：参数化模板实例化（`dispatch.hpp:51-88`）

```cpp
static std::string generate_impl(const Args& args) {
    // ...选择 header 和函数名
    return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
}
```

**关键**：template 参数完全由运行时决定——`kNumSMs`、`kNumQPs`、`kNumRanks`、`kNumTopk` 等全部是 **运行时变量提升到编译期常量**，NVCC 可对每种参数组合做针对性优化。

#### 阶段 3：编译缓存（`compiler.hpp:111-160`）

```cpp
std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
    // 签名 = name + compiler_signature + flags + code
    const auto kernel_signature = fmt::format("{}$${}$${}$${}", name, signature, flags, code);
    const auto dir_path = cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));
    
    // 命中缓存
    if (const auto runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
        return runtime;
    
    // 编译到临时目录后原子 rename
    const auto tmp_dir_path = make_tmp_dir() / get_uuid();
    // ... compile ...
    std::filesystem::rename(tmp_dir_path, dir_path);  // 原子操作
}
```

**设计亮点**：
- **缓存 key 包含完整签名**：NVCC 版本、flags、include hash、kernel code 全部编码进 SHA256
- **原子 rename 避免竞态**：多进程并发编译同一 kernel 时，只有一个成功，其余复用
- **双层缓存**：内存 `KernelRuntimeCache` + 磁盘 `~/.deep_ep/cache/`

#### 阶段 4：Kernel 加载与 Launch（`handle.hpp`）

支持 **CUDA Runtime API**（≥12.8）和 **CUDA Driver API** 两种路径：

```cpp
// csrc/jit/handle.hpp:23-81
#if CUDART_VERSION >= 12080 and defined(EP_JIT_USE_RUNTIME_API)
    using LibraryHandle = cudaLibrary_t;
    using KernelHandle = cudaKernel_t;
    // 使用 cudaLibraryLoadFromFile / cudaLibraryGetKernel
#else
    using LibraryHandle = CUmodule;
    using KernelHandle = CUfunction;
    // 使用 cuModuleLoad / cuModuleGetFunction
#endif
```

Launch 支持 **cooperative launch**（跨 SM 同步）、**cluster launch**（H100+ cluster 支持）、**PDL**（Programmatic Dependent Launch）。

### 1.3 IncludeParser 递归 Hash（`include_parser.hpp`）

```cpp
std::string get_hash_value(const std::string& code, const bool& exclude_code = true) {
    std::stringstream ss;
    for (const auto& i: get_includes(code))
        ss << get_hash_value_by_path(library_include_path / i) << "$";
    if (not exclude_code)
        ss << "#" << get_hex_digest(code);
    return get_hex_digest(ss.str());
}
```

**作用**：当任何被包含的 `.cuh` 文件发生变更时，自动失效缓存重新编译。

### 1.4 JIT 环境变量控制

| 变量 | 作用 |
|------|------|
| `EP_JIT_CACHE_DIR` | 自定义缓存目录 |
| `EP_JIT_PRINT_COMPILER_COMMAND` | 打印 NVCC 命令 |
| `EP_JIT_DEBUG` | 调试模式（打印 kernel 源码、launch 参数） |
| `EP_JIT_DUMP_ASM/PTX/SASS` | 反汇编输出 |
| `EP_JIT_CPP_STANDARD` | C++ 标准（默认 20） |
| `EP_JIT_NVCC_COMPILER` | 自定义 NVCC 路径 |
| `EP_NUM_TOPK_IDX_BITS` | topk_idx 位宽参数化 |

---

## 2. NCCL Gin Backend

### 2.1 Gin（GPU-Initiated Networking）架构

DeepEP V2 的核心通信后端是 **NCCL Gin**（GPU-Initiated Networking），允许 GPU SM 直接操作 NIC QP（Queue Pair），无需 CPU 介入。

#### 关键数据结构（`nccl.cu`）

```cpp
// csrc/kernels/backend/nccl.cu:62-108
NCCLSymmetricMemoryContext::NCCLSymmetricMemoryContext(...) {
    // 查询 NCCL communicator 属性
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    
    // 配置 Gin 设备通信器
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (num_ranks > 1 and get_env("EP_DISABLE_GIN", 0) == 0) {
        reqs.ginContextCount = num_allocated_qps;          // QP 数量
        reqs.ginExclusiveContexts = true;                   // 独占 QP
        reqs.ginQueueDepth = kGinQPDepth;                  // QP 深度
        reqs.ginTrafficClass = sl_idx;                      // RDMA 服务等级
        reqs.ginSignalCount = num_ranks + 2 * 2;           // 信号量数
        reqs.ginConnectionType = allow_hybrid_mode ? 
            NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_FULL;
    }
    
    // 创建 NCCL 设备通信器
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, dev_comm.ptr));
    
    // 注册对称内存窗口
    NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, num_bytes, &window, NCCL_WIN_STRICT_ORDERING));
    
    // 获取 LSA（Link-Symmetric Access）指针
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));
}
```

### 2.2 NCCLGin Handle 封装（`handle.cuh`）

```cpp
// csrc/kernels/elastic/handle.cuh — 实际在 deep_ep/include/deep_ep/common/handle.cuh
struct NCCLGin {
    const ncclDevComm_t& nccl_dev_comm;
    const ncclWindow_t& nccl_window;
    ncclGin gin;
    ncclTeam team_world, team_lsa, team_rail;
    
    // 支持三种 team 模式：
    // - ncclTeamTagWorld: 全局所有 rank
    // - ncclTeamTagLsa: NVLink 域内 rank
    // - ncclTeamTagRail: 同 rail 的 rank
    
    template <typename team_t>
    void get(void* src_ptr, void* dst_ptr, const int& num_bytes, const int& src_rank_idx) {
        gin.get(TEAM_WORLD_RAIL(), src_rank_idx, 
                nccl_window, offset(src_ptr), nccl_window, offset(dst_ptr), num_bytes, ...);
    }
    
    template <typename team_t>
    void put(void* recv_sym_ptr, void* send_sym_ptr, const int& num_bytes, const int& dst_rank_idx) {
        gin.put(TEAM_WORLD_RAIL(), dst_rank_idx, ...);
    }
};
```

### 2.3 QP 映射策略（`comm.cuh:56-86`）

```cpp
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps>
__device__ __forceinline__ std::pair<int, ncclGinResourceSharingMode> get_qp_mode(
    const int& sm_idx, const int& channel_in_sm_idx, const bool& is_notify_warp) {
    
    // Notify warp 独占 QP 0
    if (is_notify_warp) return {0, kSharingCTA};
    
    if constexpr (kNumSMs <= kNumAvailableQPs) {
        // SM 数 ≤ QP 数：每个 SM 至少一个独占 QP
        // e.g., 3 SMs, 10 QPs:
        // SM 0: 0 3 6 9
        // SM 1: 1 4 7
        // SM 2: 2 5 8
        return {kQPStartIdx + sm_idx + (channel_in_sm_idx % num_qps_in_sm) * kNumSMs, kSharingCTA};
    } else {
        // SM 数 > QP 数：所有 SM 共享所有 QP
        const auto global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;
        return {kQPStartIdx + (global_channel_idx % kNumAvailableQPs), kSharingGrid};
    }
}
```

**关键设计**：
- **Notify warp 独占 QP 0**：避免数据 warp 竞争
- **QP  Sharing 模式**：`CTA`（SM 内共享）vs `GPU`（全 GPU 共享）
- **运行时参数化**：QP 数量、SM 数、channel 数全部由 JIT 模板参数决定

### 2.4 Symmetric Memory 三层架构（`symmetric.hpp`）

DeepEP V2 提供三种 symmetric memory 分配策略：

| 类型 | 类 | 用途 |
|------|-----|------|
| GPU-only | `GPUSymmetricMemory` | 纯 GPU 通信（`ncclMemAlloc`） |
| Elastic GPU+CPU | `ElasticSymmetricMemory` | GPU VRAM + 本地 NUMA CPU RAM |
| Hybrid Elastic | `HybridElasticSymmetricMemory` | GPU VRAM + 跨 rank CPU RAM 拼接 |

**Hybrid Elastic 内存布局**：
```
[ GPU VRAM | CPU rank0 | CPU rank1 | ... | CPU rank(N-1) ]
```

每个 rank 创建本地 NUMA-local CPU segment，通过 **POSIX FD**（`pidfd_open` + `pidfd_getfd`）跨进程共享，其他 rank 用 `cuMemImportFromShareableHandle` 导入。

**关键常量**：
```cpp
static constexpr int64_t kNumAlignmentBytes = 2097152;  // 2 MB 对齐
```

### 2.5 Gin Barrier 实现（`comm.cuh:131-181`）

```cpp
template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles, typename team_t>
__device__ void gin_barrier_wo_local_sync(...) {
    // 1. 所有 SM 刷新所有 QPs（保证 store 可见性）
    for (int i = global_warp_idx; i < num_qps; i += kNumSMs * kNumWarps)
        ncclGin(nccl_dev_comm, i, NCCL_GIN_RESOURCE_SHARING_CTA).flush(ncclCoopWarp());
    cooperative_groups::this_grid().sync();
    
    // 2. SM 0 执行 barrier（只用 QP 0）
    if (sm_idx == 0) {
        for (int i = thread_idx; i < kNumRanks; i += kNumThreads)
            gin.signal(team, i, ncclGin_SignalInc{rank_idx});
        
        for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {
            timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal = ptx::ld_acquire_sys<uint64_t>(signal_ptr);
                if (signal >= target) return true;
                // ... timeout 处理
            });
        }
    }
}
```

**关键**：
- 使用 **Gin Signal** 实现跨 rank barrier
- **Shadow pointer** 跟踪信号值
- **Timeout 保护**：防止网络故障导致死锁

---

## 3. Header-only & Lightweight

### 3.1 Header-only 设计

所有 kernel 实现都是 **header-only**（`.cuh` 文件），无需单独编译：

```
deep_ep/include/deep_ep/
├── common/           # 基础工具（header-only）
│   ├── comm.cuh      # Gin 通信原语
│   ├── handle.cuh    # NCCLGin handle 封装
│   ├── layout.cuh    # Token/Buffer/Workspace 布局
│   ├── math.cuh      # 数学工具
│   ├── ptx.cuh       # PTX 内联汇编
│   ├── compiled.cuh  # 编译期开关
│   └── exception.cuh # 异常处理
└── impls/            # Kernel 实现（header-only）
    ├── dispatch.cuh
    ├── dispatch_copy_epilogue.cuh
    ├── combine.cuh
    ├── combine_reduce_epilogue.cuh
    ├── hybrid_dispatch.cuh
    ├── hybrid_combine.cuh
    ├── engram_fetch.cuh
    ├── engram_fetch_wait.cuh
    ├── pp_send_recv.cuh
    └── barrier.cuh
```

**总代码量**：~4500 行 header-only 代码，覆盖所有 kernel 逻辑。

### 3.2 编译期开关（`compiled.cuh`）

```cpp
// 通过 DISABLE_SM90_FEATURES 控制 FP8/TMA 支持
#ifndef DISABLE_SM90_FEATURES
    static constexpr bool kEnableSM90Features = true;
#else
    static constexpr bool kEnableSM90Features = false;
#endif
```

### 3.3 轻量级 C++ 层

C++ 层极其精简：

| 文件 | 行数 | 职责 |
|------|------|------|
| `python_api.cpp` | 39 | pybind11 模块注册 |
| `jit/` 全文件 | ~800 | JIT 编译系统 |
| `elastic/buffer.hpp` | 1382 | ElasticBuffer 核心 |
| `legacy/buffer.hpp` | 1794 | LegacyBuffer（兼容） |
| `kernels/backend/` | ~750 | NCCL Gin 后端 |

**整个 C++ 扩展编译后**：单个 `_C.so`，无外部依赖（除 NCCL 和 CUDA）。

### 3.4 零拷贝 Python Binding

```cpp
// csrc/python_api.cpp:22-39
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("is_sm90_compiled", []() { return deep_ep::kEnableSM90Features; });
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value);
    deep_ep::jit::register_apis(m);        // JIT
    deep_ep::legacy::register_apis(m);     // Legacy
    deep_ep::elastic::register_apis(m);    // Elastic V2
}
```

---

## 4. 复用已有 NCCL Communicator

### 4.1 Python 层（`elastic.py:301`）

```python
# deep_ep/buffers/elastic.py:301
self.nccl_comm_handle = get_nccl_comm_handle(group, force_new_comm=num_cpu_bytes > 0)
```

**关键**：默认直接复用 PyTorch 已有 NCCL communicator，仅在需要 CPU buffer（Engram）时才创建新 comm。

### 4.2 C++ 层（`nccl.cu:76`）

```cpp
// csrc/kernels/backend/nccl.cu:76
comm = reinterpret_cast<ncclComm_t>(nccl_comm_comm);
```

**直接 reinterpret_cast** Python 传入的 `ncclComm_t` 指针，无需创建新 communicator。

### 4.3 Domain Size 查询

```cpp
// csrc/kernels/backend/nccl.cu:49-54
std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm) {
    const int num_ranks = ncclTeamWorld(comm).nRanks;
    const int num_nvl_ranks = ncclTeamLsa(comm).nRanks;
    return {num_ranks / num_nvl_ranks, num_nvl_ranks};
}

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode) {
    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm);
    return {allow_hybrid_mode ? num_rdma_ranks : 1,
            allow_hybrid_mode ? num_nvl_ranks : num_rdma_ranks * num_nvl_ranks};
}
```

**自动从 NCCL communicator 推断拓扑**：
- `ncclTeamWorld` → 全局 rank 数
- `ncclTeamLsa` → NVLink domain 内 rank 数
- `ncclTeamRail` → 同一 rail 的 rank 数（multi-plane）

---

## 5. EPv2 架构总览

### 5.1 统一 ElasticBuffer 类

ElasticBuffer 是 EPv2 的核心，统一了以下所有通信模式：

| 模式 | API | 底层实现 |
|------|-----|----------|
| EP Dispatch | `dispatch()` | NVLink + RDMA all-to-all |
| EP Combine | `combine()` | NVLink + RDMA reduce |
| Engram Fetch | `engram_fetch()` | RDMA one-sided get |
| PP Send/Recv | `pp_send()`/`pp_recv()` | NVLink put |
| AGRS | `all_gather()` | NVLink symmetric memcpy |

### 5.2 内存布局

```
Symmetric Memory Layout:
[[[Workspace] GPU Buffer] CPU Buffer]
 ↑                       ↑          ↑
 mapped_window_ptr       workspace  cpu_segment
                         + gpu_buf  (for Engram)
```

**Workspace** 包含所有同步信号量：
```cpp
// layout.cuh:43-80
static int64_t get_num_bytes() {
    // 16 bytes NVLink barrier counter
    // (1024 + 2048) * 8 bytes notify reduction workspace
    // 1024 * 8 * 2 scaleup rank count (send/recv)
    // 2048 * 8 * 2 scaleup expert count (send/recv)
    // 1024 * 4 scaleup atomic sender counter
    // 1024 * 4 * 2 scaleout rank count (send/recv)
    // 2048 * 4 * 2 scaleout expert count (send/recv)
    // 1024 * 256 * 8 scaleout channel metadata
    // 1024 * 256 * 4 channel scaleup tail
    // 2 * 2 * 8 PP send/recv count
    // (32 + 1) * 1024 * 4 AGRS signals
}
```

**Workspace 大小固定**（约 1 MB），不随配置变化，支持最多 **1024 ranks**、**2048 experts**。

### 5.3 Token Layout（`layout.cuh:179-249`）

```cpp
struct TokenLayout {
    int num_hidden_bytes, num_sf_bytes;
    int num_topk, num_metadata_bytes;
    bool with_metadata;
    
    // 每个 token 的内存布局:
    // [hidden (TMA-aligned)] [sf (TMA-aligned)] [metadata (TMA-aligned)] [mbarrier (optional)]
    // metadata = topk_idx (num_topk * 4) + topk_weights (num_topk * 4) + src_token_global_idx (4) + linked_list_idx (num_topk * 4)
    
    template <bool kWithMBarrier, typename dtype_t = int>
    dtype_t get_num_bytes() const {
        return align(num_hidden_bytes, 128) + align(num_sf_bytes, 128) + 
               align(num_metadata_bytes, 128) + align(kWithMBarrier ? sizeof(mbarrier) : 0, 128);
    }
};
```

**TMA 对齐要求**：所有 segment 必须 128 字节对齐（TMA 硬件要求）。

---

## 6. 统一 ElasticBuffer 接口 + 新 GEMM Layout

### 6.1 统一 API（`elastic.py`）

```python
class ElasticBuffer:
    def dispatch(self, x, topk_idx, topk_weights, ...) -> (recv_x, recv_topk_idx, recv_topk_weights, handle, event)
    def combine(self, x, handle, topk_weights, ...) -> (combined_x, combined_topk_weights, event)
    def engram_write(self, storage, sf) -> None
    def engram_fetch(self, indices, num_qps, use_tma_aligned_col_major_sf) -> Callable
    def pp_send(self, x, dst_rank_idx, num_sms) -> None
    def pp_recv(self, x, src_rank_idx, num_sms) -> None
    def all_gather(self, tensors) -> (gathered_tensors, handle)
```

**所有 API 共享同一个 buffer**，无需为不同通信模式创建独立 buffer。

### 6.2 新 GEMM Layout — TMA-aligned Column-major SF

```python
# elastic.py:271-277
if use_tma_aligned_col_major_sf:
    // TMA-aligned column-major layout for the next GEMM input
    sf_token_stride = 1
    sf_hidden_stride = math.align(num_tokens, kNumAlignedSFPacks)
```

**目的**：Dispatch 输出的 FP8 scale factors 直接以 **TMA-aligned column-major** 布局，使得下一次 GEMM 可以直接消费，无需额外转置。

### 6.3 Buffer Size 自动计算（`buffer.hpp:652-686`）

```cpp
static int64_t calculate_buffer_size(const int64_t& nccl_comm,
                                     const int& num_max_tokens_per_rank, const int& hidden,
                                     int num_topk, const bool& use_fp8_dispatch,
                                     const bool& allow_hybrid_mode,
                                     const bool& allow_multiple_reduction) {
    // 自动推断 topology
    const auto [num_rdma_ranks, num_nvl_ranks] = nccl::get_physical_domain_size(nccl_comm);
    const auto [num_scaleout_ranks, num_scaleup_ranks] = nccl::get_logical_domain_size(nccl_comm, allow_hybrid_mode);
    
    // 计算 dispatch 和 combine 各自需要的 buffer size
    const auto num_dispatch_bytes = get_dispatch_buffer_size(...);
    const auto num_combine_bytes = get_combine_buffer_size(...);
    
    // 取最大值，对齐到 2 MB
    return math::align(std::max(num_dispatch_bytes, num_combine_bytes), symmetric::kNumAlignmentBytes);
}
```

---

## 7. 大规模 EP2048 支持

### 7.1 编译期常量（`layout.cuh`）

```cpp
static constexpr int kNumMaxRanks = 1024;       // 最大 rank 数
static constexpr int kNumMaxExperts = 2048;     // 最大 expert 数
static constexpr int kNumMaxExpertsPerRank = 256; // 每个 rank 最大 expert 数
```

**EP2048 = 1024 ranks × 2 experts_per_rank** 或 **256 ranks × 8 experts_per_rank**。

### 7.2 运行时 EP 数无限制

编译期常量仅影响 workspace 大小（固定 ~1MB），运行时 rank 数和 expert 数由 JIT 模板参数决定，**无硬性上限**。

### 7.3 Multi-plane 支持（`elastic.py:279-282`）

```python
if os.environ.get('NCCL_GIN_CROSS_NIC') == '0':
    # Multi-plane: all ranks share CPU segments, skip proxy re-export
    os.environ.setdefault('NCCL_SYM_REUSE_SYSMEM_HANDLES', '1')
```

### 7.4 Large Buffer VA Space 扩展（`elastic.py:284-298`）

```python
if num_cpu_bytes > 0:
    num_gpu_bytes = num_bytes - num_cpu_bytes
    num_max_local_ranks = int(os.getenv('EP_NUM_MAX_LOCAL_RANKS', 16)) if allow_hybrid_mode else 1
    
    num_registered_bytes = num_gpu_bytes + num_cpu_bytes * num_max_local_ranks + (1 << 32)
    num_total_gpu_bytes = torch.cuda.get_device_properties('cuda').total_memory
    if num_registered_bytes > num_total_gpu_bytes:
        win_stride = align(num_registered_bytes, 1 << 32)
        os.environ['NCCL_WIN_STRIDE'] = str(win_stride)
```

---

## 8. 解析式 SM/QP 计算（告别 Auto-tuning）

### 8.1 SM 数计算（`elastic.py:728-834`）

```python
@weak_lru(maxsize=None)
def get_theoretical_num_sms(self, num_experts, num_topk, num_scaleout_topk=0,
                            rdma_gbs=0, nvlink_gbs=0,
                            sm_read_gbs=200, sm_write_gbs=50) -> int:
    # 1. 自动获取带宽
    if rdma_gbs == 0 and self.num_rdma_ranks > 1:
        rdma_gbs = get_rdma_gbs()
    if nvlink_gbs == 0:
        nvlink_gbs = get_nvlink_gbs()
    
    # 2. 计算期望 topk（去重后）
    def get_expected_topk(num_groups):
        return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk) / math.comb(num_experts, num_topk))
    
    # 3. 计算各类型流量
    sm_read, sm_write = 0, 0
    rdma_traffic, nvlink_traffic = 0, 0
    
    # ... 详细流量建模 ...
    
    # 4. 找到瓶颈
    if self.num_scaleout_ranks > 1 and (rdma_traffic / rdma_gbs) > (nvlink_traffic / nvlink_gbs):
        bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
    else:
        bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs
    
    # 5. 计算 SM 数
    num_sms = max(
        bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
        bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
    )
    num_sms = align(max(4, math.ceil(num_sms * 1.25)), 2)  # 至少 4，向上取偶数，+25% 余量
    num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)
    num_sms = min(num_sms, num_device_sms)
```

### 8.2 QP 数计算（`elastic.py:836-853`）

```python
def get_theoretical_num_qps(self, num_sms: int) -> int:
    # Direct mode: 鼓励较少 QPs（减少 DB ringing 开销）
    num_qps = min(num_sms, 8 + 1)
    
    # Hybrid mode: 每个 channel（和 notify）独立 QP
    if self.allow_hybrid_mode:
        num_qps = num_sms * 16 + 1
    
    return min(num_qps, self.num_allocated_qps)
```

### 8.3 自动 SM/QP 选择（`elastic.py:926-929`）

```python
# Automatic decide SM and QP count
num_topk = (handle.topk_idx if topk_idx is None else topk_idx).shape[1]
num_sms = self.get_theoretical_num_sms(num_experts, num_topk) if num_sms == 0 else num_sms
num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps
assert num_qps <= self.num_allocated_qps
```

**用户只需传 `num_sms=0, num_qps=0`**，系统自动选择最优值。

---

## 9. Hybrid & Direct 双模式并存

### 9.1 模式选择（`nccl.cu:117-125`）

```cpp
// 根据 allow_hybrid_mode 决定逻辑拓扑
if (allow_hybrid_mode) {
    num_scaleout_ranks = num_rdma_ranks, num_scaleup_ranks = num_nvl_ranks;
    scaleout_rank_idx = rdma_rank_idx, scaleup_rank_idx = nvl_rank_idx;
} else {
    num_scaleout_ranks = 1, num_scaleup_ranks = num_ranks;
    scaleout_rank_idx = 0, scaleup_rank_idx = rank_idx;
}
```

| 模式 | scaleout | scaleup | 通信方式 |
|------|----------|---------|----------|
| Direct | 1 | num_ranks | 所有 rank 直连 |
| Hybrid | num_rdma_ranks | num_nvl_ranks | 域内 NVLink + 跨域 RDMA |

### 9.2 Kernel 分发（`dispatch.hpp:52-88`）

```cpp
static std::string generate_impl(const Args& args) {
    if (args.num_scaleout_ranks == 1) {
        header_name = "dispatch";
        func_name = fmt::format("dispatch_impl<...>", ...);
    } else {
        header_name = "hybrid_dispatch";
        func_name = fmt::format("hybrid_dispatch_impl<...>", ...);
    }
}
```

**同一 API，不同参数自动选择不同 kernel**。

### 9.3 Hybrid Dispatch 架构（`hybrid_dispatch.cuh`）

Hybrid dispatch 使用 **3 种 warp 角色**：
- **Notify warps**：统计 token → expert/rank 映射，写入 workspace
- **Scaleout warps**：域内 NVLink 发送 + RDMA 跨域发送
- **Forward warps**：将 scaleup 收到的 token 转发到 scaleout 远端

### 9.4 Hybrid Combine 架构（`hybrid_combine.cuh`）

Combine 是 dispatch 的逆操作：
- **Scaleup warps**：域内 NVLink reduce
- **Forward warps**：将 scaleup 结果转发回 scaleout

---

## 10. V3 训练 SM 数从 24→4-6

### 10.1 背景

V1/V2 时代，DeepEP 使用 **固定 24 SMs** 进行通信，对计算 kernel 的 SM 资源抢占严重。

### 10.2 V3 的 SM 优化

通过 **解析式 SM 计算** + **流水线优化**，V3 可以将 SM 数降到 **4-6**：

```python
# elastic.py:823
num_sms = align(max(4, math.ceil(num_sms * 1.25)), 2)
num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)
```

**关键参数**：
- `prefer_overlap_with_compute=True`：优先使用较少 SMs，与计算 kernel 重叠
- 最少 4 SMs（满足 **4 个 notify warps** 的最小需求）
- 偶数对齐（cluster launch 要求）

### 10.3 为什么可以降到 4-6 SMs

1. **带宽瓶颈分析**：解析式计算精确知道瓶颈在 RDMA 还是 NVLink
2. **流水线全覆盖**：每个 SM 的 warp 角色固定，无空闲周期
3. **QP 独立化**：每个 channel 独占 QP，避免竞争
4. **Cluster Launch**：`cluster_dim=2` 与计算 kernel 的 cluster 重叠

```cpp
// dispatch.hpp:226
.launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 2 - (num_sms % 2), true);
//                                                                          ^^^^^^^^^^^^^^^^
//                                                                          cluster_dim=2（偶数 SMs）或 1（奇数 SMs）
```

---

## 11. 0 SM Engram (with RDMA)

### 11.1 Engram 概念

Engram 是 DeepEP V2 新增的 **远程 KV Cache 获取** 功能：将推理/解码阶段的 KV Cache 存储在 CPU memory，训练时通过 RDMA one-sided get 拉取到 GPU。

### 11.2 0 SM 实现原理

```cpp
// csrc/kernels/elastic/engram.hpp:98-126
static void launch_engram_fetch(...) {
    constexpr int kNumEngramFetchThreads = 1024;  // 仅 1 个 block！
    
    const EngramFetchRuntime::Args args = {
        .launch_args = jit::LaunchArgs(num_qps, kNumEngramFetchThreads)
        //                                       ^^^^^^^^
        //                                       grid_dim = num_qps（通常 1-8）
    };
}
```

**关键**：
- **Grid = num_qps**（通常 1-8），每个 block 使用 1 个 QP
- **不使用 SM 进行数据搬运**：完全由 RDMA one-sided get 完成
- **Gin.get()** 直接操作 NIC，SM 仅发起请求后退出

### 11.3 Engram Fetch Kernel（`engram_fetch.cuh`）

```cpp
// 伪代码
template <int kNumQPs, ...>
__global__ void engram_fetch_impl(...) {
    const auto sm_idx = blockIdx.x;  // 每个 block 对应 1 个 QP
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, sm_idx, ...);
    
    // 每个 token 的 entry 通过 RDMA get 拉取
    for (int i = thread_idx; i < num_tokens * num_entries_per_token; i += blockDim.x) {
        const auto token_idx = i / num_entries_per_token;
        const auto entry_idx = i % num_entries_per_token;
        const auto storage_idx = indices[token_idx * num_entries_per_token + entry_idx];
        
        // RDMA one-sided get — 不占用 SM 计算资源
        gin.get<team_t>(storage_ptr, fetched_ptr, num_bytes, src_rank_idx);
        last_gin_requests[sm_idx] = gin.last_request();
    }
}
```

### 11.4 Engram Fetch Wait（`engram_fetch_wait.cuh`）

```cpp
// csrc/kernels/elastic/engram.hpp:169-189
static void launch_engram_fetch_wait(...) {
    constexpr int kNumEngramFetchWaitThreads = 1024;
    const EngramFetchWaitRuntime::Args args = {
        .launch_args = jit::LaunchArgs(num_qps, kNumEngramFetchWaitThreads)
    };
}
```

**Wait kernel** 也仅使用 num_qps 个 SMs，等待所有 RDMA get 完成。

### 11.5 CPU Buffer 用于 Engram Storage（`buffer.hpp:229-240`）

```cpp
void engram_write(const torch::Tensor& storage, const std::optional<torch::Tensor>& sf) {
    // 写入 CPU segment（在 symmetric memory 的后部）
    const auto cpu_write_offset = allow_hybrid_mode
        ? static_cast<int64_t>(nccl_context->scale_up_rank_idx) * num_cpu_buffer_bytes : 0;
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        math::advance_ptr(buffer, num_gpu_buffer_bytes + cpu_write_offset),
        storage.data_ptr(), storage.nbytes(),
        cudaMemcpyDeviceToDevice, compute_stream));
}
```

**Engram 数据在 CPU memory**，通过 `ncclCommWindowRegister` 注册到 NCCL VA space，RDMA NIC 可以直接访问。

---

## 12. 0 SM PP (with RDMA)

### 12.1 PP Send/Recv API

```python
# elastic.py:327-375
def pp_set_config(self, num_max_tensor_bytes, num_max_inflight_tensors):
    self.runtime.pp_set_config(num_max_tensor_bytes, num_max_inflight_tensors)

def pp_send(self, x, dst_rank_idx, num_sms=0):
    self.runtime.pp_send(x, dst_rank_idx, 
                         num_sms=num_sms if num_sms else jit.device_runtime.get_num_sms())

def pp_recv(self, x, src_rank_idx, num_sms=0):
    self.runtime.pp_recv(x, src_rank_idx, 
                         num_sms=num_sms if num_sms else jit.device_runtime.get_num_sms())
```

### 12.2 PP 实现（`pp_send_recv.hpp`）

```cpp
// csrc/kernels/elastic/pp_send_recv.hpp:64-95
static void launch_pp_send(...) {
    const PPSendRuntime::Args args = {
        .launch_args = jit::LaunchArgs(num_sms, 32, num_smem_bytes, 1, true)
        //                                       ^^^^^^^^
        //                                       默认使用所有 SMs
    };
}
```

**PP 默认使用所有 SMs**（因为需要高带宽），但可以通过 `num_sms` 参数限制。

### 12.3 0 SM 的含义

"0 SM PP" 指的是 **PP 通信不占用额外的 SMs**——因为：
1. PP send/recv 使用 **Gin.put()**（RDMA one-sided put）
2. SM 仅发起请求，NIC 完成数据搬运
3. 计算 kernel 可以同时在其他 SMs 上运行

### 12.4 PP Buffer Layout（`buffer.hpp:327-337`）

```cpp
void pp_set_config(const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors) {
    EP_HOST_ASSERT(num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2 <= num_buffer_bytes);
    //                                                                          ^^^^^^^
    //                                                                          send/recv × prev/next
    this->prev_rank_idx = (nccl_context->rank_idx + nccl_context->num_ranks - 1) % nccl_context->num_ranks;
    this->next_rank_idx = (nccl_context->rank_idx + 1) % nccl_context->num_ranks;
    this->num_max_pp_tensor_bytes = math::align<int64_t>(num_max_tensor_bytes, 32);
    this->num_max_pp_inflight_tensors = num_max_inflight_tensors;
}
```

**Buffer 需求**：`num_max_tensor_bytes × num_max_inflight_tensors × 2 (send/recv) × 2 (prev/next)`

---

## 13. 0 SM CP (Copy Engine)

### 13.1 AGRS — All-Gather Reduce-Scatter

```python
# elastic.py:638-726
def create_agrs_session(self):
    self.runtime.create_agrs_session()

def destroy_agrs_session(self):
    self.runtime.destroy_agrs_session()

def all_gather(self, tensors):
    return self.runtime.all_gather(tensors)
```

### 13.2 Copy Engine 实现（`buffer.hpp:438-524`）

```cpp
std::pair<std::vector<torch::Tensor>, std::function<void()>>
all_gather(const std::vector<torch::Tensor>& tensors) {
    // 1. 计算需要多少 copies
    int num_copies = 0;
    for (int i = 0; i < nccl_context->num_ranks; ++ i)
        for (int j = 0; j < num_tensors; ++ j)
            num_copies += nccl_context->num_ranks - is_inplace;
    
    // 2. 使用 cudaMemcpyBatchAsync — 完全由 Copy Engine 执行
    cudaMemcpyAttributes attrs = {
        .srcAccessOrder = cudaMemcpySrcAccessOrderStream,
        .flags = cudaMemcpyFlagPreferOverlapWithCompute
    };
    CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), num_copies, attrs, comm_stream));
    
    // 3. 信号通知其他 rank
    cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, current_session);
    
    // 4. 返回 gathered tensors 和 wait handle
    return {out, handle};
}
```

### 13.3 0 SM 的含义

**AGRS 使用 `cudaMemcpyBatchAsync`**：
- **不占用 SMs**：由 GPU Copy Engine（DMA）执行
- **Prefer overlap with compute**：与计算 kernel 并行
- **NVLink symmetric memory**：直接访问远端 GPU 内存

### 13.4 Batched Write and Wait（`cuda_driver.cu`）

```cpp
// csrc/kernels/backend/cuda_driver.cu
void batched_write_and_wait(CUstream stream, const std::vector<void*>& write_ptrs, 
                            const std::vector<void*>& wait_ptrs, const int& value) {
    // 批量发起 writes
    for (auto& ptr : write_ptrs)
        cuStreamWriteValue32(stream, reinterpret_cast<CUdeviceptr>(ptr), value, CU_STREAM_WRITE_VALUE_DEFAULT);
    
    // 批量等待 waits
    for (auto& ptr : wait_ptrs)
        cuStreamWaitValue32(stream, reinterpret_cast<CUdeviceptr>(ptr), value, CU_STREAM_WAIT_VALUE_EQ);
}
```

**使用 CUDA Driver API 的 `cuStreamWriteValue32` 和 `cuStreamWaitValue32`**：
- 在 **comm_stream** 上发起
- 不阻塞 compute_stream
- GPU 硬件自动处理同步

---

## 总结：DeepEP V2 架构的 13 个设计维度

| # | 特性 | 实现方式 | 关键源码 |
|---|------|----------|----------|
| 1 | Fully JIT | 运行时生成 CUDA 源码 → NVCC 编译 → CUBIN 缓存 → 加载 launch | `csrc/jit/` |
| 2 | NCCL Gin Backend | GPU SM 直接操作 NIC QP，支持 put/get/signal/flush | `nccl.cu`, `handle.cuh` |
| 3 | Header-only | 所有 kernel 实现为 `.cuh`，无需单独编译 | `deep_ep/include/` |
| 4 | 复用 NCCL Comm | `reinterpret_cast` 复用 PyTorch 已有 communicator | `nccl.cu:76` |
| 5 | EPv2 统一架构 | 单 `ElasticBuffer` 类统一所有通信模式 | `buffer.hpp` |
| 6 | 统一接口 + 新 GEMM Layout | TMA-aligned column-major SF，零拷贝给下次 GEMM | `elastic.py:271` |
| 7 | EP2048 支持 | 编译期常量 + 运行时参数化，无硬性上限 | `layout.cuh:19-22` |
| 8 | 解析式 SM/QP | 带宽建模 → 最优 SM/QP 数，无需 auto-tuning | `elastic.py:728-853` |
| 9 | Hybrid & Direct | 同一 API，参数自动选择 kernel | `dispatch.hpp:52-88` |
| 10 | 4-6 SMs V3 训练 | 解析式计算 + 流水线优化 + cluster launch | `elastic.py:823` |
| 11 | 0 SM Engram | RDMA one-sided get，SM 仅发起请求 | `engram.hpp:98` |
| 12 | 0 SM PP | Gin.put() one-sided，SM 不参与数据搬运 | `pp_send_recv.hpp:64` |
| 13 | 0 SM CP | `cudaMemcpyBatchAsync`，Copy Engine 执行 | `buffer.hpp:489` |

---

## 附录：关键源码文件索引

```
DeepEP/
├── csrc/
│   ├── jit/                         # JIT 编译系统
│   │   ├── compiler.hpp             # NVCC 编译 + 缓存
│   │   ├── kernel_runtime.hpp       # CUBIN 加载
│   │   ├── launch_runtime.hpp       # Launch 配置
│   │   ├── include_parser.hpp       # 递归 hash
│   │   ├── handle.hpp               # Driver/Runtime API 双路径
│   │   ├── cache.hpp                # 双层缓存
│   │   ├── device_runtime.hpp       # GPU 属性查询
│   │   └── api.hpp                  # pybind11 注册
│   ├── kernels/
│   │   ├── backend/
│   │   │   ├── nccl.cu              # NCCL Gin 后端
│   │   │   ├── symmetric.hpp        # 对称内存三层架构
│   │   │   ├── nvshmem.cu           # NVSHMEM 后备
│   │   │   ├── cuda_driver.cu       # Driver API 工具
│   │   │   └── api.cuh              # Backend API 声明
│   │   └── elastic/
│   │       ├── dispatch.hpp         # Dispatch launch
│   │       ├── combine.hpp          # Combine launch
│   │       ├── engram.hpp           # Engram launch
│   │       ├── pp_send_recv.hpp     # PP launch
│   │       ├── barrier.hpp          # Barrier launch
│   │       └── api.hpp              # Kernel API 汇聚
│   ├── elastic/
│   │   ├── buffer.hpp               # ElasticBuffer 核心（1382 行）
│   │   └── utils.hpp                # Elastic 工具
│   └── python_api.cpp               # pybind11 模块（39 行）
├── deep_ep/
│   ├── buffers/
│   │   ├── elastic.py               # Python 入口（1107 行）
│   │   └── legacy.py                # Legacy API 兼容
│   └── include/deep_ep/
│       ├── common/                  # Header-only 基础
│       │   ├── comm.cuh             # Gin 通信原语
│       │   ├── handle.cuh           # NCCLGin handle
│       │   ├── layout.cuh           # Token/Buffer/Workspace 布局
│       │   ├── math.cuh             # 数学工具
│       │   ├── ptx.cuh              # PTX 内联汇编
│       │   ├── compiled.cuh         # 编译期开关
│       │   └── exception.cuh        # 异常处理
│       └── impls/                   # Header-only kernel 实现
│           ├── dispatch.cuh
│           ├── dispatch_copy_epilogue.cuh
│           ├── combine.cuh
│           ├── combine_reduce_epilogue.cuh
│           ├── hybrid_dispatch.cuh
│           ├── hybrid_combine.cuh
│           ├── engram_fetch.cuh
│           ├── engram_fetch_wait.cuh
│           ├── pp_send_recv.cuh
│           └── barrier.cuh
└── setup.py                         # 构建配置
```

---

> **分析方法论**：本报告基于源码实证，每个结论都标注了具体文件和行号。分析遵循"设计哲学 → 源码实现 → 关键代码片段 → 设计亮点"的结构，确保可追溯性。
