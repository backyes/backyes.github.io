# DeepEP 辅助工具测试源码分析：Barrier / Engram / Gate

> 分析范围：`tests/elastic/test_barrier.py`、`tests/elastic/test_engram.py`、`tests/utils/test_gate.py`
> 关联实现：`csrc/kernels/elastic/barrier.hpp`、`csrc/kernels/elastic/engram.hpp`、`deep_ep/utils/gate.py`

---

## 1. 概览

| 测试文件 | 行数 | 测试对象 | 核心目标 |
|---------|------|---------|---------|
| `tests/elastic/test_barrier.py` | 62 | `ElasticBuffer.barrier` | 跨 rank GPU barrier 同步延迟 |
| `tests/elastic/test_engram.py` | 124 | `ElasticBuffer.engram_write` / `engram_fetch` | 远端内存（RDMA）访问的正确性与带宽 |
| `tests/utils/test_gate.py` | 57 | `get_unbalanced_scores` | MoE 负载不均衡分布生成的统计正确性 |

这三个测试是 DeepEP **弹性 EP（Elastic EP）** 基础设施的"基石性"测试——barrier 是同步原语，engram 是 RDMA 单边通信原语，gate 是 MoE 负载生成的参考实现。它们共同支撑了主 EP 测试 `test_ep.py` 中的 dispatch/combine 行为。

---

## 2. Barrier 测试分析

### 2.1 测试目标

`test_barrier.py` 是一个**纯延迟基准**：测量 `buffer.barrier()` 在 1000 次连续调用下的平均耗时（微秒级）。

### 2.2 关键代码与流程

```python
# L19-26: 核心测试逻辑
def loop_barrier(num_tests=1000):
    for i in range(num_tests):
        buffer.barrier()

t = bench_kineto(lambda: loop_barrier(), 'barrier',
                 barrier_comm_profiling=True, barrier=buffer.barrier)
dist_print(f' > EP: {buffer.rank_idx:3}/{buffer.num_ranks:3}, '
           f'barrier time: {t * 1e6:.3f} us')
```

**设计要点**：
- `bench_kineto` 的 `barrier_comm_profiling=True` 会在每次迭代前插入 `torch.cuda._sleep(int(2e7))`（~10ms）+ `buffer.barrier()`，消除 CPU 启动不平衡带来的测量噪声（参见 `utils/testing.py` L163-171）。
- 默认 8 进程（L54 `--num-processes 8`），对应单节点 8 GPU 场景。
- 支持 `--do-pressure-test` 无限循环压力测试（L35 `int(1e9)`）。

### 2.3 底层实现路径

```
buffer.barrier()  (elastic.py L497)
  → runtime.barrier(use_comm_stream, with_cpu_sync, sequential)
    → launch_barrier(...)  (barrier.hpp L56)
      → barrier_impl<kIsScaleupNVLink, kNumSMs, kNumThreads,
                      kNumScaleoutRanks, kNumScaleupRanks,
                      kNumTimeoutCycles, kSequential>  (barrier.cuh L17)
```

**`barrier_impl` 的关键设计**（`barrier.cuh` L24-39）：

| 模式 | 行为 | SM 数 |
|------|------|-------|
| `kSequential=True` | 先 scaleout barrier，再 scaleup barrier（串行） | 1 |
| `kSequential=False` | scaleout + scaleup 并行执行 | 2（仅当 `num_scaleout_ranks > 1`） |

```cuda
// barrier.cuh L24-39: 串行 vs 并行
if constexpr (kSequential) {
    if constexpr (kNumScaleoutRanks > 1)
        comm::gpu_barrier<..., kKernelBarrierTag, false, false, false>(...);
    // scaleup barrier 需要 flush RDMA 请求
    comm::gpu_barrier<..., kKernelBarrierTag, true, true, false>(...);
} else {
    comm::gpu_barrier<..., kKernelBarrierTag, false, false, false>(...);
}
```

**测试默认使用 `sequential=True`**（`elastic.py` L497 默认值），对应 `barrier.hpp` L70 的 `num_sms = 1`，即单 SM 串行执行——这是**测试同步**场景的最安全选择（参见 `elastic.py` L505-506 注释）。

### 2.4 与主 EP 测试的关系

`test_ep.py` 中 `bench_kineto` 调用均传入 `barrier=buffer.barrier`（L258-341），用于：
- 消除 CPU 启动偏斜（launch skew）
- 在 dispatch/combine 各阶段之间建立同步点

---

## 3. Engram 测试分析

### 3.1 Engram 是什么

Engram 是 DeepEP 弹性模式的**远端内存访问原语**——通过 NCCL Gin（GPU-initiated RDMA）直接读取远端 GPU 的存储窗口（`ncclWindow`），无需 CPU 介入。

| API | 语义 |
|-----|------|
| `engram_write(storage, sf)` | 将本地数据注册到 NCCL window，供远端读取 |
| `engram_fetch(indices)()` | 异步发起 RDMA get，返回 callable `hook`；调用 `hook()` 阻塞等待完成 |

### 3.2 测试流程

```python
# L17-37: 配置与 buffer 创建
num_gpu_bytes, num_cpu_bytes = deep_ep.ElasticBuffer.get_engram_storage_size_hint(
    args.num_entries, args.hidden, args.num_tokens * args.num_entries_per_token, dtype)
buffer = deep_ep.ElasticBuffer(
    group, num_bytes=num_gpu_bytes + num_cpu_bytes, num_cpu_bytes=num_cpu_bytes,
    explicitly_destroy=True, num_allocated_qps=num_qps, ...)

# L40-53: 写入阶段
local_bf16 = torch.randn((args.num_entries, args.hidden), dtype=torch.bfloat16, device='cuda')
if args.use_fp8:
    local_storage, local_sf = per_token_cast_to_fp8(local_bf16)
buffer.engram_write(local_storage, sf=sf)

# L56-57: 生成随机索引
indices = torch.randint(0, num_ranks * args.num_entries,
                        (args.num_tokens, args.num_entries_per_token), device='cuda', dtype=torch.int)
```

### 3.3 正确性验证

```python
# L60-71: 基于 all_gather 的参考实现
if not args.skip_check:
    global_storage = torch.empty((num_ranks * args.num_entries, args.hidden), dtype=dtype, device='cuda')
    dist.all_gather_into_tensor(global_storage, local_storage, group)
    ref_data = global_storage[indices.view(-1)].view(args.num_tokens, -1)

    for use_tma_aligned_col_major_sf in (False, True) if args.use_fp8 else (False,):
        data, fetched_sf = buffer.engram_fetch(indices,
                                               use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf)()
        assert torch.equal(ref_data, data), 'data mismatch'
```

**关键设计**：
- 使用 `dist.all_gather_into_tensor` 作为**ground truth**，与 RDMA get 结果逐元素比较
- FP8 模式下测试两种 SF layout（row-major vs TMA-aligned column-major）
- 默认参数：`num_entries=524288`, `hidden=128`, `num_tokens=512`, `num_entries_per_token=24`（L108-111）

### 3.4 性能测量

```python
# L79-96: issue + wait 端到端测量
def fetch_and_wait():
    hook = buffer.engram_fetch(indices, use_tma_aligned_col_major_sf=True)
    hook()

issue_t, wait_t = bench_kineto(
    fetch_and_wait,
    kernel_names=('engram_fetch_impl', 'engram_fetch_wait_impl'),
    barrier_comm_profiling=True, barrier=buffer.barrier,
    trace_path=...)
mpps = args.num_tokens * args.num_entries_per_token / (issue_t + wait_t) / 1e6
```

**指标**：
- `issue_t`：`engram_fetch_impl` kernel 耗时（发起 RDMA get）
- `wait_t`：`engram_fetch_wait_impl` kernel 耗时（等待完成）
- `MPPS`：Million Packets Per Second
- `GB/s`：有效带宽

### 3.5 底层 kernel 实现

**`engram_fetch_impl`**（`engram_fetch.cuh` L22-80）：

```cuda
// L29-35: 每个 block 对应一个 QP，warp 级并行
const auto qp_idx = static_cast<int>(blockIdx.x);
const auto warp_idx = ptx::get_warp_idx();
const auto global_warp_idx = qp_idx * kNumWarps + warp_idx;
const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);

// L55-74: 每个 warp 协作 fetch 一个 token 的多个 entry
for (int i = global_warp_idx; i < num_tokens * kNumEntriesPerToken; i += kNumQPs * kNumWarps) {
    const auto global_idx = __ldg(indices + i);
    const auto owner_rank_idx = global_idx / kNumEntriesPerRank;
    const auto local_entry_idx = global_idx % kNumEntriesPerRank;
    const auto peer_idx = owner_rank_idx / kNumRanksPerRDMAPeer;
    gin.get<team_t, ncclCoopThread, ncclGin_SegmentMixed>(...);
}
```

**核心特征**：
- 1 block = 1 QP（`blockIdx.x` 映射到 `qp_idx`）
- Warp 级分工：每个 warp 负责不同 token/entry 的 RDMA get
- 支持 request 聚合（`ncclGinOptFlagsAggregateRequests`）与周期性 flush（`kGinQPFlushDepth`）
- FP8 SF packs 由 `elect_one_sync()` 选出的 leader warp 串行收集（L78-80 TODO 注释表明未来可优化为 warp 并行）

### 3.6 与主 EP 测试的关系

Engram 是弹性模式下**跨节点参数/激活拉取**的实验性原语。`test_ep.py` 未直接使用 engram，但 engram 的底层机制（NCCL Gin RDMA get）与 dispatch/combine 的 NVLink/RDMA 通信共享同一套 `ncclDevComm` + `ncclWindow` 基础设施。

---

## 4. Gate 测试分析

### 4.1 Gate 工具是什么

`deep_ep/utils/gate.py` 提供 **MoE top-k 路由的负载分布生成器**，用于：
- 生成具有指定不均衡比（`ratio`）的 expert 选择 scores
- 为 `test_ep.py` 提供可控的、可复现的 routing 分布

### 4.2 测试目标

验证 `get_unbalanced_scores` 生成的 top-k 分布满足：
1. **实际不均衡比** `practical_ratio` 与期望 `ratio` 的相对误差 < 10%
2. **rank 间不均匀性** `inequality` < 1.02（非特殊 rank 间几乎均匀）

### 4.3 关键代码

```python
# L7-53: 穷举参数空间测试
def test_unbalanced_scores():
    for num_tokens in [1, 4096]:
        for num_experts_per_rank in [1, 4, 8, 16]:
            for num_ranks in [2, 4, 8, 16, 64, 72]:
                num_experts = num_experts_per_rank * num_ranks
                for num_topk in [1, 2, 4, 6, 8, 9]:
                    for ratio in [1.0, 2.0, 4.0]:
                        for precise in [1, 0]:
                            # L18-25: 边界条件检查
                            lower_bound_per_token = max(1, ceil_div(num_topk, num_experts_per_rank))
                            upper_bound_per_token = min(min(num_topk, num_ranks), int((num_ranks - 1) / ratio) + 1)
                            if lower_bound_per_token > upper_bound_per_token:
                                continue

                            # L27-36: 逐 rank 生成并统计
                            for rank_idx in range(num_ranks):
                                scores = get_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio, precise)
                                _topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, ...)
                                topk_idx = topk_idx // num_experts_per_rank
                                ...  # 统计每个 rank 被选中次数

                            # L39-52: 验证 ratio 与 inequality
                            practical_ratio = total_rank_count[0].item() / max(total_rank_count[1:].min().item(), 1)
                            inequality = total_rank_count[1:].max().item() / max(total_rank_count[1:].min().item(), 1)
                            if precise:
                                assert abs(practical_ratio - ratio) / ratio < 0.1 and inequality < 1.02
```

### 4.4 底层算法

`gate.py` 提供两种模式（L176-180）：

| 模式 | 函数 | 特点 |
|------|------|------|
| `precise=True` | `get_precise_unbalanced_scores` | 精确控制 rank count 分布，保证 ratio 精度 |
| `precise=False` | `get_random_unbalanced_scores` | 通过 factor 二分搜索逼近 ratio，更随机 |

**`generate_rank_count`**（L32-113）的核心逻辑：
1. 随机生成 top-k indices，统计每个 token 覆盖的 rank 数 `a[i]`
2. 根据 `ratio` 计算 special rank 的 token 数 `special_token_count`
3. 通过 permutation 将 special rank 插入到选定 token 的 top-k 中
4. `scatter_add_` 统计最终 rank count

**`generate_topk_idx`**（L4-29）将 rank count 映射为具体 expert indices：
- 在每个 rank 内随机排列 expert 顺序
- 按 rank count 截断得到 topk_idx

### 4.5 与主 EP 测试的关系

`test_ep.py` L13 导入、L75 调用：

```python
scores = get_unbalanced_scores(num_tokens, num_experts, buffer.num_ranks, num_topk,
                               args.unbalanced_ratio, args.precise_unbalanced_ratio)
```

这使得主 EP 测试可以模拟**真实 MoE 场景中的负载不均衡**（如某些 expert 被过度选中），验证 dispatch/combine 在极端分布下的正确性与性能。

---

## 5. API 使用模式总结

### 5.1 分布式测试模式

三个测试均采用 `torch.multiprocessing.spawn` 启动多进程：

```python
# 标准入口模式
@torch.inference_mode()
def test_loop(local_rank, num_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_ranks)
    buffer = deep_ep.ElasticBuffer(group, ...)
    # ... 测试逻辑
    buffer.destroy()
    dist.destroy_process_group()

if __name__ == '__main__':
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
```

### 5.2 性能测量模式

统一使用 `bench_kineto` + `buffer.barrier` 消除 CPU 启动偏斜：

```python
t = bench_kineto(fn, kernel_names=...,
                 barrier_comm_profiling=True, barrier=buffer.barrier,
                 trace_path=...)
```

### 5.3 正确性验证模式

- **Barrier**：无显式正确性断言（barrier 的正确性由 NCCL 保证），纯性能测量
- **Engram**：`torch.equal` 与 `all_gather` 参考实现逐元素比较
- **Gate**：统计检验 `practical_ratio` 与 `inequality` 的数值边界

---

## 6. 三者协作关系图

```
                    ┌─────────────────────┐
                    │   test_ep.py        │
                    │   (主 EP 测试)       │
                    └─────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
   │ barrier     │    │ engram       │    │ gate         │
   │ (同步原语)   │    │ (RDMA 原语)   │    │ (负载生成)    │
   └──────┬──────┘    └──────┬───────┘    └──────┬───────┘
          │                  │                   │
          ▼                  ▼                   ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
   │ barrier_impl│    │engram_fetch_ │    │generate_rank_│
   │ (NCCL Gin   │    │impl / wait   │    │count /       │
   │  GPU barrier)│    │(RDMA get)    │    │topk_idx      │
   └─────────────┘    └──────────────┘    └──────────────┘
```

| 原语 | 角色 | 被谁依赖 |
|------|------|---------|
| barrier | 跨 rank 同步 + 性能测量校准 | test_ep.py, test_engram.py |
| engram | 远端内存访问（实验性） | 独立测试，基础设施共享 |
| gate | 可控负载分布生成 | test_ep.py |

---

## 7. 关键发现与洞察

### 7.1 Barrier 的"测试模式"设计

`barrier(sequential=True)` 是**测试专用路径**：单 SM 串行执行 scaleout + scaleup barrier，牺牲性能换取同步确定性。这揭示了 DeepEP 的一个设计哲学——**生产路径追求并行度，测试路径追求可预测性**。

### 7.2 Engram 的"异步 hook"模式

`engram_fetch` 返回 callable 而非直接阻塞，这是 **lazy evaluation** 模式：
```python
hook = buffer.engram_fetch(indices)  # 异步发起
# ... 可插入计算 ...
hook()  # 阻塞等待，返回 (data, sf)
```
这与 dispatch/combine 的 `event` 参数（`with_previous_event`）设计一脉相承，允许用户重叠计算与通信。

### 7.3 Gate 的"精确 vs 随机"双模式

`precise=True` 通过 `generate_rank_count` 直接构造 rank count 矩阵，保证统计精度；`precise=False` 通过 factor 二分搜索逼近，更贴近真实 routing 的随机性。这种双模式设计使得测试既能**精确验证边界条件**，又能**模拟真实场景的统计波动**。

### 7.4 测试覆盖的参数空间

`test_gate.py` 的穷举参数空间极广：
- `num_tokens` × `num_experts_per_rank` × `num_ranks` × `num_topk` × `ratio` × `precise`
- 总计约 2 × 4 × 6 × 6 × 3 × 2 = **1728 种组合**（经边界条件过滤后仍有数百种）

这种**组合爆炸式测试**确保了 `get_unbalanced_scores` 在各种极端配置下的鲁棒性。

---

## 8. 与博客理论的对应关系

| 博客概念 | 测试体现 |
|---------|---------|
| GPU-centric fabric | `engram_fetch_impl` 中 SM 直接操作 Gin QP，无 CPU 介入 |
| 三阶段流水线 | `barrier_impl` 的 scaleout → scaleup 串行执行 |
| 对称内存 | `engram_write` 注册到 `ncclWindow`，远端可直接访问 |
| MoE 负载不均衡 | `get_unbalanced_scores` 的 `ratio` 参数直接建模 |

---

## 9. 文件路径索引

| 文件 | 行数 | 作用 |
|------|------|------|
| `tests/elastic/test_barrier.py` | 62 | Barrier 延迟测试 |
| `tests/elastic/test_engram.py` | 124 | Engram 正确性 + 带宽测试 |
| `tests/utils/test_gate.py` | 57 | Gate 统计正确性测试 |
| `deep_ep/buffers/elastic.py` | L497-604 | barrier / engram_write / engram_fetch Python API |
| `csrc/elastic/buffer.hpp` | L181-208 | C++ barrier 实现 |
| `csrc/kernels/elastic/barrier.hpp` | 87 | Barrier JIT runtime |
| `csrc/kernels/elastic/engram.hpp` | 160+ | Engram fetch/wait JIT runtime |
| `deep_ep/include/deep_ep/impls/barrier.cuh` | 42 | barrier_impl CUDA kernel |
| `deep_ep/include/deep_ep/impls/engram_fetch.cuh` | 80+ | engram_fetch_impl CUDA kernel |
| `deep_ep/utils/gate.py` | 181 | 负载分布生成算法 |
| `deep_ep/utils/testing.py` | L111-190 | bench_kineto 性能测量框架 |
