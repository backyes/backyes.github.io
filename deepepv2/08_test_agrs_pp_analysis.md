# DeepEP Elastic AGRS 与 PP 测试源码分析

> 分析对象：`tests/elastic/test_agrs.py`（202 行）与 `tests/elastic/test_pp.py`（139 行）
> 定位：DeepEP 实验性 Elastic Buffer 功能的集成测试与性能基准

---

## 1. 文件结构总览

| 文件 | 行数 | 核心函数 | 测试目标 |
|------|------|---------|---------|
| `test_agrs.py` | 202 | `all_gather_ref`, `generate_stress_ops`, `do_all_gather`, `test` | All-Gather/Reduce-Scatter 正确性 + 带宽 |
| `test_pp.py` | 139 | `generate_stress_ops`, `test` | Pipeline Parallel send/recv 正确性 + 延迟 |

两个文件共享相同的测试范式：**Stress Test（随机操作序列验证正确性）+ Profiling（性能基准测量）**。

---

## 2. AGRS 测试 (`test_agrs.py`) 深度分析

### 2.1 函数角色

| 函数 | 行号 | 职责 |
|------|------|------|
| `all_gather_ref` | L12-17 | CPU 参考实现：生成各 rank 随机张量并 stack 为真值 |
| `generate_stress_ops` | L20-54 | 生成随机 AGRS 操作序列（create/destroy session、ag、fetch） |
| `do_all_gather` | L57-84： | 封装 buffer 的 all_gather 调用，支持 inplace/batched 模式 |
| `test` | L89-192 | 主测试逻辑：配置 → stress → profiling → 销毁 |

### 2.2 参考实现：确定性真值生成

```python
# L12-17
def all_gather_ref(shape: tuple, rank_idx: int, num_ranks: int, round_idx: int = 0):
    ref_list = []
    for i in range(num_ranks):
        torch.manual_seed(42 + round_idx * 43 + i)  # 确定性种子
        ref_list.append(torch.randn(shape, dtype=torch.bfloat16, device='cuda'))
    return ref_list[rank_idx], torch.stack(ref_list, dim=0)
```

**关键设计**：
- 每个 rank 使用**独立但确定性**的种子 `42 + round_idx * 43 + i`，确保跨 rank 可复现
- 返回 `(local_tensor, gathered_tensor)` 对，local 用于输入，gathered 用于验证

### 2.3 压力测试：随机操作序列生成

```python
# L20-54
def generate_stress_ops(num_ops, num_max_inflight_agrs, shape, rank_idx, num_ranks):
    tensors, refs = zip(*(all_gather_ref(...) for i in range(num_ops)))
    unprocessed = random.sample(range(num_ops), num_ops)
    inflight, ops = [], [('create_session', (-1,))]
    limit = num_max_inflight_agrs
    while unprocessed or inflight:
        # 随机选择：ag / fetch / destroy
        op = random.choice(choices)
        if op == 'ag':
            b = tuple(unprocessed[-random.randint(1, max_g):])  # 随机批量
            ...
        elif op == 'fetch':
            ops.append(('fetch', inflight.pop(random.randrange(len(inflight)))))
        else:
            ops.extend([('destroy_session', (-1,)), ('create_session', (-1,))])
```

**测试覆盖维度**：
1. **Session 生命周期**：随机穿插 `create_session` / `destroy_session`
2. **Inflight 并发**：最多 `num_max_inflight_agrs` 个并发 AG 操作
3. **批量大小随机**：每次 AG 操作合并 1~max_g 个张量
4. **Fetch 顺序随机**：非 FIFO 的完成等待

### 2.4 All-Gather 执行封装

```python
# L57-84
def do_all_gather(buffer, is_inplace, is_batched, tensors, start_event=None):
    if is_inplace:
        ag_tensors = buffer.agrs_get_inplace_tensor(tuple(t.shape for t in tensors), torch.bfloat16)
        for x, y in zip(ag_tensors, tensors, strict=True):
            x.copy_(y)  # 拷贝到弹性缓冲区
    else:
        ag_tensors = tensors

    if start_event is not None:
        torch.zeros(int(256e6 // 4), dtype=torch.int, device='cuda')  # flush L2 cache
        start_event.record()

    if is_batched:
        *out_tensors, handle = buffer.all_gather(ag_tensors)
        return out_tensors, [handle]
    else:
        # 逐个调用，每个返回独立 handle
        out_tensors, handles = [], []
        for t in ag_tensors:
            out_tensor, handle = buffer.all_gather(t)
            ...
```

**关键模式**：
- **Inplace 模式**：通过 `agrs_get_inplace_tensor` 获取 buffer 内零拷贝槽位
- **L2 Cache Flush**：`torch.zeros(256e6 // 4)` 分配 256MB 冲刷 L2，确保计时准确
- **Batched vs 非 Batched**：batched 返回单个 handle，非 batched 返回 handle 列表

### 2.5 主测试流程

```python
# L93-110: 配置阶段
shape = (32, 64, 2048)  # 固定形状
num_max_session_bytes = deep_ep.ElasticBuffer.get_agrs_num_max_session_bytes(
    group, [shape for _ in range(num_max_inflight_agrs)], torch.bfloat16)
num_max_session_bytes = deep_ep.ElasticBuffer.get_agrs_buffer_size_hint(
    group, num_max_session_bytes)
buffer = deep_ep.ElasticBuffer(group, explicitly_destroy=True, num_bytes=num_max_session_bytes)
buffer.agrs_set_config(num_max_session_bytes, num_max_inflight_agrs)
```

**配置链**：
1. `get_agrs_num_max_session_bytes` → 计算 session 所需总字节（含 32 字节对齐）
2. `get_agrs_buffer_size_hint` → 对齐到 2MB 的 buffer 大小
3. `ElasticBuffer` 构造 → 显式销毁模式
4. `agrs_set_config` → 设置运行时参数（含 barrier 刷新）

### 2.6 正确性验证

```python
# L136-138
for i in range(num_ops):
    assert results[i] is not None and torch.equal(results[i], refs[i]), \
        f'Rank {rank_idx}: stress mismatch at seed={seed}, op={i}'
```

**验证策略**：
- `torch.equal` 精确比较（非 allclose），因为是无损 all-gather
- 每个 op 必须完成（`is not None`）
- 跨 seed 迭代（默认 4 次），每次 128 个 ops

### 2.7 Profiling：带宽测量

```python
# L156-187
for num_bytes in (2 ** p for p in range(20, 27)):  # 1MB 到 64MB
    shape = (num_bytes // 2, )  # bfloat16 = 2 bytes
    for is_inplace in (False, True):
        for is_batched in (False, True):
            # 50 次迭代，丢弃首次（warmup）
            times = np.array([s.elapsed_time(e) / 1e3 ...])[1:]
            bandwidth_info = f'{num_bytes * num_ranks * num_max_inflight_agrs / avg_t / 1e9:.3f} GB/s'
```

**测量矩阵**：
- 数据大小：1MB, 2MB, 4MB, 8MB, 16MB, 32MB, 64MB
- 模式组合：`(inplace=False/True) × (batched=False/True)` = 4 种
- 使用 CUDA Event 计时，丢弃首次 warmup
- 报告单位：微秒（per AG）+ 聚合带宽（GB/s）

---

## 3. PP 测试 (`test_pp.py`) 深度分析

### 3.1 函数角色

| 函数 | 行号 | 职责 |
|------|------|------|
| `generate_stress_ops` | L13-37 | 生成随机 send/recv 操作序列（含时间戳） |
| `test` | L42-121 | 主测试逻辑：配置 → stress → profiling → 销毁 |

### 3.2 压力测试：基于时间戳的 PP 操作序列

```python
# L13-37
def generate_stress_ops(rank_idx, num_ranks, num_sends, shape):
    send_times = {(s, d): [] for s in range(num_ranks) for d in range(num_ranks) if s != d}
    recv_times = {(s, d): [] for s in range(num_ranks) for d in range(num_ranks) if s != d}

    for _ in range(num_sends):
        src_rank_idx = random.randint(0, num_ranks - 1)
        dst_rank_idx = (src_rank_idx + (1 if random.randint(0, 1) else -1)) % num_ranks  # 仅相邻 rank
        st = random.randint(0, 10 ** 8)
        rt = st + random.randint(1, 3 * 10 ** 6)  # recv 在 send 之后
        send_times[(src_rank_idx, dst_rank_idx)].append(st)
        recv_times[(src_rank_idx, dst_rank_idx)].append(rt)
```

**关键约束**：
- **仅相邻 rank 通信**：`(src ± 1) % num_rings`，符合 PP ring 拓扑
- **时间戳排序**：send/recv 按时间戳排序后交错执行
- **Recv 滞后**：`rt = st + random(1, 3e6)` 模拟真实 PP 中 recv 在 send 之后

### 3.3 主测试流程

```python
# L44-59: 配置阶段
shape = (args.num_tokens, args.hidden)  # 默认 (4096, 7168)
num_max_tensor_bytes = math.prod(shape) * 2  # bfloat16
buffer = deep_ep.ElasticBuffer(
    group, explicitly_destroy=True, allow_hybrid_mode=False,  # 禁用 hybrid 模式
    num_bytes=deep_ep.ElasticBuffer.get_pp_buffer_size_hint(
        num_max_tensor_bytes, num_max_inflight_tensors))
buffer.pp_set_config(num_max_tensor_bytes, num_max_inflight_tensors)
```

**与 AGRS 的区别**：
- `allow_hybrid_mode=False`：PP 测试禁用 hybrid 模式（纯 RDMA 路径）
- `get_pp_buffer_size_hint`：buffer 大小 = `num_max_tensor_bytes × num_max_inflight × 2(send/recv) × 2(prev/next)`

### 3.4 Stress Test 执行与验证

```python
# L69-80
prev = 0
for j, (op, timestamp, peer, _, tensor) in enumerate(ops):
    if op == 'send':
        buffer.pp_send(tensor, peer)
    else:
        result = torch.empty_like(tensor)
        buffer.pp_recv(result, peer)
        assert torch.equal(result, tensor), \  # 逐张量精确匹配
            f'Rank {rank_idx}: mismatch at op {j}'
    if timestamp > prev:
        torch.cuda._sleep(int((timestamp - prev) / 10 ** 8 * args.num_sleep_cycles))
    prev = timestamp
```

**验证策略**：
- **逐张量精确匹配**：`torch.equal(result, tensor)` — 发送端 tensor 直接与接收端比较
- **时间戳驱动睡眠**：`torch.cuda._sleep` 模拟计算间隔，按时间戳差值比例睡眠
- **隐式假设**：send 和 recv 的 tensor 是同一个对象（生成时共享引用）

### 3.5 Profiling：延迟与带宽测量

```python
# L86-116
num_approx_rdma_cycles = int(num_max_tensor_bytes * 2 / get_rdma_gbs() * 1.5)

for hide_rdma_latency in (True, False):
    for num_concurrent in (1, 2, 3):
        def loop(_hide_rdma_latency=hide_rdma_latency):
            torch.zeros((131072, 32768), dtype=torch.int, device='cuda')  # L2 flush
            for t in send_tensors:
                buffer.pp_send(t, (rank_idx + 1) % num_ranks)
            if _hide_rdma_latency:
                torch.cuda._sleep(num_approx_rdma_cycles * num_concurrent)
            for t in recv_tensors:
                buffer.pp_recv(t, (rank_idx - 1) % num_ranks)

        send_t, recv_t = bench_kineto(
            loop, kernel_names=('send_impl', 'recv_impl'),
            barrier_comm_profiling=True, barrier=buffer.barrier,  # 使用 buffer 的 barrier
            trace_path=get_trace_path(...))
```

**测量矩阵**：
- `hide_rdma_latency`：True（隐藏 RDMA 延迟，测量纯计算重叠）/ False（暴露延迟）
- `num_concurrent`：1, 2, 3 并发张量
- 使用 `bench_kineto` + Kineto profiler 追踪
- 报告：send/recv 延迟（μs）+ 带宽（GB/s）

**`hide_rdma_latency` 语义**：
- `True`：send 后睡眠等待 RDMA 完成，recv 测量的是"数据已到达"后的纯读取延迟
- `False`：recv 测量的是包含 RDMA 传输的完整延迟

---

## 4. API 使用模式对比

### 4.1 AGRS API 调用链

```
ElasticBuffer(group, explicitly_destroy=True, num_bytes=...)
  ├── .agrs_set_config(num_max_session_bytes, num_max_inflight_agrs)
  ├── .create_agrs_session() / .destroy_agrs_session()
  ├── .agrs_new_session()  [context manager]
  ├── .agrs_get_inplace_tensor(shapes, dtype)  [inplace 模式]
  └── .all_gather(tensors) → (gathered_tensors, handle)
```

### 4.2 PP API 调用链

```
ElasticBuffer(group, explicitly_destroy=True, allow_hybrid_mode=False, num_bytes=...)
  ├── .pp_set_config(num_max_tensor_bytes, num_max_inflight_tensors)
  ├── .pp_send(tensor, dst_rank_idx)
  ├── .pp_recv(tensor, src_rank_idx)
  └── .barrier  [用于 profiling 同步]
```

### 4.3 共同模式

| 模式 | AGRS | PP |
|------|------|-----|
| Buffer 预计算 | `get_agrs_num_max_session_bytes` → `get_agrs_buffer_size_hint` | `get_pp_buffer_size_hint` |
| 显式配置 | `agrs_set_config` | `pp_set_config` |
| 显式销毁 | `explicitly_destroy=True` | `explicitly_destroy=True` |
| 多进程启动 | `torch.multiprocessing.spawn` | `torch.multiprocessing.spawn` |

---

## 5. 参数变化空间

### 5.1 AGRS 测试参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--num-processes` | 8 | 参与测试的 GPU 数 |
| `--num-max-inflight-agrs` | 4 | 最大并发 AG 操作数 |
| `--num-stress-iterations` | 4 | stress 轮数（每轮 128 ops） |
| `shape` | `(32, 64, 2048)` | 固定张量形状（约 8MB/张量） |
| Profiling 大小 | `2^20` ~ `2^27` bytes | 1MB ~ 64MB |

### 5.2 PP 测试参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--num-processes` | 4 | 参与测试的 GPU 数 |
| `--num-tokens` | 4096 | token 数 |
| `--hidden` | 7168 | hidden 维度 |
| `--num-max-inflight-tensors` | 4 | 最大并发张量数 |
| `--num-stress-iterations` | 4 | stress 轮数 |
| `--num-sends` | 128 | 每轮 send 操作数 |
| `--num-sleep-cycles` | 10^7 | 睡眠周期基数 |
| `--dump-profile-traces` | `''` | Kineto trace 输出目录 |

---

## 6. 验证逻辑深度分析

### 6.1 AGRS 正确性验证

```python
# L136-138
assert results[i] is not None and torch.equal(results[i], refs[i])
```

**验证链**：
1. `all_gather_ref` 生成确定性真值（各 rank 独立种子）
2. `generate_stress_ops` 生成随机操作序列
3. `do_all_gather` 执行实际 AG 操作
4. `handle()` 等待完成
5. `out.clone()` 保存结果
6. `torch.equal` 精确比较

**覆盖的边界条件**：
- Session 随机销毁/重建
- Inflight 达到上限后的 fetch/destroy 选择
- Batched vs 非 batched 混合
- Inplace vs 非 inplace 混合

### 6.2 PP 正确性验证

```python
# L76-77
assert torch.equal(result, tensor)
```

**关键假设**：
- 发送端和接收端共享同一个 `tensor` 对象引用（L31-35 生成时）
- 这意味着验证的是"数据无损传输"，而非"数据一致性"
- 时间戳排序保证了 send 在 recv 之前执行

---

## 7. 关键代码片段与模式

### 7.1 L2 Cache Flush 模式

```python
# AGRS: L71
torch.zeros(int(256e6 // 4), dtype=torch.int, device='cuda')  # 256MB

# PP: L98
torch.zeros((131072, 32768), dtype=torch.int, device='cuda')  # 131072*32768*4 = 16GB (!)
```

**注意**：PP 的 flush 量是 131072 × 32768 × 4 bytes = **16GB**，远超典型 GPU L2 缓存（H100 为 50MB）。这可能是有意为之的"全局内存写入"模式，用于清空所有缓存层级。

### 7.2 CUDA Event 计时模式

```python
# AGRS: L165-177
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
for i in range(num_tests):
    with buffer.agrs_new_session():
        _, wait_handles = do_all_gather(..., start_event=start_events[i])
        for h in wait_handles:
            h()
    end_events[i].record()
times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]  # 丢弃 warmup
```

### 7.3 时间戳驱动睡眠

```python
# PP: L78-80
if timestamp > prev:
    torch.cuda._sleep(int((timestamp - prev) / 10 ** 8 * args.num_sleep_cycles))
prev = timestamp
```

**语义**：将抽象时间戳映射到 GPU sleep 周期，模拟计算与通信重叠。

---

## 8. 实验性状态评估

### 8.1 文档标注

所有相关 API 均标注为 `(Experimental)`：

| API | 标注 |
|-----|------|
| `pp_set_config` | ✅ Experimental |
| `pp_send` | ✅ Experimental |
| `pp_recv` | ✅ Experimental |
| `create_agrs_session` | ✅ Experimental |
| `destroy_agrs_session` | ✅ Experimental |
| `agrs_new_session` | ✅ Experimental |
| `agrs_set_config` | ✅ Experimental |
| `agrs_get_inplace_tensor` | ✅ Experimental |
| `all_gather` | ✅ Experimental |

### 8.2 已知限制

| 限制 | 详情 |
|------|------|
| **PP 仅相邻 rank** | `pp_send`/`pp_recv` 文档明确"prev or next rank only" |
| **AGRS 固定 shape** | stress test 使用固定 `(32, 64, 2048)`，未覆盖变长 |
| **PP 禁用 hybrid** | `allow_hybrid_mode=False`，不测试 NCCL 回退路径 |
| **无 Reduce-Scatter** | 测试只覆盖 All-Gather，未测试 Reduce-Scatter（API 名称暗示应支持） |
| **无错误注入** | 测试仅覆盖 happy path，无超时/越界/并发溢出测试 |
| **进程数固定** | AGRS 默认 8 卡、PP 默认 4 卡，未自动化扩展 |

### 8.3 功能成熟度判断

| 维度 | 评估 |
|------|------|
| **API 完整性** | 中高 — session 生命周期、配置、执行、查询齐全 |
| **测试覆盖** | 中 — stress + profiling 双轨，但边界条件不足 |
| **生产就绪** | 低 — 明确标注 Experimental，限制较多 |
| **性能可观测** | 高 — 集成 Kineto profiling、CUDA Event 计时、带宽计算 |

---

## 9. 与核心 EP 功能的关系

### 9.1 Elastic Buffer 的定位

```
DeepEP Core (Production)
├── Dispatch / Combine (MoE expert parallelism)
├── Intra-node (NVLink)
└── Inter-node (RDMA)

Elastic Buffer (Experimental)
├── AGRS (All-Gather / Reduce-Scatter)
├── PP (Pipeline Parallelism send/recv)
└── Hybrid Mode (NCCL fallback)
```

### 9.2 设计意图推测

1. **AGRS**：为 MoE 层的 token distribution 提供比 NCCL 更灵活的 all-gather 语义（session 化、inplace、batched）
2. **PP**：为 PP 通信提供与 EP 统一的 buffer 管理，可能用于 3D/4D 并行场景
3. **共享 runtime**：两者共用 `ElasticBuffer.runtime`，共享 symmetric memory 基础设施

---

## 10. 总结

| 维度 | AGRS 测试 | PP 测试 |
|------|----------|---------|
| **测试类型** | 正确性 + 带宽 | 正确性 + 延迟 |
| **核心 API** | `all_gather`, `agrs_get_inplace_tensor` | `pp_send`, `pp_recv` |
| **验证方式** | 与 CPU 参考实现精确比较 | 发送/接收张量精确比较 |
| **压力来源** | 随机 session 生命周期 + inflight 并发 | 时间戳排序的 send/recv 交错 |
| **性能度量** | 聚合带宽 (GB/s) | 延迟 (μs) + 带宽 (GB/s) |
| **Profiler** | CUDA Event | Kineto + barrier_comm_profiling |
| **成熟度** | Experimental | Experimental |

两个测试文件展示了 DeepEP 从核心 EP 功能向更通用通信原语扩展的实验方向，采用**确定性随机测试 + 性能 profiling**的双重验证范式，为后续生产化提供了可复现的基准。
