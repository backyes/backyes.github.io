# DeepEP Legacy 测试文件对比分析

## 1. 概述

本文档对 DeepEP 三个 legacy 测试文件进行深入的源码级对比分析，覆盖 Normal（Internode）、Intranode、Low-Latency 三种通信模式。这三个文件共同构成了 DeepEP V1（NVSHMEM-based）的测试体系，与 V2（Elastic）的 `tests/elastic/test_ep.py` 形成鲜明对比。

| 文件 | 行数 | 通信模式 | 底层传输 | 核心 API |
|------|------|---------|---------|---------|
| `test_internode.py` | 395 | Normal (Internode) | NVLink + RDMA | `dispatch()` / `combine()` |
| `test_intranode.py` | 311 | Intranode | NVLink only | `dispatch()` / `combine()` |
| `test_low_latency.py` | 332 | Low-Latency | IBGDA (RDMA) | `low_latency_dispatch()` / `low_latency_combine()` |

---

## 2. 文件结构对比

### 2.1 整体结构

```
test_internode.py (395L)          test_intranode.py (311L)         test_low_latency.py (332L)
├── imports (1-14)                ├── imports (1-13)               ├── imports (1-11)
├── test_main() (18-314)          ├── test_main() (17-265)          ├── simulate_failure_and_skip() (14-30)
│   ├── Settings (28-34)          │   ├── Settings (19-25)          ├── query_mask_buffer_and_check() (33-36)
│   ├── Random data (36-56)       │   ├── Random data (28-40)       ├── test_main() (39-251)
│   ├── RDMA dispatch counts      │   ├── Expert meta (42-47)       │   ├── Settings (50-54)
│   │   (59-63)                   │   ├── Rank layout (49-62)       │   ├── Data prep (57-77)
│   ├── Expert meta (65-70)       │   ├── Layout validation(64-74)  │   ├── Dispatch loop (89-149)
│   ├── Rank layout (72-88)       │   ├── Config (77-78)            │   ├── Combine loop (151-191)
│   ├── Layout validation(90-101) │   ├── Dispatch test (90-194)    │   ├── Perf test (196-251)
│   ├── Config (104-105)          │   ├── Tune dispatch(196-230)    ├── test_loop() (255-312)
│   ├── Dispatch test (117-201)   │   ├── Tune combine (242-264)    └── main (315-332)
│   ├── Combine test (202-221)    ├── test_loop() (268-297)
│   ├── Tune dispatch (237-275)   └── main (300-311)
│   ├── Tune combine (287-313)
│   └── test_loop() (318-370)
└── main (373-395)
```

### 2.2 关键函数签名对比

```python
# test_internode.py L18-27
def test_main(args, num_sms, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group, skip_benchmark=False):

# test_intranode.py L17-18
def test_main(args, num_sms, local_rank, num_ranks, rank, buffer, group):

# test_low_latency.py L39-49
def test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
              use_logfmt=False, shrink_test=False, seed=0):
```

**关键差异**：
- internode 多了 `num_nodes` 和 `num_local_ranks` 参数（跨节点需要）
- low_latency 没有 `args` 命名空间，参数直接传递，多了 `use_logfmt` 和 `shrink_test`

---

## 3. 测试初始化差异

### 3.1 Buffer 创建

```python
# test_internode.py L327-332 — Normal 模式，跨节点
buffer = deep_ep.Buffer(group,
                        int(2e9),           # num_nvl_bytes = 2GB
                        int(1e9),           # num_rdma_bytes = 1GB
                        low_latency_mode=False,  # 显式关闭
                        num_qps_per_rank=num_qps_per_rank,
                        explicitly_destroy=True)
assert num_local_ranks == 8 and num_ranks > 8  # 必须跨节点
```

```python
# test_intranode.py L275-281 — 纯节点内，无 RDMA
buffer = deep_ep.Buffer(group,
                        int(2e9),           # num_nvl_bytes = 2GB
                        0,                  # num_rdma_bytes = 0
                        low_latency_mode=False,
                        num_qps_per_rank=1,  # 无需 RDMA QP
                        explicitly_destroy=True,
                        allow_mnnvl=args.allow_mnnvl)
```

```python
# test_low_latency.py L260-270 — 低延迟模式，IBGDA
num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
buffer = deep_ep.Buffer(group,
                        num_rdma_bytes=num_rdma_bytes,  # 自动计算大小
                        low_latency_mode=True,           # 开启低延迟
                        num_qps_per_rank=num_experts // num_ranks,  # QP 数 = 本地 expert 数
                        allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
                        explicitly_destroy=True,
                        allow_mnnvl=args.allow_mnnvl,
                        enable_shrink=args.shrink_test)
```

### 3.2 Buffer 创建参数对比表

| 参数 | Internode | Intranode | Low-Latency |
|------|-----------|-----------|-------------|
| `num_nvl_bytes` | 2GB | 2GB | 未指定（默认0） |
| `num_rdma_bytes` | 1GB | 0 | `get_low_latency_rdma_size_hint()` 计算 |
| `low_latency_mode` | `False` | `False` | `True` |
| `num_qps_per_rank` | `max(num_sms, ll_num_experts//num_ranks)` | 1 | `num_experts // num_ranks` |
| `allow_nvlink_for_low_latency_mode` | — | — | `not args.disable_nvlink` |
| `enable_shrink` | — | — | `args.shrink_test` |
| `allow_mnnvl` | — | `args.allow_mnnvl` | `args.allow_mnnvl` |

### 3.3 Config 配置

```python
# test_internode.py L104-105 — 5 参数 Config（含 RDMA）
rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (24, 48, 96, 144, 160) else 512)
config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)
#               Config(SMs, nvl_chunk, nvl_buffer, rdma_chunk, rdma_buffer)
```

```python
# test_intranode.py L77-78 — 3 参数 Config（仅 NVL）
nvl_buffer_size = 256
config = deep_ep.Config(num_sms, 8, nvl_buffer_size)
#               Config(SMs, nvl_chunk, nvl_buffer)
```

```python
# test_low_latency.py — 无 Config 对象
# low_latency_dispatch/combine 不接受 config 参数，配置由 buffer 内部管理
```

### 3.4 默认测试参数

| 参数 | Internode | Intranode | Low-Latency |
|------|-----------|-----------|-------------|
| `num_tokens` | 4096 | 4096 | **128** |
| `hidden` | 7168 | 7168 | 7168 |
| `num_topk` | 8 | 8 | 8 |
| `num_experts` | 256 | 256 | **288** |
| `num_topk_groups` | `min(num_nodes, 4)` | — | — |
| `num_processes` | 8 | 8 | 8 |

**关键洞察**：Low-Latency 模式默认 `num_tokens=128`（小 batch 低延迟场景），而 Normal/Intranode 为 4096（大 batch 高吞吐场景）。

---

## 4. API 使用模式

### 4.1 Dispatch API 对比

#### Internode/Intranode — High-Throughput API

```python
# test_internode.py L127-141
dispatch_args = {
    'x': current_x,                          # BF16 tensor 或 FP8 tuple
    'num_tokens_per_rank': num_tokens_per_rank,
    'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,  # internode only
    'is_token_in_rank': is_token_in_rank,
    'num_tokens_per_expert': num_tokens_per_expert,
    'config': config,
    'async_finish': async_mode
}
if with_topk:
    dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights})
if previous_mode:
    dispatch_args.update({'previous_event': buffer.capture()})
recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
```

#### Low-Latency — IBGDA API

```python
# test_low_latency.py L99-104
packed_recv_x, packed_recv_count, handle, event, hook = \
    buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
```

### 4.2 Combine API 对比

#### Internode/Intranode

```python
# test_internode.py L205-210
combine_args = {'x': recv_x, 'bias': (bias_0, bias_1), 'handle': handle, 'config': config, 'async_finish': async_mode}
if with_topk:
    combine_args.update({'topk_weights': recv_topk_weights})
combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
```

#### Low-Latency

```python
# test_low_latency.py L158-166
combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                     topk_idx,
                                                     topk_weights,
                                                     handle,
                                                     use_logfmt=use_logfmt,
                                                     async_finish=not return_recv_hook,
                                                     zero_copy=zero_copy,
                                                     return_recv_hook=return_recv_hook,
                                                     out=out)
```

### 4.3 API 参数对比表

| 特性 | `dispatch()` / `combine()` | `low_latency_dispatch()` / `low_latency_combine()` |
|------|---------------------------|--------------------------------------------------|
| **数据格式** | BF16 或 FP8 tuple | BF16，FP8 可选 |
| **topk_idx/topk_weights** | 可选参数 | 必须参数 |
| **config** | `deep_ep.Config` 对象 | 无（内部管理） |
| **previous_event** | 支持 | 不支持 |
| **async_finish** | 支持 | 支持 |
| **return_recv_hook** | 不支持 | 支持（延迟接收） |
| **num_worst_tokens** | 支持（intranode only） | 不支持 |
| **zero_copy** | 不支持 | 支持（combine） |
| **use_logfmt** | 不支持 | 支持（combine） |
| **round_scale/use_ue8m0** | 不支持 | 支持（dispatch） |
| **cumulative_local_expert_recv_stats** | 不支持 | 支持 |

---

## 5. 测试模式与参数组合

### 5.1 循环维度对比

```python
# test_internode.py L117-120 — 4 维组合 = 2×2×4×2 = 32 cases
for previous_mode in (False, True):          # 2
    for async_mode in (False, True):         # 2
        for current_x in (x_pure_rand, x, x_pure_rand_e4m3, x_e4m3):  # 4 (BF16/FP8 × rand/deterministic)
            for with_topk in (False, True):  # 2
```

```python
# test_intranode.py L90-93 — 4 维组合 = 2×2×3×2 = 24 cases (FP8 可能为 None)
for previous_mode in (False, True):
    for async_mode in (False, True):
        for current_x in filter(lambda elem: elem is not None, (x_pure_rand, x, x_e4m3)):  # 3
            for with_topk in (False, True):
```

```python
# test_low_latency.py L89-93 — 5 维组合 = 5×2×2×2×2 = 80 cases (最多)
for current_x in x_list:                                  # 5 (1 deterministic + 4 logfmt/rand)
    for return_recv_hook in (False, True):                # 2
        for dispatch_use_fp8 in (False, True):            # 2
            for round_scale in (False, True) if dispatch_use_fp8 else (False,):  # 1-2
                for use_ue8m0 in (False, True) if round_scale else (False,):     # 1-2
```

### 5.2 测试模式特性表

| 测试模式 | Internode | Intranode | Low-Latency |
|---------|-----------|-----------|-------------|
| `previous_event` (CUDA graph capture) | ✅ | ✅ | ❌ |
| `async_finish` | ✅ | ✅ | ✅ |
| `return_recv_hook` | ❌ | ❌ | ✅ |
| FP8 dispatch | ✅ | ✅ (SM90 only) | ✅ |
| `round_scale` (Pow2 scaling) | ❌ | ❌ | ✅ |
| `use_ue8m0` | ❌ | ❌ | ✅ |
| `zero_copy` combine | ❌ | ❌ | ✅ |
| `use_logfmt` | ❌ | ❌ | ✅ |
| `num_worst_tokens` | ✅ | ✅ | ❌ |
| Cached dispatch (handle reuse) | ✅ | ✅ | ❌ |
| Bias (0/1/2 tensors) | ✅ (固定2个) | ❌ | ❌ |

---

## 6. 验证逻辑

### 6.1 数据正确性验证

#### Internode/Intranode — `check_data()` 函数

```python
# test_internode.py L109-115
def check_data(check_x, recv_gbl_rank_prefix_sum):
    assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))  # 每行值相同（rank 编码）
    check_start = 0
    for i in range(num_ranks):
        check_end = recv_gbl_rank_prefix_sum[i].item()
        assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0  # 验证 rank 编码
        check_start = check_end
```

**核心思路**：用 `x = torch.ones(...) * rank` 构造确定性数据，接收端验证每个 token 的值等于发送端 rank。

#### Low-Latency — 逐 expert 验证

```python
# test_low_latency.py L110-149
for i in range(num_local_experts if do_check else 0):
    expert_id = rank * num_local_experts + i
    recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
    recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

    # 验证 expert 索引
    num_valid_tokens = recv_count.item()
    assert num_valid_tokens == (all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status == 0].sum().item()

    # 验证接收数据
    if current_x is x:
        recv_x_amin = recv_x[:, :-128].amin(dim=-1)
        assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
        # 验证 rank 编码
        assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
```

### 6.2 Combine 验证

#### Internode/Intranode

```python
# test_internode.py L212-214
check_x = (combined_x.float() - bias_0.float() - bias_1.float()) / is_token_in_rank.sum(dim=1).unsqueeze(1)
ref_x = x_pure_rand if is_rand else x
assert calc_diff(check_x, ref_x) < 5e-4 if current_x is x_pure_rand_e4m3 else 5e-6
```

#### Low-Latency

```python
# test_low_latency.py L178-181
diff = calc_diff(current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
assert torch.isnan(combined_x).sum().item() == 0
if not round_scale:
    assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}'
```

### 6.3 验证策略对比

| 验证项 | Internode | Intranode | Low-Latency |
|--------|-----------|-----------|-------------|
| **数据编码** | `x = ones * rank` | `x = ones * rank` | `x = ones * (rank - 128)` + 尾部 token ID |
| **行一致性** | `amin == amax` | `amin == amax` | `amin == amax` (前 128 列) |
| **rank 编码** | 直接比较 | 直接比较 | `rank - rank_offset` (避免 BF16 精度问题) |
| **token ID 追踪** | ❌ | ✅ (最后一列) | ✅ (最后 128 列) |
| **topk_idx 范围** | `[0, num_experts//num_ranks)` | `[0, num_experts//num_ranks)` | 全局 expert ID |
| **topk_weights 精度** | `< 1e-9` | `< 1e-9` | 不直接验证 |
| **combine 精度** | `< 5e-6` (BF16) / `< 5e-4` (FP8) | `< 5e-6` | `< 1e-5` (BF16) / `< 9e-4` (FP8) |
| **NaN 检查** | ❌ | ❌ | ✅ |

---

## 7. 性能调优

### 7.1 Dispatch 调优

#### Internode — 双维度网格搜索

```python
# test_internode.py L244-259
for nvl_chunk_size in range(4, 45, 4):      # NVL chunk: 4~40, step 4
    for rdma_chunk_size in range(4, 33, 4):  # RDMA chunk: 4~28, step 4
        config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
        tune_args = {'x': current_x, 'handle': handle, 'config': config}
        t, notify_t = bench_kineto(lambda: buffer.dispatch(**tune_args), ...)
```

#### Intranode — 单维度 + 默认配置

```python
# test_intranode.py L202-208
for nvl_chunk_size in tuple(range(4, 33, 2)) + (0, ):  # 0 表示默认配置
    if nvl_chunk_size > 0:
        config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
    else:
        deep_ep.Buffer.set_num_sms(num_sms)
        config = deep_ep.Buffer.get_dispatch_config(num_ranks)  # 查表法
```

#### Low-Latency — 无调优，直接 benchmark

```python
# test_low_latency.py L227-250
avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
# 分离 profiling
dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=...),
                                     kernel_names=('dispatch', 'combine'), ...)
```

### 7.2 调优策略对比

| 调优项 | Internode | Intranode | Low-Latency |
|--------|-----------|-----------|-------------|
| **NVL chunk 范围** | 4~40, step 4 | 4~30, step 2 | 不调优 |
| **RDMA chunk 范围** | 4~28, step 4 | — | 不调优 |
| **默认配置测试** | ❌ | ✅ (`get_dispatch_config`) | — |
| **benchmark 工具** | `bench_kineto` | `bench` | `bench` + `bench_kineto` |
| **SMs 调优** | 固定 24 | 固定 24 | — |
| **FP8 最佳配置收集** | `all_gather` rank 0 | `all_gather` rank 0 | — |
| **分离 profiling** | ❌ | ❌ | ✅ (dispatch/combine 分开) |

---

## 8. 特殊测试功能

### 8.1 压力测试 (Pressure Test)

```python
# test_internode.py L335-360
for seed in range(int(1e9)):
    torch.manual_seed(rank + seed)
    ref_hash = test_main(...)
    if args.pressure_test_mode == 0:
        break  # 只跑一次
    for _ in range(20):  # 重复 20 次验证确定性
        torch.manual_seed(rank + seed)
        current_hash = test_main(...)
        assert current_hash == ref_hash
```

```python
# test_low_latency.py L283-307
for seed in range(int(1e9) if do_pressure_test else 0):
    ref_hash = test_main(..., seed=seed)
    for _ in range(20):
        assert test_main(..., seed=seed) == ref_hash
```

### 8.2 失败模拟与 Shrink 测试 (Low-Latency 独有)

```python
# test_low_latency.py L14-30
def simulate_failure_and_skip(rank, api, expected_masked_ranks):
    failed_api_ranks = {
        'dispatch': 1,   # rank 1 在 dispatch 时失败
        'combine': 3,    # rank 3 在 combine 时失败
        'clean': 5       # rank 5 在 clean 时失败
    }
    if rank in expected_masked_ranks:
        return True  # 已失败，跳过
    if api in failed_api_ranks.keys():
        expected_masked_ranks.add(failed_api_ranks[api])
        if failed_api_ranks[api] == rank:
            print(f"Rank {rank} failed when first calling {api} communication API, exit...")
            return True
    return False
```

```python
# test_low_latency.py L33-36 — 查询 mask buffer 并验证
def query_mask_buffer_and_check(api, buffer, mask_status, expected_masked_ranks):
    buffer.low_latency_query_mask_buffer(mask_status)
    assert set(mask_status.nonzero().squeeze(-1).tolist()) == expected_masked_ranks
```

### 8.3 Low-Latency 与 Normal 兼容性测试

```python
# test_internode.py L362-365
if args.test_ll_compatibility:
    buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
    test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)
```

```python
# test_intranode.py L289-292
if test_ll_compatibility:
    buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
    test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)
```

**关键洞察**：Internode 和 Intranode 都通过 `import test_low_latency` 显式测试与 Low-Latency 模式的兼容性，需要先调用 `clean_low_latency_buffer` 清理缓冲区。

### 8.4 特殊功能对比

| 功能 | Internode | Intranode | Low-Latency |
|------|-----------|-----------|-------------|
| **压力测试** | ✅ (3 modes) | ❌ | ✅ |
| **失败模拟 (shrink)** | ❌ | ❌ | ✅ |
| **LL 兼容性** | ✅ (optional) | ✅ (optional) | — |
| **MNNVL 支持** | ❌ | ✅ | ✅ |
| **NVLink 禁用** | ❌ | ❌ | ✅ |
| **LogFMT** | ❌ | ❌ | ✅ |
| **hash 校验** | ✅ | ❌ | ✅ |
| **num_worst_tokens** | ✅ | ✅ | ❌ |

---

## 9. Handle 结构对比

### 9.1 Internode Handle (10 元素 tuple)

```python
# legacy.py L500-502
handle = (is_token_in_rank, rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
          recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
          recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
          recv_src_meta, send_rdma_head, send_nvl_head)
```

### 9.2 Intranode Handle (6 元素 tuple)

```python
# legacy.py L401
handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix,
          recv_src_idx, is_token_in_rank, send_head)
```

### 9.3 Low-Latency Handle (2 元素 tuple)

```python
# test_low_latency.py L113
recv_src_info, recv_layout_range = handle[0][i], handle[1][i]
# handle = (recv_src_info, recv_layout_range)  # 每个 expert 对应一个 slice
```

---

## 10. Legacy vs Elastic 对比

### 10.1 API 风格对比

| 特性 | Legacy (V1) | Elastic (V2) |
|------|-------------|--------------|
| **Buffer 类** | `deep_ep.Buffer` | `deep_ep.ElasticBuffer` |
| **Dispatch API** | `buffer.dispatch(x, num_tokens_per_rank, ...)` | `buffer.dispatch(x, topk_idx, topk_weights, num_sms, num_qps, ...)` |
| **Combine API** | `buffer.combine(x, handle, ...)` | `buffer.combine(x, handle, ...)` |
| **Config** | `deep_ep.Config(SMs, nvl_chunk, nvl_buffer, rdma_chunk, rdma_buffer)` | 无（`num_sms` + `num_qps` 直接传） |
| **Handle** | 原始 tuple | `EPHandle` 对象（具名属性） |
| **Layout 计算** | `buffer.get_dispatch_layout()` | 内部自动完成 |
| **Expand 模式** | ❌ | ✅ (`do_expand=True`) |
| **Cached 模式** | 传 `handle` 复用 | 传 `handle` 复用 |
| **zero_padding** | ❌ | ✅ |
| **allocate_on_comm_stream** | ✅ | ✅ |
| **async_with_compute_stream** | ❌ | ✅ |

### 10.2 测试风格对比

| 特性 | Legacy | Elastic |
|------|--------|---------|
| **参考实现** | 手工计算 `num_tokens_per_rank` 等 | `ref_dispatch` / `ref_combine` (NCCL-based) |
| **参数组合** | 嵌套 for 循环 | `enumerate_ep_modes()` 生成器 |
| **验证方式** | `check_data()` + `calc_diff()` | 与 NCCL 参考实现逐元素比较 |
| **默认参数** | 硬编码 | argparse |
| **分布式启动** | `torch.multiprocessing.spawn` | `torch.multiprocessing.spawn` |
| **性能报告** | 打印 GB/s | 打印 GB/s + trace 导出 |

### 10.3 关键代码对比

#### Legacy Dispatch 调用

```python
# test_internode.py L127-141
dispatch_args = {
    'x': current_x,
    'num_tokens_per_rank': num_tokens_per_rank,
    'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
    'is_token_in_rank': is_token_in_rank,
    'num_tokens_per_expert': num_tokens_per_expert,
    'config': config,
    'async_finish': async_mode
}
recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
```

#### Elastic Dispatch 调用

```python
# test_ep.py L144-153
dispatch_args = dict(
    x=x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_sms=num_sms, num_qps=num_qps,
    num_max_tokens_per_rank=num_max_tokens_per_rank, num_experts=num_experts,
    expert_alignment=expert_alignment,
    async_with_compute_stream=async_with_compute_stream,
    allocate_on_comm_stream=allocate_on_comm_stream,
    do_handle_copy=do_handle_copy, do_cpu_sync=args.do_cpu_sync)
recv_x, recv_topk_idx, recv_topk_weights, handle, dispatch_event = \
    launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, dispatch_args)
```

---

## 11. 关键代码片段

### 11.1 数据构造模式

```python
# test_internode.py L37-41 — 确定性数据 + FP8 转换
x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
x_e4m3 = per_token_cast_to_fp8(x)
x_pure_rand_e4m3 = per_token_cast_to_fp8(x_pure_rand)
x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)  # scale 转置为 column-major (TMA 兼容)
```

```python
# test_low_latency.py L57-68 — 带 token ID 追踪的数据
rank_offset = 128
x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)  # 尾部嵌入 token ID
```

### 11.2 Layout 计算与验证

```python
# test_internode.py L90-98
ref_num_tokens_per_rank, ref_num_tokens_per_rdma_rank, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
    buffer.get_dispatch_layout(topk_idx, num_experts)
assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
```

### 11.3 Cached Dispatch 测试

```python
# test_internode.py L191-200
if not with_topk:
    dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
    if previous_mode:
        dispatch_args.update({'previous_event': buffer.capture()})
    recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
    event.current_stream_wait() if async_mode else ()
    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
    if not is_rand:
        check_data(recv_x, recv_gbl_rank_prefix_sum)
```

### 11.4 num_worst_tokens 测试

```python
# test_internode.py L175-189
if with_topk:
    num_worst_tokens = num_tokens * num_ranks
    dispatch_args.update({'num_worst_tokens': num_worst_tokens})
    recv_worst_x, recv_worst_topk_idx, recv_worst_topk_weights, empty_list, _, event = buffer.dispatch(**dispatch_args)
    assert len(empty_list) == 0
    assert num_worst_tokens == recv_worst_x.size(0)
    assert torch.equal(recv_x, recv_worst_x[:recv_x.size(0)])
    assert torch.all(recv_worst_topk_idx[recv_x.size(0):] == -1).item()  # 超出部分填充 -1
```

### 11.5 Low-Latency Return Hook 模式

```python
# test_low_latency.py L99-104
packed_recv_x, packed_recv_count, handle, event, hook = \
    buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                use_fp8=dispatch_use_fp8,
                                async_finish=not return_recv_hook,
                                return_recv_hook=return_recv_hook)
hook() if return_recv_hook else event.current_stream_wait()
```

### 11.6 Low-Latency Zero-Copy Combine

```python
# test_low_latency.py L154-166
for zero_copy in (False, ) if use_logfmt else (False, True):
    if zero_copy:
        buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                         topk_idx, topk_weights, handle,
                                                         use_logfmt=use_logfmt,
                                                         zero_copy=zero_copy, out=out)
```

---

## 12. 共享模式

### 12.1 共同导入

```python
# 三个文件共享
import argparse
import torch
import torch.distributed as dist
import deep_ep
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import calc_diff, per_token_cast_to_fp8, per_token_cast_back
```

### 12.2 共同测试模式

| 模式 | Internode | Intranode | Low-Latency |
|------|-----------|-----------|-------------|
| `previous_event` | ✅ | ✅ | ❌ |
| `async_finish` | ✅ | ✅ | ✅ |
| BF16/FP8 切换 | ✅ | ✅ | ✅ |
| with/without topk | ✅ | ✅ | ❌ (必须 topk) |
| `check_data()` 行一致性 | ✅ | ✅ | ✅ |
| `calc_diff()` 精度验证 | ✅ | ✅ | ✅ |

### 12.3 共同验证策略

1. **行一致性检查**：`amin(dim=1) == amax(dim=1)` — 确保每行值相同（rank 编码）
2. **rank 编码验证**：接收数据按 rank 分段，每段值等于发送端 rank
3. **topk_idx 范围检查**：`recv_topk_idx` 在 `[0, num_local_experts)` 或 `-1`
4. **combine 精度**：`calc_diff(combined_x, ref_x) < threshold`

---

## 13. 架构洞察

### 13.1 三层 API 设计

DeepEP V1 的 API 分为三层：

1. **High-Throughput (`dispatch`/`combine`)**：面向大 batch，支持 NVLink、RDMA、NVLink+RDMA
2. **Low-Latency (`low_latency_dispatch`/`low_latency_combine`)**：面向小 batch，仅 IBGDA
3. **Elastic (V2)**：统一 API，支持 scale-up/scale-out 分离

### 13.2 Config 的演进

- **V1**：`Config` 对象封装 5 个调优参数（SMs, nvl_chunk, nvl_buffer, rdma_chunk, rdma_buffer）
- **V2**：`Config` 消失，改为 `num_sms` + `num_qps` 直接传递，更灵活

### 13.3 Handle 的演进

- **V1**：原始 tuple，元素含义需查文档
- **V2**：`EPHandle` 对象，具名属性（`psum_num_recv_tokens_per_scaleup_rank` 等）

### 13.4 测试覆盖度

| 测试场景 | Legacy 覆盖 | Elastic 覆盖 |
|---------|-------------|--------------|
| BF16/FP8 | ✅ | ✅ |
| with/without topk | ✅ | ✅ |
| async/previous_event | ✅ | ✅ |
| Expand 模式 | ❌ | ✅ |
| Cached 模式 | ✅ | ✅ |
| zero_copy | ✅ (LL only) | ✅ |
| LogFMT | ✅ (LL only) | ✅ |
| 失败恢复 (shrink) | ✅ (LL only) | ✅ |
| 压力测试 | ✅ | ✅ |
| MNNVL | ✅ | ✅ |

---

## 14. 总结

### 14.1 关键发现

1. **三种模式共享测试框架**：嵌套 for 循环 × 多维度组合，但循环维度不同（Internode 4D, Low-Latency 5D）
2. **Low-Latency 测试最复杂**：80 个测试组合（含 round_scale/ue8m0/zero_copy/logfmt），且有独有的失败模拟功能
3. **性能调优策略不同**：Internode 双维度网格搜索，Intranode 单维度+查表，Low-Latency 无调优
4. **兼容性测试**：Internode/Intranode 通过 `import test_low_latency` 显式测试与 Low-Latency 的共存
5. **数据编码技巧**：`x = ones * rank` 构造确定性数据，Low-Latency 额外嵌入 token ID 用于精确追踪

### 14.2 Legacy vs Elastic 核心差异

| 维度 | Legacy (V1) | Elastic (V2) |
|------|-------------|--------------|
| **传输层** | NVSHMEM (IBGDA) | NCCL + NVSHMEM |
| **API 复杂度** | 高（3 种 API + Config） | 低（统一 API） |
| **Handle** | 原始 tuple | 具名对象 |
| **Expand 模式** | 不支持 | 支持 |
| **测试参考** | 手工计算 | NCCL 参考实现 |
| **调优方式** | 网格搜索 | 理论建模 (`get_theoretical_num_sms`) |

---

## 附录：文件行号索引

| 文件 | 关键函数/段落 | 行号 |
|------|-------------|------|
| `test_internode.py` | `test_main()` 定义 | 18-314 |
| | Buffer 创建 | 327-332 |
| | Config 创建 | 104-105 |
| | Dispatch 测试循环 | 117-201 |
| | Combine 测试循环 | 202-221 |
| | Dispatch 调优 | 237-275 |
| | Combine 调优 | 287-313 |
| | `test_loop()` | 318-370 |
| | main | 373-395 |
| `test_intranode.py` | `test_main()` 定义 | 17-265 |
| | Buffer 创建 | 275-281 |
| | Config 创建 | 77-78 |
| | Dispatch 测试循环 | 90-194 |
| | Combine 测试循环 | 169-194 |
| | Dispatch 调优 | 196-230 |
| | Combine 调优 | 242-264 |
| | `test_loop()` | 268-297 |
| | main | 300-311 |
| `test_low_latency.py` | `simulate_failure_and_skip()` | 14-30 |
| | `query_mask_buffer_and_check()` | 33-36 |
| | `test_main()` 定义 | 39-251 |
| | Buffer 创建 | 260-270 |
| | Dispatch 测试循环 | 89-149 |
| | Combine 测试循环 | 151-191 |
| | Perf test | 196-251 |
| | `test_loop()` | 255-312 |
| | main | 315-332 |
