# DeepEP V2 ElasticBuffer 第一性原理深潜：统一接口、SM 估算、Engram 与 0 SM 通信的全景分析

---

## 📋 原始问题清单（13 问）

> 以下 13 个问题来源于对 DeepEP V2 架构的深度讨论，本报告基于源码逐一分析并给出答案。

### Q1: ElasticBuffer 为什么可以统一 Normal 和 Low-Latency 场景？

> "In V2, all EP operations — high-throughput and low-latency — are unified under a single ElasticBuffer interface. The buffer can be initialized by specifying MoE settings directly, and the optimal SM and QP counts are calculated analytically."
>
> Normal 和 Low-Latency 的系统影响主要量化因素是什么？

### Q2: SM 自动估算的本质是什么，跟 MOE 哪些关键要素有关？

```python
_num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)
```

> 例如，这儿 `num_experts`, `num_topk` 两个变量，是否描述了通信 pairs 数量？为什么这里采用专家数，而不是卡数，是专家数，还是卡数？

**用户推测**：

- SM 要处理到所有专家/卡的连接，推测 deepep buffer 是一等公民，而 buffer 是 expert 专家粒度，所以要读写内存位置（buffer）数量是 expert 的函数，而不仅仅是卡数
- SM 处理逻辑里，要针对 token level 的处理，token level 有一个 topk 的复制数量，采用了 topk 粒度的发送控制逻辑，也可能 buffer 粒度没有 topk 概念，但是 buffer 间的数据搬移量跟 topk 相关
- 每卡处理负载应该与卡数有关，每卡上的专家数量跟卡数有关，为什么不考虑卡数这个概念，来估算 SM

### Q3: dispatch_forward 函数是否包含了 topk buffer 的生成？

```python
recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_experts=num_experts, num_max_tokens_per_rank=num_max_tokens_per_rank,
    expert_alignment=expert_alignment, num_sms=_num_comm_sms,
    async_with_compute_stream=True)
```

> Token 的复制过程，是否发生在 `buffer.dispatch` 内，且利用了 `topk_idx` 函数？

### Q4: 解码是否都是走的低时延模式？为什么低时延模式下可以设置路由 cache，而 prefill 和 train 没有？

```python
if cached_handle is not None:
    # Reuse cached handle: skip layout recomputation and CPU sync
    recv_x, _, _, handle, event = _buffer.dispatch(
        x, handle=cached_handle, num_sms=_num_comm_sms,
        async_with_compute_stream=True)
    return recv_x, cached_handle.topk_idx, None, handle, event
```

> 为什么只需要 cache 设置的情况下，无需 topk 的信息了（难道 100% cache 了么？）？
>
> 从第一性原理角度，低时延模式要优先使用 SM 来处理通信的控制和数据面，且建议减少中间通信算法的跳数以减低时延，DeepEP V2 的 ElasticBuffer 做了哪些优化，是 API 上统一了，还是通信算法上也做了统一？列出关键技术以及第一性原理，另外启发到 UB、URMA 和 UBMEM 的设计里去。

### Q5: DeepEP V2 强调 0 SM，是否意味 V2 发现通信哪些先验知识了？

> 例如，0 SM 是通信数据面交给了 DMA，不需要 SM 参与么？难道抛弃了所有 SM 参与？

### Q6: Engram 是否也服用了 ElasticBuffer API 设计？

### Q7: 从 ElasticBuffer 的 `use_fp8_dispatch` 配置，回溯分析 DeepEP Buffer 的 data type 以及 dispatch 和 combine 的区别？

> **注意区分**：ElasticBuffer 对象的 `dispatch` 和 `combine` API 不同于 EP 的 dispatch 和 combine，例如训练反向传播的时候 EP dispatch API 的反向是 ElasticBuffer combine 接口

```python
ElasticBuffer.get_buffer_size_hint(
    group, num_max_tokens_per_rank, hidden,
    num_topk=num_topk, use_fp8_dispatch=use_fp8_dispatch)
```

### Q8: Elastic Buffer 均设置了 `async_with_compute_stream=True` API 参数，是否意味着 DeepSeek V2 放弃了指令集的通算融合的系统设计？

### Q9: NV Gin 架构设计理念、关键软件架构、通信算法架构、软硬协同方面工作，性能数据评测对比？

> DeepEP V1/V2 对 Gin 以及上一代设计思想的变化？对 UB 的启发有哪些？

### Q10: DeepEP 对流量隔离的建议（EP 独立运作），走 IB 的 VL（Virtual Lane），DeepEP 软件上有什么协同措施？Adaptive Routing？

### Q11: DeepEP 认为应该优先隔离流量，采用 VL，而不是开启 IB 的拥塞机制。这点对 UB 有什么启示？

> UB 对应 VL 的技术是什么，UB 上拥塞和类"VL"技术的测试是否充分，性能如何？

### Q12: NV 针对小 token 的优化技术 Compute Fabric Transport (CFT) 是什么？

> nvDev 分支跟近 CUDA CFT 特性中，建议进一步打开分析，洞察前沿。

### Q13: 其他规划特性

> 1. **Torch Zero Copy 技术**：Tencent Network Platform Department 在贡献 PyTorch tensor 到网络报文之间的 100% zero copy，类似 2012 Network Lab 的 zero 技术
> 2. **Eager 特性**：如何进一步降低 RTT 时延影响，预期收益多大？
>    > "Using a low-latency protocol removes the extra RTT latency introduced by RDMA atomic OPs"
> 3. **TMA**：是在数据面，还是控制面，优化 SM 开销，以及 NVLink 通信域的？
>    > "TMA instructions for minimal SM usage and larger NVLink domain support (Hybrid-EP branch)"

### Q14 (综合): 解读 AntGroup 团队对 DeepEP 的改进规划

> 解读相关技术：
> - **Normal-SMFree**：Eliminating SM from RDMA path
> - **LL-SBO**：Overlapping Down GEMM computation with Combine Send communication
> - **LL-Layered**：Optimizing cross-node LL operator communication using rail-optimized forwarding and data merging
>
> **All Together 建议**：建议有一个 UB 的 EP offering 提供对客户 POC 的支持，强化对融合超节点总线/1825、池化 DDR（Engram）等的支持，当前 Tencent、AntGroup、ROCm/Mori-EP 的贡献。

---

## 分析范围与源码索引

> **核心源码文件**：
> - `deep_ep/buffers/elastic.py` (1108 行) — ElasticBuffer Python 层
> - `csrc/elastic/buffer.hpp` (1382 行) — ElasticBuffer C++ 核心
> - `deep_ep/include/deep_ep/impls/dispatch.cuh` — Direct dispatch kernel
> - `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` — Hybrid dispatch kernel
> - `deep_ep/include/deep_ep/impls/combine.cuh` — Combine kernel
> - `deep_ep/include/deep_ep/impls/engram_fetch.cuh` — Engram fetch kernel
> - `deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh` — Copy epilogue
> - `deep_ep/utils/event.py` — EventOverlap 机制
> - `DeepEP/README.md` — V2 官方文档
> - `DeepEP/docs/legacy.md` — V1 官方文档（含 FP8 dispatch + BF16 combine 配置）
> - `tests/elastic/test_ep.py` — V2 EP 测试（含 cached dispatch）

---

## 目录

1. [ElasticBuffer 为什么能统一 Normal 和 Low-Latency？](#1-elasticbuffer-normal-low-latency)
2. [SM 自动估算的本质是什么？](#2-sm)
3. [dispatch 内的 TopK 复制机制](#3-dispatch-topk)
4. [Decode 路由缓存：为什么只有 Low-Latency 模式支持？](#4-decode-low-latency)
5. [0 SM 通信的先验知识](#5-0-sm)
6. [Engram 是否复用了 ElasticBuffer API？](#6-engram-elasticbuffer-api)
7. [use_fp8_dispatch 回溯：Dispatch 与 Combine 的数据类型差异](#7-use_fp8_dispatch-dispatch-combine)
8. [async_with_compute_stream 的设计哲学](#8-async_with_compute_stream)
9. [NV Gin 架构设计理念与关键架构](#9-nv-gin)
10. [流量隔离与 VL](#10-vl)
11. [VL vs 拥塞控制：对 UB 的启示](#11-vl-vs-ub)
12. [CFT (Compute Fabric Transport) 前沿分析](#12-cft-compute-fabric-transport)
13. [其他规划特性：Zero Copy、Eager、TMA](#13-zero-copyeagertma)
14. [AntGroup 优化规划解读](#14-antgroup)
15. [通信方式量化要素抽象](#_15)
16. [UB EP Offering 建议](#_16)

---

## 1. ElasticBuffer 为什么能统一 Normal 和 Low-Latency？

### 1.1 V1 时代的割裂

在 DeepEP V1 中，Normal 和 Low-Latency 是**两套完全独立的实现**：

```python
# V1 Normal: 3-phase forwarding (internode.cu)
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(x, topk_idx=topk_idx, ...)

# V1 Low-Latency: 完全不同的 API (internode_ll.cu)
recv_hidden_states, recv_expert_count, handle, event, hook = \
    buffer.low_latency_dispatch(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts)
```

**关键问题**：
- 两个 API 使用不同的 buffer 分配策略
- 不同的 kernel 文件
- 不同的 handle 结构
- Low-Latency 模式无法与 Normal 模式共享 buffer

### 1.2 V2 的统一设计哲学

V2 的核心洞察：**Normal 和 Low-Latency 的本质差异不是"通信协议"，而是"资源分配策略"**。

```python
# V2 统一 API
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_experts=num_experts,
    num_sms=num_sms,  # ← 关键参数：多=Normal, 少=Low-Latency
    ...
)
```

**统一的维度**：
1. **同一个 Buffer**：所有通信模式共享
2. **同一个 dispatch/combine API**：参数控制行为
3. **SM 数量**是主要区分因子（Normal 64-160 SMs, Low-Latency 4-8 SMs）

### 1.3 量化系统影响因素

#### 关键因子 1：Token 复制量（Replicated Token Volume）

```python
# elastic.py:770-772
def get_expected_topk(num_groups: int) -> float:
    """计算单个 token 期望去多少个不同的 rank（去重后）"""
    return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk) / math.comb(num_experts, num_topk))
```

**物理含义**：一个 token 选择 `num_topk` 个 expert，这些 expert 分布在多少个**不同的 rank**上（去重后）。

**示例**：EP=256, experts=256, topk=8
- 每个 rank 1 个 expert
- 期望去重 rank 数 ≈ 8（因为 topk=8 很少碰撞）

**示例**：EP=256, experts=2048, topk=8
- 每个 rank 8 个 expert
- 期望去重 rank 数 ≈ 8 × (1 - C(2040,8)/C(2048,8)) ≈ 0.25
- 大部分 token 都在本地 rank

#### 关键因子 2：HBM 读写带宽（SM 侧）

```python
# elastic.py:781-806
sm_read += 1 / num_expected_topk   # 每个 token 只需读一次
sm_write += ...                      # 写 send buffer + 发起 NVLink
```

**单位**：每 token 的 HBM 读写量（以 token 数据量为单位 1.0）

#### 关键因子 3：NVLink/RDMA 通信带宽（网络侧）

```python
# Direct 模式
nvlink_traffic += self.num_nvlink_ranks / self.num_ranks * (1 - 1 / self.num_nvlink_ranks)
rdma_traffic += (self.num_ranks - self.num_nvlink_ranks) / self.num_ranks
```

#### 关键因子 4：瓶颈识别

```python
# elastic.py:809-812
if self.num_scaleout_ranks > 1 and (rdma_traffic / rdma_gbs) > (nvlink_traffic / nvlink_gbs):
    bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
else:
    bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs
```

### 1.4 Normal vs Low-Latency 的统一解释

| 维度 | Normal 场景 | Low-Latency 场景 | 统一机制 |
|------|-------------|------------------|----------|
| **SM 数** | 多（64-160） | 少（4-8） | `num_sms` 参数控制 |
| **QP 数** | 多（SM×16） | 少（SM+1） | `get_theoretical_num_qps` 自动 |
| **prefer_overlap** | False | True | `prefer_overlap_with_compute` |
| **Buffer** | 相同 | 相同 | 同一 `ElasticBuffer` |
| **Kernel** | 相同 | 相同 | 同一 `dispatch_impl` |

**在 V2 中，"Low-Latency" 不是独立模式，而是 Normal 在特定参数配置下的特例**。

**来源**: `deep_ep/buffers/elastic.py` L728-834, `DeepEP/README.md` L115

---

## 2. SM 自动估算的本质是什么？

### 2.1 核心公式推导

```python
# elastic.py:818-825
num_sms = max(
    bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
    bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
)
num_sms = align(max(4, math.ceil(num_sms * 1.25)), 2)
```

#### 物理含义

```
num_sms = max(
    (网络瓶颈带宽 / 每token网络流量) × (每token HBM读量 / 每SM HBM读带宽),
    (网络瓶颈带宽 / 每token网络流量) × (每token HBM写量 / 每SM HBM写带宽)
)
```

**第一项**：为了满足网络读取需求，需要多少 SM 来提供 HBM 读数据
**第二项**：为了满足网络写入需求，需要多少 SM 来消费 HBM 写数据

#### 为什么是这个公式？—— Roofline 模型

```
时间 = max(网络时间, HBM时间)
     = max(流量/带宽, 数据量/(SM数 × 每SM带宽))
```

令 网络时间 = HBM时间（最优状态）：

```
流量 / 网络带宽 = 数据量 / (SM数 × 每SM带宽)
→ SM数 = (网络带宽 / 流量) × (数据量 / 每SM带宽)
```

### 2.2 为什么用 num_experts 和 num_topk，而不是卡数？

#### 关键洞察：SM 处理的是 Buffer，Buffer 是 Expert 粒度的

**用户的推测完全正确**：

> SM 要处理到所有专家/卡的连接，推测 deepep buffer 是一等公民，而 buffer 是 expert 专家粒度，所以要读写内存位置（buffer）数量是 expert 的函数，而不仅仅是卡数。

**源码验证**：

```python
# elastic.py:770-772
def get_expected_topk(num_groups: int) -> float:
    assert num_experts % num_groups == 0
    return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk) / math.comb(num_experts, num_topk))
```

**`num_expected_topk` 是 `num_experts` 和 `num_topk` 的函数**，而不是直接依赖卡数。

#### 为什么 Expert 数比卡数更本质？

**原因 1：Token 复制量取决于 Expert 分布，而非卡数**

考虑两种配置：
- 配置 A：EP=64, experts=64, topk=8 → 每卡 1 expert
- 配置 B：EP=64, experts=512, topk=8 → 每卡 8 expert

虽然卡数相同，但配置 A 每个 token 期望去 8 个不同 rank（去重后），配置 B 可能只有 1-2 个 rank。

**原因 2：Buffer 寻址是 Expert 粒度的**

```cpp
// dispatch.cuh:318-320
const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
```

**TopK idx 直接给出 expert index**，然后才转换为 rank index。SM 的负载取决于 **需要访问多少个不同的 expert → 多少个不同的 buffer slot**。

**原因 3：卡数的影响已隐式包含**

```python
# elastic.py:778
num_expected_topk = get_expected_topk(self.num_ranks)  # num_ranks = num_scaleout_ranks × num_scaleup_ranks
```

`num_ranks` 就是卡数（总 EP 数）。但 **卡数通过与 expert 分布的交互来影响流量**，而非独立变量。

### 2.3 TopK 的作用：Buffer 间数据搬移量

用户的第二个推测：

> SM 处理逻辑里，要针对 token level 的处理，token level 有一个 topk 的复制数量，采用了 topk 粒度的发送控制逻辑，也可能 buffer 粒度没有 topk 概念，但是 buffer 间的数据搬移量跟 topk 相关。

**源码验证**：

```cpp
// dispatch.cuh:315-328
EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
int stored_dst_rank_idx = -1;
if (lane_idx < kNumTopk) {
    const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
    const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
    stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
    tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
    if (topk_weights != nullptr)
        tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
    if (copied_topk_idx != nullptr)
        copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
}
```

**TopK 直接影响**：
1. **每个 token 需要发送多少次**：topk 个 expert → 去重后 N 个 rank → N 次发送
2. **Warp 内并行度**：topk ≤ 32（一个 warp 的 lane 数），每个 lane 处理一个 expert 选择
3. **Buffer slot 分配**：每个 topk 选择对应一个独立的 buffer slot

### 2.4 为什么不直接考虑卡数？

用户的第三个问题：

> 每卡处理负载应该与卡数有关，每卡上的专家数量跟卡数有关，为什么不考虑卡数这个概念，来估算 sms？

**回答**：卡数 **确实被考虑了**，但它是通过 `num_expected_topk` 间接体现的：

```python
# elastic.py:778
num_expected_topk = get_expected_topk(self.num_ranks)
```

**数学关系**：
- 固定 experts 和 topk 时，增加卡数 → 每卡 expert 数减少 → 去重后 rank 数增加 → 通信量增加 → 需要更多 SM
- 固定卡数和 topk 时，增加 experts → 每卡 expert 数增加 → 去重后 rank 数减少 → 通信量减少 → 需要更少 SM

**卡数本身不是独立变量**，因为：
- 如果 expert 数和卡数同比增加（保持每卡 expert 数不变），通信模式不变
- 真正决定通信量的是 **token 需要跨越多少边界**，这由 expert 分布决定

### 2.5 SM 估算的完整流程图

```
输入: num_experts, num_topk, num_ranks (卡数)
         ↓
    get_expected_topk(num_ranks)
         ↓
    num_expected_topk = num_ranks × (1 - C(E-E/R, k) / C(E, k))
         ↓
    ┌──────────────────────────────────────┐
    │  sm_read = 1 / num_expected_topk     │  (每 token HBM 读)
    │  sm_write = f(num_expected_topk, 卡数) │  (每 token HBM 写)
    │  rdma_traffic = g(卡数, num_expected_topk) │
    │  nvlink_traffic = h(卡数, num_expected_topk) │
    └──────────────────────────────────────┘
         ↓
    bottleneck = max(rdma_traffic/rdma_gbs, nvlink_traffic/nvlink_gbs)
         ↓
    num_sms = bottleneck × max(sm_read/sm_read_gbs, sm_write/sm_write_gbs)
         ↓
    num_sms = align(max(4, ceil(num_sms × 1.25)), 2)
```

**来源**: `deep_ep/buffers/elastic.py` L728-834

---

## 3. dispatch 内的 TopK 复制机制

### 3.1 用户的三个问题

1. `dispatch_forward` 函数是否包含了 topk buffer 的生成？
2. Token 的复制过程，是否发生在 `buffer.dispatch` 内？
3. 是否利用了 `topk_idx` 函数？

### 3.2 答案：是的，全部在 buffer.dispatch 内完成

`buffer.dispatch()` 是一个 **多阶段 kernel pipeline**，包含：

| 阶段 | Kernel | 作用 |
|------|--------|------|
| 1. Notify | `dispatch_impl` (notify warps) | 统计 expert → rank 映射 |
| 2. Dispatch | `dispatch_impl` (dispatch warps) | 根据 topk_idx 复制 token 到 send buffer |
| 3. Barrier | Gin barrier | 等待所有数据到达 |
| 4. Epilogue | `dispatch_copy_epilogue_impl` | 从 buffer 搬到 recv tensor |

### 3.3 阶段详解

#### 阶段 1：Notify Warps 统计（`dispatch.cuh:79-258`）

```cpp
// dispatch.cuh:95-107
const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;
for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
    // 每个 token 的 topk 选择
    const auto dst_expert_idx = lane_idx < kNumTopk ?
        static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
    if (dst_expert_idx >= 0)
        atomicAdd_block(expert_count + dst_expert_idx, 1);  // Expert 维度计数

    // Rank 维度需要去重
    const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
    if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
        atomicAdd_block(rank_count + dst_rank_idx, 1);  // Rank 维度去重计数
}
```

**关键**：`ptx::deduplicate` 确保同一个 rank 只计数一次，即使多个 expert 都在该 rank。

#### 阶段 2：Dispatch Warps 复制 Token（`dispatch.cuh:259-395`）

```cpp
// dispatch.cuh:278-394
const auto token_start = dispatch_warp_idx * kNumSMs + sm_idx;
const auto token_stride = kNumDispatchWarps * kNumSMs;
for (int token_idx = token_start; token_idx < num_tokens; token_idx += token_stride) {
    // 1. TMA load hidden data from global to shared memory
    ptx::tma_load_1d(tma_buffer.get_hidden_ptr(),
                     math::advance_ptr(x, token_i64_idx * kNumHiddenBytes), ...);

    // 2. Load top-k indices and weights
    if (lane_idx < kNumTopk) {
        const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
        const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
        stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
        tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;  // ← TopK idx 写入 smem buffer
        if (topk_weights != nullptr)
            tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
        if (copied_topk_idx != nullptr)
            copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;  // ← 复制 topk_idx
    }

    // 3. Deduplicate ranks and assign slots
    if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
        stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);

    // 4. Write dst_buffer_slot_idx (for later combine)
    if (lane_idx < kNumTopk) {
        const auto value = stored_dst_slot_idx >= 0 ?
            rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
        dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = value;  // ← 生成 slot idx
    }

    // 5. TMA store to send buffer (NVLink)
    const auto dst_ptr = stored_dst_slot_idx >= 0 ?
        gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), stored_dst_rank_idx) :
        nullptr;
    if (dst_ptr != nullptr)
        ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());

    // 6. RDMA put to remote ranks
    if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
        gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
    }
}
```

**TopK 复制的核心机制**：

```
for each token:
    for each topk selection (lane_idx < kNumTopk):
        expert_idx = topk_idx[token_idx][lane_idx]
        rank_idx = expert_idx / experts_per_rank
        slot_idx = atomicAdd(counter[rank_idx], 1)  # 原子分配 slot

        # 写入 smem TMA buffer
        tma_buffer.topk_idx[lane_idx] = expert_idx
        tma_buffer.topk_weights[lane_idx] = topk_weights[token_idx][lane_idx]

        # 发送数据到远端
        if rank_idx is NVLink-accessible:
            TMA store to symmetric buffer
        else:
            RDMA put to remote buffer
```

#### 阶段 3：Copy Epilogue（`dispatch_copy_epilogue.cuh`）

```cpp
// dispatch_copy_epilogue.cuh:70-81
for (int i = global_warp_idx; i < num_recv_tokens; i += kNumWarps * kNumSMs) {
    // 确定当前 token 来自哪个 rank
    while (i >= current_rank_end) {
        current_rank_idx += 1;
        current_rank_start = current_rank_end;
        current_rank_end = psum_num_recv_tokens_per_scaleup_rank[current_rank_idx];
    }
    const auto buffer_token = scaleup_buffer.get_rank_buffer(current_rank_idx).get_token_buffer(i - current_rank_start);

    // TMA load from buffer to smem
    ptx::tma_load_1d(tma_buffer.get_base_ptr(), buffer_token.get_base_ptr(), ...);

    // 确定目标位置
    if (kDoExpand) {
        // Expand mode: 按 expert 顺序排列
        dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);
    } else {
        // Non-expand mode: 保持到达顺序
        dst_tensor_idx = i;
    }

    // TMA store to recv tensor
    ptx::tma_store_1d(math::advance_ptr(recv_x, dst_tensor_idx * kNumHiddenBytes),
                      tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
}
```

### 3.4 TopK Buffer 的生成时机

**TopK buffer 在 dispatch kernel 内部生成**：

```cpp
// dispatch.cuh:321-325
tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
if (topk_weights != nullptr)
    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
if (copied_topk_idx != nullptr)
    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
```

**TMA buffer 的内存布局（TokenLayout）**：

```
[hidden (TMA-aligned)] [sf (TMA-aligned)] [metadata (TMA-aligned)] [mbarrier]
                                    ↓
                             metadata = topk_idx (num_topk * 4)
                                      + topk_weights (num_topk * 4)
                                      + src_token_global_idx (4)
                                      + linked_list_idx (num_topk * 4)
```

### 3.5 完整数据流图

```
input: x[num_tokens, hidden], topk_idx[num_tokens, topk], topk_weights[num_tokens, topk]
                                    ↓
                    ┌───────────────────────────────────────┐
                    │         dispatch_impl kernel          │
                    │                                       │
                    │  Notify Warps:                        │
                    │    for each token:                    │
                    │      for each topk (lane):            │
                    │        expert = topk_idx[token][lane] │
                    │        rank = expert / E_per_rank     │
                    │        atomicAdd(expert_count, 1)     │
                    │        atomicAdd(rank_count, 1)  // dedup │
                    │    → prefix sum → psum_num_recv_tokens │
                    │                                       │
                    │  Dispatch Warps:                      │
                    │    for each token:                    │
                    │      TMA load hidden → smem          │
                    │      load topk_idx → smem            │  ← TopK buffer 生成
                    │      load topk_weights → smem        │
                    │      for each topk (lane):            │
                    │        slot = atomicAdd(counter, 1)   │  ← slot 分配
                    │        if NVLink: TMA store           │  ← Token 复制
                    │        else: RDMA put                 │  ← Token 复制
                    │    → trigger PDL (Programmatic Launch)│
                    └───────────────────────────────────────┘
                                    ↓
                    ┌───────────────────────────────────────┐
                    │    dispatch_copy_epilogue_impl kernel │
                    │                                       │
                    │    for each received token:           │
                    │      TMA load from buffer → smem     │
                    │      determine dst_tensor_idx:        │
                    │        if expand: atomicAdd(psum)     │
                    │        else: dst = arrival order      │
                    │      TMA store to recv_x[dst]        │
                    │      store recv_topk_idx[dst]        │
                    │      store recv_topk_weights[dst]    │
                    └───────────────────────────────────────┘
                                    ↓
output: recv_x[num_recv_tokens, hidden], recv_topk_idx, recv_topk_weights, handle
```

### 3.6 Hybrid Dispatch 的特殊性

在 Hybrid 模式下，token 复制分为两步：

1. **Scaleout Warps**：将 token 从本地 GPU → RDMA → 远端 scaleout rank 的 scaleout_recv_buffer
2. **Forward Warps**：将 token 从 scaleout_recv_buffer → NVLink → 目标 scaleup rank 的 scaleup_buffer

```cpp
// hybrid_dispatch.cuh:447-455 — Scaleout warps
if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx != scaleout_rank_idx) {
    gin.put<ncclTeamTagRail>(  // RDMA put to remote scaleout rank
        scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
        scaleout_send_buffer.get_token_buffer(token_idx).get_base_ptr(),
        tma_buffer.get_num_bytes<false>(),
        stored_dst_scaleout_rank_idx,
        ncclGinOptFlagsAggregateRequests);
}

// hybrid_dispatch.cuh:593-598 — Forward warps
if (stored_dst_slot_idx >= 0) {
    const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(  // NVLink TMA store
        scaleup_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
        stored_dst_scaleup_rank_idx);
    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
}
```

**来源**: `deep_ep/include/deep_ep/impls/dispatch.cuh` L79-409, `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`

---

## 4. Decode 路由缓存：为什么只有 Low-Latency 模式支持？

### 4.1 EPHandle 的定义

`EPHandle` 是 `dispatch` 返回的**路由元数据对象**，包含所有用于后续 combine 的信息：

```python
class EPHandle:
    do_expand: bool
    num_experts: int
    expert_alignment: int
    num_max_tokens_per_rank: int
    num_sms: int  # dispatch 时使用的 SM 数, combine 时复用
    topk_idx: Tensor  # [num_tokens, num_topk], clone 的 expert indices
    psum_num_recv_tokens_per_scaleup_rank: Tensor  # [num_scaleup_ranks]
    psum_num_recv_tokens_per_expert: Tensor  # [num_local_experts]
    num_unaligned_recv_tokens_per_expert: Tensor  # [num_local_experts]
    num_recv_tokens_per_expert_list: list  # CPU-side per-expert counts
    recv_src_metadata: Tensor  # 源 token 元数据
    dst_buffer_slot_idx: Tensor  # 目标 buffer slot 索引
    token_metadata_at_forward: Optional[Tensor]  # Hybrid 模式
    channel_linked_list: Optional[Tensor]  # Hybrid 模式
```

**来源**: `deep_ep/buffers/elastic.py` L25-98

### 4.2 Cached Dispatch 机制

当 `handle` 参数被传入 `dispatch` 时，进入 **cached mode**：

```python
# elastic.py:937-944
if handle is not None:
    assert topk_idx is None  # 不再接受新的 topk_idx
    assert do_cpu_sync is None or not do_cpu_sync  # 禁止 CPU sync
    topk_idx = handle.topk_idx  # 复用缓存的 topk_idx
    num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, handle.num_max_tokens_per_rank)
    num_experts = value_or(num_experts, handle.num_experts)
    do_cpu_sync = False  # 关键：跳过 CPU sync
```

**Cached mode 的关键行为**：
1. **跳过 layout recomputation**：`cached_num_recv_tokens`, `cached_psum_*` 直接传入 C++ kernel
2. **跳过 CPU sync**：不需要等待 GPU→CPU 的信号
3. **跳过 topk_idx clone**：`do_handle_copy` 仍会执行
4. **C++ 层行为**：`cached_mode = cached_num_recv_tokens.has_value()`（`buffer.hpp` L734）

### 4.3 为什么需要路由缓存？

**Decode 场景的特殊性**：
- Batch size 小（128 tokens）
- Routing decisions 在连续 decode step 之间**可能不变**（如果 gating 结果一致）
- 每次重新计算 layout 有 CPU overhead
- CPU sync 会引入 latency

**使用方式**（`README.md` L271-329）：
```python
def decode_dispatch(x, topk_idx, topk_weights, ...):
    if cached_handle is not None:
        # Reuse cached handle: skip layout recomputation and CPU sync
        recv_x, _, _, handle, event = _buffer.dispatch(
            x, handle=cached_handle, ...)
        return recv_x, cached_handle.topk_idx, None, handle, event

    # 首次：完整计算
    recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(...)
    return recv_x, recv_topk_idx, recv_topk_weights, handle, event
```

### 4.4 为什么只需要 cache 设置的情况下，无需 topk 的信息了？

**回答**：是的，**100% cache 了**。

当 `cached_handle` 被传入时：
1. `topk_idx` 直接从 `handle.topk_idx` 复用 —— **不需要用户提供新的 topk_idx**
2. `psum_num_recv_tokens_per_scaleup_rank` —— 缓存了每个 scaleup rank 接收的 token 数前缀和
3. `psum_num_recv_tokens_per_expert` —— 缓存了每个 expert 接收的 token 数前缀和
4. `recv_src_metadata` —— 缓存了源 token 元数据
5. `dst_buffer_slot_idx` —— 缓存了目标 buffer slot 索引
6. `num_sms` —— 缓存了 SM 数量

**这意味着**：当 gating decisions 不变时，整个 dispatch 的 layout 计算可以完全跳过，只需要执行数据搬运。

### 4.5 为什么 Prefill 和 Train 没有路由缓存？

**Prefill 场景**：
- Batch size 大（4096-8192 tokens）
- 每个 batch 的 routing decisions **几乎不可能**与上一个 batch 相同
- Layout recomputation 的 overhead 被大 batch size 摊薄
- CPU sync 的 latency 被大计算量隐藏

**Train 场景**：
- 每个 training step 的 input data 不同
- Routing decisions 每次都变化
- 无法使用 cache

**Decode 场景**：
- Batch size 小（64-256 tokens）
- 连续 decode step 的 routing decisions **可能不变**（特别是 gating 结果稳定时）
- Layout recomputation 的 CPU overhead 占比大
- CPU sync 的 latency 直接影响 decode latency

### 4.6 从第一性原理看低时延模式的优化

用户的问题：

> 从第一性原理角度，低时延模式要优先使用 sm 来处理通信的控制和数据面，且建议减少中间通信算法的跳数以减低时延，deepv2 的 elastic buffer 做了哪些优化，是 api 上统一了，还是通信算法上也做了统一？

**V2 在两个层面都做了优化**：

#### API 层统一

1. **统一 ElasticBuffer**：所有通信模式共享同一个 Buffer 对象
2. **统一 dispatch/combine API**：参数控制行为，而非独立 API
3. **统一 EPHandle**：路由元数据结构统一
4. **统一 EventOverlap**：事件管理统一

#### 通信算法层统一

1. **统一 Kernel Pipeline**：
   - Notify Warps → Dispatch Warps → Barrier → Copy Epilogue
   - Normal 和 Low-Latency 使用**相同的 kernel 代码**
   - 区别仅在于 `num_sms` 参数

2. **统一 Buffer 模型**：
   - V1 Normal：5 层 buffer（NVL Send/Recv + RDMA Send/Recv + Expert）
   - V1 Low-Latency：2 层 buffer（ping-pong）
   - V2 Unified：统一 token layout

3. **统一同步机制**：
   - Gin barrier 替代 NVSHMEM 的复杂同步
   - mbarrier FIFO 替代传统 barrier

#### 关键技术及第一性原理

| 技术 | 第一性原理 | 启发到 UB/URMA/UBMem |
|------|-----------|---------------------|
| **Buffer 一等公民** | 内存是通信的"尺子"，所有操作围绕 buffer 展开 | Buffer 设计应先于 kernel 设计 |
| **SM 解析式计算** | Roofline 模型：带宽平衡决定最优 SM 数 | 避免 auto-tuning，用模型驱动 |
| **TopK ≤ 32 约束** | Warp lane 数 = 32，TopK 映射到 lane 并行 | 控制TopK上限以匹配硬件并行度 |
| **Cached Dispatch** | 时间局部性：routing decisions 在时间上重复 | 缓存路由元数据，跳过重复计算 |
| **EventOverlap** | 计算-通信 overlap 是隐藏延迟的唯一手段 | Stream 级 overlap 是基本设计模式 |
| **0 SM Engram** | 数据面与控制面分离：DMA 搬数据，SM 做控制 | 数据面应尽量卸载到硬件 DMA |

---

## 5. 0 SM 通信的先验知识

### 5.1 V1 的 Hook-Based Overlap

V1 Low-Latency 模式最引人注目的特性：**0 SM occupation** 通信。

```python
# docs/legacy.md:264-272
recv_hidden_states, recv_expert_count, handle, event, hook = \
    buffer.low_latency_dispatch(x, topk_idx, ...,
                                return_recv_hook=True)

# NOTES: the actual tensor will not be received only if you call `hook()`,
# it is useful for double-batch overlapping, but **without any SM occupation**
```

**机制**：
1. `return_recv_hook=True` 时，kernel 只**发起 RDMA 请求**，不等待完成
2. 返回的 `hook` 是一个 callable，调用时阻塞等待数据到达
3. RDMA 传输在**网络硬件**中进行，**不占用 GPU SMs**

### 5.2 V2 的变化

**重要**: V2 README 明确说明：

> "**Notes**: ... 0 SM RDMA low-latency EP is no longer supported"

V2 不再支持纯 0 SM 的 low-latency EP，但保留了：
- **0 SM Engram**（RDMA one-sided get）
- **0 SM PP**（RDMA send/recv）
- **0 SM CP**（Copy Engine）

### 5.3 0 SM 的本质：数据面与控制面分离

**V2 的先验知识**：

1. **数据面可以完全不需要 SM**：
   - RDMA NIC 有独立的 DMA engine
   - GPU SM 只需发起请求（写 QP doorbell）
   - 数据搬移由 NIC hardware 完成

2. **控制面仍需少量 SM**：
   - 发起 Gin 请求（`gin.put`, `gin.get`）
   - 管理 QP（Queue Pair）状态
   - 处理 completion notification

3. **V2 的设计选择**：
   - 不再追求"纯 0 SM"的 EP 通信
   - 而是通过 `async_with_compute_stream=True` 实现**计算-通信 overlap**
   - 少量 SM 用于通信控制，大部分 SM 用于计算

### 5.4 0 SM Engram 的实现

Engram 是 V2 中少数保留的 **0 SM** 功能：

```cpp
// buffer.hpp:292-309
launch_engram_fetch(
    nccl_context->dev_comm, nccl_context->window,
    math::advance_ptr(buffer, num_gpu_buffer_bytes),  // CPU segment
    fetched.data_ptr(),
    indices.data_ptr<int>(),
    ...
);
```

**机制**：
1. `engram_write` 将数据写入 CPU segment（`cudaMemcpyDeviceToDevice`）
2. `engram_fetch` 发起 **Gin RDMA get** 请求
3. 返回的 callable 在调用时等待完成
4. 数据传输由 **NIC hardware** 完成，**不占用 GPU SMs**

### 5.5 对 UB 的启发

**核心洞察**：0 SM 通信的本质是**数据面与控制面分离**。

- **数据面**：NIC DMA engine 负责数据搬移，不需要 SM
- **控制面**：SM 负责发起请求和管理状态
- **优化方向**：最小化控制面的 SM 开销，最大化数据面的 DMA 效率

**UB 设计建议**：
- 数据面采用 RDMA one-sided operations（get/put）
- 控制面采用 minimal SM footprint（< 4 SMs）
- 通过 doorbell batching 减少 SM 干预频率

---

## 6. Engram 是否复用了 ElasticBuffer API？

### 6.1 回答：是的，完全复用

Engram 的方法直接定义在 `ElasticBuffer` 类上：

```python
class ElasticBuffer:
    # ... dispatch/combine 方法 ...

    def engram_write(self, storage: torch.Tensor, sf: Optional[torch.Tensor] = None) -> None:
        """Write Engram storage data into the buffer."""
        self.runtime.engram_write(storage, sf)

    def engram_fetch(self, indices: torch.Tensor, num_qps: int = 0,
                     use_tma_aligned_col_major_sf: bool = False) -> Callable:
        """Fetch Engram entries from remote ranks via RDMA."""
        return self.runtime.engram_fetch(indices, num_qps, use_tma_aligned_col_major_sf)
```

**来源**: `deep_ep/buffers/elastic.py` L569-604

### 6.2 Engram 与 EP 共享的底层资源

| 资源 | EP Dispatch/Combine | Engram |
|------|---------------------|--------|
| **Buffer** | ✅ 共享 GPU Buffer | ✅ 共享 CPU Buffer |
| **NCCL Gin Context** | ✅ `nccl_context->dev_comm` | ✅ `nccl_context->dev_comm` |
| **NCCL Window** | ✅ `nccl_context->window` | ✅ `nccl_context->window` |
| **QP (Queue Pair)** | ✅ `num_allocated_qps` | ✅ `num_allocated_qps` |
| **Comm Stream** | ✅ `comm_stream` | ✅ `comm_stream` |
| **Barrier** | ✅ `barrier()` | ✅ `barrier()` |

### 6.3 Engram 的内存布局

```cpp
// buffer.hpp:229-236
// Write storage to CPU segment at back of buffer
const auto cpu_write_offset = allow_hybrid_mode
    ? static_cast<int64_t>(nccl_context->scaleup_rank_idx) * num_cpu_buffer_bytes : 0;
CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
    math::advance_ptr(buffer, num_gpu_buffer_bytes + cpu_write_offset),
    storage.data_ptr(), storage.nbytes(),
    cudaMemcpyDeviceToDevice, compute_stream));
```

**Hybrid Elastic Symmetric Memory**:
```
[ GPU VRAM | CPU rank0 | CPU rank1 | ... | CPU rank(N-1) ]
                       ↑ Engram storage 写入这里
```

### 6.4 Engram 与 Dispatch/Combine 的区别

| 维度 | Dispatch/Combine | Engram |
|------|------------------|--------|
| **数据方向** | All-to-All（每个 rank 都发都收） | One-sided（只读远端） |
| **数据类型** | BF16 或 FP8（dispatch） | BF16 或 FP8 |
| **Buffer 位置** | GPU Buffer | CPU Buffer |
| **SM 使用** | 多（4-160） | 0（RDMA one-sided get） |
| **同步** | Barrier + CPU sync | Barrier only |
| **使用场景** | MoE token routing | Remote KV cache fetch |

### 6.5 Engram 与 Rec Emb 查表过程的区别

**Rec Emb（推荐系统 Embedding 查表）**：
- 训练场景：需要 gradient update，使用 allreduce 或 parameter server
- 推理场景：只读，可使用 Engram 类似的远端 fetch

**Engram**：
- 只读（write 只在初始化时）
- 通过 RDMA get 直接读取远端
- 无需 CPU 介入，保持 GPU-centric

**关键区别**：
- **训练**：Embedding 查表需要写操作（gradient update），不能用 Engram
- **推理**：Embedding 查表是只读的，可以用 Engram 类似的机制

---

## 7. use_fp8_dispatch 回溯：Dispatch 与 Combine 的数据类型差异

### 7.1 官方文档的说明

V1 官方文档明确说明（`docs/legacy.md` L19, L30）：

> "FP8 dispatching and BF16 combining"

**测试配置**：
- **Training (Normal)**: 4096 tokens, 7168 hidden, top-4 groups, top-8 experts, **FP8 dispatching**, **BF16 combining**
- **Inference (Low-Latency)**: 128 tokens, 7168 hidden, top-8 experts, **FP8 dispatching**, **BF16 combining**

### 7.2 源码验证

#### Dispatch 支持 FP8

```python
# elastic.py:882-885
x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be
    `[num_tokens, hidden]`, and type must be `torch.bfloat16`; for the second type (FP8 mode),
    the first element of the tuple must be `[num_tokens, hidden]` with type `torch.float8_e4m3fn`,
    the second is the scale factors.
```

```cpp
// buffer.hpp:671-672
const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0;
```

#### Combine 只支持 BF16

```cpp
// buffer.hpp:1203
EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
```

```cpp
// combine.cuh:29
combine_impl(nv_bfloat16* x, ...)
```

### 7.3 为什么 FP8 Dispatch + BF16 Combine？

**第一性原理分析**：

1. **Dispatch 阶段**：
   - 输入是 MoE 前的 hidden states
   - 精度要求不高（只需 routing 到 expert）
   - FP8 可以节省 50% 通信带宽（1 byte vs 2 bytes）
   - 每个 token 附带 scaling factors（SF），用于反量化

2. **Combine 阶段**：
   - 输出是 MoE 后的 hidden states
   - 精度要求高（需要累加多个 expert 的输出）
   - BF16 保证累加精度
   - 避免 FP8 累加的精度损失

3. **Buffer Size 计算**：

```cpp
// buffer.hpp:671-672
const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0;
```

- FP8 dispatch：elem_size = 1 byte，SF packs = hidden/32 × 4 bytes
- BF16 dispatch：elem_size = 2 bytes，SF packs = 0

### 7.4 训练反向传播的类型对应

```python
# README.md:239-252
def combine_backward(grad_combined_x, handle):
    """The backward pass of MoE combine is actually a dispatch."""
    grad_x, _, _, _, event = _buffer.dispatch(
        grad_combined_x,
        handle=handle,  # ← 使用 dispatch API
        ...
    )
    return grad_x, event
```

**反向传播的类型对应**：

| 正向 | 反向 | API | 数据类型 |
|------|------|-----|----------|
| `dispatch_forward` | `combine_backward` | `dispatch` | BF16 → BF16（反向梯度） |
| `combine_forward` | `dispatch_backward` | `combine` | BF16 → BF16（反向梯度） |

**关键**：反向传播时，梯度是 BF16 类型，不走 FP8。

### 7.5 FP8 Dispatch 的 Buffer Size 影响

```python
# elastic.py:380-406
@staticmethod
def get_buffer_size_hint(group, num_max_tokens_per_rank, hidden,
                         num_topk=0, use_fp8_dispatch=False, ...):
    return _C.calculate_elastic_buffer_size(
        ..., use_fp8_dispatch=use_fp8_dispatch, ...)
```

**FP8 dispatch 的 buffer 影响**：
1. **Token data 大小减半**：FP8 = 1 byte/elem vs BF16 = 2 bytes/elem
2. **SF packs 增加**：每 32 个 hidden 元素需要 4 bytes SF
3. **总体 buffer 略小**：对于 hidden=7168，FP8 + SF ≈ 7168 + 7168/32×4 = 7168 + 896 = 8064 bytes/token vs BF16 = 14336 bytes/token

---

## 8. async_with_compute_stream 的设计哲学

### 8.1 API 语义

```python
# elastic.py:902-903
async_with_compute_stream: the current stream will not wait for the communication kernels to be
    finished if set.
```

**字面含义**：设置后，当前 compute stream 不会等待通信 kernel 完成。

### 8.2 实现机制

```cpp
// buffer.hpp:526-584
torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event,
                                                const bool& allocate_on_comm_stream,
                                                const bool& async_with_compute_stream) const {
    const auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(comm_stream);
    // ...
    return event;
}
```

**EventOverlap 机制**：

```python
# deep_ep/utils/event.py
class EventOverlap:
    def current_stream_wait(self, release_handle=False):
        """The current stream waits for the event to be finished."""
        assert self.event is not None
        self.event.current_stream_wait()
```

### 8.3 是否放弃了指令级的通算融合？

**回答**：不是"放弃"，而是"分层"。

**DeepSeek V2 的通信-计算 overlap 是分层的**：

1. **Stream 级别 Overlap**：
   - `async_with_compute_stream=True` 实现
   - Compute stream 和 Comm stream 并行
   - 通信 kernel 在 Comm stream 上执行
   - 计算 kernel 在 Compute stream 上执行
   - 通过 `event.current_stream_wait()` 同步

2. **Kernel 级别 Overlap**：
   - 在 dispatch kernel 内部，Notify Warps 和 Dispatch Warps 并行
   - TMA store 和计算可以 overlap
   - 这是硬件级别的 overlap（TMA engine 独立于 SM）

3. **指令级别 Overlap**：
   - PTX 指令级的并行（如 `wgmma` 与 `cp.async` overlap）
   - 这是 DeepGEMM 的特色，不是 DeepEP 的重点

**关键区别**：
- **DeepEP** 关注**通信-计算 overlap**（stream 级别）
- **DeepGEMM** 关注**计算-计算 overlap**（指令级别，wgmma 与 load overlap）

### 8.4 对 UB 的启发

**分层 Overlap 设计**：
1. **Stream 级别**：通信与计算分离到不同 stream，通过 event 同步
2. **Kernel 级别**：通知与数据搬运并行（Notify + Dispatch warps）
3. **硬件级别**：TMA engine 独立于 SM，实现真正的异步

**UB 设计建议**：
- 采用 stream-level overlap 作为基本设计模式
- 通信 kernel 使用独立 comm_stream
- 通过 event 机制实现精确同步
- 避免指令级通算融合的复杂性

---

## 9. NV Gin 架构设计理念与关键架构

### 9.1 Gin 的定义

**NCCL Gin** = **GPU-Initiated Networking**

核心思想：**GPU SM 直接操作 NIC QP（Queue Pair），无需 CPU 介入**。

### 9.2 关键软件架构

#### 9.2.1 Gin 的核心抽象

```cpp
// dispatch.cuh:69-71
const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumDispatchWarps, (kNumNotifyWarps > 0)>(
    sm_idx, warp_idx - kNumNotifyWarps, warp_idx < kNumNotifyWarps);
const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);
```

**关键组件**：
- `ncclDevComm_t`：NCCL 设备端通信器
- `ncclWindow_t`：对称内存窗口
- `qp_idx`：Queue Pair 索引
- `sharing_mode`：QP 共享模式（per-SM 或 per-CTA）

#### 9.2.2 Gin 的操作原语

| 操作 | 语义 | 使用场景 |
|------|------|----------|
| `gin.get<team_t>(dst, src, size, peer)` | RDMA one-sided read | Engram fetch |
| `gin.put<team_t>(dst, src, size, peer)` | RDMA one-sided write | Dispatch send |
| `gin.put_value<team_t>(dst, value, peer)` | RDMA write 64-bit value | Notify count |
| `gin.flush_async<team_t>(peer, request)` | 刷新 QP 请求 | 确保数据发送 |
| `gin.get_sym_ptr<team_t>(ptr, peer)` | 获取远端对称内存指针 | NVLink TMA store |

#### 9.2.3 QP 映射策略

```python
# elastic.py:836-853
def get_theoretical_num_qps(self, num_sms: int) -> int:
    # For direct mode, we encourage less QPs to reduce DB ringing overhead
    num_qps = min(num_sms, 8 + 1)

    # For hybrid mode, we encourage every channel (and notify) to have an independent QP
    if self.allow_hybrid_mode:
        num_qps = num_sms * 16 + 1

    return min(num_qps, self.num_allocated_qps)
```

**Direct 模式**：`num_qps = min(num_sms, 9)` —— 少量 QP，减少 doorbell 开销
**Hybrid 模式**：`num_qps = num_sms × 16 + 1` —— 每个 channel 独立 QP

### 9.3 通信算法架构

#### 9.3.1 Symmetric Memory 模型

```
Rank 0 GPU: [ Local Buffer ] ←→ [ Symmetric Window ]
                                      ↕ NVLink/RDMA
Rank 1 GPU: [ Local Buffer ] ←→ [ Symmetric Window ]
```

**关键**：每个 rank 的 symmetric window 映射到相同的虚拟地址，但实际访问远端内存。

#### 9.3.2 Barrier 机制

```cpp
// dispatch.cuh:74-76
comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                  kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag0, false, false, true>(
    gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);
```

**GPU-centric barrier**：
- 不使用 CPU 同步
- 通过 Gin 的 `put_value` 和 `ld_volatile` 实现
- 支持 timeout 检测死锁

### 9.4 软硬协同方面工作

| 硬件特性 | DeepEP 利用方式 |
|----------|----------------|
| **NIC DMA Engine** | RDMA one-sided get/put，无需 SM 参与 |
| **GPU TMA Engine** | TMA store/load，异步数据搬运 |
| **NVLink Fabric** | 对称内存直传，无需 CPU 介入 |
| **PCIe Atomic** | `PCI_ATOMIC_MODE=4` 提升 RDMA atomic 性能 |
| **GPU Doorbell** | SM 直接写 NIC doorbell，发起 RDMA 请求 |

### 9.5 性能数据评测对比

#### V2 官方性能（`README.md` L42-55）

| Arch | NIC type | Topo | Dispatch Bottleneck Bandwidth | Combine Bottleneck Bandwidth | #SMs |
|------|----------|------|-------------------------------|------------------------------|------|
| SM90 | CX7 | EP 8 × 2 | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |
| SM90 | CX7 | EP 8 × 4 | 61 GB/s (RDMA) | 61 GB/s (RDMA) | 6 |
| SM100 | CX7 | EP 8 × 2 | 90 GB/s (RDMA) | 91 GB/s (RDMA) | 12 |
| SM100 | N/A | EP 8 | 726 GB/s (NVLink) | 740 GB/s (NVLink) | 64 (Max perf) |
| SM100 | N/A | EP 8 | 643 GB/s (NVLink) | 675 GB/s (NVLink) | 24 (Min #SM) |

**关键**：V2 achieves up to **1.3x peak performance**, while saving up to **4x SM count** compared to V1.

#### V1 性能（`docs/legacy.md` L17-39）

**Normal kernels**：
- EP=16 internode: 43 GB/s, 4096 tokens
- EP=64 internode: 51 GB/s

**Low-Latency kernels**：
- EP=8: 77 μs latency, 98 GB/s
- EP=64: 173 μs latency, 43 GB/s
- EP=256: 194 μs latency, 39 GB/s

### 9.6 V1/V2 对 Gin 以及上一代设计思想的变化

| 维度 | V1 (NVSHMEM) | V2 (NCCL Gin) |
|------|--------------|---------------|
| **通信后端** | NVSHMEM | NCCL Gin |
| **编译方式** | 预编译 NVSHMEM kernel | Fully JIT |
| **Buffer 模型** | 分离 NVL + RDMA buffer | 统一 symmetric memory |
| **SM 控制** | 静态 `num_sms = 24` | 解析式自动计算 |
| **QP 控制** | 手动 `num_qps_per_rank` | 解析式自动计算 |
| **扩展规模** | EP ≤ 128 | EP ≤ 2048 |
| **新增功能** | 无 | Engram, PP, CP, AGRS |

### 9.7 对 UB 的启发

1. **GPU-centric 设计**：通信控制面在 GPU 上，无需 CPU 介入
2. **Symmetric Memory**：统一虚拟地址空间，简化编程模型
3. **QP 映射优化**：根据 SM 数和模式动态调整 QP 数量
4. **Barrier 硬件化**：利用 NIC 的原生 barrier 能力
5. **TMA 卸载**：数据搬运由 TMA engine 完成，释放 SM

---

## 10. 流量隔离与 VL

### 10.1 DeepEP 的官方建议

`README.md` L377-392：

> **Traffic isolation** is supported by InfiniBand through Virtual Lanes (VL).
>
> To prevent interference between different types of traffic, we recommend segregating workloads across different virtual lanes as follows:
> - expert-parallel workloads
> - other workloads
>
> For DeepEP V2, you can control the virtual lane assignment by setting the `sl_idx` argument or the `EP_OVERRIDE_RDMA_SL` environment variable.

### 10.2 软件上的协同措施

```python
# elastic.py:243
sl_idx: int = 3,  # RDMA service level index

# elastic.py:323-324
if 'EP_OVERRIDE_RDMA_SL' in os.environ:
    sl_idx = int(os.environ['EP_OVERRIDE_RDMA_SL'])
```

**SL → VL 映射**：
- InfiniBand 的 Service Level (SL) 决定 Virtual Lane (VL)
- 不同 VL 有不同的优先级和带宽分配
- DeepEP 推荐 EP 流量使用独立 VL

### 10.3 Adaptive Routing

`README.md` L388-389：

> **Adaptive routing** is an advanced routing feature provided by InfiniBand switches that can evenly distribute traffic across multiple paths. Even though adaptive routing introduces additional latency, we still recommend enabling it under all network load conditions.

**DeepEP 的建议**：
- **开启 Adaptive Routing**：即使引入额外延迟，也建议开启
- 原因：在多路径拓扑中，AR 可以均衡负载，避免热点

### 10.4 软件协同措施总结

| 措施 | 实现方式 | 目的 |
|------|----------|------|
| **VL 隔离** | `sl_idx` 参数 / `EP_OVERRIDE_RDMA_SL` 环境变量 | EP 流量与其他流量分离 |
| **Adaptive Routing** | IB 交换机配置 | 多路径负载均衡 |
| **QP 分配** | `num_allocated_qps` 参数 | 控制 RDMA 并发度 |
| **Traffic Class** | DSCP / TOS 字段 | QoS 标记 |

---

## 11. VL vs 拥塞控制：对 UB 的启示

### 11.1 DeepEP 的官方建议

`README.md` L390-392：

> **Congestion control** is disabled because it hurts maximum bandwidth. If congestion is unavoidable in some scenarios, we recommend assigning those workloads to low-priority virtual lanes.

**DeepEP 的立场**：
- **禁用拥塞控制**：因为会降低最大带宽
- **优先使用 VL 隔离**：将拥塞流量分配到低优先级 VL

### 11.2 对 UB 的启示

#### UB 对应 VL 的技术

| IB 技术 | UB 对应 | 作用 |
|---------|---------|------|
| **Virtual Lane (VL)** | Virtual Channel (VC) | 逻辑通道隔离 |
| **Service Level (SL)** | Priority / QoS | 优先级标记 |
| **Adaptive Routing** | Dynamic Routing | 动态路由 |

#### UB 上拥塞和类"VL"技术的测试

**关键问题**：
1. UB 是否支持多 VL/VC？
2. UB 的拥塞控制机制是什么？
3. VL 隔离在 UB 上的性能如何？

**建议测试**：
- 多 VL 隔离下的 EP 吞吐量
- 拥塞场景下的 VL 优先级调度
- VL 数量对性能的影响

### 11.3 设计建议

**UB 流量隔离设计**：
1. **优先 VL 隔离**：EP 流量使用独立 VL，避免与其他流量竞争
2. **禁用拥塞控制**：在数据中心内部，拥塞控制通常不必要
3. **Adaptive Routing**：在多路径拓扑中开启 AR
4. **QoS 标记**：使用 DSCP/TOS 标记 EP 流量

---

## 12. CFT (Compute Fabric Transport) 前沿分析

### 12.1 CFT 的定义

**Compute Fabric Transport (CFT)** 是 NVIDIA 针对**小 token 优化**的通信技术。

`README.md` L421-422：

> [nvDev](https://github.com/deepseek-ai/DeepEP/tree/nvDev)
> V2-based branch with the latest CUDA features, such as Compute Fabric Transport (CFT) that brings better latency on small token sizes.

### 12.2 CFT 的关键特性

| 特性 | 说明 |
|------|------|
| **目标场景** | 小 token（< 128 tokens）的通信优化 |
| **优化方向** | 降低 latency（而非提高 throughput） |
| **实现方式** | CUDA 新特性，可能涉及 GPU 硬件加速 |
| **分支** | `nvDev` 分支 |

### 12.3 CFT 与 DeepEP 的集成

**当前状态**：
- CFT 在 `nvDev` 分支中实验性支持
- 基于 V2 架构
- 目标是进一步优化小 token 场景的 latency

### 12.4 对 UB 的启发

**CFT 的设计理念**：
1. **小 token 优化**：传统 RDMA 对大消息优化，小消息 latency 高
2. **硬件加速**：利用 GPU 硬件特性降低通信延迟
3. **协议优化**：可能涉及轻量级协议，减少 RTT

**UB 设计建议**：
- 小 token 场景（decode）需要专门的通信优化
- 考虑硬件加速的通信原语
- 减少协议开销（header、doorbell 等）

---

## 13. 其他规划特性：Zero Copy、Eager、TMA

### 13.1 Torch Zero Copy

`README.md` L403-406：

> [Zero-copy](https://github.com/deepseek-ai/DeepEP/pull/453)
> Removing the copy between PyTorch tensors and communication buffers, which reduces the SM usages significantly for normal kernels
> This PR is authored by **Tencent Network Platform Department**

**核心思想**：
- 消除 PyTorch tensor 与 communication buffer 之间的拷贝
- 直接在高层次 tensor 上操作
- 显著减少 Normal kernel 的 SM 使用

**类似技术**：
- 2012 Network Lab 的 Zero Copy 技术
- GPU Direct RDMA（GDR）的扩展

**对 UB 的启发**：
- 零拷贝是减少 SM 开销的关键
- 需要 buffer 管理与 PyTorch tensor 生命周期对齐

### 13.2 Eager Protocol

`README.md` L407-408：

> [Eager](https://github.com/deepseek-ai/DeepEP/pull/437)
> Using a low-latency protocol removes the extra RTT latency introduced by RDMA atomic OPs

**核心思想**：
- 使用低延迟协议
- 消除 RDMA atomic 操作引入的额外 RTT 延迟
- 预期收益：降低 latency（特别是小消息场景）

**RDMA Atomic 的 RTT 问题**：
- RDMA atomic（如 compare-and-swap）需要 2 RTT
- Eager 协议可能使用 RDMA write + notification，只需 1 RTT

### 13.3 TMA (Tensor Memory Accelerator)

`README.md` L409-413：

> [Hybrid-EP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
> A new backend implementation using TMA instructions for minimal SM usage and larger NVLink domain support
> Fine-grained communication-computation overlap for single-batch scenarios
> PCIe kernel support for non-NVLink environments
> NVFP4 data type support

**TMA 的关键特性**：

| 特性 | 说明 |
|------|------|
| **数据面 vs 控制面** | TMA 在**数据面**，异步搬运数据，不需要 SM 参与 |
| **SM 开销** | 最小化 SM 使用（只需发起 TMA 描述符） |
| **NVLink 域** | 支持更大的 NVLink 域（跨更多 GPU） |
| **通信-计算 overlap** | 细粒度的单 batch 场景 overlap |
| **PCIe 支持** | 非 NVLink 环境下的 PCIe kernel |
| **FP4 支持** | NVFP4 数据类型 |

**TMA 在数据面还是控制面？**

**回答**：TMA 在**数据面**。

```cpp
// dispatch.cuh:289-291
ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                 mbarrier_ptr, kNumHiddenBytes);
```

**TMA 的工作方式**：
1. SM 发起 TMA 描述符（控制面，开销极小）
2. TMA engine 异步搬运数据（数据面，不占 SM）
3. mbarrier 通知完成

**对 UB 的启发**：
- 数据面应尽量使用硬件加速（TMA、DMA engine）
- 控制面只需发起请求，不参与数据搬运
- TMA 描述符可以预计算，进一步减少 SM 开销

---

## 14. AntGroup 优化规划解读

### 14.1 Normal-SMFree

`README.md` L416：

> [Normal-SMFree](https://github.com/deepseek-ai/DeepEP/pull/347) Eliminating SM from RDMA path by decoupling comm-kernel execution from NIC token transfer, freeing SMs for compute

**核心思想**：
- 将通信 kernel 执行与 NIC token 传输解耦
- 消除 RDMA 路径上的 SM 使用
- 释放 SM 给计算任务

**与 V1 的 0 SM Low-Latency 的关系**：
- V1 只有 Low-Latency 模式支持 0 SM
- AntGroup 将 0 SM 扩展到 Normal 模式

### 14.2 LL-SBO

`README.md` L417：

> [LL-SBO](https://github.com/deepseek-ai/DeepEP/pull/483) Overlapping Down GEMM computation with Combine Send communication via signaling mechanism to reduce end-to-end latency

**核心思想**：
- 将 Down GEMM 计算与 Combine Send 通信 overlap
- 通过 signaling 机制同步
- 降低端到端延迟

**SBO = Signaling-Based Overlap**

### 14.3 LL-Layered

`README.md` L418：

> [LL-Layered](https://github.com/deepseek-ai/DeepEP/pull/500) Optimizing cross-node LL operator communication using rail-optimized forwarding and data merging to reduce latency

**核心思想**：
- 优化跨节点 LL（Low-Latency）算子通信
- Rail-optimized forwarding：利用多 rail（多网卡）优化转发
- Data merging：合并小消息，减少通信次数

### 14.4 总结

| PR | 目标 | 关键技术 |
|----|------|----------|
| **Normal-SMFree** | Normal 模式 0 SM | 解耦 comm-kernel 与 NIC transfer |
| **LL-SBO** | LL 模式计算-通信 overlap | Signaling-based overlap |
| **LL-Layered** | LL 模式跨节点优化 | Rail-optimized forwarding + data merging |

---

## 15. 通信方式量化要素抽象

### 15.1 通信方式的分类

基于 DeepEP V2 的分析，可以抽象出以下通信方式：

| 通信方式 | 数据方向 | SM 使用 | 数据类型 | 优化目标 |
|----------|----------|---------|----------|----------|
| **EP Dispatch (Normal)** | All-to-All | 多（64-160） | FP8/BF16 | Throughput |
| **EP Dispatch (Low-Latency)** | All-to-All | 少（4-8） | FP8/BF16 | Latency |
| **EP Combine** | All-to-All | 多（64-160） | BF16 | Throughput |
| **Engram Fetch** | One-sided Read | 0 | BF16/FP8 | Latency |
| **PP Send/Recv** | Point-to-Point | 0 | BF16 | Latency |
| **AGRS** | All-Gather/Reduce-Scatter | 多 | BF16 | Throughput |

### 15.2 量化要素模型

#### 15.2.1 延迟模型

```
Latency = max(Network_Time, HBM_Time) + Control_Overhead

Network_Time = Data_Volume / Network_Bandwidth
HBM_Time = HBM_Read_Write_Volume / (Num_SMs × Per_SM_HBM_Bandwidth)
Control_Overhead = QP_Doorbell + Barrier + CPU_Sync
```

#### 15.2.2 吞吐量模型

```
Throughput = Num_SMs × Per_SM_Efficiency × Network_Utilization

Per_SM_Efficiency = f(TMA_Efficiency, Warp_Specialize_Overlap)
Network_Utilization = f(VL_Isolation, AR_Enable, Congestion)
```

#### 15.2.3 SM 数量模型

```
Num_SMs = max(
    (Network_Bandwidth / Per_Token_Network_Traffic) × (Per_Token_HBM_Read / Per_SM_HBM_Read_BW),
    (Network_Bandwidth / Per_Token_Network_Traffic) × (Per_Token_HBM_Write / Per_SM_HBM_Write_BW)
)
× 1.25 (margin)
```

### 15.3 关键量化参数

| 参数 | 符号 | 典型值 | 影响 |
|------|------|--------|------|
| **num_experts** | E | 256 | 决定 buffer 寻址粒度 |
| **num_topk** | K | 8 | 决定 token 复制量 |
| **num_ranks** | R | 256 | 决定通信范围 |
| **hidden** | H | 7168 | 决定 token 大小 |
| **num_max_tokens_per_rank** | T | 4096/128 | 决定 buffer 大小 |
| **rdma_gbs** | B_rdma | 50 GB/s | RDMA 带宽 |
| **nvlink_gbs** | B_nvl | 160/726 GB/s | NVLink 带宽 |
| **sm_read_gbs** | B_sm_read | 200 GB/s | 每 SM HBM 读带宽 |
| **sm_write_gbs** | B_sm_write | 50 GB/s | 每 SM HBM 写带宽 |

### 15.4 时延数据与代码分析

#### 15.4.1 Dispatch 时延分析

```
Dispatch_Time = Notify_Time + Dispatch_Time + Barrier_Time + Copy_Epilogue_Time

Notify_Time = O(num_tokens × num_topk / (num_notify_warps × 32))
Dispatch_Time = O(num_tokens / (num_dispatch_warps × num_sms))
Barrier_Time = O(num_ranks × qp_latency)
Copy_Epilogue_Time = O(num_recv_tokens / (num_sms × warp_efficiency))
```

#### 15.4.2 Combine 时延分析

```
Combine_Time = Push_Time + Reduce_Epilogue_Time

Push_Time = O(num_combined_tokens / (num_warps × num_sms))
Reduce_Epilogue_Time = O(num_combined_tokens / (num_sms × warp_efficiency))
```

#### 15.4.3 Cached Dispatch 时延分析

```
Cached_Dispatch_Time = Dispatch_Time (skip Notify_Time, skip CPU_Sync)

Speedup = (Notify_Time + CPU_Sync) / Total_Time
```

对于 Decode 场景（小 batch），Notify_Time 和 CPU_Sync 占比大，Cached Dispatch 收益显著。

---

## 16. UB EP Offering 建议

### 16.1 统一 All Together 方案

基于 DeepEP V2 的分析，建议 UB 提供以下 EP offering：

#### 16.1.1 核心能力

| 能力 | 描述 | 对标 DeepEP |
|------|------|-------------|
| **EP Dispatch/Combine** | All-to-All 通信 | DeepEP V2 ElasticBuffer |
| **Engram** | 远端内存访问 | DeepEP V2 Engram |
| **PP Send/Recv** | Pipeline Parallel | DeepEP V2 PP |
| **AGRS** | All-Gather/Reduce-Scatter | DeepEP V2 AGRS |

#### 16.1.2 差异化能力

| 能力 | 描述 | 对 POC 的价值 |
|------|------|---------------|
| **融合超节点总线/1825** | 支持 UB 超节点拓扑 | 客户 POC 关键需求 |
| **池化 DDR (NGRAM)** | 支持池化内存 | 与 Engram 类似，支持远程 KV cache |
| **SM-Free 通信** | 0 SM 通信 | AntGroup Normal-SMFree |
| **Eager Protocol** | 低延迟协议 | 消除 RTT |
| **CFT 支持** | 小 token 优化 | Decode 场景 |

### 16.2 对主要客户的支持

| 客户 | 需求 | UB 对策 |
|------|------|---------|
| **Tencent** | Zero Copy 技术 | 集成 Tencent Zero Copy PR |
| **AntGroup** | SM-Free、SBO、Layered | 集成 AntGroup 优化 PR |
| **ROCm/Mori-EP** | AMD GPU 支持 | 支持 Mori 后端 |

### 16.3 建议路线图

1. **Phase 1**：实现 EP Dispatch/Combine（对标 DeepEP V2）
2. **Phase 2**：实现 Engram + PP + AGRS
3. **Phase 3**：集成 Zero Copy、Eager、CFT
4. **Phase 4**：支持超节点拓扑和池化 DDR
5. **Phase 5**：SM-Free 通信和细粒度 overlap

---

## 总结

### Q1: ElasticBuffer 统一 Normal/Low-Latency 的量化因素

| 因子 | 公式 | 含义 |
|------|------|------|
| `num_expected_topk` | `num_ranks × (1 - C(E-E/R, k)/C(E, k))` | 每 token 期望去重 rank 数 |
| `sm_read` | `1 / num_expected_topk` | 每 token HBM 读量 |
| `sm_write` | `f(num_expected_topk, 卡数)` | 每 token HBM 写量 |
| `rdma_traffic` | `g(卡数, num_expected_topk)` | 每 token RDMA 流量 |
| `nvlink_traffic` | `h(卡数, num_expected_topk)` | 每 token NVLink 流量 |
| **瓶颈** | `max(rdma_traffic/rdma_gbs, nvlink_traffic/nvlink_gbs)` | 时间瓶颈 |

### Q2: SM 估算本质

```
num_sms = bottleneck_ratio × max(sm_read_per_token/sm_read_bw, sm_write_per_token/sm_write_bw)
```

**为什么用 experts/topk 而非卡数**：
- Buffer 寻址是 expert 粒度的
- Token 复制量取决于 expert 分布（topk 选择）
- 卡数通过 `num_expected_topk` 间接体现

### Q3: TopK 复制机制

- **是的**，`dispatch_forward`（即 `dispatch_impl`）包含 TopK buffer 生成
- **是的**，token 复制发生在 `buffer.dispatch` 内
- **是的**，利用 `topk_idx` 进行：
  1. 目标 rank 计算：`rank = expert_idx / experts_per_rank`
  2. Slot 分配：`atomicAdd(counter[rank], 1)`
  3. 数据路由：NVLink TMA store 或 RDMA put

### Q4: Decode 路由缓存

- **是的**，100% cache 了（routing decisions 不变时）
- **只有 Decode 支持**：因为 Decode 的 routing decisions 在连续 step 之间可能不变
- **Prefill/Train 不支持**：因为每个 batch 的 routing decisions 都不同
- **V2 的统一优化**：API 层统一 + 通信算法层统一

### Q5: 0 SM 先验知识

- **数据面与控制面分离**：NIC DMA 搬数据，SM 做控制
- **V2 不再支持 0 SM EP**：但保留 0 SM Engram/PP/CP
- **AntGroup 扩展**：Normal-SMFree 将 0 SM 扩展到 Normal 模式

### Q6: Engram 复用 ElasticBuffer

- **是的**，完全复用 ElasticBuffer API
- **共享底层资源**：Buffer、NCCL Gin Context、QP、Comm Stream
- **区别**：Engram 使用 CPU Buffer，EP 使用 GPU Buffer

### Q7: FP8 Dispatch vs BF16 Combine

- **FP8 Dispatch**：节省 50% 通信带宽，每 token 附带 SF
- **BF16 Combine**：保证累加精度
- **Buffer Size**：FP8 + SF < BF16

### Q8: async_with_compute_stream

- **不是放弃指令级融合**，而是分层设计
- **Stream 级别 Overlap**：通信与计算分离到不同 stream
- **Kernel 级别 Overlap**：Notify + Dispatch warps 并行
- **硬件级别 Overlap**：TMA engine 独立于 SM

### Q9: NV Gin 架构

- **GPU-Initiated Networking**：SM 直接操作 NIC QP
- **Symmetric Memory**：统一虚拟地址空间
- **QP 映射**：根据 SM 数和模式动态调整
- **性能**：V2 1.3x 峰值带宽，4x SM 节省

### Q10: 流量隔离

- **优先 VL 隔离**：EP 流量使用独立 VL
- **开启 Adaptive Routing**：多路径负载均衡
- **软件协同**：`sl_idx` 参数 / `EP_OVERRIDE_RDMA_SL` 环境变量

### Q11: VL vs 拥塞控制

- **禁用拥塞控制**：因为会降低最大带宽
- **优先 VL 隔离**：将拥塞流量分配到低优先级 VL
- **UB 启示**：需要支持多 VL/VC，测试 VL 隔离性能

### Q12: CFT

- **Compute Fabric Transport**：针对小 token 优化
- **nvDev 分支**：实验性支持
- **目标**：降低小 token 场景的 latency

### Q13: 其他特性

- **Zero Copy**：消除 PyTorch tensor 与 buffer 之间的拷贝
- **Eager Protocol**：消除 RDMA atomic 的 RTT 延迟
- **TMA**：数据面异步搬运，最小化 SM 使用

### Q14: AntGroup 优化

- **Normal-SMFree**：Normal 模式 0 SM
- **LL-SBO**：Down GEMM 与 Combine Send overlap
- **LL-Layered**：跨节点 rail-optimized forwarding

---

> **分析方法论**：本报告基于源码实证，关键结论均有代码行号标注。核心洞察是：V2 通过将 Buffer 提升为一等公民、SM/QP 解析式计算、以及统一的 Kernel Pipeline，实现了 Normal 和 Low-Latency 的统一——差异仅在于参数配置，而非架构设计。这一设计哲学对 UB EP offering 有重要的启发意义。
