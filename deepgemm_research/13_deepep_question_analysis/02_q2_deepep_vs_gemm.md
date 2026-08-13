# Q2: DeepEP 不是优化GEMM，而是优化MoE通信链路

## 1. 问题审视

**核心论断**：
> DeepEP负责Dispatch/Combine通信，不负责Expert GEMM计算。

## 2. 源码级验证

### 2.1 DeepEP的kernel只做通信

**文件**：`csrc/kernels/legacy/internode.cu`

搜索所有kernel函数名：

```bash
grep "__global__" csrc/kernels/legacy/*.cu
```

结果：
- `notify_dispatch` - 通知token数量元数据
- `dispatch` - 实际发送token到目标rank
- `notify_combine` - combine阶段的通知
- `combine` - 接收expert输出并还原

**没有任何GEMM计算**。所有kernel都是数据搬运。

### 2.2 与DeepGEMM的分工

**DeepGEMM的职责**（来自DeepGEMM仓库）：
- FP8 grouped GEMM
- MoE专家计算
- Tensor Core优化

**DeepEP的职责**（来自源码）：
```cpp
// csrc/kernels/legacy/internode.cu - dispatch kernel
// 只做：读取x → 写入rdma buffer → RDMA put到远端
```

### 2.3 数据流边界

**文件**：`deep_ep/__init__.py`

```python
from .buffers.legacy import Buffer      # V1: NVSHMEM
from .buffers.elastic import ElasticBuffer, EPHandle  # V2: NCCL GIN
```

DeepEP暴露的API：
- `dispatch()` - 发送token
- `combine()` - 接收expert输出

**不暴露**：
- GEMM计算
- Expert前向/后向传播

### 2.4 通信-计算边界

在MoE layer中的分工：

```
Router → topk_idx
   ↓
DeepEP.dispatch()     ← DeepEP负责
   ↓
token到达expert rank
   ↓
Expert GEMM           ← DeepGEMM/CUTLASS负责
   ↓
DeepEP.combine()      ← DeepEP负责
   ↓
token还原到原始rank
```

## 3. 结论

**论断验证**：✅ 正确

DeepEP是一个**纯通信库**，其所有CUDA kernel的功能是：
1. 读取本地token
2. 通过RDMA/NVLink发送到目标rank
3. 接收远端token
4. 将expert输出还原

Expert的矩阵乘法计算完全由外部库（DeepGEMM、CUTLASS等）负责。

