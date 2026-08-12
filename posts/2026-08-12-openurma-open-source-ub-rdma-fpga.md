---
title: "OpenURMA：UB 类 RDMA 传输的开源 FPGA 实现"
date: 2026-08-12
tags: ["RDMA", "FPGA", "OpenURMA", "UB", "SmartNIC", "Interconnect", "Open-Source", "Load-Store"]
excerpt: "OpenURMA 是 UB（Ultra-Bus）连接式 RDMA 传输的开源 FPGA 实现，在 Alveo U50 上实现 500ns 级远程 load/store 延迟（对比 RoCE 的 2236ns，快 4.47×），同时完整实现了 UB-Base-Specification 2.0.1 的事务层与传输层。"
---

# OpenURMA：UB 类 RDMA 传输的开源 FPGA 实现

## 这是什么

[OpenURMA](https://github.com/bojieli/OpenURMA) 是 [UB（Ultra-Bus）](https://www.ub.org/) 连接式 RDMA 传输协议的**开源 FPGA 实现**，构建在 [OpenClickNP](https://github.com/OpenClickNP) 之上。它完整实现了 *UB-Base-Specification 2.0.1* 定义的：

- **事务层**（Transaction Layer）：BTAH/ATAH 头部、18 个事务 opcode、四种服务模式（ROI/ROT/ROL/UNO）、三种执行序标签（NO/RO/SO）、Fence、两种完成序模式
- **传输层**（Transport Layer）：RTP（PSN/GoBackN 重传）、UNO 模式下的 UTP、简化 CETPH echo

上层 `libopenurma` 暴露 *UB-Software-Reference-Design-for-OS-2.0* §5.3 定义的 URMA verb 接口。

## 三个架构支柱

OpenURMA 的论文（[arXiv:2605.28717](https://arxiv.org/abs/2605.28717)）用一张图概括了 UB 协议栈的设计哲学——三个相互支撑的支柱：

1. **Transport / Transaction 分离**：状态复杂度为 O(本地 Jetties) + O(远程 endpoints)，而非两者的乘积。这使得控制器可以放在片上总线上（而非 PCIe 后面）。

2. **Native load/store 延迟**：NIC 的工作集能放进片上 SRAM，控制器与 CPU 同在片上总线上，CPU load/store 直接到达远程内存——把 RDMA READ 的四次 PCIe 遍历压缩为**一次片上总线穿越**。这是核心结果：==64 字节远程 fetch 端到端 ≈500ns==，对比同等条件下 RoCEv2 的 ==2236ns==，**快 4.47×**。

3. **分级定序（Graded Ordering）**：完整实现 §7.3 的四种服务模式 × 三种执行标签 × Fence × 两种完成模式，应用按需选择一致性级别——不请求 gating 的操作零开销。

## 不只是 RTL

OpenURMA 的一个亮点是**完整软件栈验证**：

- 同一套 `.clnp` 设计既编译为 Alveo U50 硬件 RTL，也编译为 cycle-accurate SystemC NIC
- **未经修改的官方 openEuler UMDK 栈**（`liburma → uburma.ko → ubcore.ko → openurma_ubcore.ko`）在 gem5 全系统 Linux  guest 中直接驱动它
- 真实应用跑在上面：官方 `urma_perftest`、URPC `umq` RPC 框架、KV store（最大 60KB values）、分布式原子计数器、多客户端并发、§7.3 定序负载
- 三种传输模式（RM / RC / UM）均验证通过

## 复现路径

```bash
git clone https://github.com/bojieli/OpenURMA
./reproduce.sh doctor   # 检查工具链
./reproduce.sh smoke    # 构建 + 17 项测试 + 验证 headline 数字（~2 分钟）
./reproduce.sh paper    # 完整数据集 + 所有图表 + 重建 PDF（~15 分钟）
```

## 为什么值得关注

在 AI 超节点互联的语境下，OpenURMA 提供了一个有意思的对照：

- **RoCE/RDMA** 走的是"消息传递"路径（Send/Write/Read），需要多次 PCIe 遍历
- **UB load/store** 走的是"内存语义"路径（load/store 直达远程），延迟低一个数量级
- OpenURMA 证明了这种差异**不是理论推演**，而是可以在真实 FPGA 上跑通、用真实软件栈验证的

对于研究近存通信、对称内存、GPU-centric fabric 的同学，这是一个值得细看的开源参考实现。

---

**链接：**
- 论文：[arXiv:2605.28717](https://arxiv.org/abs/2605.28717)
- 代码：[github.com/bojieli/OpenURMA](https://github.com/bojieli/OpenURMA)
- 架构导览：[docs/architecture.md](https://github.com/bojieli/OpenURMA/blob/main/docs/architecture.md)
- 评估结果：[EVAL.md](https://github.com/bojieli/OpenURMA/blob/main/EVAL.md)
