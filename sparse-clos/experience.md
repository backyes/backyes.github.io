# Sparse Clos 调研经验教训

> 完成日期：2026-07-12
> 调研类型：学术 + 工业实践
> 项目目录：sparse_clos_research/

## 本次任务总结

调查"sparse Clos"网络组网——一个散布在学术论文与工业实践但无统一术语的研究方向。覆盖了从 Clos 1953 原理论到 Google/AWS/Meta/NVIDIA 的最新工业实践（截至 2026.07）。

## 经验教训与改进

### 1. 关键词歧义早发现、早止损 ✅
- **教训**：初始将 "sparse Clos" 误判为 MoE 的 sparse expert routing（因为用户之前的兴趣方向是 MoE）。
- **好做法**：用户纠正后立即终止错误的 workflow，重新定义方向后再启动。
- **改进**：在响应前可以先用 1 个 quick search 确认关键词的主领域（网络 vs ML），避免浪费 ~2000 token 的错误 workflow。

### 2. Deep-research workflow 恢复机制不稳定 ⚠️
- **教训**：workflow 的 resume 功能在跨会话/进程边界时多次失败（"No completion record found"），journal 虽有写入但 workflow 本身未能完成验证阶段。
- **浪费**：3 次 resume 尝试产生了约 40 条 started agent 但未完成。
- **改进**：如果 workflow 首次中断，不反复 resume，而是直接读取 journal 提取已完成结果自行合成。journal 已完成 ~32 个 result agent，足够合成高质量报告。

### 3. 并行 Agent 数量需克制 ❌ → ✅
- **教训**：初始派发 5 个并行 agent 被用户中断（token 开销大）。
- **改进**：改为自己用 playwright 逐项查询——每次 navigate → evaluate 精准提取，比 agent 模式省 5-10× token。
- **原则**：对于"提取已知 URL 的特定信息"类任务，手动 playwright 远优于派 agent。agent 适合"未知探索"的开阔任务。

### 4. Playwright 效率技巧 ✅
- 用 `browser_evaluate` 提取 <1KB 关键内容，而非 `browser_snapshot`（整页 ~10-50KB token）
- 用 `browser_find` 定位"关键词"而非整页读取
- 用 Google 搜索页面快速获取搜索摘要（如 ASIC spec），再按需深入
- **教训**：arXiv HTML 版比 PDF 版更容易通过 evaluate 提取文本

### 5. 素材管理 ✅
- 32 个 agent 结果 ~1162 行 journal → 提取为 `raw_material/` 持久化
- 四个深化方向（AWS RNG、Meta DSF、芯片层、Zetta 拓扑）各自单独写 .md 持久化
- 来源清单 sources.md 按主题分类，含访问状态

### 6. 报告质量
- HTML 适配移动端成功（viewport、font、卡片式布局、responsive table）
- 每个关键结论均有 `来源：<URL>` + `原文：<英文原句>` 两种溯源
- 保留原始语言（英文引用用英文、中文引用用中文）✅
- 但报告未做"两步 prompt 法"（rule7：先写 prompt → 调研 → 改进 prompt → 再深化）—— 够用但不够体系化

### 7. 国产/非美视角缺失
- 未覆盖华为、字节、商汤等国内实践以及 Blackwell/Frontier (非 hyperscaler) 视角
- 用户特别选了"全部四个方向"而非"国产"→ 说明优先覆盖主流方向是正确的

## 改进清单
| 项 | 操作 |
|----|------|
| 关键词歧义 | 初始做一次 quick search 确认领域归属 |
| Workflow 恢复 | 失败后直接读 journal 手动合成，不反复 resume |
| Agent 使用 | 5 个并行→1 个手动 playwright，据任务复杂度决策 |
| Promopt 迭代 | 下次使用 rule7 两步 prompt 法 |
| 国产视角 | 后续补充（需 playwright 检索中文资料） |
