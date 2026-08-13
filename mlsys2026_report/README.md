# MLSys 2026 深度综合报告 · 自包含包

> 📅 会议：第 9 届 Conference on Machine Learning and Systems · Bellevue, WA · 2026/05/18–22
> 📋 覆盖：5 场 Keynote + 19 篇论文（7 大主题）
> 🗓️ 报告整理：2026-07-15 · 自包含打包：2026-08-05

---

## 🚀 快速开始

直接用浏览器打开以下任一 HTML 文件即可阅读完整报告，**无需联网**（所有资源已内嵌）。

| 版本 | 文件 | 风格 |
|------|------|------|
| **Notion 风格**（推荐） | `index_notion.html` | 白色背景、可折叠块、卡片网格 |
| 暗色风格 | `index.html` | GitHub 暗色主题 |

```bash
# macOS / Linux
open index_notion.html        # macOS
xdg-open index_notion.html    # Linux

# 或启动本地服务器（推荐，避免某些浏览器的 file:// 限制）
python3 -m http.server 8080
# 然后访问 http://localhost:8080/index_notion.html
```

---

## 📂 目录结构

```
mlsys2026_report/
├── README.md                              ← 本文件（包说明）
├── index_notion.html                      ← 📊 Notion 风格主报告（推荐）
├── index.html                             ← 📊 暗色风格主报告
├── sources.html                           ← 📎 溯源链接附件
├── keynote/                               ← 🎤 5 场 Keynote 深度笔记
│   ├── 01_saroufim_ai_writes_code.md
│   ├── 02_lidong_zhou_system_intelligence.md
│   ├── 03_vahdat_google_ai_infra.md       ← ⚠ 官方未发 abstract，含推测标记
│   ├── 04_zettlemoyer_rethinking_pretraining.md
│   ├── 05_kozyrakis_inference_efficiency.md
│   ├── SUMMARY.md                         ← 5 场综合
│   ├── sources.md
│   └── social_media_queries.md
└── papers/                                ← 📄 19 篇论文 PDF + 深度报告
    ├── README.md                          ← 19 篇详细索引
    ├── ML_Compilers_Kernels/              # 4 篇 (Dataflow / FA-4 / HipKittens / ParallelKittens)
    ├── LLM_Training_Fine-tuning/          # 2 篇 (AXLearn / veScale-FSDP)
    ├── LLM_Inference_Serving/             # 5 篇 (TokenWeave / SAKURAONE / Beyond the Buzz / MoE Tax / SpecDec Illusion)
    ├── Hardware_Accelerators/             # 3 篇 (NoC / SHIP / SuperInfer)
    ├── Distributed_Federated_ML/          # 1 篇 (fabric-lib)
    ├── Edge_Mobile_Embedded/              # 1 篇 (ExecuTorch)
    └── Data_Storage_Retrieval/            # 3 篇 (LEANN / GriNNder / SkipKV)
```

---

## 📊 报告亮点

### 六大信号（执行摘要）
1. **栈级协同是唯一出路** — 单层优化边际收益触底
2. **"同步税"成为第一性瓶颈** — GPU 体系的结构性天花板
3. **存储层级在三个方向上被重写** — 从固定硬件到编排对象
4. **通信范式从 Collective 到 P2P** — fabric-lib 是奠基工作
5. **"批判性转向"** — 三篇"反主流"论文同场，MLSys 走向工程现实主义
6. **训练框架分叉** — JAX 模块化 vs PyTorch 灵活性

### 跨论文六大战略主轴
- 主轴 1：「同步税」—— GPU 体系的结构性天花板
- 主轴 2：存储层级的重定义
- 主轴 3：通信范式转移 Collective → P2P
- 主轴 4：Superchip 对系统设计的范式冲击
- 主轴 5：批判性转向——工程现实主义
- 主轴 6：训练框架路线分叉

---

## 📖 推荐阅读路径

| 读者 | 路径 |
|------|------|
| 算法研究员 | SkipKV → MoE Serving Tax → Dataflow → SpecDec Illusion → Zettlemoyer Keynote |
| 系统/ Infra 工程师 | Beyond the Buzz → fabric-lib → SuperInfer → SAKURAONE → TokenWeave → Kozyrakis Keynote |
| 芯片/体系结构 | NoC → SHIP → Dataflow → FlashAttention-4 → HipKittens → ParallelKittens |
| 训练框架 | AXLearn → veScale-FSDP → Saroufim Keynote → Zettlemoyer Keynote |
| 管理者/决策者 | 执行摘要 → 六大主轴 → 批判性转向 → 对中国的 6 条启示 |

---

## 📐 统计

| 项目 | 数值 |
|------|------|
| Keynote 笔记 | 5 场 + 1 份综合 |
| 论文数 | 19 篇 |
| 中文深度报告 | ~9400 行 |
| PDF 总大小 | ~68 MB |
| 包总大小 | ~124 MB |
| 链接状态 | ✅ 全部内部链接有效 |

---

## ⚠️ 已知说明

1. **Vahdat (Google) Keynote**：官方未发布 abstract，笔记含推测内容，已用 ⚠ 标记
2. **GitHub 链接**：本环境中无法验证（网络限制），但均为公开仓库 URL
3. **MLSys 官方录像**：2026-06-22 起在 mlsys.org/virtual/2026/ 开放，链接已内嵌

---

## 🔄 版本历史

- **2026-07-15**：初版综合报告（基于 ~9400 行逐篇深度解读）
- **2026-08-05**：自包含打包版本
  - 所有资源复制到 `keynote/` 和 `papers/`
  - 链接改为相对路径，支持离线阅读
  - 修复外部死链接（如有）
  - 添加 README.md 与包说明

---

*报告生成：基于本地已有逐篇深度解读的二次综合 · 移动端友好 HTML · 关键结论均可溯源*
