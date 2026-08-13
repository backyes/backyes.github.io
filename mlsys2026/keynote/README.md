# MLSys 2026 Keynote 调度 Session

**会议**：第 9 届 Conference on Machine Learning and Systems · Bellevue, WA · 2026/05/18 – 22
**调度归档时间**：2026/06/18
**目录主用途**：把 MLSys 2026 的 5 场 keynote（演讲稿/官方摘要 + 视频 + 社交讨论）汇总归档，便于自己和团队复盘。

---

## 状态总览（重要先看）

| 信息维度 | 状态 |
|----------|------|
| 5 场 keynote 标题 + 时间 + 讲者 | ✅ 已从 mlsys.org 官方页确认 |
| 4/5 keynote 官方 abstract（逐字） | ✅ 已抓取并存档（仅 Vahdat 一场官方页未发布 abstract） |
| 讲者背景小传 | ✅ 已存档 |
| 单场 keynote 视频可直链 | ⏳ 官方录像约 2026/06/22 前后免费开放，**截至 06/18 尚未上架** |
| Reddit / X.com / YouTube 讨论 | ⚠️ 本调度环境网络受限，**无法直接抓取**；已在 `social_media_queries.md` 提供查询模板，待用户在能联网终端补全 |
| 跨主题综合总结 | ✅ 见 `SUMMARY.md` |

> **诚实声明**：本会话无法访问 reddit / x.com / youtube / DDG（curl 超时、WebFetch 拒绝、内置 WebSearch 持续返回空）。所有"社交媒体讨论"小节都是占位 + 查询模板，**不含编造的引文**。

---

## 5 场 Keynote 索引

| # | 日期 | 讲者 | 单位 | 标题 | 单文件 |
|---|------|------|------|------|--------|
| 1 | 周一 13:30 | Mark Saroufim | Core Automation / GPU MODE / 前 Meta PyTorch | When AI Starts Writing Systems Code | [01_saroufim_ai_writes_code.md](./01_saroufim_ai_writes_code.md) |
| 2 | 周二 10:30 | Lidong Zhou | Microsoft / MSR Asia | The Next Horizon of Systems: From MLSys to System Intelligence | [02_lidong_zhou_system_intelligence.md](./02_lidong_zhou_system_intelligence.md) |
| 3 | 周三 10:30 | Amin Vahdat | Google SVP, AI & Infrastructure | (无公开 abstract，标题：SVP and Chief Technologist, AI & Infrastructure) | [03_vahdat_google_ai_infra.md](./03_vahdat_google_ai_infra.md) |
| 4 | 周四 10:30 | Luke Zettlemoyer | UW / Meta | Rethinking Pretraining: Data and Architecture | [04_zettlemoyer_rethinking_pretraining.md](./04_zettlemoyer_rethinking_pretraining.md) |
| 5 | 周五 09:45 | Christos Kozyrakis | NVIDIA / Stanford | The Path to Inference Efficiency | [05_kozyrakis_inference_efficiency.md](./05_kozyrakis_inference_efficiency.md) |

> 全部场次地点：Grand Ballroom 1（Bellevue Hyatt Regency / 会议主场），周二、三、四 keynote 同时开放 Grand Ballroom 2 作为 overflow。

---

## 文件清单

```
mlsys2026_keynote_session/
├── README.md                                ← 当前文件
├── 01_saroufim_ai_writes_code.md
├── 02_lidong_zhou_system_intelligence.md
├── 03_vahdat_google_ai_infra.md             ← 官方未发 abstract，含⚠推测标记
├── 04_zettlemoyer_rethinking_pretraining.md
├── 05_kozyrakis_inference_efficiency.md
├── SUMMARY.md                               ← 跨 5 场综合总结（推荐先看）
├── sources.md                               ← 全部引用 URL
└── social_media_queries.md                  ← Reddit / X / Google 查询模板
```

---

## 用法

**第一次看**：先 `SUMMARY.md` 拿 5 场全景，再按兴趣点开单场 `0X_*.md`。

**等录像上架后回填**：
1. 2026/06/22 起每周访问 https://mlsys.org/virtual/2026/invited-talk/3655（其余 ID：3665 / 3684 / 3706 / 3723），逐场补 SlidesLive 直链与 transcript。
2. 在能联网的终端上执行 `social_media_queries.md` 中的搜索串，把找到的高质量讨论 URL 与摘要补回各 `0X_*.md` 的"社交媒体讨论"段。

**需要时引用**：所有源 URL 已统一在 `sources.md`，方便 copy-paste 到笔记/PPT。

---

## 元数据

- 调度归档：Claude（GLM-5.2）/ 2026-06-18
- 数据源最新一次抓取：2026-06-18（mlsys.org keynote 页面）
- 后续如果发现错别字或新增信息，直接 edit 对应 markdown 即可，无固定结构约束。
