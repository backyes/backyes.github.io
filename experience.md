# 建站经验沉淀

> backyes.github.io — LLM Infrastructure Insights (Research Hub)
> 最后更新: 2026-07-20

## 一、技术栈选型

| 层 | 选择 | 原因 |
|---|---|---|
| 托管 | GitHub Pages | 免费、与 repo 原生集成、自动 HTTPS |
| 仓库名 | `backyes.github.io` | 用户站点强制命名规则(仓库名 = 域名) |
| 发布源 | main 分支根目录 | 用户站点默认配置 |
| 构建 | 纯静态 HTML + Python 脚本 | 无 Jekyll / 无 Actions 依赖 |
| 同步 | rsync + git push | 增量同步、幂等 |
| 域名 | 自定义域名可选 | CNAME 文件 + DNS |

## 二、项目架构

```
workspace/
├── sync_reports.sh          # rsync 同步 + git 提交的入口
├── build_site.py             # 全页构建 (cards + posts + tags + search)
├── posts/                    # 手写文章源 (*.md + frontmatter)
├── vllm_research/            # 研究报告工作区
│   ├── vllm/                 # 原始源码(不发布)
│   ├── vllm-ascend/          # 原始源码(不发布)
│   └── vllm_analysis/        # 分析报告(自包含发布单元)
│       ├── index.html
│       ├── vllm_architecture_unified.html
│       ├── _archived/        # 历史迭代
│       ├── _raw/             # 原始备份
│       └── _assets/          # 图表资源
├── backyes.github.io/        # Pages 仓库克隆 (发布目标)
│   ├── index.html            # 首页 (Hero + Survey + Posts + Tags)
│   ├── posts.html            # 手写博客列表
│   ├── tags.html             # 两栏标签页 (sidebar + main)
│   ├── search-overlay        # 客户端搜索弹窗
│   ├── assets/css/main.css   # 共享设计系统
│   └── [报告目录]/           # 每份报告一个子目录
└── experience.md             # 本文件 (经验沉淀)
```

## 三、关键设计决策

### 3.1 数据驱动
- `build_site.py` 的 `REPORTS` 数组是唯一真相源
- 每份报告: `{dst, entry, visual, title, desc, cat, priority, tags}`
- `sync_reports.sh` 的 `PROJECTS` 数组负责 rsync 源路径
- 两者保持同步

### 3.2 增量同步策略
```bash
rsync -a --delete --delete-excluded \
  --include="*/" --include="*.html" --include="*.css" \
  --include="*.png" --include="*.jpg" --exclude="*" \
  "$SRC/" "$REPO/$DST/"
```
- 白名单: html + css + png/jpg/svg/webp/ico
- 排除原始素材: md / txt / pdf / py / js / json / log / sh
- 排除目录: raw/ / raw_material/ / _raw/ / logs/ / cache/

### 3.3 rsync 排除目录不删目标的修复
仅 `--exclude=raw` 不够 — 被 exclude 的路径 rsync 既不复也不删。
需要加 `--delete-excluded` 让目标端被排除的目录被主动清掉。

### 3.4 卡片视觉
- 早期: emoji (📘 🔀) → 不具备辨识度
- 中期: SVG + 固定 px 字号 → 不自适应
- 当前: HTML div + `clamp(2rem,4vw,3rem)` → 匹配首页 hero 字体风格,响应式

### 3.5 配色哲学 (Lil'Log-inspired)
- 整体: 黑白灰 + 单蓝色链接
- 去渐变、去多彩色
- `--bg:#111317` `--fg:#d7dbe0` `--accent:#4a8fe0`
- 标签统一灰色 `color:var(--muted)`,仅一个 active 时用蓝色

### 3.6 页面结构
- 首页: Hero + Survey by AI (卡片网格) + Posts 预览 + Tags 云
- Posts 页: Lil'Log 风格列表 (日期左 + 标签右)
- Tags 页: 两栏 (sidebar 标签列表 + main 按标签分组)
- Search: 客户端全文搜索 (⌘K 打开,Esc 关闭)

### 3.7 Search DB 同步
- `build_site.py` 生成 `const SEARCH_DB=[...]` JSON 数组
- 注入到 index.html 的 `<script>` 中
- 匹配 title + description + tags
- 搜索结果可点击跳转

## 四、踩过的坑

### 4.1 GitHub Pages 开关
- 用户站点 main 根目录自动发布,无需手动开 Pages
- `gh api PUT /repos/.../pages` 会 403 (fine-grained token 缺 pages scope)
- **解决**: 推送到 main 后 Pages 自动激活,等 1-3 分钟构建

### 4.2 fine-grained PAT 权限
- `github_pat_` 前缀 = fine-grained token
- 缺 Contents 写权限 → push 403
- 缺 Pages 权限 → `gh repo create` 失败
- **解决**: 编辑 token → Contents = Read and Write

### 4.3 GitHub 网络需要代理
- ping 通 github.com 但 HTTPS 443 被拦截
- **解决**: `git config http.proxy http://127.0.0.1:7897`

### 4.4 GitHub Actions 部分中断
- Pages build 依赖 Actions 基础设施
- Actions 部分中断时 build 作业调度不上 (duration=0 持续几分钟)
- **解决**: 等恢复,通常几十分钟内

### 4.5 rsync filter 顺序
- 排除目录规则**必须**放在 `--include=*/` 之前
- 否则 `--include=*/` 先匹配并进入 raw/ 目录,后面 exclude 失效

### 4.6 SEARCH_DB 双重括号
- template 有 `[]` + JSON.stringify 又加 `[]` → `[[...]]`
- **解决**: regex 改为 `const SEARCH_DB=(?:\[.*?\]|<!--SEARCH_DB-->);`

### 4.7 仓库改名
- 用户站点仓库**必须**等于 `<username>.github.io`
- 只改仓库名不能改站点域名
- 真正迁移需要新建 GitHub 账号

## 五、手写文章约定

`posts/YYYY-MM-DD-title.md`:
```markdown
---
title: "标题"
date: 2026-07-22
tags: [tag1, tag2]
excerpt: "摘要..."
---

# 正文 (标准 markdown)
```

`build_site.py` 自动解析 frontmatter → 生成 posts.html / 单篇 HTML / search DB。

## 六、维护命令

```bash
cd ~/work/claude_workspace
./sync_reports.sh              # 全量同步 + 提交 + 推送
./sync_reports.sh --dry-run    # 预览
./sync_reports.sh --no-push    # 本地试
./sync_reports.sh pd-routing   # 只同步某项目
```
