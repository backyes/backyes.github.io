# 项目经验教训

## 视觉设计规范

### 标题层级
- H2: `font-size:1.5rem` + 蓝色底部边框 (`border-bottom:2px solid var(--accent)`)
- H3: `font-size:1.25rem` + 蓝色左侧边框 (`border-left:3px solid var(--accent)`)
- **原则:** 标题通过颜色/边框标识,不要只靠字号区分

### 关键高亮
- `==text==` → 蓝色渐变下划线标记 (`background:linear-gradient(180deg,transparent 60%,var(--accent-soft) 60%)`)
- `**strong**` → 加粗 + 白色 (`color:var(--fg)`)
- `\$number\$` → 蓝色大字号数字 (`class="key-num"`)
- **注意:** `==` 内不要包含 `$` 符号,否则会被解析为 key-num

### 参考风格
- 参考 [The New Stack](https://thenewstack.io/google-frozen-gemini-chip/) 的标题渲染
- 标题要有明确的视觉标识(边框/颜色),不要只靠字号

### 表格样式
- `border-collapse:collapse`,表头灰色背景
- 斑马纹 `tr:nth-child(even)` 微差背景
- `tr:hover` 高亮
- 表格可横向滚动 (`overflow-x:auto;display:block`)

### 配色
- 主蓝: `--accent:#4a8fe0`
- 浅蓝背景: `--accent-soft:rgba(74,143,224,.12)`
- 文本白: `--fg:#d7dbe0`
- 注释灰: `--muted:#7d848d`

### 移动端适配
- `@media(max-width:768px)` 必须加
- 手机标题自动缩小 (`h2:1.3rem, h3:1.1rem`)
- 表格字号缩小 (`font-size:.8rem`),内边距减小
- 容器内边距 `padding:24px 16px`(桌面 `40px 24px`)

---

## 建站经验

### 技术选型
- GitHub Pages + 纯静态 HTML,无 Jekyll/Actions 依赖
- 仓库名 `backyes.github.io` 强制等于账号名(用户站点硬约束)
- rsync 白名单同步: html+css+png/jpg/svg/webp/ico,排除 md/txt/pdf/log/py 等原始素材

### 关键踩坑
1. **fine-grained PAT 缺 Contents 写权限** → push 403,需补权限
2. **GitHub 网络需代理** → `git config http.proxy http://127.0.0.1:7897`
3. **rsync filter 顺序** → exclude 目录规则必须在 `--include=*/` 之前
4. **`--delete-excluded` 必须加** → 仅 `--exclude` 不够,被排除的路径 rsync 既不复也不删
5. **SEARCH_DB 双重括号** → template 有 `[]` + JSON.stringify 又加 `[]`,regex 需兼容两种形式
6. **Pages build 依赖 Actions** → Actions 部分中断时 build 调度不上,等恢复
7. **仓库改名** → 用户站点仓库名必须等于账号名,只改仓库名不能改站点域名
8. **CSS 在 f-string 中** → 单 `{` `}` 必须写成 `{{` `}}`,否则 Python 报 SyntaxError
9. **双 `</style>` 标签** → CSS 内联在 f-string 中只能有一个 `<style>...</style>` 对,多余 CSS 放到 `</style>` 后会被当文字渲染

### 图片/资源路径
- 博客文章在 `posts/xxx.html`,引用资源用相对路径 `assets/xxx.png`(不是 `assets/images/`)
- 图片实际存放: `posts/assets/` 目录
- rsync 白名单包含 `*.png`,会自动同步
- **图表加水印:** `fig.text(0.99, 0.01, 'backyes.github.io', fontsize=8, color='#8b949e', ha='right', va='bottom', alpha=0.6, style='italic')`

### Markdown 渲染
- `md_to_html` 需独立处理 `![alt](url)` 图片语法,不能当成链接
- 图片行单独成块 `<img>`,不包裹在 `<p>` 中
- 表格、引用、代码块、加粗、斜体、链接都要支持
- `==text==` 高亮语法必须独立行或行内正确解析
- **行内图片正则** 优先于链接正则,否则 `![alt](url)` 会被当成链接

### 导航响应式
- JS 动态测量: 先全部显示、隐藏 `⋯`,放不下才折叠
- 桌面端空间够时全展示,不够才折叠
- 不能用 `nth-child` 硬编码隐藏

---

## 写文章经验

### 数据驱动型写作
- **先定结论** → 一句话锚点,所有数据为它服务
- **骨架优先** → 读者 30 秒扫完全文(标题+表格+图表)
- **能用表格不用文字** → 让读者自己算,不替读者做判断
- **控制字数** → 删比写更重要,去掉不影响结论的段落
- **主观判断必须有数据兜底** → 无数据支撑的结论不说
- **迭代修改** → v0.1 数据堆砌 → v2.0 精简核心

### 数据核实
- 所有报价必须来自官方文档,标注来源链接
- 计算过程要展示,让读者可验证
- 价格/数据更新时,全文所有相关数字都要同步修改

### 可视化增强
- 图表是文章的核心证据,放在每日分析表格之前
- 图表加水印(`backyes.github.io`)防止盗用
- 关键数字用 `$number$` 蓝色高亮
- 关键结论用 `==text==` 下划线标记
- 对比表格用柱状图 + 倍数曲线(双 Y 轴)

### 配色哲学
- Lil'Log 风格: 黑白灰 + 单蓝色链接
- 去渐变、去多彩色
- 标签统一灰色,仅 active 时用蓝色

### 文章结构
- **宏观→微观:** 先场景(图表)→量化(表格)→结论→展望
- **每日分析:** 每行含计算公式,便于核查
- **对比分析:** 投影到未来规模(1B/10B),不只看当前数据

### 配色哲学
- Lil'Log 风格: 黑白灰 + 单蓝色链接
- 去渐变、去多彩色
- 标签统一灰色,仅 active 时用蓝色

### 文章结构
- **宏观→微观:** 先场景(图表)→量化(表格)→结论→展望
- **每日分析:** 每行含计算公式,便于核查
- **对比分析:** 投影到未来规模(1B/10B),不只看当前数据

---

## Survey by AI 报告首页规范

### 设计风格
- **暖纸色背景**: `linear-gradient(180deg, #fcfaf6 0%, #f6f3ec 42%, #ffffff 100%)` + 绿/金色径向光晕
- **字体**: Libre Baskerville（正文/标题 serif）+ Manrope（UI sans）+ JetBrains Mono（代码）
- **主色调**: 墨绿 `#0f5d44` / 深绿 `#0a3f30` / 绿色浅底 `rgba(15,93,68,0.10)` / 金色 `#bf8b2c` / 蓝色 `#163e7a`
- **布局**: 顶部导航栏（brand + nav）+ 左侧 TOC 侧边栏（sticky）+ 主文章区
- **卡片网格**: `grid-template-columns: repeat(auto-fill, minmax(320px, 1fr))` 用于报告卡片展示

### 首页结构
1. **Hero**: 标题 + 标签 + 日期
2. **概览**: 项目简介 + blockquote 核心观点
3. **报告卡片网格**: 每张卡片含标签、标题、描述、元信息、CTA 链接
4. **架构总览**: ASCII art 分层图
5. **核心发现**: 关键洞察列表
6. **链接汇总**: 所有报告和外部资源链接

### 参考样板
- `umdk_research/analysis/index.html` — **首页标准**（topbar + hero + stats + 报告卡片 + 章节表 + 阅读路径）
- `umdk_research/analysis/cam_v2/CAM深度分析报告_v2.html` — **子页面标准**（shared.css + article-layout + toc-sidebar）
- `vllm_research/vllm_analysis/index.html` — 首页参考（章节总览表格 + 推荐阅读路径）

### 首页结构标准（参照 UMDK）
1. **Topbar** — brand + `← backyes` / `Posts` / `Survey` / `报告` 导航
2. **Hero** — 标题 + 描述 + 统计数据卡片（章节数/页数/深度）
3. **子报告卡片网格** — 每张卡片含标题、描述、元信息、CTA 链接
4. **章节总览表格** — 从主报告提取的 h2/h3 列表（§ / 章 / 核心）
5. **推荐阅读路径** — 新手/架构师/追代码 等路径

### 子页面结构标准（参照 UMDK）
1. 使用 `<link rel="stylesheet" href="shared.css">`（不内联 style）
2. `article-layout` 使用 **flexbox** 布局（TOC 侧边栏 + 主文章区）
3. `toc-sidebar` 自动生成目录（从 h1/h2/h3 提取）
4. `article-body` 使用 Libre Baskerville serif 字体

### ⚠️ 侧边栏布局关键经验（2026-08-21）

**问题背景**：deepepv2 报告使用 `shared.css`（`display: grid`），长页面点击 TOC 锚点时侧边栏异常。

**根因分析**：
- `shared.css` 的 `.article-layout` 使用 `display: grid; grid-template-columns: 220px 1fr`
- 正常 post 的 `main.css` 使用 `display: flex` 模型
- `position: sticky` 在 grid 布局中对长页面锚点跳转表现不稳定

**正确做法**（覆盖 shared.css 默认值）：
```css
/* 在报告的 <style> 中覆盖 */
.article-layout {
    display: flex;           /* 不是 grid */
    gap: 32px;
    max-width: 1160px;
    margin: 0 auto;
    padding: 0 32px;
}
.toc-sidebar {
    flex: 0 0 220px;         /* 固定宽度，不伸缩 */
    position: sticky;
    top: 24px;
    align-self: flex-start;  /* 关键：防止 sticky 被拉伸 */
    padding: 40px 0;
    max-height: 100vh;       /* 不是 80vh */
    overflow-y: auto;
}
.article-body {
    flex: 1;
    min-width: 0;            /* 防止内容溢出 */
}
```

**错误做法（避免）**：
- ❌ `position: fixed` — 脱离文档流，正文宽度计算异常
- ❌ `padding-left: 260px` 在正文上 — 与 grid 冲突
- ❌ 保留 `display: grid` + `scroll-behavior: smooth` — 长页面侧边栏消失

**移动端适配**：
```css
@media(max-width:900px){
    .article-layout{flex-direction:column; padding:0 16px}
    .toc-sidebar{position:static; flex:none; width:auto; max-height:none;
                 border-bottom:1px solid var(--line); padding:0 0 16px 0; margin-bottom:24px}
}
```

### 同步配置
- `sync_reports.sh` 的 `PROJECTS` 数组中，entry 字段指向首页 `index.html` 而非具体报告
- main `index.html` 的卡片和 `SEARCH_DB` 需同步更新链接

### 风格保留规则
- **Notion 风格已完全停用** - restyle_reports.py 不再执行任何 restyle 操作
- 所有报告统一使用暖纸色风格（Libre Baskerville + Manrope + 墨绿主题）
- 子页面使用 `shared.css`（外部样式表），不内联 style
- `shared.css` 包含完整的暖纸色 CSS 变量系统
- 首页使用内联 `<style>`（与子页面 shared.css 相同的 CSS 变量）

### 发布流程最佳实践
1. **复用源文件**: 直接使用源 `index.html` 作为首页，不重写内容
2. **仅增加导航**: 在源文件 topbar 中添加主站导航链接
3. **相对路径计算**: 根据首页实际位置计算到站点根的路径
   - `deep-ep/index.html` → 到站点根用 `../`
   - `umdk/analysis/index.html` → 到站点根用 `../../`
4. **同步后验证**: rsync 后检查 MD5 确保源文件正确同步到目标
5. **restyle 排除**: 有自定义样式的报告目录必须加入 `EXCLUDE_DIRS`
6. **入口更新**: `sync_reports.sh` + `build_site.py` 的 entry 改为 `index.html`

### Latest survey 自动刷新
- `build_site.py` 的 `gen_hero_aside()` 按 `published_at` 日期选最新报告
- 每次发布新报告时，在 REPORTS 数组中添加 `"published_at":"YYYY-MM-DD"` 字段
- 最新报告会自动显示在首页 "Latest survey" 位置
- 无 `published_at` 的报告不参与比较（向后兼容）

### 常见陷阱
- **build_site.py 覆盖手动修改**: 每次构建从 REPORTS 数组重新生成 index.html，必须改 REPORTS 而非手动编辑 index.html
- **rsync --delete 删除自定义文件**: 源目录没有 index.html 时会被删除 → 将 index.html 放入源目录
- **rsync 时间戳跳过**: 目标文件时间戳比源文件新时会跳过 → 删除目标文件或强制 rsync
- **git pull 恢复旧文件**: sync_reports.sh 先 git pull 会恢复刚删除的文件 → 手动 rsync 后直接提交
- **shared.css 404（严重）**: 报告在子目录（如 `deepepv2/html/`）时，`href="shared.css"` 相对路径指向子目录内的 shared.css。必须将 shared.css 复制到每个子目录：`cp deepepv2/shared.css deepepv2/html/shared.css`。否则所有页面样式丢失（HTTP 404）
- **scroll-behavior: smooth 长页面问题**: shared.css 设置 `html{scroll-behavior:smooth}`，长页面（>1000行）点击 TOC 锚点时粘性侧边栏会视觉"消失"。修复：在报告 `<style>` 中添加 `html{scroll-behavior:auto}` 覆盖

---

## 任务经验记录

### 2026-08-21: deepepv2 站点重构 + Engram 深度分析报告

#### 任务概述
- **目标**: 重构 deepepv2 报告站点（参照 UMDK 暖纸色风格），新增 Engram 0 SM RDMA 深度分析报告
- **产出**: 导航页 restyle + 新报告 `00_deep_ep_engram_architecture_deepdive.html` + 计数修正 46→50

#### 关键改动
1. **导航页 `deepepv2/html/index.html`**: 旧版 Notion 白底 → UMDK 暖纸色 warm paper 风格
   - topbar + hero + stat-card + 7 大类别卡片网格 + 演进表 + 阅读路径
   - 复用 shared.css 设计系统
2. **新增 Engram 报告**: 11 节完整分析（架构/Gin 交互/RoCE-NVLink/并发/对称内存/SM 调用 RoCE/部署/测试）
3. **新增 HybridElasticSymmetricMemory 深度专题**: 回答三个核心问题
   - 为什么要对称架构 → O(1) 全局地址计算
   - 对称性对单向查表意义 → 无状态路由 + 单向性 + 消除 CPU 查表
   - 是否跨节点 → **物理本地 + 逻辑全局**（CPU 段 NUMA-local，VA 布局全局对称，RDMA 跨节点访问）
4. **计数修正**: 旧索引遗漏 file 12，总数 46→49→50（含 Engram 报告）

#### 踩坑与教训
1. **shared.css 404（严重）**:
   - 报告在 `deepepv2/html/` 子目录，引用 `href="shared.css"` 相对路径
   - 文件只在 `deepepv2/shared.css`，子目录没有 → HTTP 404 → 全部页面样式丢失
   - 修复：`cp deepepv2/shared.css deepepv2/html/shared.css`
   - **预防**: 新增子目录报告时，必须同时复制 shared.css

2. **侧边栏布局问题（三次修复）**:
   - v1: `scroll-behavior:smooth` + `position:sticky` → 长页面锚点跳转时侧边栏视觉"消失"
   - v2: 改为 `position:fixed` → 侧边栏固定但正文宽度异常（脱离文档流）
   - v3（最终）: 改为 flexbox 模型（对齐 main.css）→ 正常
   - **教训**: 长页面侧边栏不要用 grid + sticky，用 flexbox + `align-self:flex-start`

3. **报告计数不一致**:
   - sync_reports.sh 描述写 47，实际 46 → 发现遗漏 file 12
   - 修正所有计数（inner index + outer index + sync_reports.sh）

#### 有效策略
1. **Explore agent 并行分析源码**: 用 Explore agent 并行读取 8+ 个源文件，快速收集技术细节
2. **先写 HTML 再调样式**: 先完成完整内容，再统一修复布局问题
3. **对比正常页面**: 出问题时对比 `main.css`（正常 post）和 `shared.css`（deepepv2）的差异

#### 方法论沉淀
- **长页面侧边栏标准模型**: flexbox + `flex:0 0 220px` + `position:sticky` + `align-self:flex-start`
- **shared.css 部署检查清单**: 子目录报告必须包含 shared.css 副本
- **站点计数同步原则**: 新增报告后必须同步更新 inner index + outer index + sync_reports.sh 三处计数

---

## 维护命令
```bash
./sync_reports.sh              # 全量同步+提交+推送
./sync_reports.sh --dry-run    # 预览
./sync_reports.sh --no-push    # 本地试
```

## 写文章经验

### 数据驱动型写作
- **先定结论** → 一句话锚点,所有数据为它服务
- **骨架优先** → 读者 30 秒扫完全文(标题+表格+图表)
- **能用表格不用文字** → 让读者自己算,不替读者做判断
- **控制字数** → 删比写更重要,去掉不影响结论的段落
- **主观判断必须有数据兜底** → 无数据支撑的结论不说
- **迭代修改** → v0.1 数据堆砌 → v2.0 精简核心

### 数据核实
- 所有报价必须来自官方文档,标注来源链接
- 计算过程要展示,让读者可验证
- 价格/数据更新时,全文所有相关数字都要同步修改
