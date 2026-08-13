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
2. `article-layout` 网格（TOC 侧边栏 + 主文章区）
3. `toc-sidebar` 自动生成目录（从 h1/h2/h3 提取）
4. `article-body` 使用 Libre Baskerville serif 字体

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
