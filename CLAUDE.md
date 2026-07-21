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

---

## 维护命令
```bash
./sync_reports.sh              # 全量同步+提交+推送
./sync_reports.sh --dry-run    # 预览
./sync_reports.sh --no-push    # 本地试
```
