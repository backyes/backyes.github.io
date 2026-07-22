#!/usr/bin/env python3
"""
build_site.py — 生成 index.html / posts.html / tags.html
从模板 + 报告元数据 + 手写 posts/*.md 构建完整静态站。

由 sync_reports.sh 在 rsync 之后调用。
"""
import os, re, json, glob, html

REPO = os.path.dirname(os.path.abspath(__file__))
POSTS_DIR = os.path.join(REPO, 'posts')

# ──── 报告元数据 (与 sync_reports.sh 的 PROJECTS 数组保持同步) ────
# visual = 卡片上的大字 (提取标题关键词)
# tags = 具体技术关键词 (用于 Tags 页聚合)
REPORTS = [
    {"dst":"umdk","entry":"analysis/cam/CAM深度分析报告.html","visual":"UMDK",
     "title":"UMDK/CAM 深度分析",
     "desc":"Communication Acceleration for Matrix 架构深度分析 · 整体架构设计哲学 / 模块分解 / framework-nda 传输抽象层 / Operator Registry 与 SOC 世代 / op_host-op_kernel-op_api 三层编程模型 / 编译系统与部署",
     "cat":"chip","priority":"p0",
     "tags":["UMDK","CAM","通信加速","传输抽象","Operator Registry","SOC","编程模型","编译系统"]},
    {"dst":"vllm_research/vllm_analysis","entry":"index.html","visual":"vLLM",
     "title":"vLLM 架构统一分析",
     "desc":"12 章统一分析 (第一性原理 / 热路径 / KV-Cache 4 层 / 分布式 / Ascend Overlay / Perf Handbook) · 合并 spine+L4 agent+源码+社区 90d pulse · 每节 source 溯源锚点",
     "cat":"inference","priority":"p0",
     "tags":["vLLM","KV-Cache","调度器","分布式推理","Ascend","性能调优"]},
    {"dst":"pd-separation","entry":"report.html","visual":"P/D",
     "title":"P/D 分离 KVCache 流通",
     "desc":"vLLM / SGLang / LMCache / Mooncake / Dynamo 五大框架的 Prefill-Decode 分离 + KV Cache 路由内部实现源码级拆解 · BootstrapQueue→WaitingQueue→InflightQueue 全生命周期",
     "cat":"inference","priority":"p0",
     "tags":["P-D分离","KV-Cache","vLLM","SGLang","Dynamo","请求路由"]},
    {"dst":"mlsys2026","entry":"index.html","visual":"MLSys",
     "title":"MLSys 2026 深度综合",
     "desc":"Keynote + 19 篇论文逐篇深度解读后的跨论文战略综合 · 6 条主轴: 同步税 / 存储层级重定义 / P2P 转移 / Superchip 冲击 / 批判性转向 / 训练路线分叉",
     "cat":"conference","priority":"p0",
     "tags":["MLSys","论文综述","训练系统","推理系统","Superchip","存储层级"]},
    {"dst":"deepseek-mtp","entry":"index.html","visual":"MTP",
     "title":"DeepSeek MTP 算力影响",
     "desc":"dspark MTP 算法对算力与总线系统行业的深度影响分析 · 算法设计者视角的范式推演",
     "cat":"model","priority":"p1",
     "tags":["MTP","DeepSeek","DSpark","算力","模型架构","推理加速"]},
    {"dst":"moe-clos","entry":"report.html","visual":"CLOS",
     "title":"Sparse CLOS × MoE 推理",
     "desc":"MoE 专家并行推理在 Sparse CLOS 网络上的效率与成本收益深度分析 · MegaScale / MixNet / UBEP / SpecMoE 多篇对比",
     "cat":"network","priority":"p1",
     "tags":["MoE","CLOS","SparseCLOS","专家并行","MegaScale","SpecMoE"]},
    {"dst":"generative-rec","entry":"generative_recommendation_report.html","visual":"RecSys",
     "title":"生成式推荐研究热点",
     "desc":"2026 年 Generative Recommendation 最新研究热点调查报告 · 算法 + 系统 + 工业落地",
     "cat":"recsys","priority":"p1",
     "tags":["推荐系统","生成式推荐","RecSys","工业落地","算法"]},
    {"dst":"sparse-clos","entry":"sparse_clos_report.html","visual":"CLOS",
     "title":"Sparse Clos 组网深度调研",
     "desc":"Sparse Clos / SlimFly / Jupiter 等无阻塞组网技术的深度调研 · 来源: 论文 + 厂商 + 学术会议",
     "cat":"network","priority":"p1",
     "tags":["CLOS","SlimFly","Jupiter","无阻塞网络","组网","数据中心网络"]},
    {"dst":"ai-supernode-bus","entry":"report.html","visual":"SuperNode",
     "title":"AI 超节点总线调研",
     "desc":"2026H1 AI 超节点总线技术市场调研 · NVLink / UALink / PCIe 6 / 光互联 + 产业格局",
     "cat":"cluster","priority":"p1",
     "tags":["超节点","NVLink","UALink","PCIe","光互联","集群"]},
    {"dst":"supernode-metrics","entry":"supernode_metrics_report.html","visual":"Metric",
     "title":"超节点指标定义",
     "desc":"超节点行业指标定义深度调研 · 制造商(NVIDIA/华为/Google) / 云商 / 学术 三视角 + 量化指标体系",
     "cat":"cluster","priority":"p1",
     "tags":["超节点","指标体系","NVIDIA","华为","Google","量化指标"]},
    {"dst":"mtp-survey","entry":"MTP_DSpark_Survey.html","visual":"MTP",
     "title":"MTP 算法 Survey",
     "desc":"大模型推理 MTP (Multi-Token Prediction) 算法 Survey · 围绕 DeepSeek DSpark 的全景调研",
     "cat":"model","priority":"p1",
     "tags":["MTP","DSpark","DeepSeek","算法调研","模型架构"]},
    {"dst":"3dls","entry":"3DLS_analysis_report.html","visual":"3DLS",
     "title":"3DLS 论文深度分析",
     "desc":"3DLS 论文深度分析报告 · 芯片 / 系统 / AI 推理架构 交叉视角",
     "cat":"chip","priority":"p2",
     "tags":["3DLS","芯片架构","3D封装","系统架构","推理架构"]},
    {"dst":"space-ecom","entry":"report.html","visual":"Space",
     "title":"太空经济联盟调研",
     "desc":"联盟首批意向成员 + 初创企业调研报告 · 含 306 家深度分析",
     "cat":"space","priority":"p2",
     "tags":["太空经济","航天","初创企业","产业调研"]},

    # ── 以下为新规模化添加 ──
    {"dst":"pd-routing","entry":"report.html","visual":"Routing",
     "title":"PD 分离 Request Routing 内部实现",
     "desc":"PD 分离架构下请求路由的源码级拆解 · SGLang / Mooncake / LMCache 内部队列生命周期 · 与 P/D KVCache 流通互补视角",
     "cat":"inference","priority":"p1",
     "tags":["P-D分离","请求路由","SGLang","LMCache","队列","源码分析"]},
    {"dst":"trillium","entry":"Trillium_vs_NVIDIA_LPX_架构分析.html","visual":"Trillium",
     "title":"Trillium vs NVIDIA LPX 微架构分析",
     "desc":"Groq Trillium 与 NVIDIA LPX 微架构 / 集群架构深度对比 · 芯片设计范式 · 互联拓扑",
     "cat":"chip","priority":"p1",
     "tags":["Trillium","NVIDIA","LPX","微架构","Groq","芯片对比"]},
    {"dst":"spacex","entry":"太空经济与SpaceX深度分析报告.html","visual":"SpaceX",
     "title":"SpaceX 深度分析 (全球视野·AI算力视角)",
     "desc":"SpaceX 全版图深度分析 · 产品布局 / 财务数据 / 政府合同 / 技术迭代 · 中国太空经济对比 · AI 算力交叉视角",
     "cat":"space","priority":"p0",
     "tags":["SpaceX","太空经济","产业调研","财务分析","AI算力","全球视野"]},
    {"dst":"hbm-cxl","entry":"report.html","visual":"HBM",
     "title":"HBM / CXL / Memory 市场调研",
     "desc":"HBM CXL NAND 内存层级市场深度调研 · 三星 / SK海力士 / 美光 · 技术路线与竞争格局",
     "cat":"storage","priority":"p1",
     "tags":["HBM","CXL","存储","内存层级","三星","SK海力士","美光"]},
    {"dst":"inference-community","entry":"web/sources/coreweave_inference.html","visual":"InferCom",
     "title":"推理社区 2026 前沿动态",
     "desc":"全球推理社区前沿动态 · CoreWeave Particula LSYS 等创新企业 · 开源与商业化路径",
     "cat":"inference","priority":"p1",
     "tags":["推理社区","CoreWeave","推理部署","前沿动态","开源"]},
]

CATS = {
    "inference":  {"label":"推理架构","color":"tag-inference"},
    "model":      {"label":"模型架构","color":"tag-model"},
    "network":    {"label":"网络拓扑","color":"tag-network"},
    "chip":       {"label":"芯片架构","color":"tag-chip"},
    "cluster":    {"label":"集群系统","color":"tag-cluster"},
    "storage":    {"label":"存储系统","color":"tag-storage"},
    "recsys":     {"label":"推荐系统","color":"tag-recsys"},
    "conference": {"label":"学术会议","color":"tag-conference"},
    "space":      {"label":"太空经济","color":"tag-space"},
}

# ──── 解析手写文章 ────
def parse_post(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.S)
    if not m:
        return None
    # 简易 frontmatter 解析 (不依赖 pyyaml)
    fm = {}
    for line in m.group(1).split('\n'):
        line = line.strip()
        if ':' in line and not line.startswith('#'):
            k, v = line.split(':', 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if v.startswith('[') and v.endswith(']'):
                v = [x.strip().strip("'\"") for x in v[1:-1].split(',') if x.strip()]
            fm[k] = v
    body = m.group(2).strip()
    fname = os.path.basename(filepath)
    dm = re.match(r'(\d{4}-\d{2}-\d{2})', fname)
    date = dm.group(1) if dm else str(fm.get('date',''))
    title = fm.get('title','')
    tags = fm.get('tags', [])
    if isinstance(tags, str): tags = [t.strip() for t in tags.split(',')]
    excerpt = fm.get('excerpt','')
    if not excerpt:
        paras = [p.strip() for p in re.split(r'\n\n+', body)
                 if p.strip() and not p.startswith('#') and not p.startswith('-')]
        excerpt = re.sub(r'[#*\`\[\]]', '', paras[0])[:200] if paras else ''
    slug = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', fname).replace('.md','')
    return {
        "title": title, "date": str(date), "tags": tags,
        "excerpt": excerpt, "slug": slug, "source": filepath,
    }

def load_posts():
    posts = []
    for fp in sorted(glob.glob(os.path.join(POSTS_DIR, '*.md')), reverse=True):
        p = parse_post(fp)
        if p:
            p["url"] = f"posts/{p['slug']}.html"
            posts.append(p)
    return posts

# ──── Markdown → HTML (完整支持表格/引用/代码/链接/加粗) ────
def md_to_html(text):
    lines = text.split('\n')
    out = []
    i = 0
    in_ul = False
    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append('</ul>')
            in_ul = False
    def inline(s):
        # 图片 ![alt](url) — 必须在链接之前处理
        s = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1" style="max-width:100%;display:block;margin:1.5em auto">', s)
        # 关键数字 $number$ → 蓝色高亮
        s = re.sub(r'\$([\d,.]+)\$', r'<span class="key-num">\1</span>', s)
        # 高亮 ==text== → 下划线标记
        s = re.sub(r'==([^=]+)==', r'<mark>\1</mark>', s)
        # 代码 `code`
        s = re.sub(r'`([^`]+)`', r'<code>\1></code>', s)
        # 加粗 **text**
        s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
        # 斜体 *text* (不在 ** 内部)
        s = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', s)
        # 链接 [text](url)
        s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
        return s

    while i < len(lines):
        s = lines[i].strip()
        # 代码块 ```
        if s.startswith('```'):
            lang = s[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            close_ul()
            out.append(f'<pre><code>{"<br>".join(code_lines)}</code></pre>')
            i += 1
            continue
        # 表格 | ... | ... |
        if '|' in s and s.startswith('|') and s.endswith('|') and i+1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i+1].strip()):
            header = [c.strip() for c in s.strip('|').split('|')]
            i += 2  # skip header + separator
            rows = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip().startswith('|'):
                row = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                rows.append(row)
                i += 1
            close_ul()
            th = ''.join(f'<th>{inline(c)}</th>' for c in header)
            out.append(f'<table><thead><tr>{th}</tr></thead><tbody>')
            for row in rows:
                td = ''.join(f'<td>{inline(c)}</td>' for c in row)
                out.append(f'<tr>{td}</tr>')
            out.append('</tbody></table>')
            continue
        # 标题
        hm = re.match(r'^(#{1,3})\s+(.*)', s)
        if hm:
            close_ul()
            level = len(hm.group(1))
            out.append(f'<h{level}>{inline(hm.group(2))}</h{level}>')
            i += 1; continue
        # 引用 >
        if s.startswith('>'):
            close_ul()
            quote = s[1:].strip()
            out.append(f'<blockquote><p>{inline(quote)}</p></blockquote>')
            i += 1; continue
        # 分隔线 ---
        if s == '---':
            close_ul()
            out.append('<hr>')
            i += 1; continue
        # 独立的图片行 ![alt](url)
        img_only = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)$', s)
        if img_only:
            close_ul()
            out.append(f'<img src="{img_only.group(2)}" alt="{img_only.group(1)}" style="max-width:100%;display:block;margin:1.5em auto">')
            i += 1; continue
        # 无序列表
        if s.startswith('- ') or s.startswith('* '):
            if not in_ul:
                out.append('<ul>'); in_ul = True
            out.append(f'<li>{inline(s[2:])}</li>')
            i += 1; continue
        # 空行
        if s == '':
            close_ul()
            i += 1; continue
        # 普通段落
        close_ul()
        out.append(f'<p>{inline(s)}</p>')
        i += 1
    close_ul()
    return '\n'.join(out)

# ──── 卡片视觉: 大字标题 (HTML 文字, 响应式字号匹配首页 hero) ────
def visual_placeholder(visual, cat):
    # 灰阶配色
    shades = {
        "inference":"#2a2f38",
        "system":   "#2d323c",
        "network":  "#2b303a",
        "chip":     "#2e333e",
        "mixed":    "#2c313b",
    }
    c = shades.get(cat, "#2a2f38")
    txt = visual or ""
    # HTML div + clamp() 字号,匹配首页 "Thoughts on AI infrastructure" 风格
    return f'<div class="visual" style="background:{c}">{txt}</div>'

# ──── 生成报告卡片 ────
def gen_cards():
    cards = ""
    for prefix in ("p0","p1","p2"):
        for r in REPORTS:
            if r["priority"] != prefix: continue
            dst, entry = r["dst"], r["entry"]
            if not os.path.isfile(os.path.join(REPO, dst, entry)): continue
            cat = CATS.get(r["cat"], CATS["inference"])
            visual = visual_placeholder(r.get("visual",""), r["cat"])
            cards += f'''<a class="card" href="{dst}/{entry}" data-cat="{r['cat']}">
  <div class="card-img">{visual}<span class="tag {cat['color']}">{cat['label']}</span></div>
  <div class="card-body">
    <h3>{r['title']}</h3>
    <div class="card-foot"><span class="more">阅读 →</span></div>
  </div>
</a>
'''
    return cards

# ──── 生成 Search DB (JSON) ────
def gen_search_db(posts):
    items = []
    for r in REPORTS:
        items.append({"t": r["title"], "d": r["desc"], "u": f"{r['dst']}/{r['entry']}",
                      "tags": " ".join(r.get("tags",[]))})
    for p in posts:
        items.append({"t": p["title"], "d": p["excerpt"], "u": p["url"],
                      "tags": " ".join(p.get("tags",[]))})
    return json.dumps(items, ensure_ascii=False)

# ──── 生成 Tag Cloud — 使用具体技术 tags ────
def gen_tag_cloud(posts):
    tag_count = {}
    for r in REPORTS:
        for t in r.get("tags", []):
            tag_count[t] = tag_count.get(t, 0) + 1
    for p in posts:
        for t in p.get("tags", []):
            tag_count[t] = tag_count.get(t, 0) + 1
    # 按计数降序,取前 20 个
    html_parts = []
    for tag, count in sorted(tag_count.items(), key=lambda x: -x[1])[:24]:
        html_parts.append(f'<a href="tags.html" class="tag-pill"># {tag} <span class="count">{count}</span></a>')
    return "\n".join(html_parts)

# ──── 生成 Posts 列表 (Lil'Log 风格) ────
def gen_posts_list(posts, full=False):
    if not posts:
        return ('<div class="empty"><div class="emoji">✍️</div>'
                '<p>Posts 栏目已预留。<br>这里将放入我个人的观察、分析与观点——独立于 AI 报告的手写洞察。<br><br>'
                '<em>敬请期待。</em></p></div>')
    items = ""
    for p in posts:
        tags_html = "".join(f'<span class="ptag">{t}</span>' for t in p.get("tags",[]))
        items += f'''<li class="posts-item">
  <div class="posts-date">{p['date']}</div>
  <div class="posts-content">
    <h3><a href="{p['url']}">{p['title']}</a></h3>
    <p class="posts-excerpt">{p['excerpt']}</p>
    <div class="posts-meta">{tags_html}</div>
  </div>
</li>
'''
    return f'<ul class="posts-list">{items}</ul>'

# ──── 生成标签 sidebar ────
def gen_tags_sidebar(posts):
    tag_count = {}
    for r in REPORTS:
        for t in r.get("tags", []):
            tag_count[t] = tag_count.get(t, 0) + 1
    for p in posts:
        for t in p.get("tags", []):
            tag_count[t] = tag_count.get(t, 0) + 1
    out = ""
    for tag, count in sorted(tag_count.items(), key=lambda x: (-x[1], x[0])):
        out += f'<li><a href="#tag-{tag}"># {tag} <span class="cnt">{count}</span></a></li>\n'
    return out

# ──── 生成 Tags 页完整内容 — 按具体技术 tag 聚合 ────
def gen_tags_full(posts):
    groups = {}
    for r in REPORTS:
        for t in r.get("tags", []):
            groups.setdefault(t,[]).append(
                {"t": r["title"], "d": r["desc"], "u": f"{r['dst']}/{r['entry']}", "kind":"AI"})
    for p in posts:
        for t in p.get("tags", []):
            groups.setdefault(t,[]).append(
                {"t": p["title"], "d": p["excerpt"], "u": p["url"], "kind":"Post"})
    out = ""
    # 按计数降序排列
    for tag in sorted(groups.keys(), key=lambda t: -len(groups[t])):
        items = groups[tag]
        out += f'<div class="tag-group" id="tag-{tag}" data-tag="{tag}">'
        out += f'<h3><span class="hash">#</span> {tag} <span class="cnt">({len(items)})</span></h3>'
        out += '<ul class="posts-list">'
        for it in items:
            kind_label = 'AI' if it.get('kind')=='AI' else 'Post'
            out += f'''<li class="posts-item">
  <div class="posts-content"><h3><a href="{it['u']}">{it['t']}</a></h3><p class="posts-excerpt">{it['d']}</p></div>
</li>'''
        out += '</ul></div>'
    return out

# ──── 生成单篇文章 HTML ────
def gen_post_page(post):
    with open(post["source"], 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.match(r'^---\s*\n.*?\n---\s*\n(.*)$', content, re.S)
    body = m.group(1).strip() if m else content
    # 跳过 body 中第一个 # 标题(与文章标题重复)
    body_lines = body.split('\n')
    start = 0
    for idx, line in enumerate(body_lines):
        if line.strip().startswith('# '):
            start = idx + 1
            break
        elif line.strip() and not line.strip().startswith('---'):
            break
    body = '\n'.join(body_lines[start:]).strip()
    body_html = md_to_html(body)
    tags_html = "".join(f'<span class="ptag">{t}</span>' for t in post.get("tags",[]))
    return f'''<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{post["title"]} · backyes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Source+Serif+Pro:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/css/main.css">
<style>
.article{{max-width:720px;margin:0 auto;padding:40px 24px 80px}}
.article h1{{font-family:var(--font-serif);font-size:2rem;font-weight:600;margin:0 0 8px}}
.article .meta{{color:var(--muted-2);font-size:.85rem;margin-bottom:40px;padding-bottom:20px;border-bottom:1px solid var(--border-soft);display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
.article-body{{font-family:var(--font-serif);font-size:1.1rem;line-height:1.8;color:var(--fg-2)}}
.article-body h2{{font-size:1.5rem;font-weight:700;margin:2em 0 .6em;padding-bottom:.4em;border-bottom:2px solid var(--accent);color:var(--fg)}}
.article-body h3{{font-size:1.25rem;font-weight:600;margin:1.5em 0 .5em;padding-left:10px;border-left:3px solid var(--accent);color:var(--fg2)}}
.article-body p{{margin:0 0 1.2em}}
.article-body strong{{color:var(--fg);font-weight:600}}
.article-body mark{{background:linear-gradient(180deg,transparent 60%,var(--accent-soft) 60%);padding:0 3px;color:var(--fg);font-weight:600}}
.article-body .key-callout{{border-left:4px solid var(--accent);background:var(--callout);padding:14px 20px;margin:1.5em 0;border-radius:0 8px 8px 0}}
.article-body .key-callout p{{margin:0 0 .5em}}
.article-body .key-callout p:last-child{{margin:0}}
.article-body .key-num{{color:var(--accent);font-weight:700;font-size:1.1em}}
.article-body .hl-blue{{background:rgba(74,143,224,.12);border-left:3px solid var(--accent);padding:10px 16px;margin:1.2em 0;border-radius:0 6px 6px 0}}
.article-body .hl-amber{{background:rgba(210,153,34,.12);border-left:3px solid #d29922;padding:10px 16px;margin:1.2em 0;border-radius:0 6px 6px 0}}
.article-body .hl-green{{background:rgba(63,185,80,.12);border-left:3px solid #3fb950;padding:10px 16px;margin:1.2em 0;border-radius:0 6px 6px 0}}
.article-body .hl-blue p,.article-body .hl-amber p,.article-body .hl-green p{{margin:0}}
.article-body table{{border-collapse:collapse;width:100%;margin:1.5em 0;font-size:.88em;display:block;overflow-x:auto}}
.article-body th{{background:var(--surface);font-weight:600;padding:8px 10px;border:1px solid var(--border);text-align:left}}
.article-body td{{padding:7px 10px;border:1px solid var(--border);vertical-align:top}}
.article-body tr:nth-child(even) td{{background:rgba(24,27,32,.25)}}
.article-body tr:hover td{{background:rgba(74,143,224,.08)}}
.article-body blockquote{{border-left:4px solid var(--accent);background:var(--callout);padding:10px 16px;margin:1.2em 0;border-radius:0 6px 6px 0}}
.article-body blockquote p{{margin:0}}
.article-body code{{background:var(--surface);border:1px solid var(--border2);padding:1px 5px;border-radius:4px;font-size:.85em;font-family:'JetBrains Mono',monospace}}
.article-body pre code{{background:none;border:none;padding:0}}
.article-body a{{color:var(--accent);text-decoration:none}}
.article-body a:hover{{text-decoration:underline}}
.article-body hr{{border:none;border-top:1px solid var(--border-soft);margin:2em 0}}
.back{{display:inline-block;margin-bottom:28px;color:var(--accent);font-size:.9rem}}
@media(max-width:768px){{
  .article{{padding:24px 16px 60px}}
  .article-body h2{{font-size:1.3rem}}
  .article-body h3{{font-size:1.1rem}}
  .article-body table{{font-size:.8rem}}
  .article-body th,.article-body td{{padding:6px 8px}}
}}
</style>
</head>
<body>
<nav class="nav"><div class="nav-inner">
  <a href="../" class="nav-brand"><span class="dot"></span>backyes</a>
  <ul class="nav-links">
    <li><a href="../#survey">Survey by AI</a></li>
    <li><a href="../posts.html" class="active">Posts</a></li>
    <li><a href="../tags.html">Tags</a></li>
  </ul>
</div></nav>
<article class="article">
  <a href="../posts.html" class="back">← 返回 Posts</a>
  <h1>{post["title"]}</h1>
  <div class="meta"><span>📅 {post["date"]}</span><div>{tags_html}</div></div>
  <div class="article-body">
{body_html}
  </div>
</article>
<footer class="footer"><div class="wrap"><p>© 2026 backyes · All rights reserved</p></div></footer>
</body>
</html>'''

# ──── 主流程 ────
def main():
    posts = load_posts()
    print(f"  解析到 {len(posts)} 篇手写文章")

    # 生成报告卡片
    cards = gen_cards()
    search_db = gen_search_db(posts)
    tag_cloud = gen_tag_cloud(posts)
    posts_preview = gen_posts_list(posts[:6])  # index 预览前 6 篇
    posts_full = gen_posts_list(posts, full=True)
    posts_search_db = json.dumps(
        [{"t":p["title"],"d":p["excerpt"],"u":p["url"],"tags":" ".join(p.get("tags",[]))}
         for p in posts], ensure_ascii=False)
    tags_full = gen_tags_full(posts)
    tags_sidebar = gen_tags_sidebar(posts)

    # index.html (从 index.html 自身读取 — template 与输出合并,placeholder 在 source 里)
    idx = open(os.path.join(REPO, 'index.html'), encoding='utf-8').read()
    # 为了幂等,保留 SEARCH_DB 的占位: 先找到并替换,写回时仍然保留 placeholder
    # 策略: 替换 SEARCH_DB 那一行,但写回 template 时保留 <!--SEARCH_DB-->
    # 简化: 直接输出,但保留 <!--SEARCH_DB--> placeholder (替换时仅替换 JSON 部分)
    idx = re.sub(r'<!--PROJECT_CARDS-->.*?<!--/PROJECT_CARDS-->',
                 f'<!--PROJECT_CARDS-->\n{cards}<!--/PROJECT_CARDS-->', idx, flags=re.S)
    # SEARCH_DB: 匹配 const SEARCH_DB=[...]; 或 const SEARCH_DB=<!--SEARCH_DB-->; 整体替换
    idx = re.sub(r'const SEARCH_DB=(?:\[.*?\]|<!--SEARCH_DB-->);',
                 f'const SEARCH_DB={search_db};', idx, flags=re.S)
    idx = re.sub(r'<!--TAG_CLOUD-->.*?<!--/TAG_CLOUD-->',
                 f'<!--TAG_CLOUD-->\n{tag_cloud}\n<!--/TAG_CLOUD-->', idx, flags=re.S)
    idx = re.sub(r'<!--PREVIEW_POSTS-->.*?<!--/PREVIEW_POSTS-->',
                 f'<!--PREVIEW_POSTS-->\n{posts_preview}\n<!--/PREVIEW_POSTS-->', idx, flags=re.S)
    open(os.path.join(REPO,'index.html'),'w',encoding='utf-8').write(idx)
    print("  ✓ index.html 已生成")

    # posts.html
    ph = open(os.path.join(REPO, 'posts.html'), encoding='utf-8').read()
    ph = re.sub(r'<!--POSTS_FULL-->.*?<!--/POSTS_FULL-->',
                f'<!--POSTS_FULL-->\n{posts_full}\n<!--/POSTS_FULL-->', ph, flags=re.S)
    ph = re.sub(r'const POSTS_DB=(?:\[.*?\]|<!--POSTS_SEARCH_DB-->);',
                f'const POSTS_DB={posts_search_db};', ph, flags=re.S)
    open(os.path.join(REPO,'posts.html'),'w',encoding='utf-8').write(ph)
    print("  ✓ posts.html 已生成")

    # tags.html
    th = open(os.path.join(REPO, 'tags.html'), encoding='utf-8').read()
    th = re.sub(r'<!--TAGS_SIDEBAR-->.*?<!--/TAGS_SIDEBAR-->',
                f'<!--TAGS_SIDEBAR-->\n{tags_sidebar}\n<!--/TAGS_SIDEBAR-->', th, flags=re.S)
    th = re.sub(r'<!--TAGS_FULL-->.*?<!--/TAGS_FULL-->',
                f'<!--TAGS_FULL-->\n{tags_full}\n<!--/TAGS_FULL-->', th, flags=re.S)
    open(os.path.join(REPO,'tags.html'),'w',encoding='utf-8').write(th)
    print("  ✓ tags.html 已生成")

    # 生成单篇文章 HTML
    for p in posts:
        page = gen_post_page(p)
        out = os.path.join(REPO, 'posts', f"{p['slug']}.html")
        open(out, 'w', encoding='utf-8').write(page)
        print(f"  ✓ 文章页: posts/{p['slug']}.html")

if __name__ == '__main__':
    main()
