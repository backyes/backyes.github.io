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
REPORTS = [
    {"dst":"vllm_research/vllm_analysis","entry":"index.html","visual":"vLLM",
     "title":"vLLM 架构统一分析",
     "desc":"12 章统一分析 (第一性原理 / 热路径 / KV-Cache 4 层 / 分布式 / Ascend Overlay / Perf Handbook) · 合并 spine+L4 agent+源码+社区 90d pulse · 每节 source 溯源锚点",
     "cat":"inference","priority":"p0"},
    {"dst":"pd-separation","entry":"report.html","visual":"P/D",
     "title":"P/D 分离 KVCache 流通",
     "desc":"vLLM / SGLang / LMCache / Mooncake / Dynamo 五大框架的 Prefill-Decode 分离 + KV Cache 路由内部实现源码级拆解 · BootstrapQueue→WaitingQueue→InflightQueue 全生命周期",
     "cat":"inference","priority":"p0"},
    {"dst":"mlsys2026","entry":"index.html","visual":"MLSys",
     "title":"MLSys 2026 深度综合",
     "desc":"Keynote + 19 篇论文逐篇深度解读后的跨论文战略综合 · 6 条主轴: 同步税 / 存储层级重定义 / P2P 转移 / Superchip 冲击 / 批判性转向 / 训练路线分叉",
     "cat":"mixed","priority":"p0"},
    {"dst":"deepseek-mtp","entry":"index.html","visual":"MTP",
     "title":"DeepSeek MTP 算力影响",
     "desc":"dspark MTP 算法对算力与总线系统行业的深度影响分析 · 算法设计者视角的范式推演",
     "cat":"inference","priority":"p1"},
    {"dst":"moe-clos","entry":"report.html","visual":"CLOS",
     "title":"Sparse CLOS × MoE 推理",
     "desc":"MoE 专家并行推理在 Sparse CLOS 网络上的效率与成本收益深度分析 · MegaScale / MixNet / UBEP / SpecMoE 多篇对比",
     "cat":"network","priority":"p1"},
    {"dst":"generative-rec","entry":"generative_recommendation_report.html","visual":"RecSys",
     "title":"生成式推荐研究热点",
     "desc":"2026 年 Generative Recommendation 最新研究热点调查报告 · 算法 + 系统 + 工业落地",
     "cat":"mixed","priority":"p1"},
    {"dst":"sparse-clos","entry":"sparse_clos_report.html","visual":"CLOS",
     "title":"Sparse Clos 组网深度调研",
     "desc":"Sparse Clos / SlimFly / Jupiter 等无阻塞组网技术的深度调研 · 来源: 论文 + 厂商 + 学术会议",
     "cat":"network","priority":"p1"},
    {"dst":"ai-supernode-bus","entry":"report.html","visual":"SuperNode",
     "title":"AI 超节点总线调研",
     "desc":"2026H1 AI 超节点总线技术市场调研 · NVLink / UALink / PCIe 6 / 光互联 + 产业格局",
     "cat":"chip","priority":"p1"},
    {"dst":"supernode-metrics","entry":"supernode_metrics_report.html","visual":"Metric",
     "title":"超节点指标定义",
     "desc":"超节点行业指标定义深度调研 · 制造商(NVIDIA/华为/Google) / 云商 / 学术 三视角 + 量化指标体系",
     "cat":"mixed","priority":"p1"},
    {"dst":"mtp-survey","entry":"MTP_DSpark_Survey.html","visual":"MTP",
     "title":"MTP 算法 Survey",
     "desc":"大模型推理 MTP (Multi-Token Prediction) 算法 Survey · 围绕 DeepSeek DSpark 的全景调研",
     "cat":"inference","priority":"p1"},
    {"dst":"3dls","entry":"3DLS_analysis_report.html","visual":"3DLS",
     "title":"3DLS 论文深度分析",
     "desc":"3DLS 论文深度分析报告 · 芯片 / 系统 / AI 推理架构 交叉视角",
     "cat":"chip","priority":"p2"},
    {"dst":"space-ecom","entry":"report.html","visual":"Space",
     "title":"太空经济联盟调研",
     "desc":"联盟首批意向成员 + 初创企业调研报告 · 含 306 家深度分析",
     "cat":"mixed","priority":"p2"},
]

CATS = {
    "inference": {"label":"推理引擎","color":"tag-inference"},
    "system":    {"label":"系统软件","color":"tag-system"},
    "network":   {"label":"网络拓扑","color":"tag-network"},
    "chip":      {"label":"芯片架构","color":"tag-chip"},
    "mixed":     {"label":"交叉领域","color":"tag-mixed"},
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

# ──── 简易 markdown → HTML ────
def md_to_html(text):
    lines = text.split('\n')
    out = []
    in_ul = False
    for line in lines:
        s = line.strip()
        if s.startswith('# '):
            if in_ul: out.append('</ul>'); in_ul = False
            out.append(f'<h1>{s[2:]}</h1>')
        elif s.startswith('## '):
            if in_ul: out.append('</ul>'); in_ul = False
            out.append(f'<h2>{s[3:]}</h2>')
        elif s.startswith('### '):
            if in_ul: out.append('</ul>'); in_ul = False
            out.append(f'<h3>{s[4:]}</h3>')
        elif s.startswith('- ') or s.startswith('* '):
            if not in_ul: out.append('<ul>'); in_ul = True
            out.append(f'<li>{s[2:]}</li>')
        elif s == '':
            if in_ul: out.append('</ul>'); in_ul = False
        else:
            if in_ul: out.append('</ul>'); in_ul = False
            out.append(f'<p>{s}</p>')
    if in_ul: out.append('</ul>')
    return '\n'.join(out)

# ──── 卡片视觉: 大字标题 (SVG 渲染, 不依赖外部图片) ────
def svg_placeholder(visual, cat):
    # 灰阶配色 + 微差, 大字用稍亮的灰
    shades = {
        "inference":("#2a2f38","#1a1e24"),
        "system":    ("#2d323c","#1c2026"),
        "network":   ("#2b303a","#1b1f25"),
        "chip":      ("#2e333e","#1d2128"),
        "mixed":     ("#2c313b","#1c2027"),
    }
    c1, c2 = shades.get(cat,("#2a2f38","#1a1e24"))
    # 根据文字长度调整字号
    txt = visual or ""
    n = len(txt)
    if n <= 3: size = 44
    elif n <= 5: size = 34
    elif n <= 8: size = 26
    else: size = 20
    return (f'<svg viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="200" height="120" fill="{c1}"/>'
            f'<rect x="0" y="84" width="200" height="36" fill="{c2}"/>'
            f'<text x="100" y="62" text-anchor="middle" fill="rgba(255,255,255,.62)" '
            f'font-family="Inter,sans-serif" font-weight="600" '
            f'font-size="{size}">{txt}</text></svg>')

# ──── 生成报告卡片 ────
def gen_cards():
    cards = ""
    for prefix in ("p0","p1","p2"):
        for r in REPORTS:
            if r["priority"] != prefix: continue
            dst, entry = r["dst"], r["entry"]
            if not os.path.isfile(os.path.join(REPO, dst, entry)): continue
            cat = CATS.get(r["cat"], CATS["mixed"])
            svg = svg_placeholder(r.get("visual",""), r["cat"])
            cards += f'''<a class="card" href="{dst}/{entry}" data-cat="{r['cat']}">
  <div class="card-img"><div class="ph">{svg}</div><span class="tag {cat['color']}">{cat['label']}</span></div>
  <div class="card-body">
    <h3>{r['title']}</h3>
    <p>{r['desc']}</p>
    <div class="card-foot"><span>📄 {dst}/{entry}</span><span class="more">阅读 →</span></div>
  </div>
</a>
'''
    return cards

# ──── 生成 Search DB (JSON) ────
def gen_search_db(posts):
    items = []
    for r in REPORTS:
        items.append({"t": r["title"], "d": r["desc"], "u": f"{r['dst']}/{r['entry']}",
                      "tags": CATS.get(r["cat"],{}).get('label','')})
    for p in posts:
        items.append({"t": p["title"], "d": p["excerpt"], "u": p["url"],
                      "tags": " ".join(p.get("tags",[]))})
    return json.dumps(items, ensure_ascii=False)

# ──── 生成 Tag Cloud ────
def gen_tag_cloud(posts):
    tag_count = {}
    for r in REPORTS:
        label = CATS.get(r["cat"],{}).get('label', r["cat"])
        tag_count[label] = tag_count.get(label, 0) + 1
    for p in posts:
        for t in p.get("tags", []):
            tag_count[t] = tag_count.get(t, 0) + 1
    html_parts = []
    for tag, count in sorted(tag_count.items(), key=lambda x: -x[1]):
        html_parts.append(f'<a href="tags.html" class="tag-pill">{tag} <span class="count">{count}</span></a>')
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

# ──── 生成 Tags 页完整内容 ────
def gen_tags_full(posts):
    # 按 tag 聚合
    groups = {}
    for r in REPORTS:
        label = CATS.get(r["cat"],{}).get('label', r["cat"])
        groups.setdefault(label,[]).append(
            {"t": r["title"], "d": r["desc"], "u": f"{r['dst']}/{r['entry']}"})
    for p in posts:
        for t in p.get("tags", []):
            groups.setdefault(t,[]).append(
                {"t": p["title"], "d": p["excerpt"], "u": p["url"]})
    out = ""
    for tag in sorted(groups.keys()):
        items = groups[tag]
        out += f'<div class="tag-group" data-tag="{tag}">'
        out += f'<h3><span class="hash">#</span>{tag} <span style="color:var(--muted-2);font-family:var(--font-mono);font-size:.9rem">({len(items)})</span></h3>'
        out += '<ul class="posts-list">'
        for it in items:
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
.article-body{{font-family:var(--font-serif);font-size:1.12rem;line-height:1.85;color:var(--fg-2)}}
.article-body h2{{font-size:1.4rem;margin:2em 0 .6em}}
.article-body h3{{font-size:1.2rem;margin:1.6em 0 .5em}}
.article-body p{{margin:0 0 1.2em}}
.article-body ul{{padding-left:1.4em;margin:0 0 1.2em}}
.article-body li{{margin-bottom:.4em}}
.back{{display:inline-block;margin-bottom:28px;color:var(--accent);font-size:.9rem}}
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
<footer class="footer"><div class="wrap"><p>© 2026 backyes · Handwritten</p></div></footer>
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
    posts_preview = gen_posts_list(posts[:5])  # index 预览前 5 篇
    posts_full = gen_posts_list(posts, full=True)
    posts_search_db = json.dumps(
        [{"t":p["title"],"d":p["excerpt"],"u":p["url"],"tags":" ".join(p.get("tags",[]))}
         for p in posts], ensure_ascii=False)
    tags_full = gen_tags_full(posts)

    # 写 index.html
    idx = open(os.path.join(REPO, 'index.html'), encoding='utf-8').read()
    idx = re.sub(r'<!--PROJECT_CARDS-->.*?<!--/PROJECT_CARDS-->',
                 f'<!--PROJECT_CARDS-->\n{cards}<!--/PROJECT_CARDS-->', idx, flags=re.S)
    idx = re.sub(r'<!--SEARCH_DB-->', search_db, idx)
    idx = re.sub(r'<!--TAG_CLOUD-->.*?<!--/TAG_CLOUD-->',
                 f'<!--TAG_CLOUD-->\n{tag_cloud}\n<!--/TAG_CLOUD-->', idx, flags=re.S)
    idx = re.sub(r'<!--PREVIEW_POSTS-->.*?<!--/PREVIEW_POSTS-->',
                 f'<!--PREVIEW_POSTS-->\n{posts_preview}\n<!--/PREVIEW_POSTS-->', idx, flags=re.S)
    open(os.path.join(REPO,'index.html'),'w',encoding='utf-8').write(idx)
    print("  ✓ index.html 已生成")

    # 写 posts.html
    ph = open(os.path.join(REPO, 'posts.html'), encoding='utf-8').read()
    ph = re.sub(r'<!--POSTS_FULL-->.*?<!--/POSTS_FULL-->',
                f'<!--POSTS_FULL-->\n{posts_full}\n<!--/POSTS_FULL-->', ph, flags=re.S)
    ph = re.sub(r'<!--POSTS_SEARCH_DB-->', posts_search_db, ph)
    open(os.path.join(REPO,'posts.html'),'w',encoding='utf-8').write(ph)
    print("  ✓ posts.html 已生成")

    # 写 tags.html
    th = open(os.path.join(REPO, 'tags.html'), encoding='utf-8').read()
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
