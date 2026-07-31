#!/usr/bin/env python3
"""
build_site.py v2 — 生成 index.html / posts.html / tags.html
Vikas Goyal 风格 + Lil'Log Posts + astrofy Survey Cards

从模板 + 报告元数据 + 手写 posts/*.md 构建完整静态站。
由 sync_reports.sh 在 rsync 之后调用。
"""
import os, re, json, glob, html

REPO = os.path.dirname(os.path.abspath(__file__))
POSTS_DIR = os.path.join(REPO, 'posts')

# ──── 报告元数据 (与 sync_reports.sh 的 PROJECTS 数组保持同步) ────
REPORTS = [
    {"dst":"amd-latest-tech-2026","entry":"index.html","visual":"AMD",
     "title":"AMD 全栈 AI 基础设施调研",
     "desc":"AMD Advancing AI 2026 全栈深度调研 · EPYC Venice (Zen 6/6c, 256 核) / MI455X (CDNA 4, 2.61 PFLOPS FP8) / Helios 机架 / Ryzen AI Gorgon · 芯片 + 集群 + 路线图",
     "cat":"chip","priority":"p0",
     "tags":["AMD","EPYC","Venice","MI455X","CDNA","Helios","Zen6","RyzenAI","Gorgon","路线图"]},
    {"dst":"umdk","entry":"analysis/cam_v2/CAM深度分析报告_v2.html","visual":"UMDK",
     "title":"UMDK/CAM 深度分析 v2",
     "desc":"CAM 通信加速库 8 章深度分析 v2 (2026-07-27) · EP Normal Zero-Buffer 流水线 / EP Low-Latency 8+4 阶段细粒度同步 / Fused Deep MoE 三变体 + Catlass GEMM / Detour All-to-All+ReduceScatter / A2E-E2A 跨域通信",
     "cat":"chip","priority":"p0",
     "tags":["UMDK","CAM","EP通信","Low-Latency","Fused Deep MoE","Catlass","Detour","A2E","跨域通信","设计洞察"]},
    {"dst":"vllm_research/vllm_analysis","entry":"index.html","visual":"vLLM",
     "title":"vLLM 架构统一分析",
     "desc":"12 章统一分析 + 3 专题 (第一性原理 / 热路径 / KV-Cache 4 层 / 分布式 / Ascend Overlay / Perf Handbook) · 百万序列 Prefill 专题 12 章 · KV Cache 即 Buffer / 1M token = 1.15 GB / NIXL vs Mooncake",
     "cat":"inference","priority":"p0",
     "tags":["vLLM","KV-Cache","调度器","分布式推理","Ascend","百万Prefill","D卡缓冲","HBM索引"]},
    {"dst":"pd-separation","entry":"report.html","visual":"P/D",
     "title":"P/D 分离 KVCache 流通",
     "desc":"vLLM / SGLang / LMCache / Mooncake / Dynamo 五大框架的 Prefill-Decode 分离 + KV Cache 路由内部实现源码级拆解",
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
    {"dst":"pd-routing","entry":"report.html","visual":"Routing",
     "title":"PD 分离 Request Routing 内部实现",
     "desc":"PD 分离架构下请求路由的源码级拆解 · SGLang / Mooncake / LMCache 内部队列生命周期",
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
    {"dst":"deep-ep","entry":"DeepEP_Final_Analysis_Report.html","visual":"DeepEP",
     "title":"DeepEP 深度设计分析 (Survey by AI)",
     "desc":"DeepSeek 开源 DeepEP 库三视角深度设计分析 · MoE 专家并行 AllToAll 通信 / NVLink+RDMA 双域融合 / Low-Latency 内核",
     "cat":"network","priority":"p0",
     "tags":["DeepEP","MoE","AllToAll","专家并行","NVLink","RDMA","通信库","DeepSeek"]},
    {"dst":"deepepv2","entry":"html/index.html","visual":"DGEMM",
     "title":"DeepGEMM & DeepEP 三向对比 (Survey by AI)",
     "desc":"36 篇深度分析报告（架构4篇 + 博客↔DeepGEMM 10篇 + 三向对比10篇 + DeepEP独立分析11篇）· 同步范式 Barrier→mbarrier FIFO / 通信模型 消息传递→Load-Store 对称直传",
     "cat":"chip","priority":"p0",
     "tags":["DeepGEMM","DeepEP","MoE","Mega MoE","SymmBuffer","Warp Specialization","NVLink","RDMA","对称内存","博客验证","三向对比"]},
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

# ──── Site identity ────
SITE = {
    "name": "backyes",
    "tagline": "Notes on AI infrastructure, chip architecture, and system design.",
    "title": "backyes — AI Infrastructure Insights",
    "description": "Research hub for AI infrastructure: chip architecture, inference systems, interconnects, and first-principles analysis.",
    "hero_eyebrow": "For AI Infrastructure Researchers",
    "hero_h1": "How to think clearly about AI system design, chip architecture, and infrastructure leverage.",
    "hero_copy": "Deep analysis of AI hardware, inference frameworks, and interconnect architectures — from first principles, with data to back it up.",
    "hero_cta_primary": "Read the essentials",
    "hero_cta_primary_url": "#essential-reads",
    "hero_cta_browse": "Browse by topic",
    "hero_cta_browse_url": "#paths",
    "newsletter_url": "#",
}

# ──── 解析手写文章 ────
def parse_post(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.S)
    if not m:
        return None
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

# ──── Markdown → HTML ────
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
        s = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1" style="max-width:100%;display:block;margin:1.5em auto">', s)
        s = re.sub(r'\$([\d,.]+)\$', r'<span class="key-num">\1</span>', s)
        s = re.sub(r'==([^=]+)==', r'<mark>\1</mark>', s)
        s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
        s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
        s = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', s)
        s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
        return s

    while i < len(lines):
        s = lines[i].strip()
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
        if '|' in s and s.startswith('|') and s.endswith('|') and i+1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i+1].strip()):
            header = [c.strip() for c in s.strip('|').split('|')]
            i += 2
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
        hm = re.match(r'^(#{1,3})\s+(.*)', s)
        if hm:
            close_ul()
            level = len(hm.group(1))
            title_text = hm.group(2)
            anchor_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', title_text)
            anchor_text = re.sub(r'=([^=]+)==', r'\1', anchor_text)
            anchor = re.sub(r'[^\w\- ]', '', anchor_text).strip().replace(' ', '-').lower()
            anchor = re.sub(r'-+', '-', anchor)[:50]
            out.append(f'<h{level} id="{anchor}">{inline(title_text)}</h{level}>')
            i += 1; continue
        if s.startswith('>'):
            close_ul()
            quote = s[1:].strip()
            out.append(f'<blockquote><p>{inline(quote)}</p></blockquote>')
            i += 1; continue
        if s == '---':
            close_ul()
            out.append('<hr>')
            i += 1; continue
        img_only = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)$', s)
        if img_only:
            close_ul()
            out.append(f'<img src="{img_only.group(2)}" alt="{img_only.group(1)}" style="max-width:100%;display:block;margin:1.5em auto">')
            i += 1; continue
        if s.startswith('- ') or s.startswith('* '):
            if not in_ul:
                out.append('<ul>'); in_ul = True
            out.append(f'<li>{inline(s[2:])}</li>')
            i += 1; continue
        if s == '':
            close_ul()
            i += 1; continue
        close_ul()
        out.append(f'<p>{inline(s)}</p>')
        i += 1
    close_ul()
    return '\n'.join(out)

# ──── 卡片视觉 ────
def visual_placeholder(visual, cat):
    shades = {
        "inference":"#2a2f38","network":"#2b303a","chip":"#2e333e",
        "cluster":"#2d323c","model":"#2c313b","storage":"#2a3038",
        "recsys":"#2b3138","conference":"#2d3038","space":"#2e3138",
    }
    c = shades.get(cat, "#2a2f38")
    return f'<div class="visual" style="background:{c}">{visual or ""}</div>'

# ════════════════════════════════════════════════════════════════════
#  VIKAS GOYLE STYLE GENERATORS
# ════════════════════════════════════════════════════════════════════

def _join(lines):
    return "\n".join(lines)

def gen_hero():
    return _join([
        '<section class="hero" aria-labelledby="hero-title">',
        '  <div class="hero-panel">',
        f'    <div class="eyebrow">{SITE["hero_eyebrow"]}</div>',
        f'    <h1 id="hero-title">{SITE["hero_h1"]}</h1>',
        f'    <p class="hero-copy">{SITE["hero_copy"]}</p>',
        '    <div class="hero-actions">',
        f'      <a class="button button-primary" href="{SITE["hero_cta_primary_url"]}">{SITE["hero_cta_primary"]}</a>',
        f'      <a class="button button-secondary" href="{SITE["hero_cta_browse_url"]}">{SITE["hero_cta_browse"]}</a>',
        '    </div>',
        '    <div class="hero-metrics">',
        '      <div class="metric">',
        '        <span class="metric-label">System Architecture</span>',
        '        <strong>Inference frameworks, KV-cache routing, P/D disaggregation, and distributed serving patterns.</strong>',
        '      </div>',
        '      <div class="metric">',
        '        <span class="metric-label">Chip &amp; Interconnect</span>',
        '        <strong>GPU microarchitecture, NVLink/NVL72, CXL memory fabric, and packaging technology deep dives.</strong>',
        '      </div>',
        '      <div class="metric">',
        '        <span class="metric-label">Industry Insight</span>',
        '        <strong>First-principles analysis of hyperscaler strategy, vendor roadmaps, and infrastructure economics.</strong>',
        '      </div>',
        '    </div>',
        '  </div>',
        '  <aside class="hero-aside" aria-label="How to use this site">',
        '    <p class="aside-kicker">Use This Page</p>',
        '    <h2 class="aside-title">Choose the lens that matches the problem in front of you.</h2>',
        '    <p class="aside-text">Start with the essentials if you are designing systems. Go to the survey section for AI-generated deep dives. Read posts for human-written analysis.</p>',
        '    <ul class="aside-list">',
        '      <li><span>For system designers</span>Begin with architecture essentials and chip microarchitecture series.</li>',
        '      <li><span>For researchers</span>Browse the AI survey section for paper-level deep dives with source tracing.</li>',
        '      <li><span>For industry watchers</span>Read posts for critical analysis of vendor strategy and market dynamics.</li>',
        '    </ul>',
        '  </aside>',
        '</section>',
    ])

def gen_signal_grid():
    cards = [
        ("AI system design, without hand-waving", "Inference architectures, KV-cache management, scheduling, and distributed serving — from first principles."),
        ("Chip & interconnect deep dives", "GPU microarchitecture, NVLink topology, CXL memory, and packaging technology analyzed quantitatively."),
        ("Infrastructure economics", "Cost models, utilization analysis, and TCO projections for AI hardware at scale."),
        ("Research synthesis", "Cross-paper analysis of MLSys/OSDI/SOSP findings, with critical evaluation and source tracing."),
    ]
    items = ""
    for title, desc in cards:
        items += f'<article class="signal-card"><strong>{title}</strong><p>{desc}</p></article>\n'
    return f'<section class="signal-grid" aria-label="What this site covers">\n{items}</section>'

def gen_featured_reports():
    p0_reports = [r for r in REPORTS if r["priority"] == "p0"][:6]
    cards = []
    for r in p0_reports:
        cat = CATS.get(r["cat"], CATS["inference"])
        cards.extend([
            '<article class="article-card">',
            f'  <div class="article-tag">{cat["label"]}</div>',
            f'  <h3>{r["title"]}</h3>',
            f'  <p>{r["desc"]}</p>',
            f'  <a class="article-link" href="{r["dst"]}/{r["entry"]}">Read the analysis →</a>',
            '</article>',
        ])
    return _join([
        '<section class="section-card" id="essential-reads" aria-labelledby="essential-reads-title">',
        '  <div class="section-heading">',
        '    <div><h2 id="essential-reads-title">Architecture &amp; AI Essentials</h2></div>',
        '    <p>Start here for the most in-depth analysis — chip architecture, inference systems, and communication libraries.</p>',
        '  </div>',
        '  <div class="featured-grid">',
        *("    " + l for l in cards),
        '  </div>',
        '</section>',
    ])

def gen_two_up(posts):
    post_items = []
    for p in posts[:6]:
        post_items.extend([
            '<article class="latest-card">',
            f'  <time datetime="{p["date"]}">{p["date"]}</time>',
            f'  <h3><a href="{p["url"]}">{p["title"]}</a></h3>',
            f'  <p>{p["excerpt"][:120]}\u2026</p>',
            '</article>',
        ])
    cat_links = []
    for cat_key, cat_info in CATS.items():
        cat_reports = [r for r in REPORTS if r["cat"] == cat_key]
        if cat_reports:
            cat_links.append(f'<a href="#survey">{cat_info["label"]} ({len(cat_reports)})</a>')
    return _join([
        '<section class="two-up">',
        '  <div class="section-card">',
        '    <div class="section-heading"><div><h2>Latest Writing</h2></div></div>',
        '    <div class="latest-grid" style="grid-template-columns:repeat(2,minmax(0,1fr));">',
        *("      " + l for l in post_items),
        '    </div>',
        '  </div>',
        '  <div class="section-card">',
        '    <div class="section-heading"><div><h2>Browse by Category</h2></div></div>',
        '    <div class="path-links">',
        *("      " + l for l in cat_links),
        '    </div>',
        '  </div>',
        '</section>',
    ])

def gen_paths_grid():
    paths = [
        ("I am designing an AI inference system", "Start with vLLM architecture, P/D disaggregation, and KV-cache routing patterns.", [("vLLM Analysis", "vllm_research/vllm_analysis/index.html"), ("P/D Separation", "pd-separation/report.html")]),
        ("I am evaluating chip architecture", "GPU microarchitecture, DeepEP communication, and packaging technology deep dives.", [("DeepEP Analysis", "deep-ep/DeepEP_Final_Analysis_Report.html"), ("DeepGEMM vs DeepEP", "deepepv2/html/index.html")]),
        ("I am researching interconnects", "NVLink, CXL, Sparse CLOS, and memory fabric architecture analysis.", [("AI Supernode Bus", "ai-supernode-bus/report.html"), ("HBM/CXL Memory", "hbm-cxl/report.html")]),
        ("I am tracking industry strategy", "Hyperscaler roadmaps, vendor positioning, and infrastructure economics.", [("AMD Full Stack", "amd-latest-tech-2026/index.html"), ("SpaceX Analysis", "spacex/太空经济与SpaceX深度分析报告.html")]),
    ]
    cards = []
    for title, desc, links in paths:
        links_html = []
        for label, url in links:
            links_html.append(f'<a href="{url}">{label} →</a>')
        cards.extend([
            '<article class="path-card">',
            f'  <h3>{title}</h3>',
            f'  <p>{desc}</p>',
            '  <div class="path-links">',
            *("    " + l for l in links_html),
            '  </div>',
            '</article>',
        ])
    return _join([
        '<section class="section-card" id="paths" aria-labelledby="paths-title">',
        '  <div class="section-heading">',
        '    <div><h2 id="paths-title">Browse by What You Need</h2></div>',
        '    <p>Organized by problem type so you can get to the right material without scanning the whole site.</p>',
        '  </div>',
        '  <div class="paths-grid">',
        *("    " + l for l in cards),
        '  </div>',
        '</section>',
    ])

def gen_latest_posts(posts):
    items = []
    for p in posts[:6]:
        items.extend([
            '<article class="latest-card">',
            f'  <time datetime="{p["date"]}">{p["date"]}</time>',
            f'  <h3><a href="{p["url"]}">{p["title"]}</a></h3>',
            f'  <p>{p["excerpt"][:140]}\u2026</p>',
            '</article>',
        ])
    return _join([
        '<section class="section-card" id="latest" aria-labelledby="latest-title">',
        '  <div class="section-heading">',
        '    <div><h2 id="latest-title">All Posts</h2></div>',
        '    <p>Human-written analysis \u2014 critical perspectives on AI infrastructure, chips, and system design.</p>',
        '  </div>',
        '  <div class="latest-grid">',
        *("    " + l for l in items),
        '  </div>',
        '</section>',
    ])

def gen_footer():
    return _join([
        '<section class="footer-grid" aria-label="About and site notes">',
        '  <article class="footer-card">',
        '    <h2>What You Will Find Here</h2>',
        '    <p>This collection focuses on AI infrastructure: chip architecture, inference systems, interconnect technology, and industry strategy. The through-line is first-principles thinking.</p>',
        '    <div class="footer-links">',
        '      <a href="posts.html">Explore all posts</a>',
        '      <a href="https://github.com/backyes" target="_blank" rel="noopener">GitHub</a>',
        '      <a href="tags.html">Browse tags</a>',
        '    </div>',
        '  </article>',
        '  <article class="footer-card">',
        '    <h3>Site Notes</h3>',
        '    <p class="note">This site is co-created with AI assistants.</p>',
        '    <p class="note" style="margin-top:14px;">These views are my own.</p>',
        '  </article>',
        '</section>',
    ])

def gen_cards():
    """Survey by AI report cards (astrofy style, adapted to light theme)"""
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

def gen_search_db(posts):
    items = []
    for r in REPORTS:
        items.append({"t": r["title"], "d": r["desc"], "u": f"{r['dst']}/{r['entry']}",
                      "tags": " ".join(r.get("tags",[]))})
    for p in posts:
        items.append({"t": p["title"], "d": p["excerpt"], "u": p["url"],
                      "tags": " ".join(p.get("tags",[]))})
    return json.dumps(items, ensure_ascii=False)

def gen_tag_cloud(posts):
    tag_count = {}
    for r in REPORTS:
        for t in r.get("tags", []):
            tag_count[t] = tag_count.get(t, 0) + 1
    for p in posts:
        for t in p.get("tags", []):
            tag_count[t] = tag_count.get(t, 0) + 1
    html_parts = []
    for tag, count in sorted(tag_count.items(), key=lambda x: -x[1])[:24]:
        html_parts.append(f'<a href="tags.html" class="tag-pill"># {tag} <span class="count">{count}</span></a>')
    return "\n".join(html_parts)

def gen_posts_list(posts, full=False):
    if not posts:
        return ('<div class="empty"><div class="emoji">✍️</div>'
                '<p>Posts 栏目已预留。<br>这里将放入我个人的观察、分析与观点。<br><br>'
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
    for tag in sorted(groups.keys(), key=lambda t: -len(groups[t])):
        items = groups[tag]
        out += f'<div class="tag-group" id="tag-{tag}" data-tag="{tag}">'
        out += f'<h3><span class="hash">#</span> {tag} <span class="cnt">({len(items)})</span></h3>'
        out += '<ul class="posts-list">'
        for it in items:
            out += f'''<li class="posts-item">
  <div class="posts-content"><h3><a href="{it['u']}">{it['t']}</a></h3><p class="posts-excerpt">{it['d']}</p></div>
</li>'''
        out += '</ul></div>'
    return out

def extract_toc(body):
    toc_items = []
    for line in body.split('\n'):
        if line.startswith('## '):
            title = line[3:].strip()
            anchor = re.sub(r'[^\w一-鿿\- ]', '', title).strip().replace(' ', '-').lower()
            anchor = re.sub(r'-+', '-', anchor)[:50]
            toc_items.append(f'<a href="#{anchor}" class="toc-h2">{title}</a>')
        elif line.startswith('### '):
            title = line[4:].strip()
            anchor = re.sub(r'[^\w一-鿿\- ]', '', title).strip().replace(' ', '-').lower()
            anchor = re.sub(r'-+', '-', anchor)[:50]
            toc_items.append(f'<a href="#{anchor}" class="toc-h3">{title}</a>')
    if not toc_items:
        return ''
    items_html = '\n'.join(f'<li>{item}</li>' for item in toc_items)
    return f'<div class="toc"><div class="toc-title">Contents</div><ul>{items_html}</ul></div>'

def gen_post_page(post):
    with open(post["source"], 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.match(r'^---\s*\n.*?\n---\s*\n(.*)$', content, re.S)
    body = m.group(1).strip() if m else content
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
    toc_html = extract_toc(body)
    tags_html = "".join(f'<span class="ptag">{t}</span>' for t in post.get("tags",[]))
    return f'''<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{post["title"]} · backyes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/css/main.css">
</head>
<body>
<div class="page-shell">
<header class="topbar">
  <div class="brand">
    <div class="brand-name"><a href="../">backyes</a></div>
    <div class="brand-tagline">{SITE["tagline"]}</div>
  </div>
  <nav class="topnav" aria-label="Primary">
    <a href="../#essential-reads">Essentials</a>
    <a href="../#survey">Survey by AI</a>
    <a href="../posts.html" class="active">Posts</a>
    <a href="../tags.html">Tags</a>
  </nav>
</header>
</div>
<div class="article-layout">
  <aside class="toc-sidebar">
    {toc_html}
  </aside>
  <article class="article">
    <a href="../posts.html" class="back">← 返回 Posts</a>
    <h1>{post["title"]}</h1>
    <div class="meta"><span>📅 {post["date"]}</span><div>{tags_html}</div></div>
    <div class="article-body">
{body_html}
    </div>
  </article>
</div>
<footer class="footer-grid" style="margin-top:60px">
  <div class="footer-card"><p class="note">© 2026 backyes · Human-driven, AI-amplified</p></div>
</footer>
</body>
</html>'''

# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    posts = load_posts()
    print(f"  解析到 {len(posts)} 篇手写文章")

    # Generate all sections
    hero = gen_hero()
    signal_grid = gen_signal_grid()
    featured_reports = gen_featured_reports()
    two_up = gen_two_up(posts)
    paths_grid = gen_paths_grid()
    latest_posts = gen_latest_posts(posts)
    footer = gen_footer()
    cards = gen_cards()
    search_db = gen_search_db(posts)
    tag_cloud = gen_tag_cloud(posts)
    posts_full = gen_posts_list(posts, full=True)
    posts_search_db = json.dumps(
        [{"t":p["title"],"d":p["excerpt"],"u":p["url"],"tags":" ".join(p.get("tags",[]))}
         for p in posts], ensure_ascii=False)
    tags_full = gen_tags_full(posts)
    tags_sidebar = gen_tags_sidebar(posts)

    # ──── index.html ────
    idx = open(os.path.join(REPO, 'index.html'), encoding='utf-8').read()
    idx = re.sub(r'<!--HERO-->.*?<!--/HERO-->', f'<!--HERO-->\n{hero}\n<!--/HERO-->', idx, flags=re.S)
    idx = re.sub(r'<!--SIGNAL_GRID-->.*?<!--/SIGNAL_GRID-->', f'<!--SIGNAL_GRID-->\n{signal_grid}\n<!--/SIGNAL_GRID-->', idx, flags=re.S)
    idx = re.sub(r'<!--FEATURED_REPORTS-->.*?<!--/FEATURED_REPORTS-->', f'<!--FEATURED_REPORTS-->\n{featured_reports}\n<!--/FEATURED_REPORTS-->', idx, flags=re.S)
    idx = re.sub(r'<!--TWO_UP-->.*?<!--/TWO_UP-->', f'<!--TWO_UP-->\n{two_up}\n<!--/TWO_UP-->', idx, flags=re.S)
    idx = re.sub(r'<!--PATHS_GRID-->.*?<!--/PATHS_GRID-->', f'<!--PATHS_GRID-->\n{paths_grid}\n<!--/PATHS_GRID-->', idx, flags=re.S)
    idx = re.sub(r'<!--LATEST_POSTS-->.*?<!--/LATEST_POSTS-->', f'<!--LATEST_POSTS-->\n{latest_posts}\n<!--/LATEST_POSTS-->', idx, flags=re.S)
    idx = re.sub(r'<!--FOOTER-->.*?<!--/FOOTER-->', f'<!--FOOTER-->\n{footer}\n<!--/FOOTER-->', idx, flags=re.S)
    idx = re.sub(r'<!--PROJECT_CARDS-->.*?<!--/PROJECT_CARDS-->', f'<!--PROJECT_CARDS-->\n{cards}<!--/PROJECT_CARDS-->', idx, flags=re.S)
    idx = re.sub(r'const SEARCH_DB=(?:\[.*?\]|<!--SEARCH_DB-->);', f'const SEARCH_DB={search_db};', idx, flags=re.S)
    idx = re.sub(r'<!--TAG_CLOUD-->.*?<!--/TAG_CLOUD-->', f'<!--TAG_CLOUD-->\n{tag_cloud}\n<!--/TAG_CLOUD-->', idx, flags=re.S)
    open(os.path.join(REPO,'index.html'),'w',encoding='utf-8').write(idx)
    print("  ✓ index.html 已生成")

    # ──── posts.html ────
    ph = open(os.path.join(REPO, 'posts.html'), encoding='utf-8').read()
    ph = re.sub(r'<!--POSTS_FULL-->.*?<!--/POSTS_FULL-->', f'<!--POSTS_FULL-->\n{posts_full}\n<!--/POSTS_FULL-->', ph, flags=re.S)
    ph = re.sub(r'const POSTS_DB=(?:\[.*?\]|<!--POSTS_SEARCH_DB-->);', f'const POSTS_DB={posts_search_db};', ph, flags=re.S)
    open(os.path.join(REPO,'posts.html'),'w',encoding='utf-8').write(ph)
    print("  ✓ posts.html 已生成")

    # ──── tags.html ────
    th = open(os.path.join(REPO, 'tags.html'), encoding='utf-8').read()
    th = re.sub(r'<!--TAGS_SIDEBAR-->.*?<!--/TAGS_SIDEBAR-->', f'<!--TAGS_SIDEBAR-->\n{tags_sidebar}\n<!--/TAGS_SIDEBAR-->', th, flags=re.S)
    th = re.sub(r'<!--TAGS_FULL-->.*?<!--/TAGS_FULL-->', f'<!--TAGS_FULL-->\n{tags_full}\n<!--/TAGS_FULL-->', th, flags=re.S)
    open(os.path.join(REPO,'tags.html'),'w',encoding='utf-8').write(th)
    print("  ✓ tags.html 已生成")

    # ──── Generate individual post pages ────
    for p in posts:
        page = gen_post_page(p)
        out = os.path.join(REPO, 'posts', f"{p['slug']}.html")
        open(out, 'w', encoding='utf-8').write(page)
        print(f"  ✓ 文章页: posts/{p['slug']}.html")

if __name__ == '__main__':
    main()
