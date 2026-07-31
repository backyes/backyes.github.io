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
    {"dst":"nvidia-specs-research","entry":"nvidia-specs-report.html","visual":"NVIDIA",
     "title":"NVIDIA 产品全规格深度调研",
     "desc":"NVIDIA GPU / CPU / 网络全产品线规格深度调研报告 · 数据中心 GPU (H100/B200/GB200) / Grace CPU / Spectrum-X 网络 / DGX 系统",
     "cat":"chip","priority":"p0",
     "tags":["NVIDIA","GPU","H100","B200","GB200","Grace","Spectrum-X","数据中心"]},
    {"dst":"nvidia-agent-reports","entry":"index.html","visual":"Agent",
     "title":"NVIDIA Agent 深度调研 (8 模块)",
     "desc":"AI Agent 自主生成的 8 篇深度调研 — GPU 微架构 / 互联总线 / 内存 / 网络 / DGX 系统 / 边缘计算 / 学术论文 / 数据手册",
     "cat":"chip","priority":"p0",
     "tags":["NVIDIA","Agent","GPU","微架构","互联","调研报告"]},
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
    {"dst":"supernode-metrics","entry":"supernode_metrics_report.html","visual":"Metrics",
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
    "tagline": "Thoughts on AI infrastructure, chip architecture, and system design.",
    "title": "backyes — AI Infrastructure Insights",
    "description": "Research hub for AI infrastructure: chip architecture, inference systems, interconnects, and first-principles analysis.",
    "hero_eyebrow": "First-Principles Analysis of AI Infrastructure",
    "hero_h1": "Thoughts on AI infrastructure — from first principles, with data.",
    "hero_copy": "Hand-written deep dives from first principles — on GPU dataflow, NVLink topology, CXL memory, KV-cache economics, and the architecture decisions that shape AI infrastructure.",
    "hero_cta_primary": "Read the posts",
    "hero_cta_primary_url": "posts.html",
    "hero_cta_browse": "AI survey reports",
    "hero_cta_browse_url": "#survey",
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
        '    <h2 class="aside-title">Two kinds of content, one through-line: first-principles thinking.</h2>',
        '    <p class="aside-text">Posts are hand-written deep dives. Survey reports are AI-generated with full source tracing. Both aim to make tradeoffs explicit and claims quantifiable.</p>',
        '    <ul class="aside-list">',
        '      <li><span>Latest post</span><a href="posts/deepseek-mtp-chip-system-impact-en.html">DeepSeek MTP: Structural Impact on Chips, Systems, and Interconnects</a> — how MTP restructures the compute-memory-interconnect triangle.</li>',
        '      <li><span>Latest survey</span><a href="deepepv2/html/index.html">DeepGEMM &amp; DeepEP 三向对比</a> — 36 reports, 3-way comparison with source-level tracing.</li>',
        '      <li><span>Browse all</span><a href="posts.html">13 posts</a> on GPU, memory, CXL, MTP, and infrastructure economics · <a href="#survey">23 AI survey reports</a> with paper-level depth.</li>',
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

def gen_recent_posts_band(posts):
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
        '<section class="section-card">',
        '  <div class="section-heading">',
        '    <div><h2>Recent Posts</h2></div>',
        '    <a href="posts.html" style="color:var(--blue);font-size:.9rem;font-weight:700">View all posts \u2192</a>',
        '  </div>',
        '  <div class="latest-grid">',
        *("    " + l for l in items),
        '  </div>',
        '</section>',
    ])

def gen_paths_grid():
    return ""  # removed from homepage

def gen_latest_posts(posts):
    return ""  # removed from homepage, nav already has Posts
def gen_footer():
    return _join([
        '<section class="section-card" style="text-align:center;padding:28px">',
        '  <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:14px;margin-bottom:14px">',
        '    <a href="posts.html" style="font-weight:700">Posts</a>',
        '    <a href="#survey" style="font-weight:700">Survey by AI</a>',
        '    <a href="tags.html" style="font-weight:700">Tags</a>',
        '    <a href="https://github.com/backyes" target="_blank" rel="noopener">GitHub</a>',
        '    <a href="https://www.zhihu.com/people/nono-nono-66" target="_blank" rel="noopener">知乎</a>',
        '    <a href="https://x.com/backyes1" target="_blank" rel="noopener">X</a>',
        '    <a href="https://space.bilibili.com/327400087" target="_blank" rel="noopener">Bilibili</a>',
        '    <a href="https://www.linkedin.com/in/yanfei-wang-5081b4126/" target="_blank" rel="noopener">LinkedIn</a>',
        '  </div>',
        '  <p style="color:var(--ink-mute);font-size:.85rem;margin:0">\u00a9 2026 backyes \u00b7 Human-driven, AI-amplified \u00b7 Built with Claude Code</p>',
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
    return '<div class="grid" id="grid">' + cards + '</div>'

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
    <div class="brand-name">
      <a href="../">backyes</a>
      <a class="icon-link" href="https://github.com/backyes" target="_blank" rel="noopener noreferrer" aria-label="GitHub"><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" fill="currentColor"/></svg></a>
      <a class="icon-link" href="https://x.com/backyes1" target="_blank" rel="noopener noreferrer" aria-label="X (Twitter)"><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" fill="currentColor"/></svg></a>
      <a class="icon-link" href="https://www.zhihu.com/people/nono-nono-66" target="_blank" rel="noopener noreferrer" aria-label="知乎"><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M5.721 0C2.251 0 0 2.25 0 5.719V18.28C0 21.751 2.252 24 5.721 24h12.56C21.751 24 24 21.75 24 18.281V5.72C24 2.249 21.75 0 18.281 0zm1.964 4.078h6.29c.09 1.017.197 2.034.312 3.051h-3.282c.09 1.45.197 2.758.32 3.926h3.572c.09.563.135 1.218.135 1.964h-4.11c.09 1.45.162 2.758.216 3.926h-1.746c-.09-1.017-.162-2.325-.216-3.926H7.09c-.09-.91-.135-1.565-.135-1.964h4.11c-.09-1.017-.162-2.325-.216-3.926H7.53c-.09-1.017-.162-2.034-.216-3.051z" fill="currentColor"/></svg></a>
      <a class="icon-link" href="https://space.bilibili.com/327400087" target="_blank" rel="noopener noreferrer" aria-label="Bilibili"><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M17.813 4.653h.854c1.51.054 2.769.578 3.773 1.574 1.004.995 1.524 2.249 1.56 3.76v7.36c-.036 1.51-.556 2.769-1.56 3.773s-2.262 1.524-3.773 1.56H5.333c-1.51-.036-2.769-.556-3.773-1.56S.036 18.858 0 17.347v-7.36c.036-1.511.556-2.765 1.56-3.76 1.004-.996 2.262-1.52 3.773-1.574h.774l-1.174-1.12a1.234 1.234 0 0 1-.373-.906c0-.356.124-.658.373-.907l.027-.027c.267-.249.573-.373.92-.373.347 0 .653.124.92.373L9.653 4.44c.071.071.134.142.187.213h4.267a.836.836 0 0 1 .16-.213l2.853-2.747c.267-.249.573-.373.92-.373.347 0 .662.151.929.4.267.249.391.551.391.907 0 .355-.124.657-.373.906L17.813 4.653zM5.333 7.24c-.746.018-1.373.276-1.88.773-.506.498-.769 1.13-.786 1.894v7.52c.017.764.28 1.395.786 1.893.507.498 1.134.756 1.88.773h13.334c.746-.017 1.373-.275 1.88-.773.506-.498.769-1.129.786-1.893v-7.52c-.017-.765-.28-1.396-.786-1.894-.507-.497-1.134-.755-1.88-.773H5.333zM8 11.2c-.355 0-.658.124-.907.373-.248.249-.373.556-.373.92v1.814c0 .355.125.657.373.906.249.249.552.373.907.373.354 0 .657-.124.906-.373.249-.249.373-.551.373-.906v-1.814c0-.364-.124-.671-.373-.92A1.236 1.236 0 0 0 8 11.2zm5.334 0c-.355 0-.658.124-.907.373-.248.249-.373.556-.373.92v1.814c0 .355.125.657.373.906.249.249.552.373.907.373.354 0 .657-.124.906-.373.249-.249.373-.551.373-.906v-1.814c0-.364-.124-.671-.373-.92a1.236 1.236 0 0 0-.906-.373z" fill="currentColor"/></svg></a>
      <a class="icon-link" href="https://www.linkedin.com/in/yanfei-wang-5081b4126/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn"><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" fill="currentColor"/></svg></a>
    </div>
    <div class="brand-tagline">{SITE["tagline"]}</div>
  </div>
  <nav class="topnav" aria-label="Primary">
    <a href="../posts.html" class="nav-accent">Posts</a>
    <a href="../#survey">Survey by AI</a>
    <a href="../#essential-reads" class="nav-hide-mobile">Essentials</a>
    <a href="../tags.html" class="nav-hide-mobile">Tags</a>
    <button class="nav-search-btn" onclick="document.getElementById('search-overlay').classList.add('open');document.getElementById('search-input').focus()" aria-label="Search" title="Search (⌘K)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg></button>
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
</div><!-- /page-shell -->
<!--SEARCH_OVERLAY-->
<div class="search-overlay" id="search-overlay">
  <div class="search-box">
    <input type="text" id="search-input" placeholder="Search posts and reports..." autocomplete="off">
    <div class="search-results" id="search-results"></div>
    <div class="search-hint">Press Esc to close · ⌘K to open</div>
  </div>
</div>
<!--/SEARCH_OVERLAY-->
<script>
const overlay = document.getElementById('search-overlay');
const input = document.getElementById('search-input');
const results = document.getElementById('search-results');
document.addEventListener('keydown', e => {{
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {{ e.preventDefault(); overlay.classList.add('open'); input.focus(); }}
  if (e.key === 'Escape') overlay.classList.remove('open');
}});
overlay.addEventListener('click', e => {{ if (e.target === overlay) overlay.classList.remove('open'); }});
input.addEventListener('input', () => {{
  const q = input.value.toLowerCase().trim();
  if (!q) {{ results.innerHTML = ''; return; }}
  const items = [
    {{t:"When Do We Need a Global Address Service",u:"../posts/when-do-we-need-global-address-service.html"}},
    {{t:"DeepEP First Principles",u:"../posts/zz-gpu-microarchitecture-deep-ep-first-principles.html"}},
    {{t:"GPU Microarchitecture: Warp Specialization",u:"../posts/z-gpu-microarchitecture-warp-specialization.html"}},
    {{t:"GPU Microarchitecture: On-chip Dataflow",u:"../posts/z-gpu-microarchitecture-from-workload-perspective-1.html"}},
    {{t:"AI Supernode Unified Addressing",u:"../posts/ai-supernode-unified-addressing-first-principles.html"}},
    {{t:"Server-Side CPU Agentic Revolution",u:"../posts/server-side-cpu-agentic-revolution.html"}},
    {{t:"Why I'm Bullish on CXL",u:"../posts/why-im-bullish-on-cxl.html"}},
    {{t:"Kimi3 Architecture Analysis",u:"../posts/kimi3-architecture-analysis.html"}},
    {{t:"Kimi3 Cost Efficiency",u:"../posts/kimi3-cost-efficiency.html"}},
    {{t:"Million-Token Storage vs Compute",u:"../posts/million-seq-storage-vs-compute.html"}},
    {{t:"Google Lustre for KV Cache",u:"../posts/google-rapid-storage-not-for-kvcache.html"}},
    {{t:"Memory Wall Model Comparison",u:"../posts/z-ai-memory-wall-model-architecture-comparison.html"}},
  ];
  const matches = items.filter(it => it.t.toLowerCase().includes(q)).slice(0, 12);
  results.innerHTML = matches.map(it => `<a href="${{it.u}}"><h4>${{it.t}}</h4></a>`).join('') || '<div style="padding:18px;color:#666">No results</div>';
}});
</script>
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
    recent_posts_band = gen_recent_posts_band(posts)
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
    idx = re.sub(r'<!--TWO_UP-->.*?<!--/TWO_UP-->', f'<!--TWO_UP-->\n{recent_posts_band}\n<!--/TWO_UP-->', idx, flags=re.S)
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
