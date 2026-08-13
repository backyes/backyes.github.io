# -*- coding: utf-8 -*-
"""Generate report scaffold (head, methodology, overview table, appendix) with a PROFILES marker."""
import json, re, html
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = json.load(open(ROOT / "digests.json", encoding="utf-8"))
S = json.load(open(ROOT / "summaries.json", encoding="utf-8"))

def esc(s): return html.escape(s or "")

def one_line(d):
    for sn in d.get("snippets", []):
        s = sn.get("s","").strip()
        if len(s) > 20:
            return s[:90]
    for sn in d.get("snippets", []):
        if sn.get("t"): return sn["t"][:90]
    return "—"

def funding_tag(d):
    txt = " ".join(sn.get("s","") for sn in d.get("snippets",[]))
    rounds = re.findall(r"(天使轮|Pre-[A-I]轮|[A-I]\+?轮|战略融资|C\+轮|IPO|上市辅导|战略投资|并购)", txt)
    amts = re.findall(r"(\d+(?:\.\d+)?\s*(?:亿|万)\s*元)", txt)
    val = re.findall(r"估值[^,，。;；\s]{0,8}(\d+(?:\.\d+)?\s*(?:亿|万)元|超?\d+(?:\.\d+)?\s*亿)", txt)
    parts = []
    if rounds: parts.append("/".join(dict.fromkeys(rounds)) )
    if amts: parts.append(dict.fromkeys(amts).popitem()[0] if amts else "")
    tag = " · ".join(p for p in parts if p)
    return tag or "—"

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, "PingFang SC", "Helvetica Neue", Arial, sans-serif;
       margin: 0; padding: 0; color: #1a1a1a; background: #f5f6f8; line-height: 1.6; }
.wrap { max-width: 900px; margin: 0 auto; padding: 12px 14px 60px; }
h1 { font-size: 20px; margin: 18px 0 4px; }
h2 { font-size: 17px; margin: 22px 0 8px; border-left: 4px solid #2b6cb0; padding-left: 8px; }
h3 { font-size: 15px; margin: 16px 0 4px; color: #1a365d; }
.meta { color: #666; font-size: 12px; margin-bottom: 12px; }
.card { background: #fff; border: 1px solid #e3e6ea; border-radius: 8px; padding: 12px 14px;
        margin: 10px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.card.startup { border-left: 4px solid #2f855a; }
.card.non { border-left: 4px solid #b7791f; opacity: 0.92; }
.tag { display:inline-block; font-size:11px; padding:1px 6px; border-radius:10px; margin-right:4px;
       background:#eef; color:#2b6cb0; }
.tag.f { background:#fef3c7; color:#92400e; }
.tag.cat { background:#e6fffa; color:#234e52; }
.tag.cat.non { background:#fdf6e3; color:#92400e; }
table { width:100%; border-collapse: collapse; font-size: 12px; background:#fff; }
th, td { border:1px solid #e3e6ea; padding:5px 6px; text-align:left; vertical-align:top; }
th { background:#edf2f7; position:sticky; top:0; }
tr:nth-child(even) td { background:#fafbfc; }
a { color:#2b6cb0; word-break: break-all; }
.src { font-size:11px; color:#555; margin-top:4px; }
.src a { margin-right:8px; }
blockquote { background:#f0f7ff; border-left:3px solid #90cdf4; margin:6px 0; padding:4px 8px;
             font-size:13px; color:#2c5282; }
.kv { font-size:13px; }
.kv b { color:#1a365d; }
ul { margin:4px 0 4px 18px; padding:0; }
.small { font-size:12px; color:#666; }
.toc a { display:inline-block; margin:2px 6px 2px 0; font-size:12px; }
@media(max-width:600px){ .wrap{padding:8px;} h1{font-size:18px;} table{font-size:11px;} }
"""

# overview table rows
rows = []
for i in sorted(D, key=int):
    d = D[i]
    cat = "非初创" if d["category"]=="non_startup" else "初创"
    rows.append(f"<tr><td>{i}</td><td>{esc(d['name'])}</td>"
                f"<td><span class='tag cat{' non' if d['category']=='non_startup' else ''}'>{cat}</span></td>"
                f"<td>{esc(one_line(d))}</td>"
                f"<td><span class='tag f'>{esc(funding_tag(d))}</span></td></tr>")
table_html = ("<table><thead><tr><th>#</th><th>企业名称</th><th>类别</th>"
              "<th>一句话定位（源自检索摘要）</th><th>关键融资标签</th></tr></thead><tbody>"
              + "".join(rows) + "</tbody></table>")

# appendix: all URLs
all_urls = {}
for i in sorted(S, key=int):
    d = S[i]
    urls = []
    for s in d.get("searches", []):
        for r in s.get("results", []):
            u = r.get("url","")
            if u.startswith("http"): urls.append(u)
    for p in d.get("pages", []):
        if p.get("url","").startswith("http"): urls.append(p["url"])
    all_urls[i] = {"name": d["name"], "urls": list(dict.fromkeys(urls))}

appx_rows = []
total_u = 0
for i in sorted(all_urls, key=int):
    a = all_urls[i]
    links = "".join(f'<a href="{esc(u)}">{esc(u[:70])}</a><br>' for u in a["urls"][:18])
    total_u += len(a["urls"])
    appx_rows.append(f"<tr><td>{i}</td><td>{esc(a['name'])}</td><td>{links}</td></tr>")
appx = (f"<p class='small'>共归档 {total_u} 条溯源链接，原始 SERP/页面 HTML 见 <code>research/raw/</code>，"
        "结构化记录见 <code>research/startups/</code>。</p>"
        "<table><thead><tr><th>#</th><th>企业</th><th>检索阅读过的关键链接（溯源）</th></tr></thead><tbody>"
        + "".join(appx_rows) + "</tbody></table>")

n_start = sum(1 for d in D.values() if d["category"]=="startup")
n_non = sum(1 for d in D.values() if d["category"]=="non_startup")

doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>联盟首批意向成员 · 初创企业调研报告</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>联盟首批意向成员 — 初创企业资本与产品调研报告</h1>
<div class="meta">生成日期：2026-07-13 ｜ 数据源：搜狗搜索（Playwright Chromium 实抓）｜
样本：96 家成员，剔除 7 家研究所/实验室，剩余 89 家企业（初创 {n_start} + 非初创 {n_non}）｜
归档：research/summaries.json、research/raw/*.html</div>

<div class="card">
<h3>调研方法与口径</h3>
<ul>
<li><b>范围</b>：从 data.md 96 家"联盟首批意向成员"中，剔除高校与研究所/实验室
（03 中国电子技术标准化研究院、04 电子六所、08 航天二院706所、10 中国电科43所、
13 航天二院空间总体部、15 雄安空天信息研究院、17 零碳信息通信网络联合实验室），剩 89 家企业。</li>
<li><b>分类</b>："初创/民营"为创业型公司（含独角兽）；"非初创"为大型央企/国企子公司、上市公司、
科技园区开发主体等（简要列示，不作为深度画像重点）。</li>
<li><b>字段</b>：企业愿景、创始人/核心团队、融资历程与估值、核心产品、营收/商业化进展。</li>
<li><b>工具链</b>：Playwright 驱动真实 Chromium 抓取搜狗 SERP + Top 结果页正文，原始 HTML 全量落盘存档；
本报告每条关键结论后附可点击溯源链接。</li>
<li><b>局限</b>：营收数据多数初创企业未公开，以"商业化进展/订单/产能"替代；部分信息以检索摘要为准，
已在附录列出全部阅读链接供人工复核。</li>
</ul>
</div>

<h2>一、总览表（89 家）</h2>
{table_html}

<h2>二、初创/民营企业详细画像</h2>
<p class='small'>按编号顺序，每家含：愿景定位 · 创始人 · 融资与估值 · 核心产品 · 商业化/营收进展 · 溯源链接。</p>
<!--PROFILES-->

<h2>三、非初创企业（大型央企/国企/上市/园区）简要</h2>
<!--NONSTARTUP-->

<h2>附录 A：调研阅读过的全部网站链接（溯源）</h2>
{appx}

<h2>附录 B：关键过程与归档说明</h2>
<div class="card small">
<ul>
<li><b>工具</b>：uv 0.8.15 + playwright 1.61 + chromium 149；脚本 research/fetch_companies.py（Sogou SERP + 结果页抓取，原始 HTML 落盘 research/raw/）。</li>
<li><b>过程</b>：4 进程并行抓取 89 家 × 2 查询 = 178 次 SERP 检索 + 111 个结果页正文；2 家因并发限流补抓成功。</li>
<li><b>归档</b>：research/raw/{'{编号}'}_serp{'{0,1}'}.html（SERP 原始 HTML）、
research/raw/{'{编号}'}_page_*.html（结果页原始 HTML）、research/startups/{'{编号}'}.json（结构化记录）、
research/summaries.json（全量汇总）、research/digests.json（压缩摘要）、research/logs/（运行日志）。</li>
<li><b>降本</b>：全部抓取由本地脚本完成，零 LLM token；LLM 仅用于读取压缩摘要撰写画像。</li>
</ul>
</div>

</div></body></html>
"""
Path(ROOT / "report.html").write_text(doc, encoding="utf-8")
print("scaffold written:", ROOT/"report.html", "size:", len(doc))
print("total archived URLs:", total_u)
