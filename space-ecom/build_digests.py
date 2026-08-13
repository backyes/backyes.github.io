# -*- coding: utf-8 -*-
"""Merge all per-company JSONs into summaries.json and build compact digests.json."""
import json, glob, re, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
files = sorted(glob.glob(str(ROOT / "startups" / "*.json")))
allc = {}
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    idx = d["idx"]
    allc[idx] = d

# merged summaries.json (full)
summ = {str(i): {"name": d["name"], "searches": d.get("searches", []),
                 "pages": [{"url": p["url"], "title": p["title"], "excerpt": p["excerpt"][:1800]}
                           for p in d.get("pages", [])],
                 "errors": d.get("errors", [])}
        for i, d in sorted(allc.items())}
json.dump(summ, open(ROOT / "summaries.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# keyword relevance scoring
KW = ["融资","轮","投资","创始人","创办","CEO","董事长","成立","创立","估值","独角兽","营收",
      "收入","产品","火箭","卫星","星座","芯片","GPU","DPU","CPU","发动机","推进","激光",
      "光通信","通信","存储","大模型","AI","算力","载荷","遥感","测控","在轨","上市","辅导",
      "注册资本","万元","亿元","A轮","B轮","C轮","天使","战略融资"]

def score(s):
    s_low = s
    return sum(s_low.count(k) for k in KW)

# known large/SOE/listed (non-startup) — verified via data; will refine in report
NON_STARTUP = {2,5,6,9,11,12,14,16,21,22,23,52,96}

digests = {}
for i, d in sorted(allc.items()):
    snips = []
    for s in d.get("searches", []):
        for r in s.get("results", []):
            snips.append(r)
    # dedupe by url, score, keep top 10
    seen = set(); uniq = []
    for r in snips:
        u = r.get("url","")
        if u in seen: continue
        seen.add(u); uniq.append(r)
    uniq.sort(key=lambda r: score((r.get("title","")+r.get("snippet",""))), reverse=True)
    top = uniq[:10]
    # best page excerpt
    pages = d.get("pages", [])
    pages.sort(key=lambda p: score(p.get("excerpt","")+p.get("title","")), reverse=True)
    best_page = pages[0] if pages else None
    digests[str(i)] = {
        "name": d["name"],
        "category": "non_startup" if i in NON_STARTUP else "startup",
        "snippets": [{"t": r.get("title","")[:120], "u": r.get("url",""),
                      "s": r.get("snippet","")[:260]} for r in top],
        "page": ({"u": best_page["url"], "t": best_page.get("title",""),
                  "x": best_page["excerpt"][:1100]} if best_page else None),
        "nsnip": len(snips), "npage": len(pages),
        "errors": d.get("errors", [])[:2],
    }
json.dump(digests, open(ROOT / "digests.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"merged {len(allc)} companies")
print(f"digests.json written. non_startup={sum(1 for v in digests.values() if v['category']=='non_startup')}, startup={sum(1 for v in digests.values() if v['category']=='startup')}")
# quick stats
for i,d in sorted(digests.items()):
    pass
print("sample (28):", json.dumps(digests["28"], ensure_ascii=False)[:600])
