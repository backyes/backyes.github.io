# -*- coding: utf-8 -*-
"""
Batch company research via Playwright Chromium + Sogou (page.goto navigation).
- Real chromium engine (playwright), Sogou SERP (no captcha, rich snippets).
- Raw SERP HTML archived to raw/ for traceability.
- Compact structured JSON per company + combined summaries.json.
- Retry + throttling. Token-free fetching (all local).
Usage: .venv/bin/python research/fetch_companies.py [--resume] [--only=1,2,3]
"""
import json, re, time, sys, os, traceback, random
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SUM = ROOT / "startups"
LOG = ROOT / "logs"
for d in (RAW, SUM, LOG):
    d.mkdir(parents=True, exist_ok=True)

SKIP = {3, 4, 8, 10, 13, 15, 17}  # research institutes / labs / state factories
COMPANIES = {}
with open(ROOT.parent / "data.md", encoding="utf-8") as f:
    for line in f:
        m = re.match(r"\s*\*\s*(\d{2})\s+(.+?)\s*$", line)
        if m:
            idx = int(m.group(1))
            name = m.group(2).strip()
            if idx not in SKIP:
                COMPANIES[idx] = name

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

def clean(s, limit=400):
    return re.sub(r"\s+", " ", s or "").strip()[:limit]

def parse_sogou(html):
    """Parse Sogou SERP: div.vrwrap -> title, real url (data-url), snippet (fz-mid)."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for e in soup.select("div.vrwrap"):
        a = e.select_one("h3 a") or e.select_one("a[href]")
        snip_el = (e.select_one("div.fz-mid") or e.select_one(".space-txt")
                   or e.select_one(".str-text-info"))
        data_url_el = e.select_one("[data-url]")
        cite_el = e.select_one(".citeLinkClass span")
        real = ""
        if data_url_el and data_url_el.get("data-url"):
            real = data_url_el["data-url"]
        elif cite_el:
            real = clean(cite_el.get_text(), 120)
        title = clean(a.get_text(), 130) if a else ""
        href = a.get("href") if a else ""
        url = real or href
        if url and not url.startswith("http"):
            url = "https://www.sogou.com" + url if url.startswith("/") else ""
        snip = clean(snip_el.get_text(), 300) if snip_el else ""
        if title or snip:
            out.append({"title": title, "url": url, "snippet": snip})
    return out

def extract_page_text(html, limit=6000):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return clean(main.get_text(" "), limit)

def goto_with_retry(page, url, tries=3):
    for t in range(tries):
        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=22000)
            time.sleep(1.0 + random.random()*0.5)
            html = page.content()
            if "vrwrap" in html or len(html) > 30000:
                return html, (resp.status if resp else 0)
            # maybe captcha / empty, wait and retry
            time.sleep(1.5)
        except Exception as e:
            time.sleep(2.0)
    return html if 'html' in dir() else None, "retry_fail"

def research_one(page, idx, name):
    rec = {"idx": idx, "name": name, "searches": [], "pages": [], "errors": []}
    queries = [f"{name} 融资 创始人 产品 营收", f"{name} 公司简介 官网 业务范围"]
    candidates = []
    seen = set()
    for qi, q in enumerate(queries):
        url = f"https://www.sogou.com/web?query={q}"
        html, status = goto_with_retry(page, url)
        if not html:
            rec["errors"].append(f"serp_fail:{q}:{status}")
            continue
        (RAW / f"{idx:02d}_serp{qi}.html").write_text(html, encoding="utf-8")
        results = parse_sogou(html)
        rec["searches"].append({"q": q, "n": len(results),
                                "results": [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
                                            for r in results[:8]]})
        for r in results:
            if r["url"] and r["url"] not in seen and r["url"].startswith("http"):
                seen.add(r["url"])
                candidates.append(r)
    # Fetch top result pages for depth (prioritize company/news/baike/36kr)
    def prio(u):
        ul = u.lower()
        s = 0
        for k in ("36kr.com","itjuzi.com","qcc.com","qichamao","tianyancha","baike.baidu",
                  "leiphone.com","qbitai.com","pedaily.cn","sina.com","sohu.com","163.com",
                  "chinanews","toutiao.com","mp.weixin","company","about"):
            if k in ul: s += 2
        if "sogou.com" in ul or "baidu.com/s" in ul: s -= 5
        return s
    candidates.sort(key=lambda r: prio(r["url"]), reverse=True)
    for r in candidates[:2]:
        u = r["url"]
        try:
            resp = page.goto(u, wait_until="domcontentloaded", timeout=18000)
            time.sleep(0.8)
            html = page.content()
            text = extract_page_text(html, 5000)
            slug = re.sub(r"[^a-z0-9]+", "_", u[:70].lower()).strip("_")[:55]
            (RAW / f"{idx:02d}_page_{slug}.html").write_text(html, encoding="utf-8")
            rec["pages"].append({"url": u, "title": r["title"],
                                 "status": (resp.status if resp else 0),
                                 "excerpt": text[:3500]})
        except Exception as e:
            rec["errors"].append(f"page_fail:{u}:{str(e)[:60]}")
        time.sleep(0.8)
    return rec

def main():
    resume = "--resume" in sys.argv
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = set(int(x) for x in a[7:].split(",") if x)
    items = [(i, n) for i, n in COMPANIES.items() if (not only or i in only)]
    items.sort()
    N = len(items)
    print(f"[start] {N} companies (engine=Sogou, chromium)", flush=True)
    alls = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=UA, locale="zh-CN",
                                  viewport={"width": 1366, "height": 900})
        ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        page = ctx.new_page()
        # warm up sogou cookies
        try:
            page.goto("https://www.sogou.com/", wait_until="domcontentloaded", timeout=15000)
            time.sleep(1)
        except Exception:
            pass
        for k, (idx, name) in enumerate(items, 1):
            out_json = SUM / f"{idx:02d}.json"
            if resume and out_json.exists():
                alls[idx] = json.loads(out_json.read_text(encoding="utf-8"))
                print(f"[{k}/{N}] {idx:02d} {name} -> cached", flush=True)
                continue
            try:
                rec = research_one(page, idx, name)
                out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                alls[idx] = rec
                ns = sum(s["n"] for s in rec["searches"])
                print(f"[{k}/{N}] {idx:02d} {name} | serp={ns} pages={len(rec['pages'])} err={len(rec['errors'])}", flush=True)
            except Exception as e:
                print(f"[{k}/{N}] {idx:02d} {name} ERROR {e}", flush=True)
                (LOG / f"{idx:02d}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            time.sleep(1.0 + random.random())
        browser.close()
    combined = {str(i): {"name": r["name"],
                         "searches": r.get("searches", []),
                         "pages": [{"url": p["url"], "title": p["title"], "excerpt": p["excerpt"][:1800]}
                                   for p in r.get("pages", [])],
                         "errors": r.get("errors", [])}
                for i, r in sorted(alls.items())}
    (ROOT / "summaries.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] summaries.json with {len(combined)} companies", flush=True)

if __name__ == "__main__":
    main()
