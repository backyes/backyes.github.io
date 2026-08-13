# -*- coding: utf-8 -*-
"""Quick test which search engine returns parseable results for a headless browser."""
import re, time
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
Q = "北京天兵科技 融资 创始人"

ENGINES = {
    "bing_intl": f"https://www.bing.com/search?q={Q}&mkt=en-US&setlang=en-US",
    "ddg_html": f"https://html.duckduckgo.com/html/?q={Q}",
    "sogou": f"https://www.sogou.com/web?query={Q}",
    "baidu": f"https://www.baidu.com/s?wd={Q}",
    "so360": f"https://www.so.com/s?q={Q}",
}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(user_agent=UA, locale="zh-CN", viewport={"width":1366,"height":900})
    ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    page = ctx.new_page()
    for name, url in ENGINES.items():
        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=20000)
            time.sleep(1.2)
            html = page.content()
            soup = BeautifulSoup(html, "lxml")
            links = []
            for a in soup.find_all("a", href=True):
                h = a["href"]
                if h.startswith("http") and "bing.com" not in h and "baidu.com" not in h \
                   and "sogou.com" not in h and "so.com" not in h and "duckduckgo" not in h \
                   and "microsoft" not in h and "go.microsoft" not in h:
                    t = re.sub(r"\s+"," ",a.get_text()).strip()[:60]
                    if t: links.append((t,h))
            cap = "captcha" in html.lower()
            print(f"=== {name} status={resp.status if resp else '?'} len={len(html)} captcha={cap} ext_links={len(links)}")
            for t,h in links[:5]:
                print(f"   {t} -> {h}")
        except Exception as e:
            print(f"=== {name} ERROR {e}")
    browser.close()
