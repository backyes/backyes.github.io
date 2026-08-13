from playwright.sync_api import sync_playwright
import sys, os, glob

url = sys.argv[1]
max_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url, wait_until='networkidle', timeout=30000)
    page.wait_for_timeout(4000)
    text = page.evaluate('() => { var m = document.querySelector("main"); return m ? m.innerText : document.body.innerText; }')
    sys.stdout.write(text[:max_chars])
    browser.close()
