#!/usr/bin/env python3
"""Fetch NVIDIA datasheets and detailed spec pages."""
import json, time, os
from playwright.sync_api import sync_playwright

OUT_DIR = "/Users/backyes/work/claude_workspace/backyes-site-toolkit/backyes.github.io/posts/research/nvidia-gpu-specs"
PROXY = {"server": "http://127.0.0.1:7897"}

# Datasheets and spec pages
URLS = [
    # H200 Datasheet
    ("H200 Datasheet", "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet"),
    # B200 Datasheet
    ("B200 Datasheet", "https://resources.nvidia.com/en-us-tensor-core/nvidia-b200-tensor-core-gpu-datasheet"),
    # Blackwell Technical Brief
    ("Blackwell Technical Brief", "https://resources.nvidia.com/en-us-tensor-core/nvidia-blackwell-architecture-technical-brief"),
    # GB200 NVL72
    ("GB200 NVL72", "https://www.nvidia.com/en-us/data-center/gb200-nvl72/"),
    # HGX B300
    ("HGX B300", "https://www.nvidia.com/en-us/data-center/hgx-b300/"),
    # Rubin NVL72 details
    ("Rubin NVL72", "https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/"),
    # H100 Datasheet (for comparison)
    ("H100 Datasheet", "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet"),
    # NVIDIA Data Center GPU Line Card
    ("GPU Line Card", "https://www.nvidia.com/en-us/data-center/data-center-gpu-line-card/"),
]

def visit_page(browser, label, url):
    """Visit a page and extract all text."""
    print(f"\n  [{label}] {url}")
    context = browser.new_context(proxy=PROXY)
    page = context.new_page()
    
    try:
        page.goto(url, timeout=45000, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle", timeout=20000)
        time.sleep(3)
        
        text = page.inner_text("body")[:15000]
        title = page.title()
        
        print(f"    OK: {title[:80]} ({len(text)} chars)")
        result = {"label": label, "url": url, "title": title, "success": True, "content": text}
    except Exception as e:
        print(f"    FAIL: {e}")
        result = {"label": label, "url": url, "success": False, "error": str(e), "content": ""}
    
    context.close()
    return result

def main():
    all_results = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        for label, url in URLS:
            result = visit_page(browser, label, url)
            all_results.append(result)
        
        browser.close()
    
    # Save
    with open(os.path.join(OUT_DIR, "datasheets.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(OUT_DIR, "datasheet_contents.txt"), "w") as f:
        for r in all_results:
            f.write(f"\n{'='*80}\n")
            f.write(f"LABEL: {r['label']}\n")
            f.write(f"URL: {r['url']}\n")
            f.write(f"TITLE: {r.get('title', 'N/A')}\n")
            f.write(f"SUCCESS: {r['success']}\n")
            f.write(f"{'='*80}\n")
            f.write(r.get("content", "")[:8000])
            f.write("\n\n")

    print(f"\n\nDone! Saved to {OUT_DIR}/datasheets.json")

if __name__ == "__main__":
    main()
