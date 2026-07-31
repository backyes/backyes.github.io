#!/usr/bin/env python3
"""Fetch GPU specs from Wikipedia and tech aggregators."""
import json, time, os
from playwright.sync_api import sync_playwright

OUT_DIR = "/Users/backyes/work/claude_workspace/backyes-site-toolkit/backyes.github.io/posts/research/nvidia-gpu-specs"
PROXY = {"server": "http://127.0.0.1:7897"}

URLS = [
    ("Wikipedia H100", "https://en.wikipedia.org/wiki/Nvidia_Hopper_(microarchitecture)"),
    ("Wikipedia B200", "https://en.wikipedia.org/wiki/Nvidia_Blackwell_(microarchitecture)"),
    ("Wikipedia H200", "https://en.wikipedia.org/wiki/Nvidia_Hopper_(microarchitecture)"),
    ("TechPowerUp H200", "https://www.techpowerup.com/gpu-specs/nvidia-h200-gpu-10866"),
    ("TechPowerUp B200", "https://www.techpowerup.com/gpu-specs/nvidia-b200-gpu-10950"),
    ("SemiAnalysis Blackwell", "https://semianalysis.com/2024/03/18/nvidia-gtc-2024-blackwell-architecture/"),
    ("NVIDIA GB200 Datasheet PDF", "https://resources.nvidia.com/en-us-tensor-core/nvidia-gb200-nvl72-datasheet"),
    ("NVIDIA B300", "https://www.nvidia.com/en-us/data-center/hgx-b300/"),
    ("NVIDIA Blackwell Ultra", "https://www.nvidia.com/en-us/data-center/technologies/blackwell-ultra/"),
    ("ServeTheHome B200", "https://www.servethehome.com/nvidia-b200-blackwell-gpu-specs/"),
]

def visit_page(browser, label, url):
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
    
    with open(os.path.join(OUT_DIR, "spec_aggregators.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(OUT_DIR, "aggregator_contents.txt"), "w") as f:
        for r in all_results:
            f.write(f"\n{'='*80}\n")
            f.write(f"LABEL: {r['label']}\n")
            f.write(f"URL: {r['url']}\n")
            f.write(f"TITLE: {r.get('title', 'N/A')}\n")
            f.write(f"SUCCESS: {r['success']}\n")
            f.write(f"{'='*80}\n")
            f.write(r.get("content", "")[:8000])
            f.write("\n\n")

    print(f"\n\nDone! Saved to {OUT_DIR}/spec_aggregators.json")

if __name__ == "__main__":
    main()
