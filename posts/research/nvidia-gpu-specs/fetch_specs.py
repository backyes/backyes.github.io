#!/usr/bin/env python3
"""Fetch NVIDIA GPU specs using playwright - all searches in one browser session."""
import json, time, os
from playwright.sync_api import sync_playwright

OUT_DIR = "/Users/backyes/work/claude_workspace/backyes-site-toolkit/backyes.github.io/posts/research/nvidia-gpu-specs"
PROXY = {"server": "http://127.0.0.1:7897"}

QUERIES = [
    "NVIDIA H200 Tensor Core GPU datasheet specifications memory bandwidth TFLOPS",
    "NVIDIA B200 GPU specifications memory bandwidth NVLink TFLOPS",
    "NVIDIA B300 GPU specifications unified with B200 memory bandwidth",
    "NVIDIA Rubin NVL72 GPU HBM4 specifications memory bandwidth TFLOPS",
    "NVIDIA GPU comparison H100 H200 B200 B300 Rubin specifications",
]

def search_and_capture(browser, query, idx):
    """Do a Google search, capture results, and visit top pages."""
    print(f"\n{'='*60}")
    print(f"Search {idx+1}: {query}")
    print(f"{'='*60}")
    
    context = browser.new_context(proxy=PROXY)
    page = context.new_page()
    
    results = {"query": query, "search_results": [], "pages": []}
    
    # Google search
    try:
        page.goto(f"https://www.google.com/search?q={query.replace(' ', '+')}&num=10", timeout=30000)
        page.wait_for_load_state("networkidle", timeout=15000)
        time.sleep(2)
        
        # Extract search result links
        links = page.query_selector_all("div#search a[href]")
        for link in links[:10]:
            href = link.get_attribute("href")
            title = link.inner_text()[:100] if link.inner_text() else ""
            if href and "google" not in href and "youtube" not in href:
                results["search_results"].append({"title": title, "url": href})
        
        print(f"  Found {len(results['search_results'])} search results")
        for r in results["search_results"][:5]:
            print(f"    - {r['title'][:60]} -> {r['url'][:80]}")
    except Exception as e:
        print(f"  Search error: {e}")
    
    # Visit top relevant pages (official NVIDIA first)
    visited = 0
    for r in results["search_results"]:
        if visited >= 4:
            break
        url = r["url"]
        # Prioritize official NVIDIA pages
        is_official = any(d in url for d in ["nvidia.com", "developer.nvidia.com", "docs.nvidia.com"])
        if not is_official and visited >= 2:
            continue
            
        try:
            print(f"  Visiting: {url[:80]}")
            page.goto(url, timeout=30000)
            page.wait_for_load_state("networkidle", timeout=15000)
            time.sleep(2)
            
            # Get page content
            text = page.inner_text("body")[:8000]
            title = page.title()
            
            results["pages"].append({
                "url": url,
                "title": title,
                "is_official": is_official,
                "content_length": len(text),
                "content_preview": text[:3000]
            })
            visited += 1
            print(f"    Got {len(text)} chars from {title[:60]}")
        except Exception as e:
            print(f"    Error visiting {url[:60]}: {e}")
    
    context.close()
    return results

def main():
    all_results = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        for idx, query in enumerate(QUERIES):
            result = search_and_capture(browser, query, idx)
            all_results.append(result)
        
        browser.close()
    
    # Save all results
    out_path = os.path.join(OUT_DIR, "raw_search_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Also extract just the page contents for easier reading
    with open(os.path.join(OUT_DIR, "page_contents.txt"), "w") as f:
        for r in all_results:
            f.write(f"\n{'='*80}\n")
            f.write(f"QUERY: {r['query']}\n")
            f.write(f"{'='*80}\n")
            for page in r.get("pages", []):
                f.write(f"\n--- PAGE: {page['title']} ---\n")
                f.write(f"URL: {page['url']}\n")
                f.write(f"OFFICIAL: {page['is_official']}\n")
                f.write(f"{'-'*40}\n")
                f.write(page.get("content_preview", "")[:3000])
                f.write("\n")

    print(f"\n\nDone! Results saved to {OUT_DIR}/")
    print(f"  - raw_search_results.json (structured)")
    print(f"  - page_contents.txt (readable)")

if __name__ == "__main__":
    main()
