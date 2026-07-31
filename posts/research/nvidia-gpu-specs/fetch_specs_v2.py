#!/usr/bin/env python3
"""Fetch NVIDIA GPU specs - direct NVIDIA pages + Bing search."""
import json, time, os
from playwright.sync_api import sync_playwright

OUT_DIR = "/Users/backyes/work/claude_workspace/backyes-site-toolkit/backyes.github.io/posts/research/nvidia-gpu-specs"
PROXY = {"server": "http://127.0.0.1:7897"}

# Key NVIDIA official pages to scrape directly
NVIDIA_URLS = [
    # H200
    "https://www.nvidia.com/en-us/data-center/h200/",
    "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet",
    # B200 / B300
    "https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/",
    "https://resources.nvidia.com/en-us-tensor-core/nvidia-b200-tensor-core-gpu-datasheet",
    # Rubin
    "https://www.nvidia.com/en-us/data-center/technologies/rubin/",
    # General
    "https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/",
]

BING_QUERIES = [
    "NVIDIA H200 specs memory bandwidth TFLOPS TDP site:nvidia.com",
    "NVIDIA B200 specs memory bandwidth NVLink TFLOPS TDP site:nvidia.com",
    "NVIDIA B300 specs memory bandwidth unified B200 TFLOPS",
    "NVIDIA Rubin NVL72 HBM4 specs memory bandwidth TFLOPS",
    "NVIDIA H100 H200 B200 B300 Rubin GPU comparison specifications",
]

def visit_page(browser, url, label=""):
    """Visit a single page and extract text content."""
    print(f"\n  Visiting: {url}")
    context = browser.new_context(proxy=PROXY)
    page = context.new_page()
    
    try:
        page.goto(url, timeout=45000, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle", timeout=20000)
        time.sleep(3)
        
        text = page.inner_text("body")[:12000]
        title = page.title()
        
        print(f"    OK: {title[:80]} ({len(text)} chars)")
        
        result = {
            "url": url,
            "title": title,
            "label": label,
            "success": True,
            "is_official": "nvidia.com" in url,
            "content": text
        }
    except Exception as e:
        print(f"    FAIL: {e}")
        result = {"url": url, "label": label, "success": False, "error": str(e), "content": ""}
    
    context.close()
    return result

def bing_search(browser, query, idx):
    """Search Bing and capture results."""
    print(f"\n{'='*60}")
    print(f"Bing Search {idx+1}: {query}")
    print(f"{'='*60}")
    
    context = browser.new_context(proxy=PROXY)
    page = context.new_page()
    
    results = {"query": query, "search_results": [], "pages": []}
    
    try:
        page.goto(f"https://www.bing.com/search?q={query.replace(' ', '+')}&count=10", timeout=30000)
        page.wait_for_load_state("networkidle", timeout=15000)
        time.sleep(2)
        
        # Bing search results
        links = page.query_selector_all("li.b_algo h2 a")
        if not links:
            links = page.query_selector_by_id("b_results")
            links = page.query_selector_all("#b_results a[href]")
        
        for link in links[:10]:
            href = link.get_attribute("href")
            title = link.inner_text()[:100] if link.inner_text() else ""
            if href and "bing" not in href and "microsoft" not in href:
                results["search_results"].append({"title": title, "url": href})
        
        print(f"  Found {len(results['search_results'])} results")
        for r in results["search_results"][:5]:
            print(f"    - {r['title'][:60]} -> {r['url'][:80]}")
    except Exception as e:
        print(f"  Search error: {e}")
    
    # Visit top pages
    visited = 0
    for r in results["search_results"]:
        if visited >= 3:
            break
        url = r["url"]
        is_official = "nvidia.com" in url
        
        try:
            print(f"  Visiting: {url[:80]}")
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle", timeout=15000)
            time.sleep(2)
            
            text = page.inner_text("body")[:8000]
            title = page.title()
            
            results["pages"].append({
                "url": url,
                "title": title,
                "is_official": is_official,
                "content": text
            })
            visited += 1
            print(f"    Got {len(text)} chars")
        except Exception as e:
            print(f"    Error: {e}")
    
    context.close()
    return results

def main():
    all_nvidia_pages = []
    all_bing_results = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # First visit NVIDIA official pages directly
        print("\n" + "="*60)
        print("PHASE 1: Visiting NVIDIA Official Pages Directly")
        print("="*60)
        
        for url in NVIDIA_URLS:
            label = url.split("/")[-2] if url.endswith("/") else url.split("/")[-1]
            result = visit_page(browser, url, label)
            all_nvidia_pages.append(result)
        
        # Then do Bing searches
        print("\n" + "="*60)
        print("PHASE 2: Bing Searches")
        print("="*60)
        
        for idx, query in enumerate(BING_QUERIES):
            result = bing_search(browser, query, idx)
            all_bing_results.append(result)
        
        browser.close()
    
    # Save results
    with open(os.path.join(OUT_DIR, "nvidia_official_pages.json"), "w") as f:
        json.dump(all_nvidia_pages, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(OUT_DIR, "bing_search_results.json"), "w") as f:
        json.dump(all_bing_results, f, indent=2, ensure_ascii=False)
    
    # Save readable text
    with open(os.path.join(OUT_DIR, "all_page_contents.txt"), "w") as f:
        f.write("="*80 + "\n")
        f.write("NVIDIA OFFICIAL PAGES\n")
        f.write("="*80 + "\n")
        for page in all_nvidia_pages:
            f.write(f"\n{'='*80}\n")
            f.write(f"URL: {page['url']}\n")
            f.write(f"Title: {page.get('title', 'N/A')}\n")
            f.write(f"Success: {page['success']}\n")
            f.write(f"{'='*80}\n")
            f.write(page.get("content", "")[:6000])
            f.write("\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("BING SEARCH RESULTS\n")
        f.write("="*80 + "\n")
        for r in all_bing_results:
            f.write(f"\n{'='*80}\n")
            f.write(f"QUERY: {r['query']}\n")
            f.write(f"{'='*80}\n")
            for page in r.get("pages", []):
                f.write(f"\n--- {page['title']} ---\n")
                f.write(f"URL: {page['url']}\n")
                f.write(f"Official: {page['is_official']}\n")
                f.write("-"*40 + "\n")
                f.write(page.get("content", "")[:4000])
                f.write("\n")

    print(f"\n\nDone! Files saved to {OUT_DIR}/")
    print(f"  - nvidia_official_pages.json")
    print(f"  - bing_search_results.json")
    print(f"  - all_page_contents.txt")

if __name__ == "__main__":
    main()
