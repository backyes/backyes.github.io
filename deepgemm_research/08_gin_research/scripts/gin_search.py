#!/usr/bin/env python3
"""
NVIDIA GIN (GPU Initiated Networking) Research Script
Downloads key pages and papers about GIN architecture.
"""
import asyncio
import os
import json
from playwright.async_api import async_playwright

OUTPUT_DIR = "/Users/backyes/work/claude_workspace/deepgemm_research/docs/08_gin_research/raw_pages"

# Key URLs to search and download
SEARCH_QUERIES = [
    "NVIDIA GIN GPU Initiated Networking NCCL",
    "GPU-initiated RDMA without CPU NVIDIA",
    "GPUDirect Async GPU-centric fabric",
    "NVIDIA Magnum IO GPU initiated networking",
    "NCCL Gin QP queue pair GPU networking",
    "GPU initiated networking DeepEP symmetric memory",
]

# Direct URLs to try accessing
DIRECT_URLS = [
    "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/gpu_direct.html",
    "https://developer.nvidia.com/gpudirect",
    "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/",
    "https://arxiv.org/abs/2104.03517",  # GPU-initiated networking paper
    "https://arxiv.org/abs/2309.00533",  # DeepEP paper
    "https://arxiv.org/abs/2409.02001",  # Mega MoE / DeepGEMM
    "https://arxiv.org/abs/2505.09372",  # DeepSeek MoE
    "https://developer.nvidia.com/blog/gpudirect-async/",
    "https://developer.nvidia.com/blog/gpudirect-storage/",
    "https://developer.nvidia.com/blog/improving-network-communications-using-gpu-initiated-networking/",
    "https://arxiv.org/abs/2002.03309",  # GPU-initiated networking (Caltech)
    "https://arxiv.org/abs/2009.10839",  # GPU-centric communication
    "https://ieeexplore.ieee.org/document/9139891",  # GPU Initiated Networking
]

async def search_and_capture(browser, query, idx):
    """Search and capture results."""
    print(f"\n=== Searching: {query} ===")
    context = browser.contexts[0]
    page = await context.new_page()
    
    try:
        # Use Google search
        await page.goto(f"https://www.google.com/search?q={query.replace(' ', '+')}&num=15", 
                       wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)
        
        # Extract search results
        results = await page.evaluate('''() => {
            const items = [];
            document.querySelectorAll('div.g, div[data-sokoban-container]').forEach(el => {
                const titleEl = el.querySelector('h3');
                const linkEl = el.querySelector('a[href]');
                const snippetEl = el.querySelector('[data-sncf], .VwiC3b, [style="-webkit-line-clamp:2"]');
                if (titleEl && linkEl) {
                    items.push({
                        title: titleEl.textContent,
                        url: linkEl.href,
                        snippet: snippetEl ? snippetEl.textContent : ''
                    });
                }
            });
            return items.slice(0, 10);
        }''')
        
        # Save results
        safe_name = query.replace(' ', '_')[:50]
        with open(f"{OUTPUT_DIR}/search_{idx:02d}_{safe_name}.json", 'w') as f:
            json.dump({"query": query, "results": results}, f, indent=2)
        
        print(f"  Found {len(results)} results")
        for r in results[:5]:
            print(f"  - {r.get('title', 'N/A')[:60]}")
            print(f"    {r.get('url', 'N/A')[:80]}")
        
        return results
    except Exception as e:
        print(f"  Error: {e}")
        return []
    finally:
        await page.close()

async def download_page(browser, url, idx):
    """Download a single page."""
    print(f"\n=== Downloading: {url[:80]} ===")
    context = browser.contexts[0]
    page = await context.new_page()
    
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)
        
        content = await page.content()
        title = await page.title()
        
        # Save HTML
        safe_name = url.replace('https://', '').replace('http://', '').replace('/', '_')[:80]
        filepath = f"{OUTPUT_DIR}/page_{idx:02d}_{safe_name}.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Also extract text
        text = await page.evaluate('''() => {
            return document.body ? document.body.innerText : '';
        }''')
        textpath = f"{OUTPUT_DIR}/page_{idx:02d}_{safe_name}.txt"
        with open(textpath, 'w', encoding='utf-8') as f:
            f.write(text[:50000])  # Limit text
        
        print(f"  Saved: {filepath}")
        print(f"  Title: {title}")
        print(f"  Content length: {len(content)} bytes")
        
        return {"url": url, "title": title, "filepath": filepath, "textpath": textpath}
    except Exception as e:
        print(f"  Error: {e}")
        return {"url": url, "error": str(e)}
    finally:
        await page.close()

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        
        # Phase 1: Search queries
        print("=" * 60)
        print("PHASE 1: Search Queries")
        print("=" * 60)
        all_results = []
        for idx, query in enumerate(SEARCH_QUERIES):
            results = await search_and_capture(browser, query, idx)
            all_results.extend(results)
        
        # Save aggregated search results
        with open(f"{OUTPUT_DIR}/all_search_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Phase 2: Download key direct URLs
        print("\n" + "=" * 60)
        print("PHASE 2: Download Direct URLs")
        print("=" * 60)
        
        key_urls = [
            "https://developer.nvidia.com/blog/gpudirect-async/",
            "https://developer.nvidia.com/blog/improving-network-communications-using-gpu-initiated-networking/",
            "https://developer.nvidia.com/gpudirect",
            "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/",
            "https://arxiv.org/abs/2002.03309",
            "https://arxiv.org/abs/2009.10839",
        ]
        
        for idx, url in enumerate(key_urls):
            await download_page(browser, url, idx)
        
        await browser.close()
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Files saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
