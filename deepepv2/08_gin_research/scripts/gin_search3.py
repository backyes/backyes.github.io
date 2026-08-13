#!/usr/bin/env python3
"""Search for GPU-initiated networking papers and content."""
import asyncio
import os
import json
from playwright.async_api import async_playwright

OUTPUT_DIR = "/Users/backyes/work/claude_workspace/deepgemm_research/docs/08_gin_research/raw_pages"

QUERIES = [
    "GPU-initiated networking RDMA GPUDirect async",
    "GPU-centric fabric communication NVIDIA",
    "GPU initiated RDMA without CPU involvement architecture",
    "NVIDIA Magnum IO GPU initiated networking architecture",
    "NCCL device initiated communication GIN kernel",
    "GPUDirect RDMA GPU-initiated networking paper",
    "GPU SM-free communication RDMA network",
]

async def search_google(browser, query, idx):
    print(f"\n=== Search: {query} ===")
    page = await browser.new_page()
    try:
        await page.goto(f"https://www.google.com/search?q={query.replace(' ','+')}&num=20", 
                       wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)
        results = await page.evaluate('''() => {
            const items = [];
            document.querySelectorAll('div.g').forEach(el => {
                const t = el.querySelector('h3');
                const l = el.querySelector('a[href]');
                const s = el.querySelector('[data-sncf], .VwiC3b, [style*="line-clamp"]');
                if (t && l) items.push({title:t.textContent, url:l.href, snippet:s?s.textContent:''});
            });
            return items.slice(0,15);
        }''')
        safe = query.replace(' ','_')[:40]
        with open(f"{OUTPUT_DIR}/srch_{idx:02d}_{safe}.json", 'w') as f:
            json.dump({"query":query,"results":results}, f, indent=2)
        for r in results[:8]:
            print(f"  {r['title'][:70]}")
            print(f"    {r['url'][:90]}")
        return results
    except Exception as e:
        print(f"  Error: {e}")
        return []
    finally:
        await page.close()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        all_r = []
        for idx, q in enumerate(QUERIES):
            r = await search_google(browser, q, idx)
            all_r.extend(r)
        await browser.close()
    with open(f"{OUTPUT_DIR}/all_searches.json", 'w') as f:
        json.dump(all_r, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
