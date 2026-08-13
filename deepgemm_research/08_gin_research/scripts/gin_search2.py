#!/usr/bin/env python3
"""Second phase: targeted downloads for GIN-specific pages."""
import asyncio
import os
import json
from playwright.async_api import async_playwright

OUTPUT_DIR = "/Users/backyes/work/claude_workspace/deepgemm_research/docs/08_gin_research/raw_pages"

URLS = [
    # NCCL docs - GIN device kernel
    "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/device_api.html",
    "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/device_api.html#gin",
    # NCCL device-initiated communication
    "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/device_api.html#device-initiated-communication",
    # GPUDirect RDMA docs
    "https://docs.nvidia.com/cuda/gpudirect-rdma/index.html",
    # CUDA toolkit docs
    "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html",
    # ArXiv papers on GPU-initiated networking
    "https://arxiv.org/abs/2104.03517",
    "https://arxiv.org/abs/2111.03744",
    "https://arxiv.org/abs/2201.04049",
    "https://arxiv.org/abs/2302.01934",
    "https://arxiv.org/abs/2309.00533",  # DeepEP
    "https://arxiv.org/abs/2409.02001",  # Mega MoE
    "https://arxiv.org/abs/2505.09372",  # DeepSeek
    # GPU-centric fabric
    "https://arxiv.org/abs/2109.02536",
    "https://arxiv.org/abs/2205.09822",
    # NCCL4Py
    "https://github.com/NVIDIA/nccl/blob/master/src/include/device_api.h",
    # NVIDIA blog on Magnum IO
    "https://developer.nvidia.com/blog/enabling-gpu-direct-async-for-io-communication/",
]

async def download_page(browser, url, idx):
    print(f"\n=== [{idx}] {url[:90]} ===")
    context = browser.contexts[0]
    page = await context.new_page()
    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2500)
        content = await page.content()
        title = await page.title()
        text = await page.evaluate('() => document.body ? document.body.innerText : ""')
        
        safe = url.replace('https://','').replace('http://','').replace('/','_')[:70]
        hp = f"{OUTPUT_DIR}/p2_{idx:02d}_{safe}.html"
        tp = f"{OUTPUT_DIR}/p2_{idx:02d}_{safe}.txt"
        with open(hp, 'w', encoding='utf-8') as f:
            f.write(content)
        with open(tp, 'w', encoding='utf-8') as f:
            f.write(text[:60000])
        print(f"  OK: {len(content)}B, title={title[:60]}")
        return {"url": url, "title": title, "status": "ok", "size": len(content)}
    except Exception as e:
        print(f"  ERR: {e}")
        return {"url": url, "error": str(e), "status": "error"}
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
        results = []
        for idx, url in enumerate(URLS):
            r = await download_page(browser, url, idx)
            results.append(r)
        await browser.close()
    with open(f"{OUTPUT_DIR}/phase2_manifest.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
