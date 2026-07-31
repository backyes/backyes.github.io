#!/usr/bin/env python3
"""Fetch final missing specs - Wikipedia full content + tech sites."""
import json, time, os
from playwright.sync_api import sync_playwright

OUT_DIR = '/Users/backyes/work/claude_workspace/backyes-site-toolkit/backyes.github.io/posts/research/nvidia-gpu-specs'
PROXY = {'server': 'http://127.0.0.1:7897'}

URLS = [
    # Wikipedia full mobile sections
    ('WP Hopper Full', 'https://en.wikipedia.org/api/rest_v1/page/mobile-sections/Hopper_(microarchitecture)'),
    ('WP Blackwell Full', 'https://en.wikipedia.org/api/rest_v1/page/mobile-sections/Blackwell_(microarchitecture)'),
    # NVIDIA product pages
    ('NVIDIA HGX B200 v2', 'https://www.nvidia.com/en-us/data-center/products/hgx-b200/'),
    ('NVIDIA GB200 Superchip', 'https://www.nvidia.com/en-us/data-center/gb200/'),
    ('NVIDIA Grace Blackwell', 'https://www.nvidia.com/en-us/data-center/grace-blackwell/'),
    # Tech sites
    ('TechPowerUp H100', 'https://www.techpowerup.com/gpu-specs/nvidia-h100-gpu-10665'),
    ('ServetheHome GB200', 'https://www.servethehome.com/nvidia-gb200-nvl72/'),
    ('NextPlatform B200', 'https://www.nextplatform.com/2024/03/18/nvidia-blackwell-b200-gpu/'),
    # Press releases
    ('NVIDIA GTC 2024 Blackwell PR', 'https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-ai-era'),
    ('NVIDIA GTC 2025 Rubin PR', 'https://nvidianews.nvidia.com/news/nvidia-vera-rubin-platform-arrives'),
]

def visit_page(browser, label, url):
    print(f'  [{label}] {url[:80]}')
    context = browser.new_context(proxy=PROXY)
    page = context.new_page()
    try:
        page.goto(url, timeout=45000, wait_until='domcontentloaded')
        page.wait_for_load_state('networkidle', timeout=20000)
        time.sleep(3)
        text = page.inner_text('body')[:15000]
        title = page.title()
        print(f'    OK: {title[:80]} ({len(text)} chars)')
        result = {'label': label, 'url': url, 'title': title, 'success': True, 'content': text}
    except Exception as e:
        print(f'    FAIL: {e}')
        result = {'label': label, 'url': url, 'success': False, 'error': str(e), 'content': ''}
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
    with open(os.path.join(OUT_DIR, 'final_specs.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, 'final_contents.txt'), 'w') as f:
        for r in all_results:
            sep = '=' * 80
            f.write(f'\n{sep}\n')
            f.write(f"LABEL: {r['label']}\n")
            f.write(f"URL: {r['url']}\n")
            f.write(f"TITLE: {r.get('title', 'N/A')}\n")
            f.write(f"SUCCESS: {r['success']}\n")
            f.write(f'{sep}\n')
            f.write(r.get('content', '')[:8000])
            f.write('\n\n')
    print('Done!')

if __name__ == '__main__':
    main()
