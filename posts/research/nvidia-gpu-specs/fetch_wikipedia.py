#!/usr/bin/env python3
"""Fetch GPU specs from Wikipedia REST API and other sources."""
import json, time, os
from playwright.sync_api import sync_playwright

OUT_DIR = '/Users/backyes/work/claude_workspace/backyes-site-toolkit/backyes.github.io/posts/research/nvidia-gpu-specs'
PROXY = {'server': 'http://127.0.0.1:7897'}

URLS = [
    ('WP REST Hopper', 'https://en.wikipedia.org/api/rest_v1/page/summary/Nvidia_Hopper_(microarchitecture)'),
    ('WP REST Blackwell', 'https://en.wikipedia.org/api/rest_v1/page/summary/Nvidia_Blackwell_(microarchitecture)'),
    ('WP REST H100', 'https://en.wikipedia.org/api/rest_v1/page/summary/Nvidia_H100'),
    ('WP REST H200', 'https://en.wikipedia.org/api/rest_v1/page/summary/Nvidia_H200'),
    ('NVIDIA HGX B200', 'https://www.nvidia.com/en-us/data-center/hgx-b200/'),
    ('NVIDIA Blackwell Ultra Datasheet', 'https://resources.nvidia.com/en-us-tensor-core/nvidia-blackwell-ultra-datasheet'),
    ('NVIDIA Rubin Datasheet', 'https://resources.nvidia.com/en-us-tensor-core/nvidia-rubin-datasheet'),
    ('AnandTech Blackwell', 'https://www.anandtech.com/show/21049/nvidia-announces-h200-tensor-core-gpu'),
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
    with open(os.path.join(OUT_DIR, 'wikipedia_rest.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, 'wikipedia_contents.txt'), 'w') as f:
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
