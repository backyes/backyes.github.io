#!/usr/bin/env python3
import json, time, os
from playwright.sync_api import sync_playwright

OUT_DIR = '/Users/backyes/work/claude_workspace/backyes-site-toolkit/backyes.github.io/posts/research/nvidia-gpu-specs'
PROXY = {'server': 'http://127.0.0.1:7897'}

URLS = [
    ('WP Hopper Mobile', 'https://en.m.wikipedia.org/wiki/Nvidia_Hopper_(microarchitecture)'),
    ('WP Blackwell Mobile', 'https://en.m.wikipedia.org/wiki/Nvidia_Blackwell_(microarchitecture)'),
    ('NVIDIA B200 Product', 'https://www.nvidia.com/en-us/data-center/b200/'),
    ('NVIDIA GB300 NVL72', 'https://www.nvidia.com/en-us/data-center/gb300-nvl72/'),
    ('NVIDIA Newsroom Blackwell', 'https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-ai-era'),
    ('Verge Blackwell', 'https://www.theverge.com/2024/3/18/24105281/nvidia-blackwell-b200-gpu-announced-specs'),
    ('Tom Hardware B200', 'https://www.tomshardware.com/pc-components/gpus/nvidia-b200-blackwell-gpu-specs'),
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
    with open(os.path.join(OUT_DIR, 'spec_aggregators_v4.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, 'aggregator_contents_v4.txt'), 'w') as f:
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
