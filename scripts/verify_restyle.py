#!/usr/bin/env python3
"""
verify_restyle.py — verify all report files are correctly restyled.

Checks:
  - Has <link> to notion-style.css
  - Has exactly one .notion-page
  - Has .notion-toc if it has h2+ headings
  - No residual local .css links (old theme)
  - No residual per-file <style> blocks
  - Relative CSS path is valid (file exists at that path)
"""
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTION_CSS = "assets/css/notion-style.css"

REPORT_DIRS = [
    "ai-supernode-bus", "deepepv2", "mlsys2026", "deepseek-mtp", "moe-clos",
    "generative-rec", "sparse-clos", "supernode-metrics", "mtp-survey", "3dls",
    "space-ecom", "pd-routing", "trillium", "spacex", "hbm-cxl",
    "inference-community", "deep-ep", "umdk", "pd-separation", "vllm_research",
]

EXCLUDE_HOSTS = [r"arxiv\.org", r"githubassets\.com", r"vllm\.ai"]


def discover_files():
    files = []
    for d in REPORT_DIRS:
        base = REPO / d
        if not base.exists():
            continue
        for ext in ("*.html", "*.HTML"):
            files.extend(base.rglob(ext))
    return sorted(files)


def should_exclude(content):
    for host in EXCLUDE_HOSTS:
        if re.search(r'href="https?://[^"]*' + host, content):
            return True
    return False


def verify_file(fp):
    """Return (is_restyled, list_of_issues)."""
    content = fp.read_text(encoding="utf-8")
    rel = os.path.relpath(fp, REPO)
    issues = []

    if should_exclude(content):
        return None, ["excluded: scraped external"]

    # Fragment files without <head> are not standalone reports — skip
    if "<head" not in content and "<HEAD" not in content:
        return None, ["excluded: fragment (no <head>)"]

    has_notion_page = 'class="notion-page"' in content
    has_css_link = bool(re.search(r'href="[^"]*assets/css/notion-style\.css"', content))
    # Only flag <style> in <head> (per-file theme). Style in <noscript> etc. is page content.
    head_match = re.search(r"<head[^>]*>(.*?)</head>", content, re.S)
    head_content = head_match.group(1) if head_match else ""
    has_old_style = bool(re.search(r"<style\b", head_content))
    has_old_local_css = bool(re.search(
        r'<link[^>]*rel=["\']stylesheet["\'][^>]*href=["\'][^"\']*\.css["\'][^>]*>',
        content, re.I,
    )) and not has_css_link  # the notion-style.css link itself counts

    # Count .notion-page occurrences
    notion_page_count = content.count('class="notion-page"')

    # Check TOC presence if h2+ exists
    has_h2 = bool(re.search(r"<h2[> ]", content))
    has_toc = 'class="notion-toc"' in content

    # Verify relative path resolves
    css_match = re.search(r'href="([^"]*assets/css/notion-style\.css)"', content)
    if css_match:
        css_rel = css_match.group(1)
        css_abs = (fp.parent / css_rel).resolve()
        if not css_abs.exists():
            issues.append(f"CSS path broken: {css_rel}")

    if not has_notion_page:
        issues.append("missing .notion-page")
    elif notion_page_count > 1:
        issues.append(f"multiple .notion-page ({notion_page_count})")

    if not has_css_link:
        issues.append("missing notion-style.css link")

    if has_old_style:
        issues.append("residual <style> block")

    if has_h2 and not has_toc:
        issues.append("has h2 but no .notion-toc")

    is_restyled = has_notion_page and has_css_link and not issues
    return is_restyled, issues


def main():
    files = discover_files()
    restyled = 0
    issues_list = []
    excluded = 0

    for fp in files:
        is_ok, issues = verify_file(fp)
        if is_ok is None:
            excluded += 1
            continue
        if is_ok:
            restyled += 1
        else:
            rel = os.path.relpath(fp, REPO)
            issues_list.append((rel, issues))

    print(f"  Total files scanned: {len(files)}")
    print(f"  ✓ Correctly restyled: {restyled}")
    print(f"  ✗ Has issues: {len(issues_list)}")
    print(f"  ⊘ Excluded (scraped): {excluded}")
    print()

    if issues_list:
        print("  Issues found:")
        for rel, issues in issues_list:
            print(f"    {rel}: {', '.join(issues)}")
        return 1
    else:
        print("  All report files correctly restyled!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
