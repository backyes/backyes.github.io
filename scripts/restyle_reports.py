#!/usr/bin/env python3
"""
restyle_reports.py — batch-apply Notion style to all Survey-by-AI report HTML files.

What it does per file:
  1. HEAD: remove per-file <style> blocks + local .css links → inject shared notion-style.css
  2. BODY: normalize wrapper (.container/.wrap/.article-body) → .notion-page
  3. Auto-generate TOC from h2/h3 → <details class="notion-toc"> after <h1>
  4. Rename .nav → .notion-nav
  5. Skip scraped external pages (arxiv / githubassets / vllm.ai CSS links)

Usage:
  python3 scripts/restyle_reports.py --dry-run           # preview all changes
  python3 scripts/restyle_reports.py --one path/to/file  # transform single file
  python3 scripts/restyle_reports.py                     # transform all (write!)
"""
import os
import re
import sys
import glob
import argparse
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parent.parent  # backyes.github.io/
NOTION_CSS = "assets/css/notion-style.css"

# Report directories (relative to REPO)
REPORT_DIRS = [
    "ai-supernode-bus",
    "deepepv2",
    "mlsys2026",
    "deepseek-mtp",
    "moe-clos",
    "generative-rec",
    "sparse-clos",
    "supernode-metrics",
    "mtp-survey",
    "3dls",
    "space-ecom",
    "pd-routing",
    "trillium",
    "spacex",
    "hbm-cxl",
    "inference-community",
    "deep-ep",
    "umdk",
    "pd-separation",
    "vllm_research",
]

# External CSS hosts — files linking to these are scraped snapshots, skip them
EXCLUDE_HOSTS = [
    r"arxiv\.org",
    r"githubassets\.com",
    r"vllm\.ai",
]

# Wrapper class names that should be normalized to .notion-page
WRAPPER_CLASSES = ["container", "wrap", "article-body", "content", "main-content"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def relpath_to_css(file_rel: str) -> str:
    """Compute relative path from a report file to assets/css/notion-style.css."""
    depth = file_rel.count(os.sep)
    return "../" * depth + NOTION_CSS


def should_exclude(content: str) -> bool:
    """Return True if file links to external scraped-page CSS (keep as-is)."""
    for host in EXCLUDE_HOSTS:
        if re.search(r'href="https?://[^"]*' + host, content):
            return True
    return False


def extract_toc(body_inner: str) -> str:
    """Generate a collapsible TOC from h2/h3 headings found in body."""
    items = []
    seen_anchors = set()

    for line in body_inner.split("\n"):
        for tag in ("h2", "h3"):
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", line, re.S)
            if m:
                title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                # Build anchor
                anchor = re.sub(r"[^\w\- ]", "", title).strip().replace(" ", "-").lower()[:50]
                # Deduplicate
                base = anchor
                n = 1
                while anchor in seen_anchors:
                    anchor = f"{base}-{n}"
                    n += 1
                seen_anchors.add(anchor)
                css_class = "toc-h2" if tag == "h2" else "toc-h3"
                items.append(f'<li><a href="#{anchor}" class="{css_class}">{title}</a></li>')
                break  # only match one tag per line

    if not items:
        return ""
    return (
        '<details class="notion-toc"><summary>Contents</summary>\n'
        "<ul>\n" + "\n".join(items) + "\n</ul>\n</details>"
    )


# --------------------------------------------------------------------------- #
# Core transform                                                              #
# --------------------------------------------------------------------------- #
def restyle(file_path: Path) -> tuple:
    """
    Transform a single HTML file. Returns (new_content, status_msg).
    status_msg: 'ok' | 'exclude: ...' | 'skip: ...'
    """
    try:
        src = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return None, f"error reading: {e}"

    file_rel = os.path.relpath(file_path, REPO)

    # --- Guard: exclude scraped external pages ---
    if should_exclude(src):
        return None, "exclude: scraped external page"

    # --- Guard: already restyled (idempotency) ---
    if 'class="notion-page"' in src:
        return None, "skip: already restyled"

    # ---- 1. HEAD: remove <style> blocks + local .css links → inject shared CSS ----
    head_match = re.search(r"(<head[^>]*>)(.*?)(</head>)", src, re.S)
    if not head_match:
        return None, "skip: no <head>"

    css_rel = relpath_to_css(file_rel)
    head_inner = head_match.group(2)

    # Remove ALL per-file <style>...</style> blocks
    head_inner = re.sub(r"<style\b[^>]*>.*?</style>", "", head_inner, flags=re.S)
    # Remove prior local stylesheet links (but keep CDN fonts, etc.)
    head_inner = re.sub(
        r'<link[^>]*rel=["\']stylesheet["\'][^>]*href=["\'][^"\']*\.css["\'][^>]*>',
        "",
        head_inner,
        flags=re.I,
    )
    head_inner = head_inner.strip()

    # Inject shared CSS link as first child of <head>
    css_tag = f'<link rel="stylesheet" href="{css_rel}">'
    new_head = f"{head_match.group(1)}\n  {css_tag}\n  {head_inner}\n{head_match.group(3)}"
    src = src[: head_match.start()] + new_head + src[head_match.end():]

    # ---- 2. BODY: normalize wrapper → .notion-page ----
    body_match = re.search(r"(<body[^>]*>)(.*?)(</body>)", src, re.S)
    if not body_match:
        return None, "skip: no <body>"

    body_inner = body_match.group(2)

    # Try to rename an existing recognized wrapper class
    wrapper_pattern = (
        r'<div class="(?:' + "|".join(WRAPPER_CLASSES) + r')(\s[^"]*)?">'
    )
    new_body_inner, n_subs = re.subn(
        wrapper_pattern, '<div class="notion-page">', body_inner, count=1
    )

    if n_subs == 0:
        # No recognized wrapper — wrap entire body content
        new_body_inner = f'<div class="notion-page">\n{body_inner}\n</div>'

    body_inner = new_body_inner

    # Rename trailing .nav → .notion-nav
    body_inner = re.sub(
        r'<div class="nav">', '<nav class="notion-nav">', body_inner
    )
    # Close tag: replace the matching </div> (heuristic: next </div> after notion-nav)
    body_inner = re.sub(
        r'(<nav class="notion-nav">.*?)</div>(?!\s*<nav)',
        r"\1</nav>",
        body_inner,
        count=1,
        flags=re.S,
    )

    # ---- 3. Inject TOC after <h1> ----
    toc = extract_toc(body_inner)
    if toc:
        body_inner = re.sub(
            r"(<h1[^>]*>.*?</h1>)",
            r"\1\n" + toc + "\n",
            body_inner,
            count=1,
            flags=re.S,
        )

    new_body = f"{body_match.group(1)}\n{body_inner}\n{body_match.group(3)}"
    src = src[: body_match.start()] + new_body + src[body_match.end():]

    return src, "ok"


# --------------------------------------------------------------------------- #
# File discovery                                                              #
# --------------------------------------------------------------------------- #
def discover_files() -> list:
    """Return list of all HTML files in report directories."""
    files = []
    for d in REPORT_DIRS:
        base = REPO / d
        if not base.exists():
            continue
        for ext in ("*.html", "*.HTML"):
            files.extend(base.rglob(ext))
    return sorted(files)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Batch-apply Notion style to Survey-by-AI report HTML files."
    )
    ap.add_argument("--dry-run", action="store_true", help="preview changes without writing")
    ap.add_argument("--one", metavar="REL_PATH", help="transform a single file (repo-relative path)")
    args = ap.parse_args()

    if args.one:
        files = [REPO / args.one]
    else:
        files = discover_files()

    stats = {"ok": 0, "exclude": 0, "skip": 0, "error": 0}

    for fp in files:
        new_content, status = restyle(fp)

        if status == "ok":
            stats["ok"] += 1
            tag = "✓"
        elif status.startswith("exclude"):
            stats["exclude"] += 1
            tag = "⊘"
        elif status.startswith("skip"):
            stats["skip"] += 1
            tag = "–"
        else:
            stats["error"] += 1
            tag = "✗"

        rel = os.path.relpath(fp, REPO)
        print(f"  {tag} {rel}  ({status})")

        if new_content and not args.dry_run:
            fp.write_text(new_content, encoding="utf-8")

    print()
    print(f"  Total: {len(files)} files")
    print(f"  ok={stats['ok']}  exclude={stats['exclude']}  skip={stats['skip']}  error={stats['error']}")

    if args.dry_run:
        print("\n  (dry-run: no files modified)")
    elif stats["ok"] > 0:
        print(f"\n  {stats['ok']} files restyled.")


if __name__ == "__main__":
    main()
