# -*- coding: utf-8 -*-
"""Print compact digests for a given index list, for LLM to read and write profiles."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
D = json.load(open(ROOT / "digests.json", encoding="utf-8"))
only = []
for a in sys.argv[1:]:
    if a.startswith("--only="):
        only = [x for x in a[7:].split(",") if x]
if not only:
    only = sorted(D.keys(), key=int)
for i in only:
    if i not in D:
        continue
    d = D[i]
    cat = "NON-STARTUP" if d["category"] == "non_startup" else "STARTUP"
    print(f"\n== [{i}] {d['name']}  ({cat}) ==")
    for sn in d.get("snippets", [])[:7]:
        print(f"  • {sn.get('t','')}")
        print(f"    URL: {sn.get('u','')}")
        print(f"    {sn.get('s','')}")
    if d.get("page"):
        p = d["page"]
        print(f"  [PAGE] {p.get('t','')} | {p.get('u','')}")
        print(f"    {p.get('x','')[:900]}")
    if d.get("errors"):
        print(f"  (errors: {d['errors']})")
