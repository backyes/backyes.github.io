# -*- coding: utf-8 -*-
"""Assemble final report.html by injecting profiles + non-startup fragments into the scaffold."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
html = (ROOT / "report.html").read_text(encoding="utf-8")

# profiles in order
prof_files = ["profiles.html", "profiles_b2.html", "profiles_b3.html",
              "profiles_b4.html", "profiles_b5.html", "profiles_b6.html", "profiles_b7.html"]
profiles = "\n".join((ROOT / f).read_text(encoding="utf-8") for f in prof_files)
html = html.replace("<!--PROFILES-->", profiles)

# non-startup in order
non_files = ["nonstartup.html", "nonstartup_b2.html", "nonstartup_b3.html", "nonstartup_b4.html"]
non = "\n".join((ROOT / f).read_text(encoding="utf-8") for f in non_files)
html = html.replace("<!--NONSTARTUP-->", non)

(ROOT / "report.html").write_text(html, encoding="utf-8")
print("assembled report.html, size:", len(html), "chars")
# count cards
print("startup cards:", html.count('class="card startup"'))
print("non cards:", html.count('class="card non"'))
print("source links (approx):", html.count('<a href'))
