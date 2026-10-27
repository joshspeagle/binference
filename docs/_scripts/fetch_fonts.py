"""Download the site's fonts (latin and latin-ext subsets, variable woff2) and their licences.

Run once when the fonts change: ``python docs/_scripts/fetch_fonts.py``. The files are
committed to ``docs/_static/fonts/`` so the site never loads fonts from another host.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).parents[1] / "_static" / "fonts"
FAMILIES = {  # family, weight range, licence directory in google/fonts
    "Sora": ("500..700", "sora"),
    "Atkinson Hyperlegible Next": ("400..700", "atkinsonhyperlegiblenext"),
    "Atkinson Hyperlegible Mono": ("400..600", "atkinsonhyperlegiblemono"),
}
SUBSETS = ("latin", "latin-ext")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    css_out = []
    for family, (weights, licence) in FAMILIES.items():
        q = family.replace(" ", "+")
        css = get(f"https://fonts.googleapis.com/css2?family={q}:wght@{weights}&display=swap").decode()
        for subset, block in re.findall(r"/\* ([\w-]+) \*/\s*(@font-face \{.*?\})", css, re.S):
            if subset not in SUBSETS:
                continue
            url = re.search(r"url\((https://[^)]+\.woff2)\)", block).group(1)
            name = f"{family.replace(' ', '')}-{subset}.woff2"
            (OUT / name).write_bytes(get(url))
            css_out.append(re.sub(r"url\(https://[^)]+\)", f"url(fonts/{name})", block))
        (OUT / f"OFL-{licence}.txt").write_bytes(
            get(f"https://raw.githubusercontent.com/google/fonts/main/ofl/{licence}/OFL.txt")
        )
    (OUT / "fonts.css").write_text(
        "/* Written by docs/_scripts/fetch_fonts.py; paths are relative to _static/. */\n"
        + "\n".join(css_out).replace("url(fonts/", "url(")
        + "\n"
    )
    print("\n".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
