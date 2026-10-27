"""Download the site's fonts (latin and latin-ext subsets, variable woff2) and their licences.

Run when the fonts in ``brand.FONTS`` change: ``python docs/_scripts/fetch_fonts.py``. The
files are committed to the theme's ``static/fonts/`` so the site never loads fonts from
another host.
"""

from __future__ import annotations

import re
import urllib.request

from brand import FONTS, GOOGLE_FONTS, STATIC

OUT = STATIC / "fonts"
SUBSETS = ("latin", "latin-ext")
UA = (  # a browser user agent, so Google Fonts serves woff2
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.woff2"):  # faces of families no longer in FONTS go too
        old.unlink()
    faces = []
    for family, _, weights, licence in FONTS.values():
        q = family.replace(" ", "+")
        css = get(
            f"https://fonts.googleapis.com/css2?family={q}:wght@{weights}&display=swap"
        ).decode()
        blocks = dict(re.findall(r"/\* ([\w-]+) \*/\s*(@font-face \{.*?\})", css, re.S))
        if missing := set(SUBSETS) - set(blocks):
            raise SystemExit(f"Google Fonts served no {sorted(missing)} face for {family}.")
        for subset, block in blocks.items():
            if subset in SUBSETS:
                url = re.search(r"url\((https://[^)]+\.woff2)\)", block).group(1)
                name = f"{family.replace(' ', '')}-{subset}.woff2"
                (OUT / name).write_bytes(get(url))
                faces.append(re.sub(r"url\(https://[^)]+\)", f"url({name})", block))
        text = get(f"{GOOGLE_FONTS}/{licence}/OFL.txt").decode()
        lines = [line.rstrip() for line in text.splitlines()]  # as the whitespace hook wants
        (OUT / f"OFL-{licence}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    header = "/* Written by docs/_scripts/fetch_fonts.py; urls are relative to this file. */\n"
    (OUT / "fonts.css").write_text(header + "\n".join(faces) + "\n", encoding="utf-8")
    print("\n".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
