"""The binference brand: mark geometry, colour tokens and the assets written from them.

This file is the one home of the design's numbers (``docs/design/brand.md`` explains them).
``python docs/_scripts/brand.py`` writes

* ``docs/_static/brand/*.svg``: the mark in three cuts, light and dark; the lockups with the
  wordmark as outlines; the favicon;
* ``docs/_static/tokens.css``: the colour and type tokens of the site theme;
* the token table in ``docs/design/brand.md`` (between its ``tokens`` markers).

``--check`` writes nothing and fails if a committed output differs from what this file
would write. The wordmark outlines need Sora's variable font (downloaded on first use to
``docs/_build/brand/``) and fontTools.
"""

from __future__ import annotations

import argparse
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

DOCS = Path(__file__).parents[1]
BRAND = DOCS / "_static" / "brand"
TOKENS_CSS = DOCS / "_static" / "tokens.css"
BRAND_MD = DOCS / "design" / "brand.md"
SORA_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/sora/Sora%5Bwght%5D.ttf"
SORA_CACHE = DOCS / "_build" / "brand" / "Sora-wght.ttf"

# --------------------------------------------------------------------------- colour tokens

#: name: (light, dark, role)
TOKENS = {
    "bg": ("#FBFBFE", "#10112A", "page"),
    "surface": ("#F1F2FA", "#181A3D", "panels, code, explainers"),
    "raised": ("#E6E8F5", "#22255A", "hover, selected"),
    "rule": ("#DADCEC", "#2B2E5E", "hairlines (decorative)"),
    "ink": ("#1B1D4A", "#E8E9FB", "text"),
    "ink-2": ("#5D5F86", "#9C9ECB", "secondary text, axes"),
    "brand": ("#4A4FC4", "#8C91F0", "the population: draws, bins, the mark"),
    "link": ("#3A3FB0", "#A5A9F5", "links, focus"),
    "sample": ("#EC5F3E", "#FF8A6B", "an object or a sample (marks only)"),
    "sample-text": ("#C2410C", "#FF8A6B", "an object or a sample, as text"),
    "bar": ("#10112A", "#0B0C20", "top bar and footer, both themes"),
    "bar-text": ("#E8E9FB", "#E8E9FB", "text on the bar"),
    "tip": ("#0E7D6E", "#4CC2B0", "'In plain words' notes"),
    "warn": ("#9A6200", "#E8AE52", "warnings"),
    "danger": ("#B42318", "#F97066", "errors"),
}
#: data colours: categories in this order, a one-hue ramp, a two-pole ramp with a grey middle
DATA = {
    "cat": (["#4A4FC4", "#C07C00", "#1A9E8C", "#8F2E74", "#5E8A1C"],
            ["#7C82E8", "#C4871C", "#25A390", "#9C4A8E", "#749E39"]),
    "seq": (["#E2E3F8", "#B8BBEE", "#8186DE", "#4C53C2", "#2A2F8F"],
            ["#1F2257", "#33388C", "#5259C7", "#8A90EC", "#C9CCFA"]),
    "div": (["#2A2F8F", "#8186DE", "#E4E4EA", "#E0A848", "#8A5600"],
            ["#C9CCFA", "#5C63C9", "#3A3B4E", "#C48A2A", "#F2C77A"]),
}
FONTS = {
    "display": "'Sora', system-ui, sans-serif",
    "body": "'Atkinson Hyperlegible Next', system-ui, sans-serif",
    "mono": "'Atkinson Hyperlegible Mono', ui-monospace, monospace",
}


def _lum(c: str) -> float:
    v = [int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    v = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]


def contrast(a: str, b: str) -> float:
    """WCAG 2 contrast ratio of two colours."""
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _against(name: str, theme: int) -> str:
    """The colour each token is read against: its text for surfaces, the bar for bar text,
    the page otherwise."""
    if name == "bar-text":
        return TOKENS["bar"][theme]
    if name in ("bg", "surface", "raised", "bar"):  # surfaces: against the text on them
        return TOKENS["bar-text" if name == "bar" else "ink"][theme]
    return TOKENS["bg"][theme]


# --------------------------------------------------------------------------- the mark

#: bins, draws, draw opacity, outline width, log-normal jitter of the draws, sample radius
CUTS = {
    "full": dict(nb=10, nd=7, op=0.17, sw=1.6, sd=0.15, dot=3.2),
    "mid": dict(nb=7, nd=4, op=0.26, sw=2.0, sd=0.13, dot=3.6),
    "tiny": dict(nb=5, nd=2, op=0.4, sw=2.6, sd=0.07, dot=4.0),
}


def _f(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def mark_geometry(cut: str = "full", seed: int = 3) -> dict:
    """The Overlay b on a 64-unit square.

    The stem is the first bin. The other bins trace a bowl: a floor of 4 units plus a
    semi-ellipse that spans them exactly, so no bin falls outside it at any cut. The draws
    jitter each bin's height log-normally; the outline is their mean; an elliptical counter
    is cut out of the bowl and holds one sample.
    """
    p = CUTS[cut]
    base, x0, x1, floor, height = 58.0, 7.0, 57.0, 4.0, 29.0
    nb = p["nb"]
    w = (x1 - x0) / nb
    a = (x1 - x0 - w) / 2
    cx, cy = x0 + w + a, base - 16.5
    mean = np.empty(nb)
    mean[0] = base - 4
    for k in range(1, nb):
        u = (x0 + (k + 0.5) * w - cx) / a
        mean[k] = floor + height * math.sqrt(max(1 - u * u, 0))

    def step(h):
        d = f"M{_f(x0)} {_f(base)}"
        for k in range(nb):
            d += f"V{_f(base - h[k])}H{_f(x0 + (k + 1) * w)}"
        return d + f"V{_f(base)}"

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(p["nd"]):
        h = mean * np.exp(rng.normal(0, p["sd"], nb))
        h[0] = min(h[0], base - 2)
        draws.append(step(h) + "Z")
    return dict(draws=draws, mean=step(mean), cx=cx, cy=cy, rx=0.34 * a, ry=7.8, **p)


def mark_body(cut: str, main: str, dot: str, uid: str) -> str:
    """The mark's SVG elements (no ``<svg>`` wrapper), in a 64 x 64 frame."""
    g = mark_geometry(cut)
    fills = "".join(f'<path d="{d}" fill="{main}" opacity="{g["op"]}"/>' for d in g["draws"])
    return (
        f'<mask id="{uid}"><rect width="64" height="64" fill="#fff"/>'
        f'<ellipse cx="{_f(g["cx"])}" cy="{_f(g["cy"])}" rx="{_f(g["rx"])}" ry="{_f(g["ry"])}" fill="#000"/></mask>'
        f'<g mask="url(#{uid})">{fills}<path d="{g["mean"]}" fill="none" stroke="{main}" '
        f'stroke-width="{g["sw"]}"/></g>'
        f'<circle cx="{_f(g["cx"])}" cy="{_f(g["cy"])}" r="{g["dot"]}" fill="{dot}"/>'
    )


def mark_svg(cut: str, theme: int, uid: str = "b") -> str:
    main, dot = TOKENS["brand"][theme], TOKENS["sample"][theme]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" '
        f'aria-label="binference">{mark_body(cut, main, dot, uid)}</svg>\n'
    )


def favicon_svg() -> str:
    """The tiny cut, following the browser's colour scheme."""
    body = mark_body("tiny", "var(--m)", "var(--d)", "f")
    lm, ld = TOKENS["brand"][0], TOKENS["sample"][0]
    dm, dd = TOKENS["brand"][1], TOKENS["sample"][1]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        f"<style>svg{{--m:{lm};--d:{ld}}}@media (prefers-color-scheme:dark){{svg{{--m:{dm};--d:{dd}}}}}</style>"
        f"{body}</svg>\n"
    )


# --------------------------------------------------------------------------- the wordmark

def _sora_600():
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    if not SORA_CACHE.exists():
        SORA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SORA_URL, timeout=60) as r:
            SORA_CACHE.write_bytes(r.read())
    return instantiateVariableFont(TTFont(SORA_CACHE), {"wght": 600})


def wordmark_paths():
    """The letters of "binference" in Sora 600 as SVG paths in font units (y up), with the
    dot of the i replaced by a measurement with an error bar. Returns (paths, tittle, width,
    ascent) where ``tittle`` holds the error bar's geometry."""
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.pens.svgPathPen import SVGPathPen

    font = _sora_600()
    gs, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
    paths, x = [], 0
    for ch in "bınference":
        name = cmap[ord(ch)]
        pen = SVGPathPen(gs)
        gs[name].draw(pen)
        paths.append((x, pen.getCommands()))
        if ch == "ı":
            bd = BoundsPen(gs)
            gs[name].draw(bd)
            i_x = x + (bd.bounds[0] + bd.bounds[2]) / 2
            stem_top = bd.bounds[3]
        x += hmtx[name][0]
    # the error bar of the design sheet, in its 24 x 34 box scaled to 0.33 em: the bar runs
    # 28 box units up from just above the stem (a 0.06 em gap), the dot 10 units up with
    # radius 7, caps 14 wide, stroke 3.2
    u = 330 / 34
    bottom = stem_top + 60
    tittle = dict(x=i_x, bottom=bottom, top=bottom + 28 * u, dot=bottom + 10 * u, r=7 * u, cap=14 * u, sw=3.2 * u)
    return paths, tittle, x, font["hhea"].ascent


def lockup_svg(theme: int) -> str:
    """The mark beside the wordmark, all outlines."""
    paths, t, width, _ = wordmark_paths()
    ink, dot = TOKENS["ink"][theme], TOKENS["sample"][theme]
    s = 64 / 1.45 / 1000  # font units to mark units: the mark is 1.45 em tall, as on the sheet
    gap, base = 14, 58  # mark-to-word gap and the shared baseline, in mark units
    letters = "".join(f'<path transform="translate({_f(x)} 0)" d="{d}"/>' for x, d in paths)
    tw = f'stroke="{dot}" stroke-width="{_f(t["sw"])}" stroke-linecap="round"'
    tittle = (
        f'<line x1="{_f(t["x"])}" y1="{_f(t["bottom"])}" x2="{_f(t["x"])}" y2="{_f(t["top"])}" {tw}/>'
        f'<line x1="{_f(t["x"] - t["cap"] / 2)}" y1="{_f(t["top"])}" x2="{_f(t["x"] + t["cap"] / 2)}" y2="{_f(t["top"])}" {tw}/>'
        f'<line x1="{_f(t["x"] - t["cap"] / 2)}" y1="{_f(t["bottom"])}" x2="{_f(t["x"] + t["cap"] / 2)}" y2="{_f(t["bottom"])}" {tw}/>'
        f'<circle cx="{_f(t["x"])}" cy="{_f(t["dot"])}" r="{_f(t["r"])}" fill="{dot}"/>'
    )
    total = 64 + gap + width * s
    word = (
        f'<g transform="translate({_f(64 + gap)} {base}) scale({s:.5f} {-s:.5f})">'
        f'<g fill="{ink}">{letters}</g>{tittle}</g>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(total)} 64" role="img" '
        f'aria-label="binference">{mark_body("full", TOKENS["brand"][theme], dot, "l")}{word}</svg>\n'
    )


# --------------------------------------------------------------------------- outputs

def tokens_css() -> str:
    def block(theme: int) -> str:
        lines = [f"  --b-{k}: {v[theme]};" for k, v in TOKENS.items()]
        for kind, (light, dark) in DATA.items():
            cols = light if theme == 0 else dark
            lines += [f"  --b-{kind}-{i}: {c};" for i, c in enumerate(cols)]
        return "\n".join(lines)

    fonts = "\n".join(f"  --b-font-{k}: {v};" for k, v in FONTS.items())
    return (
        "/* Written by docs/_scripts/brand.py; edit the tokens there. */\n"
        f":root {{\n{block(0)}\n{fonts}\n  color-scheme: light;\n}}\n"
        f'@media (prefers-color-scheme: dark) {{\n  :root:not([data-theme="light"]) {{\n{block(1)}\n  color-scheme: dark;\n  }}\n}}\n'
        f':root[data-theme="dark"] {{\n{block(1)}\n  color-scheme: dark;\n}}\n'
    )


def token_table() -> str:
    rows = ["| Token | Role | Light | Contrast | Dark | Contrast |", "|---|---|---|---|---|---|"]
    for name, (light, dark, role) in TOKENS.items():
        cl, cd = contrast(light, _against(name, 0)), contrast(dark, _against(name, 1))
        rows.append(f"| `{name}` | {role} | `{light}` | {cl:.1f} | `{dark}` | {cd:.1f} |")
    for kind, label in (("cat", "categories, in order"), ("seq", "sequential"), ("div", "diverging")):
        light, dark = DATA[kind]
        rows.append(f"| `{kind}-0…{len(light) - 1}` | {label} | {' '.join(f'`{c}`' for c in light)} | | "
                    f"{' '.join(f'`{c}`' for c in dark)} | |")
    return "\n".join(rows)


def outputs() -> dict[Path, str]:
    out = {TOKENS_CSS: tokens_css()}
    for theme, tag in ((0, "light"), (1, "dark")):
        for cut in CUTS:
            out[BRAND / f"mark-{cut}-{tag}.svg"] = mark_svg(cut, theme)
        out[BRAND / f"lockup-{tag}.svg"] = lockup_svg(theme)
    out[BRAND / "favicon.svg"] = favicon_svg()
    md = BRAND_MD.read_text()
    start, end = "<!-- tokens:start -->", "<!-- tokens:end -->"
    i, j = md.index(start) + len(start), md.index(end)
    out[BRAND_MD] = md[:i] + "\n" + token_table() + "\n" + md[j:]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if a committed output is stale")
    args = ap.parse_args(argv)
    stale = []
    for path, text in outputs().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.relative_to(DOCS.parent))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if stale:
        print("stale (run python docs/_scripts/brand.py):", *stale, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
