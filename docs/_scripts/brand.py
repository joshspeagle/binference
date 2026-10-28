"""The binference brand: the one home of the design's numbers, and the files written from them.

``docs/design/brand.md`` explains the design; every number it quotes comes from here.
``python docs/_scripts/brand.py`` writes, into the theme's ``static/`` folder,

* ``brand/*.svg``: the mark in three cuts, light and dark; the lockups with the wordmark
  as outlines; the favicon;
* ``tokens.css``: the colour, type and size tokens of the site theme;

the home page's animated mark ``hero.html`` in the theme folder, and
``docs/_ext/matplotlibrc``, the style of every executed page's plots (``conf.py`` hands
it to the notebook kernels), with the generated tables of ``brand.md`` (between its
``<!-- name:start/end -->`` markers). ``--check`` writes nothing and fails if a committed
output is stale, or if ``binference.plotting.SAMPLE_COLOR`` differs from the ``sample``
token; CI runs it.
The lockups need fontTools and Sora's variable font, downloaded on first use to
``docs/_build/brand/`` and verified against its SHA-256.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import math
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

DOCS = Path(__file__).parents[1]
STATIC = DOCS / "_themes" / "binference" / "static"
BRAND_MD = DOCS / "design" / "brand.md"
GOOGLE_FONTS = "https://raw.githubusercontent.com/google/fonts/main/ofl"
#: the Sora the lockups are drawn from, pinned to a google/fonts commit and its SHA-256
SORA_TTF = (
    "https://raw.githubusercontent.com/google/fonts/e7db6ca603fd4862010383ff17ffd6cc4ffac1c0"
    "/ofl/sora/Sora%5Bwght%5D.ttf",
    "84ff7096ae3ec6c8be47d906d1a0ba4de7f2ce78c615275c77301964a316e16c",
)
SORA_CACHE = DOCS / "_build" / "brand" / f"Sora-wght-{SORA_TTF[1][:12]}.ttf"

# --------------------------------------------------------------------------- tokens

#: name: (light, dark, role, the token it is read against for contrast)
TOKENS = {
    "bg": ("#FBFBFE", "#10112A", "page", "ink"),
    "surface": ("#F1F2FA", "#181A3D", "panels, code, explainers", "ink"),
    "raised": ("#E6E8F5", "#22255A", "hover, selected", "ink"),
    "rule": ("#DADCEC", "#2B2E5E", "hairlines (decorative)", "bg"),
    "ink": ("#1B1D4A", "#E8E9FB", "text", "bg"),
    "ink-2": ("#5D5F86", "#9C9ECB", "secondary text, axes", "bg"),
    "brand": ("#4A4FC4", "#8C91F0", "the population: draws, bins, the mark", "bg"),
    "link": ("#3A3FB0", "#A5A9F5", "links, focus", "bg"),
    "sample": ("#EC5F3E", "#FF8A6B", "an object or a sample (marks only)", "bg"),
    "sample-text": ("#C2410C", "#FF8A6B", "an object or a sample, as text", "bg"),
    "bar": ("#10112A", "#0B0C20", "top bar and footer, both themes", "bar-text"),
    "bar-text": ("#E8E9FB", "#E8E9FB", "text on the bar", "bar"),
    "tip": ("#0E7D6E", "#4CC2B0", "'In plain words' notes", "bg"),
    "warn": ("#9A6200", "#E8AE52", "warnings", "bg"),
    "danger": ("#B42318", "#F97066", "errors", "bg"),
}
#: plots are drawn on the light page in both themes; the dark theme sets them on it
FIGURE_GROUND = TOKENS["bg"][0]
#: data colours: (light, dark, use); validated with the dataviz palette checks
DATA = {
    "cat": (
        ["#4A4FC4", "#C07C00", "#1A9E8C", "#8F2E74", "#5E8A1C"],
        ["#7C82E8", "#C4871C", "#25A390", "#9C4A8E", "#749E39"],
        "categories, in this order",
    ),
    "seq": (
        ["#E2E3F8", "#B8BBEE", "#8186DE", "#4C53C2", "#2A2F8F"],
        ["#1F2257", "#33388C", "#5259C7", "#8A90EC", "#C9CCFA"],
        "sequential, one hue",
    ),
    "div": (
        ["#2A2F8F", "#8186DE", "#E4E4EA", "#E0A848", "#8A5600"],
        ["#C9CCFA", "#5C63C9", "#3A3B4E", "#C48A2A", "#F2C77A"],
        "diverging, grey middle",
    ),
}
#: role: (family, CSS stack, weight range served, google/fonts licence folder)
FONTS = {
    "display": ("Sora", "'Sora', system-ui, sans-serif", "500..700", "sora"),
    "body": (
        "Atkinson Hyperlegible Next",
        "'Atkinson Hyperlegible Next', system-ui, sans-serif",
        "400..700",
        "atkinsonhyperlegiblenext",
    ),
    "mono": (
        "Atkinson Hyperlegible Mono",
        "'Atkinson Hyperlegible Mono', ui-monospace, monospace",
        "400..600",
        "atkinsonhyperlegiblemono",
    ),
}
#: type scale: name: (role, face, weight, size in px, line height)
TYPE = {
    "title": ("page titles", "display", 600, 32, 1.15),
    "section": ("section headings", "display", 600, 20, 1.25),
    "text": ("running text", "body", 400, 16.5, 1.6),
    "code": ("code, numbers in tables", "mono", 400, 14, 1.5),
    "caption": ("captions", "body", 400, 14, 1.45),
    "subsection": ("subsection headings", "display", 600, 17, 1.3),
    "minor": ("minor headings", "body", 600, 15, 1.3),
    "label": ("labels (upper case)", "body", 600, 12, 1.4),
    "title-phone": ("page titles below 900 px", "display", 600, 28, 1.15),
}
#: other sizes in px: name: (value, use)
SIZES = {
    "accent-rule": (3, "rules marking notes, explainers and the current page"),
    "focus": (2, "focus ring"),
    "radius": (2, "the largest corner radius, on controls only"),
    "figure-text": (12, "smallest figure text at 390 px wide"),
}
#: the default figure: width and height in inches, and its text size in points. A figure
#: shown 334 px wide (the readable width of a 390 px screen) draws text at
#: pt x 334 / (72 x width) px: 13.5 px here. Legends and tick labels take this size too
#: ("medium"), and so do titles, which label panels.
FIGURE = {"width": 4.8, "height": 3.4, "pt": 14}
PHONE_READABLE_PX = 334
#: the widest a plot is shown (``--b-figure-max``), in rem: 480 px at a 16 px root, which
#: less the plot panel's 8 px padding draws its text at 14 x 464 / (72 x 4.8) = 18.8 px
FIGURE_MAX_REM = 30

# --------------------------------------------------------------------------- the mark

SIZE, BASE = 64, 58.0  # the mark's square frame, and the baseline its bins stand on
MARK_EM = 1.45  # mark height over wordmark font size in the lockup
#: name: used from px, bins, draws, draw opacity, outline width, draw jitter, sample radius
CUTS = {
    "full": {"px": 48, "nb": 10, "nd": 7, "op": 0.17, "sw": 1.6, "sd": 0.15, "dot": 3.2},
    "mid": {"px": 24, "nb": 7, "nd": 4, "op": 0.26, "sw": 2.0, "sd": 0.13, "dot": 3.6},
    "tiny": {"px": 0, "nb": 5, "nd": 2, "op": 0.4, "sw": 2.6, "sd": 0.07, "dot": 4.0},
}
#: bowl geometry in frame units: span, floor and height of the bowl, stem height, counter
BOWL = {
    "x0": 7.0,
    "x1": 57.0,
    "floor": 4.0,
    "height": 29.0,
    "stem": 54.0,
    "cy": 16.5,
    "rx": 0.34,
    "ry": 7.8,
}


def _f(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


SEED = 3  # the seed of the logo's draws


def mark_geometry(cut: str = "full", seed: int = SEED) -> dict:
    """The Overlay b: a stem bin, then bins tracing a bowl (a floor plus a semi-ellipse that
    spans them exactly), drawn as log-normally jittered draws with their mean as an outline,
    and an elliptical counter holding one sample."""
    p, b = CUTS[cut], BOWL
    nb = p["nb"]
    w = (b["x1"] - b["x0"]) / nb
    a = (b["x1"] - b["x0"] - w) / 2
    cx = b["x0"] + w + a
    mean = np.empty(nb)
    mean[0] = b["stem"]
    for k in range(1, nb):
        u = (b["x0"] + (k + 0.5) * w - cx) / a
        mean[k] = b["floor"] + b["height"] * math.sqrt(1 - u * u)

    def step(h):
        d = f"M{_f(b['x0'])} {_f(BASE)}"
        for k in range(nb):
            d += f"V{_f(BASE - h[k])}H{_f(b['x0'] + (k + 1) * w)}"
        return d + f"V{_f(BASE)}"

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(p["nd"]):
        h = mean * np.exp(rng.normal(0, p["sd"], nb))
        h[0] = min(h[0], BASE - 2)  # keep the stem inside the frame
        draws.append(step(h) + "Z")
    counter = {"cx": cx, "cy": BASE - b["cy"], "rx": b["rx"] * a, "ry": b["ry"]}
    return {"draws": draws, "mean": step(mean), **counter, **p}


def mark_fills(cut: str, main: str, seed: int = SEED) -> str:
    """The pale fills of the mark's draws."""
    g = mark_geometry(cut, seed)
    return "".join(f'<path d="{d}" fill="{main}" opacity="{g["op"]}"/>' for d in g["draws"])


def mark_body(cut: str, main: str, dot: str, uid: str, fills: str | None = None) -> str:
    """The mark's SVG elements in its ``SIZE`` frame, without an ``<svg>`` wrapper; ``fills``
    replaces the logo's own draws."""
    g = mark_geometry(cut)
    fills = mark_fills(cut, main) if fills is None else fills
    counter = f'cx="{_f(g["cx"])}" cy="{_f(g["cy"])}"'
    return (
        f'<mask id="{uid}"><rect width="{SIZE}" height="{SIZE}" fill="#fff"/>'
        f'<ellipse {counter} rx="{_f(g["rx"])}" ry="{_f(g["ry"])}" fill="#000"/></mask>'
        f'<g mask="url(#{uid})">{fills}<path d="{g["mean"]}" fill="none" stroke="{main}" '
        f'stroke-width="{g["sw"]}"/></g><circle {counter} r="{g["dot"]}" fill="{dot}"/>'
    )


#: the home page's mark cross-fades between this many sets of draws of the full cut (seeds
#: SEED onwards, the first the logo's own), showing each for HERO_STEP seconds and fading over
#: the last HERO_FADE of them
HERO_FRAMES, HERO_STEP, HERO_FADE = 6, 4, 1
HERO = DOCS / "_themes" / "binference" / "hero.html"


def hero_html() -> str:
    """The home page's mark, inlined by ``home.html`` (decorative: the page's title names it),
    with its cross-fade: each set of draws is visible for ``HERO_STEP - HERO_FADE`` s, then fades
    out as the next fades in. It runs only once ``binference.js`` adds ``moving``."""
    cycle = HERO_FRAMES * HERO_STEP
    pct = lambda s: _f(100 * s / cycle)  # noqa: E731
    style = (
        f"<style>.hero.moving .hero-draws{{animation:hero-draws {cycle}s linear infinite}}"
        f"@keyframes hero-draws{{0%,{pct(HERO_STEP - HERO_FADE)}%{{opacity:1}}"
        f"{pct(HERO_STEP)}%,{pct(cycle - HERO_FADE)}%{{opacity:0}}100%{{opacity:1}}}}</style>"
    )
    fills = "".join(
        f'<g class="hero-draws" style="animation-delay:-{(cycle - i * HERO_STEP) % cycle}s">'
        f"{mark_fills('full', 'currentColor', SEED + i)}</g>"
        for i in range(HERO_FRAMES)
    )
    body = mark_body("full", "currentColor", "var(--b-sample)", "hero-counter", fills)
    svg = f'<svg class="hero-mark" viewBox="0 0 {SIZE} {SIZE}" aria-hidden="true">'
    return f"{style}{svg}{body}</svg>\n"


def _svg(width: float, body: str, label: bool = True) -> str:
    aria = ' role="img" aria-label="binference"' if label else ""
    view = f'viewBox="0 0 {_f(width)} {SIZE}"'
    return f'<svg xmlns="http://www.w3.org/2000/svg" {view}{aria}>{body}</svg>\n'


def mark_svg(cut: str, theme: int) -> str:
    uid = f"binference-{cut}-{('light', 'dark')[theme]}"  # unique if several are inlined
    return _svg(SIZE, mark_body(cut, TOKENS["brand"][theme], TOKENS["sample"][theme], uid))


def favicon_svg() -> str:
    """The tiny cut, following the browser's colour scheme."""
    (lm, dm), (ld, dd) = TOKENS["brand"][:2], TOKENS["sample"][:2]
    style = (
        f"<style>svg{{--m:{lm};--d:{ld}}}"
        f"@media (prefers-color-scheme:dark){{svg{{--m:{dm};--d:{dd}}}}}</style>"
    )
    return _svg(
        SIZE, style + mark_body("tiny", "var(--m)", "var(--d)", "binference-favicon"), label=False
    )


# --------------------------------------------------------------------------- the wordmark


def _sora_600():
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    url, sha = SORA_TTF
    if not SORA_CACHE.exists() or hashlib.sha256(SORA_CACHE.read_bytes()).hexdigest() != sha:
        SORA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != sha:
            raise SystemExit(f"{url} does not have the SHA-256 in SORA_TTF.")
        SORA_CACHE.write_bytes(data)
    return instantiateVariableFont(TTFont(SORA_CACHE), {"wght": 600})


@functools.cache
def wordmark():
    """The word "binference" in Sora 600 as SVG paths in font units (y up), with a dotless i
    so the error bar can replace its dot. Returns (paths, width, units per em, i centre,
    stem top).

    Letters are placed by their advance widths. Sora's kerning moves this word by 7 units
    of 5588 in total (measured in Chromium), which no one can see, so it is not applied.
    """
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
            bounds = BoundsPen(gs)
            gs[name].draw(bounds)
            i_x, stem_top = x + (bounds.bounds[0] + bounds.bounds[2]) / 2, bounds.bounds[3]
        x += hmtx[name][0]
    return paths, x, font["head"].unitsPerEm, i_x, stem_top


def lockup_svg(theme: int, cut: str = "mid") -> str:
    """The mark (``cut``) beside the wordmark, all outlines, sharing the baseline ``BASE``.

    The error bar is the design sheet's 24 x 34 box scaled to 0.33 em: it starts 0.06 em
    above the stem and runs 28 box units up, with the dot 10 units up (radius 7), caps 14
    wide and a stroke of 3.2.
    """
    paths, width, em, i_x, stem_top = wordmark()
    ink, dot = TOKENS["ink"][theme], TOKENS["sample"][theme]
    s = SIZE / MARK_EM / em  # font units to frame units
    u = 0.33 * em / 34  # error-bar box units to font units
    bottom = stem_top + 0.06 * em
    top = bottom + 28 * u
    stroke = f'stroke="{dot}" stroke-width="{_f(3.2 * u)}" stroke-linecap="round"'
    bar = f'<line x1="{_f(i_x)}" y1="{_f(bottom)}" x2="{_f(i_x)}" y2="{_f(top)}" {stroke}/>'
    for y in (top, bottom):
        x1, x2 = _f(i_x - 7 * u), _f(i_x + 7 * u)
        bar += f'<line x1="{x1}" y1="{_f(y)}" x2="{x2}" y2="{_f(y)}" {stroke}/>'
    bar += f'<circle cx="{_f(i_x)}" cy="{_f(bottom + 10 * u)}" r="{_f(7 * u)}" fill="{dot}"/>'
    letters = "".join(f'<path transform="translate({_f(x)} 0)" d="{d}"/>' for x, d in paths)
    gap = 14  # mark to word, in frame units
    word = (
        f'<g transform="translate({_f(SIZE + gap)} {_f(BASE)}) scale({s:.5f} {-s:.5f})">'
        f'<g fill="{ink}">{letters}</g>{bar}</g>'
    )
    uid = f"binference-lockup-{cut}-{('light', 'dark')[theme]}"
    return _svg(SIZE + gap + width * s, mark_body(cut, TOKENS["brand"][theme], dot, uid) + word)


# --------------------------------------------------------------------------- outputs


def tokens_css() -> str:
    def colours(theme: int) -> str:
        lines = [f"  --b-{k}: {v[theme]};" for k, v in TOKENS.items()]
        for kind, v in DATA.items():
            lines += [f"  --b-{kind}-{i}: {c};" for i, c in enumerate(v[theme])]
        return "\n".join(lines)

    fixed = [f"  --b-font-{k}: {v[1]};" for k, v in FONTS.items()]
    fixed += [
        f"  --b-type-{k}: {w} {_f(px)}px/{lh} var(--b-font-{face});"
        for k, (_, face, w, px, lh) in TYPE.items()
    ]
    fixed += [f"  --b-{k}: {v}px;" for k, (v, _) in SIZES.items()]
    fixed.append(f"  --b-figure-ground: {FIGURE_GROUND};")
    fixed.append(f"  --b-figure-max: {FIGURE_MAX_REM}rem;")
    dark = colours(1)
    return (
        "/* Written by docs/_scripts/brand.py; edit the tokens there. */\n"
        f":root {{\n{colours(0)}\n" + "\n".join(fixed) + "\n  color-scheme: light;\n}\n"
        f'@media (prefers-color-scheme: dark) {{\n  :root:not([data-theme="light"]) {{\n{dark}\n'
        "  color-scheme: dark;\n  }\n}\n"
        f':root[data-theme="dark"] {{\n{dark}\n  color-scheme: dark;\n}}\n'
    )


MATPLOTLIBRC = DOCS / "_ext" / "matplotlibrc"
PLOTTING = DOCS.parent / "src" / "binference" / "plotting.py"


def matplotlibrc() -> str:
    """Plots of the executed pages: drawn for the light figure ground in both themes (the theme
    puts them on that ground in the dark theme), categories in order, readable at phone width.
    """
    phone_px = FIGURE["pt"] * PHONE_READABLE_PX / (72 * FIGURE["width"])
    if phone_px < SIZES["figure-text"][0]:
        raise SystemExit(f"FIGURE draws text at {phone_px:.1f} px on a phone.")
    light = {k: v[0].lstrip("#") for k, v in TOKENS.items()}
    cycle = ", ".join(f"'{c.lstrip('#')}'" for c in DATA["cat"][0])
    lines = {
        "figure.dpi": 144,
        "figure.facecolor": light["bg"],
        "axes.facecolor": light["bg"],
        "savefig.facecolor": light["bg"],
        "axes.prop_cycle": f"cycler('color', [{cycle}])",
        "text.color": light["ink"],
        "axes.labelcolor": light["ink"],
        "axes.edgecolor": light["ink-2"],
        "xtick.color": light["ink-2"],
        "ytick.color": light["ink-2"],
        "grid.color": light["rule"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.handlelength": 1.5,  # compact legends fit above a phone-width plot
        "legend.columnspacing": 1.0,
        "figure.figsize": f"{FIGURE['width']}, {FIGURE['height']}",
        "figure.constrained_layout.use": True,
        "font.size": FIGURE["pt"],
        "axes.titlesize": "medium",
        "lines.linewidth": 1.8,
    }
    header = "# Written by docs/_scripts/brand.py; edit the tokens there.\n"
    return header + "".join(f"{k}: {v}\n" for k, v in lines.items())


def _lum(c: str) -> float:
    v = [int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    v = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]


def contrast(a: str, b: str) -> float:
    """WCAG 2 contrast ratio of two colours."""
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _used_at(name: str) -> str:
    names = list(CUTS)
    px = CUTS[name]["px"]
    upper = CUTS[names[names.index(name) - 1]]["px"] if names.index(name) else None
    if upper is None:
        return f"{px} px and up"
    return f"below {upper} px" if px == 0 else f"{px} to {upper - 1} px"


def tables() -> dict[str, str]:
    """The generated tables of brand.md, by marker name."""
    t = ["| Token | Role | Light | Contrast | Dark | Contrast | Read against |", "|---" * 7 + "|"]
    for name, (light, dark, role, ref) in TOKENS.items():
        cl, cd = contrast(light, TOKENS[ref][0]), contrast(dark, TOKENS[ref][1])
        t.append(f"| `{name}` | {role} | `{light}` | {cl:.1f} | `{dark}` | {cd:.1f} | `{ref}` |")
    for kind, (light, dark, use) in DATA.items():
        lc, dc = " ".join(f"`{c}`" for c in light), " ".join(f"`{c}`" for c in dark)
        t.append(f"| `{kind}-0…{len(light) - 1}` | {use} | {lc} | | {dc} | | |")
    c = ["| Cut | Used at | Bins | Draws | Files |", "|---|---|---|---|---|"]
    c += [
        f"| {n} | {_used_at(n)} | {p['nb']} | {p['nd']} | `brand/mark-{n}-{{light,dark}}.svg` |"
        for n, p in CUTS.items()
    ]
    ty = ["| Role | Face | Weight | Size | Line height |", "|---|---|---|---|---|"]
    for role, face, w, px, lh in TYPE.values():
        ty.append(f"| {role} | {FONTS[face][0]} | {w} | {_f(px)} px | {lh} |")
    sz = ["| Size | Value | Use |", "|---|---|---|"]
    sz += [f"| `{k}` | {v} px | {use} |" for k, (v, use) in SIZES.items()]
    sz.append(f"| mark : wordmark | {MARK_EM} | the mark's height over the wordmark's font size |")
    sz.append(f"| `figure-max` | {FIGURE_MAX_REM} rem | the widest a plot is shown |")
    return {k: "\n".join(v) for k, v in {"tokens": t, "cuts": c, "type": ty, "sizes": sz}.items()}


def outputs() -> dict[Path, str]:
    out = {
        STATIC / "tokens.css": tokens_css(),
        STATIC / "brand" / "favicon.svg": favicon_svg(),
        MATPLOTLIBRC: matplotlibrc(),
        HERO: hero_html(),
    }
    for theme, tag in ((0, "light"), (1, "dark")):
        for cut in CUTS:
            out[STATIC / "brand" / f"mark-{cut}-{tag}.svg"] = mark_svg(cut, theme)
        out[STATIC / "brand" / f"lockup-{tag}.svg"] = lockup_svg(theme, "mid")
        out[STATIC / "brand" / f"lockup-large-{tag}.svg"] = lockup_svg(theme, "full")
    md = BRAND_MD.read_text(encoding="utf-8")
    for name, table in tables().items():
        start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        i, j = md.index(start) + len(start), md.index(end)
        md = md[:i] + "\n" + table + "\n" + md[j:]
    out[BRAND_MD] = md
    return out


def sample_color_problems() -> list[str]:
    """Whether ``binference.plotting.SAMPLE_COLOR``, the package's copy of the ``sample``
    token, still equals it."""
    found = re.findall(r'^SAMPLE_COLOR = "(#[0-9A-Fa-f]{6})"$', PLOTTING.read_text(), re.M)
    if len(found) != 1:
        return [f"{PLOTTING.name} must define SAMPLE_COLOR once as a hex string"]
    if found[0].upper() != TOKENS["sample"][0].upper():
        token = TOKENS["sample"][0]
        return [f"SAMPLE_COLOR is {found[0]} in {PLOTTING.name}, the sample token {token}"]
    return []


def font_problems() -> list[str]:
    """What in ``static/fonts/`` disagrees with ``FONTS``: missing faces or other weights.
    (fetch_fonts.py writes those files; it needs the network, so it is not run here.)"""
    css = (STATIC / "fonts" / "fonts.css").read_text(encoding="utf-8")
    problems = []
    for family, _, weights, _ in FONTS.values():
        problems += [
            f"no {family} {subset} file (run docs/_scripts/fetch_fonts.py)"
            for subset in ("latin", "latin-ext")
            if not (STATIC / "fonts" / f"{family.replace(' ', '')}-{subset}.woff2").exists()
        ]
        served = re.findall(rf"font-family: '{family}';.*?font-weight: ([\d ]+);", css, re.S)
        if served != [weights.replace("..", " ")] * 2:
            problems.append(f"fonts.css does not serve {family} at weights {weights}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if a committed output is stale")
    args = ap.parse_args(argv)
    stale = []
    for path, text in outputs().items():
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        elif not path.exists() or path.read_text(encoding="utf-8") != text:
            stale.append(f"{path.relative_to(DOCS.parent)} is stale (run docs/_scripts/brand.py)")
    stale += font_problems() + sample_color_problems()
    if stale:
        print(*stale, sep="\n")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
