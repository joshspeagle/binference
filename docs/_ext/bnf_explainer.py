"""The ``{explainer}`` directive: an interactive figure with a still frame drawn before JavaScript.

```{explainer} v1
Caption in MyST.
```

inlines frame ``v1`` from ``explainers.json`` (written by ``docs/_scripts/explainers.mjs``) as
figure V-1, with its controls disabled; ``_static/explainers/run.mjs`` enables them on pages
that have an explainer.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

JSON = Path(__file__).parent / "explainers.json"
FRAMES = json.loads(JSON.read_text())


def _control(fid: str, c: dict) -> str:
    value = f"{c['value']:.{c['digits']}f}"
    return (
        f'<label for="{fid}-{c["name"]}">{escape(c["label"])}</label>'
        f'<input type="range" id="{fid}-{c["name"]}" name="{c["name"]}" min="{c["min"]}" '
        f'max="{c["max"]}" step="{c["step"]}" value="{c["value"]}" disabled>'
        f'<output for="{fid}-{c["name"]}">{value}</output>'
    )


class Explainer(SphinxDirective):
    required_arguments = 1
    has_content = True

    def run(self):
        name = self.arguments[0]
        if name not in FRAMES:
            raise self.error(f"no explainer {name!r}: run node docs/_scripts/explainers.mjs")
        self.env.note_dependency(str(JSON))  # a redrawn frame rebuilds the page
        frame, fid = FRAMES[name], f"v-{name[1:]}"
        controls = "".join(_control(fid, c) for c in frame["controls"])
        readout = escape(frame["readout"])
        head = (
            f'<figure class="explainer" id="{fid}" data-explainer="{name}">'
            f'<p class="explainer-title">V-{name[1:]} · {escape(frame["title"])}</p>'
            f'<div class="explainer-plot">{frame["svg"]}</div>'
            f'<div class="explainer-controls">{controls}</div>'
            f'<p class="explainer-readout" aria-live="polite">{readout}</p><figcaption>'
        )
        caption = nodes.container()
        self.state.nested_parse(self.content, self.content_offset, caption)
        raw = lambda s: nodes.raw("", s, format="html")  # noqa: E731
        return [raw(head), *caption.children, raw("</figcaption></figure>")]


def _scripts(app, pagename, templatename, context, doctree) -> None:
    if doctree is not None and any(
        'data-explainer="' in n.astext() for n in doctree.findall(nodes.raw)
    ):
        app.add_js_file("explainers/run.mjs", type="module")


def setup(app):
    app.add_directive("explainer", Explainer)
    app.connect("html-page-context", _scripts)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
