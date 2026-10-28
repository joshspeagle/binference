"""The ``{explainer}`` directive: an interactive figure with a still frame drawn before JavaScript.

```{explainer} v1
Caption in MyST.
```

inlines frame ``v1`` from ``explainers.json`` (written by ``docs/_scripts/explainers.mjs``) as
figure V-1, with its controls disabled, and labels it ``explainer-v1``, so a page cites it as
{ref}`explainer-v1` ("V-1"). ``_static/explainers/run.mjs`` enables the controls on pages that
have an explainer.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

JSON = Path(__file__).parent / "explainers.json"


def _control(fid: str, c: dict) -> str:
    cid = f"{fid}-{c['name']}"
    value = f"{c['value']:.{len(str(c['step']).partition('.')[2])}f}"
    return (
        f'<label for="{cid}">{escape(c["label"])}</label>'
        f'<input type="range" id="{cid}" name="{c["name"]}" min="{c["min"]}" max="{c["max"]}" '
        f'step="{c["step"]}" value="{c["value"]}" aria-valuetext="{value}" disabled>'
        f'<output for="{cid}">{value}</output>'
    )


class Explainer(SphinxDirective):
    required_arguments = 1
    has_content = True

    def run(self):
        name = self.arguments[0]
        self.env.note_dependency(str(JSON))  # a redrawn frame rebuilds the page
        frames = json.loads(JSON.read_text())
        if name not in frames:
            raise self.error(f"no explainer {name!r}: run node docs/_scripts/explainers.mjs")
        frame, number = frames[name], f"V-{name[1:]}"
        fid = number.lower()
        self.env.get_domain("std").note_hyperlink_target(
            f"explainer-{name}", self.env.docname, fid, number
        )
        controls = "".join(_control(fid, c) for c in frame["controls"])
        head = (
            f'<figure class="explainer" id="{fid}" data-explainer="{name}" '
            f'aria-labelledby="{fid}-title">'
            f'<p class="explainer-title" id="{fid}-title">{number} · {escape(frame["title"])}</p>'
            f'<div class="explainer-plot">{frame["svg"]}</div>'
            f'<div class="explainer-controls">{controls}</div>'
            f'<p class="explainer-readout" aria-live="polite">{escape(frame["readout"])}</p>'
            "<figcaption>"
        )
        caption = nodes.container()
        self.state.nested_parse(self.content, self.content_offset, caption)
        raw = lambda s: nodes.raw("", s, format="html")  # noqa: E731
        return [raw(head), *caption.children, raw("</figcaption></figure>")]


def _scripts(app, pagename, templatename, context, doctree) -> None:
    if 'data-explainer="' in context.get("body", ""):
        app.add_js_file("explainers/run.mjs", type="module")


def setup(app):
    app.add_directive("explainer", Explainer)
    app.connect("html-page-context", _scripts)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
