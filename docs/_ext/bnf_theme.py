"""Template context and directives for the binference docs theme (``docs/_themes/binference``).

* The top bar's sections are the root toctrees with a caption. The current section is the
  one listing the page's top ancestor.
* Pages with two or more sections get the "On this page" outline.
* A page whose front matter says ``layout: home`` uses ``home.html``.
* ``{plain-words}`` is the "In plain words" note of concept pages (at most one per page).
* Source pages link back to the anchors recorded by the Python domain.
"""

from __future__ import annotations

import re
from collections import defaultdict
from html import escape

from docutils import nodes
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from sphinx import addnodes
from sphinx.util import logging

logger = logging.getLogger(__name__)

_VIEWCODE_BACKLINK = re.compile(
    r'(<div class="viewcode-block" id="([^"]+)">\s*<a class="viewcode-back" href=")'
    r'([^"#]+)#([^"]+)"'
)


class PlainWords(BaseAdmonition):
    """The "In plain words" note: one student-level summary on a concept page."""

    node_class = nodes.admonition
    required_arguments = 0

    def run(self):
        self.arguments = ["In plain words"]
        self.options["class"] = [*self.options.get("class", []), "plain-words"]
        return super().run()


def _sections(app, env) -> None:
    # not env.master_doctree: it is cached in the pickled environment and goes stale
    toctrees = list(env.get_doctree(app.config.root_doc).findall(addnodes.toctree))

    def docs(t):  # documents only: entries can also be URLs or "self"
        return [d for _, d in t["entries"] if d in env.titles]

    app.bnf_sections = [
        {"caption": t["caption"], "docname": docs(t)[0], "docs": docs(t)}
        for t in toctrees
        if t.get("caption") and docs(t)
    ]
    if not app.bnf_sections:
        logger.warning(
            "the root document has no captioned toctree, so the top bar has no sections"
        )


def _level(html: str) -> str:
    """Fix two roles Sphinx and docutils write: toctree captions are headings without a level,
    and the bibliography is a list whose children sphinxcontrib-bibtex makes link targets."""
    html = html.replace(
        'class="caption" role="heading">', 'class="caption" role="heading" aria-level="2">'
    )
    if 'role="heading">' in html:
        logger.warning("a heading role without a level remains: Sphinx's caption markup changed")
    return html.replace('<div role="list" class="citation-list">', '<div class="citation-list">')


def _source_backlinks(app, pagename: str, body: str) -> str:
    """Resolve each viewcode backlink against its destination's Python anchors."""
    anchors_by_uri: dict[str, set[str]] = defaultdict(set)
    for _, _, _, docname, anchor, _ in app.env.get_domain("py").get_objects():
        if anchor:
            uri = app.builder.get_relative_uri(pagename, docname)
            anchors_by_uri[uri].add(anchor)

    def resolve(match: re.Match[str]) -> str:
        prefix, source_name, uri, fragment = match.groups()
        anchors = anchors_by_uri[uri]
        if fragment in anchors:
            return match[0]
        # viewcode uses one documented module prefix per source page. Re-exports
        # can share that source page with objects documented under another prefix.
        suffix = f".{source_name}"
        candidates = [anchor for anchor in anchors if anchor.endswith(suffix)]
        if len(candidates) == 1:
            return f'{prefix}{uri}#{escape(candidates[0])}"'
        logger.warning("cannot resolve source backlink %s#%s on %s", uri, fragment, pagename)
        return match[0]

    rewritten, count = _VIEWCODE_BACKLINK.subn(resolve, body)
    if count != body.count('class="viewcode-back"'):
        logger.warning("cannot read all source backlinks on %s: viewcode markup changed", pagename)
    return rewritten


def _page_context(app, pagename, templatename, context, doctree):
    sections = context["bnf_sections"] = app.bnf_sections
    parents, top = app.builder.relations, pagename
    while top in parents and parents[top][0] not in (app.config.root_doc, None):
        top = parents[top][0]
    context["bnf_section_index"] = next(
        (k for k, s in enumerate(sections) if top in s["docs"]), None
    )
    toc = app.env.tocs.get(pagename)
    sections_here = toc.next_node(nodes.bullet_list) if toc else None
    context["bnf_outline"] = sections_here is not None and len(sections_here) > 1
    # Sphinx marks toctree captions as headings without a level, in the sidebar and in pages
    if "body" in context:
        body = _level(context["body"])
        context["body"] = (
            _source_backlinks(app, pagename, body) if pagename.startswith("_modules/") else body
        )
    if "toctree" in context:
        toctree = context["toctree"]
        context["toctree"] = lambda **kw: _level(toctree(**kw))
    if (context.get("meta") or {}).get("layout") == "home":
        return "home.html"
    return None


def setup(app):
    app.add_directive("plain-words", PlainWords)
    # last, so the theme wins over the extensions' stylesheets at equal specificity
    app.add_css_file("binference.css", priority=900)
    app.add_js_file("binference.js", loading_method="defer")
    app.connect("env-updated", _sections)
    app.connect("html-page-context", _page_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
