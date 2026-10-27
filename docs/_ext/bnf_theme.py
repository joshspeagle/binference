"""Template context and directives for the binference docs theme (``docs/_themes/binference``).

* The top bar's sections are the root toctrees with a caption; while the root has none,
  its entries are the sections. The current section is the one listing the page's top
  ancestor.
* Pages with two or more sections get the "On this page" outline.
* A page whose front matter says ``layout: home`` uses ``home.html``.
* ``{plain-words}`` is the "In plain words" note of concept pages (at most one per page).
"""

from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from sphinx import addnodes


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

    captioned = [t for t in toctrees if t.get("caption") and docs(t)]
    if captioned:
        app.bnf_sections = [
            {"caption": t["caption"], "docname": docs(t)[0], "docs": docs(t)} for t in captioned
        ]
        return
    entries = [d for t in toctrees for d in docs(t)]
    app.bnf_sections = [
        {"caption": env.titles[d].astext(), "docname": d, "docs": [d]} for d in entries
    ]


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
