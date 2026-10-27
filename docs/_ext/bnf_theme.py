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


def _sections(app) -> list[dict]:
    env = app.env
    toctrees = list(env.get_doctree(app.config.root_doc).findall(addnodes.toctree))
    captioned = [t for t in toctrees if t.get("caption")]
    if captioned:
        return [
            {
                "caption": t["caption"],
                "docname": t["entries"][0][1],
                "docs": [d for _, d in t["entries"]],
            }
            for t in captioned
        ]
    entries = [d for t in toctrees for _, d in t["entries"]]
    return [{"caption": env.titles[d].astext(), "docname": d, "docs": [d]} for d in entries]


def _page_context(app, pagename, templatename, context, doctree):
    sections = _sections(app)
    context["bnf_sections"] = sections
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
    app.add_js_file("binference.js", loading_method="defer")
    app.connect("html-page-context", _page_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
