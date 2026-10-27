"""Author-year citations in round brackets: "(Hogg et al. 2010; Malz 2021)" and
"Leistedt et al. (2016)", as the docs wrote them by hand (``bibtex_reference_style``)."""

from __future__ import annotations

import dataclasses

from sphinxcontrib.bibtex.plugin import register_plugin
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle


def _round(sep: str = ", ") -> BracketStyle:
    return BracketStyle(left="(", right=")", sep=sep)


@dataclasses.dataclass
class RoundAuthorYear(AuthorYearReferenceStyle):
    bracket_parenthetical: BracketStyle = dataclasses.field(default_factory=lambda: _round("; "))
    bracket_textual: BracketStyle = dataclasses.field(default_factory=_round)
    bracket_author: BracketStyle = dataclasses.field(default_factory=_round)
    author_year_sep: str = " "


def setup(app):
    register_plugin("sphinxcontrib.bibtex.style.referencing", "author_year_round", RoundAuthorYear)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
