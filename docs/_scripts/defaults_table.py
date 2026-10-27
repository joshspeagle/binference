"""Write the tables of docs/reference/defaults.md from the code.

Usage: python docs/_scripts/defaults_table.py [--check]

Lists every module-level numeric constant of binference that has a ``#:`` doc comment,
with that comment as its description, between the page's ``<!-- defaults:start -->``
and ``<!-- defaults:end -->`` markers. ``--check`` writes nothing and exits with
status 1 when the page differs from what the script would write; CI runs it.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
from pathlib import Path
from typing import Any

from sphinx.pycode import ModuleAnalyzer

import binference

PAGE = Path(__file__).resolve().parents[1] / "reference" / "defaults.md"
START, END = "<!-- defaults:start -->", "<!-- defaults:end -->"

#: (section title, module prefixes), in page order; every listed module must match one
GROUPS = [
    ("Automatic engine choice", ("binference.inference",)),
    ("Results and diagnostics", ("binference.results", "binference.diagnostics")),
    ("Engines", ("binference.samplers",)),
    ("Priors", ("binference.priors",)),
    (
        "Inputs",
        ("binference.likelihoods", "binference._", "binference.basis", "binference.response"),
    ),
    ("Selection", ("binference.selection",)),
    ("Validation", ("binference.validation",)),
]


def numeric(value: Any) -> bool:
    """A number, or a container of numbers (and ``None``): a threshold, not a registry."""
    if isinstance(value, dict):
        return all(map(numeric, value.values()))
    if isinstance(value, tuple | list):
        return all(v is None or numeric(v) for v in value)
    return isinstance(value, int | float) and not isinstance(value, bool)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return format(value, "g")
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k!r}: {fmt(v)}" for k, v in value.items()) + "}"
    if isinstance(value, tuple | list):
        return "(" + ", ".join(fmt(v) for v in value) + ")"
    return repr(value)


def markdown(text: str) -> str:
    """reStructuredText roles and literals as MyST code, safe in a table cell."""
    text = re.sub(r":\w+:`~?([^`]+)`", r"`\1`", text).replace("``", "`")
    return text.replace("|", "\\|")


def constants() -> list[tuple[str, str, str, str]]:
    """(module, name, value, description) of every documented numeric constant."""
    rows = []
    for info in pkgutil.walk_packages(binference.__path__, "binference."):
        module = importlib.import_module(info.name)
        docs = ModuleAnalyzer.for_module(info.name).find_attr_docs()
        for (scope, name), lines in docs.items():
            value = getattr(module, name, None)
            if scope or not re.fullmatch(r"_?[A-Z][A-Z0-9_]*", name) or not numeric(value):
                continue
            rows.append((info.name, name, fmt(value), " ".join(" ".join(lines).split())))
    return rows


def render() -> str:
    rows, lines = constants(), [START, ""]
    for title, prefixes in GROUPS:
        group = [r for r in rows if r[0].startswith(prefixes)]
        if group:
            lines += [f"## {title}", "", "| Name | Value | Module | What it controls |"]
            lines.append("|---|---|---|---|")
            for module, name, value, text in group:
                lines.append(f"| `{name}` | `{value}` | `{module}` | {markdown(text)} |")
            lines.append("")
    if unplaced := {r[0] for r in rows if not any(r[0].startswith(p) for _, p in GROUPS)}:
        raise ValueError(f"add {sorted(unplaced)} to GROUPS in {__file__}")
    return "\n".join([*lines, END])


def main(argv: list[str]) -> int:
    text = PAGE.read_text(encoding="utf-8")
    i, j = text.index(START), text.index(END) + len(END)
    new = text[:i] + render() + text[j:]
    if "--check" in argv:
        if new != text:
            print(f"{PAGE} is stale: run python docs/_scripts/defaults_table.py")
            return 1
        return 0
    PAGE.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
