"""The API reference pages list every public name exactly once."""

import ast
import importlib
import pkgutil
import re
import types
from collections import Counter
from pathlib import Path

import pytest

import binference

DOCS = Path(__file__).parents[1] / "docs"
API_PAGES = DOCS / "reference" / "api"
if not DOCS.exists():
    pytest.skip("docs/ is not next to the tests (sdist)", allow_module_level=True)
BLOCK = re.compile(r"^```\{eval-rst\}\n(.*?)^```", re.S | re.M)
CURRENT = re.compile(r"^\.\. currentmodule:: (\S+)", re.M)
ENTRIES = re.compile(r"^\.\. autosummary::\n((?:   .*\n|\n)*)", re.M)


def documented_names():
    """Fully qualified names in the autosummary lists of the API pages."""
    names = []
    for page in sorted(API_PAGES.glob("*.md")):
        for block in BLOCK.findall(page.read_text(encoding="utf-8")):
            module = CURRENT.search(block).group(1)
            for entries in ENTRIES.findall(block):
                for line in entries.splitlines():
                    entry = line.strip()
                    if entry and not entry.startswith(":"):
                        names.append(f"{module}.{entry}")
    return names


def public_names():
    """Names in ``binference.__all__`` and in every public submodule's ``__all__``.

    A name re-exported elsewhere is the same object, so it counts once; submodules
    and ``__version__`` are not API entries.
    """
    names = {}
    modules = [binference] + [
        importlib.import_module(f"binference.{info.name}")
        for info in pkgutil.iter_modules(binference.__path__)
        if not info.name.startswith("_")
    ]
    for module in modules:
        for name in module.__all__:
            obj = getattr(module, name)
            if name != "__version__" and not isinstance(obj, types.ModuleType):
                names.setdefault(id(obj), f"{module.__name__}.{name}")
    return names


def test_every_public_name_appears_exactly_once_in_the_api_pages():
    documented = Counter(id(pkgutil.resolve_name(n)) for n in documented_names())
    public = public_names()
    assert not public.keys() - documented.keys(), sorted(
        public[k] for k in public.keys() - documented.keys()
    )
    assert documented.keys() <= public.keys(), "an API page lists a name outside any __all__"
    assert all(n == 1 for n in documented.values()), "an API page lists a name twice"


def test_public_docstrings_cite_no_design_record():
    # the site leaves out notes/review/DECISIONS.md, so a citation there would dangle
    package = Path(binference.__file__).parent
    citing = []
    for path in sorted(package.rglob("*.py")):
        parts = path.relative_to(package).with_suffix("").parts
        if any(p.startswith("_") and p != "__init__" for p in parts):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = getattr(node, "name", "<module>")
            if (
                isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and (name == "__init__" or not name.startswith("_"))
                and "DECISIONS" in (ast.get_docstring(node) or "")
            ):
                citing.append(f"{path.relative_to(package.parent)}:{node.body[0].lineno} {name}")
    assert not citing, citing
