"""The API reference pages list every public name exactly once."""

import importlib
import pkgutil
import re
import types
from collections import Counter
from pathlib import Path

import binference

API_PAGES = Path(__file__).parents[1] / "docs" / "reference" / "api"
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


def resolve(qualname):
    module, _, name = qualname.rpartition(".")
    return getattr(importlib.import_module(module), name)


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
    documented = Counter(id(resolve(n)) for n in documented_names())
    public = public_names()
    assert not public.keys() - documented.keys(), sorted(
        public[k] for k in public.keys() - documented.keys()
    )
    assert documented.keys() <= public.keys(), "an API page lists a name outside any __all__"
    assert all(n == 1 for n in documented.values()), "an API page lists a name twice"
