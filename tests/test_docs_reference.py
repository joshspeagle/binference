"""The reference page of defaults states the values the code uses (issue #111)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs" / "_scripts" / "defaults_table.py"


def test_the_defaults_page_states_the_defaults_of_the_code():
    if not SCRIPT.exists():  # an sdist ships the tests without the docs
        pytest.skip("docs/ is not present")
    spec = importlib.util.spec_from_file_location("defaults_table", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.current() == script.render(), (
        "docs/reference/defaults.md is stale: run python docs/_scripts/defaults_table.py"
    )
