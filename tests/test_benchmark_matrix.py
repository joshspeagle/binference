"""The benchmark matrix script runs end to end (its quick mode) and reports every engine."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MATRIX = Path(__file__).parents[1] / "benchmarks" / "matrix.py"


@pytest.fixture(scope="module")
def matrix():
    spec = importlib.util.spec_from_file_location("matrix", MATRIX)
    module = importlib.util.module_from_spec(spec)
    sys.modules["matrix"] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def test_the_quick_matrix_runs_every_engine_and_reports_it(matrix, tmp_path):
    out = tmp_path / "m.json"
    matrix.run(out, quick=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    engines = [r["engine"] for r in rows]
    assert engines == ["header", "reference", *matrix.ENGINES]
    assert not [r for r in rows if "error" in r]
    nuts = rows[2]
    assert nuts["mean_err"] < 1.0 and nuts["ess_per_s"] > 0
    table = matrix.report(out)
    assert all(f"| {e} |" in table for e in matrix.ENGINES)
