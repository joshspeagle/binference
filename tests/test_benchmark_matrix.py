"""The benchmark matrix script runs end to end (its quick mode) and reports every engine."""

import json


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
