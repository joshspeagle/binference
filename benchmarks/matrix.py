"""Benchmark matrix: scenario x prior x engine (DECISIONS #14, #39, #42).

Each scenario is a mock catalogue that varies one factor around a baseline (1-D, K = 20,
N = 1000, Gaussian likelihoods one bin wide, a bimodal population drawn uniformly within
bins, no selection). For every scenario and prior, a long NUTS run is the reference, and
every engine is timed and compared with it. The results set the rules of
``method="auto"`` and are summarised in ``docs/how-it-works/auto.md``.

Run from the repository root::

    python benchmarks/matrix.py run --jobs 4               # the full matrix (90 min)
    python benchmarks/matrix.py run --quick                # a two-cell smoke run (seconds)
    python benchmarks/matrix.py report benchmarks/results/matrix.json

``run`` appends one JSON line per cell to ``--out`` (default
``benchmarks/results/matrix.json``), so an interrupted run keeps its cells; ``--only``
selects scenarios by substring. Every worker uses one BLAS thread.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import platform
import time
import warnings
import zlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

import binference
from binference import Bins, Likelihoods, PopulationModel
from binference import diagnostics as diag
from binference import priors as P
from binference.simulate import mock_1d, mock_2d, mock_categorical, population

ENGINES = ("nuts", "gibbs", "laplace", "vb", "map", "bootstrap")
REFERENCE = {"n_chains": 4, "n_warmup": 2000, "n_samples": 5000}
QUICK_REFERENCE = {"n_chains": 2, "n_warmup": 100, "n_samples": 200}
QUICK_ENGINE = {"nuts": {"n_chains": 2, "n_warmup": 100, "n_samples": 100}}


@dataclass
class Scenario:
    """A mock catalogue builder, the priors it is fitted with, and the true fractions."""

    build: Callable[[np.random.Generator], tuple[Likelihoods, np.ndarray]]
    priors: dict[str, Callable[[], object]]
    note: str = ""
    seed: int = 0


# -- populations -------------------------------------------------------------------------


def _bimodal(bins):
    return population(bins, "bimodal")


def _sparse(bins):
    # the bimodal population with its smaller half of bins emptied
    rho = _bimodal(bins)
    rho[rho < np.median(rho)] = 0.0
    return rho / rho.sum()


def _spiky(bins):
    # half the population in one bin, the rest bimodal
    rho = 0.5 * _bimodal(bins)
    rho[bins.K // 3] += 0.5
    return rho


# -- catalogue builders ------------------------------------------------------------------


def _gauss(K=20, N=1000, width=1.0, pop=_bimodal, **mock):
    """1-D Gaussian measurements ``width`` bins wide, values uniform within bins."""

    def build(rng):
        bins = Bins.linear(0.0, 1.0, K)
        rho = pop(bins)
        m = mock_1d(N, bins, density=rho, noise=width / K, rng=rng, **mock)
        return m.likelihoods, rho

    return build


def _interim(correct: bool):
    """Posteriors computed under a broad Gaussian interim prior, divided by it (correct)
    or by a flat prior (mis-specified)."""

    def build(rng):
        bins = Bins.linear(0.0, 1.0, 20)
        rho = _bimodal(bins)
        m = mock_1d(1000, bins, density=rho, noise=1 / 20, interim="gaussian", rng=rng)
        interim = m.interim if correct else np.full(bins.K, 1 / bins.K)
        return Likelihoods.from_table(m.posteriors, bins, interim=interim), rho

    return build


def _samples(S: int):
    def build(rng):
        bins = Bins.linear(0.0, 1.0, 20)
        rho = _bimodal(bins)
        m = mock_1d(1000, bins, density=rho, noise=1 / 20, n_samples=S, rng=rng)
        return Likelihoods.from_samples(m.samples, bins, interim="flat"), rho

    return build


def _grid(rng):
    edges = np.linspace(0.0, 1.0, 9)
    bins = Bins.grid(edges, edges)
    rho = population(bins, "banana")
    m = mock_2d(1000, bins, density=rho, noise=1 / 8, rng=rng)
    return m.likelihoods, rho


def _categorical(rng):
    prevalence = np.array([0.4, 0.25, 0.2, 0.1, 0.05])
    m = mock_categorical(1000, 5, prevalence=prevalence, separation=1.0, rng=rng)
    return m.likelihoods, prevalence


def _rail_like(rng):
    """Photo-z-like catalogue: K = 30 bins on [0, 3], N = 5000, Gaussian likelihoods of
    width 0.05 (1 + z), a second peak (a colour degeneracy) for 20% of objects and a
    uniform floor for 3% (catastrophic outliers)."""
    bins = Bins.linear(0.0, 3.0, 30)
    rho = population(bins, "bimodal")
    N = 5000
    labels = rng.choice(bins.K, size=N, p=rho)
    z = bins.sample_within(labels, rng)
    sigma = 0.05 * (1 + z)
    obs = z + sigma * rng.standard_normal(N)
    alias = np.where(rng.random(N) < 0.2, rng.uniform(0.0, 3.0, N), np.nan)
    e = bins.edges[0]

    def peak(mu):
        return np.diff(stats.norm.cdf((e[None, :] - mu[:, None]) / sigma[:, None]), axis=1)

    L = peak(obs) + 0.5 * np.nan_to_num(peak(np.nan_to_num(alias, nan=-10.0)))
    L = L / L.sum(1, keepdims=True) + 0.03 / bins.K
    return Likelihoods.from_table(L, bins, kind="likelihood", representation="mass"), rho


# -- the matrix --------------------------------------------------------------------------

FLAT = {"dirichlet": lambda: P.Dirichlet(1.0)}
SMOOTH = {**FLAT, "randomwalk": lambda: P.LogisticRandomWalk()}
ALL = {
    **SMOOTH,
    "dirichlet0.1": lambda: P.Dirichlet(0.1),
    "gp": lambda: P.LogisticGP(),
    "polyatree": lambda: P.PolyaTree(),
}
DETECT = 1.0 - 0.6 * np.linspace(0.0, 1.0, 20)  # detection falling from 1 to 0.4

SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario(_gauss(), ALL, "K=20, N=1000, width 1 bin"),
    "width-0.3": Scenario(_gauss(width=0.3), ALL, "narrow likelihoods"),
    "width-3": Scenario(_gauss(width=3.0), ALL, "broad likelihoods"),
    "n-100": Scenario(_gauss(N=100), SMOOTH, "few objects"),
    "n-10000": Scenario(_gauss(N=10000), SMOOTH, "many objects"),
    "k-10": Scenario(_gauss(K=10), SMOOTH, "few bins"),
    "k-50": Scenario(_gauss(K=50), SMOOTH, "many bins"),
    "pop-sparse": Scenario(_gauss(pop=_sparse), SMOOTH, "half the bins empty"),
    "pop-spiky": Scenario(_gauss(pop=_spiky), SMOOTH, "half the population in one bin"),
    "sel-truth": Scenario(_gauss(selection=DETECT), SMOOTH, "selection on the truth"),
    "sel-measured": Scenario(
        _gauss(measured_range=(None, 0.7)), SMOOTH, "selection on the measurement"
    ),
    "interim-correct": Scenario(_interim(True), FLAT, "posteriors, interim divided out"),
    "interim-wrong": Scenario(_interim(False), FLAT, "posteriors, wrong interim divided out"),
    "samples-20": Scenario(_samples(20), FLAT, "20 posterior samples per object"),
    "samples-200": Scenario(_samples(200), FLAT, "200 posterior samples per object"),
    "grid-8x8": Scenario(_grid, SMOOTH, "2-D grid, width 1 cell"),
    "categorical": Scenario(_categorical, FLAT, "classifier outputs, 5 classes"),
    "rail-like": Scenario(_rail_like, ALL, "photo-z-like, K=30, N=5000"),
}

QUICK = {
    "quick": Scenario(_gauss(K=5, N=60), {"dirichlet": FLAT["dirichlet"]}, "smoke test"),
}


# -- one cell ----------------------------------------------------------------------------


def _summaries(res) -> dict[str, list[float]]:
    q05, q95 = res.quantile([0.05, 0.95])
    return {
        "mean": res.mean.tolist(),
        "sd": res.std.tolist(),
        "q05": q05.tolist(),
        "q95": q95.tolist(),
    }


def _mixing(res) -> dict:
    d = res.diagnostics
    return {
        "ess": float(np.min(d["ess_bulk"])),
        "rhat_max": float(np.nanmax(d["rhat"])),
        "n_divergent": int(d.get("n_divergent", 0)),
    }


def _fit(model, engine, rng, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = time.perf_counter()
        res = binference.infer(model, method=engine, rng=rng, check=False, **kw)
        seconds = time.perf_counter() - t
    names = sorted({type(w.message).__name__ for w in caught})
    return res, seconds, names


def _reference(model, quick: bool, seed: int) -> dict:
    res, seconds, names = _fit(model, "nuts", seed, **(QUICK_REFERENCE if quick else REFERENCE))
    C = res.chains
    return {
        **_summaries(res),
        **_mixing(res),
        # Monte Carlo errors per bin, of the mean and of the s.d. (delta method)
        "mcse_mean": [diag.mcse(C[..., k]) for k in range(res.K)],
        "mcse_sd": [
            diag.mcse((C[..., k] - C[..., k].mean()) ** 2) / (2 * C[..., k].std())
            for k in range(res.K)
        ],
        "seconds": seconds,
        "warnings": names,
    }


def _compare(s: dict, ref: dict, truth: np.ndarray) -> dict:
    sd = np.asarray(ref["sd"])
    out = {"mean_err": float(np.max(np.abs(np.asarray(s["mean"]) - ref["mean"]) / sd))}
    out["truth_z"] = float(np.max(np.abs(np.asarray(ref["mean"]) - truth) / sd))
    if "sd" in s:
        ratio = np.asarray(s["sd"]) / sd
        out["sd_ratio_min"], out["sd_ratio_max"] = float(ratio.min()), float(ratio.max())
        out["q_err"] = float(
            max(np.max(np.abs(np.asarray(s[q]) - ref[q]) / sd) for q in ("q05", "q95"))
        )
    return out


def catalogue(name: str, quick: bool = False) -> tuple[Likelihoods, np.ndarray]:
    """The likelihoods and true fractions of a scenario, from its own seed."""
    sc = (QUICK if quick else SCENARIOS)[name]
    return sc.build(np.random.default_rng([sc.seed, zlib.crc32(name.encode())]))


def run_cell(name: str, prior_name: str, quick: bool = False) -> list[dict]:
    """Every engine on one (scenario, prior), each compared with the reference run."""
    sc = (QUICK if quick else SCENARIOS)[name]
    lk, truth = catalogue(name, quick)
    prior = sc.priors[prior_name]()
    model = PopulationModel(lk, prior)
    base = {
        "scenario": name,
        "prior": prior_name,
        "note": sc.note,
        "N": int(lk.N),
        "K": int(lk.K),
        "nnz": int(lk.nnz),
        "conjugate": bool(model.prior.conjugate),
    }
    est = binference.infer(model, method="map")
    try:
        base["missing_information"] = float(model.missing_information(est.rho))
    except Exception:  # a feature for the auto rules, not a result
        base["missing_information"] = float("nan")
    ref = _reference(model, quick, seed=1)
    rows = [{**base, "engine": "reference", **ref}]
    for engine in ENGINES:
        row = {**base, "engine": engine}
        try:
            kw = QUICK_ENGINE.get(engine, {}) if quick else {}
            res, seconds, names = _fit(model, engine, 2, **kw)
        except Exception as exc:  # an engine that cannot run is a result
            rows.append({**row, "error": f"{type(exc).__name__}: {exc}"})
            continue
        row.update(seconds=seconds, warnings=names)
        if engine == "map":
            row.update(_compare({"mean": res.rho}, ref, truth))
        else:
            row.update(_compare(_summaries(res), ref, truth))
            if engine in ("nuts", "gibbs"):
                row.update(_mixing(res))
            elif engine == "laplace":
                row["ess"] = float(res.info.get("ess_importance", res.n_draws))
                row["k_hat"] = float(res.info.get("k_hat", np.nan))
                row["k_hat_threshold"] = float(res.info.get("k_hat_threshold", np.nan))
            else:
                row["ess"] = float(res.n_draws)
            row["ess_per_s"] = row["ess"] / seconds
        rows.append(row)
    return rows


def _cells(only: str | None, quick: bool) -> list[tuple[str, str]]:
    table = QUICK if quick else SCENARIOS
    return [
        (name, p)
        for name, sc in table.items()
        if only is None or any(o in name for o in only.split(","))
        for p in sc.priors
    ]


def run(out: Path, only: str | None = None, jobs: int = 1, quick: bool = False) -> None:
    cells = _cells(only, quick)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "engine": "header",
        "binference": binference.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "jobs": jobs,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with out.open("a") as f, ProcessPoolExecutor(max(jobs, 1)) as pool:
        f.write(json.dumps(header) + "\n")
        if jobs == 1:
            results = (run_cell(n, p, quick) for n, p in cells)
        else:
            results = pool.map(run_cell, *zip(*cells, strict=True), [quick] * len(cells))
        for cell in results:
            for row in cell:
                f.write(json.dumps(row) + "\n")
                f.flush()
            print(f"{cell[0]['scenario']:>16} {cell[0]['prior']:<13} done", flush=True)


# -- report ------------------------------------------------------------------------------


def _fmt(x, spec=".2f"):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def report(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    header = [r for r in rows if r["engine"] == "header"]
    cells = [r for r in rows if r["engine"] != "header"]
    lines = []
    if header:
        h = header[-1]
        lines.append(
            f"binference {h['binference']}, numpy {h['numpy']}, Python {h['python']}, "
            f"{h['machine']}, {h['jobs']} worker(s) of one thread, started {h['started']}.\n"
        )
    lines.append(
        "| scenario | prior | f | engine | s | ESS | ESS/s | mean err | sd ratio | q err "
        "| k̂ | warnings |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in cells:
        if "error" in r:
            lines.append(
                f"| {r['scenario']} | {r['prior']} | | {r['engine']} | | | | | | | "
                f"| {r['error'][:60]} |"
            )
            continue
        sd = (
            f"{_fmt(r.get('sd_ratio_min'))}–{_fmt(r.get('sd_ratio_max'))}"
            if "sd_ratio_min" in r
            else ""
        )
        lines.append(
            f"| {r['scenario']} | {r['prior']} | {_fmt(r.get('missing_information'))} "
            f"| {r['engine']} | {_fmt(r.get('seconds'), '.1f')} | {_fmt(r.get('ess'), '.0f')} "
            f"| {_fmt(r.get('ess_per_s'), '.0f')} | {_fmt(r.get('mean_err'))} | {sd} "
            f"| {_fmt(r.get('q_err'))} | {_fmt(r.get('k_hat'))} "
            f"| {', '.join(r.get('warnings', []))} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, default=Path("benchmarks/results/matrix.json"))
    r.add_argument("--only", default=None, help="comma-separated scenario substrings")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--quick", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("path", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.out, args.only, args.jobs, args.quick)
    else:
        print(report(args.path), end="")


if __name__ == "__main__":
    main()
