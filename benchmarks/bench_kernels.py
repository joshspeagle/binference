"""Micro-benchmarks of pastry's hot kernels and engines.

Run from the repository root::

    python benchmarks/bench_kernels.py            # kernels only (seconds)
    python benchmarks/bench_kernels.py --engines  # also time every engine on a mock

Timings are wall-clock medians; set ``OMP_NUM_THREADS=1`` for single-thread numbers.
Reference targets (from the frankenz audit, N=20000, K=80, single thread): one Gibbs
label sweep <= ~15 ms and one log-likelihood-plus-gradient evaluation <= ~2 ms.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

import pastry
from pastry import PopulationModel
from pastry import _kernels as kern


def _median_time(fn, repeat: int = 20) -> float:
    fn()  # warm caches
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def bench_kernels(N: int = 20000, K: int = 80, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    bins = pastry.Bins.linear(-1.0, 7.0, K)
    mu = rng.uniform(0.0, 6.0, N)
    sigma = rng.uniform(0.05, 0.2, N) * (1.0 + mu)
    lk = pastry.Likelihoods.from_gaussian(mu + sigma * rng.standard_normal(N), sigma, bins)
    model = PopulationModel(lk)
    L, w = model._L, model._w
    rho = np.full(K, 1.0 / K)
    x = np.zeros(model.dim)
    out = {
        "loglike_and_counts": _median_time(lambda: kern.loglike_and_counts(L, w, rho)),
        "log_prob_and_grad": _median_time(lambda: model.log_prob_and_grad(x)),
        "gibbs_label_sweep": _median_time(
            lambda: kern.sample_counts(L, w, rho, rng, unit_weights=True), repeat=10
        ),
        "responsibilities": _median_time(lambda: kern.responsibilities(L, rho), repeat=10),
    }
    sparse = lk.to_sparse(rtol=1e-6)
    Ls = PopulationModel(sparse)._L
    out["loglike_and_counts_sparse"] = _median_time(lambda: kern.loglike_and_counts(Ls, w, rho))
    out["sparse_density"] = sparse.nnz / (N * K)
    return out


def bench_engines(seed: int = 42) -> list[tuple[str, float, float]]:
    mock = pastry.simulate.mock_1d(2000, rng=seed)
    rows = []
    for method, prior in (
        ("nuts", None),
        ("nuts", pastry.priors.LogisticGP()),
        ("gibbs", None),
        ("map", None),
        ("laplace", None),
        ("vb", None),
    ):
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pastry.infer(mock.likelihoods, prior, method=method, rng=1, check=False)
        dt = time.perf_counter() - t
        ess = np.nan
        if hasattr(res, "chains") and res.n_chains > 1:
            ess = float(np.min(res.diagnostics["ess_bulk"]))
        label = method + ("" if prior is None else f" + {type(prior).__name__}")
        rows.append((label, dt, ess))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engines", action="store_true", help="also time the engines")
    args = ap.parse_args()
    print("kernels at N=20000, K=80:")
    for k, v in bench_kernels().items():
        unit = "" if k == "sparse_density" else " ms"
        val = v if k == "sparse_density" else 1e3 * v
        print(f"  {k:28s} {val:9.3f}{unit}")
    if args.engines:
        print("engines on mock_1d(2000), 30 bins:")
        for label, dt, ess in bench_engines():
            print(f"  {label:28s} {dt:7.2f} s   min bulk ESS {ess:7.0f}")


if __name__ == "__main__":
    main()
