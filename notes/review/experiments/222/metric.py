"""Issue #222: LogisticGP with the default basis on a k x k grid, diagonal (auto)
against dense metric. Mock and run lengths as experiments/103_gp_basis_on_grids.py.
Run with OPENBLAS_NUM_THREADS=1; arguments: number of worker processes, grid size k
(default 16), seeds (default 3), metrics (default diag,dense). The .out files of 16 and 20
were written under the old limit of 200, when "auto" was diagonal there."""
import multiprocessing as mp, sys, time, warnings
import numpy as np
import binference
from binference import Bins, priors as P
from binference.simulate import mock_2d

def one(a):
    k, metric, seed = a
    warnings.simplefilter("ignore")
    bins = Bins.grid(np.linspace(0, 1, k + 1), np.linspace(0, 1, k + 1))
    mock = mock_2d(3000, bins, noise=0.5 / k, rng=seed)
    t = time.perf_counter()
    r = binference.infer(mock.likelihoods, P.LogisticGP(), method="nuts", n_chains=2,
                         n_warmup=300, n_samples=300, rng=seed, check=False, metric=metric)
    d = r.diagnostics
    return (k, metric, seed, time.perf_counter() - t, float(np.min(d["ebfmi"])),
            int(d.get("n_divergent", 0)), float(np.min(d["ess_bulk"])), float(np.max(d["rhat"])))

if __name__ == "__main__":
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    metrics = sys.argv[4].split(",") if len(sys.argv) > 4 else ["diag", "dense"]
    jobs = [(k, m, s) for m in metrics for s in range(n_seeds)]
    with mp.get_context("spawn").Pool(int(sys.argv[1])) as p:
        res = p.map(one, jobs, chunksize=1)
    for x in res:
        print(f"{x[0]}x{x[0]} {x[1]:6s} seed {x[2]} s {x[3]:7.1f} ebfmi {x[4]:.2f} div {x[5]:4d} ess {x[6]:6.0f} rhat {x[7]:.3f}")
