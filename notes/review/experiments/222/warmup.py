"""Issue #222: LogisticGP, default basis, 16 x 16 grid, partial centring, with the default
warm-up (1000 iterations) and 300 draws in each of two chains, by metric. Mock as
experiments/103_gp_basis_on_grids.py. Run with OPENBLAS_NUM_THREADS=1; arguments: fits at
once, seeds, and optionally the metrics (default diag,dense). warmup.out was written
before #254, when "auto" was diagonal here; warmup_0.3.out after it."""
import multiprocessing as mp, sys, time, warnings
import numpy as np
import binference
from binference import Bins, priors as P
from binference.diagnostics import ess
from binference.simulate import mock_2d

def one(a):
    metric, seed = a
    warnings.simplefilter("ignore")
    mock = mock_2d(3000, Bins.grid(*[np.linspace(0, 1, 17)] * 2), noise=0.5 / 16, rng=seed)
    t = time.perf_counter()
    r = binference.infer(mock.likelihoods, P.LogisticGP(), method="nuts", n_chains=2, n_samples=300,
                         rng=seed, check=False, metric=metric)
    t = time.perf_counter() - t
    hyp = min(ess(np.log(v)) for v in r.info["hyperparameters"].values())
    return metric, seed, t, np.round(r.diagnostics["ebfmi"], 2), hyp, float(np.min(r.diagnostics["ess_bulk"])), r.diagnostics["n_divergent"]

if __name__ == "__main__":
    metrics = sys.argv[3].split(",") if len(sys.argv) > 3 else ["diag", "dense"]
    jobs = [(m, s) for s in range(int(sys.argv[2])) for m in metrics]
    with mp.get_context("spawn").Pool(int(sys.argv[1])) as p:
        for x in p.imap(one, jobs):
            print(f"16x16 {x[0]:6s} seed {x[1]} s {x[2]:6.1f} ebfmi {x[3]} hyper-ess {x[4]:5.0f} ({x[4] / x[2]:5.2f}/s) min-ess {x[5]:5.0f} div {x[6]:3d}", flush=True)
