"""Issue #222: partial centring (the default) against the fully non-centred form, by
case. Dense metric throughout, two chains of 300 warm-up plus 300 draws run in
turn. Run with OPENBLAS_NUM_THREADS=1; arguments: fits at once, seeds, and optionally
the cases to run (comma-separated)."""
import multiprocessing as mp, sys, time, warnings
import numpy as np
import binference
from binference import Bins, priors as P
from binference.diagnostics import ess
from binference.simulate import mock_1d, mock_2d

CASES = {
    "1d-gp-30": lambda s: (mock_1d(3000, rng=s), P.LogisticGP()),
    "1d-gp-60": lambda s: (mock_1d(3000, Bins.linear(0, 3, 60), rng=s), P.LogisticGP()),
    "1d-rw2-60": lambda s: (mock_1d(3000, Bins.linear(0, 3, 60), rng=s), P.LogisticRandomWalk(order=2)),
    "1d-gp-60-n300": lambda s: (mock_1d(300, Bins.linear(0, 3, 60), rng=s), P.LogisticGP()),
    "2d-gp-8": lambda s: (mock_2d(3000, Bins.grid(*[np.linspace(0, 1, 9)] * 2), noise=0.5 / 8, rng=s), P.LogisticGP()),
    "2d-gp-12": lambda s: (mock_2d(3000, Bins.grid(*[np.linspace(0, 1, 13)] * 2), noise=0.5 / 12, rng=s), P.LogisticGP()),
    "2d-gp-16": lambda s: (mock_2d(3000, Bins.grid(*[np.linspace(0, 1, 17)] * 2), noise=0.5 / 16, rng=s), P.LogisticGP()),
}

def one(a):
    case, reparam, seed = a
    warnings.simplefilter("ignore")
    mock, prior = CASES[case](seed)
    t = time.perf_counter()
    r = binference.infer(mock.likelihoods, prior, method="nuts", n_chains=2, n_warmup=300,
                         n_samples=300, rng=seed, check=False, metric="dense", reparametrize=reparam)
    t = time.perf_counter() - t
    hyp = min(ess(np.log(v)) for v in r.info["hyperparameters"].values())
    c = r.info.get("centring")
    cen = "-" if c is None else f"{np.mean(c < 0.5):.2f}"
    return case, reparam, seed, t, float(np.min(r.diagnostics["ebfmi"])), hyp, float(np.min(r.diagnostics["ess_bulk"])), r.diagnostics["n_divergent"], cen

if __name__ == "__main__":
    seeds = range(int(sys.argv[2]))
    cases = sys.argv[3].split(",") if len(sys.argv) > 3 else CASES
    jobs = [(c, r, s) for c in cases for s in seeds for r in (True, False)]
    with mp.get_context("spawn").Pool(int(sys.argv[1])) as p:
        for x in p.imap(one, jobs):
            print(f"{x[0]:14s} {'partial' if x[1] else 'noncent':8s} seed {x[2]} s {x[3]:6.1f} ebfmi {x[4]:.2f} "
                  f"hyper-ess {x[5]:5.0f} ({x[5] / x[3]:5.2f}/s) min-ess {x[6]:5.0f} div {x[7]:3d} centred {x[8]}", flush=True)
