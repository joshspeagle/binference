"""Issue #103 item 2: LogisticGP's default basis on 2-D grids against 10 sines per axis.
Mock with 3000 objects, four chains of 300 warm-up plus 300 draws, one BLAS thread."""
import multiprocessing as mp, sys, time, warnings
import numpy as np
import binference
from binference import Bins, priors as P
from binference.simulate import mock_2d

def one(a):
    k, basis, seed = a
    warnings.simplefilter("ignore")
    bins = Bins.grid(np.linspace(0, 1, k + 1), np.linspace(0, 1, k + 1))
    mock = mock_2d(3000, bins, noise=0.5 / k, rng=seed)
    prior = P.LogisticGP() if basis is None else P.LogisticGP(n_basis=basis)
    t = time.perf_counter()
    r = binference.infer(mock.likelihoods, prior, method="nuts", n_chains=2, n_warmup=300,
                         n_samples=300, rng=seed, check=False)
    d = r.diagnostics
    return k, basis, seed, time.perf_counter() - t, float(np.min(d["ebfmi"])), int(d.get("n_divergent", 0)), float(np.min(d["ess_bulk"])), float(np.max(d["rhat"]))

if __name__ == "__main__":
    jobs = [(k, b, s) for k in (8, 12, 16) for b in (None, 10) for s in range(3)]
    with mp.get_context("spawn").Pool(int(sys.argv[1])) as p:
        res = p.map(one, jobs, chunksize=1)
    for k in (8, 12, 16):
        for b in (None, 10):
            r = [x for x in res if x[0] == k and x[1] == b]
            print(f"{k}x{k} basis {str(b):5s} s {np.mean([x[3] for x in r]):7.1f} ebfmi {min(x[4] for x in r):.2f}-{max(x[4] for x in r):.2f} div {sum(x[5] for x in r):4d} ess {min(x[6] for x in r):6.0f} rhat {max(x[7] for x in r):.3f}")
