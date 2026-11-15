"""Issue #222: hyperparameter quantiles of LogisticGP with the default basis on a 16 x 16
grid, partially centred (first argument 1) against non-centred (0), for one seed (second
argument). Dense metric, two chains of 300 warm-up plus 300 draws in two workers. Mock as
experiments/103_gp_basis_on_grids.py. Run with OPENBLAS_NUM_THREADS=1, two fits at once;
medians.out holds seeds 0, 1 and 2."""
import sys, time, warnings
import numpy as np
import binference
from binference import Bins, priors as P
from binference.simulate import mock_2d
from binference.diagnostics import ess
warnings.simplefilter("ignore")
k=int(sys.argv[3]) if len(sys.argv) > 3 else 16; seed=int(sys.argv[2]); reparam = sys.argv[1] == "1"
bins = Bins.grid(np.linspace(0, 1, k + 1), np.linspace(0, 1, k + 1))
mock = mock_2d(3000, bins, noise=0.5 / k, rng=seed)
t = time.perf_counter()
r = binference.infer(mock.likelihoods, P.LogisticGP(), method="nuts", n_chains=2, n_jobs=2,
                     n_warmup=300, n_samples=300, rng=seed, check=False, metric="dense", reparametrize=reparam)
d = r.diagnostics; I = r.info
print(f"k {k} seed {seed} reparam {reparam} time {time.perf_counter() - t:.0f} ebfmi {np.round(d['ebfmi'],2)} ess min {np.min(d['ess_bulk']):.0f} step {np.round(I['step_size'],3)} maxdepth {I['max_depth_fraction']}")
c = I.get("centring")
if c is not None:
    print("  centring: n", c.size, "==1:", np.sum(c == 1), "<0.1:", np.sum(c < 0.1), "q", np.round(np.quantile(c, [0, .25, .5, .75, 1]), 3))
for n_, v in I["hyperparameters"].items():
    print("  ", n_, v.shape, "q", np.round(np.quantile(v, [.05, .5, .95]), 4), "ess", round(ess(np.log(v))))
