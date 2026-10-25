"""NUTS vs Gibbs on the Dirichlet family: how empty-bin tails affect sampling.

Reports wall time, minimum bulk ESS, ESS per second and the fraction of NUTS
iterations that hit the maximum tree depth, for several priors on (1) a 1-D
problem with 30 bins of which about a third hold almost no objects, and (2) a
12x12 grid with many empty cells.
"""
import time, warnings, numpy as np, pastry
from pastry import priors as P
warnings.simplefilter("ignore")

def run(lk, prior, method):
    t = time.time()
    r = pastry.infer(lk, prior, method=method, rng=1, check=False, n_samples=1000, n_warmup=1000)
    dt = time.time() - t
    ess = float(np.nanmin(r.diagnostics["ess_bulk"]))
    return dt, ess, r.info.get("max_depth_fraction", np.nan)

bins = pastry.Bins.linear(0, 3, 30)
dens = pastry.simulate.population(bins, lambda z: np.exp(-0.5 * ((z - 0.9) / 0.25) ** 2) + 0.6 * np.exp(-0.5 * ((z - 1.8) / 0.2) ** 2))
lk1 = pastry.simulate.mock_1d(1000, bins, density=dens, noise=(0.03, 0.08), rng=3).likelihoods
grid = pastry.Bins.grid(np.linspace(0, 1, 13), np.linspace(0, 1, 13))
lk2 = pastry.simulate.mock_2d(1000, grid, density="banana", noise=(0.02, 0.05), rng=8).likelihoods
cases = [
    ("1-D", lk1, "Dirichlet(1)", P.Dirichlet(1.0)),
    ("1-D", lk1, "Dirichlet(0.3)", P.Dirichlet(0.3)),
    ("1-D", lk1, "Dirichlet(1/K)", P.Dirichlet("reference")),
    ("1-D", lk1, "GeneralizedDirichlet(1)", P.GeneralizedDirichlet(1.0)),
    ("1-D", lk1, "PolyaTree(c=1)", P.PolyaTree(c=1.0)),
    ("1-D", lk1, "PolyaTree(c=0.1)", P.PolyaTree(c=0.1)),
    ("2-D", lk2, "Dirichlet(1)", P.Dirichlet(1.0)),
    ("2-D", lk2, "Dirichlet(1/K)", P.Dirichlet("reference")),
]
print(f"{'data':4s} {'prior':24s} {'engine':5s} {'time':>7s} {'minESS':>7s} {'ESS/s':>7s} {'depth-capped':>12s}")
for data, lk, name, prior in cases:
    for method in ("nuts", "gibbs"):
        dt, ess, md = run(lk, prior, method)
        print(f"{data:4s} {name:24s} {method:5s} {dt:6.1f}s {ess:7.0f} {ess/dt:7.1f} {md:12.2f}", flush=True)
