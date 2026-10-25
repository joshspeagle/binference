"""C11b-2b: a smooth injection model against independent Betas, from sparse injections.

20 bins on [0, 1]; a true completeness S(z), averaged over each bin; n injections per
bin, uniform within it. Compares the root-mean-square error over bins and replicates of
each model's estimate given the injections alone: the posterior means of independent
Beta(1, 1) posteriors, and the mode of logit S under the smooth model (step scale tau).
"""

import numpy as np
from scipy import special

from binference import Bins, Selection

bins = Bins.linear(0.0, 1.0, 20)
fine = np.linspace(0.0, 1.0, 20001)
CURVES = {
    "logistic edge": lambda z: special.expit((0.6 - z) / 0.08),
    "linear": lambda z: 0.9 - 0.7 * z,
    "sharp edge": lambda z: np.where(z < 0.5, 0.9, 0.1),
}


def bin_average(f):
    k = bins.index(fine[:-1] + 0.5 * np.diff(fine))
    return np.bincount(k, f(fine[:-1] + 0.5 * np.diff(fine)), 20) / np.bincount(k, None, 20)


rng = np.random.default_rng(0)
taus = (None, 1.0, 0.3, 0.1)
print("RMSE of S over 20 bins and 200 replicates (Monte Carlo error in brackets)")
print(f"{'curve':14s} {'n/bin':>5s} " + " ".join(f"{'tau=' + str(t):>16s}" for t in taus))
for name, f in CURVES.items():
    S = bin_average(f)
    for n in (2, 5, 20):
        err = {t: [] for t in taus}
        for _ in range(200):
            z = rng.uniform(0.0, 1.0, 20 * n)
            detected = rng.random(z.size) < f(z)
            for t in taus:
                sel = Selection.from_injections(z, detected, bins, "on_truth", smooth=t)
                err[t].append(np.mean((sel.values - S) ** 2))
        cells = []
        for t in taus:
            e = np.array(err[t])
            rmse = np.sqrt(e.mean())
            cells.append(f"{rmse:.3f} ({e.std() / np.sqrt(e.size) / (2 * rmse):.3f})")
        print(f"{name:14s} {n:5d} " + " ".join(f"{c:>16s}" for c in cells))
