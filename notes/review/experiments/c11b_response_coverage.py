"""C11b-3: coverage of the population with a response tallied from few events.

A classifier with Gaussian scores, N(mu_k, 1) for classes mu = (0, 1, 2); the response
is tallied by Response.from_labelled from n_val validation objects per class (10
equal-count score bins). A target set of 1000 objects with fractions (0.2, 0.3, 0.5)
is binned through it. Gibbs (2 chains, 300 burn-in, 500 draws) fits the fractions with
the response fixed at its tallies (plug-in) and inferred jointly under three Dirichlet
priors per cell of a column: beta = 1/(J+1) (a total of 1 per column), 0.5 and 1. Reported: the share of central 68% and 95% intervals of rho_k that
contain the truth (over 3 bins and 100 simulations; Monte Carlo error about 0.03 and
0.015), and the RMSE of the posterior mean.
"""

import numpy as np

import binference
from binference import Likelihoods, Response
from binference.samplers import Gibbs

rng = np.random.default_rng(0)
mu, rho = np.array([0.0, 1.0, 2.0]), np.array([0.2, 0.3, 0.5])
J = 10
PRIORS = {"plug-in": None, "beta 1/(J+1)": 1.0 / (J + 1), "beta 0.5": 0.5, "beta 1": 1.0}
print(f"{'n_val':>5s} {'prior':>13s} {'cover68':>8s} {'cover95':>8s} {'rmse':>6s}")
for n_val in (30, 100, 1000):
    hits = {p: [] for p in PRIORS}
    err = {p: [] for p in PRIORS}
    for _ in range(100):
        labels = np.repeat(np.arange(3), n_val)
        scores = rng.normal(mu[labels], 1.0)
        base = Response.from_labelled(scores, labels)
        target = rng.choice(3, size=1000, p=rho)
        values = rng.normal(mu[target], 1.0)
        for name, prior in PRIORS.items():
            if prior is None:
                resp = base.fixed()
            else:
                resp = Response.from_counts(
                    base.counts, base.generated, base.truth_bins, base.measured_bins,
                    open_ended=True, prior=prior,
                )
            lk = Likelihoods.from_binned(resp, values=values)
            res = Gibbs(binference.PopulationModel(lk), n_chains=2, n_warmup=300, rng=rng).run(500)
            q = np.quantile(res.samples, [0.16, 0.84, 0.025, 0.975], axis=0)
            hits[name].append(
                [np.mean((q[0] <= rho) & (rho <= q[1])), np.mean((q[2] <= rho) & (rho <= q[3]))]
            )
            err[name].append(np.mean((res.samples.mean(0) - rho) ** 2))
    for name in PRIORS:
        h = np.mean(hits[name], axis=0)
        print(f"{n_val:5d} {name:>13s} {h[0]:8.2f} {h[1]:8.2f} {np.sqrt(np.mean(err[name])):6.3f}")
