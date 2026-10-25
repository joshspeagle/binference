"""Step 2.3: Monte Carlo error of the samples estimator L_ik ~ (1/S) sum_s 1[z_s in k]/pi_k.

Objects: z ~ population (two Gaussians on [0, 1]), d = z + N(0, sigma); per-object
posterior samples under a flat interim prior (truncated Gaussian on [0, 1]).
For rho = true bin masses, report the bias and scatter (over sample redraws) of the
total log-likelihood sum_i log(Lhat_i . rho) relative to the exact value, and the
same for a rho far from the interim (all mass in the left half).
"""
import numpy as np
from scipy.stats import norm, truncnorm

rng = np.random.default_rng(3)
K = 20; edges = np.linspace(0, 1, K + 1); sigma = 0.08


def pop_draw(n):
    a = rng.random(n) < 0.6
    z = np.where(a, rng.normal(0.3, 0.07, n), rng.normal(0.7, 0.1, n))
    return np.clip(z, 1e-6, 1 - 1e-6)


zt = pop_draw(200000)
rho_true = np.histogram(zt, edges)[0] / zt.size
rho_far = np.where(edges[:-1] < 0.5, 1.0, 1e-3); rho_far /= rho_far.sum()

for N in (1000, 10000):
    z = pop_draw(N); d = z + rng.normal(0, sigma, N)
    # exact bin-averaged likelihood (flat interim, truncated to [0,1]): Phi differences
    Lx = norm.cdf((edges[None, 1:] - d[:, None]) / sigma) - norm.cdf((edges[None, :-1] - d[:, None]) / sigma)
    Lx /= Lx.sum(1, keepdims=True)  # posterior bin masses under flat interim = L * width / Z
    a, b = (0 - d) / sigma, (1 - d) / sigma
    for Ssamp in (100, 1000):
        for name, rho in (("rho=truth", rho_true), ("rho far from interim", rho_far)):
            exact = np.log(Lx @ rho).sum()
            diffs = []; zero = []
            for rep in range(20):
                zs = truncnorm.rvs(a[:, None], b[:, None], loc=d[:, None], scale=sigma,
                                   size=(N, Ssamp), random_state=rng)
                idx = np.minimum((zs * K).astype(int), K - 1)
                C = np.zeros((N, K)); np.add.at(C, (np.repeat(np.arange(N), Ssamp), idx.ravel()), 1)
                C /= Ssamp
                ll = np.log(np.maximum(C @ rho, 1e-300)).sum()
                diffs.append(ll - exact); zero.append(np.mean((C @ rho) <= 1e-300))
            diffs = np.array(diffs)
            print(f"N={N:5d} S={Ssamp:4d} {name:22s} bias {diffs.mean():+8.2f}  sd {diffs.std():6.2f}")
