"""Histogram (flat-in-bin) vs linear B-spline mixture basis: same linear model, different L."""
import warnings, numpy as np, pastry
from scipy import stats, integrate
from pastry.samplers import MAP
warnings.simplefilter("ignore")
rng = np.random.default_rng(1)
beta, N, K, lo, hi = 1.5, 50_000, 10, 0.0, 3.0
# truncated exponential population
u = rng.uniform(size=N); z = -np.log(1 - u * (1 - np.exp(-beta * hi))) / beta
sig = rng.uniform(0.15, 0.3, N); d = z + sig * rng.standard_normal(N)
edges = np.linspace(lo, hi, K + 1); h = edges[1] - edges[0]
f = lambda x: beta * np.exp(-beta * x) / (1 - np.exp(-beta * hi))
truth = np.array([integrate.quad(f, a, b)[0] for a, b in zip(edges[:-1], edges[1:])])
# fine quadrature grid for building both L matrices and for mapping weights -> bin masses
zq = np.linspace(lo, hi, 601); wq = np.gradient(zq)
P = stats.norm.pdf(d[:, None], zq[None, :], sig[:, None]) * wq[None, :]       # (N, Q) p(d|z) dz
# basis 0: histogram (flat densities on bins); basis 1: linear B-splines (hats) on K knots spanning the range
hist = np.array([((zq >= a) & (zq < b) | ((b == hi) & (zq == hi))) / h for a, b in zip(edges[:-1], edges[1:])]).T
knots = np.linspace(lo, hi, K); dk = knots[1] - knots[0]
hat = np.clip(1 - np.abs(zq[:, None] - knots[None, :]) / dk, 0, None)
hat /= (hat * wq[:, None]).sum(0)                                             # each component integrates to 1
for name, Phi in [("histogram (flat in bin)", hist), ("linear B-spline mixture", hat)]:
    L = P @ Phi                                                               # L_ik = ∫ p(d_i|z) φ_k(z) dz
    est = MAP(pastry.PopulationModel(pastry.Likelihoods(L))).run().rho
    masses = np.array([(hist[:, j] * h * wq) @ (Phi @ est) for j in range(K)])
    masses /= masses.sum()
    sd = np.sqrt(truth * (1 - truth) / N)                                     # shot-noise scale only (lower bound)
    print(f"{name:26s} bin-mass error / truth: {np.round((masses - truth) / truth, 3)}")
    print(f"{'':26s} max |error| in multinomial-sd units: {np.max(np.abs(masses - truth) / sd):.1f}")
