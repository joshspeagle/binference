"""#233: histogram vs reflected Gaussian KDE for per-object posterior samples (flat interim).

usage: python kde_vs_hist.py CASE   (CASE in g20, g100, g20n, bi20)
"""
import sys, time, warnings
import numpy as np
from scipy.special import ndtr
from scipy.stats import truncnorm, norm
import binference as bf

warnings.filterwarnings("ignore")
CASES = {  # name: (K, sigma, bimodal)
    "g20": (20, 0.08, False),   # bin width 0.05 < sigma
    "g100": (100, 0.08, False),  # bin width 0.01 << sigma
    "g20n": (20, 0.02, False),  # posterior narrower than a bin
    "bi20": (20, 0.05, True),   # bimodal: mode at d and at 1-d
}
case = sys.argv[1]
K, sigma, bimodal = CASES[case]
rng = np.random.default_rng(233 + list(CASES).index(case))
edges = np.linspace(0, 1, K + 1)
width = np.diff(edges)
bins = bf.Bins(edges)


def pop_cdf(x):
    return 0.6 * norm.cdf(x, 0.3, 0.07) + 0.4 * norm.cdf(x, 0.7, 0.1)


rho_true = np.diff(pop_cdf(edges)); rho_true /= rho_true.sum()
rho_far = np.where(edges[:-1] < 0.5, 1.0, 1e-3); rho_far /= rho_far.sum()


def pop_draw(n):
    out = np.empty(0)
    while out.size < n:
        a = rng.random(n) < 0.6
        z = np.where(a, rng.normal(0.3, 0.07, n), rng.normal(0.7, 0.1, n))
        out = np.concatenate([out, z[(z > 0) & (z < 1)]])
    return out[:n]


def comps(d):
    """Posterior components (means (N, C), weights (N, C)) under a flat interim on [0,1]."""
    mu = np.stack([d, 1 - d], 1) if bimodal else d[:, None]
    m = norm.cdf((1 - mu) / sigma) - norm.cdf((0 - mu) / sigma)
    return mu, m / m.sum(1, keepdims=True)


def exact_masses(mu, wt, s):
    F = norm.cdf((edges[None, None, :] - mu[:, :, None]) / s)
    M = (wt[:, :, None] * np.diff(F, axis=2) / (F[:, :, -1:] - F[:, :, :1])).sum(1)
    return M / M.sum(1, keepdims=True)


def draw_samples(mu, wt, S):
    N, C = mu.shape
    c = (rng.random((N, S)) < wt[:, :1]).astype(int) if C == 1 else (rng.random((N, S)) > wt[:, :1]).astype(int)
    if C == 1:
        c[:] = 0
    m = np.take_along_axis(mu, c, 1)
    return truncnorm.rvs((0 - m) / sigma, (1 - m) / sigma, loc=m, scale=sigma, random_state=rng)


def hist_masses(x):
    N, S = x.shape
    idx = np.minimum((x * K).astype(int), K - 1)
    C = np.zeros((N, K)); np.add.at(C, (np.repeat(np.arange(N), S), idx.ravel()), 1)
    return C / S


def bandwidth(x, rule):
    S = x.shape[1]
    sd = x.std(1, ddof=1)
    if rule == "scott":
        return sd * S ** -0.2
    q75, q25 = np.percentile(x, [75, 25], axis=1)
    return 0.9 * np.minimum(sd, (q75 - q25) / 1.34) * S ** -0.2


def kde_masses(x, rule, chunk=None):
    """Reflected (at 0 and 1) Gaussian KDE integrated exactly over each bin."""
    N, S = x.shape
    h = bandwidth(x, rule)
    chunk = chunk or max(1, int(4e6 // (S * (K + 1))))
    out = np.empty((N, K))
    for a in range(0, N, chunk):
        xs, hs = x[a:a + chunk, :, None], h[a:a + chunk, None, None]
        e = edges[None, None, :]
        G = (ndtr((e - xs) / hs) + ndtr((e + xs) / hs) + ndtr((e - 2 + xs) / hs)).mean(1)
        out[a:a + chunk] = np.diff(G, axis=1)
    return out / out.sum(1, keepdims=True)


def loglik(M, rho):  # M: posterior masses under flat interim; L = M / width up to a row constant
    return np.log(np.maximum((M / width) @ rho, 1e-300)).sum()


def map_rho(M):
    lk = bf.Likelihoods(M / width, bins)
    return np.asarray(bf.infer(lk, method="map", check=False).rho)


def kl(p, q):  # KL(p || q) with q > 0
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


cen = 0.5 * (edges[1:] + edges[:-1])


def psd(r):
    m = r @ cen
    return float(np.sqrt(r @ (cen - m) ** 2))


def kl5(p, q):  # KL on 5 coarse bins of width 0.2: a functional the deconvolution determines
    return kl(p.reshape(5, -1).sum(1), q.reshape(5, -1).sum(1))


def rel_err(M, Mx):
    sel = Mx >= 0.01
    return float(np.sqrt(np.mean(((M[sel] - Mx[sel]) / Mx[sel]) ** 2)))


methods = ("hist", "kde-silv", "kde-scott")
est = {"hist": hist_masses, "kde-silv": lambda x: kde_masses(x, "silverman"),
       "kde-scott": lambda x: kde_masses(x, "scott")}
print(f"# case {case}: K={K} sigma={sigma} bimodal={bimodal}", flush=True)
for N, R in ((1000, 8), (10000, 3)):
    z = pop_draw(N)
    flip = (rng.random(N) < 0.5) if bimodal else np.zeros(N, bool)
    d = np.where(flip, 1 - z, z) + rng.normal(0, sigma, N)
    mu, wt = comps(d)
    Mx = exact_masses(mu, wt, sigma)
    ex = {"truth": loglik(Mx, rho_true), "far": loglik(Mx, rho_far)}
    rx = map_rho(Mx)
    print(f"N={N} exact: KL(MAP||truth)={kl(rx, rho_true):.5f} KL5={kl5(rx, rho_true):.5f}  sd(MAP)={psd(rx):.4f}  sd(truth)={psd(rho_true):.4f}", flush=True)
    for S in (30, 100, 1000):
        if N == 10000 and S == 1000 and K == 100:
            R = 2
        res = {m: {"err": [], "truth": [], "far": [], "kl": [], "sd": [], "kl5": []} for m in methods}
        sumV = []
        for r in range(R):
            x = draw_samples(mu, wt, S)
            if N == 1000:
                lk = bf.Likelihoods.from_samples(x, bins, "flat", min_samples=1, warn_outside=False)
                sumV.append(lk.sample_variance(rho_true)[0].sum())
            for m in methods:
                M = est[m](x)
                res[m]["err"].append(rel_err(M, Mx))
                res[m]["truth"].append(loglik(M, rho_true) - ex["truth"])
                res[m]["far"].append(loglik(M, rho_far) - ex["far"])
                r_ = map_rho(M)
                res[m]["kl"].append(kl(r_, rho_true)); res[m]["sd"].append(psd(r_)); res[m]["kl5"].append(kl5(r_, rho_true))
        for m in methods:
            q = {k: np.array(v) for k, v in res[m].items()}
            print(f"N={N:5d} S={S:4d} {m:9s} relerr {q['err'].mean():.3f}  "
                  f"dlogL truth {q['truth'].mean():+9.2f} (sd {q['truth'].std():6.2f})  "
                  f"far {q['far'].mean():+9.2f} (sd {q['far'].std():6.2f})  "
                  f"KL {q['kl'].mean():.5f}  KL5 {q['kl5'].mean():.5f}  sd(MAP) {q['sd'].mean():.4f}", flush=True)
        if sumV:
            print(f"   hist sum V_i at truth (package): {np.mean(sumV):.2f}", flush=True)
        if not bimodal:  # smoothing bias alone: the infinite-sample KDE is N(d, sigma^2 + h^2)
            for rule, c in (("silv", 0.9), ("scott", 1.0)):
                h = c * sigma * S ** -0.2
                Mb = exact_masses(mu, wt, np.hypot(sigma, h))
                print(f"   kde-{rule} bias only (h={h:.4f}, width x{np.hypot(1, h / sigma):.3f}): "
                      f"relerr {rel_err(Mb, Mx):.3f}  dlogL truth {loglik(Mb, rho_true) - ex['truth']:+.2f}  "
                      f"far {loglik(Mb, rho_far) - ex['far']:+.2f}  KL {kl(rb := map_rho(Mb), rho_true):.5f}  KL5 {kl5(rb, rho_true):.5f}  sd(MAP) {psd(rb):.4f}",
                      flush=True)
