"""Known true bins (no measurement error): exact posteriors by importance sampling from the flat-rate posterior."""
import numpy as np, sys
sys.argv = ["x"]; from run import RHO, SK, LAM, K
rng = np.random.default_rng(2040)
logw = {"a": lambda lam, L: -K * np.log(L),                                   # Dir(1) x 1/Lambda
        "b": lambda lam, L: 0 * L,                                            # flat rates
        "c": lambda lam, L: (1 / K - 1) * np.log(lam).sum(1),                 # Gamma(1/K) rates
        "d": lambda lam, L: (1 / K - 1) * np.log(lam).sum(1) - np.log(L)}     # Dir(1/K) x 1/Lambda
def run(Svec, n_rep=2000, M=20000):
    res = {o: [] for o in logw}
    for _ in range(n_rep):
        n = rng.poisson(LAM * RHO * Svec)
        lam = rng.gamma(n + 1, 1 / Svec, size=(M, K)); L = lam.sum(1)
        for o, f in logw.items():
            lw = f(lam, L); w = np.exp(lw - lw.max()); w /= w.sum()
            m = w @ L; s = np.sqrt(w @ (L - m) ** 2)
            o_ = np.argsort(L); c = np.cumsum(w[o_]); q = np.interp([0.025, .16, .84, .975], c, L[o_])
            res[o].append((m, s, *q))
    for o, r in res.items():
        r = np.array(r); z = (r[:, 0] - LAM) / r[:, 1]; nn = len(r)
        c68 = np.mean((r[:, 3] <= LAM) & (LAM <= r[:, 4])); c95 = np.mean((r[:, 2] <= LAM) & (LAM <= r[:, 5]))
        print(f"  {o}: z {z.mean():+.3f}±{z.std()/np.sqrt(nn):.3f} sd {z.std():.2f} cov68 {c68:.3f} cov95 {c95:.3f} bias {100*(r[:,0].mean()/LAM-1):+.2f}%")
print("known labels, tutorial selection S_k (0.016..0.79):"); run(SK)
print("known labels, constant S = 0.2085 (same expected detections):"); run(np.full(K, (RHO * SK).sum()))
