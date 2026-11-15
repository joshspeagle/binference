"""#204: expected bias of the Beta(a,b) posterior mean of S_k and its propagation to Lambda (no MCMC)."""
import sys; sys.argv = ["x"]
import numpy as np
from scipy import stats
from run import RHO, SK, LAM, K
PRI = {"(1,1)": (1, 1), "(0.5,0.5)": (.5, .5), "(0.5,2)": (.5, 2.)}
lam = LAM * RHO; ndet = lam * SK
def table(n_inj):
    n = n_inj / K
    print(f"\n== {n_inj} injections (n_inj/bin = {n:.0f}); lambda_k S_k = expected detections")
    print("bin  S_k     E[n_rec]  " + "  ".join(f"{p:>22}" for p in PRI))
    tot = {p: 0.0 for p in PRI}
    for k in range(K):
        row = f"{k:>3} {SK[k]:.4f} {n*SK[k]:8.1f}  "
        for p, (a, b) in PRI.items():
            rb = (a - (a + b) * SK[k]) / (n + a + b) / SK[k]           # E[S_hat]/S - 1, exact
            # plug-in rate n_det/S_hat: E[1/S_hat] by exact sum over r ~ Binomial(n, S) (n rounded)
            nn = int(round(n)); r = np.arange(nn + 1)
            pl = SK[k] * np.sum(stats.binom.pmf(r, nn, SK[k]) * (a + b + nn) / (a + r)) - 1
            tot[p] += lam[k] * pl
            row += f"  S {100*rb:+6.1f}% lam {100*pl:+6.1f}%"
        print(row)
    print("plug-in E[Lambda]/Lambda - 1: " + "  ".join(f"{p} {100*v/LAM:+.1f}%" for p, v in tot.items()))
for N in (3000, 30000): table(N)

# Exact posterior of Lambda with known labels (no measurement error), prior (a) on rates (density Lambda^-K),
# S_k uncertain with its Beta posterior from injections, by importance sampling (proposal: S ~ Beta posterior,
# lambda | S ~ Gamma(n_det+1, rate S); weight prod_k 1/S_k x Lambda^-K).  Also the plug-in (S fixed at the mean).
rng = np.random.default_rng(20402)
def sim(n_inj, a, b, n_rep=1000, M=20000, plug=False):
    z = []; cov = []; bias = []; ess = []
    for _ in range(n_rep):
        nd = rng.poisson(ndet); ni = rng.multinomial(n_inj, np.full(K, 1 / K)); nr = rng.binomial(ni, SK)
        al, be = a + nr, b + ni - nr
        if plug:
            S = np.broadcast_to(al / (al + be), (M, K)); lw0 = 0
        else:
            S = rng.beta(al, be, size=(M, K)); lw0 = -np.log(S).sum(1)
        L = (rng.gamma(nd + 1, 1, size=(M, K)) / S).sum(1)
        lw = lw0 - K * np.log(L); w = np.exp(lw - lw.max()); w /= w.sum()
        m = w @ L; s = np.sqrt(w @ (L - m) ** 2); o = np.argsort(L); c = np.cumsum(w[o])
        q = np.interp([.025, .16, .84, .975], c, L[o])
        z.append((m - LAM) / s); bias.append(m / LAM - 1); ess.append(1 / np.sum(w**2))
        cov.append(((q[1] <= LAM) & (LAM <= q[2]), (q[0] <= LAM) & (LAM <= q[3])))
    z = np.array(z); cov = np.array(cov).mean(0); n = len(z)
    return (f"z {z.mean():+.2f}±{z.std()/np.sqrt(n):.2f} sd {z.std():.2f} cov68 {cov[0]:.2f} cov95 {cov[1]:.2f} "
            f"bias {100*np.mean(bias):+.1f}±{100*np.std(bias)/np.sqrt(n):.1f}% (median IS ESS {np.median(ess):.0f})")
print("\n== known labels, prior (a), exact posterior by importance sampling, 1000 replicates")
print(f"{'S known (1e9 inj)':18}", sim(10**9, 1, 1, plug=True))
for lab, args, kw in [("3k (1,1)", (3000, 1, 1), {}), ("30k (1,1)", (30000, 1, 1), {}), ("3k (.5,.5)", (3000, .5, .5), {}),
                      ("3k (.5,2)", (3000, .5, 2), {}), ("3k (1,1) plug-in", (3000, 1, 1), {"plug": True}),
                      ("3k (.5,.5) plug-in", (3000, .5, .5), {"plug": True})]:
    print(f"{lab:18}", sim(*args, **kw))
