"""Step 2.3 classifier discussion: compare quantification routes under label shift.

Binary classes, x | c ~ N(+-delta/2, 1). Classifier = Bayes posterior under the
training prevalence, optionally miscalibrated by a temperature T. Routes:
  CC  classify and count (fraction of s > 0.5)
  A   calibrated posteriors / pi_train (today's from_classifier), flat prior on theta
  B   hard labels + confusion matrix from a labelled validation set (from_response)
  C   score binned into M quantile bins, response per class from validation
  B*/C*  same, with the validation uncertainty marginalised (Dirichlet draws, joint)
Reports bias, rms and 68% coverage of the posterior median/interval over mocks.
"""
import numpy as np
from scipy.special import expit, logit, logsumexp

GRID = np.linspace(0.0005, 0.9995, 1000)


def post_summary(ll):
    p = np.exp(ll - ll.max()); p /= p.sum(); c = np.cumsum(p)
    q = np.interp([0.16, 0.5, 0.84], c, GRID)
    return q


def binned_ll(n, R):  # n (J,), R (J,2) columns P(j|c)
    m = R[:, 0][:, None] * (1 - GRID) + R[:, 1][:, None] * GRID
    return (n[:, None] * np.log(np.maximum(m, 1e-300))).sum(0)


def run(theta=0.2, pi_tr=0.5, T=1.0, pi_assumed=None, delta=2.0, N=2000, Nval=500,
        M=10, shift=0.0, n_draw=200, mocks=200, seed=0):
    rng = np.random.default_rng(seed)
    pi_assumed = pi_tr if pi_assumed is None else pi_assumed
    out = {k: [] for k in ["CC", "A", "B", "B*", "C", "C*"]}

    def score(x):
        s = expit(delta * x + np.log(pi_tr / (1 - pi_tr)))
        return expit(logit(np.clip(s, 1e-12, 1 - 1e-12)) / T)

    for _ in range(mocks):
        c = rng.random(N) < theta
        x = rng.normal(np.where(c, delta / 2 + shift, -delta / 2), 1.0)
        s = score(x)
        cv = rng.random(Nval) < pi_tr
        sv = score(rng.normal(np.where(cv, delta / 2, -delta / 2), 1.0))
        out["CC"].append([np.nan, np.mean(s > 0.5), np.nan])
        L1, L0 = s / pi_assumed, (1 - s) / (1 - pi_assumed)
        ll = np.log(np.maximum(L0[:, None] * (1 - GRID) + L1[:, None] * GRID, 1e-300)).sum(0)
        out["A"].append(post_summary(ll))
        for name, edges in [("B", np.array([0, 0.5, 1.0])),
                            ("C", np.r_[0, np.quantile(sv, np.linspace(0, 1, M + 1)[1:-1]), 1.0])]:
            j = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
            jv = np.clip(np.searchsorted(edges, sv, side="right") - 1, 0, len(edges) - 2)
            J = len(edges) - 1
            n = np.bincount(j, minlength=J).astype(float)
            cnt = np.stack([np.bincount(jv[~cv], minlength=J), np.bincount(jv[cv], minlength=J)], 1) + 0.5
            R = cnt / cnt.sum(0)
            out[name].append(post_summary(binned_ll(n, R)))
            lls = np.array([binned_ll(n, np.stack([rng.dirichlet(cnt[:, 0]), rng.dirichlet(cnt[:, 1])], 1))
                            for _ in range(n_draw)])
            out[name + "*"].append(post_summary(logsumexp(lls, 0)))
    res = {}
    for k, v in out.items():
        v = np.array(v)
        med = v[:, 1]
        cov = np.mean((v[:, 0] <= theta) & (theta <= v[:, 2])) if k != "CC" else np.nan
        res[k] = (med.mean() - theta, np.sqrt(np.mean((med - theta) ** 2)), cov)
    return res


if __name__ == "__main__":
    cases = {
        "calibrated": dict(),
        "overconfident T=0.5": dict(T=0.5),
        "underconfident T=2": dict(T=2.0),
        "wrong pi_train (0.5 told, 0.3 true)": dict(pi_tr=0.3, pi_assumed=0.5),
        "label shift violated (class-1 mean +0.5)": dict(shift=0.5),
        "small validation (Nval=100)": dict(Nval=100),
    }
    for name, kw in cases.items():
        r = run(**kw)
        print(f"\n{name}")
        for k, (b, rms, cov) in r.items():
            print(f"  {k:3s} bias {b:+.3f}  rms {rms:.3f}  cov68 {cov:.2f}")
