"""Calibration of the total, low-bin rates and fractions under several priors (issue #204)."""
import sys, time, json, warnings
import numpy as np, binference
from scipy import stats, integrate

m_lo, m_hi, slope = 5.0, 85.0, -2.3
def population(x):
    m, a = np.exp(x), slope + 1
    pl = a * m**slope / (m_hi**a - m_lo**a)
    return (0.9 * pl + 0.1 * stats.norm.pdf(m, 34.0, 4.0)) * m
S = lambda m: 0.8 / (1 + (np.asarray(m) / 20.0) ** -3)
K = 16
bins = binference.Bins.linear(np.log(m_lo), np.log(m_hi), K)
e = np.asarray(bins.edges[0] if isinstance(bins.edges, tuple) else bins.edges)
RHO = np.array([integrate.quad(population, e[k], e[k + 1])[0] for k in range(K)]); RHO /= RHO.sum()
LAM = 6000.0
exact = binference.Selection.from_function(lambda x: S(np.exp(x)), bins, "on_truth")
SK = np.asarray(exact.values)
LOW = [0, 1, 2, 3]
WELL = 10

PRIORS = {
    "a": lambda: None,                                         # Dirichlet(1), derived total
    "b": lambda: binference.priors.GammaRates(1.0, 1e-6),      # flat rates
    "c": lambda: binference.priors.GammaRates(1.0 / K, 1e-6),  # rates Gamma(1/K), total prior 1/Lambda
    "d": lambda: binference.priors.Dirichlet(1.0 / K),         # Dirichlet(1/K), derived total
}

def dataset(i, injections):
    rng = np.random.default_rng([204, i])
    N = rng.poisson(LAM)
    k = rng.choice(K, N, p=RHO)
    x = rng.uniform(e[k], e[k + 1])          # uniform within bins: the binned model is exact
    det = rng.random(N) < SK[k]              # per-bin detection probability: exact selection
    sig = rng.uniform(0.05, 0.2, N)
    y = x + sig * rng.standard_normal(N)
    lk = binference.Likelihoods.from_gaussian(y[det], sig[det], bins, warn_outside=False)
    if not injections:
        return lk, exact
    z = rng.uniform(e[0], e[-1], 3000)
    found = rng.random(3000) < SK[np.clip(np.searchsorted(e, z) - 1, 0, K - 1)]
    return lk, binference.Selection.from_injections(z, found, bins, "on_truth")

def summ(d, truth):
    q = np.quantile(d, [0.025, 0.16, 0.84, 0.975], axis=0)
    return dict(mean=d.mean(0).tolist(), sd=d.std(0).tolist(), q=q.tolist(), truth=np.asarray(truth).tolist())

if __name__ == "__main__":
    opt, lo, hi, inj = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "inj"
    out = open(f"res_{opt}_{'inj' if inj else 'exact'}_{lo}_{hi}.jsonl", "w")
    for i in range(lo, hi):
        lk, sel = dataset(i, inj)
        t = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = binference.infer(lk, PRIORS[opt](), selection=sel, method="nuts", n_warmup=500, rng=i)
        dt = time.time() - t
        tot, rates, rho = np.asarray(r.total), np.asarray(r.rates), np.asarray(r.samples)
        rho = rho.reshape(-1, K); rates = rates.reshape(-1, K)
        dg = r.diagnostics
        rec = dict(i=i, opt=opt, time=dt, n_obs=int(lk.n_objects) if hasattr(lk, "n_objects") else None,
                   rhat=float(np.max(dg["rhat"])), ess=float(np.min(dg["ess_bulk"])), ndiv=int(dg.get("n_divergent", 0)),
                   warn=[str(x.message)[:120] for x in w],
                   total=summ(tot, LAM), rates=summ(rates[:, LOW], LAM * RHO[LOW]),
                   rho=summ(rho[:, LOW], RHO[LOW]), lightfrac=summ(rho[:, LOW].sum(1), RHO[LOW].sum()),
                   rho_well=summ(rho[:, WELL], RHO[WELL]))
        out.write(json.dumps(rec) + "\n"); out.flush()
