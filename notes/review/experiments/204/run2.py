"""#204 follow-up: injection-selection settings, population prior (a) throughout. Setup reused from run.py."""
import sys, time, json, warnings
import numpy as np, binference
from run import K, e, RHO, SK, LAM, LOW, summ

# option: (n_inj, prior, smooth, fixed)
OPTS = {
    "i1": (3000, (1.0, 1.0), None, False),
    "i2": (30000, (1.0, 1.0), None, False),
    "i3": (3000, (0.5, 0.5), None, False),
    "i4": (3000, (0.5, 2.0), None, False),
    "i5": (3000, (1.0, 1.0), 0.1, False),
    "i6": (3000, (1.0, 1.0), None, True),
}

def dataset(i, n_inj, prior, smooth, fixed):
    rng = np.random.default_rng([204, i])          # identical draws to run.dataset
    N = rng.poisson(LAM)
    k = rng.choice(K, N, p=RHO)
    x = rng.uniform(e[k], e[k + 1])
    det = rng.random(N) < SK[k]
    sig = rng.uniform(0.05, 0.2, N)
    y = x + sig * rng.standard_normal(N)
    lk = binference.Likelihoods.from_gaussian(y[det], binsx, sig[det], warn_outside=False)
    z = rng.uniform(e[0], e[-1], n_inj)
    kz = np.clip(np.searchsorted(e, z) - 1, 0, K - 1)
    found = rng.random(n_inj) < SK[kz]
    sel = binference.Selection.from_injections(z, found, binsx, "on_truth", prior=prior, smooth=smooth)
    return lk, (sel.fixed() if fixed else sel), np.bincount(kz, minlength=K), np.bincount(kz, weights=found, minlength=K)

from run import bins as binsx

if __name__ == "__main__":
    opt, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with open(f"res2_{opt}_{lo}_{hi}.jsonl", "w") as out:
        for i in range(lo, hi):
            lk, sel, ninj, nrec = dataset(i, *OPTS[opt])
            t = time.time()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r = binference.infer(lk, None, selection=sel, method="nuts", n_warmup=500, rng=i)
            dt = time.time() - t
            tot = np.asarray(r.total).ravel()
            rates = np.asarray(r.rates).reshape(-1, K)
            Sd = r.info.get("selection_draws")
            Sd = (np.asarray(Sd).reshape(-1, K) if Sd is not None else np.tile(np.asarray(sel.values), (2, 1)))
            dg = r.diagnostics
            rec = dict(i=i, opt=opt, time=dt, rhat=float(np.max(dg["rhat"])), ess=float(np.min(dg["ess_bulk"])),
                       ndiv=int(dg.get("n_divergent", 0)), warn=[str(x.message)[:120] for x in w],
                       ninj=ninj[LOW].tolist(), nrec=nrec[LOW].tolist(), Spoint=np.asarray(sel.values)[LOW].tolist(),
                       total=summ(tot, LAM), rates=summ(rates[:, LOW], LAM * RHO[LOW]), S=summ(Sd[:, LOW], SK[LOW]))
            out.write(json.dumps(rec) + "\n"); out.flush()
