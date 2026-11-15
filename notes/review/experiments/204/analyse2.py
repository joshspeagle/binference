import json, glob, sys
import numpy as np
def load(opt):
    recs = [json.loads(l) for f in glob.glob(f"res2_{opt}_*.jsonl") for l in open(f)]
    return sorted(recs, key=lambda r: r["i"])
def st(recs, key, j=None):
    g = lambda r, f: np.asarray(r[key][f]) if j is None else np.asarray(r[key][f])[..., j]
    m = np.array([g(r, "mean") for r in recs]); s = np.array([g(r, "sd") for r in recs])
    q = np.array([g(r, "q") for r in recs]); t = np.array([g(r, "truth") for r in recs]); n = len(recs)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (m - t) / s
    ok = np.isfinite(z)
    c68 = ((q[:, 1] <= t) & (t <= q[:, 2])).mean(); c95 = ((q[:, 0] <= t) & (t <= q[:, 3])).mean(); rb = (m - t) / t
    zs = f"z {z[ok].mean():+.2f}±{z[ok].std(ddof=1)/np.sqrt(ok.sum()):.2f} sd {z[ok].std(ddof=1):.2f}" if ok.all() else "z  (fixed)"
    return (f"{zs} | cov68 {c68:.2f} cov95 {c95:.2f} | bias {100*rb.mean():+.1f}±{100*rb.std(ddof=1)/np.sqrt(n):.1f}% | sd/truth {100*np.mean(s/t):.1f}%")
for opt in sys.argv[1:] or ["i1", "i2", "i3", "i4", "i5", "i6"]:
    recs = load(opt)
    if not recs: continue
    tm = np.array([r["time"] for r in recs]); rh = np.array([r["rhat"] for r in recs]); es = np.array([r["ess"] for r in recs]); nd = np.array([r["ndiv"] for r in recs])
    nw = sum(bool(r["warn"]) for r in recs)
    print(f"== {opt}: n={len(recs)} s/fit {np.median(tm):.1f} | max Rhat med {np.median(rh):.3f} worst {rh.max():.3f} | min ESS med {np.median(es):.0f} worst {es.min():.0f} | fits w/ div {np.mean(nd>0):.0%} | fits w/ warnings {nw}")
    print("  total ", st(recs, "total"))
    for j in range(4): print(f"  S{j}    ", st(recs, "S", j))
    sp = np.array([r["Spoint"] for r in recs]); tr = np.array(recs[0]["S"]["truth"])
    print("  S0-3 from injections alone (values): bias " + " ".join(f"{100*(sp[:,j].mean()/tr[j]-1):+.1f}%" for j in range(4)))
    for j in range(4): print(f"  rate{j} ", st(recs, "rates", j))
    if nw: print("  warn e.g.", next(r["warn"] for r in recs if r["warn"]))
