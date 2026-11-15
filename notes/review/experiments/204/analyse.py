import json, glob, sys
import numpy as np
mode = sys.argv[1] if len(sys.argv) > 1 else "exact"
def load(opt):
    recs = []
    for f in sorted(glob.glob(f"res_{opt}_{mode}_*.jsonl")):
        recs += [json.loads(l) for l in open(f)]
    return sorted(recs, key=lambda r: r["i"])
def stats(recs, key, j=None):
    g = lambda r, f: np.asarray(r[key][f]) if j is None else np.asarray(r[key][f])[..., j]
    m = np.array([g(r, "mean") for r in recs]); s = np.array([g(r, "sd") for r in recs])
    q = np.array([g(r, "q") for r in recs]); t = np.array([g(r, "truth") for r in recs])
    n = len(recs); z = (m - t) / s
    c68 = ((q[:, 1] <= t) & (t <= q[:, 2])).mean(); c95 = ((q[:, 0] <= t) & (t <= q[:, 3])).mean()
    rb = (m - t) / t
    se = lambda p: np.sqrt(p * (1 - p) / n)
    return (f"z {z.mean():+.2f}±{z.std(ddof=1)/np.sqrt(n):.2f} sd {z.std(ddof=1):.2f} | "
            f"cov68 {c68:.2f}±{se(c68):.2f} cov95 {c95:.2f}±{se(c95):.2f} | "
            f"bias {100*rb.mean():+.1f}±{100*rb.std(ddof=1)/np.sqrt(n):.1f}% | post.sd/truth {100*np.mean(s/t):.1f}%")
for opt in "abcd":
    recs = load(opt)
    if not recs: continue
    tm = np.array([r["time"] for r in recs]); rh = np.array([r["rhat"] for r in recs]); es = np.array([r["ess"] for r in recs])
    nd = np.array([r["ndiv"] for r in recs])
    print(f"== option {opt}: n={len(recs)} time/fit median {np.median(tm):.1f}s mean {tm.mean():.1f}s; "
          f"max Rhat median {np.median(rh):.3f} (>1.01 in {np.mean(rh>1.01):.0%}), min ESS median {np.median(es):.0f}, fits with divergences {np.mean(nd>0):.0%}")
    print("  total      ", stats(recs, "total"))
    for j in range(4):
        print(f"  rate bin{j}  ", stats(recs, "rates", j))
    for j in range(4):
        print(f"  rho bin{j}   ", stats(recs, "rho", j))
    print("  rho 0-3 sum", stats(recs, "lightfrac"))
    print("  rho bin10  ", stats(recs, "rho_well"))
