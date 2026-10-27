---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Mixed-stock analysis

A fishery catches salmon from several spawning populations, or *stocks*,
that mix at sea. Managers need the stock proportions of the catch, to keep
weak stocks from being overharvested. Each fish is genotyped, and a
*baseline* of fish sampled on the spawning grounds, whose stock is known,
gives the allele frequencies of every stock. A fish's genotype then has a
likelihood under each stock, and the stock proportions of the mixture are a
population over categories: binference's model with categorical bins.
The same analysis estimates the breeding origins of migrating birds and the
rookeries of foraging sea turtles.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

import binference

rng = np.random.default_rng(10)
```

## A simulated baseline and mixture

Five stocks, A to E, are typed at 60 biallelic loci (SNPs). Allele
frequencies are drawn around a common ancestral frequency with
differentiation $F_{\rm ST}=0.05$; stock B is a sister of stock A, with
frequencies drawn around A's with $F_{\rm ST}=0.01$, so the two are hard to
tell apart. The baseline has 100 fish per stock; the mixture has 1000 fish,
most from A and few from B.

```{code-cell} python
stocks = ["A", "B", "C", "D", "E"]
K, n_loci = 5, 60

def drift(p, fst):
    """Allele frequencies of a population descended from frequencies p."""
    c = (1 - fst) / fst
    return np.clip(rng.beta(p * c, (1 - p) * c), 1e-3, 1 - 1e-3)

ancestral = rng.beta(2, 2, n_loci)
freq = np.array([drift(ancestral, 0.05) for _ in range(K)])
freq[1] = drift(freq[0], 0.01)                        # B is a sister stock of A

n_base = 100
baseline = [rng.binomial(2, freq[k], size=(n_base, n_loci)) for k in range(K)]
rho_true = np.array([0.55, 0.05, 0.25, 0.10, 0.05])
origin = rng.choice(K, 1000, p=rho_true)
mixture = rng.binomial(2, freq[origin])               # allele counts, 0 to 2 per locus
print("realised stock shares of this sample:", (np.bincount(origin, minlength=K) / origin.size).round(3))
```

## Genotype likelihoods

The likelihood of a genotype under stock $k$ assumes Hardy–Weinberg
proportions and independent loci, with the allele frequencies $\hat p_{k\ell}$
estimated from the baseline (half a pseudo-allele added on each side). The
binomial coefficients are the same for every stock and drop out.

```{code-cell} python
counts = np.array([b.sum(axis=0) for b in baseline])  # allele counts per stock and locus

def log_likelihood(genotypes, allele_counts, n_fish):
    p = (allele_counts + 0.5) / (2 * n_fish + 1)
    return genotypes @ np.log(p).T + (2 - genotypes) @ np.log1p(-p).T

loglik = log_likelihood(mixture, counts, n_base)      # (fish, stocks)
bins = binference.Bins.categories(stocks)
lk = binference.Likelihoods.from_table(loglik, bins, kind="likelihood", log=True)
fit = binference.infer(lk, rng=1)
print(fit.info["auto_method"], "-", fit.info["auto_reason"])
```

"Classify and count" assigns each fish to its most likely stock and counts
the assignments. It is biased: misassigned fish move from abundant stocks
to rare ones, so the counts are pulled toward equal proportions.

```{code-cell} python
assigned = loglik.argmax(axis=1)
cc = np.bincount(assigned, minlength=K) / assigned.size
lo, hi = fit.interval(0.9)
print("stock  truth  classify-and-count  binference (90% interval)")
for k, s in enumerate(stocks):
    print(f"  {s}    {rho_true[k]:.2f}   {cc[k]:.3f}               "
          f"{fit.mean[k]:.3f} [{lo[k]:.3f}, {hi[k]:.3f}]")
```

```{code-cell} python
:tags: [remove-cell]

assert cc[0] < rho_true[0] and cc[1] > rho_true[1]
assert np.abs(fit.mean - rho_true).sum() < np.abs(cc - rho_true).sum()
assert not any(lo[k] <= rho_true[k] <= hi[k] for k in (0, 1))
```

The mixture model reduces the bias of counting, but its intervals for the
sister stocks A and B are narrow and miss the truth. The likelihoods treat the
baseline allele frequencies as exact. With 100 fish per stock they are not,
and their errors are largest relative to the small differences between A
and B, which are the differences that separate those two stocks in the
mixture.

## Carrying the baseline's uncertainty

binference cannot infer allele frequencies, but it can infer a confusion
matrix jointly with the proportions. Assign every baseline fish with
frequencies estimated from the other baseline fish (leave one out, so a
fish does not help to classify itself), and tally the assigned stock
against the known stock with {meth}`~binference.Response.from_labelled`.
Each column of the tally gets a Dirichlet posterior, inferred with the
proportions, and the mixture fish enter only through their assigned stock
({meth}`~binference.Likelihoods.from_binned`). This is the confusion-matrix
route of {doc}`06_categories_unfolding`, with the baseline as the labelled
validation set.

```{code-cell} python
labels, predicted = [], []
for k in range(K):
    for g in baseline[k]:
        c, n = counts.copy(), np.full((K, 1), float(n_base))
        c[k] -= g
        n[k] -= 1
        predicted.append(int(log_likelihood(g[None], c, n).argmax()))
        labels.append(k)
labels, predicted = np.array(labels), np.array(predicted)
print(f"baseline self-assignment rate: {np.mean(labels == predicted):.3f}")

response = binference.Response.from_labelled(
    np.array(stocks)[predicted], np.array(stocks)[labels],
    measured_bins=bins, classes=stocks,
)
print("confusion matrix, P(assigned row | true column):")
print(response.matrix.round(2))
```

```{code-cell} python
:tags: [remove-cell]

off = response.matrix - np.diag(np.diag(response.matrix))
assert min(off[0, 1], off[1, 0]) > np.delete(off.ravel(), [1, K]).max()
```

A and B are confused with each other far more often than any other pair.
The fit infers the 25 cells of the confusion matrix with the proportions.
Gibbs mixes slowly for this joint model, because the classes overlap this
much, so `method="auto"` gives way to NUTS:

```{code-cell} python
lk_c = binference.Likelihoods.from_binned(response, values=np.array(stocks)[assigned])
fit_c = binference.infer(lk_c, rng=1)
print(fit_c.info["auto_method"], "-", fit_c.info["auto_reason"])
d = fit_c.diagnostics
print(f"max R-hat {d['rhat'].max():.3f}, min bulk ESS {d['ess_bulk'].min():.0f}, "
      f"divergent {d['n_divergent']}")
```

```{code-cell} python
:tags: [remove-cell]

assert fit_c.info["auto_method"] == "nuts" and d["rhat"].max() < 1.01
```

```{code-cell} python
fig, ax = plt.subplots(figsize=(7, 3.8))
pos = np.arange(K)
for dx, res, name, color in [(-0.12, fit, "genotype likelihoods, baseline fixed", "#2a78d6"),
                             (0.12, fit_c, "confusion matrix inferred", "#1baf7a")]:
    lo, hi = res.interval(0.9)
    ax.errorbar(pos + dx, res.mean, yerr=[res.mean - lo, hi - res.mean], fmt="o",
                color=color, ms=5, capsize=3, label=name)
ax.plot(pos, cc, "s", color="#eb6834", ms=6, label="classify and count")
ax.plot(pos, rho_true, "_", color="k", ms=22, mew=2, label="truth")
ax.set(xticks=pos, xticklabels=stocks, xlabel="stock", ylabel="proportion of the mixture")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8);
```

```{code-cell} python
lo_c, hi_c = fit_c.interval(0.9)
for k, s in enumerate(stocks):
    print(f"  {s}: {fit_c.mean[k]:.3f} [{lo_c[k]:.3f}, {hi_c[k]:.3f}]   truth {rho_true[k]:.2f}")
```

```{code-cell} python
:tags: [remove-cell]

assert all(lo_c[k] <= rho_true[k] <= hi_c[k] for k in (0, 1))
assert (hi_c - lo_c)[2:4].max() < (hi_c - lo_c)[:2].min()
lo_f, hi_f = fit.interval(0.9)
assert ((hi_c - lo_c)[:2] > (hi_f - lo_f)[:2]).all()
```

With the confusion matrix inferred, the intervals for A and B widen to
reflect how weakly the baseline separates them, and they cover the truth.
Stocks C and D, which the baseline separates well, keep narrow
intervals. The price is information: each mixture fish contributes only its
assigned stock, not its full likelihood.

## What to take away

- Stock proportions are a population over categorical bins
  (`Bins.categories`), and genotype likelihoods enter through
  `Likelihoods.from_table(..., kind="likelihood", log=True)`.
- Likelihoods from estimated baseline frequencies are treated as exact, so
  intervals can be too narrow for stocks the baseline separates poorly.
- A confusion matrix tallied from the baseline (`Response.from_labelled`)
  carries that uncertainty, at the cost of using only each fish's assigned
  stock.
