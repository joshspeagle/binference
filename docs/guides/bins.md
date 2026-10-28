---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Choose bins

The bins are part of the model. The model takes the population to be flat within each
bin, so the bins decide which shapes the fit can follow. This page says where to put the bins,
how many to use, and how to check the choice.

## Where to put them

- **Cover the data.** Every object's likelihood should lie mostly inside the bins. If
  many do not, binference warns ({doc}`inputs/gaussian`). Widen the range rather than
  dropping objects.
- **Pick the coordinate.** For a quantity that spans decades, such as a mass or a flux,
  use `Bins.log`: the population is then flat in $\log z$ within each bin. The
  {term}`measure` sets this ({doc}`inputs/advanced`).
- **Unequal bins are fine.** If you use them, say whether your inputs are
  {term}`masses <mass>` or densities ({doc}`inputs/tables`).
- **Categories need no choice.** Classes, stocks and species are their own bins.

## How many

Too few bins and too many bins cause different errors.

Too few bins
: A bin wider than a feature of the population hides the feature. Worse, it biases the
  fractions: the bin-averaged likelihood weights the wrong parts of the bin. This
  {term}`binning bias` does not shrink as objects are added, while the error bars do.

Too many bins
: Each bin then holds few objects' worth of data. Its fraction is set by the prior
  more than by the data, and the fractions of neighbouring bins are strongly
  anti-correlated. The cost of a fit also grows
  with the number of bins.

A good default is bins no wider than the features you care about, and no narrower than
the typical measurement error. If you need narrower bins, add a {term}`smoothness prior`
({doc}`priors`). Report coarse summaries as functions of the fine fit, with
`result.rebin` and `result.functional` ({doc}`results`).

## What binning bias looks like

```{code-cell} python
---
tags: [remove-input]
mystnb:
  image:
    alt: >-
      Two panels. Top: a narrow peak of true density near z = 1.4. It is drawn with the
      step-shaped fit that is flat in each bin and the continuous fit on hat components.
      The steps miss the peak's shape; the hats follow it more closely. Bottom: on log axes, the
      error of the fraction in bin 1.2 to 1.5 against the number of objects. The
      posterior standard deviation falls as N grows. The bias of each fit is a
      horizontal line. The standard deviation crosses the flat fit's line first, and the
      lower hat fit's line at a larger N.
  figure:
    caption: >-
      Binning bias for a population with a peak 0.25 wide, ten bins 0.3 wide, and
      Gaussian errors of 0.2. Top: the fits with unlimited data, flat in each bin
      (steps) and on hat components (`Basis(bins, order=1)`), against the truth
      (dashed). Bottom: in the shaded bin, each fit's bias against the posterior s.d.
      of the flat fit, measured on 20,000 objects and scaled as $1/\sqrt N$. Where the
      lines cross, the bias equals the error bar.
---
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from myst_nb import glue

import binference



def peak(z):
    return np.exp(-0.5 * ((np.asarray(z) - 1.4) / 0.25) ** 2) / (0.25 * np.sqrt(2 * np.pi))


sig = 0.2
bins = binference.Bins.linear(0.0, 3.0, 10)
truth = binference.simulate.population(bins, peak)

# unlimited data: measurements on a fine grid, weighted by their exact density
zq = np.linspace(0, 3, 3001)
d = np.linspace(-0.8, 3.8, 1200)
pd = np.trapezoid(peak(zq)[None] * stats.norm.pdf(d[:, None], zq[None], sig), zq, axis=1)
w = 1e5 * pd / pd.sum()


def limit(target):
    with warnings.catch_warnings():  # the fractional weights warn; that is intended here
        warnings.simplefilter("ignore", binference.InputWarning)
        lk = binference.Likelihoods.from_gaussian(
            d, np.full(d.size, sig), target, weights=w, fractional_weights=True,
            warn_outside=False,
        )
        return binference.infer(lk, method="map")  # flat Dirichlet: the maximum-likelihood fit


flat = limit(bins)
basis = binference.Basis(bins, order=1)
hat = limit(basis)

# the hat fit's density: unit-mass components, weighted
zz = np.linspace(0, 3, 601)
H = basis.height(np.arange(basis.C)[:, None], 0, zz[None, :])
H /= np.trapezoid(H, zz, axis=1)[:, None]

# the posterior s.d. of a fit to 20,000 objects
mock = binference.simulate.mock_1d(20000, bins, density=peak, noise=sig, rng=3)
fit = binference.infer(mock.likelihoods, method="laplace", rng=1)
k = int(np.argmax(np.abs(flat.rho - truth)))
bias = {"flat in each bin": abs(flat.rho - truth)[k], "hat basis": abs(hat.rho - truth)[k]}

ink = plt.rcParams["text.color"]
e = bins.edges[0]
fig, (a, b) = plt.subplots(2, 1)
fig.set_figheight(6.6)
a.axvspan(e[k], e[k + 1], color="C0", alpha=0.12, lw=0)
a.plot(zz, peak(zz), ls="--", color=ink, label="truth")
a.stairs(flat.rho / bins.volumes, e, color="C0", lw=2, label="flat in each bin")
a.plot(zz, hat.value @ H, color="C1", lw=2, label="hat basis")
a.set(xlabel="true value $z$", ylabel="density", xlim=(0.3, 2.7))
fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncol=2)
a.set_title("Fits with unlimited data", loc="left")

N = np.logspace(2, 6, 50)
sd = fit.std[k] * np.sqrt(mock.likelihoods.W / N)
b.loglog(N, sd, color=ink, lw=1.5)
b.text(2e5, sd[-1] * 3.2, "posterior s.d.", ha="center")
for (name, value), c in zip(bias.items(), ["C0", "C1"]):
    b.axhline(value, color=c, lw=2)
    b.text(1.2e2, value * 1.15, f"bias, {name}", color=c)
    b.plot(mock.likelihoods.W * (fit.std[k] / value) ** 2, value, "o", color=c)
b.set(xlabel="number of objects $N$", ylabel="error of the fraction")
b.set_title(f"Bin [{e[k]:.1f}, {e[k + 1]:.1f})", loc="left")
plt.show()

n_equal = mock.likelihoods.W * (fit.std[k] / bias["flat in each bin"]) ** 2
glue("bins_n_equal", float(f"{n_equal:.1g}"), display=False)
glue("bins_ratio", float(bias["flat in each bin"] / bias["hat basis"]), display=False)
```

The bias is largest where the population curves most within a bin. Here it equals the
error bar at about {glue:text}`bins_n_equal:,.0f` objects. Beyond that, adding objects
makes the answer more precise but no more accurate. Hat components follow the curve
within each bin, and their bias is {glue:text}`bins_ratio:.0f` times smaller. Narrower bins with a smoothness prior reduce it
further.

## Check the binning

Two checks on a fitted result tell you whether the bins suit your data.

**Binning bias.** `result.binning_bias()` refits the same data on hat components over
the same bins. For each bin it reports the shift of the mean fraction, the fit's
posterior s.d., and their ratio, in column `z`. It warns when $|z|$ exceeds its threshold in any
bin. Gaussian likelihoods rebuild themselves. For other inputs, pass `build`, a
function that returns your data's likelihoods on the bins or basis it is given.

```{code-cell} python
bins = binference.Bins.linear(0.0, 3.0, 10)
mock = binference.simulate.mock_1d(5000, bins, density=peak, noise=0.2, rng=3)
result = binference.infer(mock.likelihoods, method="laplace", n_proposals=40000, rng=1)
check = result.binning_bias(rng=2, n_proposals=40000)
print(check)
```

```{code-cell} python
:tags: [remove-cell]
from IPython.display import Markdown

names = [f"[{a:.1f}, {b:.1f})" for a, b in zip(bins.edges[0][:-1], bins.edges[0][1:])]
top = int(np.argmax(binference.simulate.population(bins, peak)))
flagged = [k for k in range(bins.K) if abs(check.z[k]) > 2]
text = "Here " + ", ".join(f"bin {names[k]} (z = {check.z[k]:.1f})" for k in flagged)
text += " is flagged." if len(flagged) == 1 else " are flagged."
if top not in flagged:
    text += (
        f" The peak bin {names[top]} is not: its shift of {check.shift[top]:.3f} is"
        f" {abs(check.z[top]):.2f} times its posterior s.d. of {check.sd[top]:.3f},"
        " inside the threshold."
    )
glue("bins_flagged", Markdown(text), display=False)
```

{glue:md}`bins_flagged`

For a flagged bin, `n_crit` is the number of objects at which the shift equals the
statistical error. Run the check on a pilot sample to learn how many objects these bins
can take. If bins are flagged, fit on `binference.Basis(bins, order=1)`, or use
narrower bins with a smoothness prior.

**Data support.** `result.data_support` is each bin's {term}`data support`: its
objects' worth of likelihood. A bin with less than about one object's worth is
set by the prior. The summary shows it as `support`, and `result.check(support_min=1)`
flags such bins.

```{code-cell} python
print(result.data_support.round(1))
```

If the outer bins have little support, merge them or narrow the range. To compare two
binnings directly, score each fit on held-out objects with `result.log_predictive`
({doc}`results`).

The theory behind the bias is in
[binning is a modelling choice](../how-it-works/model.md#binning-bias).
