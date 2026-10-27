---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Selection and rates

Real catalogues are incomplete. If the chance of entering the catalogue depends on the
quantity you bin, the observed objects are a biased sample of the population. Ignoring
that biases the fractions $\rho$. binference corrects for it, given a
{term}`selection function`: $S_k$, the probability that an object whose true value lies
in bin $k$ is observed. The same correction gives the total number of objects,
including those the catalogue missed.

```python
import numpy as np
import binference

bins = binference.Bins([0.0, 1.0, 2.0, 3.0])
S = np.array([0.9, 0.5, 0.25])                 # detection probability per true bin
mock = binference.simulate.mock_1d(
    3000, bins, density=np.array([0.4, 0.35, 0.25]), noise=(0.3, 0.8), selection=S, rng=7
)
lk = mock.likelihoods.with_selection(None)     # the bare likelihoods
print(mock.truth.rho, mock.truth.n_drawn, "objects drawn,", lk.N, "selected")
```

## Say how the selection acts

The likelihood depends on what the selection acts on, so binference asks you to say
which: the {term}`selection mode`
([theory](../how-it-works/theory.md#selection-effects)).

`selection_mode="on_measurement"`
: Objects are selected on their **measured** data: a flux limit on measured fluxes, or
  a signal-to-noise cut. Here $S_k$ is the probability that an object in bin $k$
  passes the cut, averaged over its possible data. This mode also covers likelihoods
  that already include the detection probability, such as a response with its
  efficiencies.

`selection_mode="on_truth"`
: Objects are selected with a probability that depends on their **true** value,
  whatever the measurement noise. Examples are a completeness that depends on true
  planet radius, or a targeting rate per true class.

`Selection.combined(on_truth=..., on_measurement=...)`
: Both at once: $P(\text{observed}\mid z,d)=S^{\rm truth}(z)\,S^{\rm meas}(d)$, for
  example a completeness in redshift together with a magnitude limit.

How each mode enters the likelihood is in the
[theory](../how-it-works/theory.md#selection-effects).

The mock above selects on the true bin:

```python
right = binference.infer(lk, method="map", selection=S, selection_mode="on_truth").rho
wrong = binference.infer(lk, method="map", selection=S, selection_mode="on_measurement").rho
ignored = binference.infer(lk, method="map").rho
print("truth         ", mock.truth.rho)
print("on_truth      ", right.round(3))
print("on_measurement", wrong.round(3))
print("ignored       ", ignored.round(3))
```

Ignoring the selection over-weights the well-detected bins. The wrong mode is just as
bad. That is why an array passed as `selection` must come with `selection_mode`: there
is no safe default.

## Where to attach it

These three are equivalent:

```python
sel = binference.Selection(S, "on_truth")
a = binference.infer(lk, method="map", selection=sel).rho
b = binference.infer(lk.with_selection(S, "on_truth"), method="map").rho     # on the data
c = binference.PopulationModel(lk, selection=S, selection_mode="on_truth")    # on the model
print(np.allclose(a, b), c)
```

The simulators and `from_binned` attach the right selection themselves.
`simulate.mock_1d(..., selection=S)` selects on the truth, and
`measured_range=(lo, hi)` cuts on the measured value. `Likelihoods.from_binned`
attaches the efficiency on the measurement.

Selection values are detection probabilities, between zero and one, by default. For
the fractions only ratios between the $S_k$ matter. If you know $S$ only up to a
constant, say so with `Selection(S, mode, scale="relative")`; totals and rates are then
not available. `scale="exposure"` declares absolute values that may exceed one, such as
an observing time. A bin that can never be observed ($S_k=0$) cannot be inferred:
remove it from the bins.

## Selection that varies within bins

The model needs $S_k$ averaged over each bin. An array evaluated at bin centres is wrong
where $S$ changes quickly, and the error goes into the fractions of the bins near the
edge. binference warns when an array changes steeply between neighbouring bins
([warnings](../reference/warnings.md#inputwarning)).

If that happens, pass the selection as a function of the true value. It is then
averaged over each bin in the {term}`measure` coordinate: use
`Selection.from_function(f, bins, mode)`, or pass the function as `selection=` with
`selection_mode`. Only a raw array is checked. Values that are already bin averages,
for example from a simulation, can be passed as `Selection(S, mode)`.

```python
def completeness(z):
    return 1.0 / (1.0 + np.exp((np.asarray(z) - 2.0) / 0.1))   # a sharp edge


smooth = binference.infer(lk, method="map", selection=completeness, selection_mode="on_truth")
print(smooth.rho.round(3))
```

## Selection measured by injections

Completeness is often measured by {term}`injections`: objects of known true value are
added to the data, and the pipeline's recoveries are counted. For each bin,
`Selection.from_injections(z, detected, bins, "on_truth")` counts the injections
$n^{\rm inj}_k$ and the recoveries $n^{\rm rec}_k$. Each $S_k$ then gets a
$\mathrm{Beta}(a+n^{\rm rec}_k,\ b+n^{\rm inj}_k-n^{\rm rec}_k)$ posterior, with
`prior=(a, b)` of `(1, 1)` by default.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  image:
    alt: >-
      Two panels sharing the axis of true values from 0 to 3. Top: the number of
      injections per bin, about 30 in each, as an outline, and the number recovered as
      filled steps, close to the injected number below 1.5 and falling to almost none
      above 2.4. Bottom: the selection inferred in each bin, as a median step with 68%
      and 95% bands, following the dashed true completeness curve that falls from 0.95
      to near zero around z = 1.8.
  figure:
    caption: >-
      Injections give the selection with its uncertainty. Top: 450 injections, uniform
      in the true value, and the ones the pipeline recovered. Bottom: the posterior of
      $S_k$ in each bin from those counts (`Selection.from_injections`), against the
      true completeness (dashed). Bins with few injections get wide bands.
---
import matplotlib.pyplot as plt
import numpy as np

import binference

ink, coral = plt.rcParams["text.color"], "#EC5F3E"


def bands(ax, e, X, label):
    """Median steps with 68% and 95% bands of draws X (n_draws, K) on edges e."""
    q = np.quantile(X, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
    step = np.append(q, q[:, -1:], axis=1)
    ax.fill_between(e, step[0], step[4], step="post", color="C0", alpha=0.2, lw=0,
                    label="95% interval")
    ax.fill_between(e, step[1], step[3], step="post", color="C0", alpha=0.35, lw=0,
                    label="68% interval")
    ax.stairs(q[2], e, baseline=None, color="C0", lw=2, label=label)


def edge(z):
    return 0.95 / (1.0 + np.exp((np.asarray(z) - 1.8) / 0.25))


rng_fig = np.random.default_rng(5)
fine = binference.Bins.linear(0.0, 3.0, 15)
z_fig = rng_fig.uniform(0.0, 3.0, 450)
found = rng_fig.random(450) < edge(z_fig)
S_draws = binference.Selection.from_injections(z_fig, found, fine, "on_truth").draw(4000, rng=1)
e = fine.edges[0]
with plt.rc_context({"font.size": 12}):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.0, 6.4), sharex=True,
                                      constrained_layout=True)
    top.stairs(np.histogram(z_fig, e)[0], e, color=ink, lw=1.5, label="injected")
    top.stairs(np.histogram(z_fig[found], e)[0], e, color=coral, fill=True, alpha=0.8,
               label="recovered")
    top.set_ylabel("injections per bin")
    top.legend(loc="upper right")
    bands(bottom, e, S_draws, "median")
    zz = np.linspace(0, 3, 400)
    bottom.plot(zz, edge(zz), ls="--", color=ink, label="truth")
    bottom.set(xlabel="true value $z$", ylabel="selection $S$", ylim=(0, 1.05))
    bottom.legend(loc="lower left")
    plt.show()
```

The selection is then uncertain, and it is inferred with the population. The
catalogue's objects count as detections in their bins, and the objects it missed as
non-detections. The catalogue informs only the ratios between the $S_k$: its
likelihood does not change when all of them are scaled together. Their level, and with
it the total, comes from the injections. The draws of $S$ are in
`result.info["selection_draws"]`, and the total and the per-object methods use them
draw by draw.

```python
rng = np.random.default_rng(3)
z_inj = rng.uniform(0.0, 3.0, 600)
recovered = rng.random(600) < S[bins.index(z_inj)]     # the mock's S
measured = binference.Selection.from_injections(z_inj, recovered, bins, "on_truth")
joint = binference.infer(lk, selection=measured, method="gibbs", n_samples=300, rng=1, check=False)
print(joint.mean.round(3), joint.info["selection_draws"].mean(axis=(0, 1)).round(2))
```

Which engines infer it:

- **Gibbs** samples $S$ from its conjugate conditional, given independent Betas and a
  conjugate prior. `method="auto"` chooses it then.
- **NUTS, Laplace** and `MAP(space="unconstrained")` add $\mathrm{logit}\,S_k$ to the
  sampled coordinates, so they work with every prior. NUTS and Gibbs agree within
  Monte Carlo error.
- **VB and the bootstrap** refuse an uncertain selection. `measured.at_mean()` gives a
  fixed selection at its mean values, ignoring their uncertainty.

A selection on the measurement is fixed by the likelihoods' own cut. Injections give it
only as such a fixed value.

### Few injections per bin

With few injections per bin, `smooth=`$\tau_S$ ties neighbouring bins together. It puts
a second-order random walk on $\mathrm{logit}\,S$, with step scale $\tau_S$, and works
on 1-D bins with every engine but Gibbs. Its `values` are the mode of
$\mathrm{logit}\,S$ given the injections. $\tau_S$ is the typical change, from one bin
to the next, of the slope of $\mathrm{logit}\,S$ per bin.

In one test with 20 bins and 5 injections per bin, $\tau_S=0.3$ lowered the error of
$S$ from 0.17 to 0.13 for a smooth logistic edge. For a linear completeness it lowered
it from 0.16 to 0.09. For a step from 0.9 to 0.1 it did not help (0.171 against 0.169),
and $\tau_S=0.1$ raised the error to 0.196. These are RMS errors over bins and 200
simulations, with Monte Carlo error 0.002. A sharp edge needs independent bins or a
larger $\tau_S$.

```python
smooth = binference.Selection.from_injections(z_inj, recovered, bins, "on_truth", smooth=1.0)
fit = binference.infer(lk, selection=smooth, method="nuts", n_samples=300, rng=1, check=False)
print(fit.mean.round(3), fit.info["selection_draws"].mean(axis=(0, 1)).round(2))
```

## Totals and rates

Every fit of fractions also gives the {term}`total` and the {term}`rates`. Suppose $W$
objects were detected, and $S$ is absolute: detection probabilities, or exposures with
`scale="exposure"`. The number of detections is then Poisson with mean
$\Lambda\,S\cdot\rho$, where $\Lambda$ is the total expected number of objects.

- **The total.** Under the scale-free prior $p(\Lambda)\propto1/\Lambda$, the total
  given the fractions is $\Lambda\mid\rho\sim\mathrm{Gamma}(W,\ S\cdot\rho)$.
  `result.total` holds one such draw per draw of $\rho$. A point estimate gives
  $\hat\Lambda=W/(S\cdot\rho)$.
- **The rates.** `result.rates` holds $\lambda_k=\Lambda\rho_k$: the expected number of
  objects per bin, per unit exposure.
- **A relative selection** identifies no total, and both are `None`.
- **Non-integer weights** make $W$ their sum, and the Gamma a pseudo-posterior
  ({doc}`inputs/advanced`).

```python
fit = binference.infer(
    lk, method="gibbs", n_samples=300, rng=1, check=False,
    selection=binference.Selection(S, "on_truth", scale="exposure"),
)
print("total:", fit.total.mean().round(0), "+/-", fit.total.std().round(0),
      " actually drawn:", mock.truth.n_drawn)
```

The total inherits the prior on the fractions. With $p(\Lambda)\propto1/\Lambda$, the
default flat Dirichlet on $\rho$ is a prior on the rates proportional to
$\Lambda^{-K}$. That pulls the total down when it is poorly determined, for example
when some bins hold few detections. Check a total's calibration on simulations
({doc}`checking`) before you quote it from such data.

### The objects the catalogue missed

`result.realized_counts(full=True)` draws the histogram of the full sample: the
detected objects in their bins, plus the missed ones. For each draw, the number missed
is negative binomial, and they fall in bins in proportion to $\rho_k(1-S_k)$.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  image:
    alt: >-
      Objects per bin against the true value. Filled steps show the detected objects,
      falling from about 500 per bin at z = 0.5 to almost none at z = 2.5. An outline
      step with a band shows all objects, detected and missed, which stays well above
      the detected counts in the bins where detection is unlikely. A dashed step shows
      the true number of all objects, inside the band.
  figure:
    caption: >-
      From detected counts to the full sample. 3000 objects were drawn and each was
      detected with a probability that falls from 0.95 to 0.15 across the bins. The
      fit sees the 2000 or so detected ones (filled) and the selection. The median with
      68% and 95% bands of `realized_counts(full=True)` add the missed objects; the
      dashed line is the true histogram of all 3000.
---
coarse = binference.Bins.linear(0.0, 3.0, 10)
S10 = np.linspace(0.95, 0.15, 10)
labels = rng_fig.choice(10, size=3000, p=binference.simulate.population(coarse, "bimodal"))
e = coarse.edges[0]
z_all = rng_fig.uniform(e[labels], e[labels + 1])
seen = rng_fig.random(3000) < S10[labels]
err = rng_fig.uniform(0.05, 0.2, seen.sum())
lk_seen = binference.Likelihoods.from_gaussian(
    z_all[seen] + err * rng_fig.standard_normal(seen.sum()), err, coarse, warn_outside=False
)
fit_seen = binference.infer(lk_seen, selection=binference.Selection(S10, "on_truth"), rng=1)
detected = np.median(fit_seen.realized_counts(rng=2), axis=0)
with plt.rc_context({"font.size": 12}):
    fig, ax = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
    ax.stairs(detected, e, color=coral, fill=True, alpha=0.8, label="detected objects")
    bands(ax, e, fit_seen.realized_counts(rng=2, full=True), "all objects, inferred")
    ax.stairs(np.bincount(labels, minlength=10), e, baseline=None, color=ink, ls="--",
              lw=1.5, label="all objects, truth")
    ax.set(xlabel="true value $z$", ylabel="objects per bin")
    ax.set_ylim(0, 1.45 * ax.get_ylim()[1])
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    plt.show()
```

### Priors on the rates

A `GammaRates` prior puts independent priors on the rates themselves: bin-specific
shapes and rates, or an informative total. The data are then an inhomogeneous Poisson
process, and the selection acts as an exposure in the same way. Occurrence rates per
star, or counts per unit time or area, are estimated this way.

```python
est = binference.infer(
    lk, binference.priors.GammaRates(1.0, 1e-4), method="map",
    selection=binference.Selection(S, "on_truth", scale="exposure"),
)
print("expected total:", est.rates.sum().round(0), " actually drawn:", mock.truth.n_drawn)
```

## Next

Two mistakes are common: correcting for the selection twice, and not at all
([pitfall](troubleshooting.md#ignoring-selection-or-modelling-it-twice)).

The {doc}`selection tutorial </tutorials/05_selection>` and the
{doc}`populations-with-selection tutorial </tutorials/11_populations_with_selection>`
work through both modes and injections.
