# Selection effects and rates

Real catalogues are incomplete. If the probability of making it into the
catalogue depends on the quantity you are binning, the observed objects are
a biased sample of the population, and ignoring that biases $\rho$. binference
corrects for it given a **selection function**: $S_k$, the probability that
an object whose true value is in bin $k$ is selected.

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

## Three ways selection can act

The likelihood depends on what the selection acts on, so binference asks you to
say which ([theory](../theory.md#selection-effects)):

`selection_mode="on_measurement"`
: Objects are selected on their **measured** data, such as a flux limit on
  measured fluxes or a signal-to-noise cut. Each detected object's
  likelihood is used as is, and only the normalisation
  $(S\cdot\rho)^{-W}$ is added. Here $S_k$ is the probability that an object
  in bin $k$ would pass the cut, integrated over its possible data. This
  mode also covers likelihoods that already include the detection
  probability, such as response matrices with efficiencies folded in.

`selection_mode="on_truth"`
: Objects are selected with a probability that depends on their **true**
  value, independently of the measurement noise. Examples are a
  completeness that depends on true planet radius, or a spectroscopic
  targeting rate per true class. Each likelihood row is multiplied by $S_k$,
  and the normalisation is added.

`Selection.combined(on_truth=C, on_measurement=T)`
: Both at once: $P(\text{observed}\mid z, d) = C(z)\,T(d)$, for example a
  redshift-dependent completeness together with a magnitude limit. Rows are
  multiplied by $C_k$ only, and the normalisation uses $S_k$, the bin average of
  $C(z)\,P(d\in A\mid z)$.

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

Ignoring selection over-weights the well-detected bins. Using the wrong mode
is just as bad. This is why an array passed as `selection` must come with
`selection_mode`: there is no safe default.

## Where to attach the selection function

Any of these works, and they are equivalent:

```python
sel = binference.Selection(S, "on_truth")
a = binference.infer(lk, method="map", selection=sel).rho
b = binference.infer(lk.with_selection(S, "on_truth"), method="map").rho     # attached to the data
c = binference.PopulationModel(lk, selection=S, selection_mode="on_truth")    # on the model
print(np.allclose(a, b), c)
```

Simulators and adapters attach the right selection themselves:
`simulate.mock_1d(..., selection=S)` uses `"on_truth"`,
`simulate.mock_1d(..., measured_range=(lo, hi))` (a cut on the measured value)
uses `"on_measurement"`, both together give a combined selection, and
`Likelihoods.from_binned` uses `"on_measurement"` for the efficiency.

Selection values are detection probabilities, between zero and one, by default
(DECISIONS #46). For **fractions** only ratios between the $S_k$ matter, so a
selection known only up to a constant is fine if you say so with
`Selection(S, mode, scale="relative")`; rates and totals then cannot use it.
`scale="exposure"` declares absolute values that may exceed one (below). Bins
that can never be observed ($S_k=0$) are not identifiable from the data; remove
them from the bins.

## Selection that varies within bins

The model needs $S_k$ averaged over each bin with the population's own
within-bin shape. An array evaluated at bin centres is wrong where $S$ changes
quickly, and the error goes into the fractions of the bins near the edge.
binference warns when an array changes by more than a factor of two between
neighbouring bins (DECISIONS #47). Pass the selection as a function of the true
value instead, and it is averaged over each bin in the measure coordinate
(`Selection.from_function(f, bins, mode)`, or a callable `selection=` with
`selection_mode`). Only a raw array is checked; values that are already bin
averages, for example from a simulation, can be passed as a `Selection(S, mode)`
object.

```python
completeness = lambda z: 1.0 / (1.0 + np.exp((np.asarray(z) - 2.0) / 0.1))  # a sharp edge
smooth = binference.infer(lk, method="map", selection=completeness, selection_mode="on_truth")
print(smooth.rho.round(3))
```

## Selection measured by injections

Completeness is often measured by injecting objects of known true value and
counting how many are recovered. `Selection.from_injections(z, detected, bins,
"on_truth")` counts the injections $n_k$ and recoveries $d_k$ per bin and gives
each $S_k$ a $\mathrm{Beta}(a + d_k, b + n_k - d_k)$ posterior (prior
`(a, b) = (1, 1)`). The selection is then uncertain and is inferred with the
population: the catalogue's objects count as detections in their bins, and the
objects it missed as non-detections. The catalogue informs only the ratios
between the $S_k$ (its likelihood does not change when all of them are scaled
together), so their level, and with it the total, comes from the injections.
The draws of $S$ are in `result.info["selection_draws"]`, and the total and the
per-object methods use them draw by draw.

```python
rng = np.random.default_rng(3)
z_inj = rng.uniform(0.0, 3.0, 600)
recovered = rng.random(600) < S[mock.likelihoods.bins.index(z_inj)]  # the mock's S
measured = binference.Selection.from_injections(z_inj, recovered, mock.likelihoods.bins, "on_truth")
joint = binference.infer(lk, selection=measured, method="gibbs", n_samples=300, rng=1, check=False)
print(joint.mean.round(3), joint.info["selection_draws"].mean(axis=(0, 1)).round(2))
```

Gibbs samples $S$ from its conjugate conditional (independent Betas and a
conjugate prior; `method="auto"` chooses it then). NUTS, Laplace and
`MAP(space="unconstrained")` add $y_k = \mathrm{logit}\,S_k$ to the sampled
coordinates, so they work with every prior; NUTS and Gibbs agree within Monte
Carlo error. VB and the bootstrap refuse an uncertain selection, and
`measured.at_mean()` gives a fixed selection at its values, ignoring its
uncertainty. A selection on the measurement is fixed by the likelihoods' own
cut, so injections give it only as such a plug-in.

With few injections per bin, `smooth=τ` ties neighbouring bins together: a
second-order random walk on $\mathrm{logit}\,S$, with step scale τ per bin, lends
sparsely injected bins strength from their neighbours (1-D bins; not with
Gibbs). Its `values` are the mode of $\mathrm{logit}\,S$ given the injections. With
20 bins and 5 injections per bin, τ = 0.3 lowered the error of $S$ from
0.17 to 0.13 for a smooth logistic edge and from 0.16 to 0.09 for a linear
completeness; for a step from 0.9 to 0.1 it did not help (0.171 against 0.169),
and τ = 0.1 raised the error to 0.196 (RMSE over bins and 200 simulations, Monte
Carlo error 0.002). τ is the typical change from one bin to the next in the slope
of $\mathrm{logit}\,S$ per bin; a sharp edge needs independent bins or a larger τ.

```python
smooth = binference.Selection.from_injections(
    z_inj, recovered, mock.likelihoods.bins, "on_truth", smooth=1.0
)
fit = binference.infer(lk, selection=smooth, method="nuts", n_samples=300, rng=1, check=False)
print(fit.mean.round(3), fit.info["selection_draws"].mean(axis=(0, 1)).round(2))
```

## Rates and exposure

Every fit of fractions also gives the total and the rates. With $W$ detected
objects and an absolute selection $S$ (detection probabilities, or exposures
with `scale="exposure"`), the number of detections is Poisson with mean
$\Lambda\,S\cdot\rho$, so under the scale-free prior $p(\Lambda)\propto
1/\Lambda$ the total given the fractions is exactly
$\Lambda\mid\rho \sim \mathrm{Gamma}(W, S\cdot\rho)$. `result.total` holds
one such draw per draw of $\rho$ and `result.rates` the rates
$\lambda_k = \Lambda\rho_k$, the expected **number** of objects per bin
(per unit exposure); a point estimate gives $\hat\Lambda = W / (S\cdot\rho)$.
$S$ is taken at face value here: a selection normalised to a maximum of one
must be declared `scale="relative"`, and then the total is not identified and
both are `None`. With non-integer weights $W$ is their sum and the Gamma is a
pseudo-posterior, like the fractions' (see {doc}`inputs`).

```python
fit = binference.infer(
    lk, method="gibbs", n_samples=300, rng=1, check=False,
    selection=binference.Selection(S, "on_truth", scale="exposure"),
)
print("total:", fit.total.mean().round(0), "+/-", fit.total.std().round(0),
      " actually drawn:", mock.truth.n_drawn)
```

A `GammaRates` prior instead puts independent priors on the rates themselves
(bin-specific shapes and rates, or an informative total), as an inhomogeneous
Poisson process; the selection then acts as an exposure in the same way. This
is how occurrence rates per star, or counts per unit time or area, are
estimated with prior information on them.

```python
est = binference.infer(
    lk, binference.priors.GammaRates(1.0, 1e-4), method="map",
    selection=binference.Selection(S, "on_truth", scale="exposure"),
)
print("expected total:", est.rates.sum().round(0), " actually drawn:", mock.truth.n_drawn)
```

The inferred total accounts for the objects that were never detected.

## Common mistakes

These are the two errors that Mandel, Farr & Gair (2019) warn about:

- **Renormalising each object's likelihood** over the detectable region *and*
  dividing by $S\cdot\rho$, which corrects for selection twice. Pass the
  ordinary likelihood $p(d_i\mid z)$ at the observed data.
- **Omitting the normalisation**, for example multiplying likelihoods by $S$
  without telling binference about the selection. That corrects for nothing.
