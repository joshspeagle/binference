# Selection effects and rates

Real catalogues are incomplete. If the probability of making it into the
catalogue depends on the quantity you are binning, the observed objects are
a biased sample of the population, and ignoring that biases $\rho$. pastry
corrects for it given a **selection function**: $S_k$, the probability that
an object whose true value is in bin $k$ is selected.

```python
import numpy as np
import pastry

bins = pastry.Bins([0.0, 1.0, 2.0, 3.0])
S = np.array([0.9, 0.5, 0.25])                 # detection probability per true bin
mock = pastry.simulate.mock_1d(
    3000, bins, density=np.array([0.4, 0.35, 0.25]), noise=(0.3, 0.8), selection=S, rng=7
)
lk = mock.likelihoods.with_selection(None)     # the bare likelihoods
print(mock.truth.rho, mock.truth.n_drawn, "objects drawn,", lk.N, "selected")
```

## Two ways selection can act

The likelihood depends on what the selection acts on, so pastry asks you to
say which ([theory](../theory.md#selection-effects)):

`selection_mode="observed"`
: Objects are selected on their **measured** data, such as a flux limit on
  measured fluxes or a signal-to-noise cut. Each detected object's
  likelihood is used as is, and only the normalisation
  $(S\cdot\rho)^{-W}$ is added. Here $S_k$ is the probability that an object
  in bin $k$ would pass the cut, integrated over its possible data. This
  mode also covers likelihoods that already include the detection
  probability, such as response matrices with efficiencies folded in.

`selection_mode="true"`
: Objects are selected with a probability that depends on their **true**
  value, independently of the measurement noise. Examples are a
  completeness that depends on true planet radius, or a spectroscopic
  targeting rate per true class. Each likelihood row is multiplied by $S_k$,
  and the normalisation is added.

The mock above selects on the true bin:

```python
right = pastry.infer(lk, method="map", selection=S, selection_mode="true").rho
wrong = pastry.infer(lk, method="map", selection=S, selection_mode="observed").rho
ignored = pastry.infer(lk, method="map").rho
print("truth  ", mock.truth.rho)
print("true   ", right.round(3))
print("observed", wrong.round(3))
print("ignored", ignored.round(3))
```

Ignoring selection over-weights the well-detected bins. Using the wrong mode
is just as bad. This is why an array passed as `selection` must come with
`selection_mode`: there is no safe default.

## Where to attach the selection function

Any of these works, and they are equivalent:

```python
sel = pastry.Selection(S, "true")
a = pastry.infer(lk, method="map", selection=sel).rho
b = pastry.infer(lk.with_selection(S, "true"), method="map").rho     # attached to the data
c = pastry.PopulationModel(lk, selection=S, selection_mode="true")    # on the model
print(np.allclose(a, b), c)
```

Simulators and adapters attach the right mode themselves:
`simulate.mock_1d(..., selection=S)` uses `"true"`, and
`Likelihoods.from_response` uses `"observed"` for the efficiency.

For **fractions**, only ratios between the $S_k$ matter, so a selection
function known only up to a constant is fine. Values above one are allowed
too. Bins that can never be observed ($S_k=0$) are not identifiable from the
data; remove them from the bins.

## Rates and exposure

With a `GammaRates` prior pastry infers $\lambda_k$, the expected **number**
of objects per bin, as an inhomogeneous Poisson process. The selection
function then becomes an **exposure** and its absolute scale matters: the
expected number of detections in bin $k$ is $S_k\lambda_k$. This is how
occurrence rates per star, or counts per unit time or area, are estimated.

```python
est = pastry.infer(
    lk, pastry.priors.GammaRates(1.0, 1e-4), method="map", selection=S, selection_mode="true"
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
  without telling pastry about the selection. That corrects for nothing.
