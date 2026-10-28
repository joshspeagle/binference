# Binned measurements and classifiers

Use this page if each object is recorded only by a measured bin $j$. Examples are a
reconstructed energy bin in a detector, a classifier's score bin, or a predicted
label. Such a {term}`binned measurement` reaches the true bins through a
{term}`response matrix`.

## The response

A {class}`~binference.Response` holds the matrix
$R_{jk}=p(\text{measured in }j,\ \text{observed}\mid\text{true bin }k)$. Its column
sums are the {term}`efficiencies <efficiency>` $\epsilon_k\le1$: the probability that an
object in true bin $k$ is observed at all. A response comes from one of three places:

- **A known matrix**: `Response(R, truth_bins, measured_bins)`. A *migration* matrix,
  $p(\text{measured }j\mid\text{true }k,\ \text{observed})$, has columns that sum to
  one. Pass its efficiency separately: `Response(migration, truth_bins,
  efficiency=eps)`. Columns that sum to more than one are rejected.
- **Tallied events**: `Response.from_counts(counts, generated, truth_bins,
  measured_bins)`, or the events themselves, `Response.from_simulation(true_values,
  measured_values, truth_bins, measured_bins)`. A measured value that is NaN, or that
  lies outside the measured bins, marks an event that was not observed.
- **A labelled validation set**, for classifiers: `Response.from_labelled(scores,
  labels)`. Real-valued scores, such as the probability of one class, get 10 score
  bins with equal numbers of validation objects. Predicted labels give the confusion
  matrix.

`open_ended=True` extends the lowest and highest 1-D measured bins to all values below
and above them. `from_labelled` sets it for its score bins.

## From the response to likelihoods

`Likelihoods.from_binned(response, counts=...)` takes the number of objects in each
measured bin. `values=...` takes each object's measured value instead. Every object
measured in bin $j$ gets the likelihood row $R_{j\cdot}$ over the true bins. The
efficiency is attached as a {term}`selection` on the measurement.

```python
import numpy as np
import binference

mu = binference.simulate.mock_unfolding(3000, rng=2)
lk_unfold = binference.Likelihoods.from_binned(mu.response, counts=mu.measurements["reco_counts"])
print(lk_unfold.selection.mode, lk_unfold.K)
```

## A tallied response is inferred with the population

A response tallied from events is known only as well as its events allow. By default
it is inferred together with the population. Given the tallies, each column and its
lost share $1-\epsilon_k$ have a Dirichlet posterior,
$\mathrm{Dir}(\alpha+\text{counts}_{\cdot k},\ \alpha+\text{lost}_k)$, with `prior=`$\alpha$
(one per cell by default). If the tallies record no lost event, as for a classifier's
open-ended score bins, the columns have no lost share.

- **Gibbs**, **NUTS**, **Laplace** and `MAP(space="unconstrained")` infer the
  response with $\rho$. `method="auto"` chooses Gibbs for a conjugate prior.
- **The draws** are in `result.info["response_draws"]`, of shape
  `(chain, draw, J, K)`. The total and the per-object methods use them draw by draw.
- **A fixed response.** `response.fixed()` takes the tallied matrix as known. VB and
  the bootstrap need that. A `Response(matrix)` is always fixed.

With few events the choice matters. With tens of validation objects per class,
intervals from a fixed response covered the truth far less often than their nominal
68%. The inferred response came much closer.

## Classifiers: two routes

A classifier's output reaches the population by one of two routes. Choose by what you
have. `result.summary()` names the inputs in its title.

From calibrated probabilities and the training class mix
: `from_table(probs, Bins.categories(classes), interim=train_mix)` divides the training
  class mix out ({doc}`posteriors`). It is exact only if the probabilities are
  calibrated and `train_mix` is the class mix the classifier learned.

From scores and a labelled set
: `from_binned(Response.from_labelled(val_scores, val_labels), values=scores)` needs
  neither calibration nor the training class mix. It needs only {term}`label shift`:
  the scores within each class are distributed alike in the labelled set and the
  target set. Prefer it when you have a labelled set.

```python
mc = binference.simulate.mock_categorical(1000, 4, rng=3)
lk_prob = binference.Likelihoods.from_table(mc.posteriors, mc.bins, interim=mc.interim)
lk_score = binference.Likelihoods.from_binned(mc.response, values=mc.measurements["feature"])
print(mc.response, lk_score.bins.labels)
```

## Checking the response against the data

When there are more measured bins than true bins, the data can test the response.
`result.check()` compares the measured counts with those each posterior draw predicts
through the response. If the {term}`posterior predictive p-value` is very small, it
warns ([warnings](../../reference/warnings.md#inputwarning)). For a classifier, that
means label shift is violated.

The {doc}`classifier </tutorials/classifier-outputs>` and {doc}`unfolding </tutorials/unfolding>` tutorials run
both classifier routes and an unfolding.
