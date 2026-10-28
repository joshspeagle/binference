---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Classifier outputs: class fractions under label shift

A classifier is trained on one set of objects and applied to another, whose class
fractions differ. This change is {term}`label shift`. The fractions of the classes in the
new set, $\rho_k$ (the prevalence), are what binference infers. This tutorial follows two
routes, named by what you have:

- **posteriors and the training prevalence**: the classifier's class probabilities and
  the class fractions it was trained with;
- **scores and a labelled set**: any score from the classifier, and a labelled set that
  shows how each class scores.

```{code-cell} python
:tags: [remove-cell]

# figures 6 in wide with 17 pt text keep text above 12 px on a 390 px phone screen
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (6, 3.4), "font.size": 17,
                     "figure.constrained_layout.use": True})
```

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

import binference

OBJECT = "#EC5F3E"   # the colour of single objects in these docs
```

## Posteriors and the training prevalence

A classifier trained on balanced classes is applied to a set whose classes are skewed.
Its probability $P_{ik}$ that object $i$ is in class $k$ includes the training
prevalence $\Pi_k$ as a prior. {func}`binference.simulate.mock_categorical` simulates a
calibrated classifier on four overlapping classes:

```{code-cell} python
mock = binference.simulate.mock_categorical(3000, n_classes=4, separation=1.2, rng=5)
probs, train_prior = mock.posteriors, mock.interim
print("true class fractions:", mock.truth.rho.round(3), " training prevalence:", train_prior)
```

Two shortcuts are common. "Classify and count" assigns each object to its most probable
class and counts. "Probabilistic classify and count" averages the probabilities. Both are
biased toward the training prevalence. binference treats the training prevalence as the
{term}`interim prior` and divides it out. This is the Bayesian form of the EM method of
{cite:t}`saerens2002`:

```{code-cell} python
lk = binference.Likelihoods.from_table(probs, mock.bins, interim=train_prior)
fit = binference.infer(lk, rng=1)

cc = np.bincount(probs.argmax(axis=1), minlength=4) / len(probs)
pcc = probs.mean(axis=0)
lo, hi = fit.interval(0.95)
for k, name in enumerate(mock.bins.labels):
    print(f"{name}: truth {mock.truth.rho[k]:.3f} | CC {cc[k]:.3f} | PCC {pcc[k]:.3f} | "
          f"binference {fit.mean[k]:.3f} [{lo[k]:.3f}, {hi[k]:.3f}]")
```

```{code-cell} python
---
mystnb:
  figure:
    caption: Class fractions inferred by binference (bars with 68% intervals), the truth
      (diamonds) and the averaged probabilities (squares).
  image:
    alt: Bar chart of four classes with fractions rising from about 0.03 to 0.53. The
      diamonds of the truth sit at the bar tops. The squares of the averaged
      probabilities lie above the truth for the rare classes and below it for the
      common one.
---
fig, ax = plt.subplots(figsize=(6, 4.2))
fit.plot(ax=ax, which="fractions", truth=mock.truth.rho, stack=pcc, sample_color=OBJECT,
         label="binference")
ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1))
ax.set_xlabel("class");
```

The averaged probabilities, labelled "stacked", lie closer to equal fractions than the
truth does.

Each object's posterior under the inferred fractions, `fit.assignment_probabilities()`,
is the classifier's output recalibrated to the new set. Use it to classify objects in the
new set:

```{code-cell} python
recal = fit.assignment_probabilities()
acc_before = np.mean(probs.argmax(axis=1) == mock.truth.labels)
acc_after = np.mean(recal.argmax(axis=1) == mock.truth.labels)
print(f"accuracy with the classifier's probabilities: {acc_before:.3f}, recalibrated: {acc_after:.3f}")
```

```{code-cell} python
:tags: [remove-cell]

assert acc_after > acc_before
```

## Scores and a labelled set

Dividing out the training prevalence is exact only when the probabilities are
calibrated. A labelled set gives a route that needs no calibration
([inputs guide](../guides/inputs/binned.md)). Bin the
classifier's score, and tally how each class of the labelled set falls into the score
bins with {meth}`~binference.Response.from_labelled`. The tally is a
{term}`response matrix` $R_{jk}$: the probability of score bin $j$ for an object of class
$k$. The new set then enters as a {term}`binned measurement`.

Here the mock halves the classifier's log-likelihoods (`temperature=2`), so its
probabilities are too flat. The score is the object's one-dimensional feature:

```{code-cell} python
flat = binference.simulate.mock_categorical(
    3000, n_classes=4, separation=1.2, temperature=2.0, n_validation=2000, rng=5
)
lk_post = binference.Likelihoods.from_table(flat.posteriors, flat.bins, interim=flat.interim)
lk_score = binference.Likelihoods.from_binned(flat.response, values=flat.measurements["feature"])
fit_post = binference.infer(lk_post, rng=1)
fit_score = binference.infer(lk_score, rng=1)
print(flat.response)
print("truth                      ", flat.truth.rho.round(3))
print("posteriors and prevalence  ", fit_post.mean.round(3))
print("scores and a labelled set  ", fit_score.mean.round(3))
```

```{code-cell} python
:tags: [remove-cell]

err_post = np.abs(fit_post.mean - flat.truth.rho).max()
err_score = np.abs(fit_score.mean - flat.truth.rho).max()
assert err_score < err_post
```

The mock built `flat.response` with `Response.from_labelled(scores, labels)` from its
labelled set of 2000 objects. The scores route lands closer to the truth than the
posteriors route, whose flat probabilities bias it. `infer` also tests whether label shift
holds; the inputs guide describes the test.

## What to take away

- With calibrated probabilities, pass them with the training prevalence:
  `Likelihoods.from_table(probs, Bins.categories(classes), interim=train_prior)`.
- With a labelled set, tally the scores with `Response.from_labelled` and pass the new
  set's scores to `Likelihoods.from_binned`. This route does not need calibration.
- {doc}`mixed-stocks` uses the labelled-set route with a confusion matrix. The
  [quantification background page](../fields/quantification) places both routes in that
  field.
