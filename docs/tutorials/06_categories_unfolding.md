---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Categories and unfolding

The same model covers problems far from astronomy. This tutorial uses it for
two of them:

- **quantification**: estimating class proportions from a classifier's
  outputs when the class balance differs from the training set;
- **unfolding**: recovering a spectrum from detector counts with a response
  matrix.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

import binference

plt.rcParams["figure.dpi"] = 110
```

## Quantification under prior shift

A classifier trained on balanced classes is applied to a test set whose
class proportions are skewed. Its posteriors $p(k\mid x_i)$ include the
*training* prevalence as a prior. {func}`binference.simulate.mock_categorical`
simulates an ideal classifier on four overlapping classes:

```{code-cell} python
mock = binference.simulate.mock_categorical(3000, n_classes=4, separation=1.2, rng=5)
probs, train_prior = mock.posteriors, mock.interim
print("true test prevalence:", mock.truth.rho.round(3), " training prior:", train_prior)
```

The common shortcuts are "classify and count" (take each object's most
probable class) and "probabilistic classify and count" (average the
probabilities). Both are biased toward the training prior. binference divides the
training prior out and infers the prevalence with uncertainty. This is the
Bayesian version of the EM method of Saerens et al. (2002):

```{code-cell} python
lk = binference.Likelihoods.from_classifier(probs, train_prior, labels=mock.bins.labels)
fit = binference.infer(lk, rng=1)

cc = np.bincount(probs.argmax(axis=1), minlength=4) / len(probs)
pcc = probs.mean(axis=0)
lo, hi = fit.interval(0.95)
for k, name in enumerate(mock.bins.labels):
    print(f"{name}: truth {mock.truth.rho[k]:.3f} | CC {cc[k]:.3f} | PCC {pcc[k]:.3f} | "
          f"binference {fit.mean[k]:.3f} [{lo[k]:.3f}, {hi[k]:.3f}]")
```

```{code-cell} python
ax = fit.plot(truth=mock.truth.rho, stack=pcc)
ax.set_title("class prevalence", loc="left");
```

The per-object posteriors under the *test* population,
`fit.assignment_probabilities()`, are the recalibrated classifier outputs.
They are what to use to classify objects in the shifted test set.

```{code-cell} python
recal = fit.assignment_probabilities()
acc_before = np.mean(probs.argmax(axis=1) == mock.truth.labels)
acc_after = np.mean(recal.argmax(axis=1) == mock.truth.labels)
print(f"accuracy with training-prior posteriors: {acc_before:.3f}, after recalibration: {acc_after:.3f}")
```

## Unfolding a spectrum

In unfolding, events with a true value in bin $k$ are observed in
reconstructed bin $b$ with probability $R_{bk}$, the response matrix. Some
events are lost entirely: the efficiency $\epsilon_k=\sum_bR_{bk}$ is less
than one. {func}`binference.simulate.mock_unfolding` smears an exponential
spectrum:

```{code-cell} python
mu = binference.simulate.mock_unfolding(8000, n_true=20, n_reco=25, rng=3)
R = mu.truth.extra["response"]
counts = mu.measurements["reco_counts"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
im = axes[0].imshow(R, origin="lower", aspect="auto", cmap="Blues")
axes[0].set(xlabel="true bin k", ylabel="reconstructed bin b", title="response matrix")
plt.colorbar(im, ax=axes[0])
axes[1].bar(np.arange(len(counts)), counts, color="0.6")
axes[1].set(xlabel="reconstructed bin", ylabel="events", title="observed counts")
axes[1].spines[["top", "right"]].set_visible(False)
```

{meth}`~binference.Likelihoods.from_response` turns counts and the response into
weighted likelihood rows, one per reconstructed bin. It attaches the
efficiency as a selection function with `selection_mode="observed"`, since
the response already includes it. Inverting $R$ directly would amplify the
noise wildly. A smoothness prior regularises the unfolding the Bayesian way:
it infers how smooth the spectrum is, rather than fixing an iteration count
or a regularisation strength by hand.

```{code-cell} python
lk_u = binference.Likelihoods.from_response(counts, R, bins=mu.bins)
print(lk_u, lk_u.selection)
flat = binference.infer(lk_u, rng=1, n_samples=500, n_warmup=500)
smooth = binference.infer(lk_u, binference.priors.LogisticGP(), rng=1, check=False)
d = smooth.diagnostics
print(f"divergent: {d['n_divergent']} of {smooth.n_draws}, max R-hat {d['rhat'].max():.3f}, "
      f"min bulk ESS {d['ess_bulk'].min():.0f}")
```

`check=False` stops `infer` from warning so that we can look at the
diagnostics ourselves. One transition out of 4000 diverged, which `infer`
would have flagged. The R-hat and effective sample sizes are fine, and an
isolated divergence like this is usually harmless. Many divergences, or
divergences that persist with a higher `target_accept`, would mean the
sampler cannot explore part of the posterior ({doc}`/user-guide/pitfalls`).

```{code-cell} python
truth = mu.truth.rho
naive, *_ = np.linalg.lstsq(R, counts, rcond=None)    # unregularised inversion
naive = naive / naive.sum()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, constrained_layout=True)
flat.plot(ax=axes[0], truth=truth, which="fraction")
smooth.plot(ax=axes[1], truth=truth, which="fraction")
axes[0].plot(mu.bins.centers, naive, "x", color="0.4", label="matrix inversion")
axes[0].legend(frameon=False, fontsize=8)
axes[0].set_title("Dirichlet prior", loc="left")
axes[1].set_title("Gaussian-process prior", loc="left")
axes[1].get_legend().remove()
for ax in axes:
    ax.set_xlabel("true value")
```

The posterior bands show the uncertainty that unfolding usually hides. The
smoothness prior learns its own amplitude and length scale from the data:

```{code-cell} python
for name, draws in smooth.info["hyperparameters"].items():
    print(f"{name}: {np.quantile(draws, [0.16, 0.5, 0.84]).round(2)}")
```

```{tip}
An order-2 random walk (`LogisticRandomWalk(order=2)`) is another natural
choice, but this spectrum is an exact exponential. Its log is a straight line,
which the order-2 random walk does not penalise at all. The posterior then
pushes the random-walk scale toward zero, a funnel that makes NUTS report
divergences. A GP prior, `target_accept=0.95`, or a scale hyperprior bounded
away from zero avoids it.
```
