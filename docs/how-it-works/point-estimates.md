---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Point estimates

How does binference find the single most probable population, and how does it know it
has found it? A {term}`point estimate` maximises the posterior {eq}`eq-P` of the
{doc}`model <model>`. Three solvers share the work: EM, mix-SQP and L-BFGS. Each
estimate carries a certificate of optimality. The answer also depends on the coordinates
the posterior is written in, as the last section shows.

## EM is iterated stacking

The {term}`EM` algorithm {cite:p}`dempster1977` for {eq}`eq-P` with a flat prior is

$$
\rho_k^{(t+1)}=\frac{\hat n_k(\rho^{(t)})}{W}
=\frac1W\sum_iw_i\,r_{ik}(\rho^{(t)}).
$$

Each step averages the objects' {term}`responsibilities`: it **stacks the per-object
posteriors computed under the current population**. Two consequences follow.

- **Stacking is one EM step.** {term}`Stacking` posteriors computed under an interim
  prior $\pi$ is one EM step started from $\pi$. It is biased whenever the likelihoods
  are broader than the features of the population, which is exactly when error bars
  matter.
- **The same iteration recurs across fields.** Iterative Bayesian unfolding,
  Richardson–Lucy deconvolution, EM quantification {cite:p}`saerens2002` and the
  Shepp–Vardi algorithm are all this update. Stopping it early is an implicit
  regulariser, not a prior.

Conjugate priors keep the update in closed form. A $\mathrm{Dir}(\alpha)$ prior adds
$\alpha_k-1$ pseudo-counts to $\hat n_k$ before normalising. With a selection, the
expected counts include the missed objects, $W\rho_k(1-S_k)/(S\cdot\rho)$
([selection effects](model.md#selection-effects)). For rates with a
$\mathrm{Gamma}(a_k,b_k)$ prior, the update is
$\lambda_k\leftarrow(\hat n_k+a_k-1)/(S_k+b_k)$. binference accelerates EM with
{term}`SQUAREM` {cite:p}`varadhan2008`.

### Missing information

EM is slow when the likelihoods are broad. Its speed is set by the
{term}`missing information`, `PopulationModel.missing_information`:

$$
f=1-\min\operatorname{eig}\left(I_{\rm obs},\,I_c\right).
$$

Here $I_{\rm obs}$ is the observed information in the isometric log-ratios of $\rho$,
and $I_c$ the information the labels would give if they were known. The smallest
generalised eigenvalue picks the direction about which the data say least, relative to
known labels. Near that direction each EM step removes only a share $1-f$ of the
remaining error {cite:p}`dempster1977`.

- $f$ is near 0 when the likelihoods are narrow compared with the bins, and near 1 when
  they are broad.
- With a selection, $I_c$ counts the imputed missed objects too: $W/(S\cdot\rho)$
  objects in all.
- $f$ is computed on the support of $\rho$. A fraction at zero is an active constraint,
  not missing information.
- With `include_prior=True`, a Dirichlet prior's pseudo-counts enter $I_c$ and
  $I_{\rm obs}$, as they do in EM.

The same number sets the speed of the Gibbs sampler
([Gibbs](engines.md#gibbs-sampling)) and the choices of
{doc}`auto <auto>`.

## mix-SQP

For Dirichlet concentrations $\alpha_k\ge1$, the log posterior is concave on the
simplex. binference then uses {term}`mix-SQP` {cite:p}`kim2020` instead of EM. Each
iteration fits a quadratic model of the objective and minimises it under $\rho\ge0$, as
a non-negative least-squares problem. It converges in a handful of iterations however
broad the likelihoods are, where EM can need thousands.

A selection makes the likelihood non-concave in $\rho$. It stays concave in the
{term}`selected population` $\phi_k=S_k\rho_k/(S\cdot\rho)$:

$$
p(d_i\mid\text{detected},\rho)=\frac{\tilde L_i\cdot\rho}{S\cdot\rho}=\sum_k\frac{\tilde L_{ik}}{S_k}\,\phi_k .
$$

The maximum-likelihood estimate does not depend on the coordinates. With the flat prior,
binference therefore solves for $\phi$ and maps back, $\rho_k\propto\phi_k/S_k$. Other
priors do not keep their mode under this change of variables, so they use EM.

Rates with a `GammaRates` prior of shapes $a_k\ge1$ reduce to the same problem. Write
$\lambda_k=\Lambda\chi_k/(S_k+b_k)$ with $\chi$ on the simplex. The total
$\Lambda=W+\sum_k(a_k-1)$ then profiles out. What remains is mix-SQP in $\chi$, with
$S_k+b_k$ in place of $S_k$ and a Dirichlet term $\sum_k(a_k-1)\log\chi_k$.

## L-BFGS

The other cases maximise the density in {term}`unconstrained coordinates` with
{term}`L-BFGS`:

- priors whose density is unbounded at the edge of the simplex, such as concentrations
  below one or a Pólya tree, where the mode on the simplex can sit at an exact zero;
- priors that are not conjugate, such as the logistic family;
- a selection or response inferred with the population.

{class}`~binference.samplers.MAP` states the exact routing. `est.info["algorithm"]` and
`est.info["space"]` record the choice.

## The certificate

At the maximum on the simplex the Karush–Kuhn–Tucker conditions read

$$
g_k=\frac1W\sum_i\frac{w_iL_{ik}}{L_i\cdot\hat\rho}\le1,\quad\text{with equality where }\hat\rho_k>0 .
$$

With a prior or a selection, the gradient of their terms is added to $g_k$. binference
reports the optimality gap $\max_kg_k-1$, the {term}`KKT gap`, for every estimate from
EM or mix-SQP (`PointEstimate.kkt_gap`). It is infinite where an object has zero
likelihood. A fixed point of EM is necessary for optimality but not sufficient; the gap
checks the rest.

For rates the conditions are $\partial\log p/\partial\lambda_k\le0$, with equality
where $\hat\lambda_k>0$. The gap is the larger of
$\max_k S_k^{-1}\,\partial\log p/\partial\lambda_k$ and
$|\lambda\cdot\nabla_\lambda\log p|/(S\cdot\lambda)$, so it does not depend on the units
of the exposure.

`MAP(tol=)` bounds the gap for EM, and the same gap of the problem mix-SQP solves. For
L-BFGS it bounds the largest absolute gradient divided by $\max(1,W)$, kept in
`est.info["grad_norm"]`, and `est.kkt_gap` is NaN. A run that can no longer improve its
objective in floating point counts as converged within a looser bound
([defaults](../reference/defaults.md#engines)).

## Two coordinate systems, two modes

The same posterior has a density in each coordinate system. The two densities differ by
the Jacobian of the map between the coordinates ([coordinates](priors.md#coordinates)),
so they have different modes:

| density | coordinates | Jacobian | mode | used by |
|---|---|---|---|---|
| `PopulationModel.log_posterior(value)` | fractions $\rho$ on the simplex (rates $\lambda$) | no | `MAP(space="simplex")` | EM, mix-SQP, and the `log_posterior` of every point estimate |
| `PopulationModel.log_density_unconstrained(x)` | unconstrained $x$ | yes | `MAP(space="unconstrained")` | NUTS, Laplace, L-BFGS |

`log_posterior` is NaN for priors without a closed-form density on the simplex, the
logistic family. Take a $\mathrm{Dir}(\alpha)$ prior and objects with known labels,
$n_k$ of them in bin $k$. The log density on the simplex is then
$\sum_k(\alpha_k+n_k-1)\log\rho_k$, and the unconstrained log density
$\sum_k(\alpha_k+n_k)\log\rho_k$, up to constants. The two modes are proportional to
$\alpha_k+n_k-1$ and to $\alpha_k+n_k$.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  image:
    alt: >-
      Two stacked panels. The top panel shows a skewed density of the fraction rho 1
      between 0 and 0.8, with its peak at 0.125 marked as the simplex mode and a second
      line at 0.2 marked as the unconstrained mode. The bottom panel shows the density
      of the isometric log-ratio x, which is nearly symmetric, with its peak at the
      unconstrained mode and the simplex mode to its left. A shaded band marks the same
      central 68 per cent interval in both panels.
  figure:
    caption: >-
      One posterior in two coordinates. Two bins, one object certainly in bin 1, seven
      certainly in bin 2, and a flat prior: the fraction of bin 1 has a
      Beta(2, 8) posterior (top). Its isometric log-ratio $x$ has the density below.
      Each panel's peak is its own mode; the shaded central 68% interval holds the same
      draws in both.
    name: fig-two-modes
---
import matplotlib.pyplot as plt
import numpy as np

import binference
from binference.samplers import MAP

plt.rcParams["font.size"] = 12
INK2 = "#5D5F86"

lk = binference.Likelihoods(
    np.array([[1.0, 0.0], [0.0, 1.0]]), binference.Bins.categories(2), weights=[1, 7]
)
model = binference.PopulationModel(lk)
modes = {
    "simplex mode": MAP(model, space="simplex").run().rho[0],
    "unconstrained mode": MAP(model, space="unconstrained").run().rho[0],
}

r = np.linspace(1e-5, 1 - 1e-5, 2001)
dr = np.exp([model.log_posterior(np.array([a, 1 - a])) for a in r])
dr /= np.trapezoid(dr, r)
x = np.array([model.from_value(np.array([a, 1 - a]))[0] for a in r])
dx = np.exp([model.log_density_unconstrained(np.array([v])) for v in x])
dx /= np.trapezoid(dx, x)
cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dr[1:] + dr[:-1]) * np.diff(r))])
band = (cdf > 0.16) & (cdf < 0.84)

fig, axes = plt.subplots(2, 1, figsize=(4.8, 6.0), constrained_layout=True)
for ax, t, d, label in (
    (axes[0], r, dr, r"fraction $\rho_1$"),
    (axes[1], x, dx, r"isometric log-ratio $x$"),
):
    ax.fill_between(t[band], d[band], color="C0", alpha=0.2, lw=0)
    ax.plot(t, d, color="C0")
    for (name, value), ls, ha in zip(modes.items(), ("-", ":"), ("right", "left")):
        at = value if ax is axes[0] else model.from_value(np.array([value, 1 - value]))[0]
        ax.axvline(at, color=INK2, ls=ls, lw=1.4)
        pad = 0.01 if ax is axes[0] else 0.06
        ax.text(at + (-pad if ha == "right" else pad), d.max() * 1.3, name.replace(" ", "\n"),
                ha=ha, va="top", color=INK2)
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.set_ylim(0, d.max() * 1.32)
axes[0].set_xlim(0, 0.7)
axes[1].set_xlim(-3.2, 0.8)
plt.show()
```

Neither mode is "the" answer. The posterior mean and quantiles of any quantity are the
same in every coordinate system, because they are properties of the draws. The mode on
the simplex can sit on the boundary, with a fraction exactly zero. The unconstrained
mode is always interior, which is why the Laplace approximation is centred there
([Laplace](engines.md#laplace-with-importance-sampling)).
