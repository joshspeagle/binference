# Priors on the simplex

*What does a prior on the population fractions say, and when does it change the answer?*

## Every population is a point on the simplex

The {term}`population fractions` $\rho_1,\dots,\rho_K$ are non-negative and sum to one.
The set of all such vectors is the {term}`simplex` $\Delta^{K-1}$. With three bins it is
a triangle: each corner puts the whole population in one bin, and the centre splits it
equally. A prior on the population is a distribution over this triangle, or over its
higher-dimensional version.

## The Dirichlet prior and pseudo-counts

The default prior is the {term}`Dirichlet prior` $\mathrm{Dir}(\alpha)$, with density
proportional to $\prod_k\rho_k^{\alpha_k-1}$. Its parameters $\alpha_k$ are the
{term}`concentrations <concentration>`.

The concentrations act like objects seen before the data, or pseudo-counts. Suppose
you knew the bin of every object, with $n_k$ objects in bin $k$. The posterior would
then be $\mathrm{Dir}(\alpha+n)$, so $\alpha_k$ adds to the count of bin $k$. A prior
with this property is {term}`conjugate <conjugate prior>`. Gibbs and mean-field
VB need it, and so does EM, one of MAP's solvers.

The size of $\alpha$ sets how strongly the prior moves the estimate
(explainer {ref}`explainer-v7`):

- With every $\alpha_k=1$, the default, every population is equally probable. The
  prior is flat on the simplex.
- With large $\alpha_k$, draws crowd around the prior mean, $\alpha_k/\sum_j\alpha_j$.
- With $\alpha_k<1$, draws crowd into the corners and edges: most bins are nearly
  empty.

```{explainer} v7
1,500 draws from the symmetric Dirichlet prior $\mathrm{Dir}(\alpha,\alpha,\alpha)$ on
three bins, with $\alpha$ from 0.1 to 32. Each point is one population; each corner puts
the whole population in one bin. The readout compares the standard deviation of one
fraction over the draws with the Dirichlet's, $\sqrt{2/(9(3\alpha+1))}$.
```

The flat prior is not free. It adds one pseudo-count per bin, $K$ in all. With $K$
comparable to the number of objects, that moves the estimate toward equal fractions.
The {doc}`priors guide <../guides/priors>` gives the alternatives: the
{term}`reference prior`, concentrations from a base density or a reference sample, and
three conjugate relatives of the Dirichlet.

## Smoothness priors

A Dirichlet treats bins as unordered: shuffling them leaves the prior unchanged. With
broad likelihoods, mass can shift between neighbouring bins with little change in the
likelihood, and the posterior mean can be jagged.

A {term}`smoothness prior <logistic prior>` adds what the Dirichlet leaves out. It puts
a Gaussian distribution on the log-fractions $\eta$, with $\rho=\mathrm{softmax}(\eta)$,
and makes neighbouring values of $\eta$ similar:

- a random walk penalises jumps, or bends, between neighbouring bins;
- a Gaussian process correlates bins closer than a length scale.

Explainer {ref}`explainer-v8` compares the four kinds of prior on 16 bins. A Dirichlet's
bins are all slightly anticorrelated, whatever $\alpha$
([the Dirichlet family](../how-it-works/priors.md#the-dirichlet-family-as-trees) gives
the number). A
{term}`Pólya tree`, one of the Dirichlet's conjugate relatives,
correlates the bins within each half, quarter and so on of the range. The random walk and
the Gaussian process correlate neighbours, and anticorrelate distant bins, since the
fractions sum to one.

```{explainer} v8
Top: three draws from the chosen prior on 16 bins over $[0, 1]$. Bottom: the correlation
between every two bins' fractions over 2,000 draws. The priors are `Dirichlet(alpha)`,
`PolyaTree(c)` (four levels), `LogisticRandomWalk(scale=tau)` (order 2) and
`LogisticGP(amplitude=1, length_scale=ell, n_basis=64)` (Matérn-3/2), with the
hyperparameter fixed rather than inferred. The scale slider multiplies it by $10^s$,
from $\alpha=1$, $c=1$, $\tau=0.1$ and $\ell=0.6$. The random walk's linear trend keeps its
broad default prior, so at small $\tau$ its draws are ramps.
```

How smooth the population is, is not fixed in advance. binference infers the prior's
{term}`hyperparameters <hyperparameter>`, such as the size of the jumps or the length
scale, with the population. By default the prior is centred on a density
that is flat in the {term}`measure` coordinate. Its notion of smooth then does not
depend on how you bin. Smoothness priors are not conjugate, so Gibbs and VB cannot use
them (see {doc}`engines`).

## When the prior matters

The prior matters where the data are weak. A bin's {term}`data support` is the number
of objects' worth of likelihood it holds. Where it is low, the prior sets the answer.
With many objects in every bin, any of these priors gives nearly the same population.

To choose between priors that the data cannot tell apart, compare them on held-out
objects, with the {term}`log predictive density`. For counts per unit time or area
instead of fractions, a prior on the {term}`rates` takes the place of the Dirichlet
(see {doc}`selection`).

```{plain-words}
A prior says which populations you find plausible before you look at the data. The
Dirichlet's numbers behave like objects you pretend to have already seen; a smoothness
prior says that neighbouring bins should look alike.
```

## Where next

- Do it: the {doc}`priors guide <../guides/priors>`.
- See it: the tutorial {doc}`../tutorials/priors-and-smoothness`.
- Derivation: [the Dirichlet family](../how-it-works/priors.md#the-dirichlet-family-as-trees) in How it works.
