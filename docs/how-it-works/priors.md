# Priors in detail

What exactly does each prior on the population say, and in which coordinates do the
engines see it? This page defines the two families of priors, the coordinates the
gradient-based engines use, and the priors on hyperparameters. The
{doc}`priors guide <../guides/priors>` says which prior to choose for which data.

## The Dirichlet family as trees

Every {term}`conjugate prior` on the fractions in binference is a {term}`Dirichlet tree`
{cite:p}`minka1999`. A tree is built over the bins, with the bins as
its leaves. Each internal node $q$ splits its share of the population among its children. The split at
each node has its own Dirichlet distribution, independent of the others. A bin's
fraction $\rho_k$ is the product of the shares along the path from the root to its leaf.

Write $\rho_q$ for the total fraction of the bins below node $q$. Write $\alpha_q$ for
the concentration of the branch into $q$, and $\bar\alpha_q$ for the sum of the concentrations of
$q$'s own branches. The log density on the simplex is then linear in the logarithms of
the leaf and node fractions:

$$
\log p(\rho)=\text{const}+\sum_k(\alpha_k-1)\log\rho_k+\sum_{q\ne\text{root}}(\alpha_q-\bar\alpha_q)\log\rho_q .
$$

Given every object's bin, the posterior of each split is again a Dirichlet: its
concentrations gain the counts below each branch. This keeps Gibbs sampling, EM and
mean-field VB in closed form for every member of the family. Bins under the same node
share that node's split, so they can vary together. The four priors differ only in the
shape of their tree.

- **`Dirichlet(alpha)`** is a tree with one node. The {term}`Dirichlet prior` with every
  {term}`concentration` $\alpha_k=1$ is uniform on the simplex and is the default. Its
  correlation between any two bins is $-1/(K-1)$ when the concentrations are equal,
  whatever their value. Explainer {ref}`explainer-v8` on the
  [concepts page](../concepts/priors-on-the-simplex.md#smoothness-priors) maps it beside
  the correlations of a Pólya tree, a random walk and a Gaussian process.
  - `Dirichlet.reference()` sets $\alpha_k=1/K$, the {term}`reference prior`
    {cite:p}`berger2015`. It adds one pseudo-object in all rather than $K$, which
    matters when there are many bins and few objects.
  - `Dirichlet.from_base_measure` makes $\alpha_k$ proportional to the bin masses of a
    base measure $G_0$. The default $G_0$ is flat in the measure coordinate, so
    $\alpha_k\propto v_k$ on linear bins. This prior stays consistent when bins are
    merged; equal concentrations on unequal bins do not.
  - `Dirichlet.from_counts` encodes a reference sample with counts $n^{\rm ref}_k$ as
    $\alpha_k=1+\kappa\,n^{\rm ref}_k$ (with the default base of one). A weight
    $\kappa<1$ down-weights a reference that may not represent the population.
- **`GeneralizedDirichlet`** {cite:p}`connor1969` is a chain-shaped tree, with one bin split
  off at each node. Each stick-breaking proportion $\nu_k$ has a Beta distribution with
  parameters `a` and `b`, and $\rho_k=\nu_k\prod_{j<k}(1-\nu_j)$. With `a` equal to
  $\alpha_k$ and `b` equal to $\sum_{j>k}\alpha_j$ it is $\mathrm{Dir}(\alpha)$. Other
  parameters give variances and correlations, including positive ones, that a Dirichlet
  cannot express.
- **`DirichletTree(tree, alpha)`** takes any tree, written as nested lists of bin
  indices, with one concentration per branch.
- **`PolyaTree`** {cite:p}`lavine1992` is a balanced binary tree over ordered
  one-dimensional bins, split in halves by index. A split at depth $h$ (the root has
  $h=1$) gives a child the concentration

  $$
  c_{\rm PT}\,h^{p_{\rm PT}}\cdot\frac{2G_0(\text{child})}{G_0(\text{parent})}.
  $$

  Its prior mean is the base measure $G_0$, and its fluctuations shrink at finer scales
  (explainer {ref}`explainer-v9`). With $p_{\rm PT}=2$ its limit as the bins shrink is a
  continuous density.

```{explainer} v9
Three draws of `PolyaTree(c)` with $p_{\rm PT}=2$ and a flat base measure on $[0, 1]$,
with one to five levels: 2 to 32 bins. The dashed line is the prior mean, $G_0$. Each
level splits every bin in two with Beta concentrations $c_{\rm PT}h^2$, so the deeper
levels add less variation. The readout compares the standard deviation of a bin's
density over 1,000 draws with its exact value: a bin's fraction is a product of
independent shares, with second moment $\prod_h(a_h+1)/(2(2a_h+1))$ for
$a_h=c_{\rm PT}h^2$.
```

Concentrations below one, and the Pólya tree, give densities that are unbounded at the
edge of the simplex. Their most probable population is then found in unconstrained
coordinates ([L-BFGS](point-estimates.md#l-bfgs)).

## The logistic family

A Dirichlet treats the bins as exchangeable. With broad likelihoods, the fractions of
neighbouring bins are then anticorrelated, and the posterior mean oscillates between
bins. A {term}`logistic prior` instead puts a
Gaussian prior on the log-fractions $\eta$, with $\rho=\mathrm{softmax}(\eta)$
{cite:p}`aitchison1980`. It is written in whitened form:

$$
\eta=\mu+E\,\big(s(\theta)\odot\xi\big),\qquad\xi\sim\mathcal N(0,I).
$$

Here $E$ is a fixed basis with one column per mode, and $s_j(\theta)$ is the prior
standard deviation of mode $j$. It depends on {term}`hyperparameters <hyperparameter>`
$\theta$, such as an amplitude and a length scale, which are inferred with $\eta$. The
coefficient of mode $j$ is $c_j=s_j\xi_j$.

The mean $\mu$ defaults to the logarithm of the bin masses of a density flat in the
measure coordinate. On linear bins that is $\mu_k=\log v_k$, up to a constant. Then
$\eta_k-\mu_k$ is the log-density up to a constant, so the smoothness penalty does not
depend on the binning. Passing the logarithm of reference fractions as `mean=` centres
the prior on a reference population instead. Adding a constant to $\eta$ leaves $\rho$
unchanged, so the constant mode is dropped wherever the basis contains it.

- **`LogisticNormal(mean, cov)`** takes a general covariance $\Sigma$. The basis holds
  the eigenvectors of its zero-sum part, since only that part affects $\rho$. Without a
  covariance the components are independent, with an inferred common scale.
- **`LogisticRandomWalk(order)`** makes the order-$r$ differences of $\eta$ between
  adjacent bins independent $\mathcal N(0,\tau^2)$ {cite:p}`rue2005`.
  - The basis holds the eigenvectors of the difference precision $\mathcal D_r^\top\mathcal D_r$, with
    eigenvalues $e_j$ and scales $s_j=\tau/\sqrt{e_j}$.
  - The polynomial modes the differences do not penalise, such as the linear trend for
    $r=2$, get a broad Gaussian. `trend_scale` is the prior standard deviation of each
    such mode's root-mean-square value over the grid, whatever the number of bins.
  - On grids of several dimensions the precision is the Kronecker sum of the per-axis
    precisions. Its unpenalised modes are the products of the per-axis ones, so for
    $r=2$ in two dimensions the surface $u_1u_2$ is not penalised either.
  - Smoothness is measured per bin step, so the natural size of $\tau$ depends on the
    binning.
- **`LogisticGP(kernel)`** is a Hilbert-space reduced-rank Gaussian process
  {cite:p}`solin2020,riutortmayol2023`.
  - Let an axis have half-range $\Delta$ and centre $\bar u$ in the measure coordinate.
    Along it, the basis is sines on a box $c_{\rm box}$ times wider than the grid.
    Column $j$ evaluated at the bin centre $u_k$ is

    $$
    E_{kj}=\frac{1}{\sqrt{c_{\rm box}\Delta}}\,\sin\frac{\pi j\,(u_k-\bar u+c_{\rm box}\Delta)}{2c_{\rm box}\Delta}.
    $$
  - $s_j^2$ is the kernel's spectral density at the frequency
    $\pi j/(2c_{\rm box}\Delta)$. Matérn-1/2, -3/2 and -5/2 and squared-exponential
    kernels are available.
  - The prior on $\eta$ is stationary in the measure coordinate, such as $\log z$ on logarithmic
    bins. Grids of several dimensions use products of the per-axis sines.
  - The amplitude and one length scale per axis are inferred. By default there are
    $n_{\rm basis}$ sines per axis, one per bin up to a total in the
    [defaults](../reference/defaults.md#priors).
  - $n_{\rm basis}$ sines resolve length scales down to $f_{\rm ker}\,c_{\rm box}\Delta/n_{\rm basis}$,
    with a resolution factor $f_{\rm ker}$ per kernel from the
    [defaults](../reference/defaults.md#priors). The default length-scale prior keeps
    the length scale between that bound and the range of the axis
    ({class}`~binference.priors.LogisticGP` gives its exact form).

Logistic priors are not conjugate, so NUTS, Laplace and L-BFGS fit them; Gibbs, EM and
mean-field VB do not. NUTS changes how $\eta$ is parametrised during warm-up, without
changing the prior ([partial centring](engines.md#partial-centring)).

## Coordinates

The gradient-based engines need coordinates in which every real vector is a valid
population, the {term}`unconstrained coordinates` $x$.

- **Simplex priors** use the isometric log-ratio {cite:p}`egozcue2003`. With $Q$ the
  orthonormal Helmert basis of the vectors that sum to zero, $\eta=Qx$ for
  $x\in\mathbb R^{K-1}$ and $\rho=\mathrm{softmax}(\eta)$. The density of $x$ gains the
  Jacobian

  $$
  \log\left|\frac{\partial\rho_{1:K-1}}{\partial x}\right|=\sum_k\log\rho_k+\tfrac12\log K .
  $$

  For a $\mathrm{Dir}(\alpha)$ prior the log density of $x$ is therefore
  $\sum_k\alpha_k\log\rho_k$, up to a constant.
- **Logistic priors** are unconstrained already: $x$ holds the whitened coefficients
  $\xi$, followed by the logarithms of the inferred hyperparameters.
- **Rates** use $\log\lambda$.
- **An inferred selection** appends $\mathrm{logit}\,S_k$ for every bin, and **an
  inferred response** the isometric log-ratios of each of its columns.

The gradient of the log-likelihood with respect to the log-fractions $\eta$ is
$\hat n-W\rho$, or $\hat n-W\rho\odot S/(S\cdot\rho)$ with a selection, or
$\hat n-S\odot\lambda$ for rates. It costs the same two matrix–vector products as the
likelihood itself. Because the coordinates add a Jacobian, the most probable point in
$x$ differs from the most probable population on the simplex
([two modes](point-estimates.md#two-coordinate-systems-two-modes)).

## Hyperpriors

Hyperparameters are positive and sampled on the logarithmic scale. Each takes a
{term}`hyperprior`: `HalfNormal`, `LogNormal`, `InverseGamma` or `Gamma`. A plain number
fixes the hyperparameter instead of inferring it. Each prior's default hyperpriors are in
its signature on the {doc}`API pages <../reference/api/index>`.
