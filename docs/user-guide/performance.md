# Performance

## What costs time

Every engine is built on one operation: a pass over the likelihood matrix
$L$ to compute $L\rho$, and one back to compute $L^\top(w/s)$. A dense
$N\times K$ matrix costs $O(NK)$ per pass, and a sparse one $O(\text{nnz})$.
Some reference timings, on one CPU core with $N=20\,000$ and $K=80$
(`python benchmarks/bench_kernels.py`):

| operation | time |
|---|---|
| log-likelihood and gradient (dense) | ≈ 1.2 ms |
| the same, sparse (56% of entries kept) | ≈ 1.6 ms |
| one Gibbs sweep (a label for every object) | ≈ 9 ms |
| all per-object responsibilities | ≈ 10 ms |

The engines spend these differently:

- **NUTS** needs about 10–60 gradient evaluations per draw. With the
  defaults (4 chains of 1000 warm-up and 1000 sampling iterations) a
  2000-object, 30-bin problem takes about 30 s on one core with a Dirichlet
  prior, and 20–25 s with a smoothness prior, whose re-parametrised
  geometry needs fewer steps per draw. Time scales roughly with $N\times K$.
- **Gibbs** costs one sweep per draw. Each sweep is cheap, but successive
  draws are correlated when likelihoods are broad, so compare *effective*
  samples per second.
- **MAP** with a Dirichlet prior (the default) or `GammaRates` with shapes of at least one uses mix-SQP, which converges
  in a handful of second-order iterations: about a second for $N=10^5$
  objects and $K=100$ bins, a few seconds when likelihoods are broad. EM, used for the other conjugate priors, needs one pass per
  iteration and thousands of iterations when likelihoods are broad compared
  with the bins.
- **Laplace** costs a MAP fit plus a finite-difference Hessian ($2K$ gradient
  evaluations), and **VB** costs about as much as EM.

## Making it faster

1. **Run chains in parallel.** `binference.infer(lk, n_jobs=-1)` uses one process
   per chain and gives each process an equal share of the cores for its
   linear algebra. The random streams are the same as in a serial run, so the
   draws agree up to rounding. On four cores, the 144-bin 2-D fit of the
   {doc}`last tutorial </tutorials/08_two_dimensions>` (3000 objects, a
   Gaussian-process prior, 4 × 1000 iterations) takes about 30 s. In scripts
   on macOS and Windows, put the call under `if __name__ == "__main__":`.
2. **Use sparse likelihoods.** Narrow PDFs on many bins are mostly zeros.
   `lk.to_sparse(rtol=1e-8)` drops entries below `rtol` times each row's
   maximum. `from_samples` returns sparse matrices automatically when fewer
   than 30% of the entries are non-zero.
3. **Compress duplicates.** `lk.compress()` merges identical rows into one
   weighted row. Binned photometry, discrete measurements and unfolding data
   often shrink by orders of magnitude, and inference is unchanged.
4. **Choose bins deliberately.** Cost grows with $K$, but bins much wider than
   the population's features cause binning bias ({doc}`pitfalls`). A
   smoothness prior lets you use fine bins without noisy results.
5. **Draw what you need.** If the effective sample sizes are well above what
   you need, shorten the run: `n_samples=500, n_warmup=500` often suffices for
   well-behaved problems. Always check the diagnostics afterwards.
6. **Scale up gracefully.** For very large problems, `method="map"` gives the
   estimate in seconds, `method="laplace"` adds approximate error bars, and
   `method="auto"` switches from NUTS to Laplace above $2\times10^7$ non-zero
   likelihood values.
7. **Threads.** NumPy's BLAS may use several threads for the matrix–vector
   products. For many small fits, set `OMP_NUM_THREADS=1` and run the fits in
   parallel instead.

## Memory

A dense float64 likelihood matrix takes $8NK$ bytes: 160 MB for
$N=10^6$ and $K=20$. Sparse storage keeps 12 bytes per non-zero entry. The
per-object methods (`assignment_probabilities`, `predict_proba`) work in
chunks, so their memory use does not grow with $N$.

## Roadmap

The NumPy implementation keeps installation trivial. An optional JAX backend
(vectorised chains, GPUs) is on the roadmap for problems with $10^8$
likelihood entries and beyond; see the
[implementation plan](../design/implementation-plan.md).
