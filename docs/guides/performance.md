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
  evaluations), then, for its importance correction, the log density at each
  proposal ($2\times$ the draws by default), 256 proposals per pass over $L$:
  about 5 s for 8000 proposals with $N=10^5$ and $K=50$ on one core. **VB**
  costs about as much as EM.

## Making it faster

1. **Run chains in parallel.** `binference.infer(lk, n_jobs=-1)` uses one process
   per chain and gives each process an equal share of the cores for its
   linear algebra. The random streams are the same as in a serial run, so the
   draws agree up to rounding. On four cores of a shared container, the
   144-bin 2-D fit of the {doc}`two-dimensional tutorial
   </tutorials/two-dimensions>` took 31 s. It has 3000 objects, a
   Gaussian-process prior with 10 sines per axis, and four chains of 300
   warm-up and 300 sampling iterations. In
   scripts on macOS and Windows, put the call under
   `if __name__ == "__main__":`.
2. **Use sparse likelihoods.** Narrow PDFs on many bins are mostly zeros.
   `lk.to_sparse(rtol=1e-8)` drops entries below `rtol` times each row's
   maximum. On four cores with banded rows ($N=2\times10^5$, $K=100$ and
   $N=5\times10^4$, $K=1000$), sparse storage beat dense up to 20% and 10%
   non-zero entries with BLAS threads, and up to 40% and 30% on one thread; at
   5% it was 3–3.5× faster with threads and 9–12× on one. `from_samples`
   returns sparse matrices automatically below 10%.
3. **Compress duplicates.** `lk.compress()` merges identical rows into one
   weighted row. Binned photometry, discrete measurements and unfolding data
   often shrink by orders of magnitude, and inference is unchanged.
4. **Choose bins deliberately.** Cost grows with $K$, but bins much wider than
   the population's features cause binning bias ({doc}`pitfalls`). A
   smoothness prior lets you use fine bins without noisy results.
5. **Draw what you need.** If the effective sample sizes are well above what
   you need, shorten a NUTS run: `method="nuts", n_samples=500, n_warmup=500`
   often suffices for well-behaved problems. Under `method="auto"`, leave the
   run length to auto (see {doc}`engines`). Always check the diagnostics
   afterwards.
6. **Scale up gracefully.** For very large problems, `method="map"` gives the
   estimate in seconds, and `method="laplace"` adds approximate error bars. The
   default, `method="auto"`, already uses Gibbs or Laplace where they are
   accurate and faster than NUTS ([how it chooses](engines.md)).
7. **Threads.** NumPy's BLAS may use several threads for the matrix–vector
   products. For many small fits, set `OMP_NUM_THREADS=1` and run the fits in
   parallel instead.

## Memory

A dense float64 likelihood matrix takes $8NK$ bytes: 160 MB for
$N=10^6$ and $K=20$. Sparse storage keeps 12 bytes per non-zero entry. The
per-object methods (`assignment_probabilities`, `predict_proba`,
`log_predictive` and so `PopulationEstimator.score`) work through the objects
`chunk` at a time (4096 by default), so beyond their output they hold only one
chunk's rows. The exception is a response inferred with the population, whose
rows differ per draw: those methods hold one $N\times K$ block per draw used.

## Roadmap

On four cores with BLAS threads, a fused kernel compiled with numba (one pass
over the rows, in parallel) was slower than this NumPy path for $N\le2\times10^5$
and 16% faster at $N=10^6$, $K=50$; float32 storage changed the time by less
than 10% and let the expected counts drift by up to $5\times10^{-4}$ (relative)
at $N=10^6$. Neither is offered (DECISIONS #77). The NumPy implementation keeps installation trivial. An optional JAX backend
(vectorised chains, GPUs) is on the roadmap for problems with $10^8$
likelihood entries and beyond; see the
[implementation plan](../design/implementation-plan.md).
