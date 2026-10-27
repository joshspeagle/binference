# Scale up

This page says what makes a fit slow or large, and how to make it faster and smaller.

## What costs time

Every engine is built on one operation. It passes over the likelihood matrix $L$ to
compute $L\rho$, and back to compute $L^\top\big(w/(L\rho)\big)$, element by element. A
dense $N\times K$ matrix costs $O(NK)$ per pass. A sparse one costs O(number of non-zero
entries).

These timings come from `python benchmarks/bench_kernels.py --engines`, run on
2026-10-27 on one thread of an Intel Xeon at 2.1 GHz:

| operation | $N$ | $K$ | time |
|---|---|---|---|
| log-likelihood and its gradient, dense | 20,000 | 80 | 1.3 ms |
| the same, sparse | 20,000 | 80 | 1.7 ms |
| one Gibbs sweep (a bin for every object) | 20,000 | 80 | 10 ms |
| all per-object responsibilities | 20,000 | 80 | 8 ms |
| a NUTS fit with the defaults, Dirichlet prior | 2000 | 30 | 36 s |
| the same with `LogisticGP` | 2000 | 30 | 31 s |
| a Gibbs fit with the defaults | 2000 | 30 | 2.8 s |
| a MAP fit | 2000 | 30 | 0.01 s |
| a Laplace fit | 2000 | 30 | 0.4 s |

The engines spend these passes differently:

- **NUTS** needs tens of gradient evaluations per draw. Its time grows roughly with
  $N\times K$.
- **Gibbs** costs one sweep per draw. Each sweep is cheap. When likelihoods are broad,
  successive draws are correlated, so compare *effective* draws per second.
- **MAP** with mix-SQP converges in tens of second-order iterations. EM, used for
  the other conjugate priors, needs one pass per iteration. It can need thousands of
  iterations when likelihoods are broad compared with the bins.
- **Laplace** costs a MAP fit and a Hessian from $2K$ gradient evaluations. Its
  importance correction then evaluates the log density at every proposal. **VB** costs
  about as much as EM.

## Making it faster

1. **Run chains in parallel.** `binference.infer(lk, n_jobs=-1)` uses one process per
   chain. Each process gets an equal share of the cores for its linear algebra. The
   random streams are the same as in a serial run, so the draws agree up to rounding.
   In scripts on macOS and Windows, put the call under `if __name__ == "__main__":`.
2. **Use sparse likelihoods.** Narrow likelihoods on many bins are mostly zeros.
   `lk.to_sparse(rtol=1e-8)` drops entries below `rtol` times each row's largest
   value. Sparse storage pays when only a small share of entries is non-zero.
   `from_samples` chooses sparse storage itself below its limit
   ({doc}`defaults <../reference/defaults>`).
3. **Compress duplicates.** `lk.compress()` merges identical rows into one weighted
   row. Binned photometry, discrete measurements and unfolding data often shrink by
   orders of magnitude, and inference is unchanged.
4. **Choose bins deliberately.** Cost grows with $K$, but bins much wider than the
   population's features cause {term}`binning bias` ({doc}`bins`). A smoothness prior
   lets you use fine bins without noisy results.
5. **Draw what you need.** If the effective sample sizes are well above what you need,
   shorten a NUTS run, for example with `n_samples=500, n_warmup=500`. Under
   `method="auto"`, leave the run length to auto ({doc}`engines`). Always check the
   diagnostics afterwards.
6. **Threads.** NumPy's BLAS may use several threads for the matrix–vector products.
   For many small fits, set `OMP_NUM_THREADS=1` and run the fits in parallel instead.

## Memory

A dense float64 likelihood matrix takes $8NK$ bytes: 160 MB for $N=10^6$ and $K=20$.
Sparse storage keeps 12 bytes per non-zero entry. The per-object methods
(`assignment_probabilities`, `predict_proba`, `log_predictive`) work through the objects
`chunk` at a time, so beyond their output they hold only one chunk's rows. A response
inferred with the population is the exception: its rows differ per draw, so those
methods hold one $N\times K$ block per draw used.

The per-object methods (`assignment_probabilities`, `predict_proba`, `log_predictive`
and so `PopulationEstimator.score`) work through the objects `chunk` at a time. Beyond
their output, they hold only one chunk's rows
({doc}`defaults <../reference/defaults>`). The exception is a response inferred with
the population, whose rows differ per draw. Those methods hold one $N\times K$ block per
draw used.
