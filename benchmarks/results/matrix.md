binference 0.1.0.dev0, numpy 2.5.4, Python 3.13.16, x86_64, 4 worker(s) of one thread, started 2026-10-26 01:33:05.

| scenario | prior | f | engine | s | ESS | ESS/s | mean err | sd ratio | q err | k̂ | warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | dirichlet | 1.00 | reference | 84.9 | 8129 |  |  |  |  |  |  |
| baseline | dirichlet | 1.00 | nuts | 22.1 | 1538 | 70 | 0.03 | 0.98–1.04 | 0.08 |  |  |
| baseline | dirichlet | 1.00 | gibbs | 1.2 | 10 | 9 | 0.23 | 0.90–1.12 | 0.29 |  |  |
| baseline | dirichlet | 1.00 | laplace | 0.4 | 65 | 154 | 0.23 | 0.80–1.16 | 1.12 | 0.80 | ApproximationWarning |
| baseline | dirichlet | 1.00 | vb | 0.1 | 4000 | 76595 | 0.37 | 0.14–0.98 | 1.94 |  | ApproximationWarning |
| baseline | dirichlet | 1.00 | map | 0.0 |  |  | 1.74 |  |  |  |  |
| baseline | dirichlet | 1.00 | bootstrap | 0.5 | 200 | 398 | 1.14 | 0.00–1.36 | 3.05 |  | ApproximationWarning |
| baseline | randomwalk | 1.00 | reference | 44.4 | 18270 |  |  |  |  |  |  |
| baseline | randomwalk | 1.00 | nuts | 17.1 | 3474 | 203 | 0.03 | 0.97–1.06 | 0.14 |  |  |
| baseline | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| baseline | randomwalk | 1.00 | laplace | 0.7 | 26 | 38 | 1.18 | 1.21–2.01 | 3.63 | 1.09 | ApproximationWarning |
| baseline | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| baseline | randomwalk | 1.00 | map | 0.0 |  |  | 1.46 |  |  |  |  |
| baseline | randomwalk | 1.00 | bootstrap | 5.3 | 200 | 38 | 1.81 | 0.34–1.62 | 3.08 |  | ApproximationWarning |
| baseline | dirichlet0.1 | 1.00 | reference | 706.9 | 2807 |  |  |  |  |  |  |
| baseline | dirichlet0.1 | 1.00 | nuts | 221.5 | 354 | 2 | 0.08 | 0.93–1.04 | 0.19 |  |  |
| baseline | dirichlet0.1 | 1.00 | gibbs | 1.3 | 6 | 5 | 0.20 | 0.78–1.12 | 0.54 |  |  |
| baseline | dirichlet0.1 | 1.00 | laplace | 0.5 | 2 | 3 | 8.11 | 0.16–3.47 | 8.23 | 1.85 | ApproximationWarning |
| baseline | dirichlet0.1 | 1.00 | vb | 0.0 | 4000 | 129041 | 1.85 | 0.00–0.82 | 3.09 |  | ApproximationWarning |
| baseline | dirichlet0.1 | 1.00 | map | 0.0 |  |  | 0.61 |  |  |  |  |
| baseline | dirichlet0.1 | 1.00 | bootstrap | 1.9 | 200 | 104 | 0.52 | 0.03–0.77 | 1.88 |  | ApproximationWarning |
| baseline | gp | 1.00 | reference | 66.2 | 17194 |  |  |  |  |  |  |
| baseline | gp | 1.00 | nuts | 21.6 | 3775 | 175 | 0.01 | 0.99–1.04 | 0.08 |  |  |
| baseline | gp | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticG |
| baseline | gp | 1.00 | laplace | 1.0 | 21 | 22 | 0.23 | 0.84–1.27 | 0.86 | 0.97 | ApproximationWarning |
| baseline | gp | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| baseline | gp | 1.00 | map | 0.0 |  |  | 0.52 |  |  |  |  |
| baseline | gp | 1.00 | bootstrap | 2.9 | 200 | 69 | 0.57 | 0.29–1.18 | 1.85 |  | ApproximationWarning |
| baseline | polyatree | 1.00 | reference | 30.0 | 14544 |  |  |  |  |  |  |
| baseline | polyatree | 1.00 | nuts | 8.7 | 2938 | 339 | 0.03 | 0.98–1.03 | 0.09 |  |  |
| baseline | polyatree | 1.00 | gibbs | 3.9 | 261 | 67 | 0.12 | 0.95–1.05 | 0.19 |  |  |
| baseline | polyatree | 1.00 | laplace | 0.6 | 3305 | 5997 | 0.04 | 0.97–1.03 | 0.12 | 0.16 |  |
| baseline | polyatree | 1.00 | vb | 0.0 | 4000 | 215650 | 0.07 | 0.42–0.97 | 1.04 |  | ApproximationWarning |
| baseline | polyatree | 1.00 | map | 0.0 |  |  | 0.04 |  |  |  |  |
| baseline | polyatree | 1.00 | bootstrap | 0.7 | 200 | 269 | 0.09 | 0.31–0.65 | 1.38 |  | ApproximationWarning |
| width-0.3 | dirichlet | 0.73 | reference | 36.2 | 4823 |  |  |  |  |  |  |
| width-0.3 | dirichlet | 0.73 | nuts | 11.5 | 1195 | 103 | 0.04 | 0.97–1.06 | 0.09 |  |  |
| width-0.3 | dirichlet | 0.73 | gibbs | 1.1 | 833 | 791 | 0.06 | 0.97–1.04 | 0.10 |  |  |
| width-0.3 | dirichlet | 0.73 | laplace | 0.6 | 311 | 545 | 0.09 | 0.94–1.05 | 0.23 | 0.70 |  |
| width-0.3 | dirichlet | 0.73 | vb | 0.0 | 4000 | 749614 | 0.04 | 0.67–1.02 | 0.63 |  | ApproximationWarning |
| width-0.3 | dirichlet | 0.73 | map | 0.0 |  |  | 1.12 |  |  |  |  |
| width-0.3 | dirichlet | 0.73 | bootstrap | 1.3 | 200 | 158 | 1.12 | 0.00–1.13 | 3.11 |  | ApproximationWarning |
| width-0.3 | randomwalk | 0.88 | reference | 44.2 | 22790 |  |  |  |  |  |  |
| width-0.3 | randomwalk | 0.88 | nuts | 18.1 | 4150 | 230 | 0.05 | 0.97–1.03 | 0.13 |  |  |
| width-0.3 | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| width-0.3 | randomwalk | 0.88 | laplace | 0.8 | 17 | 21 | 2.43 | 0.72–3.78 | 10.62 | 1.13 | ApproximationWarning |
| width-0.3 | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| width-0.3 | randomwalk | 0.88 | map | 0.0 |  |  | 1.75 |  |  |  |  |
| width-0.3 | randomwalk | 0.88 | bootstrap | 8.9 | 200 | 22 | 1.79 | 0.19–1.71 | 2.41 |  | ApproximationWarning |
| width-0.3 | dirichlet0.1 | 0.83 | reference | 218.1 | 763 |  |  |  |  |  |  |
| width-0.3 | dirichlet0.1 | 0.83 | nuts | 62.5 | 113 | 2 | 0.03 | 0.96–1.03 | 0.17 |  |  |
| width-0.3 | dirichlet0.1 | 0.83 | gibbs | 1.1 | 148 | 137 | 0.05 | 0.98–1.02 | 0.13 |  |  |
| width-0.3 | dirichlet0.1 | 0.83 | laplace | 0.5 | 52 | 96 | 0.47 | 0.73–1.28 | 0.98 | 1.02 | ApproximationWarning |
| width-0.3 | dirichlet0.1 | 0.83 | vb | 0.0 | 4000 | 634142 | 0.52 | 0.31–1.05 | 2.04 |  | ApproximationWarning |
| width-0.3 | dirichlet0.1 | 0.83 | map | 0.0 |  |  | 0.60 |  |  |  |  |
| width-0.3 | dirichlet0.1 | 0.83 | bootstrap | 1.2 | 200 | 163 | 0.63 | 0.00–1.10 | 1.55 |  | ApproximationWarning |
| width-0.3 | gp | 0.77 | reference | 78.9 | 16267 |  |  |  |  |  |  |
| width-0.3 | gp | 0.77 | nuts | 24.3 | 3569 | 147 | 0.03 | 0.97–1.06 | 0.13 |  |  |
| width-0.3 | gp | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticG |
| width-0.3 | gp | 0.77 | laplace | 1.3 | 26 | 20 | 0.76 | 0.86–1.30 | 1.03 | 1.05 | ApproximationWarning |
| width-0.3 | gp | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| width-0.3 | gp | 0.77 | map | 0.0 |  |  | 1.00 |  |  |  |  |
| width-0.3 | gp | 0.77 | bootstrap | 4.0 | 200 | 50 | 1.05 | 0.29–1.28 | 2.14 |  | ApproximationWarning |
| width-0.3 | polyatree | 0.91 | reference | 25.4 | 12256 |  |  |  |  |  |  |
| width-0.3 | polyatree | 0.91 | nuts | 7.6 | 2545 | 336 | 0.04 | 0.97–1.03 | 0.12 |  |  |
| width-0.3 | polyatree | 0.91 | gibbs | 3.5 | 1323 | 383 | 0.05 | 0.97–1.02 | 0.12 |  |  |
| width-0.3 | polyatree | 0.91 | laplace | 0.6 | 4041 | 6607 | 0.03 | 0.97–1.02 | 0.09 | 0.22 |  |
| width-0.3 | polyatree | 0.91 | vb | 0.0 | 4000 | 233854 | 0.03 | 0.78–0.99 | 0.40 |  | ApproximationWarning |
| width-0.3 | polyatree | 0.91 | map | 0.0 |  |  | 0.02 |  |  |  |  |
| width-0.3 | polyatree | 0.91 | bootstrap | 0.7 | 200 | 288 | 0.12 | 0.32–0.95 | 1.36 |  | ApproximationWarning |
| width-3 | dirichlet | 1.00 | reference | 79.4 | 11640 |  |  |  |  |  |  |
| width-3 | dirichlet | 1.00 | nuts | 21.2 | 2255 | 106 | 0.03 | 0.98–1.04 | 0.10 |  |  |
| width-3 | dirichlet | 1.00 | gibbs | 1.0 | 12 | 12 | 0.31 | 0.84–1.19 | 0.69 |  |  |
| width-3 | dirichlet | 1.00 | laplace | 0.4 | 18 | 45 | 0.31 | 0.76–1.14 | 0.87 | 1.16 | ApproximationWarning |
| width-3 | dirichlet | 1.00 | vb | 0.1 | 4000 | 42516 | 0.45 | 0.12–0.64 | 2.02 |  | ApproximationWarning |
| width-3 | dirichlet | 1.00 | map | 0.0 |  |  | 4.68 |  |  |  |  |
| width-3 | dirichlet | 1.00 | bootstrap | 0.4 | 200 | 481 | 2.09 | 0.58–2.38 | 3.24 |  | ApproximationWarning |
| width-3 | randomwalk | 1.00 | reference | 96.6 | 6326 |  |  |  |  |  |  |
| width-3 | randomwalk | 1.00 | nuts | 31.5 | 657 | 21 | 0.03 | 0.96–1.04 | 0.14 |  |  |
| width-3 | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| width-3 | randomwalk | 1.00 | laplace | 0.7 | 28 | 41 | 1.29 | 0.67–1.93 | 3.23 | 0.93 | ApproximationWarning |
| width-3 | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| width-3 | randomwalk | 1.00 | map | 0.0 |  |  | 1.49 |  |  |  |  |
| width-3 | randomwalk | 1.00 | bootstrap | 2.7 | 200 | 75 | 1.31 | 0.46–1.73 | 3.14 |  | ApproximationWarning |
| width-3 | dirichlet0.1 | 1.00 | reference | 909.0 | 7745 |  |  |  |  |  |  |
| width-3 | dirichlet0.1 | 1.00 | nuts | 253.9 | 1257 | 5 | 0.03 | 0.96–1.03 | 0.16 |  |  |
| width-3 | dirichlet0.1 | 1.00 | gibbs | 1.1 | 6 | 5 | 0.78 | 0.24–1.24 | 2.20 |  |  |
| width-3 | dirichlet0.1 | 1.00 | laplace | 0.4 | 1 | 3 | 1.08 | 0.05–1.06 | 3.00 | 2.34 | ApproximationWarning |
| width-3 | dirichlet0.1 | 1.00 | vb | 0.1 | 4000 | 28017 | 2.57 | 0.00–0.40 | 3.02 |  | ApproximationWarning |
| width-3 | dirichlet0.1 | 1.00 | map | 0.0 |  |  | 0.49 |  |  |  |  |
| width-3 | dirichlet0.1 | 1.00 | bootstrap | 1.3 | 200 | 151 | 0.80 | 0.20–1.05 | 2.07 |  | ApproximationWarning |
| width-3 | gp | 1.00 | reference | 70.1 | 6639 |  |  |  |  |  |  |
| width-3 | gp | 1.00 | nuts | 21.6 | 880 | 41 | 0.04 | 0.98–1.04 | 0.11 |  |  |
| width-3 | gp | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticG |
| width-3 | gp | 1.00 | laplace | 1.1 | 24 | 23 | 0.18 | 0.82–1.19 | 0.74 | 0.95 | ApproximationWarning |
| width-3 | gp | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| width-3 | gp | 1.00 | map | 0.0 |  |  | 0.77 |  |  |  |  |
| width-3 | gp | 1.00 | bootstrap | 2.5 | 200 | 81 | 0.67 | 0.46–0.96 | 1.57 |  | ApproximationWarning |
| width-3 | polyatree | 1.00 | reference | 31.2 | 11310 |  |  |  |  |  |  |
| width-3 | polyatree | 1.00 | nuts | 8.5 | 2262 | 267 | 0.04 | 0.98–1.04 | 0.12 |  |  |
| width-3 | polyatree | 1.00 | gibbs | 3.6 | 76 | 21 | 0.10 | 0.85–1.05 | 0.40 |  |  |
| width-3 | polyatree | 1.00 | laplace | 0.6 | 2508 | 4492 | 0.04 | 0.96–1.04 | 0.14 | 0.40 |  |
| width-3 | polyatree | 1.00 | vb | 0.0 | 4000 | 89809 | 0.18 | 0.25–0.75 | 1.48 |  | ApproximationWarning |
| width-3 | polyatree | 1.00 | map | 0.0 |  |  | 0.06 |  |  |  |  |
| width-3 | polyatree | 1.00 | bootstrap | 0.6 | 200 | 338 | 0.10 | 0.27–0.57 | 1.30 |  | ApproximationWarning |
| n-100 | dirichlet | 0.96 | reference | 29.2 | 10712 |  |  |  |  |  |  |
| n-100 | dirichlet | 0.96 | nuts | 9.5 | 1632 | 171 | 0.05 | 0.95–1.03 | 0.21 |  |  |
| n-100 | dirichlet | 0.96 | gibbs | 0.4 | 223 | 600 | 0.09 | 0.95–1.04 | 0.16 |  |  |
| n-100 | dirichlet | 0.96 | laplace | 0.3 | 54 | 170 | 0.25 | 0.86–1.26 | 0.48 | 0.84 | ApproximationWarning |
| n-100 | dirichlet | 0.96 | vb | 0.0 | 4000 | 437130 | 0.22 | 0.38–1.00 | 1.26 |  | ApproximationWarning |
| n-100 | dirichlet | 0.96 | map | 0.0 |  |  | 2.19 |  |  |  |  |
| n-100 | dirichlet | 0.96 | bootstrap | 0.2 | 200 | 1009 | 1.13 | 0.00–1.78 | 3.10 |  | ApproximationWarning |
| n-100 | randomwalk | 1.00 | reference | 44.8 | 17838 |  |  |  |  |  |  |
| n-100 | randomwalk | 1.00 | nuts | 11.7 | 2948 | 252 | 0.04 | 0.94–1.04 | 0.14 |  |  |
| n-100 | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| n-100 | randomwalk | 1.00 | laplace | 0.7 | 80 | 120 | 0.84 | 0.74–1.57 | 1.98 | 0.83 | ApproximationWarning |
| n-100 | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| n-100 | randomwalk | 1.00 | map | 0.0 |  |  | 1.31 |  |  |  |  |
| n-100 | randomwalk | 1.00 | bootstrap | 2.7 | 200 | 74 | 1.27 | 0.06–1.83 | 3.04 |  | ApproximationWarning |
| n-10000 | dirichlet | 1.00 | reference | 639.6 | 6235 |  |  |  |  |  |  |
| n-10000 | dirichlet | 1.00 | nuts | 187.6 | 1210 | 6 | 0.06 | 0.97–1.05 | 0.18 |  |  |
| n-10000 | dirichlet | 1.00 | gibbs | 8.1 | 5 | 1 | 0.45 | 0.54–1.07 | 1.20 |  |  |
| n-10000 | dirichlet | 1.00 | laplace | 1.6 | 7 | 5 | 1.01 | 0.42–1.21 | 1.01 | 1.12 | ApproximationWarning |
| n-10000 | dirichlet | 1.00 | vb | 1.4 | 4000 | 2817 | 0.26 | 0.06–0.86 | 2.03 |  | ApproximationWarning, ConvergenceWarning |
| n-10000 | dirichlet | 1.00 | map | 0.0 |  |  | 1.54 |  |  |  |  |
| n-10000 | dirichlet | 1.00 | bootstrap | 2.2 | 200 | 90 | 1.03 | 0.00–1.33 | 3.05 |  | ApproximationWarning |
| n-10000 | randomwalk | 1.00 | reference | 117.9 | 19243 |  |  |  |  |  |  |
| n-10000 | randomwalk | 1.00 | nuts | 45.1 | 2661 | 59 | 0.03 | 0.98–1.06 | 0.08 |  |  |
| n-10000 | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| n-10000 | randomwalk | 1.00 | laplace | 2.1 | 7 | 3 | 1.49 | 0.82–2.26 | 2.42 | 1.23 | ApproximationWarning |
| n-10000 | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| n-10000 | randomwalk | 1.00 | map | 0.1 |  |  | 1.86 |  |  |  |  |
| n-10000 | randomwalk | 1.00 | bootstrap | 14.6 | 200 | 14 | 1.97 | 0.09–1.57 | 2.59 |  | ApproximationWarning |
| k-10 | dirichlet | 1.00 | reference | 57.4 | 6454 |  |  |  |  |  |  |
| k-10 | dirichlet | 1.00 | nuts | 18.0 | 1349 | 75 | 0.04 | 0.99–1.02 | 0.09 |  |  |
| k-10 | dirichlet | 1.00 | gibbs | 0.8 | 12 | 15 | 0.23 | 0.89–1.02 | 0.35 |  |  |
| k-10 | dirichlet | 1.00 | laplace | 0.5 | 254 | 525 | 0.18 | 0.87–1.11 | 0.92 | 0.68 |  |
| k-10 | dirichlet | 1.00 | vb | 0.1 | 4000 | 39836 | 0.21 | 0.13–0.69 | 1.60 |  | ApproximationWarning |
| k-10 | dirichlet | 1.00 | map | 0.0 |  |  | 1.20 |  |  |  |  |
| k-10 | dirichlet | 1.00 | bootstrap | 0.3 | 200 | 609 | 1.05 | 0.00–1.36 | 3.07 |  | ApproximationWarning |
| k-10 | randomwalk | 1.00 | reference | 38.4 | 11804 |  |  |  |  |  |  |
| k-10 | randomwalk | 1.00 | nuts | 14.5 | 2331 | 161 | 0.03 | 0.98–1.01 | 0.06 |  |  |
| k-10 | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| k-10 | randomwalk | 1.00 | laplace | 0.7 | 153 | 235 | 0.16 | 0.96–1.24 | 0.73 | 0.73 | ApproximationWarning |
| k-10 | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| k-10 | randomwalk | 1.00 | map | 0.0 |  |  | 0.35 |  |  |  |  |
| k-10 | randomwalk | 1.00 | bootstrap | 2.8 | 200 | 72 | 0.31 | 0.21–0.85 | 1.79 |  | ApproximationWarning |
| k-50 | dirichlet | 1.00 | reference | 130.2 | 10117 |  |  |  |  |  |  |
| k-50 | dirichlet | 1.00 | nuts | 39.7 | 1834 | 46 | 0.03 | 0.97–1.04 | 0.13 |  |  |
| k-50 | dirichlet | 1.00 | gibbs | 2.4 | 23 | 10 | 0.28 | 0.93–1.07 | 0.30 |  |  |
| k-50 | dirichlet | 1.00 | laplace | 0.8 | 2 | 3 | 2.90 | 0.43–2.25 | 2.50 | 1.79 | ApproximationWarning |
| k-50 | dirichlet | 1.00 | vb | 0.1 | 4000 | 41390 | 0.30 | 0.22–1.03 | 1.73 |  | ApproximationWarning |
| k-50 | dirichlet | 1.00 | map | 0.0 |  |  | 2.06 |  |  |  |  |
| k-50 | dirichlet | 1.00 | bootstrap | 3.0 | 200 | 66 | 1.10 | 0.00–1.54 | 3.01 |  | ApproximationWarning |
| k-50 | randomwalk | 1.00 | reference | 84.5 | 24344 |  |  |  |  |  |  |
| k-50 | randomwalk | 1.00 | nuts | 31.1 | 4492 | 145 | 0.02 | 0.98–1.08 | 0.15 |  |  |
| k-50 | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| k-50 | randomwalk | 1.00 | laplace | 1.4 | 6 | 4 | 3.21 | 0.57–5.76 | 9.40 | 1.70 | ApproximationWarning |
| k-50 | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| k-50 | randomwalk | 1.00 | map | 0.2 |  |  | 2.65 |  |  |  |  |
| k-50 | randomwalk | 1.00 | bootstrap | 45.9 | 200 | 4 | 3.31 | 0.01–3.53 | 8.35 |  | ApproximationWarning |
| pop-sparse | dirichlet | 0.99 | reference | 82.4 | 7728 |  |  |  |  |  |  |
| pop-sparse | dirichlet | 0.99 | nuts | 24.1 | 1546 | 64 | 0.04 | 0.96–1.03 | 0.16 |  |  |
| pop-sparse | dirichlet | 0.99 | gibbs | 1.0 | 20 | 19 | 0.16 | 0.94–1.06 | 0.23 |  |  |
| pop-sparse | dirichlet | 0.99 | laplace | 0.5 | 22 | 46 | 0.61 | 0.72–1.18 | 0.97 | 1.07 | ApproximationWarning |
| pop-sparse | dirichlet | 0.99 | vb | 0.0 | 4000 | 87225 | 0.39 | 0.16–1.02 | 1.82 |  | ApproximationWarning |
| pop-sparse | dirichlet | 0.99 | map | 0.0 |  |  | 1.81 |  |  |  |  |
| pop-sparse | dirichlet | 0.99 | bootstrap | 0.3 | 200 | 610 | 1.10 | 0.00–1.40 | 3.16 |  | ApproximationWarning |
| pop-sparse | randomwalk | 1.00 | reference | 60.7 | 16312 |  |  |  |  |  |  |
| pop-sparse | randomwalk | 1.00 | nuts | 22.4 | 2752 | 123 | 0.03 | 0.69–1.05 | 0.22 |  |  |
| pop-sparse | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| pop-sparse | randomwalk | 1.00 | laplace | 0.7 | 7 | 10 | 0.86 | 0.50–1.76 | 1.55 | 1.16 | ApproximationWarning |
| pop-sparse | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| pop-sparse | randomwalk | 1.00 | map | 0.0 |  |  | 0.60 |  |  |  |  |
| pop-sparse | randomwalk | 1.00 | bootstrap | 5.3 | 200 | 38 | 0.60 | 0.01–0.98 | 2.25 |  | ApproximationWarning |
| pop-spiky | dirichlet | 1.00 | reference | 62.6 | 9616 |  |  |  |  |  |  |
| pop-spiky | dirichlet | 1.00 | nuts | 18.8 | 1703 | 91 | 0.04 | 0.97–1.05 | 0.13 |  |  |
| pop-spiky | dirichlet | 1.00 | gibbs | 1.1 | 23 | 21 | 0.20 | 0.90–1.10 | 0.26 |  |  |
| pop-spiky | dirichlet | 1.00 | laplace | 0.5 | 16 | 29 | 0.46 | 0.71–1.20 | 1.24 | 0.97 | ApproximationWarning |
| pop-spiky | dirichlet | 1.00 | vb | 0.1 | 4000 | 72271 | 0.44 | 0.18–0.99 | 1.85 |  | ApproximationWarning |
| pop-spiky | dirichlet | 1.00 | map | 0.0 |  |  | 2.13 |  |  |  |  |
| pop-spiky | dirichlet | 1.00 | bootstrap | 0.6 | 200 | 363 | 1.08 | 0.00–1.48 | 3.03 |  | ApproximationWarning |
| pop-spiky | randomwalk | 1.00 | reference | 52.2 | 15962 |  |  |  |  |  |  |
| pop-spiky | randomwalk | 1.00 | nuts | 18.0 | 3183 | 177 | 0.03 | 0.97–1.03 | 0.12 |  |  |
| pop-spiky | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| pop-spiky | randomwalk | 1.00 | laplace | 0.6 | 34 | 56 | 0.64 | 0.88–1.45 | 1.74 | 1.07 | ApproximationWarning |
| pop-spiky | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| pop-spiky | randomwalk | 1.00 | map | 0.0 |  |  | 0.58 |  |  |  |  |
| pop-spiky | randomwalk | 1.00 | bootstrap | 4.7 | 200 | 42 | 0.52 | 0.03–1.14 | 1.59 |  | ApproximationWarning |
| sel-truth | dirichlet | 1.00 | reference | 83.9 | 7909 |  |  |  |  |  |  |
| sel-truth | dirichlet | 1.00 | nuts | 25.2 | 1432 | 57 | 0.06 | 0.98–1.02 | 0.13 |  |  |
| sel-truth | dirichlet | 1.00 | gibbs | 1.0 | 12 | 12 | 0.17 | 0.91–1.03 | 0.31 |  |  |
| sel-truth | dirichlet | 1.00 | laplace | 0.4 | 22 | 50 | 0.68 | 0.71–1.28 | 0.88 | 0.91 | ApproximationWarning |
| sel-truth | dirichlet | 1.00 | vb | 0.1 | 4000 | 63831 | 0.24 | 0.15–0.61 | 1.90 |  | ApproximationWarning |
| sel-truth | dirichlet | 1.00 | map | 0.0 |  |  | 1.98 |  |  |  |  |
| sel-truth | dirichlet | 1.00 | bootstrap | 0.4 | 200 | 462 | 1.27 | 0.00–1.34 | 3.00 |  | ApproximationWarning |
| sel-truth | randomwalk | 1.00 | reference | 40.8 | 18332 |  |  |  |  |  |  |
| sel-truth | randomwalk | 1.00 | nuts | 16.0 | 3209 | 200 | 0.04 | 0.93–1.03 | 0.14 |  |  |
| sel-truth | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| sel-truth | randomwalk | 1.00 | laplace | 0.8 | 12 | 16 | 1.33 | 0.81–1.95 | 2.66 | 1.08 | ApproximationWarning |
| sel-truth | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| sel-truth | randomwalk | 1.00 | map | 0.0 |  |  | 1.89 |  |  |  |  |
| sel-truth | randomwalk | 1.00 | bootstrap | 4.3 | 200 | 46 | 2.43 | 0.14–1.76 | 3.95 |  | ApproximationWarning |
| sel-measured | dirichlet | 1.00 | reference | 81.7 | 8318 |  |  |  |  |  |  |
| sel-measured | dirichlet | 1.00 | nuts | 25.6 | 1429 | 56 | 0.04 | 0.97–1.02 | 0.11 |  |  |
| sel-measured | dirichlet | 1.00 | gibbs | 1.2 | 12 | 10 | 0.36 | 0.81–1.19 | 0.69 |  |  |
| sel-measured | dirichlet | 1.00 | laplace | 0.5 | 59 | 122 | 0.27 | 0.74–1.37 | 2.03 | 0.86 | ApproximationWarning |
| sel-measured | dirichlet | 1.00 | vb | 0.1 | 4000 | 68280 | 0.33 | 0.12–0.55 | 1.95 |  | ApproximationWarning |
| sel-measured | dirichlet | 1.00 | map | 0.0 |  |  | 2.57 |  |  |  |  |
| sel-measured | dirichlet | 1.00 | bootstrap | 0.2 | 200 | 830 | 2.73 | 1.05–8.09 | 17.98 |  | ApproximationWarning |
| sel-measured | randomwalk | 1.00 | reference | 45.5 | 15399 |  |  |  |  |  |  |
| sel-measured | randomwalk | 1.00 | nuts | 16.0 | 2976 | 186 | 0.04 | 0.98–1.04 | 0.13 |  |  |
| sel-measured | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| sel-measured | randomwalk | 1.00 | laplace | 0.6 | 9 | 14 | 1.34 | 0.75–2.24 | 6.90 | 1.08 | ApproximationWarning |
| sel-measured | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| sel-measured | randomwalk | 1.00 | map | 0.0 |  |  | 1.73 |  |  |  |  |
| sel-measured | randomwalk | 1.00 | bootstrap | 4.1 | 200 | 49 | 1.59 | 0.99–1.93 | 2.37 |  | ApproximationWarning |
| interim-correct | dirichlet | 1.00 | reference | 76.4 | 7895 |  |  |  |  |  |  |
| interim-correct | dirichlet | 1.00 | nuts | 24.6 | 1664 | 68 | 0.06 | 0.96–1.04 | 0.14 |  |  |
| interim-correct | dirichlet | 1.00 | gibbs | 1.0 | 17 | 16 | 0.34 | 0.86–1.03 | 0.40 |  |  |
| interim-correct | dirichlet | 1.00 | laplace | 0.4 | 9 | 24 | 1.16 | 0.69–1.82 | 1.43 | 1.13 | ApproximationWarning |
| interim-correct | dirichlet | 1.00 | vb | 0.1 | 4000 | 64898 | 0.37 | 0.16–0.93 | 1.89 |  | ApproximationWarning |
| interim-correct | dirichlet | 1.00 | map | 0.0 |  |  | 1.66 |  |  |  |  |
| interim-correct | dirichlet | 1.00 | bootstrap | 0.6 | 200 | 315 | 1.13 | 0.00–1.44 | 3.17 |  | ApproximationWarning |
| interim-wrong | dirichlet | 1.00 | reference | 77.4 | 7656 |  |  |  |  |  |  |
| interim-wrong | dirichlet | 1.00 | nuts | 26.4 | 1774 | 67 | 0.06 | 0.99–1.03 | 0.15 |  |  |
| interim-wrong | dirichlet | 1.00 | gibbs | 1.1 | 25 | 23 | 0.30 | 0.85–1.03 | 0.35 |  |  |
| interim-wrong | dirichlet | 1.00 | laplace | 0.4 | 15 | 36 | 0.57 | 0.74–1.11 | 0.83 | 1.17 | ApproximationWarning |
| interim-wrong | dirichlet | 1.00 | vb | 0.0 | 4000 | 101677 | 0.27 | 0.14–0.95 | 1.82 |  | ApproximationWarning |
| interim-wrong | dirichlet | 1.00 | map | 0.0 |  |  | 1.72 |  |  |  |  |
| interim-wrong | dirichlet | 1.00 | bootstrap | 0.6 | 200 | 322 | 1.03 | 0.00–1.48 | 3.05 |  | ApproximationWarning |
| samples-20 | dirichlet | 0.98 | reference | 40.6 | 7197 |  |  |  |  |  |  |
| samples-20 | dirichlet | 0.98 | nuts | 14.5 | 1424 | 98 | 0.02 | 0.98–1.02 | 0.10 |  |  |
| samples-20 | dirichlet | 0.98 | gibbs | 1.2 | 126 | 103 | 0.10 | 0.90–1.03 | 0.27 |  |  |
| samples-20 | dirichlet | 0.98 | laplace | 0.4 | 103 | 270 | 0.16 | 0.87–1.19 | 0.61 | 0.94 | ApproximationWarning |
| samples-20 | dirichlet | 0.98 | vb | 0.0 | 4000 | 287990 | 0.27 | 0.26–0.96 | 1.47 |  | ApproximationWarning |
| samples-20 | dirichlet | 0.98 | map | 0.0 |  |  | 1.23 |  |  |  |  |
| samples-20 | dirichlet | 0.98 | bootstrap | 0.5 | 200 | 367 | 1.06 | 0.00–1.13 | 3.14 |  | ApproximationWarning |
| samples-200 | dirichlet | 1.00 | reference | 74.2 | 8290 |  |  |  |  |  |  |
| samples-200 | dirichlet | 1.00 | nuts | 19.4 | 1996 | 103 | 0.03 | 0.97–1.04 | 0.10 |  |  |
| samples-200 | dirichlet | 1.00 | gibbs | 1.6 | 20 | 13 | 0.23 | 0.89–1.10 | 0.38 |  |  |
| samples-200 | dirichlet | 1.00 | laplace | 0.4 | 41 | 90 | 0.27 | 0.75–1.30 | 1.60 | 0.93 | ApproximationWarning |
| samples-200 | dirichlet | 1.00 | vb | 0.0 | 4000 | 139693 | 0.45 | 0.17–0.99 | 2.09 |  | ApproximationWarning |
| samples-200 | dirichlet | 1.00 | map | 0.0 |  |  | 1.33 |  |  |  |  |
| samples-200 | dirichlet | 1.00 | bootstrap | 0.4 | 200 | 558 | 1.15 | 0.00–1.27 | 3.07 |  | ApproximationWarning |
| grid-8x8 | dirichlet | 1.00 | reference | 76.4 | 11485 |  |  |  |  |  |  |
| grid-8x8 | dirichlet | 1.00 | nuts | 21.5 | 2447 | 114 | 0.03 | 0.94–1.05 | 0.14 |  |  |
| grid-8x8 | dirichlet | 1.00 | gibbs | 2.9 | 17 | 6 | 0.20 | 0.83–1.12 | 0.52 |  |  |
| grid-8x8 | dirichlet | 1.00 | laplace | 0.4 | 10 | 23 | 0.93 | 0.47–2.81 | 5.60 | 1.91 | ApproximationWarning |
| grid-8x8 | dirichlet | 1.00 | vb | 0.1 | 4000 | 72623 | 0.41 | 0.17–0.99 | 2.04 |  | ApproximationWarning |
| grid-8x8 | dirichlet | 1.00 | map | 0.0 |  |  | 3.01 |  |  |  |  |
| grid-8x8 | dirichlet | 1.00 | bootstrap | 1.0 | 200 | 194 | 1.32 | 0.00–1.91 | 3.12 |  | ApproximationWarning |
| grid-8x8 | randomwalk | 1.00 | reference | 161.1 | 13907 |  |  |  |  |  |  |
| grid-8x8 | randomwalk | 1.00 | nuts | 73.8 | 2652 | 36 | 0.04 | 0.79–1.19 | 0.19 |  |  |
| grid-8x8 | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| grid-8x8 | randomwalk | 1.00 | laplace | 1.6 | 9 | 5 | 1.61 | 0.13–3.39 | 6.28 | 1.54 | ApproximationWarning |
| grid-8x8 | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| grid-8x8 | randomwalk | 1.00 | map | 0.0 |  |  | 0.31 |  |  |  |  |
| grid-8x8 | randomwalk | 1.00 | bootstrap | 12.1 | 200 | 16 | 0.65 | 0.02–2.17 | 4.00 |  | ApproximationWarning |
| categorical | dirichlet | 1.00 | reference | 32.0 | 3209 |  |  |  |  |  |  |
| categorical | dirichlet | 1.00 | nuts | 9.5 | 753 | 80 | 0.06 | 0.97–1.01 | 0.21 |  |  |
| categorical | dirichlet | 1.00 | gibbs | 0.8 | 18 | 24 | 0.12 | 1.02–1.07 | 0.21 |  |  |
| categorical | dirichlet | 1.00 | laplace | 0.3 | 434 | 1277 | 0.08 | 0.89–1.01 | 0.24 | 0.82 | ApproximationWarning |
| categorical | dirichlet | 1.00 | vb | 0.0 | 4000 | 133528 | 0.24 | 0.16–0.44 | 1.85 |  | ApproximationWarning |
| categorical | dirichlet | 1.00 | map | 0.0 |  |  | 0.79 |  |  |  |  |
| categorical | dirichlet | 1.00 | bootstrap | 0.1 | 200 | 1429 | 0.39 | 1.03–1.13 | 0.32 |  | ApproximationWarning |
| rail-like | dirichlet | 1.00 | reference | 388.5 | 6259 |  |  |  |  |  |  |
| rail-like | dirichlet | 1.00 | nuts | 103.6 | 1257 | 12 | 0.06 | 0.97–1.05 | 0.13 |  |  |
| rail-like | dirichlet | 1.00 | gibbs | 6.8 | 7 | 1 | 0.47 | 0.74–1.09 | 0.67 |  |  |
| rail-like | dirichlet | 1.00 | laplace | 1.1 | 7 | 6 | 0.63 | 0.52–1.11 | 1.62 | 1.62 | ApproximationWarning |
| rail-like | dirichlet | 1.00 | vb | 0.6 | 4000 | 6455 | 0.51 | 0.09–0.92 | 2.15 |  | ApproximationWarning |
| rail-like | dirichlet | 1.00 | map | 0.0 |  |  | 2.57 |  |  |  |  |
| rail-like | dirichlet | 1.00 | bootstrap | 1.3 | 200 | 153 | 1.82 | 0.00–1.45 | 3.05 |  | ApproximationWarning |
| rail-like | randomwalk | 1.00 | reference | 99.1 | 18785 |  |  |  |  |  |  |
| rail-like | randomwalk | 1.00 | nuts | 55.7 | 3627 | 65 | 0.03 | 0.97–1.08 | 0.12 |  |  |
| rail-like | randomwalk | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticR |
| rail-like | randomwalk | 1.00 | laplace | 2.8 | 13 | 5 | 2.39 | 0.61–2.79 | 4.56 | 1.42 | ApproximationWarning |
| rail-like | randomwalk | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| rail-like | randomwalk | 1.00 | map | 0.1 |  |  | 2.02 |  |  |  |  |
| rail-like | randomwalk | 1.00 | bootstrap | 24.3 | 200 | 8 | 3.00 | 0.01–2.41 | 5.60 |  | ApproximationWarning |
| rail-like | dirichlet0.1 | 1.00 | reference | 2916.8 | 1490 |  |  |  |  |  |  |
| rail-like | dirichlet0.1 | 1.00 | nuts | 858.9 | 77 | 0 | 0.06 | 0.89–1.13 | 0.96 |  |  |
| rail-like | dirichlet0.1 | 1.00 | gibbs | 6.9 | 5 | 1 | 0.94 | 0.48–1.75 | 3.05 |  |  |
| rail-like | dirichlet0.1 | 1.00 | laplace | 1.2 | 10 | 8 | 2.38 | 0.38–7.28 | 21.06 | 5.60 | ApproximationWarning |
| rail-like | dirichlet0.1 | 1.00 | vb | 0.4 | 4000 | 10242 | 1.29 | 0.00–0.83 | 3.28 |  | ApproximationWarning |
| rail-like | dirichlet0.1 | 1.00 | map | 0.0 |  |  | 1.05 |  |  |  |  |
| rail-like | dirichlet0.1 | 1.00 | bootstrap | 4.7 | 200 | 43 | 1.17 | 0.01–0.86 | 1.59 |  | ApproximationWarning |
| rail-like | gp | 1.00 | reference | 162.2 | 16254 |  |  |  |  |  |  |
| rail-like | gp | 1.00 | nuts | 51.5 | 3278 | 64 | 0.03 | 0.96–1.03 | 0.10 |  |  |
| rail-like | gp | | gibbs | | | | | | | | TypeError: Gibbs sampling needs a conjugate prior; LogisticG |
| rail-like | gp | 1.00 | laplace | 1.8 | 22 | 12 | 0.62 | 0.72–1.30 | 1.41 | 0.89 | ApproximationWarning |
| rail-like | gp | | vb | | | | | | | | TypeError: Mean-field VB needs a conjugate prior; use NUTS o |
| rail-like | gp | 1.00 | map | 0.0 |  |  | 0.78 |  |  |  |  |
| rail-like | gp | 1.00 | bootstrap | 5.9 | 200 | 34 | 0.80 | 0.04–1.17 | 2.42 |  | ApproximationWarning |
| rail-like | polyatree | 1.00 | reference | 104.4 | 8696 |  |  |  |  |  |  |
| rail-like | polyatree | 1.00 | nuts | 31.4 | 1530 | 49 | 0.03 | 0.97–1.06 | 0.13 |  |  |
| rail-like | polyatree | 1.00 | gibbs | 14.3 | 66 | 5 | 0.27 | 0.91–1.08 | 0.26 |  |  |
| rail-like | polyatree | 1.00 | laplace | 1.0 | 1825 | 1751 | 0.05 | 0.96–1.04 | 0.15 | 0.35 |  |
| rail-like | polyatree | 1.00 | vb | 0.1 | 4000 | 46898 | 0.22 | 0.24–0.83 | 1.26 |  | ApproximationWarning |
| rail-like | polyatree | 1.00 | map | 0.0 |  |  | 0.06 |  |  |  |  |
| rail-like | polyatree | 1.00 | bootstrap | 1.9 | 200 | 105 | 0.09 | 0.17–0.67 | 1.54 |  | ApproximationWarning |
