# #204: calibration of derived totals (DECISIONS #144)

Run from this folder with `OPENBLAS_NUM_THREADS=1` and `PYTHONPATH` at the checkout's `src`.

- `run.py OPT LO HI exact|inj`: NUTS fits of data sets `LO..HI-1` under population prior
  option `a`, `b`, `c` or `d` (see the file); `analyse.py exact|inj` summarises them.
- `run2.py OPT LO HI`: the injection settings `i1`..`i6` under the default population prior;
  `analyse2.py` summarises them; `injections.out` is its output for 200 data sets each.
- `known_labels.py` and `analytic.py`: exact posteriors with known labels, by importance
  sampling, and the analytic bias of the Beta posterior mean of $S_k$.

The issue's comments of 2026-11-15 give the tables.
