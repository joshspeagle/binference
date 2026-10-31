---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# How auto chooses

How does `method="auto"` pick an engine, and what evidence sets its rules? This page
states the rules and describes the benchmark behind them.
{class}`~binference.inference.Auto` gives the defaults of their options. The
{doc}`engines guide <../guides/engines>` covers when to override the choice.

## The rules

{term}`Auto` tries three engines in order and keeps the first one that passes its
check. Engine options passed to `infer` rule out the engines that do not take them.

1. **Gibbs, where it mixes well.** Gibbs needs a {term}`conjugate prior` and integer
   weights. It is chosen in two cases.
   - A selection or response is inferred with the population. Gibbs is exact and fast
     for these joint models. Gibbs refuses a selection smoothed across bins, which is
     not conjugate, so auto then tries Laplace and NUTS.
   - The [missing information](point-estimates.md#missing-information) $f$ at the most
     probable population is at most `gibbs_max_missing`. Auto computes that population
     first, with {class}`~binference.samplers.MAP`.

   In the second case Gibbs runs
   $\lceil n_{\rm ESS}\,(1+f)/(1-f)/n_{\rm chains}\rceil$ draws per chain, and at least
   its usual length. Here $n_{\rm ESS}$ is `target_ess`, the smallest bulk {term}`ESS`
   the run aims for. If that would exceed `max_gibbs_draws` per chain, auto skips Gibbs.
2. **Laplace, if its importance check passes.** Auto keeps the importance-corrected
   Laplace result when its {term}`Pareto k-hat` passes and its importance ESS reaches
   `target_ess`. A first pass with fewer proposals stops the trial early when it
   projects too small an ESS
   ([defaults](../reference/defaults.md#automatic-engine-choice)).
3. **NUTS** otherwise.

Without `n_samples`, auto extends a Gibbs run until every bulk ESS reaches `target_ess`
and the run passes the $\hat R$ and ESS limits of
{meth}`~binference.PopulationResult.check`. Each extension aims a little above
`target_ess` and is bounded in size
([defaults](../reference/defaults.md#automatic-engine-choice)). If the extensions would
need more than `max_gibbs_draws` draws per chain, auto discards the run and tries the
next engine. This catches a jointly inferred selection or response that mixes much more
slowly than $f$ predicts. Gibbs may be the last candidate, because no other engine
takes the options given. Auto then keeps it even if it did not mix, and the reason says
so.
Two kinds of run are never extended. An explicit `n_samples` fixes the length: auto
then keeps Gibbs whatever length it would need. A run that stores labels
(`store_labels`) keeps the length it was given.

Auto keeps Laplace even when its check fails in two cases. One is likelihoods with more
than `nnz_limit` stored entries. The other is options that no later engine takes. Such a
result is flagged {term}`approximate`. The result records the choice in
`result.info["auto_method"]`, and in `result.info["auto_reason"]` the reasons for it: why a
rule left Gibbs out, why earlier engines were not kept, and why this one was. Where auto
computed the most probable population without a warning, the result keeps it as
`result.best`.

## The benchmark behind the rules

The rules come from `benchmarks/matrix.py`, which compares every engine with a long
reference run on a matrix of mock catalogues. The numbers below are computed from its
committed results; the full table is
[matrix.md](https://github.com/joshspeagle/binference/blob/main/benchmarks/results/matrix.md).

```{code-cell} python
:tags: [remove-cell]

import inspect
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from IPython.display import Markdown
from myst_nb import glue

from binference.inference import Auto

ROOT = Path("../..").resolve()
sys.path.insert(0, str(ROOT / "benchmarks"))
import matrix  # noqa: E402  (the benchmark's own settings)

lines = (ROOT / "benchmarks" / "results" / "matrix.json").read_text().splitlines()
rows = [json.loads(line) for line in lines if line.strip()]
head = [r for r in rows if r["engine"] == "header"][-1]
cells = [r for r in rows if r["engine"] not in ("header", "reference") and "error" not in r]
by = {}
for r in cells:
    by.setdefault(r["engine"], []).append(r)
defaults = inspect.signature(Auto.__init__).parameters
target = defaults["target_ess"].default
f_max = defaults["gibbs_max_missing"].default


NAMES = {"randomwalk": "random walk", "dirichlet0.1": "α = 0.1 Dirichlet",
         "polyatree": "Pólya tree", "gp": "Gaussian process", "dirichlet": "flat Dirichlet"}


def g(name, value):
    glue(name, Markdown(str(value)), display=False)


def approx(r):
    return "ApproximationWarning" in r["warnings"]


lap = by["laplace"]
passed = [r for r in lap if not approx(r) and r["ess"] >= target]
short = [r for r in lap if not approx(r) and r["ess"] < target]
groups = {
    f"NUTS ({len(by['nuts'])})": by["nuts"],
    f"Gibbs ({len(by['gibbs'])})": by["gibbs"],
    f"Laplace, check passed ({len(passed)})": passed,
    f"Laplace, check failed ({len(lap) - len(passed)})": [r for r in lap if r not in passed],
    f"mean-field VB ({len(by['vb'])})": by["vb"],
}


def fmt(x):
    return f"{x:.2g}" if x < 10 else f"{x:.0f}"


table = [
    "| engine | largest mean error (reference s.d.) | s.d. ratio | smallest bulk ESS per second |",
    "|---|---|---|---|",
]
for name, rs in groups.items():
    err = max(r["mean_err"] for r in rs)
    lo, hi = min(r["sd_ratio_min"] for r in rs), max(r["sd_ratio_max"] for r in rs)
    ess = ""
    if "VB" not in name and "failed" not in name:
        e = [r["ess_per_s"] for r in rs]
        ess = f"{fmt(min(e))}–{fmt(max(e))}"
        if "Laplace" not in name:
            ess += f", median {fmt(np.median(e))}"
    table.append(f"| {name} | {fmt(err)} | {lo:.2f}–{hi:.2f} | {ess} |")
g("table", "\n".join(table))

# the setup
g("n_scenarios", len(matrix.SCENARIOS))
base = {k: v.default for k, v in inspect.signature(matrix._gauss).parameters.items()}
g("base_K", base["K"])
g("base_N", f"{base['N']:,}")
g("scenarios", "\n".join(
    f"- `{name}`: {sc.note}" for name, sc in matrix.SCENARIOS.items() if name != "baseline"
))
*first, last = (NAMES[p] for p in matrix.ALL)
g("priors", f"{', '.join(first)} and {last}")
g("n_cells", len({(r["scenario"], r["prior"]) for r in cells}))
ref = matrix.REFERENCE
g("ref_chains", ref["n_chains"])
g("ref_draws", f"{ref['n_samples']:,}")
g("ref_warmup", f"{ref['n_warmup']:,}")
g("setup", (
    f"binference {head['binference']}, NumPy {head['numpy']} and Python {head['python']} "
    f"on {head['machine']}, started on "
    f"{datetime.strptime(head['started'], '%Y-%m-%d %H:%M:%S'):%d %B %Y}"
))
g("jobs", head["jobs"])

# NUTS
nuts = by["nuts"]
g("nuts_err", f"{max(r['mean_err'] for r in nuts):.2f}")
wide = {r["prior"] for r in nuts if r["sd_ratio_min"] < 0.9 or r["sd_ratio_max"] > 1.1}
g("nuts_wide_priors", " and ".join(NAMES[p] for p in sorted(wide, reverse=True)))
small = [r for r in nuts if r["prior"] == "dirichlet0.1"]
g("nuts_small_ess", f"{fmt(min(r['ess_per_s'] for r in small))} to "
  f"{fmt(max(r['ess_per_s'] for r in small))}")
g("nuts_small_slowest", f"{max(r['seconds'] for r in small):.0f}")

# Gibbs against NUTS, either side of gibbs_max_missing
n_of = {(r["scenario"], r["prior"]): r for r in nuts}
for side, keep in (("low", lambda f: f <= f_max), ("high", lambda f: f > f_max)):
    gs = [r for r in by["gibbs"] if keep(r["missing_information"])]
    ns = [n_of[(r["scenario"], r["prior"])] for r in gs]
    wins = sum(a["ess_per_s"] > b["ess_per_s"] for a, b in zip(gs, ns, strict=True))
    fs = [r["missing_information"] for r in gs]
    g(f"g_{side}_n", len(gs))
    g(f"g_{side}_wins", wins)
    g(f"g_{side}_f", f"{(max(fs) if side == 'low' else min(fs)):.3f}")
    g(f"g_{side}_g", f"{fmt(min(r['ess_per_s'] for r in gs))}–"
      f"{fmt(max(r['ess_per_s'] for r in gs))}")
    g(f"g_{side}_nuts", f"{fmt(min(r['ess_per_s'] for r in ns))}–"
      f"{fmt(max(r['ess_per_s'] for r in ns))}")

# Laplace
g("lap_priors", " and ".join(NAMES[p] for p in sorted({r["prior"] for r in passed})))
g("lap_err", f"{max(r['mean_err'] for r in passed):.2f}")
g("lap_s", f"{min(r['seconds'] for r in passed):.1f} to "
  f"{max(r['seconds'] for r in passed):.1f}")
g("lap_fail_err", f"{max(r['mean_err'] for r in lap if approx(r)):.0f}")
g("lap_short_n", len(short))
g("lap_short_q", f"{max(r['q_err'] for r in short):.1f}")
g("vb_sd", f"{min(r['sd_ratio_min'] for r in by['vb']):.2f}")
```

**Catalogues.** There are {glue:md}`n_scenarios` catalogues, each varying one factor
around a baseline. The baseline has one dimension, {glue:md}`base_K` bins and
{glue:md}`base_N` objects with Gaussian errors one bin wide. Its population is bimodal,
and true values are uniform within each bin. The other catalogues are:

```{glue:md} scenarios
:format: myst
```

**Fits.** Each catalogue was fitted with the priors that apply to it, from the
{glue:md}`priors` priors: {glue:md}`n_cells` combinations in all. Each engine ran at its
defaults. The reference was NUTS with {glue:md}`ref_chains` chains of
{glue:md}`ref_draws` draws after {glue:md}`ref_warmup` warm-up iterations. Errors are in
reference standard deviations: the largest error of a bin's mean, and the ratio of each
bin's standard deviation to the reference's.

**Setup.** {glue:md}`setup`. Each fit used one thread, with {glue:md}`jobs` fits at a
time.

```{glue:md} table
:format: myst
```

For Laplace the last column is the importance ESS per second, the effective number of
independent draws among the weighted proposals, not a bulk ESS of a chain.

What the table shows:

- **NUTS** was within {glue:md}`nuts_err` reference standard deviations in every mean.
  Its ratios outside 0.9–1.1 all come from the {glue:md}`nuts_wide_priors` priors. It is
  slow with Dirichlet concentrations below one. At 0.1 it managed
  {glue:md}`nuts_small_ess` ESS per second, and its slowest fit took
  {glue:md}`nuts_small_slowest` s.
- **Gibbs** beat NUTS in {glue:md}`g_low_wins` of the {glue:md}`g_low_n` combinations
  with $f$ at most `gibbs_max_missing` (the largest was {glue:md}`g_low_f`). There it
  reached {glue:md}`g_low_g` ESS per second, against {glue:md}`g_low_nuts` for NUTS.
- Above that threshold (from $f=$ {glue:md}`g_high_f`), Gibbs beat NUTS in only
  {glue:md}`g_high_wins` of {glue:md}`g_high_n` combinations. It reached
  {glue:md}`g_high_g` ESS per second, against {glue:md}`g_high_nuts`. The benchmark
  ran Gibbs at its default length. Below the threshold, auto's run length from $f$ is
  designed to make up for the autocorrelation; the benchmark did not measure it.
- **Laplace** passed its check only with the {glue:md}`lap_priors` prior. There it
  agreed with the reference to {glue:md}`lap_err` standard deviations, in
  {glue:md}`lap_s` s. Where the check failed, its errors reached
  {glue:md}`lap_fail_err` standard deviations.
- The ESS requirement rejected {glue:md}`lap_short_n` more Laplace fits, whose
  $\hat k$ passed with fewer effective draws than `target_ess`. Their quantiles were
  up to {glue:md}`lap_short_q` standard deviations off.
- **Mean-field VB** was fast, but its standard deviation ratios went down to
  {glue:md}`vb_sd`, so auto never chooses it.

The slow test suite re-checks four of these combinations against stored reference
posteriors, including the engine auto picks in each
([the test suite](validation.md#the-statistical-tests)).
