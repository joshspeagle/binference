# The software landscape

Which tools come closest to binference, across fields? This page compares them. The
field pages describe what each tool does in its field. The {doc}`index <index>` gives
the date of the survey.

## Comparison

The census found no maintained, general Python package that samples the population's
posterior from an arbitrary likelihood matrix, or from posterior samples. The maintained tools take one field's data products, stop at
point estimates, or are written in R or Julia. The tools that sample the full
hierarchical model are paper code.

The table is the one home of each tool's language, licence and status. A dash means the
census recorded nothing.

| tool | field | language | licence | status | uncertainty on $\rho$ |
|---|---|---|---|---|---|
| [RAIL `VarInfStackSummarizer`](https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/var_inf.py) | photo-$z$ | Python | MIT | pz-rail-base 2.0.11, September 2026; smoke tests only | mean-field VB |
| [Photoz-tools](https://github.com/ixkael/Photoz-tools) | photo-$z$ | Python notebooks | MIT | last push 2020 | Gibbs |
| [BHM-nz](https://github.com/Harry45/BHM-nz) | photo-$z$ | Python | MIT | last push 2023 | Gibbs |
| [CHIPPR](https://github.com/aimalz/chippr) | photo-$z$ | Python 2 | MIT | README: "stale", "a prototype" | emcee |
| [frankenz](https://github.com/joshspeagle/frankenz) | photo-$z$ | Python | MIT | last push 2018; PyPI 0.1.5 | Gibbs; invalid Metropolis |
| [Starlight](https://github.com/ixkael/Starlight) | colour–magnitude | Python, Cython | MIT | last commit 2017; never packaged | Gibbs |
| [extreme-deconvolution](https://github.com/jobovy/extreme-deconvolution) | densities | C, Python | — | maintained; not on PyPI | none |
| [pygmmis](https://github.com/pmelchior/pygmmis) | densities | Python | — | 1.2.3, 2022 | none |
| [StarFormationHistories.jl](https://github.com/cgarling/StarFormationHistories.jl) | Hess diagrams | Julia | MIT | 2.0.0, October 2026 | Hessian, HMC, MCMC |
| [exopop](https://github.com/dfm/exopop) | occurrence rates | Python | MIT | 2014 paper code | slice sampling |
| [DR25-occurrence-public](https://github.com/stevepur/DR25-occurrence-public) | occurrence rates | — | — | "not intended to be used by the community" | — |
| [PosteriorStacker](https://github.com/JohannesBuchner/PosteriorStacker) | astronomy | Python | GPL/AGPL, inconsistent | 0.6.4, September 2026 | nested sampling |
| [MRExo](https://github.com/shbhuk/mrexo) | exoplanets | Python | GPL-3 | 1.1.5, 2023; no tests | bootstrap |
| [dftools](https://github.com/obreschkow/dftools) | distribution functions | R | GPL-3 | last commit 2021 | — |
| [gwpopulation](https://github.com/ColmTalbot/gwpopulation) | GW | Python | MIT | 1.3.1, December 2025 | any bilby sampler |
| [BinnedCBCMasses](https://git.ligo.org/RatesAndPopulations/BinnedCBCMasses) | GW | Stan | MIT | last commit 2019 | Stan's sampler |
| [PixelPop](https://git.ligo.org/jack.heinzel/pixelpop) | GW | JAX, NumPyro | MIT | — | NUTS |
| [gppop](https://github.com/AnaryaRay1/gppop) | GW | PyTorch, Pyro | GPL | not on PyPI; empty test directory | NUTS |
| GWInferno | GW | — | — | not on PyPI | — |
| icarogw | GW | — | — | not on PyPI | — |
| FIGARO | GW | — | — | 54-line test suite | — |
| [RooUnfold](https://gitlab.cern.ch/RooUnfold/RooUnfold) | unfolding | C++, ROOT | BSD-3 | 3.1.0; conda-forge, not PyPI | propagated errors, toys |
| TUnfold | unfolding | C++, ROOT | — | last standalone release 2019 | — |
| [PyUnfold](https://github.com/jrbourbeau/pyunfold) | unfolding | Python | MIT | last release 2018 | per-bin errors |
| [PyFBU](https://github.com/pyFBU/fbu) | unfolding | Python, PyMC3, Theano | — | "no longer actively maintained"; needs `numpy<1.20` | NUTS |
| [mixsqp](https://cran.r-project.org/package=mixsqp) | empirical Bayes | R, C++ | MIT | — | none |
| [REBayes](https://cran.r-project.org/package=REBayes) | empirical Bayes | R | GPL (≥ 2) | needs commercial MOSEK | none |
| [ashr](https://cran.r-project.org/package=ashr) | empirical Bayes | R | GPL (≥ 3) | — | mean-field VB |
| [deconvolveR](https://github.com/bnaras/deconvolveR) | empirical Bayes | R | GPL (≥ 2) | active 2026; no tests | delta method |
| [ebnm](https://cran.r-project.org/package=ebnm) | empirical Bayes | R | GPL (≥ 3) | — | none for $g$ |
| [Empirikos.jl](https://github.com/nignatiadis/Empirikos.jl) | empirical Bayes | Julia | MIT | active 2026; MOSEK or Hypatia | frequentist intervals |
| [npeb](https://github.com/jake-soloff/NPEB) | empirical Bayes | Python | MIT | alpha; no tests | none |
| [mixsqpx](https://pypi.org/project/mixsqpx/) | empirical Bayes | Python, JAX | MIT | alpha; Python 3.12 only | none |
| [QuaPy](https://github.com/HLT-ISTI/QuaPy) | quantification | Python | BSD-3 | 0.2.3, October 2026; 16 test modules | NUTS; bootstrap regions |
| mlquantify | quantification | Python | BSD-3 | — | bootstrap regions |
| [qunfold](https://github.com/mirkobunse/qunfold) | quantification, unfolding | Python | — | — | none |
| [rubias](https://github.com/eriqande/rubias) | fisheries genetics | R, C++ | CC0 | archived on CRAN, February 2026 | Gibbs |
| SPAM, ONCOR | fisheries genetics | desktop programs | — | legacy | bootstrap |
| [mixstock](https://github.com/bbolker/mixstock) | sea turtles | R, C | GPL | archived on CRAN, 2014 | Gibbs |
| [SourceTracker2](https://github.com/caporaso-lab/sourcetracker2) | microbial ecology | Python | modified BSD | last PyPI release 2016 | Gibbs |
| [salmon](https://github.com/COMBINE-lab/salmon) | RNA-seq | Rust | BSD-3 | active 2026 | bootstrap or Gibbs replicates |
| [kallisto](https://github.com/pachterlab/kallisto) | RNA-seq | C++ | BSD-2 | active 2026 | bootstrap |
| [density-deconvolution](https://github.com/bayesiains/density-deconvolution) | neural | PyTorch | none | one commit, 2020 | none |
| [NEB](https://github.com/MaximeVandegar/NEB) | neural | PyTorch notebooks | MIT | last touched 2022 | none |
| [Denoising-Milky-Way-with-NF](https://github.com/Ziyang-Yan/Denoising-Milky-Way-with-NF) | neural | PyTorch | MIT | no packaging, no tests | none |
| [neuralG](https://github.com/shijiew97/neuralG) | neural | R, PyTorch | — | no tests | none |
| [diem](https://github.com/francois-rozet/diem) | neural | JAX | MIT | needs a Slurm cluster | none |
| [flowjax](https://github.com/danielward27/flowjax) | flows | JAX | MIT | maintained | — |
| [zuko](https://github.com/probabilists/zuko) | flows | PyTorch | MIT | maintained | — |
| nflows | flows | PyTorch | — | no release since 2020 | — |
| flow_matching (Meta) | flows | PyTorch | CC BY-NC 4.0, no commercial use | — | — |

The {doc}`guides <../guides/index>` describe binference's own inputs, engines, priors
and selection.

## Notes on the landscape

- **Many tools stop being maintained.** CRAN archived rubias because the maintainer's
  email bounced. PyFBU stopped working when Theano was discontinued. Several neural
  repositories have no licence. emcee moved MPI pools and parallel tempering into other
  packages, because "there aren't enough resources to maintain" them.
- **Tests check that code runs, not that intervals cover.** RAIL's summarizer tests are
  end-to-end smoke tests. Several tools have no tests at all.
- **Uncertainty is where the tools differ most.** Most engines converge to the same
  maximum, but their error bars mean different things. Mean-field VB under-covers.
  Convex NPMLE solvers return none. Bootstrap regions measure an estimator's
  variability. Gibbs and NUTS sample the exact posterior ({doc}`simplex-priors`).
- **Memory limits the scale.** The established remedies merge identical rows (salmon,
  Empirikos.jl), store sparse matrices (npeb), or compute each likelihood once on fixed
  bins (Starlight). binference offers the first two
  ([performance guide](../guides/performance.md)).

binference's interface borrows patterns from other scientific Python packages; the
{doc}`design notes <../design/python-patterns>` list them. The
{doc}`validation guide <../guides/validation>` describes how binference checks a fit.
