<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/joshspeagle/binference/main/docs/_themes/binference/static/brand/lockup-large-dark.svg">
    <img alt="binference" width="300" src="https://raw.githubusercontent.com/joshspeagle/binference/main/docs/_themes/binference/static/brand/lockup-large-light.svg">
  </picture>
</h1>

[![PyPI version][pypi-badge]][pypi]
[![Supported Python versions][python-badge]][pypi]
[![CI status on main][ci-badge]][ci]
[![Documentation][docs-badge]][docs]

**Population inference from noisy measurements.** binference combines per-object
likelihoods, posteriors, samples, binned counts, or classifier outputs. Fit the
population distribution, quantify its uncertainty, and use it to update each
object's posterior.

[Quick start][quickstart] · [Choose an input][inputs] · [Tutorials][tutorials] · [API reference][api]

## Install

```bash
pip install "binference[plot]"
```

See the [installation guide][install] for supported Python versions and extras.

## Fit a population

This example estimates a redshift distribution from measured values and their
Gaussian errors. Replace the simulated arrays with your own.

```python
import binference

bins = binference.Bins.linear(0.0, 3.0, 15)
mock = binference.simulate.mock_1d(2000, bins, rng=42)
z_obs, z_err = mock.measurements["mean"], mock.measurements["sigma"]

lk = binference.Likelihoods.from_gaussian(z_obs, bins, z_err)
result = binference.infer(lk, rng=1)
print(result.summary())
```

![Population posteriors for a mock catalogue][figure]

*Population posteriors for 2,000 simulated objects under a Dirichlet prior (top)
and a Gaussian-process prior (bottom). Each panel shows the posterior median,
68% and 95% intervals, the true density (dashed), and the stacked estimate (dotted).*

## Status, citation, and contributions

Version 0.3.0 is the first beta. The API may change before 1.0. If you use
binference in your work, see [how to cite it][citing]. Contributions follow the
[contributor guide][contributing]. binference is released under the [MIT License][license].

[pypi-badge]: https://img.shields.io/pypi/v/binference
[python-badge]: https://img.shields.io/pypi/pyversions/binference
[ci-badge]: https://github.com/joshspeagle/binference/actions/workflows/ci.yml/badge.svg?branch=main
[docs-badge]: https://img.shields.io/badge/docs-online-4A4FC4
[pypi]: https://pypi.org/project/binference/
[ci]: https://github.com/joshspeagle/binference/actions/workflows/ci.yml
[docs]: https://joshspeagle.github.io/binference/
[quickstart]: https://joshspeagle.github.io/binference/start/quickstart.html
[inputs]: https://joshspeagle.github.io/binference/guides/inputs/index.html
[tutorials]: https://joshspeagle.github.io/binference/tutorials/index.html
[api]: https://joshspeagle.github.io/binference/reference/api/index.html
[install]: https://joshspeagle.github.io/binference/start/install.html
[figure]: https://raw.githubusercontent.com/joshspeagle/binference/main/docs/_static/quickstart.png
[citing]: https://joshspeagle.github.io/binference/reference/citing.html
[contributing]: https://github.com/joshspeagle/binference/blob/main/CONTRIBUTING.md
[license]: https://github.com/joshspeagle/binference/blob/main/LICENSE
