# Install

binference needs Python 3.13 or later. Its core depends on three packages: NumPy
(2.1 or later), SciPy (1.14.1 or later) and threadpoolctl, which keeps parallel chains
from oversubscribing the cores.

binference is in alpha. Until it is released on PyPI, install it from GitHub, with the
plotting extra:

```bash
pip install "binference[plot] @ git+https://github.com/joshspeagle/binference"
```

Leave out `[plot]` for the core alone, or name several extras, such as `[plot,xarray]`.

## Extras

| Extra | Adds | For |
|---|---|---|
| `plot` | matplotlib | `result.plot()` and the functions of `binference.plotting` |
| `xarray` | xarray | `result.to_datatree()`, an ArviZ-compatible export |
| `all` | both of the above | |
| `test` | pytest, pytest-cov and matplotlib | running the test suite |
| `docs` | Sphinx, myst-nb and the rest of the documentation toolchain | building these pages |
| `dev` | `test`, `docs`, `plot`, `xarray` and pre-commit | working on binference itself |

To work on binference itself, follow the set-up in the
[contributing guide](../contributing.md).

## Check the installation

This fits a simulated catalogue of 500 objects on 10 bins and prints the fitted
population fractions.

```python
import binference

print(binference.__version__)
mock = binference.simulate.mock_1d(500, binference.Bins.linear(0, 3, 10), rng=0)
print(binference.infer(mock.likelihoods, method="map").rho.round(3))
```

Next, the {doc}`quickstart` takes you from your own arrays to a population.
