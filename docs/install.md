# Installation

pastry needs Python ≥ 3.13 with NumPy ≥ 2.1, SciPy ≥ 1.14.1 and the small
pure-Python package threadpoolctl, which keeps parallel chains from
oversubscribing the cores. Plotting needs matplotlib, and export to
ArviZ-style `DataTree`s needs xarray. Both are optional extras.

pastry is in alpha. Until it is released on PyPI as `pastry-inference` (the
name `pastry` is taken), install it from GitHub:

```bash
pip install "pastry-inference[plot] @ git+https://github.com/joshspeagle/pastry"
```

For development:

```bash
git clone https://github.com/joshspeagle/pastry
cd pastry
pip install -e ".[dev]"
```

| extra | adds |
|---|---|
| `plot` | matplotlib, for `result.plot()` and friends |
| `xarray` | xarray, for `result.to_datatree()` |
| `test` | pytest, coverage and matplotlib (for the plotting tests) |
| `docs` | Sphinx and the notebook toolchain used to build these pages |
| `dev` | all of the above plus ruff and pre-commit |
| `all` | `plot` and `xarray` |

Check the installation:

```python
import pastry

print(pastry.__version__)
mock = pastry.simulate.mock_1d(500, pastry.Bins.linear(0, 3, 10), rng=0)
print(pastry.infer(mock.likelihoods, method="map").rho.round(3))
```

Then work through the [quick-start tutorial](tutorials/01_quickstart).
