# Installation

binference needs Python ≥ 3.13 with NumPy ≥ 2.1, SciPy ≥ 1.14.1 and the small
pure-Python package threadpoolctl, which keeps parallel chains from
oversubscribing the cores. Plotting needs matplotlib, and export to
ArviZ-style `DataTree`s needs xarray. Both are optional extras.

binference is in alpha. Until it is released on PyPI as `binference` (the
name `binference` is taken), install it from GitHub:

```bash
pip install "binference[plot] @ git+https://github.com/joshspeagle/binference"
```

For development:

```bash
git clone https://github.com/joshspeagle/binference
cd binference
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
import binference

print(binference.__version__)
mock = binference.simulate.mock_1d(500, binference.Bins.linear(0, 3, 10), rng=0)
print(binference.infer(mock.likelihoods, method="map").rho.round(3))
```

Then work through the [quick-start tutorial](tutorials/01_quickstart).
