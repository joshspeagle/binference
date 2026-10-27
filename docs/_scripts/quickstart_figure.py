"""Regenerate docs/_static/quickstart.png (the README figure).

Run from the repository root, with the docs style:
``MATPLOTLIBRC=docs/_ext/matplotlibrc python docs/_scripts/quickstart_figure.py``.
It fits the example of the README and the home page.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import binference

bins = binference.Bins.linear(0.0, 3.0, 15)
mock = binference.simulate.mock_1d(2000, bins, rng=42)
z_obs, z_err = mock.measurements["mean"], mock.measurements["sigma"]
lk = binference.Likelihoods.from_gaussian(z_obs, z_err, bins)
stack = lk.stack()

flat = binference.infer(lk, rng=1)
smooth = binference.infer(lk, binference.priors.LogisticGP(), rng=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
flat.plot(ax=axes[0], truth=mock.truth.rho, stack=stack)
smooth.plot(ax=axes[1], truth=mock.truth.rho, stack=stack)
axes[0].set_title("Dirichlet prior (default)", loc="left")
axes[1].set_title("Gaussian-process prior", loc="left")
for ax in axes:
    ax.set_xlabel("z")
axes[1].set_ylabel("")
axes[1].get_legend().remove()
fig.savefig("docs/_static/quickstart.png", dpi=150)
print(flat.summary())
print(smooth.diagnostics)
