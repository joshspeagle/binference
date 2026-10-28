"""Regenerate docs/_static/quickstart.png (the README figure).

Run from the repository root, with the docs style:
``MATPLOTLIBRC=docs/_ext/matplotlibrc PYTHONPATH=src python docs/_scripts/quickstart_figure.py``.
The style sets the text size and the figure's width; this script sets only the height of its
two stacked panels.
It fits the example of the README and the home page.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import binference
from binference.plotting import SAMPLE_COLOR

bins = binference.Bins.linear(0.0, 3.0, 15)
mock = binference.simulate.mock_1d(2000, bins, rng=42)
z_obs, z_err = mock.measurements["mean"], mock.measurements["sigma"]
lk = binference.Likelihoods.from_gaussian(z_obs, z_err, bins)
stack = lk.stack()

flat = binference.infer(lk, rng=1)
smooth = binference.infer(lk, binference.priors.LogisticGP(), rng=1)

fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
fig.set_figheight(6.4)
for ax, result in zip(axes, (flat, smooth), strict=True):
    result.plot(ax=ax, truth=mock.truth.rho, stack=stack, sample_color=SAMPLE_COLOR)
legend = axes[0].get_legend()  # one legend above both panels, with the bands' entries
handles, labels = legend.legend_handles, [t.get_text() for t in legend.texts]
for ax in axes:
    ax.get_legend().remove()
axes[0].set_title("Dirichlet prior (default)", loc="left")
axes[1].set_title("Gaussian-process prior", loc="left")
axes[0].set_xlabel("")
axes[1].set_xlabel("z")
fig.legend(handles, labels, loc="outside upper center", ncol=2)
fig.savefig("docs/_static/quickstart.png")
print(flat.summary())
print(smooth.diagnostics)
