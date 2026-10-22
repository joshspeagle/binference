"""Regenerate docs/_static/quickstart.png (the README figure).

Run from the repository root: ``python docs/_scripts/quickstart_figure.py``.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pastry

mock = pastry.simulate.mock_1d(2000, rng=42)
lk = mock.likelihoods
stack = lk.stack()

flat = pastry.infer(lk, rng=1)
smooth = pastry.infer(lk, pastry.priors.LogisticGP(), rng=1)

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
