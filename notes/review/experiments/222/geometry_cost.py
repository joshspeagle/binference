"""Issue #222: time of NUTS's warm-up curvature step (`_adapt_geometry`) for LogisticGP
with the default basis, four chains (the default), against the dense limit's two settings:
every chain's position, or one. Mock as experiments/103_gp_basis_on_grids.py. Run with
OPENBLAS_NUM_THREADS=1."""
import time, warnings
import numpy as np
from binference import Bins, priors as P
from binference.engines import nuts
from binference.model import PopulationModel
from binference.simulate import mock_2d

warnings.simplefilter("ignore")
for k in (16, 20, 22):
    mock = mock_2d(3000, Bins.grid(*[np.linspace(0, 1, k + 1)] * 2), noise=0.5 / k, rng=0)
    model = PopulationModel(mock.likelihoods, P.LogisticGP())
    g = model.log_density_unconstrained_and_grad
    x = np.zeros(model.dim)
    t = time.perf_counter(); [g(x) for _ in range(50)]; per = (time.perf_counter() - t) / 50
    for limit in (nuts._DENSE_MAX_DIM, 0):
        nuts._DENSE_MAX_DIM = limit
        eng = nuts.NUTS(model, n_chains=4, n_warmup=0, metric="dense", rng=0)
        eng._setup()
        t = time.perf_counter(); eng._adapt_geometry(); t = time.perf_counter() - t
        print(f"{k}x{k} dim {model.dim} gradient {per * 1e3:.1f} ms; curvature at "
              f"{'every chain' if limit else 'one chain  '} {t:5.1f} s", flush=True)
    nuts._DENSE_MAX_DIM = 500
