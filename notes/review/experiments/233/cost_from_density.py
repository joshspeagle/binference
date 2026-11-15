"""#233 cost: a KDE fed through from_density's adaptive quadrature (the issue's route), N=1000, S=1000."""
import time, warnings
import numpy as np
from scipy.special import ndtr
from scipy.stats import truncnorm
import binference as bf

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)
N, S, K, sigma = 1000, 1000, 20, 0.08
bins = bf.Bins(np.linspace(0, 1, K + 1))
d = rng.uniform(0.2, 0.8, N)
x = truncnorm.rvs(-d / sigma, (1 - d) / sigma, loc=d[:, None], scale=sigma, size=(N, S), random_state=rng)
h = 0.9 * x.std(1, ddof=1) * S ** -0.2
c = 1 / (np.sqrt(2 * np.pi) * h)


def kde(z):  # (M,) -> (N, M), reflected at 0 and 1
    out = np.zeros((N, z.size))
    for a in range(0, N, 50):
        xs, hs = x[a:a + 50, :, None], h[a:a + 50, None, None]
        zz = z[None, None, :]
        out[a:a + 50] = (np.exp(-0.5 * ((zz - xs) / hs) ** 2) + np.exp(-0.5 * ((zz + xs) / hs) ** 2)
                         + np.exp(-0.5 * ((zz - 2 + xs) / hs) ** 2)).mean(1) * c[a:a + 50, None]
    return out


calls = []
def counted(z):
    calls.append(z.size)
    return kde(z)


t = time.time()
lk = bf.Likelihoods.from_density(counted, bins, interim="flat")
print(f"from_density(KDE callable) N={N} S={S}: {time.time() - t:.1f} s, {len(calls)} calls, {sum(calls)} points")
