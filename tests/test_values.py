"""Value draws from each input's within-bin information (DECISIONS #50)."""

import pickle

import numpy as np
import pytest
from scipy import stats

import binference
from binference import Basis, Bins, Likelihoods

BINS = Bins.linear(0, 1, 5)
MU, SD = np.array([0.37, 0.52, 0.71]), 0.08
N_COPY = 4000  # draws of object 0, as copies of its row


# log pi = -10 z: weights vary by e^2 across a bin, so dividing it out shows, while the
# posteriors stay N(MU - SD^2 / 0.1, SD) on z >= 0
INTERIM = stats.expon(scale=0.1)
POST_MU = MU - SD**2 / 0.1


def _values(lk, **kw):
    est = binference.infer(lk, method="map")
    return est.sample_values(idx=np.zeros(N_COPY, int), rng=1, **kw)[0]


def _check_truncnorm(vals, mu, sd, ess=None):
    # within each bin with enough draws, the mean is that of N(mu, sd) truncated to it;
    # resampled draws also carry the error of the samples, ess(lo, hi) of them in the bin
    k = BINS.index(vals)
    for b in np.unique(k):
        v = vals[k == b]
        if v.size < 200:
            continue
        lo, hi = BINS.edges[0][b], BINS.edges[0][b + 1]
        ref = stats.truncnorm((lo - mu) / sd, (hi - mu) / sd, loc=mu, scale=sd)
        var = 1 / v.size + (0 if ess is None else 1 / ess(lo, hi))
        assert abs(v.mean() - ref.mean()) < 4 * ref.std() * np.sqrt(var)


def test_density_likelihoods_draw_values_from_the_density_inside_the_bin():
    def dens(z):
        return stats.norm.pdf(z[None, :], MU[:, None], SD)

    lk = Likelihoods.from_density(dens, BINS, kind="likelihood")
    _check_truncnorm(_values(lk), MU[0], SD)
    dists = [stats.norm(m, SD) for m in MU]
    _check_truncnorm(_values(Likelihoods.from_density(dists, BINS, kind="likelihood")), MU[0], SD)


def test_posterior_densities_divide_out_the_interim_prior_inside_the_bin():
    post = [stats.norm(m, SD) for m in POST_MU]
    lk = Likelihoods.from_density(post, BINS, interim=INTERIM)
    _check_truncnorm(_values(lk), MU[0], SD)


def test_samples_are_resampled_inside_the_drawn_bin_with_the_interim_divided_out():
    samples = POST_MU[:2, None] + SD * np.random.default_rng(0).standard_normal((2, 100_000))
    lk = Likelihoods.from_samples(samples, BINS, interim=INTERIM)
    vals = _values(lk)
    assert np.isin(vals, samples[0]).all()

    def ess(lo, hi):  # of the weights 1 / pi of object 0's samples in [lo, hi)
        z = samples[0][(samples[0] >= lo) & (samples[0] < hi)]
        w = 1 / INTERIM.pdf(z)
        return w.sum() ** 2 / (w**2).sum()

    _check_truncnorm(vals, MU[0], SD, ess)


def test_samples_inside_a_hat_are_weighed_by_its_height():
    # flat samples: inside a hat the values follow the hat (sd = half-width / sqrt(6))
    basis = Basis(Bins.linear(0, 1, 4), order=1)
    samples = np.random.default_rng(0).random((2, 40_000))
    lk = Likelihoods.from_samples(samples, basis, interim="flat")
    est = binference.infer(lk, method="map")
    c = 2  # an interior component, centred at 0.625 with half-width 0.25
    labels = est._component_labels(np.zeros(N_COPY, int), None, np.random.default_rng(2))
    vals = est.sample_values(idx=np.zeros(N_COPY, int), rng=2)[0]
    v = vals[labels[0] == c]
    assert abs(v.mean() - 0.625) < 4 * 0.25 / np.sqrt(6) / np.sqrt(v.size)
    assert abs(v.std() - 0.25 / np.sqrt(6)) < 4 * 0.25 / np.sqrt(6) / np.sqrt(2 * v.size)


def test_correlated_gaussians_draw_jointly_within_the_box():
    grid = Bins.grid([np.linspace(0, 1, 3), np.linspace(0, 1, 3)])
    mu, cov = np.array([0.45, 0.4]), 0.15**2 * np.array([[1.0, 0.9], [0.9, 1.0]])
    lk = Likelihoods.from_gaussian(np.tile(mu, (3, 1)), bins=grid, cov=cov)
    vals = _values(lk)
    k = grid.index(vals)
    box = vals[k == np.bincount(k).argmax()]  # the box with most draws
    lo, hi = box.min(0).round(1), box.max(0).round(1)  # its edges, on this 0.5 grid
    ref = np.random.default_rng(3).multivariate_normal(mu, cov, 400_000)
    ref = ref[np.all((ref >= lo) & (ref <= hi), axis=1)]
    r, r_ref = np.corrcoef(box.T)[0, 1], np.corrcoef(ref.T)[0, 1]
    assert abs(r - r_ref) < 4 * (1 - r_ref**2) / np.sqrt(box.shape[0])
    assert np.all(np.abs(box.mean(0) - ref.mean(0)) < 4 * ref.std(0) / np.sqrt(box.shape[0]))


def test_only_inputs_without_within_bin_information_warn():
    gauss = Likelihoods.from_gaussian(MU, SD, BINS)
    with pytest.warns(binference.InputWarning, match=r"\(table\) keep no information"):
        _values(Likelihoods.from_table(gauss.values, BINS, kind="likelihood"))
    samples = np.random.default_rng(0).normal(0.5, 0.1, (3, 500))
    with pytest.warns(binference.InputWarning, match="keep no information"):
        _values(Likelihoods.from_samples(samples, BINS, interim="flat", keep_samples=False))
    _values(Likelihoods.from_samples(samples, BINS, interim="flat")[1:])  # rows follow subsets


def test_within_bin_data_stay_out_of_pickles():
    # NUTS sends the model to its workers: a lambda density must not stop that
    lk = Likelihoods.from_density(lambda z: np.ones((2, z.size)), BINS, kind="likelihood")
    copy = pickle.loads(pickle.dumps(binference.PopulationModel(lk)))
    assert lk.meta["within"].data is not None and copy.likelihoods.meta["within"].data is None
