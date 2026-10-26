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


def test_workers_get_no_within_bin_data_and_unpicklable_data_pickle_as_empty():
    from binference.samplers.nuts import _worker_model

    # NUTS sends the model to its workers: a lambda density must not stop that
    lk = Likelihoods.from_density(lambda z: np.ones((2, z.size)), BINS, kind="likelihood")
    model = binference.PopulationModel(lk)
    assert "within" not in _worker_model(model).likelihoods.meta
    assert "within" in model.likelihoods.meta  # the caller's model keeps them
    with pytest.warns(binference.ApproximationWarning, match="cannot be pickled"):
        assert pickle.loads(pickle.dumps(model)).likelihoods.meta["within"].data is None
    samples = np.random.default_rng(0).normal(0.5, 0.1, (2, 50))
    kept = Likelihoods.from_samples(samples, BINS, interim="flat").meta["within"]
    assert pickle.loads(pickle.dumps(kept)).data is not None


def test_a_bin_no_sample_reached_draws_flat_within_it():
    # with a floor, bins without samples keep a likelihood; there the values are flat
    rng = np.random.default_rng(0)  # object i's samples all lie in bin i % 5
    samples = 0.2 * (np.arange(20)[:, None] % 5 + rng.uniform(0.0, 1.0, (20, 200)))
    lk = Likelihoods.from_samples(samples, BINS, interim="flat", floor=50.0)
    vals = _values(lk)
    far = vals[vals >= 0.2]
    k = BINS.index(far)
    assert far.size > 200
    mid = BINS.centers[k]
    assert abs(np.mean(far - mid)) < 4 * 0.2 / np.sqrt(12 * far.size)


def test_truncated_normal_draws_keep_their_precision_in_far_tails():
    from binference.results import _truncnorm

    a, b = np.array([-1.0, 8.0, -12.0]), np.array([0.5, 9.0, -11.0])
    x = _truncnorm(np.repeat(a, 20_000), np.repeat(b, 20_000), np.random.default_rng(0))
    for i, (lo, hi) in enumerate(zip(a, b, strict=True)):
        xi = x[i * 20_000 : (i + 1) * 20_000]
        ref = stats.truncnorm(lo, hi)
        assert np.all((xi >= lo) & (xi <= hi))
        assert abs(xi.mean() - ref.mean()) < 4 * ref.std() / np.sqrt(xi.size)


def test_samples_with_zero_weights_leave_the_other_objects_and_bins_intact():
    # object 0 has only zero-weight samples (kept by the floor); object 1's samples in
    # [0.4, 0.6) weigh nothing, so its draws there are flat rather than one sample
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 1.0, (3, 2000))
    sw = np.ones_like(samples)
    sw[0] = 0.0
    sw[1][(samples[1] >= 0.4) & (samples[1] < 0.6)] = 0.0
    lk = Likelihoods.from_samples(
        samples,
        BINS,
        interim="flat",
        sample_weights=sw,
        floor=2000,
        min_samples=0,
        warn_outside=False,
    )  # object 0 has neither effective samples nor mass inside the bins
    assert np.all(np.isfinite(lk.meta["within"].data[-1]))
    prior = binference.priors.Dirichlet(5.0)  # keeps every bin in the population
    est = binference.samplers.MAP(binference.PopulationModel(lk, prior=prior)).run()
    vals = est.sample_values(idx=np.ones(N_COPY, int), rng=1)[0]
    mid = vals[(vals >= 0.4) & (vals < 0.6)]
    assert mid.size > 20 and np.unique(mid).size == mid.size
    assert np.unique(est.sample_values(idx=np.full(200, 2), rng=1)).size > 100


def test_an_interim_prior_that_vanishes_inside_a_bin_does_not_collapse_the_draws():
    post = [stats.uniform(0.0, 0.95)] * 3  # posteriors under the same interim prior
    lk = Likelihoods.from_density(post, BINS, interim=stats.uniform(0.0, 0.95))
    vals = _values(lk)
    last = vals[vals >= 0.8]
    assert last.size > 200 and last.max() - last.min() > 0.1


def test_distributions_without_a_density_fall_back_with_a_warning():
    class CdfOnly:
        def __init__(self, m):
            self.m = m

        def cdf(self, z):
            return stats.norm.cdf(z, self.m, SD)

    lk = Likelihoods.from_density([CdfOnly(m) for m in MU], BINS, kind="likelihood")
    with pytest.warns(binference.InputWarning, match="keep no information"):
        _values(lk)


def test_density_functions_are_evaluated_a_few_bins_at_a_time(monkeypatch):
    lk = Likelihoods.from_density(
        lambda z: stats.norm.logpdf(z[None, :], MU[:, None], SD), BINS, kind="likelihood", log=True
    )
    whole = _values(lk)
    monkeypatch.setattr(binference.results, "EVAL_BUDGET", 3 * 64)  # one bin per call
    np.testing.assert_array_equal(_values(lk), whole)
    dists = Likelihoods.from_density([stats.norm(m, SD) for m in MU], BINS, kind="likelihood")
    assert pickle.loads(pickle.dumps(dists)).meta["within"].data is not None
