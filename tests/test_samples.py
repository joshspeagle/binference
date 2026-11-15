"""Tests for the samples estimator (DECISIONS #31, audit I-23)."""

import warnings

import numpy as np
import pytest
from scipy import stats

import binference
from binference import Bins, Density, InputWarning, Likelihoods


def _posterior_samples(rng, n_obj, n_samp, like, interim, lo, hi):
    # rejection samples from like(z) * interim(z) on [lo, hi]
    out = []
    for _ in range(n_obj):
        z = rng.uniform(lo, hi, 40 * n_samp)
        p = like(z) * interim(z)
        z = z[rng.uniform(0, p.max(), z.size) < p]
        out.append(z[:n_samp])
    return out


def test_a_pointwise_interim_is_divided_out_exactly_within_bins():
    rng = np.random.default_rng(0)
    bins = Bins([0.0, 0.5, 1.0])  # two wide bins
    like = stats.norm(0.45, 0.3).pdf
    interim = lambda z: np.exp(-3 * z)  # noqa: E731  varies strongly within the bins
    samples = _posterior_samples(rng, 20, 20000, like, interim, 0.0, 1.0)
    exact = Likelihoods.from_gaussian([0.45], bins, 0.3).toarray()[0]
    pointwise = Likelihoods.from_samples(samples, bins, interim).toarray().mean(axis=0)
    per_bin = (
        Likelihoods.from_samples(
            samples, bins, Density.from_masses(Density.from_function(interim).masses(bins))
        )
        .toarray()
        .mean(axis=0)
    )
    np.testing.assert_allclose(pointwise, exact, atol=0.01)  # MC error ~ 0.002
    assert abs(per_bin[1] - exact[1]) > 0.05  # the per-bin division is biased here


def test_likelihood_samples_are_averaged_in_the_measure_coordinate():
    rng = np.random.default_rng(1)
    bins = Bins.log(1.0, 100.0, 4)
    samples = [stats.norm(7.0, 3.0).rvs(40000, random_state=rng) for _ in range(10)]
    est = Likelihoods.from_samples(samples, bins, Density.flat("z")).toarray().mean(axis=0)
    exact = Likelihoods.from_gaussian([7.0], bins, [3.0], tol=1e-8).toarray()[0]
    np.testing.assert_allclose(est, exact, atol=0.01)


def test_a_floor_keeps_empty_bins_alive():
    bins = Bins.linear(0, 1, 4)
    samples = [np.full(50, 0.1)]
    bare = Likelihoods.from_samples(samples, bins, "flat", min_samples=1)
    floored = Likelihoods.from_samples(samples, bins, "flat", floor=1.0, min_samples=1)
    assert np.all(bare.toarray()[0, 1:] == 0)
    # counts (50 + 0.25) / 0.25 = 201 in the sampled bin, 0.25 / 0.25 = 1 elsewhere
    np.testing.assert_allclose(floored.toarray()[0], [1.0, 1 / 201, 1 / 201, 1 / 201], rtol=1e-12)


def test_monte_carlo_variance_falls_like_one_over_the_sample_size():
    rng = np.random.default_rng(2)
    bins = Bins.linear(0, 1, 5)
    rho = np.full(5, 0.2)
    v = []
    for n in (100, 10000):
        s = rng.normal(0.5, 0.2, size=(30, n))
        lk = Likelihoods.from_samples(s, bins, "flat", warn_outside=False)
        var, n_eff = lk.sample_variance(rho)
        v.append(var.mean())
        np.testing.assert_allclose(n_eff, n * (1 - lk.meta["outside_fraction"]), rtol=1e-12)
    assert 50 < v[0] / v[1] < 200


def test_check_warns_when_sample_noise_is_large():
    rng = np.random.default_rng(3)
    bins = Bins.linear(0, 1, 10)
    s = rng.uniform(0.0, 1.0, size=(2000, 3))  # 3 samples per object
    with pytest.warns(InputWarning, match="fewer than"):
        lk = Likelihoods.from_samples(s, bins, "flat", floor=0.5)
    res = binference.infer(lk, method="gibbs", n_samples=50, n_warmup=50, rng=0, check=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        msgs = res.check()
    assert any("Monte Carlo" in m for m in msgs)
    assert any(
        issubclass(w.category, InputWarning) and "Monte Carlo" in str(w.message) for w in caught
    )


def test_the_scale_of_sample_weights_changes_neither_the_floor_nor_the_variance():
    rng = np.random.default_rng(4)
    bins = Bins.linear(0, 1, 5)
    s = rng.normal(0.5, 0.2, size=(20, 200))
    kw = {"floor": 1.0, "warn_outside": False}
    unit = Likelihoods.from_samples(s, bins, "flat", sample_weights=np.ones_like(s), **kw)
    norm = Likelihoods.from_samples(s, bins, "flat", sample_weights=np.full(s.shape, 0.005), **kw)
    np.testing.assert_allclose(norm.toarray(), unit.toarray(), rtol=1e-12)
    rho = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
    np.testing.assert_allclose(norm.sample_variance(rho)[0], unit.sample_variance(rho)[0])
    assert unit.sample_variance(rho)[0].min() > 0


def test_a_floored_object_without_samples_in_the_bins_has_no_sampling_error():
    bins = Bins.linear(0, 1, 5)
    samples = [np.linspace(0.05, 0.95, 50), np.full(10, 5.0)]
    lk = Likelihoods.from_samples(
        samples, bins, "flat", floor=1.0, min_samples=0, warn_outside=False
    )
    var, _ = lk.sample_variance(np.full(5, 0.2))
    assert np.isfinite(var).all() and var[1] == 0.0


def test_an_interim_distribution_is_divided_out_in_its_far_tail():
    rng = np.random.default_rng(5)
    bins = Bins.linear(30, 45, 3)
    # the pdf underflows to zero here; the log-pdf does not
    lk = Likelihoods.from_samples(
        [rng.uniform(36, 40, 200)], bins, stats.norm(0, 1), min_samples=0
    )
    np.testing.assert_array_equal(lk.toarray()[0] > 0, [False, True, False])


def test_an_interim_with_only_a_cdf_is_divided_out_per_bin():
    class CdfOnly:
        def cdf(self, z):
            return stats.uniform(0, 1).cdf(z)

    rng = np.random.default_rng(6)
    bins = Bins.linear(0, 1, 4)
    s = [rng.uniform(0, 1, 400)]
    via_cdf = Likelihoods.from_samples(s, bins, CdfOnly()).toarray()
    np.testing.assert_allclose(via_cdf, Likelihoods.from_samples(s, bins, "flat").toarray())


def test_a_transform_measure_works_on_an_axis_of_small_values():
    m = binference.Measure.transform(np.sqrt, np.square)
    z = np.array([1e-9, 1e-8, 3e-8])
    np.testing.assert_allclose(m.log_jacobian(z), np.log(0.5 / np.sqrt(z)), rtol=1e-6)


def test_samples_are_stored_sparsely_only_below_the_measured_fill():
    # sparse storage pays only at low fill (DECISIONS #77); the broad case is 28% full
    rng = np.random.default_rng(0)
    bins = binference.Bins.linear(0, 1, 100)
    narrow = rng.uniform(0.1, 0.9, 200)[:, None] + 0.005 * rng.standard_normal((200, 50))
    broad = rng.uniform(0.3, 0.7, 200)[:, None] + 0.05 * rng.standard_normal((200, 400))
    a = binference.Likelihoods.from_samples(narrow, bins, interim="flat")
    b = binference.Likelihoods.from_samples(broad, bins, interim="flat")
    fill = binference._kernels.SPARSE_FILL
    assert a.is_sparse and a.nnz < fill * a.N * a.K
    assert not b.is_sparse and b.nnz > fill * b.N * b.K


def test_integer_sample_weights_of_ragged_samples_act_as_repeated_samples():
    rng = np.random.default_rng(6)
    bins = Bins.linear(0, 1, 5)
    samples = [rng.uniform(0, 1, 30), rng.uniform(0, 1, 70)]  # ragged
    sw = [rng.integers(1, 5, 30).astype(float), rng.integers(1, 5, 70).astype(float)]
    kw = {"min_samples": 0, "warn_outside": False, "sparse": False}
    weighted = Likelihoods.from_samples(samples, bins, "flat", sample_weights=sw, **kw)
    repeated = Likelihoods.from_samples(
        [np.repeat(s, w.astype(int)) for s, w in zip(samples, sw, strict=True)], bins, "flat", **kw
    )
    unweighted = Likelihoods.from_samples(samples, bins, "flat", **kw)

    def shape(lk):  # rows up to their scale
        L = lk.toarray()
        return L / L.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(shape(weighted), shape(repeated), rtol=1e-12)
    assert not np.allclose(shape(weighted), shape(unweighted), rtol=0.05)


def test_samples_padded_with_nan_are_refused_with_a_pointer_to_ragged_lists():
    padded = np.array([[0.2, 0.3, np.nan, np.nan], [0.1, 0.4, 0.6, 0.8]])
    with pytest.raises(ValueError, match=r"samples must be finite.*list of arrays"):
        Likelihoods.from_samples(padded, Bins.linear(0, 1, 2), "flat", min_samples=0)


def test_samples_of_a_likelihood_can_say_so_with_kind():
    bins = Bins.linear(0, 1, 5)
    x = np.random.default_rng(0).uniform(0, 1, (4, 50))
    by_kind = Likelihoods.from_samples(x, bins, kind="likelihood")
    by_interim = Likelihoods.from_samples(x, bins, interim=binference.Density.flat("z"))
    np.testing.assert_array_equal(by_kind.toarray(), by_interim.toarray())


def test_a_kde_with_a_vanishing_bandwidth_is_the_histogram():
    bins = Bins.log(1.0, 100.0, 8)  # a measure that is not z, and a non-flat interim
    rng = np.random.default_rng(1)
    x = np.exp(rng.uniform(0.1, 4.5, (6, 80)))
    interim = stats.loguniform(0.5, 200.0)
    hist = Likelihoods.from_samples(x, bins, interim).toarray()
    kde = Likelihoods.from_samples(x, bins, interim, smooth="kde", bandwidth=1e-9).toarray()
    np.testing.assert_allclose(kde, hist, rtol=1e-9, atol=1e-12)


def test_a_kde_is_closer_than_the_histogram_on_bins_narrower_than_the_posteriors():
    rng = np.random.default_rng(2)
    bins, sigma = Bins.linear(0, 1, 100), 0.08
    d = rng.uniform(0.3, 0.7, 300)
    x = d[:, None] + sigma * rng.standard_normal((300, 30))  # inside [0, 1] for these d
    e = bins.edges[0]
    exact = np.diff(stats.norm.cdf((e[None, :] - d[:, None]) / sigma), axis=1)

    def error(lk):
        p = lk.toarray() / lk.toarray().sum(1, keepdims=True)
        big = exact >= 0.01
        return np.sqrt(np.mean(((p - exact) / np.where(big, exact, 1))[big] ** 2))

    hist = error(Likelihoods.from_samples(x, bins, kind="likelihood", warn_outside=False))
    kde = error(
        Likelihoods.from_samples(x, bins, kind="likelihood", smooth="kde", warn_outside=False)
    )
    assert kde < 0.5 * hist  # about 0.3 against 1.1


def test_a_kde_keeps_the_mass_of_samples_at_the_ends_inside_the_bins():
    bins = Bins.linear(0, 1, 4)
    x = np.array([[0.0, 0.001, 0.999, 1.0]])
    lk = Likelihoods.from_samples(
        x, bins, kind="likelihood", smooth="kde", bandwidth=0.3, min_samples=1
    )
    np.testing.assert_allclose(lk.toarray()[0, 0], lk.toarray()[0, 3], rtol=1e-12)


def test_kde_arguments_are_checked():
    x = np.random.default_rng(3).uniform(0, 1, (2, 30))
    bins = Bins.linear(0, 1, 3)
    with pytest.raises(ValueError, match="smooth"):
        Likelihoods.from_samples(x, bins, kind="likelihood", smooth="hist")
    for bad in (0.0, [0.1, 0.2], "scott"):
        with pytest.raises(ValueError, match="bandwidth"):
            Likelihoods.from_samples(x, bins, kind="likelihood", smooth="kde", bandwidth=bad)
    with pytest.raises(ValueError, match="needs smooth"):
        Likelihoods.from_samples(x, bins, kind="likelihood", bandwidth=0.1)
    grid = Bins.grid([0, 0.5, 1], [0, 0.5, 1])
    with pytest.raises(ValueError, match="1-D bins"):
        Likelihoods.from_samples(np.stack([x, x], axis=-1), grid, kind="likelihood", smooth="kde")
    masked = Bins.linear(0, 1, 4, mask=[True, False, True, True])
    with pytest.raises(ValueError, match="without a mask"):
        Likelihoods.from_samples(x, masked, kind="likelihood", smooth="kde")


def test_a_kde_leaves_the_noise_of_the_samples_to_the_histogram():
    bins = Bins.linear(0, 1, 50)
    x = 0.5 + 0.08 * np.random.default_rng(4).standard_normal((40, 30))
    hist = Likelihoods.from_samples(x, bins, kind="likelihood", warn_outside=False)
    kde = Likelihoods.from_samples(x, bins, kind="likelihood", smooth="kde", warn_outside=False)
    rho = np.full(50, 1 / 50)
    for a, b in zip(kde.sample_variance(rho), hist.sample_variance(rho), strict=True):
        np.testing.assert_allclose(a, b)


def test_a_kde_runs_on_the_cells_of_a_basis():
    basis = binference.Basis(Bins.linear(0, 1, 6), order=1)
    x = np.random.default_rng(5).uniform(0.1, 0.9, (3, 200))
    lk = Likelihoods.from_samples(x, basis, kind="likelihood", smooth="kde")
    assert lk.K == basis.K and np.all(lk.toarray() > 0)
