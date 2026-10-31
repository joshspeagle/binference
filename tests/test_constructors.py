"""Tests for the constructors by input form (DECISIONS #32), adaptive quadrature (#23),
measure-aware averages (#21) and mass outside the support (#18)."""

import warnings
from itertools import pairwise

import numpy as np
import pytest
from scipy import integrate, stats

from binference import Bins, Density, InputWarning, Likelihoods
from binference import _quadrature as quadrature


def test_adaptive_averages_of_narrow_gaussians_match_the_exact_ones():
    bins = Bins.linear(0, 1, 10)
    mu, sig = np.array([0.33, 0.71]), np.array([0.002, 0.01])  # 2% and 10% of a bin
    exact = Likelihoods.from_gaussian(mu, sig, bins).toarray()
    logf = lambda z: stats.norm.logpdf(z[None, :], mu[:, None], sig[:, None])  # noqa: E731
    adaptive = Likelihoods.from_density(logf, bins, kind="likelihood", log=True, tol=1e-5)
    np.testing.assert_allclose(adaptive.toarray(), exact, atol=1e-5)


def test_frozen_distributions_of_one_family_are_evaluated_together(monkeypatch):
    calls = {"logcdf": 0, "logpdf": 0}
    family = type(stats.norm)

    def counting(name):
        method = getattr(family, name)

        def counted(self, *args, **kwargs):
            calls[name] += 1
            return method(self, *args, **kwargs)

        return counted

    for name in calls:
        monkeypatch.setattr(family, name, counting(name))

    def n_calls(dists, bins):
        calls.update(logcdf=0, logpdf=0)
        Likelihoods.from_density(dists, bins, kind="likelihood")
        return calls.copy()

    dists = [stats.norm(m, 0.1) for m in (0.2, 0.5, 0.7)]
    # exact masses on linear bins; quadrature in the measure coordinate on log bins
    for bins in (Bins.linear(0, 1, 4), Bins.log(0.1, 1.0, 4)):
        assert n_calls(dists, bins) == n_calls(dists * 2, bins)


def test_frozen_histograms_are_not_taken_for_one_family():
    # a frozen rv_histogram's generator holds its histogram, so one call would use the first
    data = ([0.1, 0.2, 0.2, 0.3], [0.6, 0.7, 0.7, 0.9])
    hists = [stats.rv_histogram(np.histogram(x, 3)).freeze() for x in data]
    bins = Bins.linear(0, 1, 4)
    lk = Likelihoods.from_density(hists, bins, kind="likelihood")
    one = [Likelihoods.from_density([h], bins, kind="likelihood").toarray()[0] for h in hists]
    np.testing.assert_allclose(lk.toarray(), one)


def test_a_single_distribution_is_refused_with_a_hint():
    with pytest.raises(TypeError, match="a list of one"):
        Likelihoods.from_density(stats.norm(0.5, 0.1), Bins.linear(0, 1, 4), kind="likelihood")


def test_bins_that_do_not_converge_warn():
    bins = Bins.linear(0, 1, 4)
    logf = lambda z: stats.norm.logpdf(z[None, :], 0.3, 0.001)  # noqa: E731
    with pytest.warns(InputWarning, match="did not reach"):
        Likelihoods.from_density(logf, bins, kind="likelihood", log=True, max_depth=0)


def test_likelihoods_on_log_bins_are_averaged_in_log_z():
    bins = Bins.log(1.0, 100.0, 4)
    mu, sig = 7.0, 3.0
    e = bins.edges[0]
    # the bin average under a population flat in log z: integral of L(z) / z dz over the log width
    want = [
        integrate.quad(lambda z: stats.norm.pdf(z, mu, sig) / z, a, b, epsabs=0, epsrel=1e-12)[0]
        / np.log(b / a)
        for a, b in pairwise(e)
    ]
    for lk in (
        Likelihoods.from_gaussian([mu], [sig], bins, tol=1e-8),
        Likelihoods.from_density([stats.norm(mu, sig)], bins, kind="likelihood", tol=1e-8),
    ):
        np.testing.assert_allclose(lk.toarray()[0], np.array(want) / max(want), atol=1e-7)


def test_distributions_far_below_the_bins_keep_finite_rows():
    bins = Bins.linear(0, 1, 10)
    dists = [stats.norm(-0.35, 0.05), stats.norm(-0.6, 0.05)]  # 7 and 12 sigma below
    with pytest.warns(InputWarning, match="outside the bins"):
        via_dist = Likelihoods.from_density(dists, bins, kind="likelihood")
    with pytest.warns(InputWarning, match="outside the bins"):
        exact = Likelihoods.from_gaussian([-0.35, -0.6], 0.05, bins)
    np.testing.assert_allclose(via_dist.toarray(), exact.toarray(), rtol=1e-9, atol=1e-300)
    assert np.all(via_dist.toarray()[:, 1] > 0)


def test_a_logpdf_only_object_far_from_the_bins_keeps_a_finite_row():
    class OnlyLogpdf:
        def logpdf(self, z):
            return stats.norm.logpdf(z, -0.8, 0.02)

    bins = Bins.linear(0, 1, 5)
    lk = Likelihoods.from_density([OnlyLogpdf()], bins, kind="likelihood")
    with pytest.warns(InputWarning, match="outside the bins"):
        exact = Likelihoods.from_gaussian([-0.8], 0.02, bins)
    flat = np.full(5, 0.2)
    np.testing.assert_allclose(
        lk.log_evidence_terms(flat), exact.log_evidence_terms(flat), rtol=1e-6
    )
    assert lk.toarray()[0, 1] > 0


def test_outside_fractions_are_recorded_per_object():
    bins = Bins.linear(0, 1, 4)
    lk = Likelihoods.from_gaussian([0.5, 1.0], 0.1, bins)
    np.testing.assert_allclose(lk.meta["outside_fraction"], [0.0, 0.5], atol=1e-6)


def test_a_peak_the_first_estimate_misses_does_not_warn():
    bins = Bins.log(0.1, 10.0, 50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lk = Likelihoods.from_gaussian([8.588], [0.01], bins)
    assert np.isfinite(lk.log_scale).all()


def test_distributions_without_a_density_are_refused_where_one_is_needed():
    class OnlyCdf:
        def cdf(self, z):
            return stats.norm.cdf(z, 5.0, 1.0)

    with pytest.raises(TypeError, match="logpdf or pdf"):
        Likelihoods.from_density([OnlyCdf()], Bins.log(1.0, 100.0, 5), kind="likelihood")

    class OnlyLogcdf:
        def logcdf(self, z):
            return stats.norm.logcdf(z, 0.5, 0.2)

    lk = Likelihoods.from_density([OnlyLogcdf()], Bins.linear(0, 1, 5), kind="likelihood")
    assert lk.toarray()[0, 2] == 1.0


def test_n_dimensional_distributions_receive_points_as_rows():
    class Blob:
        def logpdf(self, x):
            return -0.5 * np.sum((np.asarray(x) - 0.5) ** 2, axis=1) / 0.01

    grid = Bins.grid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    lk = Likelihoods.from_density([Blob(), Blob()], grid, kind="likelihood")
    assert lk.toarray().shape == (2, 9) and lk.toarray()[0, 4] == 1.0


def test_a_non_positive_tolerance_is_refused():
    with pytest.raises(ValueError, match="tol must be positive"):
        Likelihoods.from_gaussian([5.0], [1.0], Bins.log(1, 10, 4), tol=0)


def test_refining_rows_in_blocks_gives_the_same_averages(monkeypatch):
    bins = Bins.log(1.0, 100.0, 10)
    mu = np.linspace(2, 90, 7)
    whole = Likelihoods.from_gaussian(mu, 0.05 * mu, bins).toarray()
    monkeypatch.setattr(quadrature, "MAX_ELEMENTS", 200)
    blocked = Likelihoods.from_gaussian(mu, 0.05 * mu, bins).toarray()
    np.testing.assert_allclose(blocked, whole, rtol=1e-12)


def test_the_outside_warning_points_at_the_caller():
    with pytest.warns(InputWarning, match="outside the bins") as record:
        Likelihoods.from_gaussian([0.5, 50.0], [0.1, 0.1], Bins.linear(0, 1, 5))
    assert record[0].filename == __file__


def test_density_masses_meet_the_tolerance_on_unequal_bins():
    edges = [1.0, 1.001, 10.0, 100.0, 1000.0]
    m = Density.from_function(lambda z: 1 / z).masses(Bins(edges), tol=1e-5)
    exact = np.diff(np.log(edges)) / np.log(1000.0)
    np.testing.assert_allclose(m, exact, rtol=1e-5)


def test_a_function_with_no_rows_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        Likelihoods.from_density(
            lambda z: np.zeros((0, np.shape(z)[0])), Bins.linear(0, 1, 3), kind="likelihood"
        )


def test_errors_name_the_interim_argument():
    with pytest.raises(ValueError, match=r"^interim must be"):
        Likelihoods.from_table(np.ones((1, 3)), Bins.linear(0, 1, 3), interim="gaussian")
