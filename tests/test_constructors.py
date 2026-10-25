"""Tests for the constructors by input form (DECISIONS #32), adaptive quadrature (#23),
measure-aware averages (#21) and mass outside the support (#18)."""

import numpy as np
import pytest
from scipy import integrate, stats

from binference import Bins, InputWarning, Likelihoods


def test_adaptive_averages_of_narrow_gaussians_match_the_exact_ones():
    bins = Bins.linear(0, 1, 10)
    mu, sig = np.array([0.33, 0.71]), np.array([0.002, 0.01])  # 2% and 10% of a bin
    exact = Likelihoods.from_gaussian(mu, sig, bins).toarray()
    logf = lambda z: stats.norm.logpdf(z[None, :], mu[:, None], sig[:, None])  # noqa: E731
    adaptive = Likelihoods.from_density(logf, bins, kind="likelihood", log=True, tol=1e-5)
    np.testing.assert_allclose(adaptive.toarray(), exact, atol=1e-5)


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
        for a, b in zip(e[:-1], e[1:], strict=True)
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
