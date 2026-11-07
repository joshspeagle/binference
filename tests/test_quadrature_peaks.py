"""Adaptive bin averages keep what a coarse cell read (issue #179)."""

import numpy as np
import pytest
from scipy.stats import norm

from binference import Bins, InputWarning
from binference._quadrature import bin_log_average


def _gaussian(sigma):
    return lambda z: norm.logpdf(z, 0.5, sigma)[None]


@pytest.mark.parametrize("sigma", [1e-4, 1e-3, 3e-3, 1e-2])
def test_a_peak_on_the_split_point_keeps_its_mass_to_tol(sigma):
    mass = np.exp(bin_log_average(_gaussian(sigma), Bins.linear(0, 1, 1)))
    assert mass[0, 0] == pytest.approx(1.0, abs=1e-3)


def test_a_peak_in_the_middle_of_a_bin_stays_in_that_bin():
    mass = 1.0 * np.exp(bin_log_average(_gaussian(1e-3), Bins.linear(0, 2, 2)))
    assert mass[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert mass[0, 1] < 1e-3


def test_a_peak_read_per_row_keeps_its_mass_to_tol():
    logf = lambda z, row: norm.logpdf(z, 0.5, 1e-3)  # noqa: E731
    mass = np.exp(bin_log_average(logf, Bins.linear(0, 1, 1), rows=3))
    np.testing.assert_allclose(mass[:, 0], 1.0, atol=1e-3)


def test_an_overshooting_coarse_read_does_not_loosen_the_other_bins():
    # bin 0's first reading of the peak is about 11 times its mass; a reference that kept
    # it would accept bin 1's error at 11 times tol
    def logf(z):
        return np.logaddexp(norm.logpdf(z, 0.5, 0.01), norm.logpdf(z, 1.5, 0.03))[None]

    mass = np.exp(bin_log_average(logf, Bins.linear(0, 2, 2)))[0]
    np.testing.assert_allclose(mass, 1.0, atol=2e-3)


def test_a_peak_too_narrow_for_max_depth_warns():
    with pytest.warns(InputWarning, match="did not reach"):
        bin_log_average(_gaussian(1e-7), Bins.linear(0, 1, 1))


def test_a_box_that_only_the_coarse_cell_read_warns():
    def logf(z):
        return np.where(np.abs(z - 0.5) < 1e-3, np.log(500.0), -np.inf)[None]

    with pytest.warns(InputWarning, match="read zero"):
        bin_log_average(logf, Bins.linear(0, 1, 1))
