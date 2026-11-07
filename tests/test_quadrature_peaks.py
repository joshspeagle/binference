"""Adaptive bin averages keep what a coarse cell read."""

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
    mass = np.exp(bin_log_average(_gaussian(1e-3), Bins.linear(0, 2, 2)))
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


def _spike_on(background):
    return lambda z: np.logaddexp(np.log(0.5) + norm.logpdf(z, 0.5, 1e-3), background(z))[None]


def test_a_spike_above_a_broad_background_keeps_its_mass():
    logf = _spike_on(lambda z: np.full_like(z, np.log(0.5)))
    assert np.exp(bin_log_average(logf, Bins.linear(0, 1, 1)))[0, 0] == pytest.approx(
        1.0, abs=2e-3
    )


def test_a_spike_stays_in_its_bin_when_another_bin_sets_the_row_scale():
    logf = _spike_on(lambda z: norm.logpdf(z, 1.5, 0.03))
    mass = np.exp(bin_log_average(logf, Bins.linear(0, 2, 2)))[0]
    np.testing.assert_allclose(mass, [0.5, 1.0], atol=2e-3)


def test_a_spike_whose_tails_underflow_to_zero_is_still_found():
    def logf(z):
        with np.errstate(divide="ignore"):
            return np.log(norm.pdf(z, 0.5, 1e-4))[None]

    assert np.exp(bin_log_average(logf, Bins.linear(0, 1, 1)))[0, 0] == pytest.approx(
        1.0, abs=1e-3
    )


def test_a_box_that_only_the_coarse_cell_read_is_found():
    def logf(z):
        return np.where(np.abs(z - 0.5) < 1e-3, np.log(500.0), -np.inf)[None]

    assert np.exp(bin_log_average(logf, Bins.linear(0, 1, 1)))[0, 0] > 0.9


def test_a_spike_on_a_two_dimensional_grid_keeps_its_mass():
    def logf(z):
        spike = norm.logpdf(z, 0.5, 1e-2).sum(axis=-1)
        return np.logaddexp(np.log(0.5) + spike, np.log(0.5))[None]

    bins = Bins.grid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert np.exp(bin_log_average(logf, bins))[0, 0] == pytest.approx(1.0, abs=2e-3)


def test_a_negligible_point_value_does_not_force_refinement():
    calls = []

    def logf(z):
        calls.append(np.size(z))
        spike = np.where(np.abs(z - 0.5) < 1e-3, np.log(1e-9), -np.inf)
        return np.logaddexp(spike, 0.0)[None]

    bin_log_average(logf, Bins.linear(0, 1, 1))
    assert sum(calls) == 8  # 5 and 3 points on the one cell


def test_a_spike_only_a_check_rule_reads_is_followed():
    # on the 3-point rule's first node, which the 5-point rule misses
    def logf(z):
        return np.logaddexp(norm.logpdf(z, 0.5 - 0.15**0.5, 1e-4), 0.0)[None]

    assert np.exp(bin_log_average(logf, Bins.linear(0, 1, 1)))[0, 0] == pytest.approx(
        2.0, abs=2e-3
    )


def test_a_background_above_the_share_does_not_confirm_a_tall_bump():
    def logf(z):
        return np.log(1 + 9 * np.exp(-0.5 * ((z - 0.5) / 0.006) ** 2))[None]

    exact = 1 + 9 * 0.006 * (2 * np.pi) ** 0.5
    assert np.exp(bin_log_average(logf, Bins.linear(0, 1, 1)))[0, 0] == pytest.approx(
        exact, abs=2e-3
    )


def test_a_one_sided_feature_on_a_split_point_does_not_warn():
    def logf(z):
        with np.errstate(divide="ignore"):
            return np.where(z >= 0.5, norm.logpdf(z, 0.5, 1e-2) + np.log(2.0), -np.inf)[None]

    mass = np.exp(bin_log_average(logf, Bins.linear(0, 1, 1), jumps=True, max_depth=16))
    assert mass[0, 0] == pytest.approx(1.0, abs=2e-3)


def test_a_claim_that_turns_negligible_stops_without_a_warning():
    def logf(z):
        wide = norm.logpdf(z, 1.5, 0.1)
        return np.logaddexp(wide, np.log(0.01) - 0.5 * ((z - 0.5) / 1e-7) ** 2)[None]

    bin_log_average(logf, Bins.linear(0, 2, 2))  # warnings are errors


def test_memory_stays_bounded_by_the_chunk_size(monkeypatch):
    from binference import _quadrature

    seen = []

    def logf(z):
        seen.append(z.shape[0])
        return np.zeros((2, z.shape[0]))

    monkeypatch.setattr(_quadrature, "MAX_ELEMENTS", 200)
    bin_log_average(logf, Bins.linear(0, 1, 40), n_sub=4)
    assert len(seen) > 2 and max(seen) * 2 <= 200
