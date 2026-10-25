"""Tests for selection functions (DECISIONS #7, #8, #10, #46, #47, #48)."""

import pickle
import warnings

import numpy as np
import pytest

import binference
from binference import Bins, InputWarning, PopulationModel, Selection
from binference import priors as P
from binference.simulate import mock_1d


def test_the_old_mode_names_point_at_the_new_ones():
    with pytest.raises(ValueError, match="'on_truth'"):
        Selection([0.5, 1.0], "true")
    with pytest.raises(ValueError, match="'on_measurement'"):
        Selection([0.5, 1.0], "observed")


def test_probabilities_above_one_need_a_declared_scale():
    with pytest.raises(ValueError, match="scale='exposure'"):
        Selection([0.5, 2.0], "on_truth")
    assert Selection([0.5, 2.0], "on_truth", scale="exposure").scale == "exposure"
    with pytest.raises(ValueError, match="DECISIONS #46"):
        binference.simulate.mock_1d(100, Bins.linear(0, 3, 2), selection=[2.0, 1.0], rng=0)


def test_rates_refuse_a_relative_selection():
    mock = mock_1d(200, Bins.linear(0, 3, 3), rng=1)
    sel = Selection([1.0, 2.0, 3.0], "on_truth", scale="relative")
    with pytest.raises(ValueError, match="absolute"):
        PopulationModel(mock.likelihoods, P.GammaRates(1.0, 1e-3), selection=sel)
    PopulationModel(mock.likelihoods, selection=sel)  # fractions need only ratios


def test_a_function_is_averaged_over_each_bin_in_the_measure_coordinate():
    lin = Selection.from_function(lambda z: z / 2, Bins.linear(0, 2, 2), "on_truth")
    np.testing.assert_allclose(lin.values, [0.25, 0.75], rtol=1e-8)
    log = Selection.from_function(np.log, Bins.log(1.0, np.e**2, 2), "on_truth", scale="exposure")
    np.testing.assert_allclose(log.values, [0.5, 1.5], rtol=1e-6)  # the mean of u = log z


def test_combined_selection_scales_rows_by_the_truth_part_only():
    sel = Selection.combined([0.5, 1.0], [0.8, 0.4])
    np.testing.assert_allclose(sel.row, [0.5, 1.0])
    np.testing.assert_allclose(sel.values, [0.4, 0.4])
    bins = Bins.linear(0, 2, 2)
    f = Selection.combined(lambda z: z / 2, lambda z: 1 - z / 2, bins)
    np.testing.assert_allclose(f.row, [0.25, 0.75], rtol=1e-8)
    np.testing.assert_allclose(f.values, [1 / 6, 1 / 6], rtol=1e-8)  # mean of z/2 - z^2/4
    m = PopulationModel(np.ones((3, 2)), bins=bins, selection=sel)
    np.testing.assert_allclose(
        m._L.toarray() if hasattr(m._L, "toarray") else m._L, [[0.5, 1.0]] * 3
    )


def test_a_steep_array_warns_but_a_selection_object_or_function_does_not():
    bins = Bins.linear(0, 3, 3)
    mock = mock_1d(200, bins, rng=2)
    with pytest.warns(InputWarning, match="factor of 10"):
        mock.likelihoods.with_selection([0.05, 0.5, 1.0], "on_truth")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mock.likelihoods.with_selection(Selection([0.05, 0.5, 1.0], "on_truth"))
        mock.likelihoods.with_selection(lambda z: 0.05 + 0.3 * z, "on_truth")


def test_a_selection_is_immutable_and_survives_pickling():
    sel = Selection.combined([0.5, 1.0], [0.8, 0.4], scale="relative")
    with pytest.raises(AttributeError):
        sel.mode = "on_truth"
    with pytest.raises(ValueError):
        sel.values[0] = 1.0
    back = pickle.loads(pickle.dumps(sel))
    assert back == sel and back.scale == "relative" and not back.row.flags.writeable
    truth = pickle.loads(pickle.dumps(Selection([0.5, 1.0], "on_truth", scale="exposure")))
    assert truth.mode == "on_truth" and truth.scale == "exposure"


def test_the_pass_probabilities_of_complementary_cuts_add_to_one():
    from binference.simulate import _measured_pass

    z = np.array([0.1, 1.0, 2.0])
    below = _measured_pass(z, (0.05, 0.5), -np.inf, 1.0)
    above = _measured_pass(z, (0.05, 0.5), 1.0, np.inf)
    np.testing.assert_allclose(below + above, 1.0, rtol=1e-12)
    assert np.all((below > 0) & (above > 0))


_S = [1.0, 0.8, 0.6, 0.45, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]


@pytest.mark.parametrize(
    "kw",
    [
        {"selection": _S},
        {"measured_range": (0.5, None)},
        {"selection": _S, "measured_range": (0.5, None)},
    ],
    ids=["on_truth", "on_measurement", "combined"],
)
def test_mocks_with_each_kind_of_selection_recover_the_population(kw):
    from binference.simulate import population

    bins = Bins.linear(0, 3, 10)
    rho = population(bins, "bimodal")
    mock = mock_1d(20000, bins, density=rho, noise=(0.02, 1.0), rng=3, **kw)
    fit = binference.infer(mock.likelihoods, method="laplace", rng=0, check=False)
    sd = fit.samples.std(axis=0)
    naive = binference.infer(mock.likelihoods.with_selection(None), method="map").value
    # z-scores against the truth: below 2.5 in 12 trial fits; ignoring selection, 11 to 23
    assert np.abs((fit.mean - rho) / sd).max() < 4
    assert np.abs((naive - rho) / sd).max() > 8
