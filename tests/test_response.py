"""Tests for binned measurements and classifier routes (DECISIONS #27, #29)."""

import pickle
import warnings

import numpy as np
import pytest
from scipy import stats

import binference
from binference import Bins, InputWarning, Likelihoods, Response


def _route_c(mock, shift=0.0):
    m = mock.measurements
    resp = Response.from_labelled(
        m["validation_feature"] + shift, m["validation_labels"], classes=mock.bins.labels
    )
    return Likelihoods.from_binned(resp, values=m["feature"])


def test_score_bins_recover_the_prevalence_when_the_posteriors_are_miscalibrated():
    mock = binference.simulate.mock_categorical(6000, 3, temperature=2.0, n_validation=6000, rng=1)
    a = binference.infer(mock.likelihoods, method="map").value
    c = binference.infer(_route_c(mock), method="map").value
    # route C's error is about 0.01 here (validation and target noise); route A is biased
    np.testing.assert_allclose(c, mock.truth.rho, atol=0.04)
    assert np.abs(a - mock.truth.rho).max() > 0.06


def test_a_simulated_response_matches_the_known_one():
    rng = np.random.default_rng(2)
    truth, measured = Bins.linear(0, 1, 4), Bins.linear(-0.2, 1.2, 7)
    z = rng.uniform(0, 1, 100_000)
    m = z + 0.1 * rng.standard_normal(z.size)
    m[rng.random(z.size) < 0.2] = np.nan  # lost events
    resp = Response.from_simulation(z, m, truth, measured)
    x, w = np.polynomial.legendre.leggauss(32)
    exact = np.empty((7, 4))
    for k in range(4):
        zk = (k + 0.5 * (x + 1)) / 4
        cdf = stats.norm.cdf((measured.edges[0][None, :] - zk[:, None]) / 0.1)
        exact[:, k] = 0.8 * (np.diff(cdf, axis=1) * 0.5 * w[:, None]).sum(axis=0)
    # each entry is a binomial share of about 25,000 events: sd below 0.003
    np.testing.assert_allclose(resp.matrix, exact, atol=0.015)
    np.testing.assert_allclose(resp.generated.sum(), z.size)
    assert resp.origin == "simulation"


def test_predicted_labels_give_the_confusion_matrix():
    labels = np.array(["a", "a", "a", "b", "b"])
    predicted = np.array(["a", "a", "b", "b", "b"])
    resp = Response.from_labelled(predicted, labels)
    np.testing.assert_allclose(resp.matrix, [[2 / 3, 0.0], [1 / 3, 1.0]])
    assert resp.measured_bins.labels == ("a", "b")


def test_observed_events_from_outside_the_truth_range_are_reported():
    with pytest.warns(InputWarning, match="outside the truth bins"):
        Response.from_simulation(
            [0.2, 0.7, 2.0], [0.5, 0.5, 0.5], Bins.linear(0, 1, 2), Bins.linear(0, 1, 2)
        )
    with pytest.raises(ValueError, match="truth bin"):
        Response.from_simulation([0.2], [0.2], Bins.linear(0, 1, 2), Bins.linear(0, 1, 2))


def test_a_response_is_immutable_and_survives_pickling():
    resp = binference.simulate.mock_unfolding(500, rng=3).response
    with pytest.raises(AttributeError):
        resp.origin = "x"
    with pytest.raises(ValueError):
        resp.matrix[0, 0] = 1.0
    back = pickle.loads(pickle.dumps(resp))
    np.testing.assert_array_equal(back.matrix, resp.matrix)
    assert not back.matrix.flags.writeable


def _measured_count_messages(lk):
    res = binference.infer(lk, method="gibbs", n_samples=300, n_warmup=300, rng=0, check=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [m for m in res.check() if "measured counts" in m], res


def test_the_predictive_check_flags_label_shift():
    # power at a 0.5 shift with 20,000 objects: 20 of 20 replicates gave p < 1e-3;
    # without a shift none of 40 replicates of 4,000 objects did
    mock = binference.simulate.mock_categorical(20000, 3, n_validation=20000, rng=4)
    good, res = _measured_count_messages(_route_c(mock))
    assert good == []
    assert "response from labelled data" in res.summary().title
    bad, _ = _measured_count_messages(_route_c(mock, shift=0.5))
    assert len(bad) == 1
