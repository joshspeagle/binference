"""Tests for binned measurements and classifier routes (DECISIONS #27, #29)."""

import pickle
import warnings

import numpy as np
import pytest
from scipy import stats

import binference
from binference import Bins, InputWarning, Likelihoods, Response


def _route_c(mock, shift=0.0):
    return Likelihoods.from_binned(mock.response, values=mock.measurements["feature"] + shift)


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
    for resp in (
        binference.simulate.mock_unfolding(500, rng=3).response,
        binference.simulate.mock_categorical(100, 3, rng=3).response,
    ):
        with pytest.raises(AttributeError):
            resp.origin = "x"
        with pytest.raises(ValueError):
            resp.matrix[0, 0] = 1.0
        back = pickle.loads(pickle.dumps(resp))
        np.testing.assert_array_equal(back.matrix, resp.matrix)
        assert not back.matrix.flags.writeable
        assert (back.origin, back.open_ended) == (resp.origin, resp.open_ended)
        assert (back.generated is None) == (resp.generated is None)


def _measured_count_messages(lk):
    res = binference.infer(lk, method="gibbs", n_samples=300, n_warmup=300, rng=0, check=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = [m for m in res.check() if "measured counts" in m]
    if found:  # a problem with the inputs, not with the sampler
        assert any(
            issubclass(w.category, InputWarning) and "measured counts" in str(w.message)
            for w in caught
        )
    return found, res


def test_the_predictive_check_flags_label_shift():
    # power at a 0.5 shift with 20,000 objects: 20 of 20 replicates gave p < 1e-3;
    # without a shift none of 40 replicates of 4,000 objects did
    mock = binference.simulate.mock_categorical(20000, 3, n_validation=20000, rng=4)
    good, res = _measured_count_messages(_route_c(mock))
    assert good == []
    assert "response from labelled data" in res.summary().title
    bad, _ = _measured_count_messages(_route_c(mock, shift=0.5))
    assert len(bad) == 1


def test_the_predictive_p_value_matches_the_pseudo_inverse_form():
    rng = np.random.default_rng(7)
    J, K = 8, 3
    counts = rng.integers(20, 200, size=(J, K)).astype(float)
    generated = counts.sum(axis=0) / 0.8
    resp = Response.from_counts(counts, generated, Bins.categories(K))
    q0 = resp.matrix @ [0.2, 0.3, 0.5]
    n = rng.multinomial(2000, q0 / q0.sum()).astype(float)
    rho = np.array([0.25, 0.25, 0.5])
    R, N = resp.matrix, n.sum()
    q = R @ rho / (R.sum(0) @ rho)
    c = rho**2 / generated / (R.sum(0) @ rho) ** 2
    A = np.eye(J) - np.outer(q, np.ones(J))  # derivative of q in the columns of R
    cov = N * (np.diag(q) - np.outer(q, q)) + N**2 * A @ (np.diag(R @ c) - (R * c) @ R.T) @ A.T
    d = n - N * q
    expected = stats.chi2.sf(d @ np.linalg.pinv(cov) @ d, J - 1)
    np.testing.assert_allclose(resp.predictive_pvalue(n, [rho]), expected, rtol=1e-8)


def test_attaching_other_likelihoods_reruns_the_predictive_check():
    mock = binference.simulate.mock_categorical(20000, 3, n_validation=20000, rng=4)
    _, res = _measured_count_messages(_route_c(mock))
    res.attach(_route_c(mock, shift=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert any("measured counts" in m for m in res.check())


def test_open_ended_bins_tally_and_measure_by_the_same_rule():
    rng = np.random.default_rng(8)
    z = rng.uniform(0, 1, 5000)
    m = z + 0.3 * rng.standard_normal(z.size)  # many values beyond the measured edges
    truth, measured = Bins.linear(0, 1, 3), Bins.linear(0, 1, 5)
    resp = Response.from_simulation(z, m, truth, measured, open_ended=True)
    np.testing.assert_allclose(resp.efficiency, 1.0)
    lk = Likelihoods.from_binned(resp, values=m)
    np.testing.assert_allclose(lk.W, z.size)


def test_simulation_weights_are_tallied_as_effective_events():
    rng = np.random.default_rng(9)
    z = rng.uniform(0, 1, 4000)
    m = z + 0.1 * rng.standard_normal(z.size)
    w = rng.uniform(0.5, 1.5, z.size)
    truth, measured = Bins.linear(0, 1, 3), Bins.linear(-0.5, 1.5, 6)
    a = Response.from_simulation(z, m, truth, measured, weights=w)
    b = Response.from_simulation(z, m, truth, measured, weights=1e3 * w)
    np.testing.assert_allclose(b.matrix, a.matrix)
    np.testing.assert_allclose(b.generated, a.generated)
    assert np.all(a.generated < np.bincount(truth.index(z), minlength=3))


def test_labelled_objects_need_finite_scores():
    with pytest.raises(ValueError, match="scores must be finite"):
        Response.from_labelled([0.1, np.nan, 0.3, 0.8, 0.5, 0.2], [0, 1, 0, 1, 1, 0])


def test_an_efficiency_must_be_probabilities():
    with pytest.raises(ValueError, match="efficiency"):
        Response(np.eye(2), Bins.categories(2), efficiency=[0.5, np.nan])


def test_a_migration_matrix_off_one_by_rounding_gets_exactly_its_efficiency():
    # the first column sums to 1.0005
    resp = Response([[0.5005, 0.3], [0.5, 0.7]], Bins.categories(2), efficiency=[1.0, 0.9])
    np.testing.assert_allclose(resp.efficiency, [1.0, 0.9], rtol=1e-12)


def test_more_tallied_than_generated_events_name_the_truth_bins():
    with pytest.raises(ValueError, match=r"more events than generated in truth bin\(s\) \[0\]"):
        Response.from_counts([[6, 1], [5, 1]], [10, 10], Bins.categories(2))
    with pytest.raises(ValueError, match="must have shape"):  # not a count per truth bin
        Response.from_counts([6, 7], [10, 10], Bins.categories(2))


def test_open_ended_needs_one_dimensional_measured_bins_whatever_the_constructor():
    with pytest.raises(ValueError, match="open_ended needs 1-D"):
        Response.from_simulation(
            [0.1, 0.6], [0, 1], Bins.linear(0, 1, 2), Bins.categories(2), open_ended=True
        )


def test_a_truth_bin_the_response_never_observes_is_named():
    resp = Response(np.array([[0.5, 0, 0.3], [0.5, 0, 0.3]]), Bins.categories(3))
    with pytest.raises(ValueError, match=r"truth bin\(s\) \[1\] sum to zero"):
        Likelihoods.from_binned(resp, counts=[3, 4])
