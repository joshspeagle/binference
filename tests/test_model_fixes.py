"""Tests for the model fixes of DECISIONS #15, #56, #57 and audit M-1 to M-4, M-7, M-14."""

import warnings

import numpy as np
import pytest

import binference
from binference import Bins, ConvergenceWarning, Likelihoods, PopulationModel, Selection
from binference import priors as P
from binference._simplex import helmert_basis, softmax
from binference.samplers import MAP, Bootstrap
from binference.samplers.base import em_update


def _one_object_bin(n=40, K=5, seed=0):
    # n objects spread over bins 0..K-2 and one object alone in bin K-1
    rng = np.random.default_rng(seed)
    L = rng.uniform(0.0, 0.1, size=(n + 1, K))
    L[np.arange(n), rng.integers(0, K - 1, size=n)] = 1.0
    L[:n, K - 1] = 0.0
    L[n] = 0.0
    L[n, K - 1] = 1.0
    return L


def test_a_zero_weight_row_is_the_same_as_no_row():
    L = _one_object_bin()
    w = np.ones(L.shape[0])
    w[-1] = 0.0
    m = PopulationModel(Likelihoods(L, Bins.categories(5)), P.Dirichlet(1.0))
    with_zero = MAP(m._with_weights(w), algorithm="em").run().value
    dropped = PopulationModel(Likelihoods(L[:-1], Bins.categories(5)), P.Dirichlet(1.0))
    without = MAP(dropped, algorithm="em").run().value
    np.testing.assert_allclose(with_zero, without, atol=1e-6)
    assert with_zero[-1] < 1e-6  # not reset to the uniform vector


def test_a_rate_bootstrap_never_collapses_to_zero():
    L = _one_object_bin()
    m = PopulationModel(Likelihoods(L, Bins.categories(5)), P.GammaRates(1.0, 1e-3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = Bootstrap(m, rng=3).run(60)
    totals = res.rates.sum(axis=1)
    assert np.all(totals > 0) and np.all(np.isfinite(res.samples))


def test_em_stops_with_a_warning_where_an_object_becomes_impossible():
    L = _one_object_bin(n=8, K=4)
    w = np.ones(L.shape[0])
    w[-1] = 0.3  # alone in its bin, clipped to zero by a concentration below one
    with pytest.warns(binference.InputWarning, match="pseudo-likelihood"):
        lk = Likelihoods(L, Bins.categories(4), weights=w, fractional_weights=True)
    m = PopulationModel(lk, P.Dirichlet(0.5))
    for accelerate in (True, False):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # also warns that the prior is unbounded
            est = MAP(m, space="simplex", accelerate=accelerate).run()
        assert any(
            issubclass(w.category, ConvergenceWarning) and "zero likelihood" in str(w.message)
            for w in caught
        )
        assert np.isfinite(est.log_posterior) and not np.allclose(est.value, 0.25)


def test_missing_information_is_nan_where_a_sparse_prior_outweighs_the_data():
    mock = binference.simulate.mock_1d(5, Bins.linear(0, 3, 10), rng=7)
    m = PopulationModel(mock.likelihoods, P.Dirichlet(0.3))
    assert np.isnan(m.missing_information(np.full(10, 0.1), include_prior=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MAP(m, space="simplex", max_iter=3).run()  # warns, does not crash


def _em_jacobian_radius(m, value, h=1e-6):
    # spectral radius of the EM map at its fixed point, in isometric log-ratios (fractions,
    # which need not be normalised: V's columns sum to zero) or log-rates
    if m.kind == "rates":
        to_x, to_value = np.log, np.exp
    else:
        V = helmert_basis(m.K)
        to_x, to_value = (lambda v: V.T @ np.log(v)), (lambda x: softmax(V @ x))

    def F(x):
        return to_x(em_update(m, to_value(x)))

    x0 = to_x(value)
    J = np.column_stack([(F(x0 + h * e) - F(x0 - h * e)) / (2 * h) for e in np.eye(x0.size)])
    return float(np.max(np.abs(np.linalg.eigvals(J))))


@pytest.mark.parametrize("mode", [None, "on_truth", "on_measurement"])
@pytest.mark.parametrize(
    "prior",
    [P.Dirichlet(1.0), P.Dirichlet(50.0), P.GammaRates(1.0, 1e-9), P.GammaRates(200.0, 1.0)],
    ids=repr,
)
def test_missing_information_is_the_rate_of_em(mode, prior):
    # an interior estimate: at a fraction of zero the EM rate there is that of an active
    # constraint, not of missing information
    mock = binference.simulate.mock_1d(
        1500, Bins.linear(0, 3, 6), density="uniform", noise=0.25, rng=1
    )
    sel = None if mode is None else Selection(np.linspace(1.0, 0.3, 6), mode)
    m = PopulationModel(mock.likelihoods, prior, selection=sel)
    mode_value = MAP(m, algorithm="em", max_iter=100000).run().value
    f = m.missing_information(mode_value, include_prior=True)
    np.testing.assert_allclose(f, _em_jacobian_radius(m, mode_value), atol=1e-4)


def test_the_kkt_gap_is_infinite_where_an_object_is_impossible():
    m = PopulationModel(Likelihoods(np.eye(3), Bins.categories(3)))
    assert m.kkt_gap(np.array([0.0, 0.5, 0.5])) == np.inf


def test_the_rates_kkt_gap_does_not_depend_on_the_units_of_the_exposure():
    mock = binference.simulate.mock_1d(800, Bins.linear(0, 3, 6), rng=2)

    def model(unit):
        sel = Selection(np.full(6, unit), "on_truth", scale="exposure")
        return PopulationModel(mock.likelihoods, P.GammaRates(1.0, 1e-9 * unit), selection=sel)

    lam = 1.01 * MAP(model(1.0)).run().value  # the same point, in each unit of exposure
    gaps = [model(unit).kkt_gap(lam / unit) for unit in (1e-3, 1.0, 1e3)]
    np.testing.assert_allclose(gaps, gaps[1], rtol=1e-9)
    assert gaps[1] > 1e-3  # a 1% perturbation is visible


def test_the_rates_start_is_the_observed_composition_over_the_exposure():
    from binference.samplers.base import interior_start

    # broad likelihoods, so the composition must be fitted on the columns of L over S: for
    # seeds 0-19 of this mock the start's log posterior is 0.2-2.2 below the mode's (26-47
    # when fitted on L)
    lk = binference.simulate.mock_1d(1000, Bins.linear(0, 3, 6), noise=0.8, rng=3).likelihoods
    S = np.linspace(0.9, 0.3, 6)
    for mode in ("on_truth", "on_measurement"):
        m = PopulationModel(lk, P.GammaRates(1.0, 1e-9), selection=Selection(S, mode))
        start, mle = interior_start(m), MAP(m).run().value
        np.testing.assert_allclose(start.sum(), mle.sum(), rtol=0.1)
        assert m.log_posterior(mle) - m.log_posterior(start) < 8.0


def test_a_gp_prior_works_on_two_bins():
    mock = binference.simulate.mock_1d(200, Bins.linear(0, 3, 2), rng=4)
    est = binference.infer(mock.likelihoods, P.LogisticGP(), method="map")
    assert np.isfinite(est.value).all()


def test_a_weighted_copy_describes_its_own_rows():
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 6), rng=5)
    m = PopulationModel(mock.likelihoods)
    w = np.random.default_rng(0).poisson(1.0, m.N).astype(float)
    mw = m._with_weights(w)
    assert mw.N == np.count_nonzero(w) == mw._L.shape[0] and mw.W == w.sum()
    value = MAP(mw).run().value
    assert 0 <= mw.missing_information(value, max_objects=100) <= 1  # subsamples its own rows


def test_squarem_rejects_an_extrapolation_that_makes_an_object_impossible():
    m = PopulationModel(Likelihoods(np.eye(3) + 0.1, Bins.categories(3)))
    target = np.array([0.2, 0.3, 0.5])
    calls = []

    def F(x, s):  # an EM-like contraction; every extrapolated trial meets an impossible object
        calls.append(x)
        return None if len(calls) > 2 else 0.5 * (x + target)

    v = np.array([0.6, 0.3, 0.1])
    new, f, s = MAP(m)._squarem_step(m, F, v, m._L @ v, fractions=True)
    r2 = 0.5 * (0.5 * (v + target) + target)  # the two plain EM steps
    np.testing.assert_allclose(new, r2)
    assert np.isfinite(f) and len(calls) > 3  # trials were made and rejected
    np.testing.assert_allclose(s, m._L @ r2)  # the evidences at the point returned
