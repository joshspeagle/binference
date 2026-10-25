"""Tests for the Gibbs, bootstrap and validation fixes of audit G-1 to G-12 (DECISIONS #75)."""

import time

import numpy as np
import pytest
import scipy.sparse as sp

import binference
from binference import Bins, PopulationModel
from binference import _kernels as kern
from binference.samplers import MAP, Bootstrap, Gibbs


def test_weighted_label_counts_on_sparse_likelihoods_stay_sparse_and_correct():
    rng = np.random.default_rng(0)
    N, K = 20000, 2500
    cols = rng.integers(0, K, size=(N, 10))
    L = sp.csr_array((rng.random(N * 10), (np.repeat(np.arange(N), 10), cols.ravel())), (N, K))
    L = sp.csr_array(sp.vstack([L, np.ones((1, K))]))  # one flat object must not densify the rest
    N += 1
    rho = np.full(K, 1.0 / K)
    w = rng.integers(1, 1000, size=N).astype(float)
    t = time.perf_counter()
    counts, _ = kern.sample_counts(L, w, rho, rng, unit_weights=False)
    # the dense path took 13 s and 1.5 GB here (audit G-2); a label per object exceeded 2 s
    assert time.perf_counter() - t < 2.0
    assert counts.sum() == w.sum()
    # each count is binomial: its expectation from the responsibilities
    expected = rho * (L.T @ (w / (L @ rho)))  # no N x K array
    assert np.max(np.abs(counts - expected) / np.sqrt(expected + 1)) < 6


def _model(n=200, K=4):
    mock = binference.simulate.mock_1d(n, Bins.linear(0, 3, K), rng=1)
    return PopulationModel(mock.likelihoods)


def test_gibbs_inputs_are_checked_when_the_engine_is_made():
    m = _model()
    Gibbs(m, chunk=None).run(2)  # no chunking, as the kernels allow
    with pytest.raises(ValueError, match="store_labels"):
        Gibbs(m, store_labels=[0, 5000])
    with pytest.raises(ValueError, match="init"):
        Gibbs(m, init=[0.5, 0.5, 0.0, 0.0])


def test_gibbs_chains_from_one_init_start_apart():
    eng = Gibbs(_model(), init=[0.25, 0.25, 0.25, 0.25], n_chains=3, rng=0)
    eng._setup()
    starts = np.array(eng._state)
    assert not np.allclose(starts[0], starts[1])  # R-hat needs overdispersed starts


def test_gibbs_starts_correct_for_the_objects_a_selection_missed():
    mock = binference.simulate.mock_1d(
        3000, Bins.linear(0, 3, 3), selection=[1.0, 0.5, 0.2], noise=0.3, rng=3
    )
    m = PopulationModel(mock.likelihoods)
    eng = Gibbs(m, n_chains=20, rng=0)
    eng._setup()
    # starts drawn around the observed composition were 0.1 or more off in a fraction;
    # the start posterior SD is about 0.01, so the mean of 20 is within 0.01
    np.testing.assert_allclose(np.mean(eng._state, axis=0), MAP(m).run().rho, atol=0.02)


def test_bootstrap_counts_replicates_that_did_not_converge():
    m = PopulationModel(
        binference.simulate.mock_1d(500, Bins.linear(0, 3, 6), rng=2).likelihoods,
        binference.priors.Dirichlet(2.0),
    )
    with (
        pytest.warns(binference.ApproximationWarning),  # always: not a posterior
        pytest.warns(binference.ConvergenceWarning) as rec,
    ):
        res = Bootstrap(m, max_iter=1, rng=0).run(5)
    msgs = [str(r.message) for r in rec]
    assert any(s.startswith("EM did not converge") for s in msgs)  # the base fit is not silenced
    assert "5 of 5 bootstrap fits did not converge; raise max_iter." in msgs
    assert res.info["n_not_converged"] == 5
