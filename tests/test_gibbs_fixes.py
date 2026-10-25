"""Tests for the Gibbs, bootstrap and validation fixes of audit G-1 to G-12 (DECISIONS #75)."""

import time

import numpy as np
import pytest
import scipy.sparse as sp

import binference
from binference import Bins, PopulationModel
from binference import _kernels as kern
from binference.samplers import Bootstrap, Gibbs


def test_weighted_label_counts_on_sparse_likelihoods_stay_sparse_and_correct():
    rng = np.random.default_rng(0)
    N, K = 20000, 2500
    cols = rng.integers(0, K, size=(N, 10))
    L = sp.csr_array((rng.random(N * 10), (np.repeat(np.arange(N), 10), cols.ravel())), (N, K))
    rho = np.full(K, 1.0 / K)
    w = np.full(N, 2.0)
    t = time.perf_counter()
    counts, _ = kern.sample_counts(L, w, rho, rng, unit_weights=False)
    # the dense path took 13 s and 1.5 GB here (audit G-2)
    assert time.perf_counter() - t < 2.0
    assert counts.sum() == w.sum()
    # each count is binomial: its expectation from the responsibilities, SD below 7
    expected = (w[:, None] * kern.responsibilities(L, rho)).sum(axis=0)
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


def test_bootstrap_counts_replicates_that_did_not_converge():
    m = PopulationModel(
        binference.simulate.mock_1d(500, Bins.linear(0, 3, 6), rng=2).likelihoods,
        binference.priors.Dirichlet(2.0),
    )
    with (
        pytest.warns(binference.ConvergenceWarning, match="did not converge"),
        pytest.warns(binference.ApproximationWarning),  # always: not a posterior
    ):
        res = Bootstrap(m, max_iter=1, rng=0).run(5)
    assert res.info["n_not_converged"] == 5
