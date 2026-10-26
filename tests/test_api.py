"""Tests for the high-level API: infer(), PopulationEstimator, namespace (spec 8)."""

from pathlib import Path

import numpy as np
import pytest

import binference
from binference import Bins, PopulationEstimator, PopulationModel, Selection


def test_namespace():
    for name in binference.__all__:
        assert hasattr(binference, name), name
    assert isinstance(binference.__version__, str)
    assert binference.samplers.NUTS and binference.priors.Dirichlet and binference.Selection


def test_citation_file_version_matches_package():
    cff = Path(__file__).parents[1] / "CITATION.cff"
    if not cff.exists():
        pytest.skip("CITATION.cff is not next to the tests (installed package)")
    line = next(ln for ln in cff.read_text().splitlines() if ln.startswith("version:"))
    assert line.split(":", 1)[1].strip().strip("'\"") == binference.__version__


def test_infer_accepts_arrays_likelihoods_models(small_1d):
    raw = small_1d.likelihoods.toarray()
    a = binference.infer(raw, method="map")
    assert a.bins.is_categorical
    b = binference.infer(raw, bins=small_1d.bins, method="map")
    assert b.bins == small_1d.bins
    np.testing.assert_allclose(a.rho, b.rho, atol=1e-8)
    m = PopulationModel(small_1d.likelihoods)
    c = binference.infer(m, method="map")
    np.testing.assert_allclose(c.rho, b.rho, atol=1e-8)
    with pytest.raises(ValueError):
        binference.infer(m, prior=binference.priors.Dirichlet(2.0))


def test_infer_methods_and_errors(small_1d):
    lk = small_1d.likelihoods
    with pytest.raises(ValueError, match="Unknown method"):
        binference.infer(lk, method="hmc")
    with pytest.raises(TypeError):
        binference.infer(lk, method="gibbs", n_chainz=2)
    with pytest.raises(ValueError):
        binference.infer(lk, method="map", n_samples=10)
    res = binference.infer(
        lk, method="auto", rng=1, n_samples=30, n_warmup=30, n_chains=1, check=False
    )
    assert res.method == res.info["auto_method"] and res.info["auto_reason"]
    em = binference.infer(lk, method="em")
    assert em.method == "map"


def test_infer_reproducible(small_1d):
    kw = {"method": "gibbs", "n_samples": 20, "n_warmup": 5, "check": False}
    a = binference.infer(small_1d.likelihoods, rng=7, **kw)
    b = binference.infer(small_1d.likelihoods, rng=7, **kw)
    np.testing.assert_array_equal(a.samples, b.samples)
    with pytest.raises(TypeError):
        binference.infer(small_1d.likelihoods, rng=np.random.RandomState(1), **kw)


def test_estimator_contract(small_1d):
    est = PopulationEstimator(method="gibbs", n_samples=100, rng=1, engine_kwargs={"n_warmup": 20})
    params = est.get_params()
    assert params["method"] == "gibbs" and params["n_samples"] == 100
    clone = PopulationEstimator(**params)
    assert clone.get_params() == params
    est.set_params(n_samples=80)
    assert est.n_samples == 80
    with pytest.raises(ValueError):
        est.set_params(n_sample=3)
    with pytest.raises(RuntimeError):
        est.predict_proba(small_1d.likelihoods)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", binference.ConvergenceWarning)
        assert est.fit(small_1d.likelihoods) is est
    assert est.population_.shape == (12,)
    p = est.predict_proba(small_1d.likelihoods)
    assert p.shape == (small_1d.likelihoods.N, 12)
    assert est.predict(small_1d.likelihoods).shape == (small_1d.likelihoods.N,)
    assert np.isfinite(est.score(small_1d.likelihoods))
    assert est.sample(5, rng=1).shape == (5, 12)
    assert "PopulationEstimator(" in repr(est)


def test_estimator_map_and_raw_arrays(small_1d):
    est = PopulationEstimator(method="map", bins=small_1d.bins).fit(small_1d.likelihoods.toarray())
    assert isinstance(est.result_, binference.PointEstimate)
    assert est.sample(3).shape == (3, 12)


def test_selection_through_infer():
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 3), selection=[0.9, 0.5, 0.2], rng=3)
    with pytest.raises(ValueError, match="selection_mode"):
        binference.infer(mock.likelihoods, selection=[0.9, 0.5, 0.2], method="map")
    a = binference.infer(mock.likelihoods, method="map")
    b = binference.infer(
        mock.likelihoods.with_selection(None),
        selection=Selection([0.9, 0.5, 0.2], "on_truth"),
        method="map",
    )
    np.testing.assert_allclose(a.rho, b.rho, atol=1e-8)
