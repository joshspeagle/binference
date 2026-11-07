"""Tests for the high-level API: infer(), PopulationEstimator, namespace (spec 8)."""

import importlib
from pathlib import Path

import numpy as np
import pytest

import binference
from binference import Bins, PopulationEstimator, PopulationModel, Selection


def test_namespace():
    for name in binference.__all__:
        assert hasattr(binference, name), name
    assert isinstance(binference.__version__, str)
    assert binference.engines.NUTS and binference.priors.Dirichlet and binference.Selection


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
        binference.infer(m, prior=binference.priors.Dirichlet(2.0), method="map")


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


def test_the_old_samplers_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("binference.samplers")


class _Halves(binference.engines.base.Engine):
    def __init__(self, model, *, rng=None, progress=False, n=2):
        super().__init__(model, rng=rng, progress=progress)
        self.n = n

    def run(self, n_samples=None):
        value = np.full(self.model.K, 1.0 / self.model.K)
        return binference.PointEstimate(self.model, value, method="halves")


def test_infer_runs_an_engine_class_with_its_options():
    lk = binference.Likelihoods(np.ones((3, 2)), Bins.linear(0, 1, 2))
    assert binference.infer(lk, method=_Halves, n=3, rng=0).method == "halves"
    with pytest.raises(TypeError, match="unexpected keyword"):
        binference.infer(lk, method=_Halves, nope=1)


def test_infer_refuses_a_class_that_is_not_an_engine():
    lk = binference.Likelihoods(np.ones((3, 2)), Bins.linear(0, 1, 2))
    with pytest.raises(ValueError, match="Engine subclass"):
        binference.infer(lk, method=dict)


def test_a_raw_array_of_normalised_rows_warns_that_it_may_hold_posteriors():
    table = np.array([[0.2, 0.8], [0.5, 0.5]])
    with pytest.warns(binference.InputWarning, match="may be posteriors"):
        PopulationModel(table, bins=Bins.linear(0, 1, 2))
    PopulationModel(binference.Likelihoods(table, Bins.linear(0, 1, 2)))  # stated: no warning
    PopulationModel(2.0 * table, bins=Bins.linear(0, 1, 2))
    with pytest.warns(binference.InputWarning, match="may be posteriors"):
        PopulationModel(np.round(table / 3 + 0.333, 3), bins=Bins.linear(0, 1, 2))


@pytest.mark.parametrize("method", ["laplace", "vb", "bootstrap", "map"])
@pytest.mark.filterwarnings("ignore::binference.ApproximationWarning")
def test_infer_checks_the_inputs_after_every_engine(method):
    rng = np.random.default_rng(0)
    bins = Bins.linear(0, 1, 4)
    lk = binference.Likelihoods.from_samples(rng.uniform(0, 1, (2000, 80)), bins, interim="flat")
    with pytest.warns(binference.InputWarning, match="Monte Carlo variance"):
        binference.infer(lk, method=method, rng=0)
