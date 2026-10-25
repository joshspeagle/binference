"""Selections measured by injections and inferred with the population (DECISIONS #10, #30, #79)."""

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods, Selection
from binference.samplers import MAP, Gibbs
from binference.validation import SBCResult


def _injections(S, bins, n_per_bin, rng):
    z = np.repeat(bins.centers, n_per_bin) + rng.uniform(
        -0.5, 0.5, bins.K * n_per_bin
    ) * np.repeat(bins.volumes, n_per_bin)
    detected = rng.random(z.size) < np.repeat(S, n_per_bin)
    return z, detected


def test_injections_give_a_beta_posterior_per_bin():
    bins = Bins.linear(0, 1, 4)
    z = np.array([0.1, 0.1, 0.3, 0.6, 0.6, 0.6, 0.9, 1.5])
    detected = np.array([1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    with pytest.warns(binference.InputWarning, match="outside"):
        sel = Selection.from_injections(z, detected, bins, "on_truth", prior=(1.0, 2.0))
    injected, recovered, prior = sel.injections
    assert prior == (1.0, 2.0)
    np.testing.assert_array_equal(injected, [2, 1, 3, 1])
    np.testing.assert_array_equal(recovered, [1, 1, 2, 0])
    np.testing.assert_allclose(sel.values, (1 + recovered) / (3 + injected))
    assert sel.draw(5, rng=0).shape == (5, 4) and sel.at_mean().injections is None


def test_injection_inputs_are_checked_by_name():
    bins = Bins.linear(0, 1, 4)
    z = np.linspace(0.05, 0.95, 8)
    with pytest.raises(ValueError, match="detected"):
        Selection.from_injections(z, np.full(8, 0.5), bins, "on_truth")
    with pytest.raises(ValueError, match="injections but detected"):
        Selection.from_injections(z, np.ones(7, bool), bins, "on_truth")
    with pytest.raises(ValueError, match="prior"):
        Selection.from_injections(z, np.ones(8, bool), bins, "on_truth", prior=(0, 1))
    with pytest.raises(ValueError, match="Basis"):
        Selection.from_injections(z, np.ones(8, bool), binference.Basis(bins), "on_truth")


def _has_xarray():
    try:
        import xarray  # noqa: F401
    except ImportError:
        return False
    return True


def _problem(mode="on_truth", seed=0):
    rng = np.random.default_rng(seed)
    bins = Bins.linear(0, 1, 4)
    S = np.array([0.9, 0.7, 0.4, 0.2])
    mock = binference.simulate.mock_1d(300, bins, noise=0.1, rng=seed)
    sel = Selection.from_injections(*_injections(S, bins, 40, rng), bins, mode)
    return mock.likelihoods.with_selection(None), sel


def test_only_gibbs_infers_a_selection_from_injections():
    lk, sel = _problem()
    m = binference.PopulationModel(lk, selection=sel)
    with pytest.raises(ValueError, match="method='gibbs'"):
        MAP(m)
    est = binference.infer(lk, selection=sel.at_mean(), method="map")  # the plug-in
    assert np.isfinite(est.rho).all()
    res = binference.infer(lk, selection=sel, method="auto", n_samples=50, rng=0, check=False)
    assert res.method == "gibbs" and res.info["selection_draws"].shape == (4, 50, 4)
    assert res.total.shape == (200,)
    if _has_xarray():  # exported with the population
        assert res.to_datatree()["posterior"]["selection"].shape == (4, 50, 4)


def test_a_selection_on_the_measurement_is_only_a_plug_in():
    # the likelihoods' cut fixes S; with S free the catalogue adds only missed objects and
    # drove S to zero in a first version (DECISIONS #79)
    lk, sel = _problem("on_measurement")
    with pytest.raises(ValueError, match="on the truth"):
        binference.PopulationModel(lk, selection=sel)
    assert np.isfinite(binference.infer(lk, selection=sel.at_mean(), method="map").rho).all()


@pytest.mark.slow
def test_joint_inference_of_population_and_selection_is_calibrated():
    # SBC over (rho, S): rho ~ Dir(1), S_k ~ Beta(1, 1), 30 injections per bin, 100
    # detected objects drawn from rho * S, Gaussian errors of 0.15 on bins of 0.25
    bins = Bins.linear(0, 1, 4)
    rng = np.random.default_rng(2026)
    n_draws, ranks = 49, []
    for _ in range(200):
        rho, S = rng.dirichlet(np.ones(4)), rng.uniform(size=4)
        p = rho * S
        k = rng.choice(4, size=100, p=p / p.sum())
        z = bins.edges[0][k] + 0.25 * rng.random(100)
        lk = Likelihoods.from_gaussian(
            z + 0.15 * rng.standard_normal(100), 0.15, bins, warn_outside=False
        )
        sel = Selection.from_injections(*_injections(S, bins, 30, rng), bins, "on_truth")
        res = Gibbs(
            binference.PopulationModel(lk, selection=sel), n_chains=2, n_warmup=200, rng=rng
        ).run(500)
        idx = np.linspace(0, res.n_draws - 1, n_draws).round().astype(int)
        R, Sd = res.samples[idx], res.info["selection_draws"].reshape(-1, 4)[idx]
        ranks.append(np.concatenate([(R < rho).sum(0), (Sd < S).sum(0)]))
    names = [f"rho[{k}]" for k in range(4)] + [f"S[{k}]" for k in range(4)]
    p = SBCResult(np.array(ranks), n_draws, names).pvalues(n_bins=10)
    assert np.all(p > 1e-3 / p.size), dict(zip(names, p, strict=True))
