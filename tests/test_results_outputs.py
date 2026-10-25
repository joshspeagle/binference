"""Selected population, full-sample counts, KKT checks, data support and the prior-vs-data
ratio (DECISIONS #12, #41, #49, #53)."""

import numpy as np
import pytest

import binference
from binference import Bins, Selection
from binference.samplers import MAP, Gibbs

S = np.array([1.0, 0.7, 0.4, 0.2])


def _fit(selection=S, n=300, **kw):
    bins = Bins.linear(0, 1, 4)
    lk = binference.simulate.mock_1d(n, bins, noise=0.1, rng=0).likelihoods.with_selection(None)
    sel = None if selection is None else Selection(selection, "on_truth")
    model = binference.PopulationModel(lk, selection=sel, **kw)
    return lk, Gibbs(model, n_warmup=50, rng=0).run(200)


def test_the_selected_population_reweights_each_draw_by_its_selection():
    _, res = _fit()
    phi = res.samples * S / (res.samples @ S)[:, None]
    np.testing.assert_allclose(res.selected, phi)
    np.testing.assert_allclose(res.summary(which="selected")["mean"], phi.mean(0))
    plain = _fit(selection=None)[1]
    np.testing.assert_allclose(plain.selected, plain.samples)
    est = binference.infer(
        res.model.likelihoods, method="map", selection=S, selection_mode="on_truth"
    )
    np.testing.assert_allclose(est.selected, est.rho * S / (est.rho @ S))


def test_an_inferred_selection_gives_each_draw_its_own_selected_population():
    bins = Bins.linear(0, 1, 4)
    lk = binference.simulate.mock_1d(200, bins, noise=0.1, rng=0).likelihoods
    z = np.linspace(0.01, 0.99, 80)
    sel = Selection.from_injections(z, np.random.default_rng(0).random(80) < 0.6, bins, "on_truth")
    res = Gibbs(binference.PopulationModel(lk, selection=sel), n_warmup=30, rng=0).run(40)
    Sd = res.info["selection_draws"].reshape(-1, 4)
    phi = res.samples * Sd
    np.testing.assert_allclose(res.selected, phi / phi.sum(1, keepdims=True))


def test_full_sample_counts_add_the_missed_objects():
    # with S known, the full sample holds W / (S . rho) objects on average
    _, res = _fit()
    full = res.realized_counts(n_draws=None, rng=1, full=True)
    seen = res.realized_counts(n_draws=None, rng=1)
    np.testing.assert_allclose(seen.sum(1), res.model.W)
    expect = res.model.W / (res.samples @ S)
    gap = full.sum(1) - expect  # NegBin noise, sd sqrt(W (1 - p)) / p per draw
    sd = np.sqrt(res.model.W * (1 - res.samples @ S)) / (res.samples @ S)
    assert abs(gap.mean()) < 4 * np.sqrt(np.mean(sd**2) / gap.size)
    relative = binference.PopulationModel(
        res.model.likelihoods, selection=Selection(S, "on_truth", scale="relative")
    )
    rel = Gibbs(relative, n_warmup=0, rng=0).run(2)
    with pytest.raises(ValueError, match="relative"):
        rel.realized_counts(full=True)


def test_a_point_estimate_check_flags_its_kkt_gap_and_convergence():
    lk, _ = _fit(selection=None)
    good = MAP(binference.PopulationModel(lk)).run()
    assert good.check() == []
    with pytest.warns(binference.ConvergenceWarning):
        bad = MAP(binference.PopulationModel(lk), algorithm="em", max_iter=2).run()
    with pytest.warns(binference.ConvergenceWarning, match="KKT gap"):
        bad.check()
    with pytest.raises(binference.DiagnosticsError, match="did not converge"):
        bad.check(strict=True)


def test_data_support_counts_objects_worth_of_likelihood_per_bin():
    lk, res = _fit(selection=None)
    L = lk.toarray()
    np.testing.assert_allclose(res.data_support, (L / L.sum(1, keepdims=True)).sum(0))
    np.testing.assert_allclose(res.summary()["support"], res.data_support)
    with pytest.warns(binference.BinferenceWarning) as caught:  # short chains warn too
        res.check(support_min=res.data_support.max() + 1)
    assert any("data support below" in str(w.message) for w in caught)


def test_the_prior_information_ratio_falls_with_the_data_and_rises_with_the_prior():
    ratios = {}
    for n in (100, 400):
        for alpha in (1.0, 10.0):
            _, res = _fit(selection=None, n=n, prior=binference.priors.Dirichlet(alpha))
            ratios[n, alpha] = res.prior_information_ratio()
    assert ratios[400, 1.0] < ratios[100, 1.0] < ratios[100, 10.0]
    np.testing.assert_allclose(ratios[100, 1.0] / ratios[400, 1.0], 4.0, rtol=0.3)
