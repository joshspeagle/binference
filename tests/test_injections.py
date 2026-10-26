"""Selections measured by injections and inferred with the population (DECISIONS #10, #30,
#79, #80)."""

import pickle
import warnings

import numpy as np
import pytest
from scipy import special

import binference
from binference import Bins, Likelihoods, Selection
from binference.samplers import MAP, NUTS, Gibbs, Laplace, MeanFieldVB
from binference.selection import _logit_density
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
    injected, recovered, prior, smooth = sel.injections
    assert prior == (1.0, 2.0) and smooth is None
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


def test_engines_that_cannot_infer_a_selection_refuse_and_name_those_that_can():
    lk, sel = _problem()
    m = binference.PopulationModel(lk, selection=sel)
    with pytest.raises(ValueError, match="method='nuts'"):
        MeanFieldVB(m)
    for kw in ({"space": "simplex"}, {"algorithm": "em"}):
        with pytest.raises(ValueError, match="space='unconstrained'"):
            MAP(m, **kw)
    est = binference.infer(lk, selection=sel.at_mean(), method="map")  # the plug-in
    assert np.isfinite(est.rho).all()
    res = binference.infer(lk, selection=sel, method="auto", n_samples=50, rng=0, check=False)
    assert res.method == "gibbs" and res.info["selection_draws"].shape == (4, 50, 4)
    assert res.total.shape == (200,)
    if _has_xarray():  # exported with the population
        assert res.to_datatree()["posterior"]["selection"].shape == (4, 50, 4)


def test_the_joint_density_has_the_gradient_of_its_values():
    # both blocks, for a simplex and a logistic prior, independent and smooth selections
    lk, _ = _problem()
    rng = np.random.default_rng(1)
    z, detected = _injections(np.array([0.9, 0.7, 0.4, 0.2]), lk.bins, 5, rng)
    for prior in (None, binference.priors.LogisticRandomWalk()):
        for smooth in (None, 0.5):
            sel = Selection.from_injections(z, detected, lk.bins, "on_truth", smooth=smooth)
            m = binference.PopulationModel(lk, prior, selection=sel)
            x = m.from_value(np.full(4, 0.25)) + 0.3 * rng.standard_normal(m.dim)
            assert m.dim == m.prior.dim + 4
            _, g = m.log_density_unconstrained_and_grad(x)
            h = 1e-5
            fd = [
                (m.log_density_unconstrained(x + h * e) - m.log_density_unconstrained(x - h * e))
                / (2 * h)
                for e in np.eye(m.dim)
            ]
            np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-6)


def test_map_and_laplace_infer_the_selection_in_unconstrained_coordinates():
    lk, sel = _problem()
    m = binference.PopulationModel(lk, selection=sel)
    est = MAP(m).run()  # space="auto" resolves to unconstrained
    S = est.info["selection_draws"].ravel()
    assert est.info["space"] == "unconstrained" and est.x.size == m.dim
    np.testing.assert_allclose(S, m.selection_at(est.x))
    assert est.total == pytest.approx(m.W / (S @ est.rho))  # the mode's S, not the means
    assert est.log_likelihood == pytest.approx(m.log_likelihood(est.rho, S=S))
    # the log posterior includes the Beta density of S given the injections
    n, r, (a, b), _ = sel.injections
    log_beta = np.sum((a + r - 1) * np.log(S) + (b + n - r - 1) * np.log1p(-S))
    assert est.log_posterior == pytest.approx(est.log_likelihood + m.log_prior(est.rho) + log_beta)
    with pytest.raises(ValueError, match="shape"):
        m.log_likelihood(est.rho, S=np.full(3, 0.5))
    with pytest.raises(ValueError, match="inferred"):
        binference.PopulationModel(lk).log_likelihood(est.rho, S=S)
    assert binference.PointEstimate(None, est.rho, info=est.info).total is None
    with pytest.warns(binference.ApproximationWarning):
        res = Laplace(m, importance=False, rng=0).run(200)
    assert res.info["selection_draws"].shape == (1, 200, 4)
    np.testing.assert_allclose(res.info["mode"], est.x, atol=1e-4)


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


def test_per_object_methods_use_each_draws_selection():
    lk, sel = _problem()
    res = Gibbs(binference.PopulationModel(lk, selection=sel), n_warmup=30, rng=0).run(40)
    S = res.info["selection_draws"].reshape(-1, 4)
    L = lk.values[:3]
    expect = np.mean(
        [L * r * s / (L @ (r * s))[:, None] for r, s in zip(res.samples, S, strict=True)], axis=0
    )
    np.testing.assert_allclose(res.assignment_probabilities(idx=[0, 1, 2], n_draws=None), expect)
    assert "ess_bulk_selection" in res.diagnostics and "rhat_selection" in res.diagnostics


def test_equal_means_from_different_injections_are_different_selections():
    bins = Bins.linear(0, 1, 2)
    a = Selection.from_injections([0.2, 0.2, 0.7, 0.7], [1, 0, 1, 0], bins, "on_truth")
    b = Selection.from_injections([0.2] * 4 + [0.7] * 4, [1, 0] * 4, bins, "on_truth")
    assert np.allclose(a.values, b.values) and a != b  # Beta(2, 2) against Beta(3, 3)
    back = pickle.loads(pickle.dumps(a))
    assert back == a and not back.injections[0].flags.writeable


@pytest.mark.slow
def test_nuts_agrees_with_joint_gibbs():
    lk, sel = _problem()
    m = binference.PopulationModel(lk, selection=sel)
    fits = [NUTS(m, rng=1).run(2000), Gibbs(m, rng=1).run(5000)]
    stats = []
    for r in fits:
        draws = np.concatenate([r.samples, r.info["selection_draws"].reshape(-1, 4)], axis=1)
        d = r.diagnostics
        ess = np.concatenate([d["ess_bulk"], d["ess_bulk_selection"]])
        stats.append((draws.mean(0), draws.std(0), ess))
    (m1, s1, e1), (m2, s2, e2) = stats
    # Monte Carlo errors: of a mean sd / sqrt(ESS), of an sd about sd / sqrt(2 ESS)
    np.testing.assert_array_less(np.abs(m1 - m2), 4 * np.hypot(s1 / np.sqrt(e1), s2 / np.sqrt(e2)))
    se_sd = np.hypot(s1 / np.sqrt(2 * e1), s2 / np.sqrt(2 * e2))
    np.testing.assert_array_less(np.abs(s1 - s2), 4 * se_sd)


@pytest.mark.slow
def test_joint_inference_with_a_logistic_prior_and_nuts_is_calibrated():
    # SBC over (rho, S): rho from LogisticRandomWalk, S_k ~ Beta(1, 1), as in the Gibbs SBC
    bins = Bins.linear(0, 1, 4)
    prior = binference.priors.LogisticRandomWalk()
    rng = np.random.default_rng(2027)
    n_draws, ranks = 49, []
    for _ in range(60):
        rho, S = prior.bind(bins).sample(1, rng)[0], rng.uniform(size=4)
        p = rho * S
        k = rng.choice(4, size=100, p=p / p.sum())
        z = bins.edges[0][k] + 0.25 * rng.random(100)
        lk = Likelihoods.from_gaussian(
            z + 0.15 * rng.standard_normal(100), 0.15, bins, warn_outside=False
        )
        sel = Selection.from_injections(*_injections(S, bins, 30, rng), bins, "on_truth")
        m = binference.PopulationModel(lk, prior, selection=sel)
        res = NUTS(m, n_chains=2, n_warmup=200, rng=rng).run(150)
        idx = np.linspace(0, res.n_draws - 1, n_draws).round().astype(int)
        R, Sd = res.samples[idx], res.info["selection_draws"].reshape(-1, 4)[idx]
        ranks.append(np.concatenate([(R < rho).sum(0), (Sd < S).sum(0)]))
    names = [f"rho[{k}]" for k in range(4)] + [f"S[{k}]" for k in range(4)]
    p = SBCResult(np.array(ranks), n_draws, names).pvalues(n_bins=7)
    assert np.all(p > 1e-3 / p.size), dict(zip(names, p, strict=True))


def test_a_smooth_selection_is_the_mode_of_its_injection_posterior():
    bins = Bins.linear(0, 1, 6)
    z, detected = _injections(np.linspace(0.9, 0.2, 6), bins, 3, np.random.default_rng(4))
    sel = Selection.from_injections(z, detected, bins, "on_truth", smooth=0.3)
    _, g = _logit_density(special.logit(sel.values), sel.injections)
    np.testing.assert_allclose(g, 0.0, atol=1e-8)
    indep = Selection.from_injections(z, detected, bins, "on_truth")
    assert sel != indep and np.ptp(np.diff(sel.values)) < np.ptp(np.diff(indep.values))
    assert pickle.loads(pickle.dumps(sel)) == sel


def test_a_smooth_selection_is_refused_where_it_does_not_apply():
    bins = Bins.linear(0, 1, 6)
    z, detected = np.linspace(0.05, 0.95, 12), np.tile([True, False], 6)
    sel = Selection.from_injections(z, detected, bins, "on_truth", smooth=0.3)
    with pytest.raises(ValueError, match="independent Betas"):
        sel.draw(3)
    m = binference.PopulationModel(Likelihoods(np.ones((5, 6)), bins), selection=sel)
    with pytest.raises(TypeError, match="smooth"):
        Gibbs(m)
    assert binference.inference.Auto(m).candidates()[0][0] != "gibbs"
    with pytest.raises(ValueError, match="smooth"):
        Selection.from_injections(z, detected, bins, "on_truth", smooth=0.0)
    for few in (Bins.grid([0, 1, 2], [0, 1, 2]), Bins.linear(0, 1, 2), Bins.categories(4)):
        with pytest.raises(ValueError, match="ordered 1-D bins"):
            Selection.from_injections(few.centers[:1], [1], few, "on_truth", smooth=1.0)


def test_nuts_centres_a_logistic_prior_beside_an_inferred_selection():
    # partial centring acts on the prior's block; draws come back non-centred with logit S
    lk, sel = _problem()
    m = binference.PopulationModel(lk, binference.priors.LogisticRandomWalk(), selection=sel)
    eng = NUTS(m, n_chains=2, n_warmup=60, rng=0, reparametrize=True)
    eng._geometry_iter = 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", binference.ConvergenceWarning)
        res = eng.run(20)
    assert "centring" in res.info and res.x.shape == (2, 20, m.dim)
    x, lp = res.x.reshape(-1, m.dim), res.log_density_unconstrained.ravel()
    np.testing.assert_allclose([m.log_density_unconstrained(v) for v in x], lp, rtol=1e-8)
    np.testing.assert_allclose(res.info["selection_draws"].reshape(-1, 4), m.selection_at(x))


def test_auto_and_exact_choose_what_they_can_infer():
    lk, sel = _problem()
    m = binference.PopulationModel(lk, binference.priors.LogisticRandomWalk(), selection=sel)
    assert binference.inference.Auto(m).candidates()[0][0] != "gibbs"
    m = binference.PopulationModel(
        Likelihoods(lk.toarray()[:, :3], Bins.linear(0, 1, 3)),
        selection=Selection.from_injections([0.1, 0.5], [1, 1], Bins.linear(0, 1, 3), "on_truth"),
    )
    with pytest.raises(ValueError, match="at_mean"):
        binference.validation.exact_posterior(m)
    with pytest.raises(ValueError, match="prior"):
        Selection.from_injections([0.1], [1], Bins.linear(0, 1, 3), "on_truth", prior=1.0)
    with pytest.raises(ValueError, match="z"):
        Selection.from_injections(np.ones((2, 2)), [1, 1], Bins.linear(0, 1, 3), "on_truth")
