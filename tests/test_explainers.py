"""The docs' explainers (docs/_static/explainers) reproduce binference's own numbers."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import binference
from binference.samplers import Gibbs

DOCS = Path(__file__).parents[1] / "docs"
SCRIPT = DOCS / "_scripts" / "explainers.mjs"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


def explainer(name, **state):
    run = subprocess.run(
        ["node", SCRIPT, "--compute", name, json.dumps(state)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(run.stdout)


def em_steps(lk, rho, steps):
    model = binference.PopulationModel(lk)
    for _ in range(steps):
        rho = model.expected_counts(rho) / model.W
    return rho


@pytest.mark.parametrize(("sigma", "width"), [(0.1, 0.25), (0.45, 0.4)])
def test_bin_average_explainer_matches_from_gaussian(sigma, width):
    v1 = explainer("v1", sigma=sigma, width=width)
    lk = binference.Likelihoods.from_gaussian([v1["d"]], [sigma], binference.Bins(v1["edges"]))
    L = lk.toarray()[0] * np.exp(lk.log_scale[0])
    # the explainer's normal CDF has error below 1.2e-7
    np.testing.assert_allclose(v1["L"], L, rtol=1e-6, atol=1e-6)


def test_interim_prior_explainer_fits_by_binference_em_steps():
    v3 = explainer("v3", centre=2.4, width=0.4)
    lk = binference.Likelihoods(np.array(v3["L"]))
    post = binference.PopulationModel(lk).responsibilities(np.array(v3["prior"]))
    np.testing.assert_allclose(v3["posteriors"], post, rtol=1e-12)
    flat = np.full(post.shape[1], 1 / post.shape[1])
    np.testing.assert_allclose(v3["right"], em_steps(lk, flat, v3["steps"]), rtol=1e-9)
    np.testing.assert_allclose(
        v3["wrong"], em_steps(binference.Likelihoods(post), flat, v3["steps"]), rtol=1e-9
    )


def test_em_explainer_starts_from_the_stack_and_follows_binference_em():
    v4 = explainer("v4", step=1)
    lk = binference.Likelihoods(np.array(v4["L"]))
    np.testing.assert_allclose(v4["rho"], lk.stack(), rtol=1e-9)
    np.testing.assert_allclose(
        explainer("v4", step=25)["rho"], em_steps(lk, lk.stack(), 24), rtol=1e-9
    )


def test_shrinkage_explainer_matches_binference_responsibilities():
    v5 = explainer("v5", width=0.3)
    bins = binference.Bins(v5["edges"])
    one = binference.Likelihoods.from_gaussian([v5["d"]], [v5["sigma"]], bins)
    np.testing.assert_allclose(v5["L"], one.toarray()[0] * np.exp(one.log_scale[0]), atol=1e-6)
    pop = binference.simulate.population(bins, stats.norm(v5["centre"], 0.3))
    np.testing.assert_allclose(v5["rho"], pop, atol=1e-6)
    post = binference.PopulationModel(one).responsibilities(np.array(v5["rho"]))[0]
    np.testing.assert_allclose(v5["posterior"], post, atol=1e-6)


@pytest.mark.parametrize("loga", [-1.0, 0.0, 1.5])
def test_dirichlet_explainer_draws_have_the_dirichlet_mean_and_variance(loga):
    v7 = explainer("v7", loga=loga)
    draws, a = np.array(v7["draws"]), v7["alpha"]
    n = len(draws)
    # each fraction of Dir(a, a, a) is Beta(a, 2a); the sample mean and variance of n draws
    # have standard errors sd / sqrt(n) and sqrt((mu4 - var^2) / n); allow 5 of them
    mean, var, _, kurt = stats.beta(a, 2 * a).stats(moments="mvsk")
    mu4 = (kurt + 3) * var**2
    np.testing.assert_array_less(np.abs(draws.mean(axis=0) - mean), 5 * np.sqrt(var / n))
    np.testing.assert_array_less(np.abs(draws.var(axis=0) - var), 5 * np.sqrt((mu4 - var**2) / n))


def _batch_correlation_se(draws, n):
    # the standard error of a correlation from n draws: the spread of the correlations of
    # batches of n binference draws
    batches = draws.reshape(-1, n, draws.shape[1])
    return np.std([np.corrcoef(b, rowvar=False) for b in batches], axis=0, ddof=1)


@pytest.mark.parametrize(
    ("prior", "s"),
    [(0, 0.0), (1, -1.0), (2, -1.0), (2, 0.0), (2, 1.0), (3, -1.0), (3, 0.0), (3, 1.0)],
)
def test_prior_explainer_correlations_match_binference_prior_draws(prior, s):
    v8 = explainer("v8", prior=prior, s=s)
    corr, value = np.array(v8["corr"]), v8["value"]
    bins = binference.Bins.linear(0.0, 1.0, 16)
    spec = [
        binference.priors.Dirichlet(value),
        binference.priors.PolyaTree(c=value),
        binference.priors.LogisticRandomWalk(order=2, scale=value),
        binference.priors.LogisticGP(amplitude=1.0, length_scale=value, n_basis=64),
    ][prior]
    model = binference.PopulationModel(binference.Likelihoods(np.ones((1, 16)), bins), spec)
    n, batches = 2000, 20  # the explainer's 2,000 draws; binference's 20 times as many
    draws = model.sample_prior(n * batches, rng=3)
    se = _batch_correlation_se(draws, n)
    # the explainer's error and binference's (sqrt(20) times smaller), 5 standard errors
    tol = 5 * se * np.sqrt(1 + 1 / batches) + 1e-12  # the diagonal is 1 in both
    np.testing.assert_array_less(np.abs(corr - np.corrcoef(draws, rowvar=False)), tol)


@pytest.mark.parametrize(("levels", "logc"), [(2, 0.0), (5, -1.0), (5, 0.0)])
def test_polya_explainer_matches_binference_mean_and_variance_at_every_level(levels, logc):
    v9 = explainer("v9", levels=levels, logc=logc)
    draws, K = np.array(v9["draws"]), 2**levels
    model = binference.PopulationModel(
        binference.Likelihoods(np.ones((1, K)), binference.Bins.linear(0.0, 1.0, K)),
        binference.priors.PolyaTree(c=v9["c"]),
    )
    ref = model.sample_prior(40000, rng=4)
    n = len(draws)
    # the mean is the base measure, flat; 5 standard errors of the mean of n draws
    np.testing.assert_array_less(
        np.abs(draws.mean(axis=0) - 1 / K), 5 * ref.std(axis=0) / np.sqrt(n)
    )
    for h in range(1, levels + 1):  # the masses of the 2^h nodes at depth h
        js, bnf = (x.reshape(len(x), 2**h, -1).sum(axis=2) for x in (draws, ref))
        m2 = bnf.var(axis=0)
        m4 = ((bnf - bnf.mean(axis=0)) ** 4).mean(axis=0)
        # standard errors of the two sample variances, sqrt((mu4 - var^2) / n); 5 of them
        tol = 5 * np.sqrt((m4 - m2**2) * (1 / n + 1 / len(ref)))
        np.testing.assert_array_less(np.abs(js.var(axis=0) - m2), tol)


@pytest.mark.parametrize("width", [0.05, 0.3, 0.6])
def test_missing_information_explainer_matches_binference_at_the_mock_population(width):
    v14 = explainer("v14", width=width)
    model = binference.PopulationModel(binference.Likelihoods(np.array(v14["L"])))
    # the mock's population: two normal peaks cut to [0, 3], and a flat share of 0.5 that
    # grows a little because a value of the peaks outside [0, 3] is redrawn from the start
    edges = np.linspace(0, 3, 11)
    peaks = [(0.6, 0.9, 0.25), (0.4, 2.1, 0.3)]
    cdf = lambda x: sum(a * stats.norm.cdf(x, c, w) for a, c, w in peaks)  # noqa: E731
    q = cdf(3) - cdf(0)
    share = 0.5 / (0.5 + 0.5 * q)
    rho = share * np.diff(edges) / 3 + (1 - share) * np.diff(cdf(edges)) / q
    np.testing.assert_allclose(v14["rho"], rho, rtol=1e-6)  # the explainer's CDF: error < 1.2e-7
    # the same formula at the same population: round-off
    np.testing.assert_allclose(
        v14["f"], model.missing_information(np.array(v14["rho"])), atol=1e-9
    )


@pytest.mark.parametrize("width", [0.05, 0.3, 0.6])
def test_missing_information_explainer_gibbs_draws_the_binference_posterior(width):
    v14 = explainer("v14", width=width)
    model = binference.PopulationModel(binference.Likelihoods(np.array(v14["L"])))
    js = np.array(v14["chain"])
    ref = Gibbs(model, n_chains=1, n_warmup=200, rng=6).run(8000).chains[0]
    # a mean of n draws with autocorrelation time at most tau = (1 + f) / (1 - f) has standard
    # error sd sqrt(tau / n); 5 standard errors of the difference of the two means
    tau = (1 + v14["f"]) / (1 - v14["f"])
    sd = ref.std(axis=0)
    tol = 5 * sd * np.sqrt(tau / len(js) + tau / len(ref))
    np.testing.assert_array_less(np.abs(js.mean(axis=0) - ref.mean(axis=0)), tol)


def test_a_control_with_options_is_a_disabled_select_with_its_default_chosen(monkeypatch):
    monkeypatch.syspath_prepend(str(DOCS / "_ext"))
    from bnf_explainer import _control

    html = _control(
        "v-8", {"name": "prior", "label": "prior", "options": ["a", "b<c"], "value": 1}
    )
    assert '<select id="v-8-prior" name="prior" disabled>' in html
    assert '<option value="0">a</option><option value="1" selected>b&lt;c</option>' in html
