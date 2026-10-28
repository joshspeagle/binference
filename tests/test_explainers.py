"""The docs' explainers (docs/_static/explainers) reproduce binference's own numbers."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import binference

SCRIPT = Path(__file__).parents[1] / "docs" / "_scripts" / "explainers.mjs"
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
