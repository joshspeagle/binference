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


@pytest.mark.parametrize(("sigma", "width"), [(0.1, 0.25), (0.45, 0.4)])
def test_bin_average_explainer_matches_from_gaussian(sigma, width):
    v1 = explainer("v1", sigma=sigma, width=width)
    lk = binference.Likelihoods.from_gaussian([1.37], [sigma], binference.Bins(v1["edges"]))
    L = lk.toarray()[0] * np.exp(lk.log_scale[0])
    np.testing.assert_allclose(
        v1["L"], L, rtol=1e-6, atol=1e-6
    )  # the explainer's CDF: error < 1.2e-7


def test_interim_prior_explainer_fits_by_binference_em_steps():
    v3 = explainer("v3", centre=2.4)
    lk, post = (
        binference.Likelihoods(np.array(v3["L"])),
        binference.Likelihoods(np.array(v3["posteriors"])),
    )
    for rows, fit in ((lk, v3["right"]), (post, v3["wrong"])):
        model, rho = (
            binference.PopulationModel(rows),
            np.full(len(v3["prior"]), 1 / len(v3["prior"])),
        )
        for _ in range(v3["steps"]):
            rho = model.expected_counts(rho) / model.W
        np.testing.assert_allclose(fit, rho, rtol=1e-9)
    L = np.array(v3["L"]) * v3["prior"]
    np.testing.assert_allclose(v3["posteriors"], L / L.sum(axis=1, keepdims=True), rtol=1e-12)


def test_em_explainer_starts_from_the_stack_and_follows_binference_em():
    v4 = explainer("v4", step=1)
    lk = binference.Likelihoods(np.array(v4["L"]))
    np.testing.assert_allclose(v4["rho"], lk.stack(), rtol=1e-9)
    model, rho = binference.PopulationModel(lk), lk.stack()
    for _ in range(24):
        rho = model.expected_counts(rho) / model.W
    np.testing.assert_allclose(explainer("v4", step=25)["rho"], rho, rtol=1e-9)


def test_shrinkage_explainer_matches_binference_responsibilities():
    v5 = explainer("v5", width=0.3)
    bins = binference.Bins(np.linspace(0, 3, 31))
    one = binference.Likelihoods.from_gaussian([1.9], [0.4], bins)
    np.testing.assert_allclose(v5["L"], one.toarray()[0] * np.exp(one.log_scale[0]), atol=1e-6)
    np.testing.assert_allclose(
        v5["rho"], binference.simulate.population(bins, stats.norm(1.2, 0.3)), atol=1e-6
    )
    post = binference.PopulationModel(one).responsibilities(np.array(v5["rho"]))[0]
    np.testing.assert_allclose(v5["posterior"], post, atol=1e-6)
