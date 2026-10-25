"""Tests for Density (DECISIONS #26), interim priors (#24) and explicit stacking (#11)."""

import numpy as np
import pytest
from scipy import stats

import binference
from binference import Bins, Density, Likelihoods, priors


def test_flat_is_flat_in_the_measure_coordinate_unless_asked_for_z():
    logbins = Bins.log(1.0, 1000.0, 3)
    np.testing.assert_allclose(Density.flat().masses(logbins), 1 / 3)
    np.testing.assert_allclose(Density.flat("z").masses(logbins), np.array([9, 90, 900]) / 999)


def test_masses_and_densities_differ_by_bin_volumes():
    bins = Bins([0.0, 1.0, 3.0])
    np.testing.assert_allclose(Density.from_masses([1, 1]).masses(bins), [0.5, 0.5])
    np.testing.assert_allclose(Density.from_densities([1, 1]).masses(bins), [1 / 3, 2 / 3])


def test_a_distribution_cdf_and_its_pdf_give_the_same_masses():
    bins = Bins.linear(-2, 2, 8)
    d = stats.norm(0.3, 0.7)
    exact = Density.from_distribution(d).masses(bins)
    quad = Density.from_function(d.pdf).masses(bins, n_sub=8)
    np.testing.assert_allclose(quad, exact, rtol=1e-6)


def test_a_callable_on_categories_receives_the_labels():
    bins = Bins.categories(["star", "galaxy"])
    d = Density.from_function(lambda labs: np.array([3.0 if s == "star" else 1.0 for s in labs]))
    np.testing.assert_allclose(d.masses(bins), [0.75, 0.25])


def test_an_array_interim_is_read_the_same_way_by_every_constructor():
    bins = Bins.log(0.1, 10.0, 12)
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.5, 5.0, 30)
    pi = 1.0 / bins.centers  # an interim density per unit z
    dists = [stats.norm(m, 0.3) for m in mu]
    via_dist = Likelihoods.from_distributions(
        dists, bins, kind="posterior", interim_prior=pi, interim_representation="density"
    )
    via_func = Likelihoods.from_function(
        lambda z: np.array([d.pdf(z) for d in dists]),
        bins,
        n_sub=8,
        kind="posterior",
        interim_prior=pi,
        interim_representation="density",
    )
    np.testing.assert_allclose(via_func.toarray(), via_dist.toarray(), rtol=1e-3, atol=1e-8)
    for build in (
        lambda: Likelihoods.from_distributions(dists, bins, kind="posterior", interim_prior=pi),
        lambda: Likelihoods.from_function(
            lambda z: np.ones((1, z.size)), bins, kind="posterior", interim_prior=pi
        ),
    ):
        with pytest.raises(ValueError, match="ambiguous"):
            build()


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        Likelihoods.from_function(lambda z: np.ones((1, z.size)), Bins.linear(0, 1, 3), kind="post")


def test_stack_reproduces_the_mean_of_the_input_posteriors():
    bins = Bins.linear(0, 1, 4)
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(4), size=50)
    interim = np.array([0.1, 0.2, 0.3, 0.4])
    lk = Likelihoods.from_posteriors(P, bins, interim)
    np.testing.assert_allclose(lk.stack(), P.mean(axis=0), rtol=1e-12)
    flat = lk.stack("flat")
    assert not np.allclose(flat, P.mean(axis=0))


def test_stack_names_rows_without_support_under_the_interim_prior():
    bins = Bins.linear(0, 1, 3)
    lk = Likelihoods(np.array([[1.0, 0.0, 0.0], [0.2, 0.5, 0.3]]), bins)
    with pytest.raises(ValueError, match=r"Rows \[0\]"):
        lk.stack(Density.from_masses([0.0, 0.5, 0.5]))


def test_the_default_base_measure_is_flat_in_the_measure_coordinate():
    logbins = Bins.log(1.0, 100.0, 4)
    alpha = priors.Dirichlet.from_base_measure(2.0).resolve_alpha(logbins)
    np.testing.assert_allclose(alpha, 0.5)
    by_z = priors.Dirichlet.from_base_measure(2.0, Density.flat("z")).resolve_alpha(logbins)
    np.testing.assert_allclose(by_z, 2.0 * logbins.volumes / logbins.volumes.sum())


def test_population_accepts_a_density():
    bins = Bins.linear(0, 1, 5)
    d = Density.from_masses([1, 2, 3, 2, 1])
    np.testing.assert_allclose(binference.simulate.population(bins, d), d.masses(bins))
