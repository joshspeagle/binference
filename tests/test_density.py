"""Tests for Density (DECISIONS #26), interim priors (#24) and explicit stacking (#11)."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

import binference
from binference import Bins, Density, Likelihoods, priors
from binference.priors.logistic import _resolve_mean


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


def test_a_distribution_with_a_cdf_and_an_sf_keeps_its_upper_tail():
    # nine standard deviations up the CDF rounds to one; the survival function does not
    norm = SimpleNamespace(cdf=stats.norm.cdf, sf=stats.norm.sf, pdf=stats.norm.pdf)
    bins = Bins.linear(9, 12, 3)
    exact = Density.from_distribution(stats.norm()).masses(bins)  # logcdf and logsf
    np.testing.assert_allclose(Density.from_distribution(norm).masses(bins), exact, rtol=1e-12)


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
    via_dist = Likelihoods.from_density(dists, bins, interim=pi, interim_representation="density")
    via_func = Likelihoods.from_density(
        lambda z: np.array([d.pdf(z) for d in dists]),
        bins,
        interim=pi,
        interim_representation="density",
        tol=1e-6,
    )
    via_table = Likelihoods.from_table(
        np.array([np.diff(d.cdf(bins.edges[0])) for d in dists]),
        bins,
        interim=pi,
        interim_representation="density",
    )
    np.testing.assert_allclose(via_func.toarray(), via_dist.toarray(), rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(via_table.toarray(), via_dist.toarray(), rtol=1e-8, atol=1e-10)
    for build in (
        lambda: Likelihoods.from_density(dists, bins, interim=pi),
        lambda: Likelihoods.from_table(np.ones((1, bins.K)), bins, interim=pi),
    ):
        with pytest.raises(ValueError, match="ambiguous"):
            build()


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        Likelihoods.from_density(lambda z: np.ones((1, z.size)), Bins.linear(0, 1, 3), kind="post")


def test_stack_reproduces_the_mean_of_the_input_posteriors():
    bins = Bins.linear(0, 1, 4)
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(4), size=50)
    interim = np.array([0.1, 0.2, 0.3, 0.4])
    lk = Likelihoods.from_table(P, bins, interim=interim)
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


def test_densities_are_equal_when_built_alike_from_equal_values():
    a = np.linspace(1.0, 2.0, 10)
    func = lambda z: z  # noqa: E731
    ramp = _Ramp()
    equal = [
        (Density.flat(), Density.flat()),
        (Density.from_masses(a), Density.from_masses(list(a))),
        (Density.from_distribution(stats.norm(0, 1)), Density.from_distribution(stats.norm(0, 1))),
        (Density.from_function(func), Density.from_function(func)),
        # numbers as floats; a bound method as Python compares it (a new object per access)
        (
            Density.from_distribution(stats.norm(0, 1)),
            Density.from_distribution(stats.norm(0.0, 1.0)),
        ),
        (Density.from_function(ramp.pdf), Density.from_function(ramp.pdf)),
    ]
    for d1, d2 in equal:
        assert d1 == d2 and hash(d1) == hash(d2)
    unequal = [
        (Density.flat(), Density.flat("z")),
        (Density.from_masses(a), Density.from_masses(a[::-1])),
        (Density.from_masses(a), Density.from_densities(a)),
        (Density.from_distribution(stats.norm(0, 1)), Density.from_distribution(stats.norm(0, 2))),
        # a frozen rv_histogram's family name and arguments leave out its histogram
        tuple(
            Density.from_distribution(stats.rv_histogram((h, [0, 1, 2]))())
            for h in ([1, 2], [2, 1])
        ),
        (Density.from_function(func), Density.from_function(lambda z: z)),
        # a distribution outside SciPy's public families by identity, not by its repr
        (Density.from_distribution(ramp), Density.from_distribution(_Ramp())),
        (Density.from_function(func), Density.from_function(func, bounded=True)),
    ]
    for d1, d2 in unequal:
        assert d1 != d2
    # a prior keys a bounded density apart from the same function unbounded
    keys = {
        priors.PolyaTree(base=Density.from_function(_ramp, bounded=b))._fingerprint()
        for b in (False, True)
    }
    assert len(keys) == 2


class _Ramp:
    """A distribution whose repr does not tell two apart."""

    def pdf(self, z):
        return np.asarray(z, float)

    def __repr__(self):
        return "_Ramp()"


def _ramp(z):
    return 1.0 + z


def test_a_prior_keys_a_named_function_by_its_name_and_others_by_identity(monkeypatch):
    key = priors.PolyaTree(base=_ramp)._fingerprint()
    # the same function defined anew under its name, as in another process
    twin = type(_ramp)(_ramp.__code__, globals(), "_ramp")
    monkeypatch.setitem(globals(), "_ramp", twin)
    assert priors.PolyaTree(base=twin)._fingerprint() == key
    # lambdas have no name that finds them
    lambdas = [lambda z: 1.0 + z, lambda z: 1.0 + z]
    assert len({priors.PolyaTree(base=f)._fingerprint() for f in lambdas}) == 2


def test_population_accepts_a_density():
    bins = Bins.linear(0, 1, 5)
    d = Density.from_masses([1, 2, 3, 2, 1])
    np.testing.assert_allclose(binference.simulate.population(bins, d), d.masses(bins))


def test_every_prior_default_is_flat_in_the_measure_coordinate():
    logbins = Bins.log(1.0, 100.0, 4)
    flat = Density.flat().masses(logbins)
    alpha = priors.Dirichlet.from_base_measure(1.0).resolve_alpha(logbins)
    np.testing.assert_allclose(alpha, flat)
    mean = _resolve_mean(None, logbins)  # the logistic priors' default mean
    np.testing.assert_allclose(np.exp(mean) / np.exp(mean).sum(), flat)


def test_a_numeric_callable_on_integer_categories_gets_numbers():
    d = Density.from_function(lambda k: np.exp(-k))
    w = np.exp(-np.arange(3.0))
    np.testing.assert_allclose(d.masses(Bins.categories(3)), w / w.sum())


def test_tiny_unequal_bins_still_need_a_representation():
    tiny = Bins([0.0, 1e-9, 3e-9])
    with pytest.raises(ValueError, match="ambiguous"):
        Density.coerce([1.0, 1.0]).masses(tiny)


def test_the_interim_uses_the_constructor_quadrature():
    bins = Bins.linear(0, 1, 2)
    interim = lambda z: np.exp(8 * z)  # noqa: E731
    lk = Likelihoods.from_density(lambda z: np.exp(8 * z)[None, :], bins, n_sub=2, interim=interim)
    np.testing.assert_allclose(lk.toarray(), 1.0, rtol=1e-12)  # the interim divides out exactly
