"""Tests for pastry.Likelihoods and its constructors (spec 3.4-3.5, 4.2; R-INPUT-*)."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import stats

import pastry
from pastry import Bins, Likelihoods
from pastry.warnings import InputWarning


def test_basic_construction_and_scales():
    L = np.array([[1.0, 2.0, 4.0], [0.0, 3.0, 1.0]])
    lk = Likelihoods(L)
    assert lk.N == 2 and lk.K == 3 and len(lk) == 2
    assert lk.bins.is_categorical
    np.testing.assert_allclose(lk.values.max(axis=1), 1.0)
    np.testing.assert_allclose(lk.toarray() * np.exp(lk.log_scale)[:, None], L)
    np.testing.assert_allclose(lk.weights, 1.0)
    assert lk.W == 2.0
    assert lk.nnz == 5
    assert "Likelihoods(N=2" in repr(lk)


def test_sparse_input_matches_dense():
    L = np.array([[1.0, 0.0, 4.0], [0.0, 3.0, 1.0], [2.0, 2.0, 0.0]])
    a = Likelihoods(L)
    b = Likelihoods(sp.csr_array(L))
    assert b.is_sparse
    np.testing.assert_allclose(b.toarray(), a.toarray())
    np.testing.assert_allclose(b.log_scale, a.log_scale)
    np.testing.assert_allclose(a.to_sparse().toarray(), a.toarray())
    np.testing.assert_allclose(b.to_dense().toarray(), a.toarray())


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[1.0, -1.0]]),
        np.array([[1.0, np.nan]]),
        np.array([[1.0, np.inf]]),
        np.array([1.0, 2.0]),
        np.array([[1.0]]),
    ],
)
def test_invalid_values(bad):
    with pytest.raises(ValueError):
        Likelihoods(bad)


def test_empty_rows_raise_or_drop():
    L = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 1.0]])
    with pytest.raises(ValueError, match="zero likelihood"):
        Likelihoods(L)
    with pytest.warns(InputWarning, match="Dropping"):
        lk = Likelihoods(L, drop_empty=True)
    assert lk.N == 2
    np.testing.assert_array_equal(lk.meta["dropped"], [1])


def test_weights_validation():
    L = np.ones((3, 2))
    with pytest.raises(ValueError):
        Likelihoods(L, weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        Likelihoods(L, weights=[1.0, 0.0, 1.0])
    assert Likelihoods(L, weights=2.0).W == 6.0


def test_from_log_stable():
    logL = np.array([[-1e4, -1e4 + 1.0], [0.0, -np.inf]])
    lk = Likelihoods.from_log(logL)
    np.testing.assert_allclose(lk.toarray()[0], [np.exp(-1.0), 1.0])
    np.testing.assert_allclose(lk.toarray()[1], [1.0, 0.0])
    np.testing.assert_allclose(lk.log_scale, [-1e4 + 1.0, 0.0])
    with pytest.raises(ValueError):
        Likelihoods.from_log(np.array([[np.nan, 0.0]]))


def test_mass_vs_density_representation():
    bins = Bins([0.0, 1.0, 3.0])  # unequal widths
    dens = np.array([[2.0, 1.0]])
    mass = dens * bins.volumes
    a = Likelihoods(dens, bins, representation="density")
    b = Likelihoods(mass, bins, representation="mass")
    np.testing.assert_allclose(a.toarray(), b.toarray())
    with pytest.raises(ValueError):
        Likelihoods(dens, bins, representation="per-bin")


def test_from_posteriors_requires_interim_prior():
    bins = Bins.linear(0, 1, 3)
    with pytest.raises(TypeError):
        Likelihoods.from_posteriors(np.ones((2, 3)), bins)  # missing interim prior


def test_from_posteriors_removes_interim_prior(rng):
    bins = Bins([0.0, 0.5, 1.0, 2.0, 4.0])
    L = rng.uniform(0.1, 1.0, size=(20, 4))
    pi = np.array([0.1, 0.2, 0.3, 0.4])
    P = L * pi
    P /= P.sum(axis=1, keepdims=True)
    lk = Likelihoods.from_posteriors(P, bins, pi)
    ratio = lk.toarray() / L
    np.testing.assert_allclose(ratio / ratio[:, :1], 1.0, rtol=1e-12)
    # density representations give the same result
    lk2 = Likelihoods.from_posteriors(
        P / bins.volumes,
        bins,
        pi / bins.volumes,
        representation="density",
        interim_representation="density",
    )
    np.testing.assert_allclose(lk2.toarray(), lk.toarray(), rtol=1e-12)
    # a callable density that is constant within bins gives the same masses
    dens = pi / bins.volumes
    lk3 = Likelihoods.from_posteriors(P, bins, lambda z: dens[bins.index(z)])
    np.testing.assert_allclose(lk3.toarray(), lk.toarray(), rtol=1e-10)
    # "flat" means masses proportional to bin volumes
    lk4 = Likelihoods.from_posteriors(P, bins, "flat")
    expected = P / bins.volumes
    np.testing.assert_allclose(lk4.toarray(), expected / expected.max(1, keepdims=True))


def test_from_posteriors_zero_interim_prior():
    bins = Bins.linear(0, 1, 3)
    with pytest.raises(ValueError, match="interim prior is zero"):
        Likelihoods.from_posteriors(np.array([[0.5, 0.5, 0.0]]), bins, [0.0, 0.5, 0.5])
    lk = Likelihoods.from_posteriors(np.array([[0.0, 0.5, 0.5]]), bins, [0.0, 0.5, 0.5])
    np.testing.assert_allclose(lk.toarray(), [[0.0, 1.0, 1.0]])


def test_representation_warning_on_unequal_bins():
    bins = Bins([0.0, 0.1, 1.0])
    dens = np.array([[5.0, 0.5555555555555556]])  # integrates to one as a density
    with pytest.warns(InputWarning, match="density"):
        Likelihoods.from_posteriors(dens, bins, "flat", representation="mass")


def test_unequal_bins_representation_matters():
    """Unequal bins: the right representation recovers the population, the wrong one biases it."""
    rng = np.random.default_rng(0)
    bins = Bins([0.0, 0.2, 0.4, 1.0, 2.0])
    rho = np.array([0.1, 0.2, 0.3, 0.4])
    mock = pastry.simulate.mock_1d(20000, bins, density=rho, noise=0.15, rng=rng)
    good = pastry.infer(mock.likelihoods, method="map").rho
    np.testing.assert_allclose(good, rho, atol=0.025)
    # declaring bin-averaged likelihoods as masses divides columns by the volumes,
    # which biases the population when PDFs straddle bins of unequal width
    bad_lk = Likelihoods(mock.likelihoods.toarray(), bins, representation="mass")
    bad = pastry.infer(bad_lk, method="map").rho
    assert np.max(np.abs(bad - rho)) > 0.08


def test_from_samples_histogram_and_meta(rng):
    bins = Bins.linear(0, 1, 4)
    samples = [rng.uniform(0, 1, 400), rng.uniform(0.0, 0.5, 300), np.array([0.1, 0.2, 1.5])]
    with pytest.warns(InputWarning):  # third object has few samples (median check)
        lk = Likelihoods.from_samples(samples, bins, min_samples=500)
    assert lk.N == 3
    assert lk.meta["source"] == "samples"
    np.testing.assert_allclose(lk.meta["n_eff"], [400, 300, 3])
    np.testing.assert_allclose(lk.meta["outside_fraction"][2], 1 / 3)
    # object 1 has no samples above 0.5
    np.testing.assert_allclose(lk.toarray()[1, 2:], 0.0)


def test_from_samples_flat_vs_interim(rng):
    bins = Bins([0.0, 1.0, 3.0])
    samples = rng.uniform(0, 3, size=(5, 40000))
    lk = Likelihoods.from_samples(samples, bins, "flat")
    # uniform samples under a flat interim prior -> flat likelihood (up to Monte Carlo noise)
    np.testing.assert_allclose(lk.toarray(), 1.0, atol=0.05)
    lk_s = Likelihoods.from_samples(samples, bins, sparse=True)
    assert lk_s.is_sparse
    # with weights
    lk_w = Likelihoods.from_samples(samples, bins, sample_weights=np.ones_like(samples))
    np.testing.assert_allclose(lk_w.toarray(), lk.toarray())


def test_from_samples_2d(rng):
    bins = Bins.grid([0, 0.5, 1.0], [0, 0.5, 1.0])
    s = rng.uniform(0, 1, size=(3, 500, 2))
    lk = Likelihoods.from_samples(s, bins)
    assert lk.K == 4 and lk.N == 3


def test_from_gaussian_exact_1d():
    bins = Bins([0.0, 0.3, 1.0, 1.2])
    mu = np.array([0.5, 0.1])
    sig = np.array([0.2, 0.05])
    lk = Likelihoods.from_gaussian(mu, sig, bins)
    e = bins.edges[0]
    mass = stats.norm.cdf(e[None, 1:], mu[:, None], sig[:, None]) - stats.norm.cdf(
        e[None, :-1], mu[:, None], sig[:, None]
    )
    expected = mass / bins.volumes
    expected /= expected.max(axis=1, keepdims=True)
    # (pastry's tail computation is more accurate than CDF differences, which underflow)
    np.testing.assert_allclose(lk.toarray(), expected, rtol=1e-9, atol=1e-60)


def test_from_gaussian_far_tails_are_finite():
    bins = Bins.linear(0, 1, 5)
    lk = Likelihoods.from_gaussian([40.0], [0.1], bins)
    assert np.all(np.isfinite(lk.log_scale))
    assert np.argmax(lk.toarray()[0]) == 4


def test_from_gaussian_2d_is_product():
    bins = Bins.grid([0, 0.5, 1.0], [0, 0.3, 1.0])
    mu = np.array([[0.2, 0.6]])
    sig = np.array([[0.1, 0.2]])
    lk = Likelihoods.from_gaussian(mu, sig, bins)
    ax = Likelihoods.from_gaussian(mu[:, 0], sig[:, 0], Bins([0, 0.5, 1.0])).toarray()[0]
    ay = Likelihoods.from_gaussian(mu[:, 1], sig[:, 1], Bins([0, 0.3, 1.0])).toarray()[0]
    expected = np.outer(ax, ay).ravel()
    np.testing.assert_allclose(lk.toarray()[0], expected / expected.max(), rtol=1e-9)
    # per-axis shared sigma broadcasts
    lk2 = Likelihoods.from_gaussian(mu, np.array([0.1, 0.2]), bins)
    np.testing.assert_allclose(lk2.toarray(), lk.toarray())
    with pytest.raises(ValueError):
        Likelihoods.from_gaussian(mu, [[0.0, 0.1]], bins)


def test_from_distributions_matches_gaussian():
    bins = Bins([0.0, 0.3, 1.0, 1.2])
    mu, sig = np.array([0.5, 0.1]), np.array([0.2, 0.05])
    a = Likelihoods.from_gaussian(mu, sig, bins)
    b = Likelihoods.from_distributions(
        [stats.norm(m, s) for m, s in zip(mu, sig, strict=True)], bins
    )
    np.testing.assert_allclose(b.toarray(), a.toarray(), rtol=1e-9, atol=1e-60)

    class OnlyLogpdf:
        def __init__(self, d):
            self.d = d

        def logpdf(self, x):
            return self.d.logpdf(x)

    c = Likelihoods.from_distributions(
        [OnlyLogpdf(stats.norm(m, s)) for m, s in zip(mu, sig, strict=True)], bins, n_sub=12
    )
    np.testing.assert_allclose(c.toarray(), a.toarray(), rtol=2e-3, atol=1e-6)
    with pytest.raises(ValueError):
        Likelihoods.from_distributions([stats.norm(0, 1)], bins, kind="posterior")


def test_from_function_quadrature_converges():
    bins = Bins([0.0, 0.3, 1.0, 1.2])
    mu, sig = np.array([0.5, 0.1]), np.array([0.2, 0.05])
    a = Likelihoods.from_gaussian(mu, sig, bins)

    def f(z):
        return stats.norm.logpdf(z[None, :], mu[:, None], sig[:, None])

    b = Likelihoods.from_function(f, bins, n_sub=16, log=True)
    np.testing.assert_allclose(b.toarray(), a.toarray(), rtol=1e-4, atol=1e-8)
    c = Likelihoods.from_function(lambda z: np.exp(f(z)), bins, n_sub=16)
    np.testing.assert_allclose(c.toarray(), a.toarray(), rtol=1e-4, atol=1e-8)


def test_from_response():
    R = np.array([[0.7, 0.1], [0.2, 0.6], [0.05, 0.2]])  # columns sum < 1 (efficiency)
    counts = np.array([10, 0, 3])
    lk = Likelihoods.from_response(counts, R)
    assert lk.N == 2
    np.testing.assert_allclose(lk.weights, [10, 3])
    np.testing.assert_allclose(lk.toarray(), R[[0, 2]] / R[[0, 2]].max(1, keepdims=True))
    assert lk.selection.mode == "observed"
    np.testing.assert_allclose(lk.selection.values, R.sum(0))
    assert (
        Likelihoods.from_response(counts, R / R.sum(0), efficiency="from_response").selection
        is None
    )
    assert Likelihoods.from_response(counts, R, efficiency=None).selection is None


def test_from_classifier():
    p = np.array([[0.7, 0.3], [0.2, 0.8]])
    pi = np.array([0.8, 0.2])
    lk = Likelihoods.from_classifier(p, pi, labels=["a", "b"])
    expected = p / pi
    np.testing.assert_allclose(lk.toarray(), expected / expected.max(1, keepdims=True))
    assert lk.bins.labels == ("a", "b")


def test_compress_preserves_inference():
    L = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 1.0, 1.0], [2.0, 4.0, 6.0]])
    lk = Likelihoods(L)
    c = lk.compress()
    assert c.N == 2
    np.testing.assert_allclose(np.sort(c.weights), [1.0, 3.0])
    rho = np.array([0.2, 0.3, 0.5])
    m1 = pastry.PopulationModel(lk)
    m2 = pastry.PopulationModel(c)
    assert m1.log_likelihood(rho) == pytest.approx(m2.log_likelihood(rho), rel=1e-12)
    assert lk.subset([0, 2]).compress() is not None


def test_subset_getitem_with_weights():
    lk = Likelihoods(np.ones((4, 2)) + np.arange(4)[:, None] * np.array([0.0, 1.0]))
    s = lk[1:3]
    assert s.N == 2
    w = lk.with_weights([1, 2, 3, 4])
    np.testing.assert_allclose(w.subset([3]).weights, [4.0])


def test_selection_requires_mode():
    with pytest.raises(ValueError, match="selection_mode"):
        Likelihoods(np.ones((2, 2)), selection=[0.5, 1.0])
    lk = Likelihoods(np.ones((2, 2)), selection=[0.5, 1.0], selection_mode="true")
    assert lk.selection.mode == "true"
    assert lk.with_selection(None).selection is None
    with pytest.raises(ValueError):
        Likelihoods(np.ones((2, 2)), selection=[0.0, 1.0], selection_mode="true")
    with pytest.raises(ValueError):
        pastry.Selection([0.5, 1.0], mode="maybe")


def test_stack_one_em_step():
    bins = Bins.linear(0, 1, 3)
    lk = Likelihoods(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]), bins)
    np.testing.assert_allclose(lk.stack(), [0.25, 0.5, 0.25])
