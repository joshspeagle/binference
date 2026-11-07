"""Tests for the mixture basis and internal resolution (DECISIONS #4, #5, #47, #76)."""

import itertools
import json
import pickle

import numpy as np
import pytest
from scipy import integrate, stats

import binference
from binference import Basis, Bins, Likelihoods, PointEstimate, PopulationModel, PopulationResult
from binference import priors as P
from binference.engines import MAP


def _hats(bins, z):
    # brute-force order-1 shapes in 1-D, linear bins: hats at the centres, constant ends
    c = bins.centers
    H = np.array([np.interp(z, c, np.eye(bins.K)[k]) for k in range(bins.K)])
    return H / np.trapezoid(H, z, axis=1)[:, None]


def test_hat_masses_are_exact_and_columns_sum_to_one():
    A = Basis(Bins.linear(0, 3, 4), order=1).matrix.toarray()
    # the first hat is 1 on [0, 0.375] then falls to 0 at 1.125: 0.65625 of its 0.75 in bin 0
    np.testing.assert_allclose(A[:, 0], [0.875, 0.125, 0, 0], atol=1e-15)
    np.testing.assert_allclose(A[:, 1], [0.125, 0.75, 0.125, 0], atol=1e-15)
    basis = Basis(Bins.log(1, 100, 5), order=1, resolution=2)
    np.testing.assert_allclose(basis.grid_masses.sum(axis=0), 1.0)
    np.testing.assert_allclose(basis.matrix.sum(axis=0), 1.0)


def test_masked_grids_take_flat_components_only():
    G = Bins.grid([0, 1, 2, 3], [0, 1, 2], mask=np.array([[1, 1], [1, 0], [1, 1]], bool))
    basis = Basis(G, resolution=2)
    assert basis.matrix.shape == (5, 20)
    np.testing.assert_allclose(basis.matrix.sum(axis=0), 1.0)
    # next to a masked cell the hats would not sum to one, biasing the bins there
    with pytest.raises(ValueError, match="unmasked"):
        Basis(G, order=1)


def test_resolution_is_checked_by_name():
    for bad in (0, 1.5, (2, 2), "x"):
        with pytest.raises(ValueError, match="resolution"):
            Basis(Bins.linear(0, 1, 4), resolution=bad)


def test_refine_and_coarsen_are_inverse():
    for bins in (
        Bins.log(1, 100, 4),
        Bins.grid([0, 1, 2], [0, 1, 2, 3], mask=[[1, 0, 1], [1, 1, 1]]),
    ):
        fine, fmap = bins.refine(3 if bins.ndim == 1 else [2, 1])
        assert fine.coarsen(3 if bins.ndim == 1 else [2, 1])[0] == bins
        np.testing.assert_array_equal(np.bincount(fmap), [3 if bins.ndim == 1 else 2] * bins.K)
    with pytest.raises(ValueError, match="factor"):
        Bins.linear(0, 1, 3).refine(0)


def test_the_flat_basis_at_resolution_one_changes_nothing():
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 8), rng=1)
    d, s = mock.measurements["mean"], mock.measurements["sigma"]
    plain = Likelihoods.from_gaussian(d, s, mock.likelihoods.bins)
    on_basis = Likelihoods.from_gaussian(d, s, Basis(mock.likelihoods.bins))
    np.testing.assert_array_equal(on_basis.values, plain.values)
    np.testing.assert_array_equal(
        MAP(PopulationModel(on_basis)).run().rho, MAP(PopulationModel(plain)).run().rho
    )


def test_flat_components_are_the_refined_bins_summed_back():
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 6), rng=2)
    d, s = mock.measurements["mean"], mock.measurements["sigma"]
    basis = Basis(mock.likelihoods.bins, resolution=3)
    fine, fmap = mock.likelihoods.bins.refine(3)
    lk, lk_fine = Likelihoods.from_gaussian(d, s, basis), Likelihoods.from_gaussian(d, s, fine)
    np.testing.assert_allclose(lk.values, lk_fine.values, rtol=1e-12)
    prior = P.LogisticGP(length_scale=1.0)
    rho = MAP(PopulationModel(lk, prior)).run().rho
    w = MAP(PopulationModel(lk_fine, prior)).run().rho
    np.testing.assert_allclose(rho, np.bincount(fmap, weights=w), atol=1e-8)


@pytest.mark.filterwarnings("ignore:Non-integer weights:binference.InputWarning")
def test_resolution_and_hats_remove_the_binning_bias():
    # the large-N limit: data on a fine grid weighted by their exact density, so the
    # fit has no noise and its error is the binning bias alone (DECISIONS #5)
    beta, s = 1.5, 0.2
    bins = Bins.linear(0, 3, 10)
    f = lambda x: beta * np.exp(-beta * x) / (1 - np.exp(-beta * 3))  # noqa: E731
    e = bins.edges[0]
    truth = np.array([integrate.quad(f, a, b)[0] for a, b in itertools.pairwise(e)])
    zq = np.linspace(0, 3, 6001)
    d = np.linspace(-0.8, 3.8, 1500)
    pd = np.trapezoid(f(zq)[None] * stats.norm.pdf(d[:, None], zq[None], s), zq, axis=1)
    err = {}
    for order, res in [(0, 1), (1, 1), (1, 4)]:
        basis = Basis(bins, order=order, resolution=res)
        lk = Likelihoods.from_gaussian(
            d,
            np.full(d.size, s),
            basis,
            weights=1e4 * pd / pd.sum(),
            fractional_weights=True,
            warn_outside=False,  # data beyond the range, as the population convolved
        )
        rho = MAP(PopulationModel(lk, P.LogisticRandomWalk(order=2))).run().rho
        err[order, res] = np.max(np.abs(rho / truth - 1))
    # measured: 7.4%, 3.3% and 0.11%
    assert err[0, 1] > 0.05 and err[1, 1] < 0.6 * err[0, 1] and err[1, 4] < 0.005


def test_a_selection_function_is_averaged_with_each_component_as_weight():
    bins = Bins.linear(0, 3, 8)
    basis = Basis(bins, order=1)
    S = lambda z: 1 / (1 + np.exp(-3 * (np.asarray(z) - 1.0)))  # noqa: E731
    z = np.linspace(0, 3, 300001)
    brute = np.trapezoid(_hats(bins, z) * S(z), z, axis=1)
    sel = binference.Selection.from_function(S, basis, "on_truth")
    # the grid cells average S, then the hats weight them: an O(h^2) error, h = 3/64
    np.testing.assert_allclose(sel.values, brute, atol=1e-3)
    assert sel.basis == basis
    per_bin = np.linspace(0.2, 0.9, 8)
    sel = binference.selection.as_selection(per_bin, "on_measurement", basis)
    np.testing.assert_allclose(sel.values, basis.matrix.T @ per_bin)
    assert binference.selection.as_selection(sel, None, basis) is sel  # not mapped twice
    with pytest.raises(ValueError, match="different basis"):
        binference.selection.as_selection(sel, None, Basis(bins, order=1, resolution=2))


def test_per_bin_likelihoods_move_onto_a_basis_as_flat_within_bins():
    bins = Bins.linear(0, 1, 4)
    basis = Basis(bins, order=1)
    L = np.array([[1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 3.0, 1.0]])
    lk = Likelihoods(L, bins).on_basis(basis)
    expect = L @ basis.matrix.toarray()
    np.testing.assert_allclose(lk.toarray(), expect / expect.max(axis=1, keepdims=True))
    np.testing.assert_allclose(Likelihoods(L, basis).toarray(), lk.toarray())
    with pytest.raises(ValueError, match="already"):
        lk.on_basis(basis)
    with pytest.raises(ValueError, match="bins"):
        Likelihoods(L, Bins.linear(0, 2, 4)).on_basis(basis)


@pytest.mark.parametrize("route", ["gaussian", "loglike"])
def test_values_drawn_inside_hats_follow_the_population(route):
    # broad likelihoods: an object's value then follows the population density itself
    bins = Bins.linear(0, 3, 6)
    basis = Basis(bins, order=1)
    n = 4000
    lk = Likelihoods.from_gaussian(np.full(n, 1.5), np.full(n, 50.0), basis, warn_outside=False)
    w = np.array([0.0, 0.7, 0.0, 0.0, 0.3, 0.0])  # separate hats: triangles, not boxes
    est = binference.PointEstimate(PopulationModel(lk), w)
    loglike = (lambda z, i: -0.5 * ((z - 1.5) / 50.0) ** 2) if route == "loglike" else None
    v = est.sample_values(rng=0, loglike=loglike, n_sub=256)[0]
    z = np.linspace(0, 3, 30001)
    cdf = np.concatenate([[0], integrate.cumulative_trapezoid(w @ _hats(bins, z), z)])
    assert stats.kstest(v, lambda x: np.interp(x, z, cdf)).pvalue > 1e-3


def test_hat_labels_are_shared_among_bins_by_mass():
    # broad likelihoods: an object's component follows the weights, its bin then A @ w
    bins = Bins.linear(0, 3, 6)
    basis = Basis(bins, order=1)
    n = 20000
    lk = Likelihoods.from_gaussian(np.full(n, 1.5), np.full(n, 50.0), basis, warn_outside=False)
    w = np.array([0.05, 0.3, 0.25, 0.2, 0.15, 0.05])
    labels = binference.PointEstimate(PopulationModel(lk), w).sample_assignments(rng=1)[0]
    p = basis.to_bins(w)
    freq = np.bincount(labels, minlength=6) / n
    assert np.all(np.abs(freq - p) < 5 * np.sqrt(p * (1 - p) / n))


def test_results_report_bins_and_keep_component_draws(tmp_path):
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 6), rng=3)
    d, s = mock.measurements["mean"], mock.measurements["sigma"]
    basis = Basis(mock.likelihoods.bins, order=1, resolution=2)
    lk = Likelihoods.from_gaussian(d, s, basis)
    res = binference.infer(lk, method="gibbs", n_samples=100, n_warmup=50, rng=0, check=False)
    assert res.bins == basis.bins and res.samples.shape[1] == 6
    np.testing.assert_allclose(res.samples, basis.to_bins(res.component_draws.reshape(-1, 12)))
    counts = res.realized_counts(n_draws=3, rng=1)
    assert np.all(counts == np.round(counts)) and np.allclose(counts.sum(axis=1), 300)
    np.testing.assert_allclose(res.assignment_probabilities(idx=[0, 1]).sum(axis=1), 1.0)
    path = tmp_path / "r.npz"
    res.save(str(path))
    back = PopulationResult.load(str(path))
    assert back.basis == basis
    np.testing.assert_allclose(back.mean, res.mean)
    back.attach(lk).assignment_probabilities(idx=[0])
    with pytest.raises(ValueError, match="basis"):
        back.attach(Likelihoods.from_gaussian(d, s, mock.likelihoods.bins))


def test_sample_likelihoods_moved_onto_a_basis_keep_their_monte_carlo_error():
    rng = np.random.default_rng(4)
    bins = Bins.linear(0, 3, 6)
    samples = rng.normal(rng.uniform(0.5, 2.5, 50)[:, None], 0.3, (50, 400))
    flat = Likelihoods.from_samples(samples, bins, interim="flat")
    basis = Basis(bins, resolution=2)
    w = np.full(12, 1 / 12)
    for lk in (flat.on_basis(basis), Likelihoods.from_samples(samples, basis, interim="flat")):
        var, _ = lk.sample_variance(w)
        assert var.shape == (50,) and np.all(np.isfinite(var))


def test_an_interim_array_per_bin_works_with_a_basis():
    rng = np.random.default_rng(5)
    bins = Bins.linear(0, 3, 6)
    samples = rng.normal(rng.uniform(0.5, 2.5, 30)[:, None], 0.3, (30, 300))
    basis = Basis(bins, order=1)
    a = Likelihoods.from_samples(samples, basis, interim=np.full(6, 1 / 6))
    b = Likelihoods.from_samples(samples, basis, interim="flat")
    np.testing.assert_allclose(a.toarray(), b.toarray(), rtol=1e-12)


def test_the_stack_on_a_basis_is_in_the_model_parameters():
    mock = binference.simulate.mock_1d(200, Bins.linear(0, 3, 6), rng=6)
    basis = Basis(mock.likelihoods.bins, order=1, resolution=2)
    lk = Likelihoods.from_gaussian(mock.measurements["mean"], mock.measurements["sigma"], basis)
    st = lk.stack()
    assert st.shape == (12,)
    assert np.isfinite(PopulationModel(lk).kkt_gap(st))


def test_values_on_log_bins_are_flat_in_log_within_bins():
    # broad likelihoods: values follow the population, flat in log z within each bin
    bins = Bins.log(1.0, 100.0, 4)
    n = 4000
    lk = Likelihoods.from_gaussian(np.full(n, 50.0), np.full(n, 1e4), bins, warn_outside=False)
    rho = np.array([0.1, 0.4, 0.3, 0.2])
    v = binference.PointEstimate(PopulationModel(lk), rho).sample_values(rng=0)[0]
    e = np.log(bins.edges[0])
    cdf = lambda x: np.interp(np.log(x), e, np.concatenate([[0], np.cumsum(rho)]))  # noqa: E731
    assert stats.kstest(v, cdf).pvalue > 1e-3


def test_a_reloaded_result_attaches_on_a_transform_axis(tmp_path):
    tb = Bins(np.linspace(0, 3, 7), measure=binference.Measure.transform(np.arcsinh, np.sinh))
    mock = binference.simulate.mock_1d(100, tb, rng=7)
    lk = Likelihoods.from_gaussian(
        mock.measurements["mean"], mock.measurements["sigma"], Basis(tb, order=1)
    )
    res = binference.infer(lk, method="gibbs", n_samples=20, n_warmup=10, rng=0, check=False)
    res.save(str(tmp_path / "r.npz"))
    back = PopulationResult.load(str(tmp_path / "r.npz"))
    back.attach(lk).assignment_probabilities(idx=[0])
    assert back.to_dict()["meta"]["basis"]["order"] == 1


def test_a_reloaded_result_with_a_selection_attaches_on_a_transform_axis(tmp_path):
    # the saved selection's basis loads with a tabulated measure, as the result's does
    tb = Bins(np.linspace(0, 3, 7), measure=binference.Measure.transform(np.arcsinh, np.sinh))
    mock = binference.simulate.mock_1d(100, tb, rng=7)
    lk = Likelihoods.from_gaussian(
        mock.measurements["mean"], mock.measurements["sigma"], Basis(tb, order=1)
    )
    sel = np.linspace(0.5, 1.0, tb.K)
    res = binference.infer(
        lk,
        selection=sel,
        selection_mode="on_truth",
        method="gibbs",
        n_samples=20,
        n_warmup=10,
        rng=0,
        check=False,
    )
    res.save(str(tmp_path / "r.npz"))
    back = PopulationResult.load(str(tmp_path / "r.npz"))
    back.attach(lk).assignment_probabilities(idx=[0])


def _bins_of_every_kind():
    mask = np.ones(8, bool)
    mask[[0, 5]] = False
    return {
        "categories": Bins.categories(["a", "b", "c", "d", "e", "f"]),
        "linear": Bins.linear(0, 3, 6),
        "log": Bins.log(0.1, 10, 6),
        "masked": Bins.linear(0, 4, 8, mask=mask),
        "grid": Bins.grid([0, 1, 2, 3], [0, 1, 2]),
    }


@pytest.mark.parametrize("kind", _bins_of_every_kind())
def test_plain_bins_are_the_identity_basis_of_every_kind_of_bins(kind):
    bins = _bins_of_every_kind()[kind]
    basis = Basis(bins)
    assert basis.is_identity and basis.components is bins and basis.grid is bins
    np.testing.assert_array_equal(basis.grid_to_bins, np.arange(bins.K))
    np.testing.assert_array_equal(basis.matrix.toarray(), np.eye(bins.K))
    np.testing.assert_array_equal(basis.grid_masses.toarray(), np.eye(bins.K))
    np.testing.assert_array_equal(basis.to_bins(np.eye(bins.K)), np.eye(bins.K))
    assert pickle.loads(pickle.dumps(basis)) == basis

    values = np.random.default_rng(0).random((40, bins.K))
    lk = Likelihoods(values, bins)
    assert lk.basis == basis and lk.bins is bins and lk.subset([0, 1]).basis == basis
    res = binference.infer(lk, method="map")
    assert res.basis == basis and res.bins == bins
    draws = binference.infer(lk, method="gibbs", n_samples=20, n_warmup=10, rng=0, check=False)
    assert draws.basis == basis and draws.bins == bins
    assert draws.component_draws is draws._params and draws.chains.shape[-1] == bins.K


def test_a_result_without_a_model_has_the_identity_basis_of_its_bins():
    draws = np.random.default_rng(1).dirichlet(np.ones(4), 5)
    assert PopulationResult(None, draws, method="gibbs").basis == Basis(Bins.categories(4))
    bins = Bins.linear(0, 1, 4)
    assert PopulationResult(None, draws, method="gibbs", bins=bins).basis == Basis(bins)
    assert PointEstimate(None, draws[0]).basis == Basis(Bins.categories(4))


@pytest.mark.parametrize(
    "options", [{"order": 1}, {"resolution": 2}, {"oversample": 2}], ids=lambda o: next(iter(o))
)
def test_categories_take_only_the_identity_basis(options):
    with pytest.raises(ValueError, match="needs a grid"):
        Basis(Bins.categories(3), **options)


def test_categories_have_no_geometry_for_a_constructor_to_use():
    with pytest.raises(ValueError, match="ordered"):
        Likelihoods.from_gaussian([0.5], [0.1], Basis(Bins.categories(3)))


@pytest.mark.filterwarnings("ignore::binference.warnings.InputWarning")  # raw values
@pytest.mark.parametrize("kind", ["linear", "log", "masked"])
def test_an_explicit_identity_basis_gives_the_fit_of_its_plain_bins(kind):
    bins = _bins_of_every_kind()[kind]
    rng = np.random.default_rng(2)
    lo, hi = bins.range[0]
    mean, sigma = rng.uniform(lo, hi, 200), rng.uniform(0.2, 0.5, 200)
    samples = np.clip(mean[:, None] + sigma[:, None] * rng.normal(size=(200, 30)), lo, hi)
    S = np.linspace(0.3, 0.9, bins.K)
    selection = {"selection": S, "selection_mode": "on_truth"}
    make = {
        "gaussian": lambda b: Likelihoods.from_gaussian(mean, sigma, b, **selection),
        "samples": lambda b: Likelihoods.from_samples(
            samples, b, "flat", drop_empty=True, **selection
        ),
        "values": lambda b: Likelihoods(rng_values, b, **selection),
    }
    rng_values = rng.random((50, bins.K))
    for build in make.values():
        plain, explicit = build(bins), build(Basis(bins))
        np.testing.assert_array_equal(plain.values, explicit.values)
        np.testing.assert_array_equal(plain.log_scale, explicit.log_scale)
        assert plain.selection == explicit.selection and plain.basis == explicit.basis
        assert plain.meta.keys() == explicit.meta.keys()
        fit = {"method": "gibbs", "n_samples": 30, "n_warmup": 10, "rng": 3, "check": False}
        a, b = binference.infer(plain, **fit), binference.infer(explicit, **fit)
        np.testing.assert_array_equal(a.chains, b.chains)
        np.testing.assert_array_equal(
            a.sample_values(n_draws=2, idx=[0, 1], rng=4),
            b.sample_values(n_draws=2, idx=[0, 1], rng=4),
        )


def test_a_selection_on_an_identity_basis_is_a_selection_per_bin():
    bins = Bins.linear(0, 3, 6)
    S = lambda z: 0.2 + 0.2 * np.asarray(z)  # noqa: E731
    plain = binference.Selection.from_function(S, bins, "on_truth")
    assert binference.Selection.from_function(S, Basis(bins), "on_truth") == plain
    assert plain.basis is None
    assert binference.selection.as_selection(plain, None, Basis(bins)) is plain


def test_a_selection_from_injections_passes_an_identity_basis_unchanged():
    bins = Bins.linear(0, 3, 6)
    z = np.linspace(0, 3, 90, endpoint=False)
    sel = binference.Selection.from_injections(z, z > 1, bins, "on_truth")
    assert Likelihoods(np.ones((3, 6)), bins, selection=sel).selection is sel
    with pytest.raises(ValueError, match="injections"):
        Likelihoods(np.ones((3, 6)), Basis(bins, order=1), selection=sel)


def test_values_drawn_on_an_oversampled_identity_basis_come_from_their_objects_samples():
    # the samples are binned on the quadrature cells, so a label is not their bin
    bins = Bins.linear(0, 4, 4)
    samples = np.full((2, 40), [[1.21], [2.83]])
    lk = Likelihoods.from_samples(samples, Basis(bins, oversample=3), "flat", sparse=False)
    res = binference.infer(lk, method="gibbs", n_samples=20, n_warmup=10, rng=0, check=False)
    z = res.sample_values(n_draws=5, rng=1)
    np.testing.assert_array_equal(z[:, 0], 1.21)
    np.testing.assert_array_equal(z[:, 1], 2.83)


def test_likelihoods_on_an_oversampled_identity_basis_are_already_on_a_basis():
    bins = Bins.linear(0, 3, 6)
    lk = Likelihoods(np.ones((3, 6)), Basis(bins, oversample=2))
    with pytest.raises(ValueError, match="already on a basis"):
        lk.on_basis(Basis(bins, order=1))


def _save_old_layout(path, bins, draws, basis=None, selection=None):
    # a file as 0.1.0 wrote it: a plain fit has neither "basis" nor component draws
    meta = {
        "bins": "grid",
        "ndim": 1,
        "measures": ["linear"],
        "schema": 1,
        "binference_version": "0.1.0",
        "method": "gibbs",
        "kind": "fractions",
        "approximate": False,
        "prior": None,
        "prior_key": None,
        "info": {},
    }
    arrays = {"edges_0": bins.edges[0], "draws": draws, "total": np.full(draws.shape[:2], 9.0)}
    if basis is not None:
        meta["basis"] = basis
        arrays["component_draws"] = draws
    if selection is not None:
        meta["selection"] = {"scale": "probability", "row": "values"}
        arrays["selection_values"] = selection
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


@pytest.mark.parametrize(
    "saved", [None, {"order": 0, "resolution": [1], "oversample": 1}], ids=["plain", "explicit"]
)
def test_a_file_of_0_1_0_loads_on_the_identity_basis(tmp_path, saved):
    bins = Bins.linear(0, 3, 6)
    draws = np.random.default_rng(4).dirichlet(np.ones(6), (2, 5))
    S = np.linspace(0.3, 0.9, 6)
    path = tmp_path / "old.npz"
    _save_old_layout(path, bins, draws, basis=saved, selection=S)
    back = PopulationResult.load(str(path))
    assert back.basis == Basis(bins) and back.bins == bins
    np.testing.assert_array_equal(back.chains, draws)
    assert back.selection == binference.Selection(S, "on_truth")
    lk = Likelihoods(np.ones((3, 6)), bins, selection=back.selection)
    assert back.attach(lk).assignment_probabilities(idx=[0]).shape == (1, 6)


@pytest.mark.parametrize("kind", _bins_of_every_kind())
def test_a_saved_plain_fit_round_trips_without_basis_metadata(tmp_path, kind):
    bins = _bins_of_every_kind()[kind]
    lk = Likelihoods(np.random.default_rng(5).random((30, bins.K)), bins)
    res = binference.infer(lk, method="gibbs", n_samples=20, n_warmup=10, rng=0, check=False)
    assert "basis" not in res.to_dict()["meta"] and "component_draws" not in res.to_dict()
    path = tmp_path / "r.npz"
    res.save(str(path))
    back = PopulationResult.load(str(path))
    assert back.basis == res.basis and back.bins == bins
    np.testing.assert_array_equal(back.chains, res.chains)
    assert back.attach(lk).assignment_probabilities(idx=[0]).shape == (1, bins.K)


def test_a_saved_oversampled_identity_basis_keeps_its_oversampling(tmp_path):
    bins = Bins.linear(0, 3, 6)
    basis = Basis(bins, oversample=2)
    lk = Likelihoods.from_gaussian([1.0, 2.0], [0.3, 0.3], basis)
    res = binference.infer(lk, method="gibbs", n_samples=20, n_warmup=10, rng=0, check=False)
    path = tmp_path / "r.npz"
    res.save(str(path))
    assert PopulationResult.load(str(path)).basis == basis


def test_a_result_on_grid_bins_attaches_only_to_likelihoods_on_its_bins():
    draws = np.random.default_rng(6).dirichlet(np.ones(6), 5)
    res = PopulationResult(None, draws, method="gibbs", bins=Bins.grid([0, 1, 2], [0, 1, 2, 3]))
    with pytest.raises(ValueError, match="other bins"):
        res.attach(Likelihoods(np.ones((3, 6)), Bins.grid([0, 1, 2, 3], [0, 1, 2])))
    with pytest.raises(ValueError, match="other bins"):
        res.attach(Likelihoods(np.ones((3, 6))))


def test_new_likelihoods_of_a_plain_fit_must_be_on_its_bins():
    bins = Bins.linear(0, 3, 6)
    res = binference.infer(Likelihoods(np.ones((3, 6)), bins), method="map")
    assert res.predict_proba(Likelihoods(np.ones((2, 6)), Basis(bins))).shape == (2, 6)
    with pytest.raises(ValueError, match="same bins"):
        res.predict_proba(Likelihoods(np.ones((2, 6)), Bins.linear(0, 6, 6)))
    with pytest.raises(ValueError, match="same bins"):
        res.predict_proba(Likelihoods(np.ones((2, 6)), Basis(bins, order=1)))


def _plain_and_oversampled():
    bins = Bins.linear(0.0, 1.0, 4)
    values = np.random.default_rng(0).uniform(0.1, 1.0, (50, 4))
    return bins, Likelihoods(values, bins), Likelihoods(values, Basis(bins, oversample=2))


def test_an_explicit_plain_basis_on_plain_likelihoods_changes_nothing():
    bins, plain, _ = _plain_and_oversampled()
    assert plain.on_basis(Basis(bins)) is plain


def test_a_fit_and_likelihoods_on_different_quadrature_cells_do_not_mix():
    _, plain, oversampled = _plain_and_oversampled()
    res = binference.infer(
        oversampled, method="gibbs", n_samples=20, n_warmup=5, rng=0, check=False
    )
    with pytest.raises(ValueError, match="other bins or another basis"):
        res.attach(plain)
    with pytest.raises(ValueError, match="same bins"):
        res.predict_proba(plain)
    assert "basis=" in repr(oversampled)


def test_binning_bias_refuses_a_fit_on_finer_quadrature_cells():
    _, _, oversampled = _plain_and_oversampled()
    res = binference.infer(
        oversampled, method="gibbs", n_samples=50, n_warmup=10, rng=0, check=False
    )
    with pytest.raises(ValueError, match="binning_bias"):
        res.binning_bias(order=1)
