"""Tests for measures (DECISIONS #21) and masked grids (DECISIONS #19) on Bins."""

import pickle

import numpy as np
import pytest

import binference
from binference import Bins, InputWarning, Likelihoods
from binference.io import _bins_from_parts, _bins_to_parts
from binference.measure import Measure


def test_log_bins_carry_a_log_measure_with_geometric_centres():
    b = Bins.log(1.0, 100.0, 2)
    assert [m.name for m in b.measures] == ["log"]
    np.testing.assert_allclose(b.u_edges[0], np.log([1.0, 10.0, 100.0]))
    np.testing.assert_allclose(b.centers, [np.sqrt(10.0), np.sqrt(1000.0)])
    np.testing.assert_allclose(b.volumes, [9.0, 90.0])


def test_hand_built_geometric_edges_warn_unless_a_measure_is_given():
    edges = np.geomspace(1, 1000, 4)
    with pytest.warns(InputWarning, match="geometrically spaced"):
        Bins(edges)
    assert Bins(edges, measure="log").measures[0].name == "log"
    assert Bins(edges, measure="linear").measures[0].name == "linear"


def test_log_measure_rejects_non_positive_edges():
    with pytest.raises(ValueError, match="outside the domain"):
        Bins([0.0, 1.0, 2.0], measure="log")


def test_from_centers_uses_midpoints_in_the_measure_coordinate():
    c = np.geomspace(0.01, 1.0, 3)
    b = Bins.from_centers(c, measure="log")
    assert b.edges[0][0] > 0
    np.testing.assert_allclose(b.centers, c)
    np.testing.assert_allclose(np.diff(np.log(b.edges[0])), np.log(10.0))


def test_density_measure_reproduces_the_log_map():
    m = Measure.density(lambda z: 1.0 / z, 1.0, 10.0, n=20001)
    z = np.array([1.5, 3.0, 7.0])
    np.testing.assert_allclose(m.forward(z), np.log(z), rtol=1e-6)
    np.testing.assert_allclose(m.inverse(m.forward(z)), z, rtol=1e-9)


def test_transform_measure_sets_edges_and_centres_in_u():
    m = Measure.transform(np.sqrt, np.square, name="sqrt")
    b = Bins([0.0, 1.0, 4.0], measure=m)
    np.testing.assert_allclose(b.u_edges[0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(b.centers, [0.25, 2.25])


def test_measures_compare_by_name_or_table():
    assert Measure.log() == Measure.coerce("log") and Measure.log() != Measure.linear()
    a = Measure.tabulated([0.0, 1.0], [0.0, 2.0])
    assert a == Measure.tabulated([0.0, 1.0], [0.0, 2.0])
    assert pickle.loads(pickle.dumps(a)) == a
    with pytest.raises(ValueError, match="measure must be"):
        Measure.coerce("sqrt")


def test_negative_zero_edges_hash_like_positive_zero():
    a, b = Bins([-0.0, 1.0, 2.0]), Bins([0.0, 1.0, 2.0])
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1


def test_index_rejects_multi_column_values_for_1d_bins():
    b = Bins.linear(0, 1, 4)
    np.testing.assert_array_equal(b.index([[0.1], [0.9]]), [0, 3])
    with pytest.raises(ValueError, match="shape"):
        b.index([[0.1, 0.3], [0.5, 0.9]])


def test_sample_within_is_uniform_in_the_measure_coordinate():
    b = Bins.log(1.0, 100.0, 1)
    v = b.sample_within(np.zeros(20000, dtype=int), np.random.default_rng(0))
    assert v.min() >= 1.0 and v.max() <= 100.0
    # uniform in log z: half of the draws fall below 10
    assert abs(np.mean(v < 10.0) - 0.5) < 4 * 0.5 / np.sqrt(v.size)


# --------------------------------------------------------------------- masks
@pytest.fixture
def masked():
    mask = np.ones((3, 4), dtype=bool)
    mask[0, 0] = mask[2, 3] = False
    return Bins.grid(np.linspace(0, 3, 4), np.linspace(0, 4, 5), mask=mask)


def test_masked_grid_counts_and_orders_only_active_cells(masked):
    assert masked.K == 10 and masked.shape == (3, 4) and masked.is_masked
    np.testing.assert_array_equal(masked.cells, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    np.testing.assert_allclose(masked.centers[0], [0.5, 1.5])
    assert masked.volumes.shape == (10,)


def test_masked_cells_are_outside_the_support(masked):
    np.testing.assert_array_equal(masked.index([[0.5, 0.5], [0.5, 1.5], [2.5, 3.5]]), [-1, 0, -1])


def test_unflatten_fills_masked_cells_and_flatten_inverts_it(masked):
    a = np.arange(10.0)
    g = masked.unflatten(a)
    assert np.isnan(g[0, 0]) and np.isnan(g[2, 3]) and g[0, 1] == 0.0
    np.testing.assert_array_equal(masked.flatten(g), a)


def test_neighbours_skip_masked_cells(masked):
    pairs = masked.neighbors()
    assert pairs.min() >= 0 and pairs.max() < masked.K
    assert len(pairs) == len(Bins.grid(*masked.edges).neighbors()) - 4


def test_coarsen_keeps_wholly_active_blocks_and_drops_masked_ones():
    mask = np.ones(4, dtype=bool)
    mask[:2] = False
    b = Bins(np.linspace(0, 4, 5), mask=mask)
    coarse, mapping = b.coarsen(2)
    assert coarse.K == 1
    np.testing.assert_array_equal(mapping, [0, 0])


def test_masked_bins_compare_pickle_and_save_with_their_measure(masked):
    b = Bins.grid(masked.edges[0] + 1.0, masked.edges[1] + 1.0, measure="log", mask=masked.mask)
    assert b != Bins.grid(*b.edges, measure="log")
    assert pickle.loads(pickle.dumps(b)) == b
    meta, arrays = _bins_to_parts(b)
    assert _bins_from_parts(meta, arrays) == b
    t = Bins([1.0, 2.0, 5.0], measure=Measure.density(lambda z: 1.0 / z, 1.0, 5.0))
    meta, arrays = _bins_to_parts(t)
    assert _bins_from_parts(meta, arrays) == t


def test_invalid_masks_are_rejected():
    with pytest.raises(ValueError, match="must have the grid"):
        Bins.linear(0, 1, 3, mask=[True, False])
    with pytest.raises(ValueError, match="must have the grid"):  # a transposed 2-D mask
        Bins.grid(np.arange(3.0), np.arange(4.0), mask=np.ones((3, 2), dtype=bool))
    with pytest.raises(ValueError, match="no bins"):
        Bins.linear(0, 1, 2, mask=[False, False])


def test_gaussian_likelihoods_on_a_masked_grid_are_the_active_columns(masked):
    rng = np.random.default_rng(1)
    mu, sig = rng.uniform(0, 3, (20, 2)), np.full((20, 2), 0.7)
    full = Likelihoods.from_gaussian(mu, sig, Bins.grid(*masked.edges)).toarray()[:, masked.cells]
    part = Likelihoods.from_gaussian(mu, sig, masked).toarray()
    np.testing.assert_allclose(part, full / full.max(axis=1, keepdims=True), rtol=1e-12)


def test_mocks_on_masked_grids_put_no_object_in_a_masked_cell(masked):
    mock = binference.simulate.mock_2d(300, masked, rng=2)
    assert mock.likelihoods.K == masked.K
    assert np.all(masked.index(mock.truth.values) >= 0)


def test_smoothness_priors_refuse_masked_grids_for_now(masked):
    with pytest.raises(NotImplementedError, match="masked"):
        binference.priors.LogisticGP().bind(masked)


def test_a_user_measure_named_log_is_not_the_log_measure():
    fake = Measure.transform(np.log, np.exp, name="log")
    assert fake != Measure.log() and not fake.is_named


def test_bins_cell_index_matches_grid_index_of_every_bin():
    mask = np.ones((3, 4), dtype=bool)
    mask[0, 0] = mask[2, 3] = False
    b = Bins.grid(np.arange(4.0), np.arange(5.0), mask=mask)
    for got, want in zip(b.cell_index, np.unravel_index(b.cells, b.shape), strict=True):
        np.testing.assert_array_equal(got, want)


def test_coarsen_rejects_a_partly_masked_block():
    b = Bins(np.linspace(0, 4, 5), mask=[True, False, True, True])
    with pytest.raises(ValueError, match="wholly active"):
        b.coarsen(2)


def test_from_centers_refuses_outer_bins_beyond_a_tabulated_measure():
    m = Measure.density(lambda z: 1.0 + 0.0 * z, 0.0, 10.0)
    with pytest.raises(ValueError, match="beyond the range"):
        Bins.from_centers([0.0, 5.0, 10.0], measure=m)


def test_cdf_likelihoods_on_masked_1d_bins_keep_only_active_bins():
    from scipy import stats

    b = Bins.linear(0, 4, 4, mask=[True, False, True, True])
    d = [stats.norm(1.5, 0.3)]
    with pytest.warns(InputWarning, match="outside the bins"):  # the masked bin holds most mass
        L = Likelihoods.from_density(d, b, kind="likelihood").toarray()
    full = Likelihoods.from_density(d, Bins.linear(0, 4, 4), kind="likelihood").toarray()
    ratio = L / full[:, b.cells]  # likelihoods are defined up to a per-object scale
    np.testing.assert_allclose(ratio, ratio[0, 0], rtol=1e-6)


@pytest.mark.parametrize("interim", [None, "uniform"])
def test_mock_posterior_samples_avoid_masked_bins(interim):
    mask = np.ones(30, dtype=bool)
    mask[10:15] = False
    b = Bins.linear(0, 3, 30, mask=mask)
    mock = binference.simulate.mock_1d(300, b, interim=interim, n_samples=50, rng=3)
    assert np.all(b.contains(mock.samples.ravel()))


def test_masked_1d_results_plot_with_gaps():
    import matplotlib

    matplotlib.use("Agg")
    mask = np.ones(30, dtype=bool)
    mask[10:15] = False
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 30, mask=mask), rng=3)
    res = binference.infer(mock.likelihoods, method="gibbs", rng=1, n_samples=20, check=False)
    ax = res.plot(truth=mock.truth.rho, stack=mock.likelihoods.stack())
    assert ax.get_xlim() == (0.0, 3.0)
