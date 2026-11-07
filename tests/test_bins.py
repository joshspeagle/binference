"""Tests for binference.Bins (spec 4.1; R-BINS-1)."""

import copy
import pickle

import numpy as np
import pytest

from binference import Bins


def test_1d_basic():
    b = Bins([0.0, 0.5, 1.0, 2.0])
    assert b.K == 3
    assert b.ndim == 1
    assert b.shape == (3,)
    assert b.ordered and not b.is_categorical
    np.testing.assert_allclose(b.volumes, [0.5, 0.5, 1.0])
    np.testing.assert_allclose(b.centers, [0.25, 0.75, 1.5])
    assert len(b) == 3
    assert "Bins(K=3" in repr(b)


@pytest.mark.parametrize("bad", [[0.0], [0.0, 0.0], [1.0, 0.0], [0.0, np.nan, 1.0]])
def test_invalid_edges(bad):
    with pytest.raises(ValueError):
        Bins(bad)


def test_linear_log_from_centers():
    b = Bins.linear(0, 3, 30)
    assert b.K == 30
    np.testing.assert_allclose(b.volumes, 0.1)
    lb = Bins.log(1, 100, 4)
    np.testing.assert_allclose(lb.edges[0], np.geomspace(1, 100, 5))
    with pytest.raises(ValueError):
        Bins.log(0, 1, 3)
    c = Bins.from_centers([0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(c.centers, [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(c.edges[0], [-0.05, 0.05, 0.15, 0.25, 0.35])


def test_grid():
    g = Bins.grid([0, 1, 2], [0, 1, 2, 3])
    assert g.K == 6 and g.shape == (2, 3) and g.ndim == 2
    np.testing.assert_allclose(g.volumes, np.ones(6))
    assert g.centers.shape == (6, 2)
    # C order: last axis fastest
    np.testing.assert_allclose(g.centers[:3, 0], 0.5)
    np.testing.assert_allclose(g.centers[:3, 1], [0.5, 1.5, 2.5])
    a = np.arange(6.0)
    np.testing.assert_allclose(g.flatten(g.unflatten(a)), a)
    assert g.unflatten(a).shape == (2, 3)
    g2 = Bins.grid([[0, 1, 2], [0, 2, 4]])
    np.testing.assert_allclose(g2.volumes, 2.0)
    g3 = Bins.grid([0, 1], [0, 1, 2], [0, 0.5, 1.0])
    assert g3.K == 4 and g3.shape == (1, 2, 2)
    np.testing.assert_allclose(g3.volumes, 0.5)


def test_categories():
    c = Bins.categories(["a", "b", "c"])
    assert c.K == 3 and c.is_categorical and not c.ordered
    np.testing.assert_allclose(c.volumes, 1.0)
    assert c.labels == ("a", "b", "c")
    with pytest.raises(AttributeError):
        _ = c.edges
    c4 = Bins.categories(4)
    assert c4.labels == (0, 1, 2, 3)
    with pytest.raises(ValueError):
        Bins.categories(["a", "a"])
    with pytest.raises(ValueError):
        Bins.categories(1)
    with pytest.raises(ValueError):
        c.neighbors()


def test_index_and_contains():
    b = Bins([0.0, 1.0, 2.0])
    idx = b.index([-0.1, 0.0, 0.5, 1.0, 1.999, 2.0, 2.1, np.nan])
    np.testing.assert_array_equal(idx, [-1, 0, 0, 1, 1, 1, -1, -1])
    assert b.contains([0.5]).all()
    g = Bins.grid([0, 1, 2], [0, 1])
    np.testing.assert_array_equal(g.index([[0.5, 0.5], [1.5, 0.2], [2.5, 0.5]]), [0, 1, -1])
    c = Bins.categories(["x", "y"])
    np.testing.assert_array_equal(c.index(["y", "x", "z"]), [1, 0, -1])


def test_neighbors():
    b = Bins.linear(0, 1, 4)
    np.testing.assert_array_equal(b.neighbors(), [[0, 1], [1, 2], [2, 3]])
    g = Bins.grid([0, 1, 2], [0, 1, 2])
    pairs = {tuple(p) for p in g.neighbors().tolist()}
    assert pairs == {(0, 2), (1, 3), (0, 1), (2, 3)}


def test_coarsen():
    b = Bins.linear(0, 1, 6)
    cb, m = b.coarsen(2)
    assert cb.K == 3
    np.testing.assert_array_equal(m, [0, 0, 1, 1, 2, 2])
    np.testing.assert_allclose(np.bincount(m, weights=b.volumes), cb.volumes)
    with pytest.raises(ValueError):
        b.coarsen(4)


def test_immutable_equality_pickle():
    b = Bins.linear(0, 1, 3)
    with pytest.raises(AttributeError):
        b._shape = (4,)
    with pytest.raises(ValueError):
        b.volumes[0] = 3.0
    assert b == Bins([0, 1 / 3, 2 / 3, 1.0])
    assert b != Bins.linear(0, 1, 4)
    assert hash(b) == hash(Bins.linear(0, 1, 3))
    for obj in (b, Bins.grid([0, 1, 2], [0, 1]), Bins.categories(["a", "b"])):
        assert pickle.loads(pickle.dumps(obj)) == obj


def test_pickled_and_copied_bins_keep_their_edges_read_only():
    b = Bins.grid([1.0, 2.0, 4.0], [0.0, 1.0, 3.0], measure=["log", "linear"])
    for new in (pickle.loads(pickle.dumps(b)), copy.copy(b)):
        assert new == b and not any(e.flags.writeable for e in new.edges)


def test_bin_counts_and_factors_must_be_positive_integers():
    for build, name in (
        (lambda: Bins.linear(0, 1, 2.5), "n"),
        (lambda: Bins.log(1, 10, 3.7), "n"),
        (lambda: Bins.linear(0, 1, 6).coarsen(1.5), "factor"),
    ):
        with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
            build()
    assert Bins.linear(0, 1, 6.0) == Bins.linear(0, 1, 6)  # as refine, an integral float
    for labels in (2.0, "star"):  # a string is one label, not four
        with pytest.raises(TypeError, match="labels must be"):
            Bins.categories(labels)
