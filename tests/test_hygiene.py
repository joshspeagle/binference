"""Tests for Likelihoods validation, storage and compress (issue #26; DECISIONS #1)."""

import numpy as np
import pytest
import scipy.sparse as sp

from binference import Bins, InputWarning, Likelihoods, PopulationModel, Selection


@pytest.fixture
def lk():
    return Likelihoods(np.array([[1.0, 2.0, 0.5], [0.2, 1.0, 3.0]]), Bins.linear(0, 1, 3))


def test_weights_are_integer_multiplicities_unless_asked_otherwise(lk):
    with pytest.raises(ValueError, match="integer multiplicities"):
        Likelihoods(lk.values, lk.bins, weights=[1.5, 1.0])
    with pytest.warns(InputWarning, match="pseudo-likelihood"):
        frac = lk.with_weights([1.5, 1.0], fractional=True)
    assert frac.W == 2.5
    assert lk.with_weights([2, 3]).W == 5


def test_rtol_outside_zero_to_one_is_refused(lk):
    for rtol in (-0.1, 1.5):
        with pytest.raises(ValueError, match="rtol"):
            lk.to_sparse(rtol)
    assert lk.to_sparse(1.0).nnz == 2  # the row maxima survive


def test_gaussian_means_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError, match="mean must have shape"):
        Likelihoods.from_gaussian(np.full((3, 2), 0.5), 0.1, Bins.linear(0, 1, 4))


def test_an_ambiguous_sigma_is_refused():
    grid = Bins.grid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    with pytest.raises(ValueError, match="ambiguous"):
        Likelihoods.from_gaussian(np.full((2, 2), 0.5), [0.1, 0.3], grid)
    lk = Likelihoods.from_gaussian(np.full((2, 2), 0.5), [[0.1, 0.3]], grid)
    np.testing.assert_allclose(lk.meta["gaussian_sigma"], [[0.1, 0.3], [0.1, 0.3]])


def test_no_objects_is_an_error(lk):
    with pytest.raises(ValueError, match="at least one object"):
        lk.subset([])
    with pytest.raises(ValueError, match="no observed events"):
        Likelihoods.from_response([0, 0], 0.5 * np.eye(2))


def test_a_bad_representation_is_refused_before_any_work():
    with pytest.raises(ValueError, match="representation"):
        Likelihoods(np.full((2, 3), np.nan), representation="probability")


def test_stored_arrays_are_read_only_and_not_shared(lk):
    for arr in (lk.values, lk.log_scale, lk.weights):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0.0
    sparse = lk.to_sparse()
    with pytest.raises(ValueError, match="read-only"):
        sparse.values.data[0] = 0.0
    model = PopulationModel(lk)
    assert not model.likelihoods.values.flags.writeable


def test_selection_is_immutable():
    s = Selection([0.5, 1.0], "true")
    with pytest.raises(AttributeError, match="immutable"):
        s.mode = "observed"


def test_compress_keeps_sparse_storage_and_always_copies():
    bins = Bins.linear(0, 1, 50)
    rows = sp.csr_array(np.eye(50)[[0, 3, 0, 3, 7]])
    lk = Likelihoods(rows, bins)
    c = lk.compress()
    assert c.is_sparse and c.N == 3 and sorted(c.weights) == [1, 2, 2]
    np.testing.assert_allclose(
        c.log_evidence_terms(np.full(50, 0.02)) @ c.weights,
        lk.log_evidence_terms(np.full(50, 0.02)).sum(),
    )
    unique = Likelihoods(np.eye(3), Bins.linear(0, 1, 3))
    assert unique.compress() is not unique


def test_broad_gaussians_give_flat_rows():
    with pytest.warns(InputWarning, match="outside the bins"):
        lk = Likelihoods.from_gaussian([0.5], [1e8], Bins.linear(0, 1, 4))
    np.testing.assert_allclose(lk.toarray(), 1.0, rtol=1e-14)


def test_grid_explains_a_two_dimensional_array():
    with pytest.raises(ValueError, match="must be 1-D"):
        Bins.grid(np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))
    with pytest.raises(ValueError, match="one array of edges per axis"):
        Bins.grid([])


def test_fractional_weights_reach_every_constructor():
    with pytest.warns(InputWarning, match="pseudo-likelihood"):
        lk = Likelihoods.from_gaussian(
            [0.3, 0.6], 0.1, Bins.linear(0, 1, 4), weights=[0.5, 1.5], fractional_weights=True
        )
    assert lk.W == 2.0


def test_scipy_gaussians_keep_precision_near_the_mean_too():
    from scipy import stats

    with pytest.warns(InputWarning, match="outside the bins"):
        lk = Likelihoods.from_density(
            [stats.norm(0.5, 1e8)], Bins.linear(0, 1, 4), kind="likelihood"
        )
    np.testing.assert_allclose(lk.toarray(), 1.0, rtol=1e-6)
