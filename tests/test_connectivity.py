import numpy as np

from scdfc.connectivity import (
    edges_to_matrix,
    expected_windows,
    matrix_to_edges,
    nearest_correlation,
    nonoverlap_horizon,
    sliding_window_fc,
)


def test_edge_matrix_roundtrip():
    rng = np.random.default_rng(1)
    edges = rng.normal(size=(3, 6)).astype(np.float32)
    matrix = edges_to_matrix(edges, 4)
    assert matrix.shape == (3, 4, 4)
    np.testing.assert_allclose(matrix, matrix.transpose(0, 2, 1))
    np.testing.assert_allclose(matrix_to_edges(matrix), edges)
    np.testing.assert_allclose(np.diagonal(matrix, axis1=-2, axis2=-1), 1)


def test_sliding_fc_matches_numpy():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(30, 4))
    z, starts = sliding_window_fc(data, 10, 5)
    assert len(z) == expected_windows(30, 10, 5) == 5
    expected = np.arctanh(np.corrcoef(data[:10], rowvar=False)[np.triu_indices(4, 1)])
    np.testing.assert_allclose(z[0], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(starts, [0, 5, 10, 15, 20])


def test_nearest_correlation_is_valid():
    invalid = np.array([[1.0, 1.2, -0.9], [1.2, 1.0, 0.8], [-0.9, 0.8, 1.0]])
    projected = nearest_correlation(invalid)
    np.testing.assert_allclose(projected, projected.T, atol=1e-7)
    np.testing.assert_allclose(np.diag(projected), 1)
    assert np.linalg.eigvalsh(projected).min() >= -1e-6


def test_nonoverlap_horizons():
    assert [nonoverlap_horizon(window, 5) for window in (83, 42, 125)] == [17, 9, 25]

