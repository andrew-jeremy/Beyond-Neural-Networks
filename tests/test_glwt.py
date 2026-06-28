import numpy as np
from glwt_logic.graph_utils import karate_graph, normalized_laplacian, structural_node_signals
from glwt_logic.glwt import GLWTFilterBank, soft_shrink


def test_soft_shrink():
    x = np.array([-2.0, -0.2, 0.0, 0.3, 3.0])
    y = soft_shrink(x, 0.5)
    assert np.allclose(y, np.array([-1.5, 0.0, 0.0, 0.0, 2.5]))


def test_glwt_shapes():
    _, A, _ = karate_graph()
    L = normalized_laplacian(A)
    X = structural_node_signals(A)
    bank = GLWTFilterBank(L)
    coeffs = bank.transform(X)
    assert len(coeffs) == len(bank.scales)
    assert coeffs[0].shape == X.shape
    feats = bank.features(X)
    assert feats.shape[0] == A.shape[0]
