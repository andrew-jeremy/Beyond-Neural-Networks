# Algorithm Notes

This repository implements the algorithm proposed in *Beyond Neural Networks: Symbolic Reasoning over Wavelet Logic Graph Signals*.

The implementation keeps the architecture neural-network-free:

1. Build a graph Laplacian `L` from the input graph.
2. Define Graph Laplacian Wavelet Transform filters `g_k(s_k L)`.
3. Compute multiscale coefficients `c_k = g_k(s_k L) f`.
4. Apply soft shrinkage and modulation `phi(c_k) = gamma_k sign(c_k) max(|c_k|-lambda_k,0) cos(theta_k)`.
5. Reconstruct each scale component `f^(k) = g_k(s_k L) phi(c_k)`.
6. Combine scales with convex weights.
7. Threshold spectral activations and induce symbolic rules using a shallow decision tree.

The benchmark script uses exact eigendecomposition for the small Karate Club graph and includes a Chebyshev approximation utility for larger graphs.
