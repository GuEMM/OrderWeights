"""Quick checks of closed forms used in the paper (run: python -m pytest tests)."""
import sys
from fractions import Fraction as F
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orderweights import measures as M, bias as B, gaussian as G, dyadic  # noqa: E402


def test_order_weights_closed_forms():
    for N in range(3, 10):
        closed = {"TC": lambda m: 1, "DTC": lambda m: m - 1, "Sigma": lambda m: m, "Omega": lambda m: 2 - m,
                  "II": lambda m: (-1) ** N * (m == N), "TSE": lambda m: F((N + 1) * (m - 1), 2 * (m + 1))}
        for name, f in closed.items():
            lam = M.order_weights(M.c_measure(name, N))
            assert lam[0] == 0 and all(lam[m - 1] == f(m) for m in range(2, N + 1))
        for k in range(1, N):
            assert M.coefficients(M.order_weights(M.c_u(N, k))) == M.c_u(N, k)


def test_exact_bias_two_forms():
    for q in (2, 3):
        for name in ("Omega", "Sigma", "II", "phiSR"):
            c = M.coeffs(name, 5)
            for T in (q ** 3, q ** 5):
                assert np.isclose(B.exact_bias(c, 5, q, T), B.exact_bias_lambda(c, 5, q, T))


def test_gaussian_beta_form():
    c = M.coeffs("Omega", 8)
    lam = [float(x) for x in M.order_weights(c)]
    alt = sum(lam[m - 1] * comb(8, m) * G.beta_G(m, 39) for m in range(2, 9))
    assert np.isclose(G.exact_bias(c, 8, 40), alt)


def test_bias_bound_attained_by_phiSR():
    for q in (2, 3):
        for N in range(3, 12):
            lam = [float(x) for x in M.order_weights(M.c_phiSR(N))]
            b = sum(lam[m - 1] * comb(N, m) * (q - 1) ** m for m in range(2, N + 1)) / abs(lam[-1])
            assert np.isclose(b, -(q - 1) ** 2 * (q ** (N - 2) - 1))


def test_dyadic_triadic_same_entropies():
    assert np.allclose(dyadic.entropies(dyadic.endpoint("dyadic")), dyadic.entropies(dyadic.endpoint("triadic")))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
