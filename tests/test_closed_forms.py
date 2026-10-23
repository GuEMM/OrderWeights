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
                  "coI": lambda m: (-1) ** N * (m == N), "TSE": lambda m: F((N + 1) * (m - 1), 2 * (m + 1))}
        for name, f in closed.items():
            lam = M.order_weights(M.c_measure(name, N))
            assert lam[0] == 0 and all(lam[m - 1] == f(m) for m in range(2, N + 1))
        for k in range(1, N):
            assert M.coefficients(M.order_weights(M.c_u(N, k))) == M.c_u(N, k)


def test_exact_bias_two_forms():
    for q in (2, 3):
        for name in ("Omega", "Sigma", "coI", "OmegaStar"):
            c = M.coeffs(name, 5)
            for T in (q ** 3, q ** 5):
                assert np.isclose(B.exact_bias(c, 5, q, T), B.exact_bias_lambda(c, 5, q, T))


def test_gaussian_beta_form():
    c = M.coeffs("Omega", 8)
    lam = [float(x) for x in M.order_weights(c)]
    alt = sum(lam[m - 1] * comb(8, m) * G.beta_G(m, 39) for m in range(2, 9))
    assert np.isclose(G.exact_bias(c, 8, 40), alt)


def test_bias_bound_attained_by_omega_star():
    for q in (2, 3):
        for N in range(3, 12):
            lam = [float(x) for x in M.order_weights(M.c_omega_star(N))]
            b = sum(lam[m - 1] * comb(N, m) * (q - 1) ** m for m in range(2, N + 1)) / abs(lam[-1])
            assert np.isclose(b, -(q - 1) ** 2 * (q ** (N - 2) - 1))


def test_cheapest_measure_is_omega():
    """Proposition on the cheapest measure: lambda_2 = 0 and lambda_N != 0 need at least 2N+1 entropies,
    reached only by the O-information divided by N-2 (brute force over the subset sizes used)."""
    import itertools as it
    for N in range(4, 9):
        best = None
        for r in (2, 3):
            for sizes in it.combinations(range(1, N + 1), r):
                if N not in sizes:
                    continue
                A = np.array([[comb(N - m, k - m) if k >= m else 0 for k in sizes] for m in (1, 2)], float)
                if np.linalg.matrix_rank(A) == len(sizes):
                    continue                       # only the zero solution
                v = np.linalg.svd(A)[2][-1]
                if abs(v[sizes.index(N)]) < 1e-12:
                    continue                       # does not respond to GXOR
                cost = sum(comb(N, k) for k in sizes)
                if best is None or cost < best[0]:
                    best = (cost, sizes, v / v[sizes.index(N)])
        assert best[0] == 2 * N + 1 == M.cost(M.c_measure("Omega", N))
        assert best[1] == (1, N - 1, N)
        assert np.allclose(best[2], [1 / (N - 2), -1 / (N - 2), 1])


def test_omega_ratio():
    """Eq. (omega-ratio): leading bias of Omega/(N-2) over the bound, < q/(q-1), equal to one for N = 3, 4."""
    for q in (2, 3, 4, 5):
        for N in range(3, 21):
            lam = [float(x) for x in M.order_weights(M.c_measure("Omega", N))]
            b_om = sum(lam[m - 1] * comb(N, m) * (q - 1) ** m for m in range(2, N + 1)) / (N - 2)
            ratio = abs(b_om) / ((q - 1) ** 2 * (q ** (N - 2) - 1))
            closed = (N * (q - 1) * q ** (N - 1) - 2 * q ** N + N * (q - 1) + 2) / ((N - 2) * (q - 1) ** 2 * (q ** (N - 2) - 1))
            assert np.isclose(ratio, closed) and ratio < q / (q - 1) + 1e-12
            if N in (3, 4):
                assert np.isclose(ratio, 1)


def test_dyadic_triadic_same_entropies():
    assert np.allclose(dyadic.entropies(dyadic.endpoint("dyadic")), dyadic.entropies(dyadic.endpoint("triadic")))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)


def test_field_mixture():
    """Two independent regimes (U and a common field): Omega < 0 near U with the leading term of
    Proposition 'mixture', and the pairwise MaxEnt path (Ising with field and coupling) stays >= 0."""
    from orderweights import paths
    mix, me = paths.field_paths(4, 0.05, n=80)
    assert mix["y"][0] < 0 and mix["y"].max() > 0
    assert me["y"].min() > -1e-10


def test_field_mixture_leading_terms():
    """Independent regimes: leading terms near U, near the biased product Q (Efron-Stein, SI
    Eq. si-field-Q) and along the MaxEnt path (SI Eq. si-field-maxent), for the O-information."""
    import itertools as it
    from math import comb
    from orderweights import paths
    N, p = 4, 0.05
    mu = 1 - 2 * p; nu2 = mu ** 2 / (1 - mu ** 2)
    s = paths._setup(N, 2)
    X = np.array(list(it.product((0, 1), repeat=N))); k = X.sum(1)
    Q = p ** k * (1 - p) ** (N - k); U = s["U"]
    lead_U = 0.5 * sum((2 - m) * comb(N, m) * mu ** (2 * m) for m in range(3, N + 1))
    lead_Q = 0.5 * sum((2 - m) * comb(N, m) * nu2 ** m for m in range(3, N + 1))
    e = 1e-4; assert abs(s["omega"]((1 - e) * U + e * Q) / (lead_U * e ** 2) - 1) < 2e-3
    d = 1e-7; assert abs(s["omega"]((1 - d) * Q + d * U) / (lead_Q * d ** 2) - 1) < 2e-3
    sp = 1 - 2 * X; M = sp.sum(1).astype(float); E = (M ** 2 - N) / 2
    e = 1e-3; tgt = np.array([N * e * mu, comb(N, 2) * e * mu ** 2])
    _, Pme = paths.fit_maxent(np.column_stack([M, E]), tgt)
    assert abs(s["omega"](Pme) / (comb(N, 3) * mu ** 6 * e ** 3) - 1) < 1e-2


def test_fit_maxent_matches_moments():
    """The Newton fit reproduces the target moments and the pairwise marginals of the mixture."""
    import itertools as it
    from math import comb
    from orderweights import paths
    N, p, e = 5, 0.05, 0.3
    mu = 1 - 2 * p
    X = np.array(list(it.product((0, 1), repeat=N))); k = X.sum(1)
    M = (N - 2 * k).astype(float); E = (M ** 2 - N) / 2
    _, P = paths.fit_maxent(np.column_stack([M, E]), [N * e * mu, comb(N, 2) * e * mu ** 2])
    Pmix = (1 - e) / 2 ** N + e * p ** k * (1 - p) ** (N - k)
    for i, j in it.combinations(range(N), 2):
        a = np.array([[P[(X[:, i] == u) & (X[:, j] == v)].sum() for v in (0, 1)] for u in (0, 1)])
        b = np.array([[Pmix[(X[:, i] == u) & (X[:, j] == v)].sum() for v in (0, 1)] for u in (0, 1)])
        assert np.abs(a - b).max() < 1e-12
