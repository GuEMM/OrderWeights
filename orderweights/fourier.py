"""Discrete Fourier analysis of Boolean distributions (Methods 'Discrete Fourier analysis',
Fig. 'fourier-rules'; SI Table S1).

rho_hat(r) = E_P[chi_r(x)] = sum_x P(x) (-1)^{r.x}: the coefficients of the relative deviation
rho = P/U - 1 from uniformity.
"""
import itertools as it

import numpy as np

__all__ = ["from_function", "from_constraints", "supports", "rho_hat", "character_table"]


def from_function(f, N=4):
    """x_1..x_{N-1} uniform and independent, x_N = f(x_1, ..., x_{N-1})."""
    P = np.zeros((2,) * N)
    for x in it.product((0, 1), repeat=N - 1):
        P[x + (int(f(*x)),)] += 1
    return P / P.sum()


def from_constraints(ok, N=4):
    """Uniform over the states x for which ok(x) is True."""
    P = np.zeros((2,) * N)
    for x in it.product((0, 1), repeat=N):
        P[x] = float(ok(x))
    return P / P.sum()


def supports(N):
    """All non-empty supports a(r), ordered by weight and lexicographically within a weight."""
    return [a for k in range(1, N + 1) for a in it.combinations(range(N), k)]


def rho_hat(P, a):
    """Coefficient of the Boolean direction with support a."""
    N = P.ndim
    sign = np.ones((2,) * N)
    for i in a:
        sh = [1] * N; sh[i] = 2
        sign = sign * np.array([1.0, -1.0]).reshape(sh)
    return (P * sign).sum()


def character_table(N=3):
    """Signs of chi_r(x) = (-1)^{r.x} for all r, x in {0,1}^N (SI Table S1)."""
    states = list(it.product((0, 1), repeat=N))
    return states, {r: [(-1) ** (np.dot(r, x) % 2) for x in states] for r in states}
