"""Dyadic and triadic distributions of James & Crutchfield (Entropy 19, 531, 2017):
three variables X, Y, Z with q = 4 states, each written as two bits, x = 2 x0 + x1
(SI Sec. 'A limitation of entropy-based measures').

Arrays over the six bits have axes ordered X0, X1, Y0, Y1, Z0, Z1.
"""
import itertools as it

import numpy as np
from scipy.optimize import brentq

__all__ = ["GENERATORS", "endpoint", "maxent", "entropies", "kappa", "z4_components",
           "indistinguishable_pair", "LN2"]

GROUPS = [(0, 1), (2, 3), (4, 5)]
NB = 6
LN2 = np.log(2)

# Rules as products of +-1 characters of the bits: chi(g) = prod_{i in g} (-1)^{bit_i}
GENERATORS = {"dyadic": [(0, 3), (2, 5), (4, 1)],                # X0=Y1, Y0=Z1, Z0=X1
              "triadic": [(0, 2, 4), (1, 3), (3, 5), (1, 5)]}    # X0+Y0+Z0=0 mod 2, X1=Y1=Z1


def _chi(bits):
    c = np.ones((2,) * NB)
    for i in bits:
        sh = [1] * NB; sh[i] = 2
        c = c * np.array([1.0, -1.0]).reshape(sh)
    return c


def endpoint(kind):
    """Uniform distribution over the eight states that satisfy the rules."""
    P = np.ones((2,) * NB)
    for g in GENERATORS[kind]:
        P = P * (_chi(g) > 0)
    return P / P.sum()


def maxent(kind, J, J3=None):
    """P ~ exp(J sum of the generators). For 'triadic', J3 is the coupling of the
    three-variable term and J that of the pairwise terms (default J3 = J)."""
    E = np.zeros((2,) * NB)
    for g in GENERATORS[kind]:
        E = E + (J3 if (len(g) == 3 and J3 is not None) else J) * _chi(g)
    P = np.exp(E - E.max())
    return P / P.sum()


def _H(P, vars_):
    keep = [i for v in vars_ for i in GROUPS[v]]
    m = P.sum(axis=tuple(i for i in range(NB) if i not in keep)).ravel()
    m = m[m > 1e-300]
    return -(m * np.log(m)).sum()


def entropies(P):
    """The seven subset entropies (nats), ordered X, Y, Z, XY, XZ, YZ, XYZ."""
    subs = [a for k in (1, 2, 3) for a in it.combinations(range(3), k)]
    return np.array([_H(P, a) for a in subs])


def kappa(P):
    """(kappa of the pair XY, kappa_XYZ) in nats."""
    h = entropies(P)
    k_pair = h[0] + h[1] - h[3]
    tc = h[0] + h[1] + h[2] - h[6]
    pairs = (h[0] + h[1] - h[3]) + (h[0] + h[2] - h[4]) + (h[1] + h[2] - h[5])
    return k_pair, tc - pairs


def z4_components(P6, tol=1e-9):
    """Non-zero Fourier coefficients rho_hat(r), r in Z_4^3, of a distribution given on the bits.
    Returns a list of (weight, r, |rho_hat|^2)."""
    P = P6.reshape(4, 4, 4)            # x = 2 x0 + x1 for each variable (C-order reshape)
    X = np.array(list(it.product(range(4), repeat=3)))
    w = 1j * np.pi / 2
    out = []
    for r in it.product(range(4), repeat=3):
        if not any(r):
            continue
        rho = np.sum(P.ravel() * np.exp(-w * (X @ np.array(r))))
        if abs(rho) > tol:
            out.append((sum(1 for x in r if x), r, abs(rho) ** 2))
    return sorted(out)


def indistinguishable_pair(J3):
    """Full-support pair with identical subset entropies (SI, Table S-dt-family).
    Returns (J2, J): the pairwise coupling of the triadic member at which kappa_XYZ = 0,
    and the coupling of the dyadic member with the same pair entropies."""
    J2 = brentq(lambda j2: kappa(maxent("triadic", j2, J3))[1], 1e-3, 10, xtol=1e-12)
    h_pair = entropies(maxent("triadic", J2, J3))[3]
    J = brentq(lambda j: entropies(maxent("dyadic", j))[3] - h_pair, 1e-4, 20, xtol=1e-12)
    return J2, J
