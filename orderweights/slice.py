"""The pairwise-triplet slice of the simplex for N = 3 variables on Z_q
(main text Fig. 'slice', SI Sec. 'Visualization of local geometry').

The slice is P = U (1 + a psi_2 + b psi_3), with psi_2 the symmetric combination of pairwise
directions and psi_3 the triplet directions, normalized so that chi^2_2 = a^2, chi^2_3 = b^2.
"""
import itertools
from math import comb

import numpy as np

__all__ = ["Slice", "GEOMETRY", "targets"]

N = 3
SQ2, SQ3, SQ6, SQ32 = np.sqrt(2), np.sqrt(3), np.sqrt(6), np.sqrt(1.5)

# Domain of the slice (P >= 0) and radius of convergence ||rho||_inf = 1
GEOMETRY = {
    2: dict(domain=np.array([(-1 / SQ3, 0.0), (0.0, -1.0), (SQ3, 0.0), (0.0, 1.0)]),
            labels=["non-COPY", "~XOR", "COPY", "XOR"],
            radius=np.array([(-1 / SQ3, 0.0), (0.0, -1.0), (1 / SQ3, 0.0), (0.0, 1.0)]),
            zoom=(-0.3, 0.7, -0.4, 0.4)),
    3: dict(domain=np.array([(SQ6, SQ2), (-SQ32, SQ2), (0.0, -1 / SQ2)]),
            labels=["COPY", "non-COPY", "~GXOR"],
            radius=np.array([(0.0, -1 / SQ2), (4 / (3 * SQ6), -1 / (3 * SQ2)),
                             (0.0, 1 / SQ2), (-4 / (3 * SQ6), 1 / (3 * SQ2))]),
            zoom=(-0.25, 0.45, -0.15, 0.6)),
}


class Slice:
    """Exact O-information and its variations on the slice, for N = 3 and alphabet size q."""

    def __init__(self, q, c=None):
        self.q = q
        self.X = np.array(list(itertools.product(range(q), repeat=N)))
        p2 = sum(q * (self.X[:, i] == self.X[:, j]) - 1.0 for i, j in [(0, 1), (1, 2), (0, 2)])
        p3 = q * ((self.X.sum(1) % q) == 0) - 1.0
        self.p2 = p2 / np.sqrt(np.mean(p2 ** 2))
        self.p3 = p3 / np.sqrt(np.mean(p3 ** 2))
        # O-information for N = 3: c_1 = 1, c_2 = -1, c_3 = 1
        self.c = c if c is not None else {1: 1.0, 2: -1.0, 3: 1.0}
        self.lam = {m: -sum(self.c.get(k, 0.0) * comb(N - m, k - m) for k in range(m, N + 1))
                    for m in range(1, N + 1)}
        self.subsets = [a for k in range(1, N + 1) for a in itertools.combinations(range(N), k)]

    def P(self, a, b):
        return (1 + a * self.p2 + b * self.p3) / self.q ** N

    def coords(self, P):
        rho = P * self.q ** N - 1
        return np.mean(rho * self.p2), np.mean(rho * self.p3)

    def _marg(self, P, a):
        ax = tuple(i for i in range(N) if i not in a)
        return P.reshape((self.q,) * N).sum(axis=ax).ravel() if ax else P

    def omega(self, P):
        P = np.clip(P, 0, None)
        tot = 0.0
        for a in self.subsets:
            p = self._marg(P, a)
            p = p[p > 0]
            tot += self.c.get(len(a), 0.0) * (-(p * np.log(p)).sum())
        return tot

    def terms(self, P, nmax=3):
        """[delta^2 phi, delta^3 phi, ...] at U, from Methods Eq. nth-rho."""
        out = []
        for n in range(2, nmax + 1):
            s = sum(self.c.get(len(a), 0.0) * np.mean((self._marg(P, a) * self.q ** len(a) - 1) ** n)
                    for a in self.subsets)
            out.append((-1) ** (n - 1) / (n * (n - 1)) * s)
        return out

    def r_max(self, theta):
        """Distance from U to the boundary of the simplex along the direction theta in (a, b)."""
        d = np.cos(theta) * self.p2 + np.sin(theta) * self.p3
        neg = d < -1e-12
        return np.min(-1.0 / d[neg])


def targets(sl):
    """Slice coordinates (a, b) of COPY and GXOR."""
    X, q = sl.X, sl.q
    copy = (X.min(1) == X.max(1)).astype(float); copy /= copy.sum()
    gxor = ((X.sum(1) % q) == 0).astype(float); gxor /= gxor.sum()
    return np.array(sl.coords(copy)), np.array(sl.coords(gxor))
