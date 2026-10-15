"""Labeling-symmetric Shannon measures and their order weights.

A measure of the class is

    phi = sum_{k=1}^{N} c_k sum_{|a|=k} H(X^a),

and its order weights are (main text, Eq. lam_general)

    lambda_m = - sum_{k=m}^{N} c_k C(N-m, k-m),          m = 1..N,
    c_k      = - sum_{m=k}^{N} (-1)^(m-k) C(N-k, m-k) lambda_m.

Coefficients are returned as lists [c_1, ..., c_N] of Fractions, so that the
closed forms of the paper can be checked exactly.
"""
from fractions import Fraction as F
from math import comb, log

__all__ = ["order_weights", "coefficients", "c_measure", "coeffs", "c_u", "c_TSE",
           "c_phiSR", "c_binomial", "hallmark_values", "hallmark_scale", "cost", "LABEL"]


def order_weights(c, m_max=None):
    """Order weights [lambda_1, ..., lambda_N] of the coefficients c = [c_1, ..., c_N]
    (a dict {k: c_k} is also accepted; missing k count as zero)."""
    if isinstance(c, dict):
        N = max(c)
        c = [c.get(k, 0) for k in range(1, N + 1)]
    N = len(c)
    m_max = N if m_max is None else min(m_max, N)
    return [-sum(c[k - 1] * comb(N - m, k - m) for k in range(m, N + 1))
            for m in range(1, m_max + 1)]


def coefficients(lam):
    """Inverse map: [lambda_1, ..., lambda_N] -> [c_1, ..., c_N] (main text, Eq. c-from-lambda)."""
    N = len(lam)
    return [-sum((-1) ** (m - k) * comb(N - k, m - k) * lam[m - 1] for m in range(k, N + 1))
            for k in range(1, N + 1)]


def c_TSE(N):
    """TSE complexity: c_k = 1/C(N,k) for k < N, c_N = -(N-1)/2."""
    return [F(1, comb(N, k)) for k in range(1, N)] + [F(-(N - 1), 2)]


def c_u(N, k):
    """Han's basis element u_k, k = 1..N-1:
    c_{k-1} = -1/C(N,k-1), c_k = 2/C(N,k), c_{k+1} = -1/C(N,k+1)."""
    if not 1 <= k <= N - 1:
        raise ValueError("k must be in 1..N-1")
    c = [F(0)] * N
    if k - 1 >= 1:
        c[k - 2] = F(-1, comb(N, k - 1))
    c[k - 1] = F(2, comb(N, k))
    c[k] = F(-1, comb(N, k + 1))
    return c


def c_phiSR(N):
    """Least biased valid synergy-redundancy index, phi^SR = u_1 - u_{N-1} (main text, Eq. phi-SR)."""
    return [x - y for x, y in zip(c_u(N, 1), c_u(N, N - 1))]


def c_binomial(N, d):
    """Member d of the family lambda_m = (-1)^d C(m-2, d) (Methods, Eq. binomial-family)."""
    lam = [F(0)] + [F((-1) ** d * comb(m - 2, d)) for m in range(2, N + 1)]
    return coefficients(lam)


def c_measure(name, N, k=None):
    """Coefficients [c_1, ..., c_N] of the measures of Table 1."""
    c = [F(0)] * N
    if name == "TC":
        c[0] += 1; c[N - 1] += -1
    elif name == "DTC":
        c[N - 2] += 1; c[N - 1] += -(N - 1)
    elif name == "Omega":
        c[0] += 1; c[N - 2] += -1; c[N - 1] += N - 2
    elif name == "Sigma":
        c[0] += 1; c[N - 2] += 1; c[N - 1] += -N
    elif name == "II":
        c = [F(-(-1) ** j) for j in range(1, N + 1)]
    elif name == "TSE":
        c = c_TSE(N)
    elif name == "u":
        c = c_u(N, k)
    else:
        raise ValueError(name)
    return c


def coeffs(name, N):
    """Like c_measure, but also accepts 'phiSR'."""
    if name == "phiSR":
        return c_phiSR(N)
    return c_measure(name, N)


LABEL = {"Omega": r"$\Omega$", "Sigma": r"$\Sigma$", "II": "II", "TC": "TC", "DTC": "DTC",
         "TSE": "TSE", "phiSR": r"$\phi^{SR}$"}


def hallmark_values(c):
    """(phi(COPY), phi(GXOR)) in units of ln q (main text, Eq. extremals-spectral)."""
    lam = [float(x) for x in order_weights(c)]
    N = len(lam)
    copy = sum((-1) ** m * comb(N, m) * lam[m - 1] for m in range(2, N + 1))
    return copy, lam[-1]


def hallmark_scale(c, q):
    """max(|phi(COPY)|, |phi(GXOR)|) in nats: the natural size of a measure."""
    copy, gxor = hallmark_values(c)
    return max(abs(copy), abs(gxor)) * log(q)


def cost(c):
    """Number of subset entropies needed: all subsets of the sizes with c_k != 0."""
    N = len(c)
    return sum(comb(N, k) for k in range(1, N + 1) if c[k - 1] != 0)
