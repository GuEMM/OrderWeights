"""Sampling bias of plug-in estimates of discrete measures (main text Sec. 'Sampling bias',
Methods 'Exact mean of plug-in estimates', SI Sec. 'Sampling bias of plug-in estimates').

The simulation routines keep the algorithms (and therefore the random streams) of the
scripts that produced the published figures and tables, so that a run with the same seed
reproduces the published numbers.
"""
import itertools as it
from math import comb, log

import numpy as np
from scipy.stats import binom

from .measures import order_weights, coeffs

__all__ = ["h_T", "expected_entropy", "exact_bias", "beta_m", "exact_bias_lambda",
           "leading_bias", "simulate_uniform", "expected_entropy_product",
           "exact_bias_product", "omega_sigma_exact", "omega_saturation",
           "simulate_autocorrelated"]


def h_T(p, T):
    """Mean of (n/T) ln(n/T) for n ~ Binomial(T, p) (Methods, Eq. hT)."""
    n = np.arange(1, T + 1)
    return np.sum(binom.pmf(n, T, p) * (n / T) * np.log(n / T))


def expected_entropy(K, T):
    """Exact mean plug-in entropy (nats) of T samples from the uniform distribution on K states."""
    return -K * h_T(1.0 / K, T)


def exact_bias(c, N, q, T):
    """Exact mean of the plug-in estimate of phi on independent uniform data (true value 0),
    E[phi_hat] = sum_k c_k C(N,k) E[H_hat_k]  (SI, Eq. S-mean-uniform)."""
    return sum(float(c[k - 1]) * comb(N, k) * expected_entropy(q ** k, T) for k in range(1, N + 1))


def beta_m(m, q, T):
    """beta_m(T) = sum_k (-1)^(m-k) C(m,k) q^k h_T(q^-k): mean plug-in estimate of kappa_S for
    m independent uniform variables (main text, Eq. bias-exact)."""
    return sum((-1) ** (m - k) * comb(m, k) * q ** k * h_T(float(q) ** -k, T) for k in range(1, m + 1))


def exact_bias_lambda(c, N, q, T):
    """The same exact mean, written through the order weights: sum_m lambda_m C(N,m) beta_m(T)."""
    lam = [float(x) for x in order_weights(c)]
    return sum(lam[m - 1] * comb(N, m) * beta_m(m, q, T) for m in range(2, N + 1))


def leading_bias(c, N, q, T):
    """Large-sample approximation (1/2T) sum_m lambda_m C(N,m) (q-1)^m (main text, Eq. bias)."""
    lam = [float(x) for x in order_weights(c)]
    return sum(lam[m - 1] * comb(N, m) * (q - 1) ** m for m in range(2, N + 1)) / (2 * T)


def simulate_uniform(N, q, T, R, names, rng):
    """R plug-in estimates of each measure in `names` on T independent samples of N
    independent uniform q-ary variables. Returns {name: array of length R}."""
    subs = [a for k in range(1, N + 1) for a in it.combinations(range(N), k)]
    cs = {n: [float(x) for x in coeffs(n, N)] for n in names}
    out = {n: np.empty(R) for n in names}
    for r in range(R):
        counts = rng.multinomial(T, np.full(q ** N, q ** -N)).reshape((q,) * N) / T
        Hk = np.zeros(N + 1)            # sum of the plug-in entropies of all subsets of each size
        for a in subs:
            p = counts.sum(axis=tuple(i for i in range(N) if i not in a)).ravel()
            p = p[p > 0]
            Hk[len(a)] -= (p * np.log(p)).sum()
        for n in names:
            out[n][r] = sum(cs[n][k - 1] * Hk[k] for k in range(1, N + 1))
    return out


# ---------------------------------------------------------------- non-uniform marginals
def expected_entropy_product(sub, T, marg):
    """Exact mean plug-in entropy of the subset `sub` of independent variables with
    marginals marg[i] (a list of state probabilities per variable)."""
    return -sum(h_T(np.prod([marg[i][xi] for i, xi in zip(sub, x)]), T)
                for x in it.product(*[range(len(marg[i])) for i in sub]))


def exact_bias_product(c, T, marg):
    """Exact mean of the plug-in estimate on independent data with arbitrary marginals
    (SI, Eq. S-mean-general). Only the subset sizes with c_k != 0 are evaluated."""
    N = len(marg)
    tot = 0.0
    for k in range(1, N + 1):
        if c[k - 1] == 0:
            continue
        tot += float(c[k - 1]) * sum(expected_entropy_product(a, T, marg)
                                     for a in it.combinations(range(N), k))
    return tot


# ---------------------------------------------------------------- fewer samples than states
def omega_sigma_exact(N, q, T):
    """Exact means of the plug-in O- and S-information on independent uniform data."""
    H1, HN1, HN = (expected_entropy(q ** k, T) for k in (1, N - 1, N))
    return N * H1 - N * HN1 + (N - 2) * HN, N * H1 + N * HN1 - N * HN


def omega_saturation(N, q, T):
    """Saturation formula for T << q^(N-1), with the first correction for repeated
    observations (SI, Eq. S-omega-sat)."""
    return N * expected_entropy(q, T) - 2 * log(T) + (T - 1) / q ** N * (N * (q - 1) + 2) * log(2)


# ---------------------------------------------------------------- temporal correlations
def simulate_autocorrelated(N, T, s, R, rng):
    """Plug-in estimates of (Omega, Sigma, coI) for N independent binary variables, each keeping
    its previous value with probability s and otherwise drawn anew (SI, Table S-autocorr).
    Same algorithm and random stream as the script that produced the table."""
    subsets = [a for k in range(1, N + 1) for a in it.combinations(range(N), k)]
    cO = {1: 1, N - 1: -1, N: N - 2}
    cS = {1: 1, N - 1: 1, N: -N}
    cI = {k: (-1) ** (k + 1) for k in range(1, N + 1)}
    pw = 2 ** np.arange(N)

    def estimates(X):
        H = {}
        for a in subsets:
            code = X[:, list(a)] @ pw[:len(a)]
            cnt = np.bincount(code)
            cnt = cnt[cnt > 0] / len(X)
            H[a] = -np.sum(cnt * np.log(cnt))
        return [sum(cc.get(len(a), 0) * H[a] for a in subsets) for cc in (cO, cS, cI)]

    out = np.empty((R, 3))
    for r in range(R):
        X = np.empty((T, N), dtype=np.int64)
        X[0] = rng.integers(0, 2, N)
        keep = rng.random((T, N)) < s
        new = rng.integers(0, 2, (T, N))
        for t in range(1, T):
            X[t] = np.where(keep[t], X[t - 1], new[t])
        out[r] = estimates(X)
    return out
