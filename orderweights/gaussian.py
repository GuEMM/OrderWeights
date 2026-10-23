"""Plug-in (sample-covariance) estimates of the measures for Gaussian variables
(main text Sec. 'Sampling bias', Methods 'Gaussian systems', SI Sec. 'Sampling bias for
Gaussian variables').

H_hat(X^a) = (1/2) ln det(2 pi e S_a), with S the sample covariance of T samples, n = T - 1.
Since n S_a is Wishart, E ln det S_a = sum_{i<k} psi((n-i)/2) + k ln(2/n) for every covariance,
and the terms linear in k cancel because lambda_1 = 0:

    E[phi_hat] - phi = (1/2) sum_k c_k C(N,k) sum_{i<k} [psi((n-i)/2) - psi(n/2)].
"""
import itertools as it
from math import comb, factorial

import numpy as np
from scipy.special import digamma

from .measures import order_weights, coeffs

__all__ = ["exact_bias", "beta_G", "expansion", "leading_bias", "simulate", "correlated_check"]


def exact_bias(c, N, T):
    """Exact mean of phi_hat - phi for any Gaussian distribution, T > N (SI, Eq. S-gauss-exact)."""
    n = T - 1
    d = np.cumsum([0.0] + [digamma((n - i) / 2) - digamma(n / 2) for i in range(N)])
    return 0.5 * sum(float(c[k - 1]) * comb(N, k) * d[k] for k in range(1, N + 1))


def beta_G(m, n):
    """beta^G_m(n) = (1/2) sum_j (-1)^(m-j) C(m-1, j) psi((n-j)/2)  (Methods, Eq. gauss-bias)."""
    return 0.5 * sum((-1) ** (m - j) * comb(m - 1, j) * digamma((n - j) / 2) for j in range(m))


def expansion(c, N, T, order=2):
    """Large-sample expansion of the bias up to 1/n^2 (SI, Eq. S-gauss-expansion)."""
    n = T - 1
    lam = [float(x) for x in order_weights(c)]
    l2, l3 = lam[1], (lam[2] if N >= 3 else 0.0)
    out = l2 * comb(N, 2) / (2 * n)
    if order >= 2:
        out += (2 * l3 * comb(N, 3) + 3 * l2 * comb(N, 2)) / (4 * n * n)
    return out


def leading_bias(c, N, T):
    """Leading term of each beta^G_m, beta^G_m ~ (m-2)!/(2 n^(m-1)) with n = T - 1 (main text, Sec. bias):
    sum_m lambda_m C(N, m) (m-2)! / (2 n^(m-1))."""
    n = T - 1
    lam = [float(x) for x in order_weights(c)]
    return sum(lam[m - 1] * comb(N, m) * factorial(m - 2) / (2 * n ** (m - 1)) for m in range(2, N + 1))


def simulate(N, T, R, names, rng):
    """R plug-in estimates on T samples of N independent standard Gaussian variables."""
    cs = {m: [float(x) for x in coeffs(m, N)] for m in names}
    used = [k for k in range(1, N + 1) if any(cs[m][k - 1] != 0 for m in names)]
    idx = {k: np.array(list(it.combinations(range(N), k))) for k in used}
    out = {m: np.empty(R) for m in names}
    for r in range(R):
        X = rng.standard_normal((T, N))
        S = np.cov(X, rowvar=False)
        Hk = np.zeros(N + 1)          # sum over subsets of size k of (1/2) ln det S_a; constants cancel
        for k, I in idx.items():
            Hk[k] = 0.5 * np.linalg.slogdet(S[I[:, :, None], I[:, None, :]])[1].sum()
        for m in names:
            out[m][r] = sum(cs[m][k - 1] * Hk[k] for k in range(1, N + 1))
    return out


def correlated_check(N=6, T=30, R=20000, seeds=(1, 2, 3),
                     names=("TC", "DTC", "Sigma", "Omega", "coI")):
    """The exact mean holds for correlated Gaussians too (SI, 'Numerical check').
    Covariance C = A A^T + 0.05 I with A a random Gaussian matrix. Returns a list of rows."""
    cs = {m: [float(x) for x in coeffs(m, N)] for m in names}
    idx = {k: np.array(list(it.combinations(range(N), k))) for k in range(1, N + 1)}

    def phi_from_cov(Ss):
        Hk = np.zeros((Ss.shape[0], N + 1))
        for k, I in idx.items():
            sub = Ss[:, I[:, :, None], I[:, None, :]]
            Hk[:, k] = 0.5 * np.linalg.slogdet(sub)[1].sum(axis=1)
        return {m: sum(cs[m][k - 1] * Hk[:, k] for k in range(1, N + 1)) for m in names}

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((N, N))
        C = A @ A.T + 0.05 * np.eye(N)
        true = phi_from_cov(C[None])
        L = np.linalg.cholesky(C)
        X = rng.standard_normal((R, T, N)) @ L.T
        Xc = X - X.mean(axis=1, keepdims=True)
        est = phi_from_cov(np.einsum("rti,rtj->rij", Xc, Xc) / (T - 1))
        Xi = rng.standard_normal((R, T, N))
        Xi -= Xi.mean(axis=1, keepdims=True)
        ind = phi_from_cov(np.einsum("rti,rtj->rij", Xi, Xi) / (T - 1))
        for m in names:
            b = est[m] - true[m][0]
            ex = exact_bias(cs[m], N, T)
            rows.append(dict(seed=seed, measure=m, bias=b.mean(), exact=ex,
                             z=(b.mean() - ex) / (b.std() / np.sqrt(R)),
                             sd=b.std(), sd_ratio=b.std() / ind[m].std()))
    return rows
