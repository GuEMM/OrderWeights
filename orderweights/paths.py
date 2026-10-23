"""O-information along two paths from the uniform distribution U (main text Sec.
'Synergy created by mixing regimes', Fig. 'mixing'; SI Sec. 'Linear-mixing and maximum-entropy paths').

  - linear mixture   P_eps = (1 - eps) U + eps P_target
  - pairwise MaxEnt  the maximum-entropy distribution with the pairwise marginals of P_eps.
    For the target COPY it is the ferromagnetic Potts model P_J ~ exp[J sum_{i<j} delta(x_i, x_j)],
    with J fixed by P_J(x_1 = x_2) = 1/q + eps (1 - 1/q). For independent variables biased by a
    common field (field_paths) it is the Ising model with a uniform field and a uniform coupling.
    For two independent copies it is the product of the two pairs.

The variations are computed from delta^n Omega = (-1)^(n-1)/(n(n-1)) sum_a c_a E_U[rho_a^n]
(Methods, Eq. nth-rho). All quantities are exact, by enumeration of the q^N states.
"""
import itertools as it
from math import comb, log

import numpy as np

__all__ = ["copy_paths", "two_copies_paths", "field_paths", "fit_maxent", "A_m"]


def A_m(m, q):
    """Number of non-zero entry assignments on m variables that sum to 0 mod q (SI, Eq. Am)."""
    return ((q - 1) ** m + (-1) ** m * (q - 1)) / q


def _setup(N, q):
    X = np.array(list(it.product(range(q), repeat=N)), dtype=np.int8)
    # Both paths are invariant under permutations of the variables, and the O-information involves
    # only subsets of 1, N-1 and N variables: one subset of each size, with its multiplicity, suffices.
    subsets = [((0,), N), (tuple(range(N - 1)), N), (tuple(range(N)), 1)]
    c = {k: 0.0 for k in range(1, N + 1)}
    c[1] += 1; c[N - 1] -= 1; c[N] += N - 2
    E = sum((X[:, i] == X[:, j]).astype(float) for i, j in it.combinations(range(N), 2))
    agree = (X[:, 0] == X[:, 1])
    U = np.full(len(X), float(q) ** -N)
    COPY = (X.min(1) == X.max(1)).astype(float); COPY /= COPY.sum()

    def marginal(P, a):
        axes = tuple(i for i in range(N) if i not in a)
        return P.reshape((q,) * N).sum(axis=axes).ravel() if axes else P

    def omega(P):
        tot = 0.0
        for a, mult in subsets:
            p = marginal(P, a); p = p[p > 1e-300]
            tot -= mult * c[len(a)] * (p * np.log(p)).sum()
        return tot

    def term(P, n):
        return (-1) ** (n - 1) / (n * (n - 1)) * sum(
            mult * c[len(a)] * np.mean((marginal(P, a) * q ** len(a) - 1) ** n) for a, mult in subsets)

    return dict(E=E, agree=agree, U=U, COPY=COPY, omega=omega, term=term)


def copy_paths(N, q, n=260, eps_min=1e-3):
    """Omega / ln q along the mixture towards COPY and along the Potts (MaxEnt) path.
    Returns (mix, maxent) dictionaries; the MaxEnt path is labelled by the eps of the
    mixture with the same pairwise marginals."""
    s = _setup(N, q)
    lnq = log(q)
    eps = np.logspace(np.log10(eps_min), 0, n)
    mix = dict(eps=eps, y=np.array([s["omega"]((1 - e) * s["U"] + e * s["COPY"]) for e in eps]) / lnq)
    # the variations of the mixture are homogeneous in eps: evaluate them once at eps = 1
    d2, d3 = s["term"](s["COPY"], 2) / lnq, s["term"](s["COPY"], 3) / lnq
    assert np.isclose(d2 * lnq, 0.5 * sum((2 - m) * comb(N, m) * A_m(m, q) for m in range(3, N + 1)))
    mix.update(o2=d2 * eps ** 2, o23=d2 * eps ** 2 + d3 * eps ** 3, radius=1 / (q ** (N - 1) - 1))
    J = np.logspace(-4, np.log10(14), n)
    pJ, yJ, t2, t3 = [], [], [], []
    for j in J:
        w = np.exp(j * (s["E"] - s["E"].max())); P = w / w.sum()
        pJ.append(P[s["agree"]].sum()); yJ.append(s["omega"](P) / lnq)
        t2.append(s["term"](P, 2) / lnq); t3.append(s["term"](P, 3) / lnq)
    me = dict(eps=(np.array(pJ) - 1 / q) / (1 - 1 / q), J=J, y=np.array(yJ), o2=np.array(t2),
              o23=np.array(t2) + np.array(t3),
              lead=comb(N, 3) * (q - 1) / q ** 3 * J ** 3 / lnq)   # main text, Eq. maxent-omega
    return mix, me


def two_copies_paths(n=260, eps_min=1e-3):
    """Mixture of U with two independent copies (x1 = x2, x3 = x4; N = 4 Boolean variables) and
    the product of two independent pairs, U (1 + eps s1 s2)(1 + eps s3 s4)."""
    lnq = log(2)
    X = np.array(list(it.product(range(2), repeat=4)))
    sp = 1 - 2 * X
    c = {1: 1.0, 2: 0.0, 3: -1.0, 4: 2.0}
    subsets = [a for k in (1, 3, 4) for a in it.combinations(range(4), k)]

    def marginal(P, a):
        axes = tuple(i for i in range(4) if i not in a)
        return P.reshape((2,) * 4).sum(axis=axes).ravel() if axes else P

    def omega(P):
        tot = 0.0
        for a in subsets:
            p = marginal(P, a); p = p[p > 1e-300]
            tot -= c[len(a)] * (p * np.log(p)).sum()
        return tot

    def term(P, k):
        return (-1) ** (k - 1) / (k * (k - 1)) * sum(
            c[len(a)] * np.mean((marginal(P, a) * 2 ** len(a) - 1) ** k) for a in subsets)

    s12, s34 = sp[:, 0] * sp[:, 1], sp[:, 2] * sp[:, 3]
    eps = np.logspace(np.log10(eps_min), 0, n)
    out = []
    for rho in (lambda e: e * (s12 + s34 + s12 * s34), lambda e: e * s12 + e * s34 + e * e * s12 * s34):
        Ps = [(1 + rho(e)) / 16 for e in eps]
        t2 = np.array([term(P, 2) for P in Ps]) / lnq
        t3 = np.array([term(P, 3) for P in Ps]) / lnq
        out.append(dict(eps=eps, y=np.array([omega(P) for P in Ps]) / lnq, o2=t2, o23=t2 + t3))
    out[0]["radius"] = 1 / 3          # ||rho||_inf = 3 eps along the mixture
    # checks quoted in the text: delta^2 = -eps^2, delta^2 + delta^3 = -eps^2 + 2 eps^3; Omega = 0 for the product
    assert np.allclose(out[0]["o2"] * lnq, -eps ** 2)
    assert np.allclose(out[0]["o23"] * lnq, -eps ** 2 + 2 * eps ** 3)
    assert np.abs(out[1]["y"]).max() < 1e-12
    return out


def fit_maxent(features, target, theta0=None, tol=1e-12, max_iter=100):
    """Maximum-entropy distribution P ~ exp(theta . features) over the enumerated states with
    E_P[features] = target. Newton's method on the convex dual, log Z(theta) - theta . target,
    whose gradient is E_P[features] - target and whose Hessian is the covariance of the features,
    with step halving; it converges quadratically and needs no tolerance tuning."""
    F = np.asarray(features, dtype=float)               # shape (n_states, n_features)
    t = np.asarray(target, dtype=float)
    theta = np.zeros(F.shape[1]) if theta0 is None else np.array(theta0, dtype=float)

    def dual(th):
        a = F @ th; amax = a.max(); w = np.exp(a - amax)
        return amax + np.log(w.sum()) - th @ t, w / w.sum()

    D, P = dual(theta)
    for _ in range(max_iter):
        mean = P @ F
        grad = mean - t
        if np.abs(grad).max() < tol:
            return theta, P
        cov = (F - mean).T @ ((F - mean) * P[:, None])
        step = np.linalg.solve(cov, grad)
        lam = 1.0
        while True:                                     # step halving keeps the dual decreasing
            D_new, P_new = dual(theta - lam * step)
            if D_new <= D + 1e-15 or lam < 1e-10:
                break
            lam /= 2
        theta, D, P = theta - lam * step, D_new, P_new
    raise RuntimeError(f"maximum-entropy fit did not converge: |grad| = {np.abs(grad).max():.1e}")


def field_paths(N, p, n=260, eps_min=1e-3):
    """Mixture of U with independent Boolean variables biased by a common field,
    Q_p(x) = prod_i p^{x_i} (1 - p)^{1 - x_i}, and the pairwise MaxEnt path, the Ising model
    P ~ exp[h sum_i s_i + J sum_{i<j} s_i s_j] (s_i = 1 - 2 x_i) with the magnetization and the
    pair correlation of the mixture. Omega / ln 2 along both; both regimes are independent.
    Near U, Omega = (eps^2 / 2) sum_{m >= 3} (2 - m) C(N, m) mu^(2m) + O(eps^3), mu = 1 - 2p
    (Proposition 'mixture': the target has a component mu^|S| on every subset S)."""
    s = _setup(N, 2)
    lnq = log(2)
    X = np.array(list(it.product(range(2), repeat=N)))
    k = X.sum(1)
    Q = p ** k * (1 - p) ** (N - k)
    M = (N - 2 * k).astype(float)                 # sum_i s_i
    Epair = (M ** 2 - N) / 2                      # sum_{i<j} s_i s_j
    mu = 1 - 2 * p
    eps = np.logspace(np.log10(eps_min), 0, n)
    mix = dict(eps=eps, y=np.array([s["omega"]((1 - e) * s["U"] + e * Q) for e in eps]) / lnq)
    d2, d3 = s["term"](Q, 2) / lnq, s["term"](Q, 3) / lnq
    assert np.isclose(d2 * lnq, 0.5 * sum((2 - m) * comb(N, m) * mu ** (2 * m) for m in range(3, N + 1)))
    mix.update(o2=d2 * eps ** 2, o23=d2 * eps ** 2 + d3 * eps ** 3,
               radius=1 / ((2 * max(p, 1 - p)) ** N - 1))
    # MaxEnt: match <sum_i s_i> = N eps mu and <sum_{i<j} s_i s_j> = C(N, 2) eps mu^2 (those of the mixture)
    features = np.column_stack([M, Epair])
    yME, theta = [], np.zeros(2)
    for e in eps[eps < 1]:
        target = np.array([N * e * mu, comb(N, 2) * e * mu ** 2])
        theta, P = fit_maxent(features, target, theta0=theta)
        yME.append(s["omega"](P) / lnq)
    me = dict(eps=eps[eps < 1], y=np.array(yME))
    return mix, me
