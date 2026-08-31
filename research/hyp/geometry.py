"""Hyperbolic geometry, Gromov delta, and Koopman/EDMD fitting.

The central comparison this file exists to support: a nonlinear map can be made
linear either by *lifting* it (Koopman/EDMD -- more dimensions) or by *curving*
the space it lives in (the Poincare chart -- same dimensions, negative
curvature). Both are conjugations f = phi^-1 . L . phi, so both can be scored
the same way: fit a linear operator in the chart, map back, and measure the
error in the original ambient space on held-out data.
"""
import numpy as np

EPS = 1e-7


# --------------------------------------------------------------------------
# Poincare ball, curvature -c  (c = 0 degenerates to Euclidean, exactly)
# --------------------------------------------------------------------------
def expmap0(v, c):
    if c <= 0:
        return v
    sc = np.sqrt(c)
    n = np.linalg.norm(v, axis=-1, keepdims=True).clip(EPS)
    return np.tanh(sc * n) * v / (sc * n)


def logmap0(x, c):
    if c <= 0:
        return x
    sc = np.sqrt(c)
    n = np.linalg.norm(x, axis=-1, keepdims=True).clip(EPS)
    return np.arctanh((sc * n).clip(max=1 - 1e-6)) * x / (sc * n)


def poincare_dist(x, y, c):
    """Geodesic distance; small c falls back to Euclidean.

    The closed form divides by sqrt(c) after an arccosh that tends to 0, so it
    loses all precision as c -> 0. The limit is exactly the Euclidean distance,
    so take it directly rather than computing 0/0.
    """
    if c <= 1e-6:
        return np.linalg.norm(x - y, axis=-1)
    sq = np.sum((x - y) ** 2, -1)
    xx = np.sum(x * x, -1)
    yy = np.sum(y * y, -1)
    num = 2 * c * sq
    den = ((1 - c * xx) * (1 - c * yy)).clip(EPS)
    return np.arccosh(1 + (num / den).clip(min=0)) / np.sqrt(c)


def to_ball(h, mu, c, fill=0.95):
    """Center activations and rescale so they occupy the ball of curvature c.

    `fill` keeps the cloud off the boundary, where artanh and float64 both die.
    This scaling is the one free choice in the chart; it is applied identically
    to every curvature in the sweep so the comparison stays fair.
    """
    z = h - mu
    r = np.linalg.norm(z, axis=-1).max().clip(EPS)
    radius = 1.0 / np.sqrt(c) if c > 0 else 1.0
    return z * (fill * radius / r), (fill * radius / r)


# --------------------------------------------------------------------------
# Gromov four-point delta-hyperbolicity
# --------------------------------------------------------------------------
def delta_hyperbolicity(D, n_samples=200000, seed=0):
    """Sampled four-point delta, normalised by diameter.

    For points x,y,z,w the three pairings d(x,y)+d(z,w), d(x,z)+d(y,w),
    d(x,w)+d(y,z) satisfy, in a tree, 'the two largest are equal'. delta is
    half the gap between the largest two; delta_rel = 2*delta/diam puts it on
    [0,1] where 0 is an exact tree and ~1 is maximally non-hyperbolic.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    diam = D.max()
    if diam <= 0:
        return dict(delta_rel_mean=np.nan, delta_rel_p95=np.nan, diam=0.0)
    i, j, k, l = (rng.integers(0, n, n_samples) for _ in range(4))
    ok = (i != j) & (i != k) & (i != l) & (j != k) & (j != l) & (k != l)
    i, j, k, l = i[ok], j[ok], k[ok], l[ok]
    s = np.sort(np.stack([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]]), axis=0)
    delta = (s[2] - s[1]) / 2.0
    return dict(delta_rel_mean=float(2 * delta.mean() / diam),
                delta_rel_p95=float(2 * np.percentile(delta, 95) / diam),
                diam=float(diam), n=int(len(i)))


# --------------------------------------------------------------------------
# Linear operator fitting, always scored out of sample
# --------------------------------------------------------------------------
def fit_linear(X, Y, Xte, Yte, ridge=1e-3):
    """Least squares Y ~ A X + b, scored by held-out R^2.

    The ridge is proportional to each column's own Gram diagonal. Every chart in
    the curvature sweep rescales its coordinates, so an absolute ridge -- or one
    scaled by the whole trace, which the intercept column then dominates --
    would regularise the charts by different amounts, and the sweep would be
    measuring regularisation strength instead of geometry.
    """
    Xa = np.hstack([X, np.ones((len(X), 1))])
    G0 = Xa.T @ Xa
    G = G0 + ridge * np.diag(np.diag(G0).clip(min=1e-12))
    W = np.linalg.solve(G, Xa.T @ Y)
    pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ W
    return W, pred


def r2(Y, P):
    ss_res = ((Y - P) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def koopman_spectrum(W, d):
    A = W[:d, :]
    ev = np.linalg.eigvals(A)
    return ev


def delta_max(D, basepoint=0, block=64):
    """True Gromov delta: a maximum over triples, not an average.

    delta = max_{i,j} ( (A o A)_{ij} - A_{ij} ), where A is the Gromov-product
    matrix about a basepoint and (o) is the max-min matrix product. The sampled
    *mean* four-point delta turns out not to separate WordNet from a Gaussian at
    realistic sample sizes -- the max does, so it is the one to report.
    """
    n = D.shape[0]
    w = basepoint
    A = 0.5 * (D[:, w][:, None] + D[w, :][None, :] - D)
    best = np.full((n, n), -np.inf)
    for s in range(0, n, block):
        e = min(s + block, n)
        m = np.minimum(A[:, s:e][:, :, None], A[s:e, :][None, :, :]).max(1)
        np.maximum(best, m, out=best)
    delta = float((best - A).max())
    diam = float(D.max())
    return dict(delta=delta, delta_rel=2 * delta / diam if diam > 0 else np.nan, diam=diam)


def matched_gaussian(H, seed=0):
    """Gaussian with the same mean and covariance as H.

    delta_rel is depressed by distance concentration in high dimension, so a
    raw value is not interpretable on its own. Comparing activations against a
    Gaussian carrying their exact second-order statistics isolates whatever
    tree-likeness is *not* explained by the covariance.
    """
    rng = np.random.default_rng(seed)
    mu = H.mean(0)
    Z = H - mu
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    G = rng.normal(size=(len(H), Vt.shape[0])) * (S / np.sqrt(len(H)))
    return G @ Vt + mu


def rff_lift(X, D, gamma, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    Om = rng.normal(0, np.sqrt(2 * gamma), size=(d, D // 2))
    P = X @ Om
    return np.hstack([np.cos(P), np.sin(P)]) * np.sqrt(2.0 / D)
