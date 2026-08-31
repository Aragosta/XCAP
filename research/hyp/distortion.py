"""Does a point cloud embed better in H^k than in R^k, at the same k?

This replaces delta-hyperbolicity as the primary structural test. delta_rel does
not separate WordNet from a Gaussian at realistic sample sizes (measured, see
wordnet_control.py), so it cannot support a claim about activations either.
Embedding distortion does separate them, and it asks the question directly:
holding dimension fixed, does negative curvature buy you a better fit?
"""
import numpy as np
import torch

EPS = 1e-9


def _poincare_dist_t(x, c):
    sq = torch.cdist(x, x).clamp(min=EPS) ** 2
    nn = (x * x).sum(-1)
    den = ((1 - c * nn)[:, None] * (1 - c * nn)[None, :]).clamp(min=1e-6)
    arg = (1 + 2 * c * sq / den).clamp(min=1 + 1e-7)
    return torch.acosh(arg) / np.sqrt(c)


def _expmap0_t(v, c):
    sc = np.sqrt(c)
    n = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.tanh((sc * n).clamp(max=15.0)) * v / (sc * n)


def embed_distortion(D, k=8, c=1.0, steps=700, lr=0.05, seed=0, euclidean=False):
    """Fit a k-dim embedding to the target distances; return scale-free stress.

    Loss is squared log-ratio of distances, so it is invariant to a global
    rescaling -- otherwise the comparison would just be measuring which space
    happened to be scaled better.
    """
    torch.manual_seed(seed)
    n = D.shape[0]
    T = torch.tensor(D, dtype=torch.float32).clamp(min=EPS)
    iu = torch.triu_indices(n, n, offset=1)
    logT = torch.log(T[iu[0], iu[1]])
    v = torch.nn.Parameter(torch.randn(n, k) * 0.01)
    opt = torch.optim.Adam([v], lr=lr)
    for i in range(steps):
        if euclidean:
            d = torch.cdist(v, v).clamp(min=EPS)
        else:
            x = _expmap0_t(v, c)
            x = x * torch.clamp(0.999 / np.sqrt(c) / x.norm(dim=-1, keepdim=True).clamp(min=EPS),
                                max=1.0)
            d = _poincare_dist_t(x, c).clamp(min=EPS)
        pred = torch.log(d[iu[0], iu[1]])
        loss = ((pred - pred.mean()) - (logT - logT.mean())).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        r = (pred - pred.mean()) - (logT - logT.mean())
        stress = float(r.pow(2).mean())
        # average multiplicative distortion after the best global rescale
        dist = float(torch.exp(r.abs()).mean())
    return dict(stress=stress, distortion=dist)


def hyp_advantage(D, k=8, c=1.0, steps=700, seed=0):
    h = embed_distortion(D, k, c, steps, seed=seed, euclidean=False)
    e = embed_distortion(D, k, c, steps, seed=seed, euclidean=True)
    return dict(k=k, c=c, hyp_stress=h["stress"], euc_stress=e["stress"],
                hyp_distortion=h["distortion"], euc_distortion=e["distortion"],
                # >1 means negative curvature fits better at the same dimension
                advantage=e["stress"] / max(h["stress"], 1e-12))
