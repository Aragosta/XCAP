"""A3 - the long-range structure of the data itself.

Lin & Tegmark (Criticality in Formal Languages and Statistical Physics, 2017)
showed that mutual information between symbols in natural language decays as a
power law I(s) ~ s^-k, not exponentially - a regular grammar gives exponential
decay, a context-free one can give a power law, and real text is power law.

That gives the *data* an exponent, in the same units as the mask's span exponent.
If a mask is a prior over which pairs may interact, the natural question is how
the exponent of the prior relates to the exponent of the dependence structure it
is meant to cover. This measures the data-side number on the same corpus every
training run in this project used.

Estimated two ways because the byte-level joint has 65536 cells:
  bytes      256 symbols, with the Miller-Madow bias correction
  reduced    27 symbols (a-z plus everything else), where the joint is 729 cells
             and the estimate is essentially unbiased at this sample size
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flyattn import textdata  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
LAGS = np.unique(np.round(np.geomspace(1, 4096, 45)).astype(int))


def reduce_alphabet(arr):
    """a-z -> 0..25, everything else -> 26."""
    low = arr | 0x20                      # crude lowercase for ASCII letters
    out = np.full(len(arr), 26, np.int16)
    m = (low >= 97) & (low <= 122)
    out[m] = low[m] - 97
    return out


def mutual_information(x, s, k, miller_madow=True):
    """I(X_t ; X_{t+s}) in nats, from the empirical joint."""
    a, b = x[:-s], x[s:]
    joint = np.bincount(a.astype(np.int64) * k + b, minlength=k * k).astype(np.float64)
    n = joint.sum()
    joint /= n
    pj = joint.reshape(k, k)
    px = pj.sum(1)
    py = pj.sum(0)
    nz = pj > 0
    hxy = -np.sum(pj[nz] * np.log(pj[nz]))
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    mi = hx + hy - hxy
    if miller_madow:
        # first-order bias correction: (support - 1) / (2n) per entropy term
        mi -= ((nz.sum() - 1) - (np.sum(px > 0) - 1) - (np.sum(py > 0) - 1)) / (2 * n)
    return float(mi)


def fit_powerlaw(lags, mi, lo, hi):
    m = (lags >= lo) & (lags <= hi) & (mi > 0)
    if m.sum() < 5:
        return np.nan, np.nan
    sl, ic = np.polyfit(np.log(lags[m]), np.log(mi[m]), 1)
    r2 = np.corrcoef(np.log(lags[m]), np.log(mi[m]))[0, 1] ** 2
    return float(-sl), float(r2)


def fit_exponential(lags, mi, lo, hi):
    m = (lags >= lo) & (lags <= hi) & (mi > 0)
    if m.sum() < 5:
        return np.nan
    return float(np.corrcoef(lags[m], np.log(mi[m]))[0, 1] ** 2)


def main():
    corp = textdata.load_corpora()
    out = {}
    for name, arr in (("gutenberg_train", corp["train"]),
                      ("brown", corp["ood_brown"]),
                      ("reuters", corp["ood_reuters"])):
        for enc, (x, k) in (("bytes", (arr.astype(np.int16), 256)),
                            ("reduced", (reduce_alphabet(arr), 27))):
            mi = np.array([mutual_information(x, int(s), k) for s in LAGS])
            kk, r2p = fit_powerlaw(LAGS, mi, 4, 1024)
            r2e = fit_exponential(LAGS, mi, 4, 1024)
            out[f"{name}/{enc}"] = dict(lags=LAGS.tolist(), mi=mi.tolist(),
                                        exponent=kk, r2_powerlaw=r2p,
                                        r2_exponential=r2e)
            print(f"{name:16s} {enc:8s} I(1)={mi[0]:.4f} I(64)={mi[LAGS.searchsorted(64)]:.5f} "
                  f"I(1024)={mi[LAGS.searchsorted(1024)]:.6f}  "
                  f"exponent k={kk:.3f} (R2 power={r2p:.3f}, R2 exp={r2e:.3f})",
                  flush=True)
    json.dump(out, open(os.path.join(RES, "a3_mutual_information.json"), "w"))


if __name__ == "__main__":
    main()
