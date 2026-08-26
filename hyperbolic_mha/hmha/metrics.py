"""Metrics, statistics, and geometry diagnostics.

Split into three groups:
  * scalar conversions and resource accounting,
  * statistics for comparing arms across seeds (effect size + interval, not a
    bare p-value -- with 3 seeds a p-value alone would be close to meaningless),
  * geometry probes that ask whether hyperbolic attention is *using* curvature
    rather than merely being reparameterised by it.
"""

from __future__ import annotations

import math
import resource

import torch

LN2 = math.log(2.0)


# ------------------------------------------------------------------ scalars


def bits_per_byte(mean_nats: float) -> float:
    """Cross-entropy in nats/byte -> bits/byte, the tokeniser-independent metric."""
    return mean_nats / LN2


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB."""
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024.0


def attention_flops_per_token(seq_len: int, d_model: int, kind: str) -> float:
    """Analytic attention FLOPs per token (multiply-adds counted as 2).

    Projections are linear in ``seq_len``; the score and aggregation matmuls are
    linear per token and therefore quadratic per sequence. The hyperbolic arm
    adds one extra ambient dimension per head plus transcendental work in the
    exp/log maps, which is a constant-factor overhead -- estimated here at the
    conventional ~10 flops per transcendental op.
    """
    proj = 4 * 2 * d_model * d_model
    scores = 2 * 2 * seq_len * d_model
    if kind == "euclidean":
        return proj + scores
    # exp map on q, k, v plus log map on the output, each touching d_model entries
    transcendental = 4 * 10 * d_model
    extra_dim = 2 * 2 * seq_len  # the time coordinate, one per head, in both matmuls
    return proj + scores + transcendental + extra_dim


# --------------------------------------------------------------- statistics


def mean_std(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    std = float(t.std(unbiased=True)) if t.numel() > 1 else 0.0
    return float(t.mean()), std


def welch_t_test(a: list[float], b: list[float]) -> dict:
    """Welch's unequal-variance t-test. Returns the statistic, dof and p-value."""
    from scipy import stats

    if len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "dof": float("nan"), "p_value": float("nan")}
    result = stats.ttest_ind(a, b, equal_var=False)
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    va, vb = float(ta.var(unbiased=True)), float(tb.var(unbiased=True))
    na, nb = len(a), len(b)
    dof_num = (va / na + vb / nb) ** 2
    dof_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    return {
        "t": float(result.statistic),
        "dof": dof_num / dof_den if dof_den > 0 else float("nan"),
        "p_value": float(result.pvalue),
    }


def cohens_d(a: list[float], b: list[float]) -> float:
    """Standardised mean difference (a - b), pooled SD."""
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    if ta.numel() < 2 or tb.numel() < 2:
        return float("nan")
    na, nb = ta.numel(), tb.numel()
    pooled = math.sqrt(
        ((na - 1) * float(ta.var(unbiased=True)) + (nb - 1) * float(tb.var(unbiased=True)))
        / (na + nb - 2)
    )
    return (float(ta.mean()) - float(tb.mean())) / pooled if pooled > 0 else float("nan")


def bootstrap_diff_ci(
    a: list[float], b: list[float], n_resamples: int = 20000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Percentile bootstrap CI for the mean difference ``mean(a) - mean(b)``.

    With three seeds per arm this interval is wide by construction. That width
    is the point: it reports honestly how much the data can support.
    """
    g = torch.Generator().manual_seed(seed)
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)

    ia = torch.randint(0, ta.numel(), (n_resamples, ta.numel()), generator=g)
    ib = torch.randint(0, tb.numel(), (n_resamples, tb.numel()), generator=g)
    diffs = ta[ia].mean(dim=1) - tb[ib].mean(dim=1)

    lo = float(diffs.quantile(alpha / 2))
    hi = float(diffs.quantile(1 - alpha / 2))

    # A bootstrap over fewer than three observations per arm is degenerate: with
    # one value the resamples are all identical and the interval collapses to a
    # point, which would read as a significant result no matter how small the
    # difference. Refuse the significance claim rather than emit a false one.
    n_min = min(ta.numel(), tb.numel())
    excludes_zero = bool(lo > 0 or hi < 0) and n_min >= 3

    return {
        "diff": float(ta.mean() - tb.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": 1 - alpha,
        "n_min": int(n_min),
        "excludes_zero": excludes_zero,
        "underpowered": n_min < 3,
    }


def compare_arms(a: list[float], b: list[float], name_a: str, name_b: str) -> dict:
    """Full comparison of one metric between two arms (lower is better)."""
    mean_a, std_a = mean_std(a)
    mean_b, std_b = mean_std(b)
    return {
        "metric_lower_is_better": True,
        name_a: {"mean": mean_a, "std": std_a, "n": len(a), "values": a},
        name_b: {"mean": mean_b, "std": std_b, "n": len(b), "values": b},
        "difference": bootstrap_diff_ci(a, b),
        "welch": welch_t_test(a, b),
        "cohens_d": cohens_d(a, b),
        "winner": name_a if mean_a < mean_b else name_b,
    }


def log_log_slope(xs: list[float], ys: list[float]) -> dict:
    """Fit ``y = k * x^p`` by least squares in log space; return ``p`` and R^2.

    For attention this recovers the scaling exponent: ~1 would mean linear in
    sequence length, ~2 confirms the quadratic term dominates.
    """
    lx = torch.log(torch.tensor(xs, dtype=torch.float64))
    ly = torch.log(torch.tensor(ys, dtype=torch.float64))
    n = lx.numel()
    slope = ((lx * ly).mean() - lx.mean() * ly.mean()) / (
        (lx * lx).mean() - lx.mean() ** 2
    )
    intercept = ly.mean() - slope * lx.mean()

    pred = slope * lx + intercept
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {
        "exponent": float(slope),
        "coefficient": float(intercept.exp()),
        "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_points": int(n),
    }


# ----------------------------------------------------------------- geometry


def gromov_delta(
    points: torch.Tensor,
    distance: str = "euclidean",
    curvature: float = 1.0,
    n_samples: int = 200,
    seed: int = 0,
) -> dict:
    """Gromov delta-hyperbolicity of a point cloud, via the four-point condition.

    delta measures how tree-like a metric space is: 0 is an exact tree, larger
    values mean flatter/more grid-like. Reported **relative to the diameter**,
    since raw delta scales with the space and would otherwise be incomparable
    between arms.

    Computed on a random subsample -- the exact statistic is O(n^4).
    """
    g = torch.Generator().manual_seed(seed)
    n = min(n_samples, points.shape[0])
    idx = torch.randperm(points.shape[0], generator=g)[:n]
    p = points[idx].double()

    if distance == "euclidean":
        d = torch.cdist(p, p)
    elif distance == "lorentz":
        from .lorentz import expmap0, lorentz_distance

        x = expmap0(p, curvature)
        d = lorentz_distance(x.unsqueeze(1), x.unsqueeze(0), curvature)
    else:
        raise ValueError(f"unknown distance {distance!r}")

    diameter = float(d.max())
    if diameter <= 0:
        return {"delta": float("nan"), "delta_rel": float("nan"), "diameter": diameter}

    # Gromov product w.r.t. base point 0, then delta = max-min gap of the
    # two largest of the three pairings, computed as the standard matrix form.
    row0 = d[0].unsqueeze(0)
    gromov = 0.5 * (row0 + row0.T - d)
    # max-min matrix product: (A o B)_ij = max_k min(A_ik, B_kj)
    maxmin = torch.min(gromov.unsqueeze(-1), gromov.unsqueeze(0)).max(dim=1).values
    delta = float((maxmin - gromov).max())

    return {
        "delta": delta,
        "delta_rel": delta / diameter,
        "diameter": diameter,
        "n_points": n,
    }


def embedding_geometry(model, curvature: float | None = None) -> dict:
    """Geometry of the learned token embeddings.

    ``delta_rel`` near 0 means the learned metric structure is tree-like, which
    is the property hyperbolic space is supposed to accommodate. Reporting it
    for both arms is what separates "hyperbolic helped because hierarchy" from
    "hyperbolic helped for some unrelated optimisation reason".
    """
    emb = model.embed.weight.detach()
    stats = {
        "embedding_norm_mean": float(emb.norm(dim=-1).mean()),
        "embedding_norm_std": float(emb.norm(dim=-1).std()),
        "euclidean": gromov_delta(emb, "euclidean"),
    }
    if curvature is not None:
        stats["lorentz"] = gromov_delta(emb, "lorentz", curvature=curvature)
    return stats
