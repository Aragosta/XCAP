#!/usr/bin/env python3
"""Test the energy view where its theory makes sharp predictions -- no training.

Four experiments, each a direct check of a published claim:

  capacity     retrieval vs number of stored patterns (Ramsauer: capacity is
               exponential in d, so at d=64 nothing here should saturate it)
  metastable   retrieval when patterns are NOT well separated.  Theory says the
               softmax fixed point collapses to the *mean* of the similar
               patterns; the sparse-Hopfield margin result says sparsemax can
               still retrieve exactly.  This is the decisive comparison.
  beta         retrieval vs inverse temperature: the phase transition between
               one global attractor and pattern-specific attractors
  iterate      does running the update repeatedly (the Energy Transformer's
               recurrent scheme) help, and does the energy actually descend?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from torch import Tensor

from ebt.hopfield import (energy, make_patterns, retrieval_accuracy, retrieve,
                          separation, update)

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("softmax", "sparsemax", "sigmoid")


def _query(X: Tensor, noise: float, g: torch.Generator):  # noqa: F821
    target = torch.arange(X.size(0))
    return X + noise * torch.randn(X.shape, generator=g), target


def capacity(d: int, noise: float, beta: float, seeds: int, ms: list[int]) -> list[dict]:
    rows = []
    for m in ms:
        for kind in KINDS:
            acc = {"correct": 0.0, "euclid_correct": 0.0, "exact_frac": 0.0,
                   "relative_error": 0.0, "norm_ratio": 0.0}
            for s in range(seeds):
                g = torch.Generator().manual_seed(s)
                X = make_patterns(m, d, g)
                q, t = _query(X, noise, g)
                r = retrieval_accuracy(q, X, t, beta, kind, steps=1)
                for k in acc:
                    acc[k] += r[k] / seeds
            rows.append({"experiment": "capacity", "d": d, "m": m, "kind": kind, **acc})
    return rows


def metastable(d: int, m: int, clusters: int, spreads: list[float], beta: float,
               seeds: int) -> list[dict]:
    rows = []
    for spread in spreads:
        for kind in KINDS:
            acc = {"correct": 0.0, "exact_frac": 0.0, "relative_error": 0.0,
                   "sep": 0.0, "dist_to_cluster_mean": 0.0}
            for s in range(seeds):
                g = torch.Generator().manual_seed(s)
                X = make_patterns(m, d, g, clusters=clusters, spread=spread)
                q, t = _query(X, spread * 0.5, g)
                r = retrieval_accuracy(q, X, t, beta, kind, steps=1)
                out, _ = retrieve(q, X, beta, kind, steps=1)
                idx = torch.arange(m) % clusters
                means = torch.stack([X[idx == c].mean(0) for c in range(clusters)])
                # is the fixed point closer to the pattern or to its cluster mean?
                d_pat = (out - X[t]).norm(dim=-1)
                d_mean = (out - means[idx[t]]).norm(dim=-1)
                acc["dist_to_cluster_mean"] += float((d_mean < d_pat).float().mean()) / seeds
                acc["sep"] += float(separation(X).mean()) / seeds
                for k in ("correct", "exact_frac", "relative_error"):
                    acc[k] += r[k] / seeds
            rows.append({"experiment": "metastable", "spread": spread, "clusters": clusters,
                         "kind": kind, **acc})
    return rows


def beta_sweep(d: int, m: int, noise: float, betas: list[float], seeds: int) -> list[dict]:
    rows = []
    for beta in betas:
        for kind in KINDS:
            acc = {"correct": 0.0, "exact_frac": 0.0}
            for s in range(seeds):
                g = torch.Generator().manual_seed(s)
                X = make_patterns(m, d, g, clusters=m // 4, spread=0.35)
                q, t = _query(X, noise, g)
                r = retrieval_accuracy(q, X, t, beta, kind, steps=1)
                for k in acc:
                    acc[k] += r[k] / seeds
            rows.append({"experiment": "beta", "beta": beta, "kind": kind, **acc})
    return rows


def iterate(d: int, m: int, clusters: int, spread: float, beta: float, steps: int,
            seeds: int, noise: float) -> list[dict]:
    rows = []
    for kind in KINDS:
        per_step, descent_ok = [0.0] * (steps + 1), True
        for s in range(seeds):
            g = torch.Generator().manual_seed(s)
            X = make_patterns(m, d, g, clusters=clusters, spread=spread)
            q, t = _query(X, noise, g)
            for step in range(steps + 1):
                r = retrieval_accuracy(q, X, t, beta, kind, steps=step)
                per_step[step] += r["correct"] / seeds
            if kind != "sigmoid":
                _, traj = retrieve(q, X, beta, kind, steps=steps)
                e = torch.stack(traj)
                tol = 1e-5 * e.abs().max()      # float32 round-off, not a violation
                descent_ok &= bool((e[1:] <= e[:-1] + tol).all())
        rows.append({"experiment": "iterate", "kind": kind,
                     "acc_by_step": [round(v, 4) for v in per_step],
                     "monotone_energy_descent": descent_ok})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "results" / "memory.json"))
    a = ap.parse_args()

    rows = []
    print(f"=== capacity (d={a.dim}, noise={a.noise}, beta={a.beta}, "
          f"{a.seeds} seeds): fraction retrieved to the right pattern")
    rows += capacity(a.dim, a.noise, a.beta, a.seeds, [8, 32, 128, 512, 2048])
    ms = sorted({r["m"] for r in rows if r["experiment"] == "capacity"})
    print("   (cosine-nearest: scale-invariant, so gates that do not normalise are judged fairly)")
    print("kind".ljust(10) + "".join(f"M={m}".rjust(12) for m in ms))
    for kind in KINDS:
        line = kind.ljust(10)
        for m in ms:
            r = next(r for r in rows if r["experiment"] == "capacity"
                     and r["kind"] == kind and r["m"] == m)
            line += f"{r['correct']:12.3f}"
        print(line)

    print(f"\n=== metastable: patterns in 4 tight clusters, spread controls separation")
    print("   'avg' = fraction of queries whose fixed point is closer to the CLUSTER MEAN")
    print("   than to the pattern itself -- the failure the theory predicts for softmax")
    meta = metastable(a.dim, 32, 4, [0.05, 0.15, 0.35, 0.75], a.beta, a.seeds)
    rows += meta
    spreads = sorted({r["spread"] for r in meta})
    print("kind".ljust(10) + "".join(f"spread={s}".rjust(16) for s in spreads))
    for kind in KINDS:
        line = kind.ljust(10)
        for s in spreads:
            r = next(r for r in meta if r["kind"] == kind and r["spread"] == s)
            line += f"{r['correct']:7.2f}/avg{r['dist_to_cluster_mean']:5.2f}"
        print(line)

    print(f"\n=== beta sweep (clustered patterns, one update step)")
    bs = beta_sweep(a.dim, 32, 0.2, [0.1, 0.25, 0.5, 1.0, 2.0, 8.0], a.seeds)
    rows += bs
    betas = sorted({r["beta"] for r in bs})
    print("kind".ljust(10) + "".join(f"b={b}".rjust(10) for b in betas))
    for kind in KINDS:
        line = kind.ljust(10)
        for b in betas:
            r = next(r for r in bs if r["kind"] == kind and r["beta"] == b)
            line += f"{r['correct']:10.3f}"
        print(line)

    print(f"\n=== iterating the update (Energy Transformer's recurrent scheme)")
    it = iterate(a.dim, 32, 4, 0.35, a.beta, 5, a.seeds, noise=0.7)
    rows += it
    for r in it:
        print(f"{r['kind']:10s} acc by step {r['acc_by_step']}   "
              f"monotone energy descent: {r['monotone_energy_descent']}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "rows": rows}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
