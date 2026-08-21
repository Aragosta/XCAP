#!/usr/bin/env python3
"""The three properties a coherent energy-attention framework should have.

Memory is `sep(sim(q, M)) @ P` (Universal Hopfield Networks, Millidge et al.
2022).  This script tests the design space for the three things we actually
want from it:

  1. STRUCTURE     dissimilarity is encoded by the similarity stage, so "this
                   query matches nothing" is representable.  Measured as
                   retrieval robustness when memories have unequal norms --
                   the case where a dot product confuses "large" with "close".
  2. COMPRESSION   the memory M is a free argument, so it can be k prototypes
                   instead of N patterns.  Measured as accuracy vs memory
                   budget.
  3. GENERALISATION queries that were never stored are answered by their
                   distance to what is stored.  Measured on held-out points
                   from the same clusters, which is nearest-prototype
                   classification rather than lookup.

Patterns are clustered with unequal norms throughout, because that is the
regime that separates the mechanisms; on well-separated equal-norm memories
every combination scores 1.000 and the experiment says nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ebt.uhn import SEPARATIONS, SIMILARITIES, compress, uhn

ROOT = Path(__file__).resolve().parents[1]


def make_world(n_clusters: int, per_cluster: int, d: int, spread: float,
               norm_lo: float, norm_hi: float, g: torch.Generator):
    """Clustered patterns with unequal norms, plus their cluster labels."""
    centres = torch.randn(n_clusters, d, generator=g)
    label = torch.arange(n_clusters).repeat_interleave(per_cluster)
    X = centres[label] + spread * torch.randn(n_clusters * per_cluster, d, generator=g)
    X = X * torch.empty(X.size(0), 1).uniform_(norm_lo, norm_hi, generator=g)
    return X, label, centres


def structure(a, g_seed: int) -> list[dict]:
    rows = []
    for sim in SIMILARITIES:
        for sep in SEPARATIONS:
            acc = 0.0
            for s in range(a.seeds):
                g = torch.Generator().manual_seed(g_seed + s)
                X, label, _ = make_world(a.clusters, a.per_cluster, a.dim, a.spread,
                                         a.norm_lo, a.norm_hi, g)
                q = X + a.noise * torch.randn(X.shape, generator=g)
                out = uhn(q, X, sim=sim, sep=sep, beta=a.beta)
                acc += float((torch.cdist(out, X).argmin(-1)
                              == torch.arange(X.size(0))).float().mean()) / a.seeds
            rows.append({"experiment": "structure", "sim": sim, "sep": sep, "acc": acc})
    return rows


def compression(a, g_seed: int) -> list[dict]:
    rows = []
    total = a.clusters * a.per_cluster
    budgets = [b for b in (total, total // 2, total // 4, a.clusters * 2, a.clusters) if b >= 1]
    for sim in SIMILARITIES:
        for k in budgets:
            acc = 0.0
            for s in range(a.seeds):
                g = torch.Generator().manual_seed(g_seed + s)
                X, label, _ = make_world(a.clusters, a.per_cluster, a.dim, a.spread,
                                         a.norm_lo, a.norm_hi, g)
                mem, assign = compress(X, k, generator=g)
                # the value read out is the cluster label of each memory slot,
                # one-hot: a heteroassociative memory from pattern to class
                if k >= X.size(0):
                    slot_label = label
                else:
                    slot_label = torch.stack(
                        [torch.bincount(label[assign == c], minlength=a.clusters).argmax()
                         if (assign == c).any() else torch.tensor(0) for c in range(k)])
                P = torch.nn.functional.one_hot(slot_label, a.clusters).float()
                q = X + a.noise * torch.randn(X.shape, generator=g)
                pred = uhn(q, mem, P, sim=sim, sep="softmax", beta=a.beta).argmax(-1)
                acc += float((pred == label).float().mean()) / a.seeds
            rows.append({"experiment": "compression", "sim": sim, "budget": k,
                         "ratio": k / total, "acc": acc})
    return rows


def generalisation(a, g_seed: int) -> list[dict]:
    """Queries drawn from the same clusters but never stored."""
    rows = []
    for sim in SIMILARITIES:
        for sep in ("softmax", "sparsemax", "max"):
            seen, unseen = 0.0, 0.0
            for s in range(a.seeds):
                g = torch.Generator().manual_seed(g_seed + s)
                X, label, centres = make_world(a.clusters, a.per_cluster, a.dim, a.spread,
                                               a.norm_lo, a.norm_hi, g)
                P = torch.nn.functional.one_hot(label, a.clusters).float()
                # novel points from the same clusters, with fresh norms
                novel_label = torch.arange(a.clusters).repeat_interleave(a.per_cluster)
                novel = centres[novel_label] + a.spread * torch.randn(
                    novel_label.numel(), a.dim, generator=g)
                novel = novel * torch.empty(novel.size(0), 1).uniform_(
                    a.norm_lo, a.norm_hi, generator=g)
                # the same corruption as the stored queries, or "novel" would
                # simply be the easier condition and the comparison meaningless
                novel = novel + a.noise * torch.randn(novel.shape, generator=g)
                for name, (q, y) in (("seen", (X + a.noise * torch.randn(X.shape, generator=g), label)),
                                     ("unseen", (novel, novel_label))):
                    pred = uhn(q, X, P, sim=sim, sep=sep, beta=a.beta).argmax(-1)
                    hit = float((pred == y).float().mean()) / a.seeds
                    if name == "seen":
                        seen += hit
                    else:
                        unseen += hit
            rows.append({"experiment": "generalisation", "sim": sim, "sep": sep,
                         "seen": seen, "unseen": unseen})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--per-cluster", type=int, default=16)
    ap.add_argument("--spread", type=float, default=0.6)
    ap.add_argument("--noise", type=float, default=2.0)
    ap.add_argument("--norm-lo", type=float, default=0.5)
    ap.add_argument("--norm-hi", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "framework.json"))
    a = ap.parse_args()

    rows = []
    print(f"{a.clusters} clusters x {a.per_cluster} patterns in {a.dim}d, spread {a.spread}, "
          f"norms U({a.norm_lo},{a.norm_hi}), query noise {a.noise}, {a.seeds} seeds\n")

    print("=== 1. STRUCTURE: exact retrieval of the queried pattern ===")
    st = structure(a, 0); rows += st
    print("sim".ljust(11) + "".join(s.rjust(11) for s in SEPARATIONS))
    for sim in SIMILARITIES:
        print(sim.ljust(11) + "".join(
            f"{next(r for r in st if r['sim'] == sim and r['sep'] == sep)['acc']:11.3f}"
            for sep in SEPARATIONS))

    print("\n=== 2. COMPRESSION: class accuracy vs how many memory slots are kept ===")
    cp = compression(a, 100); rows += cp
    budgets = sorted({r["budget"] for r in cp}, reverse=True)
    print("sim".ljust(11) + "".join(f"k={b}".rjust(10) for b in budgets))
    for sim in SIMILARITIES:
        print(sim.ljust(11) + "".join(
            f"{next(r for r in cp if r['sim'] == sim and r['budget'] == b)['acc']:10.3f}"
            for b in budgets))

    print("\n=== 3. GENERALISATION: class accuracy on points never stored ===")
    gn = generalisation(a, 200); rows += gn
    print(f"{'sim':<11}{'sep':<11}{'stored':>9}{'novel':>9}")
    for r in gn:
        print(f"{r['sim']:<11}{r['sep']:<11}{r['seen']:9.3f}{r['unseen']:9.3f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "rows": rows}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
