#!/usr/bin/env python3
"""Does the energy landscape lose the answer, or does the averaging readout?

E(q,m) = ||q-m||^2 is zero exactly when q = m, so the lowest-energy memory is
by definition the right one.  Attention, however, does not return the lowest
energy memory: it returns sum_j p_j m_j, a convex combination.  This script
separates the two:

  argmin      hard readout: the memory of lowest pairwise energy
  softmax     the attention readout, at a range of inverse temperatures
  sparsemax   the same, with a gate that can put weight on a single memory

Theory (Ramsauer et al.) says the retrieval error falls like exp(-beta * Delta)
for pattern separation Delta, so a failure at fixed beta should be curable by
raising beta -- unless the memories are genuinely indistinguishable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ebt.hopfield import (argmin_energy, energy_gap, make_patterns,
                          retrieval_accuracy, separation)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--patterns", type=int, default=32)
    ap.add_argument("--clusters", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--spreads", nargs="+", type=float, default=[0.05, 0.15, 0.35])
    ap.add_argument("--betas", nargs="+", type=float, default=[1, 4, 16, 64, 256, 1024])
    ap.add_argument("--out", default=str(ROOT / "results" / "readout.json"))
    a = ap.parse_args()

    rows = []
    print("Retrieval accuracy.  Query = pattern + noise (noise = 0.5 * spread).")
    print("'argmin' is the readout the energy definition implies; the rest are attention.\n")
    for spread in a.spreads:
        sep, gap, hard = 0.0, 0.0, 0.0
        for s in range(a.seeds):
            g = torch.Generator().manual_seed(s)
            X = make_patterns(a.patterns, a.dim, g, clusters=a.clusters, spread=spread)
            t = torch.arange(a.patterns)
            q = X + 0.5 * spread * torch.randn(X.shape, generator=g)
            sep += float(separation(X).mean()) / a.seeds
            gap += float(energy_gap(q, X, t).mean()) / a.seeds
            hard += float((argmin_energy(q, X) == t).float().mean()) / a.seeds
        print(f"spread={spread}   separation={sep:8.2f}   mean energy gap={gap:7.3f}")
        print(f"  {'argmin (hard, beta=inf)':<26}{hard:8.3f}")
        rows.append({"spread": spread, "kind": "argmin", "beta": None,
                     "acc": hard, "sep": sep, "gap": gap})
        for kind, score in (("softmax", "dot"), ("sparsemax", "dot"),
                            ("softmax", "energy"), ("sparsemax", "energy")):
            line = f"  {kind + ' [' + score + ']':<26}"
            for beta in a.betas:
                acc = 0.0
                for s in range(a.seeds):
                    g = torch.Generator().manual_seed(s)
                    X = make_patterns(a.patterns, a.dim, g, clusters=a.clusters, spread=spread)
                    t = torch.arange(a.patterns)
                    q = X + 0.5 * spread * torch.randn(X.shape, generator=g)
                    acc += retrieval_accuracy(q, X, t, beta, kind, 1, score)["correct"] / a.seeds
                rows.append({"spread": spread, "kind": kind, "score": score,
                             "beta": beta, "acc": acc,
                             "sep": sep, "gap": gap})
                line += f"{acc:8.3f}"
            print(f"  {'':<0}{line}")
        print(f"  {'':<28}" + "".join(f"b={b:<6g}".rjust(8) for b in a.betas) + "\n")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(a), "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
