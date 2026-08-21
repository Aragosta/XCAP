#!/usr/bin/env python3
"""Turn results.json / scaling.json into results/REPORT.md (+ plots if matplotlib is present)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebt.variants import VARIANT_NAMES

ROOT = Path(__file__).resolve().parents[1]

BLURB = {
    "baseline-softmax": "dense attention, softmax (control)",
    "dense-sparsemax": "dense attention, sparsemax rows",
    "mosa-softmax": "MoSA top-k routing + softmax (MoSA as published)",
    "mosa-sparsemax": "MoSA top-k routing + sparsemax (the two-tier proposal)",
    "smaxroute-softmax": "sparsemax router + softmax (router ablation)",
    "smaxroute-sparsemax": "sparsemax router + sparsemax (fully differentiable sparse)",
}


def agg(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    if not vals:
        return None, None
    return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)


def fmt(m, s, prec=3):
    return "-" if m is None else f"{m:.{prec}f} ± {s:.{prec}f}"


def table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(line(r) for r in rows)])


def build(results_path: Path, scaling_path: Path, out_path: Path) -> str:
    data = json.loads(results_path.read_text())
    cfg, results = data["config"], data["results"]
    by = defaultdict(list)
    for r in results:
        by[(r["task"], r["variant"])].append(r)
    tasks = [t for t in cfg["tasks"] if any(k[0] == t for k in by)]
    variants = [v for v in VARIANT_NAMES if any(k[1] == v for k in by)]

    md = ["# Sparse attention bake-off: MoSA x sparsemax vs. softmax attention", ""]
    md += [f"* {cfg['seeds']} seeds x {cfg['steps']} steps, N={cfg['seq_len']}, "
           f"d_model={cfg['d_model']}, {cfg['n_heads']} heads, {cfg['n_layers']} layers, "
           f"batch {cfg['batch_size']}, lr {cfg['lr']}",
           f"* routed variants get a capacity ratio of {cfg['capacity_ratio']} "
           f"(k = {int(cfg['capacity_ratio'] * cfg['seq_len'])} of {cfg['seq_len']} tokens per head)",
           "* every number is mean ± population std over seeds", ""]

    md += ["## Variants", "", table(
        [[v, BLURB[v]] for v in variants], ["variant", "what it is"]), ""]

    md += ["## Accuracy (final eval)", ""]
    rows = []
    for v in variants:
        row = [v]
        for t in tasks:
            row.append(fmt(*agg(by[(t, v)], "final_acc")))
        rows.append(row)
    md += [table(rows, ["variant", *tasks]), ""]

    md += ["## Loss (final eval)", ""]
    md += [table([[v] + [fmt(*agg(by[(t, v)], "final_loss")) for t in tasks] for v in variants],
                 ["variant", *tasks]), ""]

    md += ["## Sample efficiency (steps to 90% accuracy, '-' = never reached)", ""]
    rows = []
    for v in variants:
        row = [v]
        for t in tasks:
            hit = [r["steps_to_acc"] for r in by[(t, v)] if r["steps_to_acc"]]
            row.append(f"{mean(hit):.0f} ({len(hit)}/{len(by[(t, v)])} seeds)" if hit else "-")
        rows.append(row)
    md += [table(rows, ["variant", *tasks]), ""]

    md += ["## Mechanism diagnostics (averaged over tasks and seeds)", "",
           "`attn_zero_frac`: share of *exactly zero* attention weights inside the block. "
           "`attn_support`: non-zero weights per query row. `token_coverage`: share of tokens "
           "picked by at least one head. `route_support`: tokens per head after routing. "
           "`router_grad_frac`: share of router logits receiving non-zero gradient.", ""]
    keys = ["attn_zero_frac", "attn_support", "attn_entropy", "token_coverage",
            "route_support", "route_support_std", "router_grad_frac"]
    rows = []
    for v in variants:
        runs = [r for t in tasks for r in by[(t, v)]]
        flat = [{**r["final"], **{k: r.get(k) for k in ("router_grad_frac",)}} for r in runs]
        rows.append([v] + [fmt(*agg(flat, k), 3) for k in keys])
    md += [table(rows, ["variant", *keys]), ""]

    md += ["## Cost (as trained: N=%d, batch %d, CPU)" % (cfg["seq_len"], cfg["batch_size"]), ""]
    rows = []
    base = mean(r["fwd_ms"] for r in by[(tasks[0], "baseline-softmax")]) if \
        (tasks[0], "baseline-softmax") in by else None
    for v in variants:
        runs = [r for t in tasks for r in by[(t, v)]]
        f_m, _ = agg(runs, "fwd_ms")
        fb_m, _ = agg(runs, "fwd_bwd_ms")
        fl_m, _ = agg(runs, "flops_per_seq")
        mem_m, _ = agg(runs, "attn_matrix_bytes")
        rows.append([v, f"{f_m:.1f}", f"{fb_m:.1f}",
                     f"{fl_m/1e6:.1f}", f"{mem_m/1e6:.1f}",
                     f"{f_m/base:.2f}x" if base else "-"])
    md += [table(rows, ["variant", "fwd ms/batch", "fwd+bwd ms/batch",
                        "MFLOPs/seq", "attn matrix MB", "rel fwd"]), ""]

    if scaling_path.exists():
        sc = json.loads(scaling_path.read_text())
        md += ["## Scaling with sequence length (forward ms/batch, batch %d)"
               % sc["config"]["batch_size"], ""]
        lens = sorted({r["seq_len"] for r in sc["rows"]})
        rows = []
        for v in VARIANT_NAMES:
            row = [v]
            for n in lens:
                hit = [r for r in sc["rows"] if r["variant"] == v and r["seq_len"] == n]
                row.append(f"{hit[0]['fwd_ms']:.1f}" if hit else "-")
            rows.append(row)
        md += [table(rows, ["variant", *[f"N={n}" for n in lens]]), ""]
        rows = []
        for v in VARIANT_NAMES:
            row = [v]
            for n in lens:
                hit = [r for r in sc["rows"] if r["variant"] == v and r["seq_len"] == n]
                row.append(f"{hit[0]['attn_bytes']/1e6:.1f}" if hit else "-")
            rows.append(row)
        md += ["Attention-matrix memory (MB):", "",
               table(rows, ["variant", *[f"N={n}" for n in lens]]), ""]

    out_path.write_text("\n".join(md) + "\n")
    return "\n".join(md)


def plots(results_path: Path, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                  # pragma: no cover
        print(f"skipping plots: {exc}")
        return
    data = json.loads(results_path.read_text())
    by = defaultdict(list)
    for r in data["results"]:
        by[(r["task"], r["variant"])].append(r)
    tasks = data["config"]["tasks"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        for v in VARIANT_NAMES:
            runs = by.get((t, v))
            if not runs:
                continue
            steps = [h["step"] for h in runs[0]["history"]]
            acc = [mean(r["history"][i]["acc"] for r in runs) for i in range(len(steps))]
            ax.plot(steps, acc, label=v, lw=1.6)
        ax.set_title(t); ax.set_xlabel("step"); ax.set_ylabel("eval accuracy")
        ax.grid(alpha=0.3); ax.set_ylim(0, 1)
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=130)
    print(f"wrote {out_dir / 'learning_curves.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "results.json"))
    ap.add_argument("--scaling", default=str(ROOT / "results" / "scaling.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "REPORT.md"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    md = build(Path(a.results), Path(a.scaling), Path(a.out))
    print(md)
    if not a.no_plots:
        plots(Path(a.results), Path(a.out).parent)


if __name__ == "__main__":
    main()
