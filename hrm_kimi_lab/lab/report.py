"""Collect results/*.json into a markdown report, aggregating over seeds."""
import argparse, json, statistics
from pathlib import Path

from lab.variants import VARIANTS


def load(results: Path):
    runs = {}
    for f in sorted(results.glob("*.json")):
        if f.name == "loop_scaling.json":
            continue
        r = json.loads(f.read_text())
        runs.setdefault(r["variant"], []).append(r)
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    a = p.parse_args()
    runs = load(a.results)
    if not runs:
        print("no results yet"); return

    rows = []
    for name, rs in runs.items():
        bpcs = [r["val_bpc"] for r in rs]
        r0 = rs[0]
        rows.append({
            "variant": name, "n_seeds": len(rs),
            "bpc_mean": statistics.mean(bpcs),
            "bpc_sd": statistics.pstdev(bpcs) if len(bpcs) > 1 else None,
            "bpc_all": sorted(bpcs),
            "total_params": r0["total_params"], "active_params": r0["active_params"],
            "blocks": r0.get("block_applications"), "unique": r0.get("unique_blocks"),
            "tok_s": r0["tokens_per_s"], "train_s": r0["train_time_s"],
            "sample": r0["sample"],
        })
    rows.sort(key=lambda r: r["bpc_mean"])
    r0 = next(iter(runs.values()))[0]

    out = ["# HRM x Kimi linear attention — small-scale results", "",
           f"Char-level LM on tiny Shakespeare (1.1 MB real text, val = disjoint final 10%). "
           f"d_model={r0['hidden_size']}, heads={r0['num_heads']}, seq_len={r0['seq_len']}, "
           f"batch={r0['batch_size']}, {r0['steps']} steps "
           f"({r0['steps']*r0['batch_size']*r0['seq_len']:,} tokens — identical budget for every "
           f"variant), lr={r0['lr']}. CPU only.", "",
           "`blocks/fwd` = block forward passes per token per model forward (compute proxy); "
           "`unique` = distinct blocks holding parameters. HRM trades the second for the first.", "",
           "| Variant | Val bits/char | seeds | Params | Active | blocks/fwd | unique | tok/s | Train s |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bpc = f"**{r['bpc_mean']:.4f}**"
        if r["bpc_sd"] is not None:
            bpc += f" ±{r['bpc_sd']:.4f}"
        out.append(f"| {r['variant']} | {bpc} | {r['n_seeds']} | {r['total_params']:,} | "
                   f"{r['active_params']:,} | {r['blocks']} | {r['unique']} | "
                   f"{r['tok_s']:,.0f} | {r['train_s']:.0f} |")

    out += ["", "## What each variant is", ""]
    for r in rows:
        out.append(f"- **{r['variant']}** — {VARIANTS[r['variant']].note}")

    loop = a.results / "loop_scaling.json"
    if loop.exists():
        data = json.loads(loop.read_text())
        cycles = sorted({int(c) for v in data.values() for c in v["curve"]})
        out += ["", "## Test-time loop scaling (no retraining)", "",
                "Val bits/char when a trained model is *run* with a different number of outer "
                "H cycles. A model that improves past its trained cycle count has learned "
                "iterative refinement rather than a fixed-depth function.", "",
                "| Variant | trained H | " + " | ".join(f"H={c}" for c in cycles) + " |",
                "|---|---|" + "|".join("---" for _ in cycles) + "|"]
        for name, v in data.items():
            cells = [f"{v['curve'][str(c)]['val_bpc']:.4f}" if str(c) in v["curve"] else "—"
                     for c in cycles]
            out.append(f"| {name} | {v['trained_H_cycles']} | " + " | ".join(cells) + " |")

    out += ["", "## Samples (seed 0, temperature 0.8)", ""]
    for r in rows:
        out += [f"### {r['variant']} (bpc {r['bpc_mean']:.3f})", "", "```", r["sample"], "```", ""]
    (a.results / "REPORT.md").write_text("\n".join(out))
    print("\n".join(out[:6 + len(rows) + 2]))


if __name__ == "__main__":
    main()
