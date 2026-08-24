"""Collect results/*.json into a markdown report."""
import argparse, json
from pathlib import Path

from lab.variants import VARIANTS

COLS = [
    ("variant", "Variant", "{}"),
    ("total_params", "Params", "{:,}"),
    ("active_params", "Active", "{:,}"),
    ("val_bpc", "Val bits/char", "{:.4f}"),
    ("val_loss", "Val loss", "{:.4f}"),
    ("final_train_loss", "Train loss", "{:.4f}"),
    ("tokens_per_s", "tok/s (CPU)", "{:,.0f}"),
    ("train_time_s", "Train s", "{:.0f}"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    a = p.parse_args()
    rows = [json.loads(f.read_text()) for f in sorted(a.results.glob("*.json"))
            if f.name != "loop_scaling.json"]
    rows.sort(key=lambda r: r["val_bpc"])

    ref = rows[0]
    out = ["# HRM x Kimi linear attention -- small-scale results", ""]
    if rows:
        r0 = rows[0]
        out += [f"Char-level LM on tiny Shakespeare (1.1 MB real text). "
                f"d_model={r0['hidden_size']}, heads={r0['num_heads']}, seq_len={r0['seq_len']}, "
                f"batch={r0['batch_size']}, {r0['steps']} steps "
                f"({r0['steps']*r0['batch_size']*r0['seq_len']:,} tokens), lr={r0['lr']}, seed={r0['seed']}. "
                f"CPU only.", ""]
    out.append("| " + " | ".join(c[1] for c in COLS) + " |")
    out.append("|" + "|".join("---" for _ in COLS) + "|")
    for r in rows:
        out.append("| " + " | ".join(fmt.format(r[key]) for key, _, fmt in COLS) + " |")
    out += ["", "## What each variant is", ""]
    for r in rows:
        out.append(f"- **{r['variant']}** — {VARIANTS[r['variant']].note}")

    loop = a.results / "loop_scaling.json"
    if loop.exists():
        data = json.loads(loop.read_text())
        cycles = sorted({int(c) for v in data.values() for c in v["curve"]})
        out += ["", "## Test-time loop scaling (no retraining)", "",
                "Val bits/char when the trained model is *run* with a different "
                "number of outer H cycles.", "",
                "| Variant | trained H | " + " | ".join(f"H={c}" for c in cycles) + " |",
                "|---|---|" + "|".join("---" for _ in cycles) + "|"]
        for name, v in data.items():
            cells = [f"{v['curve'][str(c)]['val_bpc']:.4f}" if str(c) in v["curve"] else "-"
                     for c in cycles]
            out.append(f"| {name} | {v['trained_H_cycles']} | " + " | ".join(cells) + " |")

    out += ["", "## Samples", ""]
    for r in rows:
        out += [f"### {r['variant']} (bpc {r['val_bpc']:.3f})", "", "```", r["sample"], "```", ""]
    (a.results / "REPORT.md").write_text("\n".join(out))
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
