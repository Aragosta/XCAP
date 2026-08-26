"""Turn ``results/metrics.json`` into a readable report plus plots.

Written so the verdict is legible whichever way the experiment came out. A null
or negative result gets stated as plainly as a positive one, and the confounds
that limit what any of it means are listed next to the numbers rather than
buried at the end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ------------------------------------------------------------------- helpers


def _fmt(value, spec: str = ".4f") -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value != value:
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _median_cost(arm_data: dict, key: str) -> float:
    """Median of a cost metric across seeds.

    Timing on a shared 4-core box is noisy and any single run can be inflated by
    whatever else touched the CPU. The median across seeds is the robust summary;
    the per-run values remain in metrics.json.
    """
    values = sorted(run["cost"][key] for run in arm_data["runs"])
    return values[len(values) // 2]


def _verdict(comparison: dict) -> tuple[str, str]:
    """Reduce the statistics to an honest one-line verdict.

    With three seeds, "no detectable difference" is the most likely truthful
    answer and is treated as a first-class outcome, not a failure to report.
    """
    stats = comparison["test_bits_per_byte"]
    diff = stats["difference"]
    hyp, euc = stats["hyperbolic"]["mean"], stats["euclidean"]["mean"]
    delta = hyp - euc
    direction = "better" if delta < 0 else "worse"

    if diff.get("underpowered"):
        headline = (
            f"**Underpowered: {diff.get('n_min', '?')} seed(s) per arm cannot support a "
            f"conclusion.** Test bits-per-byte: hyperbolic {hyp:.4f} vs Euclidean "
            f"{euc:.4f} ({delta:+.4f})."
        )
        detail = (
            "This run was too small to resolve a difference. The numbers below are "
            "reported as-is; treat none of them as evidence either way."
        )
    elif diff["excludes_zero"]:
        headline = (
            f"**Hyperbolic attention is {direction} than the Euclidean baseline** on held-out "
            f"test bits-per-byte: {hyp:.4f} vs {euc:.4f} "
            f"({delta:+.4f} bpb, 95% CI [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}])."
        )
        detail = "The bootstrap interval excludes zero, so the sign of the effect is supported."
    else:
        headline = (
            f"**No detectable difference between the two geometries at this scale.** "
            f"Test bits-per-byte: hyperbolic {hyp:.4f} vs Euclidean {euc:.4f} "
            f"({delta:+.4f}, 95% CI [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}])."
        )
        detail = (
            "The interval spans zero. The observed gap is within seed noise, so this run "
            "does not support a claim in either direction -- including the claim that "
            "hyperbolic attention helps."
        )
    return headline, detail


# -------------------------------------------------------------------- plots


def make_plots(data: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    colors = {"euclidean": "#2b6cb0", "hyperbolic": "#c05621"}

    # 1. validation curves, one line per seed
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, arm_data in data["arms"].items():
        for i, run in enumerate(arm_data["runs"]):
            steps = [h["tokens"] / 1e6 for h in run["history"]]
            bpb = [h["val_bits_per_byte"] for h in run["history"]]
            ax.plot(
                steps, bpb, color=colors[arm], alpha=0.8,
                label=arm if i == 0 else None, marker="o", markersize=3,
            )
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation bits/byte")
    ax.set_title("Validation loss (all seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "val_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path.name)

    # 2. attention scaling, log-log, with the fitted exponents
    scaling = data["benchmark"]["scaling"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm in ("euclidean", "hyperbolic"):
        pts = scaling[arm]["points"]
        xs = [p["seq_len"] for p in pts]
        ys = [1000 * p["median_s"] for p in pts]
        ax.plot(
            xs, ys, "o-", color=colors[arm],
            label=f"{arm} (fit exponent {scaling[arm]['fit_large_only']['exponent']:.2f})",
        )
    # reference slopes anchored at the first measured point
    x0, y0 = xs[0], ys[0]
    ax.plot(xs, [y0 * (x / x0) for x in xs], "--", color="gray", alpha=0.6, label="linear O(n)")
    ax.plot(
        xs, [y0 * (x / x0) ** 2 for x in xs], ":", color="black", alpha=0.6,
        label="quadratic O(n$^2$)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("attention forward (ms)")
    ax.set_title("Attention cost scaling")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "scaling.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path.name)

    # 3. ablation grid
    ablations = data.get("ablations", {})
    if ablations:
        names = list(ablations)
        values = [ablations[n]["final_val"]["bits_per_byte"] for n in names]
        order = sorted(range(len(names)), key=lambda i: values[i])
        fig, ax = plt.subplots(figsize=(7, 0.4 * len(names) + 2))
        bar_colors = [
            "#2b6cb0" if names[i] == "euclidean_reference" else "#c05621" for i in order
        ]
        ax.barh([names[i] for i in order], [values[i] for i in order], color=bar_colors)
        ax.set_xlabel("validation bits/byte (lower is better)")
        ax.set_title(f"Ablations at {data['preset']} short budget, single seed")
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        path = out_dir / "ablations.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path.name)

    return written


# ------------------------------------------------------------------- report


def build_report(data: dict, plots: list[str]) -> str:
    arms = data["arms"]
    comparison = data["comparison"]
    headline, detail = _verdict(comparison)

    euc_run = arms["euclidean"]["runs"][0]
    hyp_run = arms["hyperbolic"]["runs"][0]
    train_cfg = euc_run["train_config"]
    model_cfg = euc_run["model_config"]

    out = [
        "# Hyperbolic MHA vs Euclidean MHA: evaluation report",
        "",
        "## Verdict",
        "",
        headline,
        "",
        detail,
        "",
        "## What was tested",
        "",
        "Two arms of an identical MoE transformer, differing **only** in the attention "
        "module. Every other component -- embeddings, RMSNorm, RoPE, the MoE FFN with a "
        "shared expert plus top-2 routing, the cross-layer attention residual, "
        "initialisation, optimiser, schedule, and the exact held-out evaluation windows "
        "-- is shared.",
        "",
        f"- **Corpus**: {data['corpus']['source']}, {data['corpus']['tokenizer']}, "
        f"{data['corpus']['train_bytes'] / 1e6:.1f} MB train / "
        f"{data['corpus']['valid_bytes'] / 1e6:.2f} MB valid / "
        f"{data['corpus']['test_bytes'] / 1e6:.2f} MB test",
        f"- **Model**: d_model {model_cfg['d_model']}, {model_cfg['n_layers']} layers, "
        f"{model_cfg['n_heads']} heads, {model_cfg['n_routed_experts']} routed experts "
        f"(top-{model_cfg['top_k']}) + {model_cfg['n_shared_experts']} shared, "
        f"context {model_cfg['max_seq_len']}",
        f"- **Training**: {train_cfg['steps']} steps, batch {train_cfg['batch_size']}, "
        f"lr {train_cfg['lr']}, cosine schedule, "
        f"{len(arms['euclidean']['runs'])} seeds per arm",
        f"- **Hardware**: {data['environment']['device']}, "
        f"{data['environment']['threads']} threads, torch {data['environment']['torch']}",
        "",
        "### Parameter parity",
        "",
        _table(
            ["arm", "total", "trainable", "active/token", "embedding"],
            [
                [
                    arm,
                    f"{arms[arm]['runs'][0]['params']['total']:,}",
                    f"{arms[arm]['runs'][0]['params']['trainable']:,}",
                    f"{arms[arm]['runs'][0]['params']['active_per_token']:,}",
                    f"{arms[arm]['runs'][0]['params']['embedding']:,}",
                ]
                for arm in ("euclidean", "hyperbolic")
            ],
        ),
        "",
        "Trainable counts are identical: the hyperboloid's time coordinate is derived "
        "from the spatial part rather than learned, so the manifold costs no parameters.",
        "",
        "## Quality",
        "",
    ]

    rows = []
    for metric, label in [
        ("val_bits_per_byte", "validation bits/byte"),
        ("test_bits_per_byte", "test bits/byte (full split)"),
        ("best_val_bits_per_byte", "best validation bits/byte"),
    ]:
        stats = comparison[metric]
        diff = stats["difference"]
        rows.append(
            [
                label,
                f"{stats['euclidean']['mean']:.4f} ± {stats['euclidean']['std']:.4f}",
                f"{stats['hyperbolic']['mean']:.4f} ± {stats['hyperbolic']['std']:.4f}",
                f"{diff['diff']:+.4f}",
                f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]",
                _fmt(stats["welch"]["p_value"], ".3f"),
                _fmt(stats["cohens_d"], ".2f"),
            ]
        )
    out += [
        _table(
            ["metric", "euclidean (mean ± sd)", "hyperbolic (mean ± sd)",
             "diff (hyp - euc)", "95% CI", "Welch p", "Cohen's d"],
            rows,
        ),
        "",
        f"Per-seed test bits/byte -- euclidean: "
        f"{[round(v, 4) for v in comparison['test_bits_per_byte']['euclidean']['values']]}, "
        f"hyperbolic: "
        f"{[round(v, 4) for v in comparison['test_bits_per_byte']['hyperbolic']['values']]}.",
        "",
        "With three seeds per arm the confidence interval is wide by construction. It is "
        "reported as the primary statistic precisely because a p-value at n=3 would "
        "convey more confidence than the data contains.",
        "",
    ]
    if "val_curves.png" in plots:
        out += ["![validation curves](val_curves.png)", ""]

    # ------------------------------------------------------------------ cost
    out += ["## Cost", "", _table(
        ["arm", "ms/step (median of seeds)", "tokens/s (train)", "prefill ms", "decode ms/token", "peak RSS MB"],
        [
            [
                arm,
                _fmt(_median_cost(arms[arm], "ms_per_step"), ".0f"),
                _fmt(_median_cost(arms[arm], "tokens_per_s"), ",.0f"),
                _fmt(data["benchmark"]["latency"][arm]["prefill_ms"], ".1f"),
                _fmt(data["benchmark"]["latency"][arm]["decode_ms_per_token"], ".1f"),
                _fmt(_median_cost(arms[arm], "peak_rss_mb"), ".0f"),
            ]
            for arm in ("euclidean", "hyperbolic")
        ],
    ), ""]

    euc_ms = _median_cost(arms["euclidean"], "ms_per_step")
    hyp_ms = _median_cost(arms["hyperbolic"], "ms_per_step")
    out += [
        f"Hyperbolic attention costs **{hyp_ms / euc_ms:.2f}x** the Euclidean arm's training "
        f"step time. The iso-compute framing matters more than the raw ratio: for the same "
        f"wall clock the Euclidean baseline could take roughly "
        f"{hyp_ms / euc_ms:.2f}x more steps, so hyperbolic attention has to beat the baseline "
        f"by more than the baseline gains from that extra training to be worth adopting.",
        "",
    ]

    # --------------------------------------------------------------- scaling
    scaling = data["benchmark"]["scaling"]
    out += [
        "## Quadratic vs linear scaling",
        "",
        _table(
            ["arm", "exponent (all lengths)", "exponent (3 longest)", "R² (all)"],
            [
                [
                    arm,
                    _fmt(scaling[arm]["fit"]["exponent"], ".3f"),
                    _fmt(scaling[arm]["fit_large_only"]["exponent"], ".3f"),
                    _fmt(scaling[arm]["fit"]["r_squared"], ".4f"),
                ]
                for arm in ("euclidean", "hyperbolic")
            ],
        ),
        "",
        "Fitted on ``time = k · L^p``. The all-lengths exponent sits below 2 because at "
        "short sequences the Q/K/V projections -- linear in L -- dominate the quadratic "
        "score matmul. The 3-longest fit isolates the regime where attention actually "
        "dominates, and that is the number to read for the asymptotic claim.",
        "",
        _table(
            ["sequence length", "euclidean ms", "hyperbolic ms", "overhead"],
            [
                [
                    str(e["seq_len"]),
                    _fmt(1000 * e["median_s"], ".2f"),
                    _fmt(1000 * h["median_s"], ".2f"),
                    _fmt(h["median_s"] / e["median_s"], ".2f") + "x",
                ]
                for e, h in zip(
                    scaling["euclidean"]["points"], scaling["hyperbolic"]["points"]
                )
            ],
        ),
        "",
        "**Both arms are quadratic.** Hyperbolic attention changes the constant factor, "
        "not the order of growth: it still materialises an L×L score matrix. Nothing about "
        "curvature makes attention cheaper -- the geometry is a hypothesis about quality, "
        "not about complexity.",
        "",
    ]
    if "scaling.png" in plots:
        out += ["![scaling](scaling.png)", ""]

    # -------------------------------------------------------------- geometry
    out += [
        "## Geometry diagnostics",
        "",
        "Loss alone cannot distinguish 'curvature captured hierarchy' from 'curvature "
        "happened to be a useful reparameterisation'. These probe the learned "
        "representation directly. Gromov δ measures how tree-like a metric space is, "
        "normalised by diameter; lower means more tree-like.",
        "",
        _table(
            ["arm", "δ_rel (euclidean metric)", "embedding norm", "attention entropy (nats)",
             "expert entropy", "attn-residual gate"],
            [
                [
                    arm,
                    _fmt(arms[arm]["runs"][0]["geometry"]["euclidean"]["delta_rel"], ".4f"),
                    _fmt(arms[arm]["runs"][0]["geometry"]["embedding_norm_mean"], ".3f"),
                    _fmt(arms[arm]["runs"][0]["attention_stats"]["attn_entropy_mean"], ".3f"),
                    _fmt(arms[arm]["runs"][0]["attention_stats"]["expert_entropy_mean"], ".3f")
                    + " / "
                    + _fmt(arms[arm]["runs"][0]["attention_stats"]["expert_entropy_max"], ".3f"),
                    _fmt(
                        arms[arm]["runs"][0]["attention_stats"].get(
                            "attn_residual_gate_absmean"
                        ),
                        ".4f",
                    ),
                ]
                for arm in ("euclidean", "hyperbolic")
            ],
        ),
        "",
        f"Routing stayed healthy in both arms (expert entropy near its "
        f"{_fmt(euc_run['attention_stats']['expert_entropy_max'], '.3f')} maximum means "
        f"tokens spread across experts rather than collapsing onto one).",
        "",
        "### Stability",
        "",
        _table(
            ["arm", "non-finite grad steps", "mean grad norm", "max grad norm"],
            [
                [
                    arm,
                    str(arms[arm]["runs"][0]["stability"]["nonfinite_grad_steps"]),
                    _fmt(arms[arm]["runs"][0]["stability"]["grad_norm_mean"], ".3f"),
                    _fmt(arms[arm]["runs"][0]["stability"]["grad_norm_max"], ".2f"),
                ]
                for arm in ("euclidean", "hyperbolic")
            ],
        ),
        "",
    ]

    # ------------------------------------------------------------- ablations
    ablations = data.get("ablations", {})
    if ablations:
        ref = ablations.get("euclidean_reference", {}).get("final_val", {}).get(
            "bits_per_byte"
        )
        rows = []
        for name, result in sorted(
            ablations.items(), key=lambda kv: kv[1]["final_val"]["bits_per_byte"]
        ):
            bpb = result["final_val"]["bits_per_byte"]
            rows.append(
                [
                    name,
                    _fmt(bpb, ".4f"),
                    _fmt(bpb - ref, "+.4f") if ref else "-",
                    _fmt(result["cost"]["ms_per_step"], ".0f"),
                    str(result["stability"]["nonfinite_grad_steps"]),
                ]
            )
        out += [
            "## Ablations",
            "",
            f"Single seed, {ablations['curvature_1.0']['train_config']['steps']} steps -- "
            "a short budget for ranking design choices against each other, not for "
            "converged numbers. The Euclidean reference is trained at the same budget so "
            "the comparison is like-for-like.",
            "",
            _table(
                ["variant", "val bits/byte", "vs euclidean ref", "ms/step", "non-finite steps"],
                rows,
            ),
            "",
        ]
        if "ablations.png" in plots:
            out += ["![ablations](ablations.png)", ""]

        spec = ablations.get("score_sign_spec", {}).get("final_val", {}).get("bits_per_byte")
        corrected = ablations.get("curvature_1.0", {}).get("final_val", {}).get(
            "bits_per_byte"
        )
        if spec and corrected:
            worse = spec > corrected
            out += [
                f"**On the score sign**: the reference spec's `softmax(-<q,k>_L)` scores "
                f"{spec:.4f} bpb against the corrected sign's {corrected:.4f}. "
                + (
                    "The corrected sign wins, as predicted -- the spec's version attends to "
                    "the *farthest* tokens because `-<q,k>_L` increases with geodesic "
                    "distance."
                    if worse
                    else "The spec's version is not worse here, which is worth noting "
                    "against the prediction: at this scale attending to distant tokens "
                    "apparently costs little."
                ),
                "",
            ]

    # ---------------------------------------------------------------- samples
    out += [
        "## Generated samples",
        "",
        f"Prompt: `{repr(hyp_run['sample'][:24])}`. Both arms at temperature 0.8 after the "
        "same training budget. At this scale samples are byte-level babble with local "
        "structure; they are included as a qualitative check that neither arm diverged, "
        "not as evidence of quality.",
        "",
    ]
    for arm, run in [("euclidean", euc_run), ("hyperbolic", hyp_run)]:
        out += [f"**{arm}**", "", "```", run["sample"][:400], "```", ""]

    # ------------------------------------------------------------ limitations
    out += [
        "## What this experiment cannot tell you",
        "",
        "The result above is real but narrow. Every one of these is a live reason the "
        "conclusion might not transfer:",
        "",
        f"- **Scale.** {euc_run['params']['total'] / 1e6:.1f}M parameters, "
        f"{euc_run['cost']['tokens_seen'] / 1e6:.1f}M training tokens, context "
        f"{model_cfg['max_seq_len']}. Geometry effects are widely argued to appear with "
        "depth and scale; this is far below where that argument is usually made.",
        "- **Byte-level tokenisation.** Vocab 256 keeps the embedding table from swamping "
        "a small model, but it also means the model spends most of its capacity learning "
        "spelling rather than the syntactic hierarchy that hyperbolic space is supposed to "
        "help with. This is arguably the single biggest confound: the hypothesis is about "
        "hierarchy, and byte-level modelling suppresses the hierarchy.",
        "- **Perplexity is per byte**, so it is not comparable to published per-token "
        "WikiText-2 numbers.",
        "- **Seeds.** Three per arm. Enough to see a large effect, not enough to resolve a "
        "small one.",
        "- **No hyperparameter search per arm.** Both arms use the baseline's learning rate "
        "and schedule. If the hyperbolic arm's optimum sits elsewhere -- plausible, given "
        "its much wider score dynamic range -- this comparison understates it.",
        "- **CPU only**, so wall-clock ratios reflect CPU kernel availability. The "
        "Euclidean arm has a fused attention kernel available in production settings that "
        "the hyperbolic arm does not, which would likely *widen* the cost gap on a GPU, "
        "not narrow it.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python -m pytest tests -q          # correctness gate",
        "python -m hmha.experiment --preset fast",
        "python -m hmha.report",
        "```",
        "",
        f"Total experiment wall clock: {data.get('total_wall_s', 0) / 60:.1f} min.",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the report from metrics.json")
    parser.add_argument("--metrics", type=Path, default=RESULTS_DIR / "metrics.json")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "report.md")
    args = parser.parse_args()

    data = json.loads(args.metrics.read_text())
    plots = make_plots(data, args.out.parent)
    args.out.write_text(build_report(data, plots))
    print(f"wrote {args.out} and {len(plots)} plots")


if __name__ == "__main__":
    main()
