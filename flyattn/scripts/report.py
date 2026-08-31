"""Collect every results/*.json into a single markdown report."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "RESULTS.md")


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(v, n=4):
    return "-" if v is None else f"{v:.{n}f}"


def m1(w):
    r = load("m1_temperature.json")
    if not r:
        return
    f = r["full"]
    w(f"\n## M1 - geometric temperature\n")
    w(f"Full >=5-synapse graph, giant component: N={f['n']}, "
      f"<k>={f['mean_degree']:.2f}, mean local clustering={f['clustering']:.4f}, "
      f"gamma={f['gamma']['gamma']:.3f} (k_min={f['gamma']['k_min']}, "
      f"KS={f['gamma']['ks']:.4f}).\n")
    w("| subsample | n | <k> | clustering | gamma | beta_fit | status |")
    w("|---|---|---|---|---|---|---|")
    for x in r["fits"]:
        w(f"| {x['scale']} (rep {x['rep']}) | {x['n']} | {x['mean_degree']:.2f} | "
          f"{x['clustering']:.4f} | {x['gamma']['gamma']:.3f} | "
          f"**{x['beta']:.3f}** | {x['status']} |")
    b = np.array([x["beta"] for x in r["fits"]])
    phase = ("hot / non-geometric" if b.mean() < 1 else
             "quasi-geometric" if b.mean() < 2 else "cold / over-clustered")
    w(f"\n**beta = {b.mean():.3f} +- {b.std():.3f}** across scales -> {phase} "
      f"(beta_c = 1, beta = 2D = 2).\n")


def m2(w):
    r = load("m2_curvature.json")
    if not r:
        return
    w("\n## M2 - Balanced Forman curvature\n")
    w(f"{r['n_edges_sampled']} edges sampled from {r['n_edges_total']}, "
      f"{r['n_null']} degree-preserving rewirings.\n")
    w("| | mean | sd | median | frac negative | 1st pct | 5th pct |")
    w("|---|---|---|---|---|---|---|")
    for nm, d in (("FlyWire", r["empirical"]), ("degree-preserving null", r["null_pooled"])):
        w(f"| {nm} | {d['mean']:.4f} | {d['sd']:.4f} | {d['median']:.4f} | "
          f"{d['frac_negative']:.4f} | {d['q01']:.4f} | {d['q05']:.4f} |")
    w(f"\nDifference in mean curvature {r['delta_mean']:.4f} "
      f"(z = {r['z_of_mean']:.1f} against the spread of null replicate means); "
      f"negative-edge fraction {r['delta_frac_negative']:+.4f}.\n")


def m3(w):
    r = load("m3_richclub.json")
    if not r:
        return
    w("\n## M3 - asymmetry census and rich club\n")
    w("| min total degree | neurons | broadcasters | integrators | asymmetric fraction of brain |")
    w("|---|---|---|---|---|")
    for x in r["asymmetry_census"]:
        w(f"| {x['min_total_degree']} | {x['n_neurons']} | {x['broadcasters']} | "
          f"{x['integrators']} | {x['asym_fraction_of_brain']:.4f} |")
    rc = r["rich_club"]
    w("\nNormalised rich-club coefficient phi(k)/phi_null(k):\n")
    w("| k | phi | phi normalised |")
    w("|---|---|---|")
    for k, p, pn in zip(rc["k"], rc["phi"], rc["phi_normalised"]):
        w(f"| {k} | {p:.5f} | {pn:.4f} |")


def t2(w):
    runs = []
    for p in glob.glob(os.path.join(RES, "t2_keybias*.json")):
        runs += json.load(open(p))["runs"]
    if not runs:
        return
    w("\n## T2 - key-norm bias\n")
    keys = ["val", "val_hard", "ood_book", "ood_brown", "ood_reuters"]
    w("| arm | seeds | " + " | ".join(keys) + " |")
    w("|---" * (len(keys) + 2) + "|")
    for arm in sorted({r["arm"] for r in runs}):
        f = [r["final"] for r in runs if r["arm"] == arm]
        cells = " | ".join(f"{np.mean([x[k] for x in f]):.4f} ± "
                           f"{np.std([x[k] for x in f]):.4f}" for k in keys)
        w(f"| {arm} | {len(f)} | {cells} |")


def t1(w):
    r = load("t1_pruning.json")
    if not r:
        return
    w("\n## T1 - pruning and the shuffling threshold\n")
    w(f"W_V/W_O parameters: {r['n_vo_params']}. Dense val loss "
      f"{r['dense']['loss']:.4f} (hard {r['dense']['hard_loss']:.4f}). "
      "No retraining.\n")
    arms = ("trained", "shuffled_w", "shuffled_mask", "random_both")
    w("| density | kept | " + " | ".join(arms) + " | trained - shuffled_w |")
    w("|---" * (len(arms) + 3) + "|")
    for x in r["rows"]:
        gap = x["shuffled_w"]["val"][0] - x["trained"]["val"][0]
        w(f"| {x['density']:g} | {x['kept_weights']} | "
          + " | ".join(f"{x[a]['val'][0]:.4f}" for a in arms)
          + f" | {gap:+.4f} |")
    b = load("t1b_finetune_shard0.json")
    if b:
        w("\n### T1b - after fine-tuning under a fixed mask "
          f"({b['steps']} steps)\n")
        agg = defaultdict(list)
        for x in b["rows"]:
            agg[(x["density"], x["arm"])].append(x)
        dens = sorted({k[0] for k in agg}, reverse=True)
        arms2 = ("trained", "shuffled_w", "shuffled_mask")
        w("| density | " + " | ".join(arms2) + " |")
        w("|---" * (len(arms2) + 1) + "|")
        for d in dens:
            cells = []
            for a in arms2:
                v = [x["val"] for x in agg.get((d, a), [])]
                cells.append(f"{np.mean(v):.4f} ± {np.std(v):.4f}" if v else "-")
            w(f"| {d:g} | " + " | ".join(cells) + " |")


def sweeps(w):
    for grid, title in (("t3", "T3 - geometric temperature sweep"),
                        ("t4", "T4 - QK-init criticality x structure")):
        runs = []
        for p in sorted(glob.glob(os.path.join(RES, f"sweep_{grid}_shard*.json"))):
            runs += json.load(open(p))["runs"]
        if not runs:
            continue
        w(f"\n## {title}\n")
        keys = ["val", "val_hard", "ood_brown", "ood_reuters"]
        group = defaultdict(list)
        for r in runs:
            group[(r["structure"], r["data_scale"], r["qk_gain"])].append(r["final"])
        w("| structure | data | qk gain | n | mask density | " + " | ".join(keys) + " |")
        w("|---" * (len(keys) + 5) + "|")
        for k in sorted(group, key=lambda t: (t[1], t[2], t[0])):
            f = group[k]
            md = next(r["mask_density"] for r in runs
                      if (r["structure"], r["data_scale"], r["qk_gain"]) == k)
            cells = " | ".join(f"{np.mean([x[m] for x in f]):.4f} ± "
                               f"{np.std([x[m] for x in f]):.4f}" for m in keys)
            w(f"| {k[0]} | {k[1]} | {k[2]:g} | {len(f)} | {md:.4f} | {cells} |")


def t5(w):
    r = load("t5_curvature.json")
    if not r:
        return
    w("\n## T5 - curvature of the attention masks, and SDRF surgery\n")
    w("| mask | edges | mean BFC | 5th pct BFC | frac negative | mean span | frac long |")
    w("|---|---|---|---|---|---|---|")
    for nm, s in r["masks"].items():
        w(f"| {nm} | {s['n_edges']} | {s['bfc_mean']:.4f} | {s['bfc_q05']:.4f} | "
          f"{s['bfc_frac_neg']:.4f} | {s['mean_span']:.1f} | {s['frac_long']:.4f} |")
    if r["runs"]:
        w("\n| mask | gap | accuracy | eval loss |")
        w("|---|---|---|---|")
        g = defaultdict(list)
        for x in r["runs"]:
            g[(x["mask"], x["gap"])].append(x)
        for k in sorted(g):
            v = g[k]
            w(f"| {k[0]} | {k[1]} | {np.mean([x['acc'] for x in v]):.3f} ± "
              f"{np.std([x['acc'] for x in v]):.3f} | "
              f"{np.mean([x['eval_loss'] for x in v]):.4f} |")


def main():
    lines = []
    w = lines.append
    w("# Results\n")
    w("Every number here comes from the real FlyWire v783 connectome or from "
      "training runs on real text (Project Gutenberg, with held-out books, "
      "Brown and Reuters as increasingly distant OOD sets). Baselines are "
      "degree-preserving rewirings or hot-limit S1 samples, never Erdos-Renyi.\n")
    for f in (m1, m2, m3, t1, t2, sweeps, t5):
        f(w)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
