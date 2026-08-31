"""Consolidate probe outputs into the tables the write-up quotes."""
import json, os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, "results", name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(v, n=3):
    return "  n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"


def main():
    files = sys.argv[1:] or ["moe.json", "dense.json", "moe_shuf.json", "moe_init.json"]
    runs = [(f, load(f)) for f in files]
    runs = [(f, r) for f, r in runs if r]
    wn = load("wordnet.json")

    print("=" * 78)
    print("P1b  STRUCTURE: hyperbolic advantage relative to a covariance-matched null")
    print("     calibration -- tree-metric cloud 3.95 | isotropic gaussian 1.23 | matched gaussian 1.05")
    print("=" * 78)
    for f, r in runs:
        v = [x["ratio"] for x in r["p1b_curvature"]]
        print(f"{f:16s} layers: " + " ".join(fmt(x, 2) for x in v) + f"   mean {np.mean(v):.2f}")
    if wn:
        print(f"{'wordnet (delta)':16s} delta_rel max {wn['wordnet_max']['delta_rel']:.3f} "
              f"mean {wn['wordnet']['delta_rel_mean']:.4f}")

    print()
    print("=" * 78)
    print("P4  CHART COMPARISON: held-out R^2 on the residual UPDATE  (higher = more linear)")
    print("    euclid = R_max->0 limit of the hyperbolic chart, so the sweep is nested")
    print("=" * 78)
    for f, r in runs:
        print(f"\n--- {f}")
        print(f"{'layer':>5} {'euclid':>8} {'hyp_best':>9} {'gain':>8} {'+-sd':>7} "
              f"{'R*':>5} {'rff192':>8} {'rff1024':>8} {'tanh1':>7}")
        for x in r["p4_charts"]:
            print(f"{x['layer']:>5} {fmt(x['euclid_d'])!s:>8} {fmt(x['hyp_best_d'])!s:>9} "
                  f"{fmt(x['hyp_gain_d'],4)!s:>8} {fmt(x.get('hyp_gain_d_sd'),4)!s:>7} "
                  f"{fmt(x['hyp_best_R'],2)!s:>5} {fmt(x['rff192_d'])!s:>8} "
                  f"{fmt(x['rff1024_d'])!s:>8} {fmt(x['tanh1_d'])!s:>7}")
        g_h = np.mean([x["hyp_gain_d"] for x in r["p4_charts"]])
        g_r = np.mean([x["rff1024_d"] - x["euclid_d"] for x in r["p4_charts"]])
        print(f"   mean gain over Euclidean:  curving {g_h:+.4f}   lifting(rff1024) {g_r:+.4f}")

    print()
    print("=" * 78)
    print("P3  KOOPMAN IN DEPTH: linearity of the layer map")
    print("=" * 78)
    for f, r in runs:
        print(f"{f:16s} next-state R2 " + " ".join(fmt(x["r2"], 3) for x in r["p3_koopman"]))
        print(f"{'':16s} UPDATE  R2    " + " ".join(fmt(x["r2_update"], 3) for x in r["p3_koopman"]))
        print(f"{'':16s} spectral radius " + " ".join(fmt(x["spec_radius"], 2) for x in r["p3_koopman"])
              + f"   global(update) {fmt(r['p3_global'].get('global_r2_update'))}")

    print()
    print("=" * 78)
    print("P2/P5/P7  DEPTH, ENERGY, SIBLING DISTANCE")
    print("=" * 78)
    for f, r in runs:
        p2 = [x["r_vs_depth_partial"] for x in r["p2_radial"]]
        p5 = [x["E_hyp_vs_logp"] for x in r["p5_energy"]]
        p5e = [x["E_euc_vs_logp"] for x in r["p5_energy"]]
        sr = [x["slope_ratio"] for x in r["p7_siblings"]]
        print(f"{f:16s} radius~depth|pos " + " ".join(fmt(x, 2) for x in p2))
        print(f"{'':16s} corr(-E_hyp,logp) " + " ".join(fmt(x, 2) for x in p5))
        print(f"{'':16s} corr(-E_euc,logp) " + " ".join(fmt(x, 2) for x in p5e))
        print(f"{'':16s} dist/radius slope " + " ".join(fmt(x, 2) for x in sr)
              + "   (hyperbolic prediction: 2.0)")

    print()
    print("=" * 78)
    print("P6  MoE ROUTING: what does the router partition on?")
    print("    angle and radius probes both get 128+ input dims -- capacity is matched")
    print("=" * 78)
    for f, r in runs:
        if "p6_routing" not in r:
            continue
        print(f"\n--- {f}")
        print(f"{'layer':>5} {'majority':>9} {'angle':>8} {'radius':>8} {'full':>8}")
        for x in r["p6_routing"]:
            print(f"{x['layer']:>5} {fmt(x['majority'])!s:>9} {fmt(x['angle_only'])!s:>8} "
                  f"{fmt(x['radius_only'])!s:>8} {fmt(x['full'])!s:>8}")


if __name__ == "__main__":
    main()
