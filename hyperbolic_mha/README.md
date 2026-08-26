# Hyperbolic MHA evaluation lab

A standalone, self-contained experiment: does **Lorentz-model hyperbolic attention**
beat ordinary Euclidean attention on real text, at matched parameters?

This folder shares no code with the rest of the repository and can be copied out
whole.

## The claim under test

Language is hierarchical. Hyperbolic space has exponential volume growth, matching
the exponential node growth of trees, so it embeds hierarchies with far less
distortion than flat space. Therefore attention scored by *geodesic distance on a
hyperboloid* should model language better than attention scored by dot product.

That is a hypothesis about **quality**, not about cost — a point worth keeping
straight, because hyperbolic attention still materialises an L×L score matrix and
is still quadratic. The benchmark here measures that rather than assuming it.

## Experimental design

Two arms of the same MoE transformer. `ModelConfig.attention` is the *only*
difference — embeddings, RMSNorm, RoPE, the MoE FFN (1 shared expert + top-2 of 4
routed), the cross-layer attention residual, initialisation, optimiser, schedule
and the exact held-out evaluation windows are all shared. Trainable parameter
counts are identical and asserted in a test: the hyperboloid's time coordinate is
*derived* from the spatial part, so curvature costs no parameters.

- **Data**: WikiText-2, real text, byte-level (vocab 256), headline metric
  **bits-per-byte**.
- **Statistics**: 3 seeds per arm, bootstrap CI on the difference plus Welch's
  t-test and Cohen's d. The interval is the primary statistic — at n=3 a p-value
  would imply more confidence than the data holds.
- **Scaling**: attention-only microbenchmark across L ∈ 128…4096, fitted in
  log-log space to recover the exponent empirically.
- **Geometry**: Gromov δ-hyperbolicity of the learned embeddings, attention
  entropy, expert-utilisation entropy — evidence about *why* a difference exists,
  not just that it does.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests -q                  # correctness gate: run this first
python -m hmha.experiment --preset fast    # ~35 min on 4 CPU cores
python -m hmha.report                      # -> results/report.md + plots
```

Or `bash scripts/run_all.sh`. Use `--preset smoke` for a ~1 minute end-to-end
check of the whole pipeline.

Results land in `results/`: `metrics.json` (everything, written incrementally
after each stage), `report.md`, and PNG plots.

## Corrections to the reference specification

The spec this was built from has three formulas that are wrong or belong to a
different model of hyperbolic space. Each correction is the default, each has the
literal version available behind a config flag so the difference is *measured*,
and each is pinned by a test.

1. **Exponential map.** The spec's spatial part `(v/‖v‖)·sinh(√c‖v‖)` is missing a
   `1/√c` factor, so its output does not satisfy `⟨x,x⟩_L = -1/c` and is not on the
   manifold at all (except coincidentally at `c = 1`).
   → `tests/test_lorentz.py::test_spec_expmap_without_sqrt_c_is_off_manifold`

2. **Attention score sign.** The spec scores with `-⟨q,k⟩_L`. But
   `-⟨q,k⟩_L = (1/c)·cosh(√c·d)`, which *increases* with geodesic distance — so
   `softmax(-⟨q,k⟩_L)` puts its mass on the **farthest** tokens, the opposite of
   what attention is for. The corrected default is `+⟨q,k⟩_L`, which is still
   exactly `Q·M·Kᵀ`, so the fast matmul trick is fully preserved; only the sign
   changes. `score_sign="spec"` runs the original.
   → `tests/test_attention.py::test_spec_sign_attends_to_the_farthest_token`

3. **Aggregation.** The spec's Lorentz factor `γ = 1/√(1-c‖v‖²)` is the *Klein*
   model's, not the Lorentz model's. `klein_gyromidpoint` converts coordinates
   first (`k = x_s/(√c·x₀)`) and is the faithful Einstein gyromidpoint; the
   default `lorentz_centroid` is the direct Lorentz analogue and needs no γ.

A fourth change is numerical rather than mathematical: the log map uses
`asinh(√c‖x_s‖)/√c` instead of `arcosh(√c·x₀)`. They agree on the manifold, but
`arcosh` has an infinite derivative at 1 — exactly where points near the origin
sit — and produces NaN gradients there.

## Layout

| path | contents |
|---|---|
| `hmha/lorentz.py` | manifold primitives: exp/log maps, distance, centroids |
| `hmha/attention.py` | `EuclideanMHA` and `HyperbolicMHA` behind one interface |
| `hmha/moe.py` | shared + top-k routed experts, load-balance and z-loss |
| `hmha/model.py` | the transformer; the arm is one config flag |
| `hmha/data.py` | WikiText-2 download, cache, byte tokenisation |
| `hmha/train.py` | training loop and held-out evaluation |
| `hmha/metrics.py` | bits/byte, bootstrap + Welch, Gromov δ, FLOP estimates |
| `hmha/bench.py` | scaling and latency microbenchmarks |
| `hmha/experiment.py` | driver: arms × seeds, ablations, benchmark |
| `hmha/report.py` | renders `results/report.md` and plots |
| `tests/` | correctness gate (manifold, attention, MoE, model, metrics) |

## Reading the results honestly

`results/report.md` ends with a section on what the experiment **cannot** tell
you. It is not boilerplate. At ~3.6M parameters, byte-level tokenisation and a
256-byte context, this setup is far below the scale where geometry arguments are
usually made, and byte-level modelling actively suppresses the syntactic hierarchy
that hyperbolic space is meant to accommodate. A null result here is a genuine
result about this regime and no more than that.
