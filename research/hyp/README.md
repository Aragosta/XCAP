# Hyperbolic geometry vs Koopman lifting in a MoE MHA transformer

Testing, on real text, whether the residual stream and attention of a
mixture-of-experts multi-head-attention transformer are better described by
**negative curvature** (hyperbolic embeddings) or by **Koopman/kernel lifting**
(linear attention, EDMD).

Both are the same algebraic move — a conjugation `f = phi^-1 . L . phi` that
turns a nonlinear map into a linear one — but they pay for it differently.
Lifting buys linearity with extra dimensions; curving buys capacity with
curvature at fixed dimension. That difference is measurable.

## Setup

No GPU and no pretrained MoE was reachable from this environment, so the model
is trained from scratch:

| | |
|---|---|
| data | GSM8K chain-of-thought traces (`openai/grade-school-math`), calculator annotations stripped |
| tokenizer | word/digit level, 4096 types, newline is its own token (marks solution steps) |
| model | 6 layers, d=192, **6-head MHA**, **8 experts top-2**, 8.8M params / 2.65M active per token |
| control | dense twin with **matched active parameters** (2.65M), same data, same schedule |
| depth labels | step index within the solution, from the newline tokens — ground truth, not position |

`python train.py` / `python train.py --dense`, then `./run_all.sh`.

## Probes

| | question | file |
|---|---|---|
| P1b | is the residual stream tree-like beyond its own covariance? | `distortion.py` |
| P2 | does reasoning depth live in the radius? | `probes.py` |
| P3 | how linear is the layer-to-layer map (Koopman in depth)? | `probes.py` |
| P4 | **curving vs lifting, head to head** | `probes.py` |
| P5 | does a hyperbolic energy track model confidence? | `probes.py` |
| P6 | does the MoE router partition angle or radius? | `probes.py` |
| P7 | is cross-branch distance ~ 2r? | `probes.py` |
| P8 | **which kernel does trained attention actually behave like?** | `attention_kernel.py` |

## Two instruments that had to be thrown away

Recorded here because they are the standard ones, and because a lot of
"LLM representations are hyperbolic" evidence rests on them.

1. **delta-hyperbolicity does not work at these scales.** Measured on the
   WordNet noun hypernym graph — the canonical hierarchy — `delta_rel` is
   `0.030` (sampled mean) or `0.32` (true max). A Gaussian in R^192 gives
   `0.034` and `0.25`. The real tree scores *worse* than the Gaussian on the
   max statistic. It does not separate hierarchy from noise in either
   direction. (`wordnet_control.py`)

2. **"embeds better in H^k than R^k" does not work either.** A tree-metric
   cloud gets a 3.3-3.5x hyperbolic advantage at k=4-8 — but an isotropic
   Gaussian in R^192 gets **12-14x**. Negative curvature helps any cloud whose
   distances are too spread for a flat low-dimensional fit, which includes
   structureless high-dimensional noise. (`distortion.py`)

The fix for both is the same: score each cloud against a null carrying its own
mean and covariance, so only structure beyond second order registers.
Calibrated in `validate_instrument.py`:

```
cloud               adv(data)  adv(null)   RATIO
tree metric MDS         10.27       2.60    3.95     <- real hierarchy
isotropic gaussian      13.97      11.39    1.23     <- no structure
gaussian w/ tree spectrum 2.61      2.50    1.05     <- no structure
```

## Three corrections that changed the answer

- **The ridge must be scale-invariant.** Each chart in the curvature sweep
  rescales its coordinates; a fixed ridge then regularises the charts by
  different amounts and the sweep measures regularisation, not geometry.
  Fixed by scaling the penalty per column of the Gram matrix.
- **Curvature is not the free parameter.** Rescaling a fixed cloud to fill the
  ball of curvature `c` makes the chart *identical* for every `c` — curvature
  and fill fraction are the same knob. The sweep is over `R_max`, how many
  curvature radii the cloud spans, at `c = 1`. `R_max -> 0` is exactly the
  Euclidean chart, so the comparison is properly nested.
- **Score the residual update, not the next state.** A residual block is
  `h + f(h)` with `||f|| << ||h||`, so R^2 against `h_{l+1}` is dominated by the
  identity and every chart scores ~0.99. Scoring `f(h)` is the only version
  with discriminating power.

Results and interpretation: `RESULTS.md`.
