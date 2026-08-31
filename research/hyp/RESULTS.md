# Results

Model: 6 layers, d=192, 6-head MHA, 8 experts top-2, 8.8M params (2.65M active
per token), trained on GSM8K chain-of-thought to val ppl **10.81**. Dense
control with matched active parameters reaches ppl 10.5. Probes run on 5000
held-out answer tokens with ground-truth reasoning-step depth.

Four conditions throughout: **moe** (trained), **dense** (trained control),
**shuf** (same trained model, word order destroyed), **init** (same
architecture, untrained).

---

## The headline

On this model, **negative curvature buys nothing anywhere it was tested**, and
where a curved and a flat version of the same mechanism could be compared
directly, the flat one won. The one thing that clearly matters is the softmax
normalisation — which is the *lifting* side of the duality, not the curving one.

---

## P4 — curving vs lifting, head to head (the central test)

Held-out R² on the residual **update** `f(h)`, not on `h + f(h)`. `euclid` is
exactly the `R_max -> 0` limit of the hyperbolic chart, so this is nested.

| layer | euclid | hyp best | gain | ±sd | R\* | rff1024 |
|---|---|---|---|---|---|---|
| 0 | 0.783 | 0.783 | +0.0000 | 0.0000 | 0.00 | 0.809 |
| 1 | 0.438 | 0.438 | +0.0000 | 0.0000 | 0.00 | 0.413 |
| 2 | 0.426 | 0.426 | +0.0000 | 0.0000 | 0.00 | 0.393 |
| 3 | 0.376 | 0.376 | +0.0000 | 0.0000 | 0.67 | 0.354 |
| 4 | 0.494 | 0.494 | +0.0000 | 0.0000 | 0.00 | 0.511 |
| 5 | 0.514 | 0.514 | +0.0000 | 0.0000 | 0.00 | 0.508 |

Mean gain over Euclidean — **curving +0.0000, lifting (rff1024) −0.0070**.
The optimal curvature radius is 0 at almost every layer, in all four
conditions. Neither conjugation linearises the layer map better than plain
linear regression in the ambient space.

## P3 — Koopman in depth

| | L0 | L1 | L2 | L3 | L4 | L5 | global |
|---|---|---|---|---|---|---|---|
| next-state R² | 0.788 | 0.908 | 0.928 | 0.871 | 0.895 | 0.921 | — |
| **update R²** | 0.786 | 0.434 | 0.425 | 0.374 | 0.493 | 0.507 | **0.220** |

Per-layer maps are moderately linear, but a *single* operator applied at every
depth captures only 0.22 of the update. The stack is not one Koopman operator
iterated in depth — each layer is a materially different map.

## P1b — is the residual stream tree-like?

Hyperbolic advantage relative to a covariance-matched null.
Calibration: tree-metric cloud **3.95**, isotropic Gaussian **1.23**, matched
Gaussian **1.05**.

| condition | mean over layers |
|---|---|
| moe (trained) | **1.08** |
| dense (trained) | 1.28 |
| moe, shuffled text | 1.01 |
| moe, untrained | 1.03 |

Indistinguishable from a Gaussian carrying the same covariance. Training moves
it no further from the null than shuffling the input does.

## P2 — does reasoning depth live in the radius?

Partial correlation of `||h − mu||` with solution-step depth, controlling for
token position: **−0.01 to 0.13** across layers (dense: −0.06 to 0.03).
No radial depth code.

## P5 — hyperbolic energy vs model confidence

`corr(−E, logp)` with and without the `(n−1)log(sinh r / r)` volume term:

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| E_hyp | 0.05 | −0.07 | −0.08 | −0.07 | −0.10 | −0.10 | −0.06 |
| E_euc | 0.05 | −0.07 | −0.08 | −0.06 | −0.10 | −0.10 | −0.06 |

The volume term — the whole source of the exponential-capacity argument — adds
nothing to two decimal places at every layer. (Curiously the correlation is
*stronger* on shuffled text, +0.31 to +0.39, which suggests what it tracks is
token frequency, not reasoning state.)

## P7 — the `d ≈ 2r` sibling prediction

Ratio of cross-branch-distance slope to radius slope, per layer:
`4.96, −1.59, −0.70, −0.52, −0.48, −46.83, 2.14`. No stable value, and nothing
near the predicted 2.0. Not supported.

## P6 — MoE routing: a non-result

Angle-only probe predicts the top-1 expert at **0.93–0.99**; radius-only at
0.37–0.53 with matched probe capacity. This looks like strong confirmation that
experts partition angular sectors.

**It is an architectural identity, not a finding.** The router reads
`RMSNorm(x)`, so its input is already scale-normalised. Rescaling the residual
stream by 0.01x, 0.5x, 2x and 50x leaves the top-1 expert **100% unchanged**.
The radius cannot carry routing information in this architecture, and the
angular probe is measuring the definition of the router. Consistent with that,
the angular probe already reaches 0.72–0.83 on the *untrained* model.

Any claim of the form "MoE experts occupy angular sectors, matching hyperbolic
branch structure" needs this control before it means anything.

---

## P8 — attention: which kernel is softmax actually close to?

Surrogates swapped into the trained model, each with its free scalar tuned on
held-out data. No retraining.

| kernel | mechanism | val loss | Δ |
|---|---|---|---|
| softmax MHA (trained) | — | 2.3807 | — |
| softmax + temperature | — | 2.3807 | +0.000 |
| **Euclidean distance kernel** | flat, keeps softmax | **3.0608** | **+0.680** |
| **hyperbolic distance kernel** | curved, keeps softmax | **3.2336** | **+0.853** |
| linear RFF D=64 | finite lift | 3.7271 | +1.346 |
| linear RFF D=256 | finite lift | 3.7613 | +1.381 |
| linear elu | finite lift | 3.8287 | +1.448 |

Three things, in order of size:

1. **The softmax is what matters.** Kernels that keep the normalisation lose
   ~0.7 nats; kernels that discard it lose ~1.4. That gap is twice as large as
   anything else measured here.
2. **Curvature actively hurts.** Hyperbolic is 0.17 nats *worse* than the flat
   distance kernel it should generalise. The optimum was interior to a wide
   grid (β up to 128, scale up to 0.95), so this is not a tuning artefact.
3. **The finite Koopman truncation is the worst option** — exactly what the
   theory predicts, since a rank-D dictionary cannot reproduce the
   infinite-dimensional feature map that `exp(q·k)` is.

An earlier run on a mid-training checkpoint had hyperbolic *beating* the flat
control; on the converged model the ordering reverses. The flat-distance control
is what makes either reading possible — without it, the hyperbolic-vs-linear gap
alone reads as strong evidence for hyperbolic attention, and it is not.

---

## What this does and does not show

**Does not test the mathematics.** Koopman lifting and hyperbolic charts really
are both conjugations `f = phi^-1 . L . phi`; nothing here bears on that.

**Does test the empirical claim** that transformer activations on reasoning text
carry hyperbolic structure that a curved chart or a curved energy can exploit.
On this model: no support, in seven independent probes, with the controls that
would have caught a false positive.

**Limits, in order of how much they should worry you:**

- **Scale.** 8.8M parameters, d=192, 6 layers, 1.09M training tokens, one narrow
  domain. Structure that only appears in a 4096-dim stream trained on trillions
  of tokens would be invisible here.
- **Reasoning depth.** GSM8K solutions have a median of 4 steps and a maximum of
  10. The exponential-branching regime the hyperbolic argument is really about
  needs far deeper search than this data contains.
- **Inductive bias.** This model was trained with Euclidean, softmax-normalised
  machinery. Asking whether its activations are hyperbolic asks whether a flat
  architecture spontaneously curves. A model *trained* with hyperbolic layers is
  a different experiment, and this says nothing about it.
- **Substitution ≠ architecture.** P8 measures how close each kernel is to the
  *trained softmax solution*. It does not say linear-attention architectures
  fail — those are trained from scratch and do far better than a swapped-in
  surrogate.

**The negative methodological results travel further than the model does.**
δ-hyperbolicity failing to separate WordNet from a Gaussian, low-dimensional
embedding advantage favouring a Gaussian over a tree, and RMSNorm making the
MoE angular-routing result a tautology are all properties of the instruments,
not of this 8.8M-parameter model. Any hyperbolic-representation claim resting
on them needs the matched-null and scale-invariance controls first.
