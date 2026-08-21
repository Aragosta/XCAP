# EBT — a bench for sparse attention ideas

A small, self-contained testbed for one question:

> Does replacing softmax with **sparsemax** inside attention, and/or replacing
> "every head sees every token" with **MoSA-style expert-choice routing**,
> actually buy anything over a plain softmax attention baseline?

Everything here is deliberately tiny (a 2-layer encoder, ~100k params, synthetic
tasks, CPU-only) so that the whole grid runs end to end in under an hour and
every claim in the report is reproducible from one command.

*(This lives inside the XCAP repo for convenience only; it shares no code with
the rest of it.)*

## The mechanisms

Two orthogonal choices, six combinations, one shared implementation:

|                   | softmax rows        | sparsemax rows          |
|-------------------|---------------------|-------------------------|
| **dense** (no routing)      | `baseline-softmax` (control) | `dense-sparsemax` |
| **top-k routing** (MoSA)    | `mosa-softmax`               | `mosa-sparsemax` ← the two-tier proposal |
| **sparsemax routing**       | `smaxroute-softmax`          | `smaxroute-sparsemax` ← fully differentiable |

**Tier 1 — macro routing.** Each head owns a router `W_r: R^d -> R` and scores
every token. *Top-k (MoSA)*: the head hard-selects its own `k = ratio * N`
tokens, gathers them into a contiguous `k x d` block, attends inside the block,
and scatters the result back; unselected positions get exactly zero from that
head. Head outputs are gated by `sigmoid(router score)` so the router receives
gradient. *Sparsemax routing*: the router scores go through sparsemax, tokens
with exact-zero probability are dropped, and the survivors are gathered and
gated by their (normalised) routing weight — so **the number of tokens a head
takes is learned per sequence and per head** instead of hardcoded.

**Tier 2 — micro sparsity.** Inside the selected block, attention rows are
normalised with either softmax (never exactly zero) or sparsemax (the Euclidean
projection onto the simplex, which produces exact zeros and a genuinely sparse
attention graph).

All six variants share the same projections, head layout, MLP, initialisation
and a learnable per-head temperature, so the parameter counts match to within
0.5% and the only difference is the mechanism. (Temperature matters: sparsemax
is *not* scale invariant, so a fixed `1/sqrt(d)` scale would silently decide how
sparse it is allowed to be. Giving every variant a learnable temperature removes
that confound.)

## The tasks

Three probes chosen so the grid cannot be won by one inductive bias:

| task | what it needs | who should win |
|---|---|---|
| `associative_recall` | `[k1 v1 k2 v2 ... QUERY kq]` → the value bound to `kq` | pure retrieval; a delta-shaped attention row is optimal → favours sparsity |
| `needle` | noise everywhere, a few TAG-value pairs, a query naming one tag | ~90% of tokens are irrelevant → favours routing to a small subset |
| `majority` | the most frequent symbol in the whole sequence | every token counts → favours dense, high-entropy attention; the honest counter-example |

Each is per-position classification with a loss mask on the final position, so
one training loop covers all three. `tests/test_tasks.py` checks the labels are
correct, unpredictable from a constant, and (for `majority`) genuinely require
the full sequence.

## What is measured

* **Quality** — final/best eval accuracy and loss, plus steps-to-90% as a
  sample-efficiency proxy.
* **Mechanism** — fraction of *exactly zero* attention weights, non-zero
  weights per query row, attention entropy, token coverage (share of tokens
  picked by at least one head), routed support size and its spread across
  heads/sequences.
* **Differentiability** — `router_grad_frac`, the measured share of router
  logits that receive a non-zero gradient. This turns the usual hand-wave
  ("top-k is non-differentiable") into a number.
* **Cost** — wall-clock forward and forward+backward ms/batch, analytic
  FLOPs/sequence, bytes held by the materialised attention matrices, and a
  separate sweep of all of that against sequence length.
* **Stability** — mean and std of the gradient norm over training.

## Running it

```bash
pip install -r requirements.txt
python -m pytest -q                       # ~90 tests, seconds
python experiments/run_benchmark.py --steps 1000 --seeds 2 --workers 4 --threads 1
python experiments/run_scaling.py
python experiments/report.py              # writes results/REPORT.md + learning_curves.png
```

`results/REPORT.md` is the generated write-up; `results/results.json` holds
every raw run record including the full eval history.

## Layout

```
ebt/sparsemax.py    sparsemax as an autograd.Function (+ masked softmax)
ebt/attention.py    the 2x3 attention grid and its diagnostics
ebt/variants.py     the six named configurations
ebt/tasks.py        the three synthetic probes
ebt/model.py        tiny encoder transformer + FLOP/param accounting
ebt/metrics.py      eval, speed benchmark, router-gradient coverage
ebt/train.py        training loop and the single-run experiment record
experiments/        benchmark, scaling sweep, report generator
tests/              unit tests for every mechanism and task
```

## Caveats, stated up front

* **Expert-choice routing is not causal.** Which tokens a head selects depends
  on the whole sequence, so it leaks future information in an autoregressive
  setting (this is inherent to expert choice, not an implementation bug). The
  benchmark tasks are therefore bidirectional/encoder-style. The `causal` flag
  masks attention *within* a block by original position and is exercised by
  the tests, but it does not make the routing decision causal.
* **Wall-clock is CPU wall-clock at small N.** Top-k routing does a real gather
  into a `k x k` block, so its speedup is real, but at these sizes the
  projections dominate and none of this reflects a fused GPU kernel. The FLOP
  and attention-memory columns are the scale-free part of the cost story.
* **The sparsemax router's block width is `max` support across the batch**,
  padded with dead slots, because ragged supports would need a custom kernel.
  The maths is identical to dropping the zero-probability tokens; the wall
  clock is pessimistic relative to what a ragged kernel would give.
* Small model, synthetic tasks, 2 seeds. Directional evidence, not a paper.
