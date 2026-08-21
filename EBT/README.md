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
| `associative_recall` | key-value pairs hidden among noise, query names a key → its value | content-addressed retrieval; a delta-shaped attention row is optimal → favours sparsity |
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

## What came out of it

Full numbers in [`results/REPORT.md`](results/REPORT.md); raw runs in
`results/results.json`. Setup: d_model 64, 4 heads, 2 layers, N=64, 1500 steps,
2 seeds, capacity ratio 0.25 (k=16 of 64). Chance is 0.125 on all three tasks.

| variant | associative_recall | needle | majority |
|---|---|---|---|
| baseline-softmax | 0.320 | **0.357** | **1.000** |
| dense-sparsemax | **0.340** | 0.314 | **1.000** |
| mosa-softmax | 0.123 | 0.129 | 0.316 |
| mosa-sparsemax | 0.121 | 0.111 | 0.223 |
| smaxroute-softmax | 0.189 | 0.129 | 0.188 |
| smaxroute-sparsemax | 0.115 | 0.182 | 0.191 |

**1. Sparsemax rows are free quality-wise, and they really are sparse.** In the
dense setting sparsemax matches softmax everywhere (0.340 vs 0.320 on recall,
0.314 vs 0.357 on needle, both perfect on majority) while zeroing **79% of the
attention weights on average and 95% on `needle`** — 13 non-zero weights per
query row instead of 64. That is the micro-tier claim, and it holds: you get a
genuinely sparse attention graph without paying for it in accuracy.

**2. But sparsemax costs wall clock, not saves it.** Dense sparsemax is
**2.7x slower** than softmax at N=64 and 2.5x at N=512 (498 ms vs 199 ms
forward), because the threshold search is a sort. Exact zeros are a
*structural* property here, not a speed win — nothing downstream exploits them
unless you write a kernel that does.

**3. Macro-routing is where the speed actually is.** At N=512 top-k routing is
**13x faster** forward (14.9 ms vs 199.3 ms) and holds 4.2 MB of attention
matrices instead of 67 MB, and the gap widens with N exactly as the k²/N²
argument predicts.

**4. At this scale, routing destroys the tasks.** Every routed variant sits at
or barely above chance on both retrieval tasks, and drops from 1.000 to
0.19–0.32 on `majority`. Two distinct causes, both visible in the diagnostics:
on `majority` routing is *provably* lossy — heads cover only 33–66% of tokens,
and you cannot count a majority over tokens you never look at; on the retrieval
tasks the router has to learn *which* tokens matter at the same time as the
attention learns what to do with them, and 1500 steps is not enough for that
chicken-and-egg to resolve. Stacking sparsemax on top of routing
(`mosa-sparsemax`) never beat routing alone.

**5. The differentiability argument did not survive contact.** The claim is
that top-k gives unselected tokens zero gradient while sparsemax routing is
"fully differentiable". Measured: top-k routing puts non-zero gradient on
**1.5%** of router logits, sparsemax routing on **0.2–2.2%** — the same order.
The reason is structural: sparsemax's own Jacobian is zero outside its support,
so a token with exact-zero routing probability gets exactly zero gradient too.
Sparsemax routing moves *where* the cut-off is, it does not remove it.

**6. What sparsemax routing does deliver is a learned k.** Support size comes
out at 6.8–8.4 tokens per head with a std of 3.3–4.0 *across heads and
sequences* — the model genuinely allocates different budgets to different
heads instead of a hardcoded 16, and it does so while using less than half the
capacity top-k was given. That part of the proposal works; it just did not pay
off in accuracy here.

**Bottom line for the idea as pitched.** The two tiers do not compose the way
the pitch assumes. Sparsemax is the cheap, safe half — free exact zeros at
equal accuracy, at a wall-clock cost. MoSA is the fast half — big, scalable
savings — but it is the half that has to *earn* its routing, and on tasks small
enough to train on a CPU it never gets there. The honest next experiment is a
longer budget on the retrieval tasks (does routed accuracy ever catch up?) and
a router warm-start (dense for the first N steps, then route), which would
separate "routing is wrong" from "routing is slow to learn".

### Caveats on these numbers
2 seeds, 1500 steps, a 2-layer 30k-param model on CPU. The baseline itself only
reaches ~0.34 on the retrieval tasks, so those columns compare *rates of early
learning*, not converged accuracy. Every routed variant might look different
with a longer schedule; nothing here says MoSA is a bad idea at scale, only
that it is not free at small scale.

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
