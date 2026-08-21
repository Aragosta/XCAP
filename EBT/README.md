# EBT — does sparsemax attention beat softmax attention?

A small, self-contained testbed for one question:

> If you replace softmax with **sparsemax** — the Euclidean projection onto the
> probability simplex — inside attention, what do you gain and what do you pay?

Everything is deliberately tiny (a 2-layer encoder, ~30k params, synthetic
tasks, CPU-only) so the whole comparison runs end to end in minutes and every
claim in the report is reproducible from one command.

*(This lives inside the XCAP repo for convenience only; it shares no code with
the rest of it.)*

## The mechanism

Softmax gives every key a strictly positive weight, however irrelevant it is: a
64-token row always has 64 non-zero weights. Sparsemax solves

    sparsemax(z) = argmin_{p in simplex} ||p - z||^2

whose solution is `max(z - tau(z), 0)` for a data-dependent threshold `tau`.
Coordinates below the threshold are **exactly zero**, so the attention row
becomes a genuinely sparse graph rather than a dense one with small numbers in
it. The support size is not a hyperparameter — it falls out of the score
distribution, per row.

The two variants under test are `baseline-softmax` and `sparsemax`. They share
projections, head layout, MLP, initialisation and a learnable per-head
temperature, so parameter counts and FLOPs are *identical* and the only
difference is the normaliser. (The temperature is deliberately learnable and
given to both: sparsemax is not scale invariant, so a hardcoded `1/sqrt(d)`
would silently decide how sparse it is allowed to be.)

## The tasks

Three probes chosen so the comparison cannot be won by one inductive bias:

| task | what it needs | what it rewards |
|---|---|---|
| `associative_recall` | key-value pairs hidden among noise, query names a key → its value | content-addressed retrieval; a delta-shaped attention row is optimal → rewards sparsity |
| `needle` | noise everywhere, a few TAG-value pairs, a query naming one tag | ~90% of tokens are irrelevant → rewards ignoring them |
| `majority` | the most frequent symbol in the whole sequence | every token counts → rewards dense, high-entropy attention; the honest counter-example |

Each is per-position classification with a loss mask on the final position, so
one training loop covers all three. `tests/test_tasks.py` checks the labels are
correct, not predictable from a constant, and (for `majority`) that they really
require the full sequence.

## What is measured

* **Quality** — final/best eval accuracy and loss, plus steps-to-90% as a
  sample-efficiency proxy.
* **Mechanism** — fraction of *exactly zero* attention weights, non-zero
  weights per query row, row entropy, largest single weight.
* **Cost** — wall-clock forward and forward+backward ms/batch, analytic
  FLOPs/sequence, attention-matrix bytes, and a sweep of all of that against
  sequence length.
* **Stability** — mean and std of the gradient norm over training.

## What came out of it

`results/` starts empty — running the two commands below regenerates
`results/results.json` and `results/REPORT.md`. The numbers quoted here come
from a 2-seed run at d_model 64, N=64, 1500 steps (raw records preserved in git
history at commit `1bf6bc2`). Chance accuracy is 0.125 on all three tasks.

**1. Sparsemax matches softmax on accuracy.** Across all three tasks the two are
within noise of each other, including a perfect 1.000 for both on `majority`.
Whatever sparsemax throws away, it is not signal the model needed.

**2. And it really is sparse — adaptively so.** Non-zero weights per query row,
out of 64:

| task | softmax | sparsemax |
|---|---|---|
| associative_recall | 64 | **3.4** |
| needle | 64 | **2.9** |
| majority | 64 | **34.1** |

That last row is the interesting one. Nothing tells the model which task it is
on, yet it goes near-one-hot for retrieval and stays broad for counting.
Softmax cannot express either endpoint: it can approximate the first and never
reaches exact zero. This is the real result — a normaliser that picks its own
support size per row, per task, for free.

**3. The bill is wall clock, not accuracy.** Sparsemax is ~2.5–2.7x slower,
because finding `tau` is a sort. FLOPs and memory are identical by construction.
So the exact zeros are a *structural* property, not a speed win: nothing
downstream exploits them unless you write a kernel that skips them.

**When it is worth it:** you want the sparse attention graph itself —
interpretability, attention-based pruning or routing decisions, extracting a
discrete structure from a trained model. **When it is not:** you only want
throughput. For a cheaper middle ground, α-entmax with α=1.5 has a closed-form
solution and needs no sort.

### Caveats
2 seeds, 1500 steps, a 2-layer 30k-param model on CPU. The baseline itself only
reaches ~0.34 on the two retrieval tasks, so those columns compare *rates of
early learning*, not converged accuracy; `majority` is the only task both
variants solve outright.

### What was dropped
This bench originally also tested MoSA-style expert-choice routing (hard top-k
per head) and sparsemax-as-a-router, in a 2x3 grid. Both mechanisms have been
removed from the code. They were dropped because every routed variant sat at or
near chance on all three tasks, and because the sparsemax router's claimed
advantage — differentiable routing — did not survive measurement: non-zero
router gradient reached 0.2–2.2% of logits versus 1.5% for hard top-k, since
sparsemax's Jacobian is also zero outside its support. That code and its
results are in git history at commit `1bf6bc2` if they are ever wanted back.

## Running it

```bash
pip install -r requirements.txt
python -m pytest -q                       # 72 tests, seconds
python experiments/run_benchmark.py --steps 1500 --seeds 2 --seq-len 64 --d-model 64 --lr 3e-3
python experiments/run_scaling.py
python experiments/report.py              # writes results/REPORT.md + learning_curves.png
```

## Layout

```
ebt/sparsemax.py    sparsemax as an autograd.Function (+ masked softmax)
ebt/attention.py    the attention module and its diagnostics
ebt/variants.py     the two named configurations
ebt/tasks.py        the three synthetic probes
ebt/model.py        tiny encoder transformer + FLOP/param accounting
ebt/metrics.py      eval and speed benchmark
ebt/train.py        training loop and the single-run experiment record
experiments/        benchmark, scaling sweep, report generator
tests/              unit tests for the normaliser, the attention and the tasks
```
