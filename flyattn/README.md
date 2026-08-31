# flyattn — connectome geometry as an attention prior

Standalone research project living inside the XCAP repository. It shares nothing
with the XCAP pipeline; treat `flyattn/` as its own tree.

The question: does the geometric structure of the *Drosophila* connectome buy
anything as an inductive bias for attention, and if so, which part of it?

## Pre-registered nulls

Most of this is expected to come back negative, and the value is in knowing
which part fails and why:

* conn2res found nothing for the fly (p = 0.11) where mouse, rat and macaque
  passed at p < 0.002.
* Frankle et al. found mask position irrelevant at initialisation.
* The ESA structural advantage lives in one corner of hyperparameter space
  (spectral radius ~0.99, gone at 0.25, reversed under strong regularisation).

House rules, applied everywhere:

1. The baseline is a **degree-preserving rewiring or a hot-limit S¹/H² sample**,
   never Erdős–Rényi, never "random".
2. Report **sample efficiency, hard-subset and OOD performance**, not
   in-distribution loss.
3. Report **interactions** with criticality and data scale, not main effects.

## Data

* **FlyWire v783** (Dorkenwald et al. 2024; Schlegel et al. 2024) from the public
  Codex snapshot. Edges aggregated over neuropils, ≥5 synapses:
  134,181 neurons / 2,700,513 directed edges, ⟨k⟩ = 37.4, undirected density
  0.0279 %.
* **Text**: NLTK Gutenberg for train/val; three OOD sets at increasing distance
  (held-out Gutenberg books → Brown → Reuters newswire). Byte-level tokenisation,
  so OOD numbers are not tokeniser artefacts.

## Layout

```
src/flyattn/
  connectome.py   FlyWire loading, thresholding, graph construction
  nulls.py        degree-preserving rewiring (directed and undirected)
  s1h2.py         S¹/H² sampling, hidden-degree calibration, β and γ fitting
  curvature.py    Balanced Forman curvature and SDRF
  masks.py        attention masks from S¹/H², configuration model, windows,
                  symmetric vs asymmetric global tokens
  model.py        the MoE-MHA baseline and its switches
  train.py        shared training/eval harness (OOD + hard-subset + curves)
  textdata.py     corpora and the hard-position definition
  synth.py        MQAR and gap-recall
scripts/          one script per experiment; results land in results/*.json
```

Run anything with the project venv, e.g.
`python scripts/m1_temperature.py`.
