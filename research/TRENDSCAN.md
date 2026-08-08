# Trend scanning — XCAP v0

The single record for `research/trendscan.py`. Supersedes and replaces
`RESULTS_trendscan_vs_heuristic.md`, `SYNTHESIS_breadth_not_signal.md` and `decisions.py`,
whose durable content is folded in below.

> ## Status, 2026-08-08 — cut back to the signal, and only the signal
>
> `trendscan.py` now emits exactly one thing per rung: `tb_L`, the raw slope t of a
> centred OLS fit on log adjusted price. Everything else that used to come out of the
> build is gone from it: the winsorised path (`wb`), the sign/runs path (`sb`), the scale
> column (`sd`), the forward targets, the ADV column and the skip reduction. Bar
> integrity (§2.1) stays — it is data rejection, not a signal choice.
>
> **The scoring harness is also cut.** There is no `trendscan_eval.py` in the tree: no IC
> tables, no neutralisation, no deciles, no costed book. It exists in history at
> `git show 9810fa7:research/trendscan_eval.py` and should be recovered from there, not
> rewritten — that file is the only reason §3's numbers were reproducible at all. Nothing
> below this note can currently be re-run.
>
> Winsorisation is not disowned, it is not being done *here*. §3.5 and §3.13 stand as the
> record of what it measured; a return-preprocessing path is a decision, and decisions
> belong downstream of a signal file, not baked into it.
>
> **Consequence, stated plainly:** the status block below claims the signal is
> `sb_top`/`wb_top`, and nothing in the tree produces those columns any more. §3.13's
> central result — that `tb` is fully explained away by cheap controls while the robust
> paths are not — is unaffected as a *finding*, but it is now a finding about a signal
> this repo no longer computes. Re-deriving from a clean `tb` base is the deliberate
> choice; treat everything from here as needing re-measurement.
>
> Targets, liquidity screen, the skip, κ-rescaling, ranking, deciles and costs are all
> things you do *with* a t, after the fact. None of them belong in the build, and for now
> none of them are anywhere.

> ## Status, 2026-08-07 — there is a real signal, and it is `sb_top` / `wb_top`
>
> Re-measured end to end on the post-bar-integrity panel, with the harness now living
> inside `trendscan.py`. The deliverable at this stage is a **signal**, not a strategy, and
> §3.13 is the section that answers that; §3.12's backtest is reported but does **not** gate
> it.
>
> 1. **The scan carries information no cheap baseline has** (§3.13). Projecting out
>    vol-normalised 12−1, low-vol and size *jointly*, `wb_top` retains 25–50% of its IC and
>    `sb_top` 29–43%, both clearing t=2 at **every** horizon (t up to 8.5).
> 2. **`tb_top` — the plain OLS t-stat — does not.** Against the same joint control it
>    retains 1% at h252 (t 0.10). Its apparent strength is reconstructible from things you
>    can compute more cheaply. **The robust input paths are the signal; the raw one is a
>    proxy.** Raw IC could not see this — all three look like ties (§3) because they share a
>    large common component. Only the orthogonal part discriminates.
> 3. **It is a long-horizon signal and the IC never peaks** (§3.13). `wb_top` runs
>    0.0172 → 0.0518 from h1 to h252, still climbing at the edge of the grid.
> 4. **The payoff is 2.2–3.4× larger on the short side** (§3.13). Deciles are monotone
>    (9/9 rising at h63 and h252), but D1 sits ~763bp *below* the cross-sectional mean at
>    h252 while D10 sits only ~287bp above.
>
> That last point explains §3.12: the long-only top-decile book tested there **uses only
> D10 and discards the stronger half of the signal**, so its loss to a flat baseline is a
> statement about that construction, not about the signal.
>
> Also settled: curvature cut entirely (§3.7); the tent exponent is flat under CPCV, so
> `p=1` stands (§3.5).
>
> **Still not a deployment recommendation.** Everything is in-sample on one panel with a
> ~40-trial count, there is no sector or market-cap control available in this data, and no
> long/short book has been costed.

---

## 1. What is fitted

For every **(asset, date `t`, rung `L`)**, one OLS on the trailing window `[t−L+1, t]`:

```
y_i = a + b·u_i + ε_i                 y = log(adjusted close), centred per asset
                                       u_i = i − (L−1)/2,  i = 0…L−1
```

The design is **centred**, so `Σu = Σu³ = 0`. The quadratic term `c·u_i²` was fitted until
2026-08-07 and is gone (§3.7): the centred basis made `b` numerically identical with or
without it (verified Δ = 3.5e-18), so dropping `c` changes no trend arithmetic — it only
removes a second accumulator chain.

Everything uses data ≤ `t`. López de Prado's trend scanning is a forward-looking
*labelling* method; this is the same machinery run backward so it can be traded.

**Speed.** The naive scan is O(T·ΣL). Each window is a rolling OLS updated in O(1) per bar
from four centred sums (`Σy, Σy², Σi·y, Σi²·y`), njit-compiled, parallel over assets, with
an exact recompute every 2048 bars to stop drift. 19M rows × 10 rungs × 3 input paths in
**~3.5 s**; the whole build in **~48 s**, and nothing is written to disk (§2).

Recursions and moment formulas (`Su2 = L(L²−1)/12`) verified
symbolically and against `scipy.stats.linregress` / `numpy.linalg.lstsq` — max |Δ| ~1e-6,
which is float32 storage rounding.

## 2. What it outputs

**Nothing.** `build()` returns an in-memory frame — 18,999,392 rows × 51 columns — and the
surface is **not persisted.** A 4.6 GB parquet cost more to store than the ~48 s it takes
to recompute, and a stale one outlives the harness that validated it. That is not a
micro-optimisation: the previous parquet *did* go stale, and §3's whole findings section
had to be thrown away because of it.

```
GRID = [5, 10, 21, 42, 63, 84, 126, 168, 210, 252]     # one ladder, roughly geometric
```

| vector | is | from which fit |
|---|---|---|
| `tb_5 … tb_252` | `b / se(b)` — slope t | linear (df `L−2`) |
| `sd_5 … sd_252` | `√(SSE/(L−2))` — the fit's own residual vol | linear |
| `wb_5 … wb_252` | the same slope t, on a price path rebuilt from returns **clipped at ±3·MAD(252)** | linear |
| `sb_5 … sb_252` | the same slope t, on `cumsum(sign(Δlog P))` — no prices, no vol | linear |

plus `security_id, date, close, adv21`, `fwd1/5/21/63` (targets, diagnostics only), and
`seg` / `y` / `rw` (segment id, centred log price, winsorised returns) which the harness
and the sweep consume.

`wb` and `sb` are the *same kernel on a different input series* (§3.8), not a different
estimator. On the clean panel **all three tie** — `wb_252` beats `tb_252` only at h5
(§3.5), and `sb_252` ties it everywhere while assuming nothing about the distribution.
`sb` exists as the baseline any future trend column must beat; §4's recorded failure mode
was killing a baseline without ever measuring it.

**One grid, no fast/slow.** A 5-day rung is just a rung (§3.3). Roughly geometric because
adjacent rungs are near-redundant — 117 daily-spaced windows measured N_eff 2.14, so fine
spacing buys columns, not information, and a dense grid measures worth exactly zero (§3.2).

**The build emits the surface and nothing else.** No skip, no argmax, no ladder mean, no
composite. Those are *reductions*, applied downstream against stored columns —
`skip()`, `argmax_abs()`, `rowmean()`, `rung()`. The surface is raw: `tb_L[t]` is the
window *ending at t*, and `skipped[t] == raw[t−k]`, so the 12−1 construction is one shift.

That is the load-bearing design choice. Every error found in the review was a reduction
baked into the build where it could not be questioned.

### 2.1 Bar integrity — the build rejects prices before it fits anything

Added after `LOSS.md` §9.7 found that the panel these results are computed on is corrupt.
Two vendor defects, and neither is visible to the existing `phase1_checks` gate:

| defect | example | treatment |
|---|---|---|
| **isolated foreign print** — a whole OHLC bar from another instrument, unwinding on the next bar | 270686, 2015-09-18: 519.17 → 14,199.60 → 519.17, volume 421,071 vs a normal 500 | drop the bars (4,030, across 123 securities) |
| **level break** — missed corporate action, or a spliced/recycled ticker | 283067, 2004-07-30: 8.99 → 74.30 overnight, `split_factor` 1.0, dollar volume continuous through it. 258690 alternates a $0.005 stub at volume 0 with a real $4,700 name at volume 250k | **cut the series** (785 cuts) |

Three things make this belong in the build and nowhere else:

* a print corrupts **every window that contains it** — up to 252 bars of `tb`/`tc`/`sd`
  for one security — and both legs of `fwd{h}`. There is no downstream repair.
* the bar is internally consistent, so `low ≤ {open,close} ≤ high` passes it, and
  `split_factor` is 1.0, so the corporate-action checks pass it too.
* **the liquidity screen selects for it.** The spike inflates `adv21` for 21 bars and
  pulls the name *into* the top-1000-by-ADV universe. Screening harder makes it worse.

Detection is on **adjusted** price and by **reversal**, which is what makes it split-safe:
a real split is already in `adj` so it never appears, and a missed split does not unwind.
Level breaks **cut** rather than drop — the data either side is fine, only its continuity
is not — so no window and no `fwd{h}` spans a break, and the universe is preserved
(4,505 securities reach the screened panel, against 4,506 before).

**No winsorisation of prices, returns, or P&L — here or downstream.** Anything that
survives the cleaning is treated as real return, and every IC and every backtest in this
document is measured on unclipped outcomes.

That policy is about honesty of *outcomes*, and it does **not** extend to a signal's own
*input*. `wb_L` scans a path built from returns clipped at ±3·MAD(252). The clipping is
inside the filter; nothing it touches is ever counted as a return.

Note the interaction, because it is the most interesting thing this section produced:
**once bar integrity landed, winsorisation's measured edge largely disappeared** (§3.5).
The clipper had been doing the cleaner's job. Two fixes for one defect look like two
independent wins until you apply both.

**Windows longer than an asset's history are skipped** rather than voiding the row.
Consequence: any reduction that averages across rungs averages a *varying* number of them,
and since |t| ∝ √L its scale tracks listing age — mean |trend| runs 4.12 (2 rungs) →
10.53 (12 rungs), monotone, with the cross-sectional rank compressed toward 0.5. Moot for
a single-rung reduction, which simply has no value until the rung fits (~5% of screened
rows). Anything that must make rungs commensurable divides by **√L** (§3.8).

## 3. Findings

> **Re-measured 2026-08-07 on the post-§2.1 panel** by `python research/trendscan_eval.py eval`,
> which now ships inside `trendscan.py` rather than in a file that can go missing. The
> tables below replace the pre-bar-integrity ones outright; they are not annotated,
> because the old numbers had already misled once. One headline changed sign of
> importance — see §3.5 on winsorisation.

Sample: 18,999,392 → **17,143,673** rows after a 1-bar lag and the screen (ADV21 ≥ $1M,
close ≥ $5), 6,903 dates, 5,603 securities, 2000–2026.

Convention, and it is stricter than everything that preceded it:

| | before | now |
|---|---|---|
| timing | signal(t) vs return **from t** — free fills at the signal's own close | **1-bar lag**: signal(t) earns t+1 → t+1+h, matching `BACKTEST.py` |
| NW bandwidth | `lag = h`, barely above the induced MA(h−1) | `⌈1.5h⌉` |

Both push t **down**. Reported ICs are cross-sectional Spearman, Newey-West t.

### The reductions

Every reduction is `_top` = the 252 rung with a 21-bar skip. Curvature is gone (§3.7);
argmax and the ladder mean are not re-listed because §3.1/§3.2 killed both twice.

| signal | IC_h1 | t | IC_h5 | t | IC_h21 | t | IC_h63 | t |
|---|---|---|---|---|---|---|---|---|
| `tb_top` incumbent | 0.0174 | 6.72 | 0.0212 | 5.15 | 0.0241 | 3.33 | 0.0332 | 2.80 |
| **`wb_top`** winsorised | 0.0172 | 6.60 | **0.0243** | **5.94** | **0.0258** | **3.60** | **0.0355** | **3.05** |
| `sb_top` sign-only | 0.0172 | **7.22** | 0.0220 | 6.20 | 0.0213 | 3.55 | 0.0316 | 3.25 |
| `nsd` −`sd_top` | 0.0109 | 3.69 | 0.0093 | 1.89 | 0.0067 | 0.74 | 0.0156 | 1.03 |
| `flat_ret` 12−1, unfitted | 0.0142 | 5.18 | 0.0181 | 4.13 | 0.0161 | 2.14 | 0.0165 | 1.36 |
| `flat_ret_vol` 12−1 / σ252 | 0.0157 | 5.86 | 0.0195 | 4.53 | 0.0203 | 2.65 | 0.0254 | 2.09 |

**Levels are ~0.8 correlated, so the only test that says anything is the paired one**
(NW t of the *difference* vs `tb_top`):

| paired − `tb_top` | h1 | h5 | h21 | h63 |
|---|---|---|---|---|
| `wb_top` | −0.0003 (−0.21) | **+0.0031 (2.29)** | +0.0017 (1.18) | +0.0024 (1.29) |
| `sb_top` | −0.0002 (−0.10) | +0.0008 (0.30) | −0.0027 (−0.82) | −0.0016 (−0.35) |
| `flat_ret` | **−0.0032 (−2.01)** | −0.0030 (−1.54) | **−0.0079 (−2.45)** | **−0.0166 (−3.36)** |
| `flat_ret_vol` | −0.0017 (−1.22) | −0.0017 (−1.09) | −0.0038 (−1.75) | **−0.0077 (−2.84)** |
| `tb_21` | −0.0121 (−3.87) | −0.0154 (−3.64) | −0.0237 (−3.43) | −0.0276 (−2.54) |
| `tb_63` | −0.0079 (−2.77) | −0.0128 (−3.33) | −0.0169 (−2.57) | −0.0175 (−1.70) |
| `tb_126` | −0.0074 (−3.13) | −0.0102 (−3.41) | −0.0091 (−2.14) | −0.0125 (−1.66) |

Three things this settles, all of them on the clean panel:

1. **The top rung wins**, confirming §3.1. Every shorter rung loses at every horizon,
   most of them past t −2.5. Not re-litigated further.
2. **The scan beats a plain return** — `flat_ret` loses at h1/h21/h63 and the gap *widens
   with horizon* (−0.0166, t −3.36 at h63). Against the harder `flat_ret_vol` baseline the
   margin survives only at h63 (t −2.84). So the trend scan's edge over "12−1 divided by
   vol" is a long-horizon effect, and a modest one.
3. **Nothing Gaussian is load-bearing.** `sb_top` — cumsum of *signs*, no prices, no vol,
   no distributional assumption — ties the OLS t-stat at every horizon, |t| ≤ 0.82. The
   regression is a convenient way to compute a tent-weighted average of returns (§3.8),
   not a source of edge.

### 3.1 The ladder mean loses to a single rung

`|t| ∝ √L` (§3.8 — measured `L^0.51`, not the `L^1.5` claimed here before), so a ladder
mean is dominated by its top rung by construction (corr +0.856). The conclusion was right
and the mechanism was wrong: the exponent is the random-walk null exponent, so the ladder
mean is dominated by the top rung even when every rung is pure noise. Paired against it, the single top rung is better at every horizon —
**+0.0046 (t 2.60)** at h1 — at **27% of the turnover**. `trend_lad` also collapses in
2009-16 (h21 IC 0.0014) where `trend_top` holds 0.0121.

The prior claim "ladder wins" was only ever measured against *argmax*, never against "use
the longest rung."

And the scaling artifact is not what was hiding a good ladder. Commensurating the rungs
first — `z_L = tb_L/(κ_b√L)`, so every rung has a unit null (§3.9) — the equal-weight mean
over {21, 63, 126, 252} scores **0.0148 / 0.0190 / 0.0293** at h1/h21/h63 against `wb_252`'s
**0.0161 / 0.0272 / 0.0367**. The ladder still loses with the artifact removed. The short
rungs are simply worse signals.

### 3.2 The endogenous horizon buys nothing, and a dense grid buys nothing

Held to an identical grid and skip, for curvature the **ladder beats the argmax**
(up to t −1.63) — the reverse of the prior claim. Without the skip the two are
indistinguishable (|t| ≤ 1.62). For trend, argmax loses to the single top rung.

A dense 21…252 argmax (232 of the old build's 260 window-passes) against a 12-rung
monthly argmax: **+0.0004 / +0.0006 / +0.0007 / +0.0000**, all |t| ≤ 0.65. Zero. That was
89% of the scan.

The prior comparison was confounded — it varied mechanism *and* grid density *and* skip at
once, so it could never attribute the result.

### 3.3 No fast/slow split

Drop-one-leg on the old 3-leg composite, paired:

| dropped | ΔIC_h1 | t | ΔIC_h21 | t | ΔIC_h63 | t |
|---|---|---|---|---|---|---|
| fast curvature | −0.0008 | −1.17 | **+0.0005** | 0.36 | **+0.0008** | 0.56 |
| slow curvature | +0.0014 | 1.76 | −0.0033 | −1.22 | −0.0017 | −0.45 |
| **trend** | **−0.0064** | **−6.04** | **−0.0096** | **−2.35** | **−0.0136** | **−2.16** |

Only the trend leg is measurably load-bearing. Dropping the fast leg is free at h1/h5 and
mildly *better* at h21/h63, while it carried turnover 0.2698 — 23× the trend leg. On a
fixed grid, "fast ladder beats fast argmax and turns over less" fails on both counts.

Folding the short rungs into **one** grid made curvature *stronger* than the monthly-only
version (h5 0.0126 vs 0.0092, h21 0.0184 vs 0.0163). The content was real; the hardcoded
split was not.

### 3.4 The skip is asymmetric — hardcoding it degraded a leg

| | ΔIC_h21 | t | ΔIC_h63 | t |
|---|---|---|---|---|
| curvature, ladder: no-skip − skip | 0.0050 | 1.47 | **0.0101** | **2.20** |
| curvature, argmax: no-skip − skip | **0.0072** | **2.20** | 0.0087 | 1.97 |

Trend wants the window to end a month early — it removes short-horizon reversal from the
slope. Curvature is *hurt* by it: it measures the bend happening **now**. Window length and
end-lag are two axes of one 2-D surface, which is why the skip must stay out of the build.

### 3.5 The tent is insensitive, and winsorisation was being paid twice

Two weighting questions, both now closed by measurement rather than argument.

**The tent exponent buys nothing.** §3.8 shows `tb_L` is, up to scale, a tent-weighted
average of returns with `w_j ∝ j(L−j)`. That exponent was never fitted — it falls out of
OLS algebra, not out of any belief about markets — which made it the one remaining
structural lever with plausible upside. Sweeping `w_j ∝ (j(L−j))^p` over
`p ∈ {0.5, 0.75, 1, 1.5, 2}` at L=252, one FIR pass each over the winsorised returns:

| paired − p=1.0 | h1 | h5 | h21 | h63 |
|---|---|---|---|---|
| p = 0.5 | +0.0004 (0.42) | −0.0005 (−0.48) | −0.0006 (−0.43) | +0.0004 (0.28) |
| p = 0.75 | +0.0003 (0.39) | +0.0001 (0.16) | −0.0001 (−0.16) | +0.0010 (1.04) |
| p = 1.5 | −0.0006 (−0.83) | +0.0009 (1.02) | −0.0001 (−0.10) | +0.0003 (0.28) |
| p = 2.0 | −0.0015 (−1.56) | −0.0001 (−0.09) | −0.0010 (−0.79) | −0.0003 (−0.20) |

Every |t| ≤ 1.56. Under CPCV (5 groups, leave-one-out, **252-bar purge** — the purge must
equal the longest rung or training windows overlap test labels through the scan itself —
and a 21-bar embargo) no arm beats `p=1` in more than 2 of 5 splits, against a 70%
adoption bar. **Keep `p=1`.** The result is not "the tent is optimal"; it is "the tent is
*flat* in its exponent" — anything vaguely hump-shaped over 252 bars scores the same.
That closes the last structural lever, and it closes it negative.

**Winsorisation's edge largely evaporated on the clean panel.** The pre-§2.1 record had
`wb_L` beating `tb_L` at every horizon (+0.0013 t 6.27 at h1, +0.0029 t 3.83 at h21). Now:

| `wb_top` − `tb_top` | h1 | h5 | h21 | h63 |
|---|---|---|---|---|
| pre-bar-integrity (stale) | +0.0013 (6.27) | — | +0.0029 (3.83) | +0.0051 (4.39) |
| clean panel | −0.0003 (−0.21) | **+0.0031 (2.29)** | +0.0017 (1.18) | +0.0024 (1.29) |

Only h5 still clears t=2, on four horizons tested. The mechanism is not mysterious:
**clipping at ±3·MAD was cleaning up the same foreign prints that §2.1's bar integrity
now removes at the source.** You get paid for that fix once, not twice. The right reading
is that on **raw** IC `wb` and `tb` are now near-ties and the earlier "beats it everywhere"
was substantially an artifact of dirty inputs.

**Amended by §3.13.** That paragraph is correct about raw IC and wrong as a verdict on
clipping. Neutralised on `flat_ret_vol` + `nsd` + `ladv`, `wb_top` retains 25–50% of its IC
with t ≥ 2.8 at every horizon while `tb_top` retains 1–29% and dies at h252 (t 0.10).
Clipping's contribution is invisible in raw IC because both signals share the same large
common component with the baselines; it shows up entirely in the orthogonal part. `wb_top`
is the candidate for a better reason than "not worse anywhere".

### 3.6 Scale (`sd`) is real, cheap, and a different anomaly

Flat across rungs (IC_h1 ≈ 0.011–0.013, t ≈ 4 at every rung), which is the expected shape:
there is no term structure to a volatility estimate. This is idiosyncratic vol falling out
of sums already computed — **the low-vol effect arriving free**, not trend scanning, and it
should not be described as such.

Standalone on the clean panel, `nsd` = −`sd_top` scores 0.0109 / 0.0093 / 0.0067 / 0.0156
at h1/h5/h21/h63 — **only h1 clears t=3, and h21 is indistinguishable from zero (t 0.74).**
Paired against `tb_top` it loses everywhere (−0.0065 to −0.0187, t −1.69 to −2.46). It is
cheap (turnover 0.0082, the lowest of anything measured) and it is a real, separate
anomaly, but as a *standalone* signal at these horizons it is weak and it is not a trend
signal. Its prior recorded value was as a third leg in a composite that no longer exists;
that claim is not carried forward, because the composite it improved is gone.

Keep emitting it — it costs nothing, arriving from sums already computed — and treat it
as an input to a future risk or sizing model, not as alpha.

### 3.7 Removed, with the algebra

Two objects were emitted, evaluated, and cut. The algebra is kept so neither has to be
rediscovered by rescanning 19M rows.

#### `tc` (curvature) — cut 2026-08-07

The quadratic leg of the same fit, `t_c = c·√(Su4_c)/sd` on the orthogonalised basis
`{1, u, u² − Su2/L}`, where `Su2 = L(L²−1)/12` and `Su4_c = Su4 − Su2²/L`. Orthogonality
is what made it cheap: `b` and `c` come from one pass of the same rolling sums, and the
centred basis makes the two estimates independent by construction, so `tc` cost only a
second accumulator chain (`W2`, a rank-2 recursion) rather than a second scan.

It was cut for one reason: **it is zero on returns, on all ten rungs, signed and absolute**
(§3.9), and standalone it backtested at alpha −0.246 on 6.2× turnover (`LOSS.md` §4). The
earlier case for it rested entirely on its covering the 2009-16 "CTA winter" inside an
equal-weight composite — but that composite was significantly *worse* than its own trend
leg at h1 (−0.0043, t −3.89), so the coverage was never actually purchasable at a weight
anyone had fitted. A signal that is zero on the thing you are predicting does not become
positive by being averaged with one that isn't.

The forward-**vol** use is the one thing that was real, and it has no consumer and no
advantage: `sd_21`/`sd_63` beat `tc` there by ~10×. Removing `tc` deleted the whole `W2`
recursion, `Su4_c`, `sse2` and two 19M×10 scratch surfaces from the kernel — a real
simplification, not a column drop.

**The 2009-16 hole is now uncovered, and that is a stated, accepted exposure** (§3.10's
era table), not something hidden inside a full-sample mean.

#### `rz` (endpoint residual)

Emitted from the first build, never evaluated. Its IC rose **monotonically with rung
length** (h21: 0.0015 at L=5 → **0.0201** at L=252, t 4.09), so it was not short-term
reversal. But `corr(nrz_252, ncurv) = **+0.851**`, and the reason is mechanical.

`rz` was the endpoint residual of the **linear** fit while the surface also fits a
quadratic. With the centred basis the linear intercept is `a = ȳ = a' + c·Su2/L`, so

```
y_t − ȳ − b·m  =  c·(m² − Su2/L) + ε_t ,    m² − Su2/L ≈ L²/6  for large L
```

**The endpoint is exactly where the quadratic term is largest, so the linear fit dumps the
entire curvature signal into its last residual.** `rz_L ≈ c·L²/6/sd`, and `t_c ∝ c/sd`.
Same object, rescaled. Adding it to a composite already holding curvature was significantly
**harmful** (−0.0020, t −4.05 at h1) at turnover 0.1491–0.4555, the highest measured
anywhere.

If ever re-introduced, take it from the **quadratic** fit —
`ŷ_t = ȳ + b·m + c·(m² − Su2/L)` over `√(SSE₂/(L−3))` — which is orthogonal to both slope
and curvature by construction. That version is untested.

### 3.8 What `tb` actually is, what its null is, and where the edge comes from

Measured on the **pre-bar-integrity** panel (~15.7M rows / 6,385 dates), same convention as
§3 otherwise. **These numbers have not been re-measured**, and the two above them that were
re-measured both moved, so treat the third decimal as unreliable. They are kept because
this section is structural — algebraic identities and null distributions, verified against
`lstsq` and simulation, which a change of panel does not alter — whereas §3's tables were
comparative and had to go. Where §3.8's *comparisons* conflict with §3's, §3 wins.

It answers two questions the record never did: is an OLS t-stat meaningful on a
non-Gaussian unit-root path, and does it beat a plain trailing return.

**`tb_L` is a tent-weighted mean of daily returns over its own residual vol.** With the
centred design `b = Σ_j w_j r_j / Σ_j w_j`, `w_j = j(L−j)/2` — an inverted parabola over
the window. Verified: reconstructing the slope from returns by FFT convolution reproduces
the stored `tb·sd/√Su2` at **corr 1.000000, median relative difference 4.2e-8** (17.5M
rows). No regression is happening that a 252-tap FIR filter does not already do.
Corollary: the weight on the most recent bar is ≈ 4/L of peak, so **`tb_252` already
soft-skips the last month** — the 12−1 `SKIP=21` is partly redundant with the weighting.

**Non-Gaussianity is not the problem. The unit root is.** Null of `tb_252`, 20k paths each:

| innovations | sd(tb) | p99 \|tb\| |
|---|---|---|
| iid noise around a line (what OLS assumes) | 1.00 | 2.60 |
| Gaussian **random walk** | 22.23 | 62.75 |
| random walk, Student-t(3) | 22.32 | 62.37 |
| random walk, α=1.5 symmetric stable (**infinite variance**) | 21.57 | 56.27 |
| random walk, α=1.1 | 20.99 | 52.05 |

The statistic is **self-normalising and effectively distribution-free** — infinite variance
moves its null by <6%. Fat tails, jumps and skew do not break it. What breaks it is that
residuals around a linear trend in log price are I(1), so `se(b)` is understated ~22× and
`tb` is not a t-statistic under any DGP resembling a price.

Precisely what that does and does not forbid:

* **Absolute magnitude, thresholds, probabilities: dead.** `|tb_252| = 45` is an ordinary
  random-walk draw (p99 = 63), not evidence of a trend. Nothing here maps to a p-value.
  This is unchanged by the robust paths — winsorising moves the null sd by **0.00**
  (22.28 → 22.28), because the tails were never what inflated it. Null sd at L=252 is
  22.3 for `tb`, `wb` **and** `sb` under Gaussian, t(3) and α=1.5 innovations alike.
* **Cross-sectional value vs rank: a tie, so use the rank.** Regressing on the per-date
  z-scored *value* rather than the rank (target held as a rank throughout) moves IC by
  ≤ 0.0008 at every horizon for all three columns, every |t| ≤ 1.0. The magnitude carries
  nothing the ordering doesn't, and destroys nothing either. Rank is the honest form and
  costs zero. `LOSS.md`'s kill of conviction-weighting on raw `tb` is consistent with this:
  there was never a magnitude to weight by.

Only `sb_252` shows real dispersion meaningfully above its own null — 26.17 vs 22.40
(ratio 1.17), against 1.01 for `tb` and 1.04 for `wb`. Suggestive only; pooled dispersion
also absorbs vol clustering and cross-sectional vol heterogeneity.

`data/_staging/meta_score.parquet`'s `p_correct` was **checked and is not affected**: it
spans 0.072–0.900 with median 0.520 and p99 0.603, nothing above 0.99, and
corr(|`trend`|, `p_correct`) = 0.056. It is a calibrated classifier output, not a CDF of a
scan statistic. (It remains an orphan — no code in the tree produces it.)

Confirming it on real data: `mean|tb_L|` across the 10 rungs fits **`L^0.51`** (2.58 at
L=5 → 18.09 at L=252), matching the driftless random-walk null exactly (sim 6.27 / 10.93 /
22.23 at L = 21 / 63 / 252). The observed dispersion of `tb_252` is indistinguishable from
noise; only its cross-sectional *ordering* carries anything.

**Against a plain return.** All signals over the identical window `[t−273, t−21]`, identical
lag and screen. IC (NW t):

| signal | numerator | denominator | h1 | h21 | h63 |
|---|---|---|---|---|---|
| `ret` (the 12−1 return) | flat | none | 0.0122 | 0.0161 (2.11) | 0.0169 (1.38) |
| `bL` | tent | none | 0.0124 (6.15) | 0.0210 (2.84) | 0.0240 (1.92) |
| `zmom` | flat | σ(daily ret, 252) | 0.0145 (7.17) | 0.0199 (2.56) | 0.0258 (2.06) |
| `ret/sd` | flat | `sd_252` | 0.0147 (7.49) | 0.0204 (2.68) | 0.0274 (2.27) |
| `b/σ` | tent | σ(daily ret) | 0.0145 (7.51) | 0.0239 (3.22) | 0.0305 (2.43) |
| **`tb_252`** | tent | `sd_252` | **0.0147 (7.82)** | **0.0242 (3.30)** | **0.0315 (2.61)** |

Paired vs `tb_252`: `ret` −0.0025 (t −3.11) / −0.0081 (−2.58) / −0.0147 (−2.98). **Yes,
`tb_252` beats the plain 12−1 return, significantly, at every horizon.** But decomposing
the h21 gap:

* vol normalisation: **+0.0038** — *any* denominator; nothing to do with trend scanning
* tent weighting: **+0.0041** — a one-line FIR filter
* the OLS residual sd instead of a rolling σ: **+0.0003, t −0.34.** Nothing.

Swapping `sd_252` for σ(returns) or EWMA(hl=63): ΔIC −0.0003 / +0.0001 / −0.0008, all
|t| ≤ 1.34. **The regression, the residuals and the t-normalisation contribute nothing that
σ of daily returns does not.** The tent weighting is the one part of the OLS that is
load-bearing, and it is not just a longer skip — flat windows with the centroid pushed back
to match (`[−273,−63]`, `[−273,−105]`) lose by −0.0049 (t −2.95) and −0.0056 (t −1.99) at
h21.

**Where the remaining edge is: the input, not the estimator.** Both extra paths, measured
off the stored surface:

| signal | h1 | h21 | h63 |
|---|---|---|---|
| `wb_252` — tent × winsorised returns | **0.0161** (8.50) | **0.0272** (3.71) | **0.0366** (3.07) |
| Δ vs `tb_252`, paired | **+0.0013 (t 6.27)** | **+0.0029 (t 3.83)** | **+0.0051 (t 4.39)** |
| `sb_252` — tent × sign(r), nothing else | 0.0149 (9.67) | 0.0226 (3.73) | 0.0300 (3.08) |
| Δ vs `tb_252`, paired | +0.0001 (0.13) | −0.0017 (−0.65) | −0.0016 (−0.40) |

The statistic is robust to fat tails; its **input is not**, and clipping the input pays
+9% IC at h1 and +16% at h63. Meanwhile a sign-only, fully nonparametric version ties
`tb_252` everywhere (all |t| < 0.7) and has the highest date-stability at h1: **nothing in
the Gaussian machinery is load-bearing.** Both are stored, so neither claim needs a rescan
to re-check.

Two implementation consequences, now fixed in `trendscan.py`: a degenerate window emits
`NaN` rather than `0.0` (0.035% of rows; `0.0` put them at *mid-rank*, and the bottom-1%-`sd`
rows carried mean |`tb_252`| 33.7 vs 17.9), and the surfaces are NaN-filled at allocation
rather than relying on `MIN_SEG` two functions away. Stale prices are **not** a
contaminant: corr(21d zero-return fraction, |`tb_21`|) = −0.023, and names with >30% zero
returns have *lower* |`tb_21`| (3.41 vs 4.66).

### 3.9 The unit root, the orthogonality of `b` and `c`, and what the scan is made of

All numbers here: 15.5–15.8M screened rows, ~6,390 dates, 1-bar lag, `SKIP=21`, ADV21 ≥ $1M,
close ≥ $5, per-date Spearman, Newey–West `⌈1.5h⌉`; sims are 40k paths per rung.

**Same caveat as §3.8: this is the pre-bar-integrity panel and has not been re-measured.**
The simulation results (κ scaling, the null distribution, the tent decomposition) are
panel-independent and stand. The IC comparisons in it do not — §3 supersedes them. The
curvature results are retained as the evidence trail behind §3.7's cut, not as live
findings; `tc` no longer exists.

**What a unit root is.** A series has one when `y_t = y_{t−1} + ε_t`: shocks are *permanent*,
`Var(y_t) = σ²t` grows without bound, and there is no level or line the series returns to.
OLS assumes the opposite — `y_t = a + b·t + ε_t` with iid `ε` (*trend-stationary*), where
shocks are temporary and the path is pinned to the line. Log prices are the first; the
regression in `trendscan.py` assumes the second. The cost is not bias, it is the standard
error:

| | trend-stationary (assumed) | unit root (actual) |
|---|---|---|
| true `sd(b̂)` | `∝ σ/L^1.5` | `∝ σ/√L` |
| OLS's `se(b̂)` = `sd_resid/√Su2` | `∝ σ/L^1.5` ✓ | `∝ σ/L` ✗ |
| ratio = null sd of `tb` | `1` | **`∝ √L`** |

Information about a trend accumulates at `L^1.5` if the model is true and at `√L` if the
series is a random walk. `tb` divides by the former; the gap **is** the `√L` inflation.
Measured null, `sd = κ·√L`, flat for `L ≥ 10`:

| L | 5 | 10 | 21 | 42 | 63 | 84 | 126 | 168 | 210 | 252 |
|---|---|---|---|---|---|---|---|---|---|---|
| `κ_b` | 1.83 | 1.39 | 1.37 | 1.38 | 1.40 | 1.39 | 1.39 | 1.40 | 1.41 | **1.41** |
| `κ_c` | 1.85 | 0.81 | 0.78 | 0.79 | 0.79 | 0.80 | 0.81 | 0.81 | 0.81 | **0.81** |

The kernel is not wrong — under a genuinely trend-stationary DGP the same code returns
sd(`tc`) = **1.00** exactly. The data is not iid. `KAPPA_B`/`KAPPA_C` and `zscale()` in
`trendscan.py` are this table.

**`b ⊥ c`, by construction and under the unit root.** With a centred `u_i = i − (L−1)/2` and
a centred square `u²_c = u² − Su2/L`, the design columns are `[1, u, u²_c]` and

* `Σ 1·u = 0` (u is centred), `Σ 1·u²_c = Su2 − L·(Su2/L) = 0` (by construction),
* `Σ u·u²_c = Σu³ − (Su2/L)Σu = 0` (odd symmetry).

So `X'X` is **diagonal**, every coefficient is a univariate projection, and adding or
dropping the quadratic term cannot move `b̂`. That is why both come out of one pass.
Diagonality gives `Cov(b̂, ĉ) = 0` only under iid errors, which is what the unit root breaks
— but a symmetry argument rescues it: for iid *increments* the time-reversed path has the
same law, and reversal sends `u → −u`, hence `b̂ → −b̂` while `ĉ → ĉ`. So
`(b̂, ĉ) =ᵈ (−b̂, ĉ)`, forcing `E[b̂ĉ] = −E[b̂ĉ] = 0` for **any** iid-increment process,
whatever its distribution.

| DGP | corr(`tb`,`tc`) | corr(\|`tb`\|,\|`tc`\|) |
|---|---|---|
| Gaussian RW | −0.009 | **−0.194** |
| t(3) RW | +0.002 | −0.192 |
| RW + drift | −0.009 | −0.223 |
| trend-stationary (OLS's world) | −0.007 | **−0.009** |
| **real data (`wb_252` vs `tc_252`), per-date rank** | **−0.0009** | **−0.152** |

The signed orthogonality is exact and holds on real data. The *magnitudes* are dependent
under a unit root and independent when OLS's assumptions hold — and real prices sit on the
unit-root side. A straight-looking Brownian path has less room left to look curved; that is
a property of the null, not a signal.

**Orthogonal, and empty for returns.** On every rung, signed and absolute:

| | h1 | h21 | h63 |
|---|---|---|---|
| `tc_21` | −0.0008 (−0.7) | −0.0006 (−0.3) | −0.0002 (−0.2) |
| `tc_63` | −0.0020 (−1.7) | −0.0049 (−1.2) | −0.0055 (−1.4) |
| `tc_126` | +0.0001 (0.1) | −0.0053 (−1.1) | −0.0055 (−0.8) |
| `tc_252` | +0.0008 (0.6) | −0.0060 (−1.3) | −0.0011 (−0.1) |
| −\|`tc_252`\| | +0.0002 (0.2) | −0.0044 (−1.4) | −0.0107 (−2.1) |
| `wb_252` (yardstick) | +0.0161 (8.5) | +0.0272 (3.7) | +0.0367 (3.1) |

No rung holds a sign across horizons or clears t = 2. Orthogonality guarantees `c` is not a
restatement of `b`; it guarantees nothing about `c` being informative, and it is not.

**But curvature predicts RISK, incrementally.** Target = realised vol of the next 21 days:

| signal | raw IC | \| `sd_252` | \| `sd_252`, `sd_21` |
|---|---|---|---|
| \|`tc_252`\| | +0.0562 (12.6) | **−0.1847 (−40.1)** | **−0.1387 (−36.6)** |
| \|`tc_126`\| | +0.0444 (10.7) | −0.0448 (−12.6) | — |
| `sd_21` | +0.5925 (133) | +0.3256 (79) | — |
| `sd_63` | +0.6267 (145) | +0.3161 (69) | +0.1959 (54) |
| \|`wb_252`\| | +0.0026 (0.4) | — | — |

The sign flips under conditioning: *at a given trailing vol*, a strongly curved path is
followed by a quieter one, and it survives controlling for both long and short trailing vol
at t = −36. Unresolved caveat: `tc = c·√((L−3)Su4_c/sse2)` divides by the residual sd
*after* the quadratic, so |`tc`| conflates curvature with smoothness, and `sse2` is not
emitted, so the two cannot be separated from stored columns. §6 carries the one-column
experiment that would.

**The denominator is not the lever — a falsified hypothesis, recorded so it is not re-run.**
`se_OLS` uses the residual spread of the *level*, which under a unit root is a Brownian-bridge
range — effectively one observation — where `sd(returns)` averages `L`. Replacing it should
have been free IC. At L=252, `b` reconstructed exactly as `tb·sd/√Su2`:

| h21 | IC | vs `tb_252` |
|---|---|---|
| `tb_252` | +0.0242 (3.30) | — |
| `b_252` (no denominator at all) | +0.0210 (2.84) | −0.0033 (−1.19) |
| `b`/`rv252` | +0.0239 (3.21) | −0.0003 (−0.42) |
| `b`/`rv63` | +0.0245 (3.40) | +0.0003 (0.30) |
| `sd_252`/`rv252` (the denominator alone) | +0.0043 (1.04) | — |

Same at h1 and h63. Vol-normalising is worth ~+0.003 over the bare slope; *which* vol
estimator does it is a coin flip; the denominator carries no alpha of its own. **The unit
root corrupts the scale of `tb`, not its ordering, and there is no statistical fix that
improves the signal.**

**Where the edge actually lives: the tent.** `b = Σ_j w_j r_j / Σ_j w_j` with
`w_j = j(L−j)/2`. The alternative is flat weights, i.e. the plain L-day return:

| | h1 | h21 | h63 |
|---|---|---|---|
| `tb_252` | +0.0147 (7.8) | +0.0242 (3.3) | +0.0315 (2.6) |
| tent `b`/`rv252` | +0.0145 (7.5) | +0.0239 (3.2) | +0.0305 (2.4) |
| flat 252d return | +0.0122 (5.8) | +0.0161 (2.1) | +0.0169 (1.4) |
| flat 252d / `rv252` | +0.0145 (7.2) | +0.0199 (2.6) | +0.0258 (2.1) |
| 12−1 momentum | +0.0116 (5.7) | +0.0168 (2.3) | +0.0172 (1.4) |
| 12−1 / `rv252` | +0.0139 (7.2) | +0.0211 (2.9) | +0.0259 (2.1) |
| Δ `tb_252` − flat 252d, paired | **+0.0025 (3.1)** | **+0.0081 (2.6)** | **+0.0147 (3.0)** |
| Δ `tb_252` − 12−1/`rv252`, paired | +0.0009 (2.2) | +0.0031 (2.0) | +0.0056 (2.9) |

The tent beats flat momentum and beats 12−1 at every horizon, significantly. **This is the
answer to "is trend scanning better than simple returns": yes, and the reason is the
`j(L−j)` weighting — mid-window emphasis, endpoints downweighted — not the regression, not
the t-statistic, and not the curvature.**

**The absolute form, and the gate.** `z_L = tb_L/(κ_b√L)` has a unit null on every rung,
security and date. It is a monotone rescale within a rung, so it changes no per-date
ordering — but `|z| = 2` becomes a genuine ~5% event, where `|tb_252| = 45` meant nothing.
IC of `wb_252` inside **fixed** (not per-date) `|z|` buckets:

| \|z\| | share | h1 | h21 | h63 |
|---|---|---|---|---|
| 0.0 – 0.5 | 38.3% | +0.0075 | +0.0157 | +0.0226 |
| 0.5 – 1.0 | 28.8% | +0.0131 | +0.0241 | +0.0321 |
| 1.0 – 1.5 | 17.3% | +0.0155 | +0.0247 | +0.0367 |
| 1.5 – 2.0 | 8.9% | +0.0136 | +0.0287 | +0.0375 |
| 2.0 + | 6.8% | +0.0150 | +0.0230 | +0.0296 |

The bottom 38% of the panel carries roughly **half** the IC of everything above it, at every
horizon, and bucket shares are stable to 0.1% across horizons — which is what makes a fixed
threshold viable rather than a per-date quantile. The top bucket **rolls over**: this is a
gate, not a monotone weight, consistent with §3.8's finding that value ties rank linearly.
The magnitude says *whether to trust the sign*, not *how much to size*.

### 3.10 Stability by era — the full-sample mean hides a dead decade

IC by era, `eval` section 3. The full-sample number is an average over three very
different regimes and should not be quoted alone:

| signal | 2000–2008 h1 / h21 | 2009–2016 h1 / h21 | 2017–2026 h1 / h21 |
|---|---|---|---|
| `tb_top` | 0.0198 / **0.0342** | 0.0110 / **0.0111** | 0.0209 / 0.0265 |
| `wb_top` | 0.0210 / **0.0351** | 0.0060 / **0.0131** | 0.0233 / 0.0287 |
| `sb_top` | 0.0221 / 0.0367 | 0.0099 / 0.0075 | 0.0193 / 0.0202 |
| `flat_ret` | 0.0164 / 0.0234 | 0.0095 / 0.0088 | 0.0162 / 0.0163 |
| `flat_ret_vol` | 0.0189 / 0.0285 | 0.0091 / 0.0090 | 0.0185 / 0.0229 |

**2009–2016 is a hole.** `tb_top`'s h21 IC falls to 0.0111, roughly a third of its
2000–08 value, and `wb_top` to 0.0131. That is the post-GFC "CTA winter" and it is
market-wide, not an artifact — every trend column and both flat baselines fall together.
Curvature used to be the proposed patch for exactly this era; §3.7 explains why that patch
was never actually purchasable. **The exposure is now uncovered and is accepted, not
hidden.** Any deployment sized off the full-sample mean is sized off a number the strategy
did not earn in eight of its twenty-six years.

Note also that the scan's *advantage over the flat baselines* is not stable: in 2009–16
`flat_ret_vol` (0.0090) essentially matches `tb_top` (0.0111).

### 3.11 Delisting truncation — measured, and small

`fwd{h}` is NaN in a security's final `h` bars, so a name that delists has its terminal
decline excluded from every h21/h63 IC. Sized directly (`eval` section 4):

* 3,479 of 5,930 segments end more than 90 days before the panel ends.
* Of the rows in those segments, **0.80%** (h21) and **2.38%** (h63) have no target at all.

So the direct truncation is small. An earlier version of this section compared full-sample
IC against a "survivors only" subset and reported Δ = −0.0102, which was **wrong** — that
split removes 40.95% of rows concentrated in the highest-IC era (2000–08, h21 IC 0.0342),
so it measured era mix, not truncation. The confounded comparison has been removed rather
than annotated. The row shares above are the honest figure.

Truncation in the *IC* is small; the delisting write-off in the *backtest* is not (§3.12).

### 3.12 Net of costs on a long-only top-decile book — which tests the weak half

> **Read §3.13 first.** This section costs one specific portfolio construction, and §3.13
> shows that construction throws away most of the signal: the payoff is 2.2–3.4× larger on
> the short side, and a long-only top-decile book holds only D10. The result below is
> sound for what it measures. What it measures is not "is this a good signal".

`research/BACKTEST.py`, IBKR commissions + `spread_costs` from ADV, long-only equal-weight
top decile, rebalanced every 21 bars, $50M, 1-bar lag, impact coefficient 0.02, borrow fee
0.0, and a −30% terminal write-off on the 3,153 names that stop printing >90d before the
panel ends.

| book | ann_return | ann_vol | Sharpe | maxDD | ann_turnover | ann_cost_drag |
|---|---|---|---|---|---|---|
| `wb_top` candidate | 0.0205 | 0.1998 | 0.2019 | −0.640 | 3.19 | 0.0455 |
| `tb_top` incumbent | 0.0143 | 0.2021 | 0.1718 | −0.659 | 3.27 | 0.0480 |
| **`flat_ret` unfitted baseline** | **0.0330** | 0.2441 | **0.2557** | −0.661 | 3.91 | 0.0584 |

**The unfitted 12−1 return beats both scan books, on CAGR and on Sharpe.** It does so
*despite* higher turnover (3.91 vs 3.19) and a higher cost drag (5.84% vs 4.55% a year),
so this is not a cost artifact — the baseline earns more gross as well as net.

Sensitivity to the delisting assumption, which is the single largest modelling choice here:

| book | Sharpe w/ −30% write-off | hold-flat (biased high) | Δ |
|---|---|---|---|
| `wb_top` | 0.2019 | 0.2495 | +0.0477 |
| `tb_top` | 0.1718 | 0.2293 | +0.0575 |
| `flat_ret` | 0.2557 | 0.3230 | +0.0673 |

Holding delisted names flat inflates Sharpe by 0.048–0.067 — **larger than the entire gap
between the candidate and the incumbent.** Every historical result in this project that did
not model delisting is overstated by about this much. It does not change the ranking.

**What this table cannot tell you, stated plainly:** a long-only top-decile equity book is
dominated by market beta, and `BACKTEST.py`'s summary returns no benchmark-relative alpha
or information ratio, so the plan's requirement to report *alpha and IR beside Sharpe* is
**not met here**. `flat_ret` also runs 4.4 points more annualised vol, i.e. more beta, which
flatters its CAGR. A long/short or beta-neutral construction is the right test of whether
the scan has alpha, and it has not been run. What the table does establish is narrower and
sufficient for the decision at hand: **on the construction that was actually specified, the
scan does not beat the free baseline.**

**Trial count, for deflation (LdP ch14):** this configuration is the survivor of roughly
40 comparisons on one panel — ~15 in §3.1–3.6, 10 rungs, 5 tent exponents, plus the
input-path and baseline comparisons. Against a Sharpe of 0.20 on a 26-year daily track,
with that many trials and a variance across them of the order of the spread in §3's tables,
the deflated Sharpe does not approach 0.95. **No number in this section supports
deployment.** They are reported because a negative result that cost 40 trials is worth
exactly as much as a positive one, and unlike a positive one it is probably true.

---

### 3.13 Signal diagnostics — the section that answers "is this a good signal"

`python research/trendscan_eval.py signal`. No portfolio, no costs, no construction choices.
Four questions, and the second is the one that matters.

**Where does the IC peak? It doesn't.**

| signal | h1 | h5 | h21 | h63 | h126 | h252 |
|---|---|---|---|---|---|---|
| `tb_top` | 0.0174 | 0.0212 | 0.0241 | 0.0332 | 0.0402 | 0.0456 |
| **`wb_top`** | 0.0172 | 0.0243 | 0.0258 | 0.0355 | 0.0456 | **0.0518** |
| `sb_top` | 0.0172 | 0.0220 | 0.0213 | 0.0316 | 0.0357 | 0.0428 |
| `flat_ret_vol` | 0.0157 | 0.0195 | 0.0203 | 0.0254 | 0.0345 | 0.0442 |
| `nsd` (low-vol) | 0.0109 | 0.0093 | 0.0067 | 0.0156 | 0.0280 | 0.0459 |
| `ladv` (size) | 0.0028 | −0.0022 | −0.0114 | −0.0216 | −0.0302 | −0.0374 |

IC triples from h1 to h252 and is **still rising at the edge of the grid.** This is a
long-horizon object, which is why the h63 ceiling the old grid used was the wrong place to
stop and why a 21-day rebalance (§3.12) is not its natural holding period. Note also that
at h252 both `nsd` (0.0459) and `flat_ret_vol` (0.0442) reach `tb_top`'s level — at exactly
the horizon where trend looks strongest, two cheap controls look equally strong. Hence:

**Does it survive the cheap alternatives?** Per-date *joint* OLS residual on the controls,
scored by the identical statistic (residual-of-a-rank is not re-ranked — that would swap
estimators mid-comparison). Retained % is of the raw IC above.

*After projecting out `flat_ret_vol` alone:*

| | h1 | h21 | h63 | h252 | retained |
|---|---|---|---|---|---|
| `tb_top` | 0.0034 (3.78) | 0.0124 (3.52) | 0.0168 (3.20) | 0.0114 (2.13) | 20–52% |
| `wb_top` | 0.0063 (7.00) | 0.0168 (4.72) | 0.0234 (4.45) | 0.0240 (3.91) | 36–66% |
| `sb_top` | 0.0071 (9.18) | 0.0116 (3.81) | 0.0151 (3.35) | 0.0194 (3.15) | 40–54% |

*After projecting out `nsd` + `ladv` (vol and size):* retained **77–90%** at h21 and beyond.
It is not a low-vol or small-cap bet wearing a trend costume.

*After projecting out all three jointly — the real test:*

| | h1 | h21 | h63 | h126 | h252 | retained |
|---|---|---|---|---|---|---|
| `tb_top` | 0.0012 (1.37) | 0.0068 (2.16) | 0.0096 (2.14) | 0.0051 (0.83) | **0.0005 (0.10)** | 1–29% |
| **`wb_top`** | 0.0044 (5.37) | 0.0118 (3.92) | 0.0176 (4.06) | 0.0178 (2.98) | 0.0151 (2.83) | **25–50%** |
| **`sb_top`** | 0.0056 (8.47) | 0.0092 (3.69) | 0.0122 (3.40) | 0.0120 (2.75) | 0.0123 (2.54) | **29–43%** |

**This is the most important table in the file, and it reverses §3's ordering.** On raw IC
the three input paths are ties — `sb` within |t| ≤ 0.82 of `tb`, `wb` clearing only at h5 —
so they were documented as interchangeable. Jointly neutralised they are nothing of the
kind: **`tb_top` is fully explained away** (1% retained at h252, t 0.10) while `wb_top` and
`sb_top` clear t=2 at every horizon. Raw IC cannot see this because all three share the
same large common component with the baselines; only the orthogonal part discriminates.

The practical consequence: **`sb_top` is the most defensible signal in the project.**
Cumsum of return *signs* — no prices, no vol estimate, no distributional assumption, the
cheapest thing here to compute — and it retains ~a third of its IC against every control
this panel can construct. It was kept only as a floor to stop a baseline being killed
unmeasured (§4); it turns out to be the thing itself. `wb_top` retains slightly more and is
the alternative; `tb_top` should not be the headline column.

This also partly rehabilitates winsorisation, which §3.5 struck. The strike was right about
the *raw* claim — clipping does not raise raw IC on a clean panel. But clipping's value
shows up in the **orthogonal** component (66% retained vs `tb`'s 52% at h63 against
`flat_ret_vol`; 50% vs 29% against all three). Both statements are true and they are about
different quantities.

**Is it monotone, or tail-only?** Per-date deciles, mean forward return in bp, averaged
over dates. 92.4% of rows carry a signal (the top rung needs 273 bars).

| `wb_top` | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | D10−D1 | mono |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h21 | −69 | −8 | 8 | 33 | 45 | 53 | 66 | 67 | 65 | 66 | 135 | 8/9 |
| h63 | −193 | −7 | 35 | 102 | 140 | 155 | 193 | 194 | 192 | 186 | 379 | 7/9 |
| h252 | −464 | 35 | 159 | 344 | 471 | 528 | 593 | 630 | 668 | 672 | 1136 | 9/9 |

Monotone, not tail-only — 9/9 rising steps at h252 for all three signals. But the payoff is
**badly asymmetric**, and this is the finding that reconciles §3.13 with §3.12:

| | h21 | h63 | h252 |
|---|---|---|---|
| `wb_top` D1 − mean | −102 | −293 | −828 |
| `wb_top` D10 − mean | +33 | +86 | +308 |
| **short side / long side** | **3.0×** | **3.4×** | **2.7×** |

**The bottom decile is worth 2.2–3.4× the top decile.** Most of what the scan knows is
which names to *avoid or short*, not which to buy. A long-only top-decile book — the
construction §3.12 costed — holds D10 and nothing else, so it captures the weaker third of
the signal and pays full costs for it. That is a portfolio-construction result, not a
signal result, and it is the single clearest instruction this project has produced about
what to build next.

**Two caveats that do not go away.** There is no sector or market-cap field in
`alpha_panel.parquet` (`security_id, date, close, high, low, volume, adj, split_factor`),
so "survives size neutralisation" rests on log-ADV as a proxy and **sector neutrality is
untested** — a plausible way for a residual this size to be an industry bet. And all of
this is in-sample on one panel; §3.12's ~40-trial count still applies to any number here.

---

## 4. Settled — do not retry

| | verdict |
|---|---|
| sign-sum heuristic (Σ sign over 4 windows) | dead short side: a −4 score predicts +9.9 bp vs +9.1 bp for 0. Beaten at every horizon at 5× the turnover. **NOTE** this is a *flat* sum over 4 fixed windows; the tent-weighted `sb_L` (§3.8) is a different object and ties `tb_L` |
| ~~`zmom` (mean Δlog P / σ̂√L) — weakest thing tested; beats nothing~~ | **WRONG, struck.** Measured in §3.8: `zmom` is IC 0.0145 / 0.0199 / 0.0258, **beats the raw 12−1 return** and is 82% of `tb_252`. Vol normalisation is worth +0.0038 at h21 on its own. This unmeasured line is why the plain-return baseline went unrun for the life of the file |
| `curv_rel` = sign(trend) × curvature | ~0 at every horizon. Curvature is **not** "accelerating trends continue" — it is direction-blind second-derivative mean reversion |
| `tc_L` (curvature), any rung, any use | **Cut from the build entirely, 2026-08-07** (§3.7). As a return signal: zero on all ten rungs, signed and absolute, nothing clears t = 2, and standalone it backtested at alpha −0.246 on 6.2× turnover. Its one real use — incremental forward-**vol** prediction — has no consumer and is beaten ~10× by `sd_21`/`sd_63`. Do not re-add it without a consumer that pays for the second accumulator chain |
| swapping `tb`'s denominator for a return-vol one | §3.9 — the unit root makes `se_OLS` a one-observation estimate of the vol it implicitly divides by, so this looked free. Measured: `b`/`rv252` and `b`/`rv63` are indistinguishable from `tb` (all \|t\| < 0.8), and the denominator alone carries IC 0.0043. Vol-normalising at all is worth ~+0.003 over the bare slope; the estimator is a coin flip. The unit root corrupts the **scale**, not the ordering |
| trend scanning vs a plain trailing return | **Split verdict — read both halves.** On *IC* the tent wins, re-confirmed on the clean panel: `flat_ret − tb_top` = −0.0032/−0.0079/−0.0166 (t −2.01/−2.45/−3.36) at h1/h21/h63, widening with horizon. **Net of costs on a long-only book it loses** (§3.12): `flat_ret` Sharpe 0.256 vs 0.202/0.172. A higher IC that does not survive a cost model is not an edge. Keep the baseline in every future comparison — it has now beaten the thing it was a control for |
| tent exponent `p` in `w_j ∝ (j(L−j))^p` | §3.5 — swept `{0.5, 0.75, 1, 1.5, 2}` under CPCV. All paired \|t\| ≤ 1.56, no arm wins more than 2 of 5 splits. The tent is **endpoint-insensitive, not optimal**; `p=1` stands. Do not re-sweep |
| winsorisation (`wb`) as an established win | §3.5 — **struck.** The pre-bar-integrity "+0.0029 at h21, t 3.83" does not replicate on the clean panel (+0.0017, t 1.18); only h5 clears t=2. The clipper was doing bar integrity's job. `wb` is a near-tie with `tb` **on raw IC** — but §3.13 shows the win is real in the *neutralised* component (retains 25–50% vs `tb`'s 1–29%), so the strike stands against the original evidence, not against the conclusion |
| adjacent differences of the surface | N_eff 10.07, the highest breadth in the project, and **zero IC**. Differencing near-identical estimates decorrelates them because what is left is noise |
| more columns on the trend axis | 8 EWMA spans → N_eff 1.9; 24-col tensor → 1.9; 6 trend estimators → 1.10. Saturated |
| daily-spaced windows | 117 columns → N_eff 2.14 |
| ladder mean for trend | §3.1 — loses to a single rung |
| argmax / endogenous horizon | §3.2 — no measurable edge on a fixed grid |
| dense scan grid | §3.2 — worth exactly zero, and it was 89% of the scan |
| fast/slow leg split | §3.3 — the fast leg is free to drop |
| hardcoding the skip | §3.4 — asymmetric; it degrades curvature |
| equal-weight trend+curvature composite | Loses to its own trend leg at h1 (−0.0043, t −3.89). Moot now that curvature is cut (§3.7), but the lesson generalises: an equal-weight blend of a strong and a weak signal is a weighting failure dressed as diversification |
| `rz` as emitted | §3.7 — 0.851 correlated with curvature, by construction |

## 5. Why breadth, not signal, is the scarce resource

The context this work sits in, from Man AHL's *Dynamics of Dispersion* (2025), Tan/Roberts/
Zohren's *Spatio-Temporal Momentum* (arXiv:2302.10175), and this project's own measurements.

**The trend signal is commodity; what is scarce is the number of genuinely independent bets,
and that number is always far below the nominal column count.** AHL's 20 designed-to-differ
proxies ran 0.87 average pairwise correlation pre-2007 and 0.78 after. Two completely
different construction routes hit the same ceiling, which is evidence of a structural
property of the momentum premium rather than an artifact of either method.

The actionable corollary: **adding columns to a saturated axis buys nothing; changing the
geometry of an axis beats extending it.** Curvature is the one change that ever raised
N_eff here (1.10 → 1.84 at the time), while six trend estimators sat at 1.10.

And **N_eff is gameable** — it can be inflated without limit by adding orthogonal junk. The
measured counterexample is in §4. The objective is N_eff *of streams that each carry alpha*,
or simply the composite's out-of-sample IC.

Two further findings that hold independently on both sides of the literature:

* **Complexity loses.** A single-layer perceptron with L1 beat MLP, CNN and LSTM on both
  STMOM datasets (US equities: 2.609 vs 1.040 / 1.015 / 0.192). Low-SNR panels overfit.
* **Equal weight is the boss fight** — and here it finally lost, to something simpler still
  (§3.5). Fitted allocation is where estimation error compounds; if you cannot beat the
  trivial baseline out of sample, the fit is noise. Turnover regularisation is the
  exception that reliably pays.
* **Expect the tail to pay.** All 20 AHL proxies sit below y = x — crisis Sharpe exceeds
  all-period Sharpe. Evaluate crisis and full-period performance separately or the whole
  point of the allocation is invisible.

## 6. Open

Five items closed since the last revision — the tent sweep (§3.5, negative), costs (§3.12,
negative on one construction), delisting truncation (§3.11, measured and small), curvature
(§3.7, cut), and *does the signal survive cheap alternatives* (§3.13, yes for `wb`/`sb`, no
for `tb`). What remains:

* **Build the long/short book.** §3.13's clearest instruction: the short side is worth
  2.2–3.4× the long side at every horizon, and §3.12's long-only top-decile construction
  throws that away, which is most of why it loses to an unfitted 12−1 net of costs. Until a
  short-side-inclusive book is costed, §3.12 is a result about one construction and not
  about the signal. This is the single most valuable open item.
* **Sector neutrality is untested and cannot be tested here.** `alpha_panel.parquet` has no
  sector, market-cap, or shares-outstanding field, so §3.13's "survives size" rests on
  log-ADV as a proxy and the residual could still be an industry bet. This is the most
  plausible remaining way for §3.13's headline to be wrong. Needs a mapping table, not
  another pass over this panel.
* **`sb_top` is now the reference signal, not `tb_top`** (§3.13). Docs, `EMIT` ordering, and
  any downstream consumer still lead with `tb`, which is fully explained away by cheap
  controls. Nothing is broken by this, but every table that reports `tb` first is
  misleading about which column matters.
* **Long-only top-decile cannot separate alpha from beta** (§3.12). The test that would
  change the verdict rather than explain it: run the same three books **beta-neutral or
  long/short**, and report alpha and IR, which `BACKTEST.py`'s summary dict does not
  currently produce. `flat_ret` wins partly by carrying 4.4 more points of vol; a
  vol-matched or beta-matched comparison is the honest rematch.
* **Everything is still in-sample** over one 26-year panel except the tent sweep. The
  ~40-trial count in §3.12 is the reason to distrust any positive number in §3; it is also
  why the negative result is the more credible one — searching 40 ways and *failing* to beat
  a free baseline is hard to explain by luck.
* **The 2009–16 hole is uncovered** (§3.10). `tb_top`'s h21 IC falls to 0.0111 there. No
  candidate patch remains now that curvature is cut, and none should be sought by searching
  the same panel again.
* **Where the `|z| ≥ 0.5` gate belongs** (§3.9) — universe screen, weight, or nothing. It
  drops 38% of names and claims *higher* IC at *lower* turnover, which is an unusually
  favourable combination and therefore one to distrust until costed. It was scoped into the
  sweep's CPCV and not run; it is the one untested lever that could plausibly move §3.12,
  because it attacks turnover and selection at once.
* **The grid still stops too early — now measured, not suspected.** §3.13: IC triples from
  h1 to h252 and is *still rising* at h252 for all three signals. Momentum is expected to
  decay past ~12m and reverse at 3–5y, so the peak is probably just past the edge — but the
  measurement now says the edge is not it. An 18M forward horizon and an 18M rung would
  settle where it turns. Note this is a statement about the *scan's* horizon, not a claim
  that a 12-month holding period is tradeable.
* **`rz` from the quadratic fit** (§3.7). Note this now requires restoring the quadratic
  accumulator chain that §3.7 deleted — no longer free.
* **The trend-scan label** — LdP as published, forward-looking — lifted target IC 1.9–3.5×
  over a fixed-horizon forward return. Still a live proposal for a model's target vector. It
  leaks up to `L_max` bars, so it needs purging **and a 252-bar embargo**.

## 7. Reproducing

As of 2026-08-08 only the build reproduces. The four `trendscan_eval.py` commands that
produced §3 are cut from the tree; see the status note at the top.

```
python research/trendscan.py check        # kernel vs lstsq, NaN emission,           ~15 s
                                          # RESYNC drift, κ-null
python research/trendscan.py build        # scan, print, discard                     ~11 s
```

`check` must pass before anything else is believed. The build reads only
`data/_staging/alpha_panel.parquet` and produces 18,999,392 rows × 16 columns (10 `tb`
rungs plus `security_id`, `seg`, `date`, `close`, `volume`, `y`); the scan itself is ~3 s
of that and the DuckDB read is most of the rest. There is nothing else to run: every
reduction, target, screen and cost that used to follow it was removed on purpose, and
re-adding one means recovering the harness from `9810fa7`, not reinventing it — every
measured mistake in §4 was a reduction baked in where it could not be questioned.

**The surface is not written to disk** (§2), and the harness ships in the same directory,
imported from the module it scores. Both are deliberate: the previous harness was deleted
in `c6c8e41` and the loss went unnoticed for two commits, `research/loss_test.py` vanished
entirely (see the note atop `LOSS.md`), and the stale `trendscan.parquet` is why all of §3
had to be re-measured. A surface that is never stored cannot go stale, and a harness that
fails to import when the signal changes under it does not get orphaned.

**`sweep` is gone.** §3.5 settled the tent exponent at `p = 1` under CPCV and the result was
negative — no `p` beat the tent in ≥7/10 splits. The 116 lines existed to answer a question
that is now answered, so they were deleted rather than left to be re-run. The §3.5 numbers
stand as recorded; reproducing them needs `git show` of this commit's parent.

`data/_staging` still holds orphans with no producing code — `trendscan.parquet` (4.6 GB),
`trendscan_features.parquet` (472 MB), `meta_score.parquet` (156 MB), `signal_corr.parquet`,
`signal_surface.parquet`. Nothing reads any of them. Treat all as stale; they are safe to
delete. `data/_raw` is never touched by anything here.
