# Trend scanning — XCAP v0

The single record for `research/trendscan.py`. Supersedes and replaces
`RESULTS_trendscan_vs_heuristic.md`, `SYNTHESIS_breadth_not_signal.md` and `decisions.py`,
whose durable content is folded in below.

---

## 1. What is fitted

For every **(asset, date `t`, rung `L`)**, one OLS on the trailing window `[t−L+1, t]`:

```
y_i = a + b·u_i + c·u_i² + ε_i        y = log(adjusted close), centred per asset
                                       u_i = i − (L−1)/2,  i = 0…L−1
```

The design is **centred**, so `Σu = Σu³ = 0` and the basis `{1, u, u²}` is orthogonal.
Two consequences, and they are what make this a genuine two-axis object rather than a
reparametrisation:

* `b` is **numerically identical** in the linear and the quadratic fit (verified: Δ = 3.5e-18)
* `c` is the **exact orthogonal complement** of `b` — curvature is not contaminated by trend

Everything uses data ≤ `t`. López de Prado's trend scanning is a forward-looking
*labelling* method; this is the same machinery run backward so it can be traded.

**Speed.** The naive scan is O(T·ΣL). Each window is a rolling OLS updated in O(1) per bar
from four centred sums (`Σy, Σy², Σi·y, Σi²·y`), njit-compiled, parallel over assets, with
an exact recompute every 2048 bars to stop drift. 19M rows × 10 rungs in **3.4 s**; the
whole build, including the parquet write, in 41 s.

Recursions and moment formulas (`Su2 = L(L²−1)/12`, `Su4 = L(L²−1)(3L²−7)/240`) verified
symbolically and against `scipy.stats.linregress` / `numpy.linalg.lstsq` — max |Δ| ~1e-6,
which is float32 storage rounding.

## 2. What it outputs

`data/_staging/trendscan.parquet` — 19,003,721 rows × 38 columns.

```
GRID = [5, 10, 21, 42, 63, 84, 126, 168, 210, 252]     # one ladder, roughly geometric
```

| vector | is | from which fit |
|---|---|---|
| `tb_5 … tb_252` | `b / se(b)` — slope t | linear (df `L−2`) |
| `tc_5 … tc_252` | `c / se(c)` — curvature t | quadratic (df `L−3`) |
| `sd_5 … sd_252` | `√(SSE/(L−2))` — the fit's own residual vol | linear |

plus `security_id, date, close, adv21` and `fwd1/5/21/63` (targets, diagnostics only).

**One grid, no fast/slow.** A 5-day rung is just a rung (§3.3). Roughly geometric because
adjacent rungs are near-redundant — 117 daily-spaced windows measured N_eff 2.14, so fine
spacing buys columns, not information, and a dense grid measures worth exactly zero (§3.2).

**The build emits the surface and nothing else.** No skip, no argmax, no ladder mean, no
composite. Those are *reductions*, applied downstream against stored columns —
`skip()`, `argmax_abs()`, `rowmean()`, `rung()`. The surface is raw: `tb_L[t]` is the
window *ending at t*, and `skipped[t] == raw[t−k]`, so the 12−1 construction is one shift.

That is the load-bearing design choice. Every error found in the review was a reduction
baked into the build where it could not be questioned.

**Windows longer than an asset's history are skipped** rather than voiding the row.
Consequence: any reduction that averages across rungs averages a *varying* number of them,
and since |t| ∝ L^1.5 its scale tracks listing age — mean |trend| runs 4.12 (2 rungs) →
10.53 (12 rungs), monotone, with the cross-sectional rank compressed toward 0.5. Moot for
a single-rung reduction, which simply has no value until the rung fits (~5% of screened
rows).

## 3. Findings

Sample: 19,003,721 → **15,895,611** rows after a 1-bar lag and the screen (ADV21 ≥ $1M,
close ≥ $5), 6,622 dates, 5,566 securities, 2001–2026.

Convention, and it is stricter than everything that preceded it:

| | before | now |
|---|---|---|
| timing | signal(t) vs return **from t** — free fills at the signal's own close | **1-bar lag**: signal(t) earns t+1 → t+1+h, matching `BACKTEST.py` |
| NW bandwidth | `lag = h`, barely above the induced MA(h−1) | `⌈1.5h⌉` |

Both push t **down**. Reported ICs are cross-sectional Spearman, Newey-West t.

### The reductions

| reduction | IC_h1 | t | IC_h5 | t | IC_h21 | t | IC_h63 | t | turnover |
|---|---|---|---|---|---|---|---|---|---|
| **trend_top** `tb_252`, 12−1 | **0.0182** | **6.94** | 0.0222 | 5.30 | 0.0250 | 3.38 | **0.0343** | 2.82 | **0.0116** |
| trend_lad ladder mean | 0.0139 | 5.16 | 0.0194 | 4.46 | 0.0195 | 2.62 | 0.0309 | 2.86 | 0.0430 |
| trend_am argmax\|t\| | 0.0160 | 6.05 | 0.0184 | 4.40 | 0.0238 | 3.32 | 0.0324 | 3.04 | 0.0324 |
| ncurv_lad `−mean tc` | 0.0027 | 1.13 | 0.0126 | 3.91 | 0.0184 | 3.67 | 0.0187 | 2.46 | 0.1127 |
| ncurv_am | 0.0019 | 0.84 | 0.0089 | 2.93 | 0.0159 | 3.34 | 0.0132 | 1.86 | 0.0722 |
| nsd_lad `−mean sd` | 0.0134 | 4.29 | 0.0122 | 2.35 | 0.0102 | 1.06 | 0.0202 | 1.29 | 0.0145 |
| combo_ew (trend+curv) | 0.0125 | 6.01 | 0.0229 | 6.70 | **0.0263** | **4.93** | 0.0321 | 3.36 | — |

`corr(IC)` between the trend and curvature legs is **+0.131**, N_eff **1.95 / 2.00**. All
reductions are monotone in quintiles except `ncurv_am`.

### 3.1 The ladder mean loses to a single rung

`|t| ∝ L^1.5`, so a ladder mean is dominated by its top rung by construction
(corr +0.856). Paired against it, the single top rung is better at every horizon —
**+0.0046 (t 2.60)** at h1 — at **27% of the turnover**. `trend_lad` also collapses in
2009-16 (h21 IC 0.0014) where `trend_top` holds 0.0121.

The prior claim "ladder wins" was only ever measured against *argmax*, never against "use
the longest rung."

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

### 3.5 Weighting is the open question, not inclusion

Equal-weight rank composites, paired **against the trend leg alone** — a baseline that had
never been run:

| composite | ΔIC_h1 | t | ΔIC_h21 | t | ΔIC_h63 | t |
|---|---|---|---|---|---|---|
| trend + curv (ladder, no skip) | −0.0043 | **−3.89** | −0.0006 | −0.14 | −0.0029 | −0.52 |
| trend + curv (argmax, no skip) | −0.0041 | **−3.80** | −0.0004 | −0.10 | −0.0033 | −0.61 |

**Every equal-weight composite is significantly worse than its own trend leg at h1**, and
a wash at h21/h63. Equal weight is badly wrong at h1, where trend is ~9× stronger.

It is a weighting failure, not a signal failure. Curvature is a genuine near-orthogonal bet
that does the one thing trend cannot — **cover the 2009-16 hole**:

```
h21 IC by era        2000-2008   2009-2016   2017-2026
trend_top               0.0354     [0.0121]     0.0273
ncurv_lad               0.0214     [0.0191]     0.0155
combo_ew                0.0347     [0.0184]     0.0261
```

That is the post-GFC "CTA winter", and curvature is at its **strongest** there. At h21 the
blend buys +0.0063 in the weak era for −0.0007 and −0.0012 in the other two — a good trade
the full-sample mean does not price. At h1 the same blend is a −0.0035 (t −3.14) loss.

**Do not ship an equal-weight composite as the default.** The weight has never been fitted,
and it is probably horizon-dependent.

### 3.6 Scale (`sd`) is real, cheap, and a different anomaly

Flat across rungs (IC_h1 ≈ 0.011–0.013, t ≈ 4 at every rung), which is the expected shape:
there is no term structure to a volatility estimate. This is idiosyncratic vol falling out
of sums already computed — **the low-vol effect arriving free**, not trend scanning, and it
should not be described as such.

It is the only thing tested that improves the incumbent pair:

```
T+curv+sd − T+curv :  +0.0043 (t 3.52)   +0.0019   +0.0004   +0.0029
```

Significant at h1 only, turnover **0.0145** — second cheapest after the trend leg, and the
least correlated with curvature (−0.112). Weakness: h21 IC by era 0.0193 / **−0.0004** /
0.0115. It dies in exactly the era curvature carries; neither leg covers both.

### 3.7 `rz` (endpoint residual) — removed, with the algebra

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

---

## 4. Settled — do not retry

| | verdict |
|---|---|
| sign-sum heuristic (Σ sign over 4 windows) | dead short side: a −4 score predicts +9.9 bp vs +9.1 bp for 0. Beaten at every horizon at 5× the turnover |
| `zmom` (mean Δlog P / σ̂√L) | weakest thing tested; beats nothing |
| `curv_rel` = sign(trend) × curvature | ~0 at every horizon. Curvature is **not** "accelerating trends continue" — it is direction-blind second-derivative mean reversion |
| adjacent differences of the surface | N_eff 10.07, the highest breadth in the project, and **zero IC**. Differencing near-identical estimates decorrelates them because what is left is noise |
| more columns on the trend axis | 8 EWMA spans → N_eff 1.9; 24-col tensor → 1.9; 6 trend estimators → 1.10. Saturated |
| daily-spaced windows | 117 columns → N_eff 2.14 |
| ladder mean for trend | §3.1 — loses to a single rung |
| argmax / endogenous horizon | §3.2 — no measurable edge on a fixed grid |
| dense scan grid | §3.2 — worth exactly zero, and it was 89% of the scan |
| fast/slow leg split | §3.3 — the fast leg is free to drop |
| hardcoding the skip | §3.4 — asymmetric; it degrades curvature |
| equal-weight composite as default | §3.5 — loses to its own trend leg at h1 |
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

* **Everything is in-sample** over one 26-year panel, and this configuration is the survivor
  of dozens of comparisons scored on the same statistic. No CPCV, no holdout, no
  multiple-testing adjustment. The reversals in §3 are what decisions inside the noise band
  look like.
* **No costs.** Turnover is a proxy. The arbiter is `BACKTEST.py` with `spread_costs` and
  the IBKR preset, and it has never been run on any of this. It is the one result that can
  still invalidate the rest — the curvature leg at 0.1127 turnover is the exposure.
* **Weighting** (§3.5), per horizon.
* **Does the grid still stop too early?** Trend IC was still rising at the ceiling the last
  time the grid was cut, and that cost half the signal. Momentum is expected to decay past
  ~12m and reverse at 3–5y, so 252 is probably near the peak — but that is a prior, not a
  measurement. An 18M rung would settle it.
* **Delisting truncation.** `fwd{h}` is NaN in the final h bars of each security, so a name
  that delists has its terminal decline excluded from every h21/h63 IC. `BACKTEST.py` has
  the same hole (a name leaving `prices` is held flat, not written down).
* **`rz` from the quadratic fit** (§3.7).
* **The trend-scan label** — LdP as published, forward-looking — lifted target IC 1.9–3.5×
  over a fixed-horizon forward return. Still a live proposal for a model's target vector. It
  leaks up to `L_max` bars, so it needs purging **and a 252-bar embargo**.

## 7. Reproducing

```
python research/trendscan.py build     # surface → data/_staging/trendscan.parquet  (41 s)
python research/trendscan.py diag      # score the reductions in `reductions()`     (101 s)
python research/trendscan.py           # both
```

`diag` prints IC (with paired differences), quintile monotonicity, turnover, N_eff and
era stability. Reductions live in `reductions()` in `trendscan.py` — a visible dict,
deliberately not in the build. Edit it freely; nothing downstream depends on it.
