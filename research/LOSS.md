# The loss — XCAP v0

The single record for `research/loss_test.py`. Companion to `TRENDSCAN.md` (what the
features are); this records what is done with them, and what has been killed. **Replaces
its previous 673-line version**, which argued for a differentiable economic loss, a horizon
vector and a Gârleanu–Pedersen trade rule — all built, all measured, **none survived.**

## 1. What the harness is

It stopped being a learning system and became a **falsification harness**. Six methods do
the work; each caught something reasoning alone had not. Every result below is unfitted.

| method | what it caught |
|---|---|
| `BACKTEST.py` as sole arbiter, **surrogate gap reported** | 59% of training cost was charged on dates the arbiter charges nothing. Gap now −0.017 / +0.004 |
| **exact numerical proofs on the position map** | the DMN map's loss swung 3× over a 64× gross change worth **0.004 Sharpe** |
| **placebo controls** (random groups, same sizes) | sector's edge is shrinkage, not information — random sectors match or beat real ones |
| **alpha / IR reported beside Sharpe** | the same trap 3×: inverse-vol, curvature and `tsv` raise Sharpe by *de-risking* while killing active return |
| **unfitted baseline in every table** | fitting loses — IC worse than raw `tb_252` in 9/10 CPCV splits, every mode |
| **deflated Sharpe** | the raw signal sits inside the noise of the search that found it |

Validation frame: CPCV, 5 groups, 10 splits, 252-bar purge, 21-bar embargo.

## 2. Proved exactly (`proof`) — `w = z/‖z‖₁`, `z = s−mean(s)`; long-only `relu(z)/‖relu(z)‖₁`

| | | measured |
|---|---|---|
| P1 | `w(λs) = w(s)` — score scale is an **exact** null; no gradient is spent on it | 2.2e-19; `dL/dλ` −5.7e-21 (DMN −9.85e-06) |
| P2 | `Σw = 0`, `Σ\|w\| = 1` — leverage and market tilt are not free parameters | 5e-17 / 1.000000000000 |
| P2' | `w_lo = 2·relu(w_ls)` — long-only **is** the neutral book minus its short leg | 4.3e-19 |
| P3 | `s → s + c_t` leaves `w` unchanged — date-level features are annihilated | 1.5e-18 |
| P5 | block gradients unbiased; a batch-Sharpe loss is not | cos 0.9937 vs 0.9678 |

## 3. What survived

**NYSE + NASDAQ common stock**, ADV21 ≥ $1M, close ≥ $5, no top-N cap. 16,983,991 rows,
5,537 securities, median 2,575/date, 2000-01→2026-07, delisted included.

```
LONG-ONLY  vol-neutralised tb_252 (12-1), eq-wt top 50%, monthly
           ret +15.03%  vol 22.12%  Sharpe 0.744  IR vs EW +0.527  tno 2.01
           equal-weight benchmark  +12.72%  21.78%  0.659
NEUTRAL    tb_252 (12-1), gross-1 dollar-neutral, Barroso vol overlay (tgt 0.040, cap 3)
           ret  +2.04%  vol  4.53%  Sharpe 0.468  alpha Sh +0.699  tno 2.05
           maxDD -15.8%  skew -0.40    unscaled: 0.261 / 0.576 / -23.7% / -1.18

deflated (T=26.6yr, sd 0.194)      N=25     N=50    N=100
   long-only  IR    +0.527        76.4%    67.0%    57.3%
   neutral    alpha +0.699        94.6%    90.7%    85.8%
```

**The neutral book is the stronger result** — long-only's 0.744 Sharpe is mostly equity
beta, and against its proper benchmark it is barely a coin flip past N=100. Four changes:

* **`tb` carries no magnitude.** `t² = (L−2)R²/(1−R²)`, so `tb = sign(b)·g(R²)`, pure path
  straightness — and it tilts −0.209 with log vol, into the tercile where §9.8 measured trend
  IC 0.0108 against 0.0310 in the top. Neutralising: IR **+0.313 → +0.527**, robust to form
  (linear 0.527 / quadratic 0.515 / rank-in-vol-decile 0.522).
* **Barroso–Santa-Clara vol overlay**, neutral book only: Sharpe 0.261 → 0.468, skew −1.18 →
  −0.40, **all of it from 2005-2010** (−0.504 → −0.064). Insensitive across settings.
* **The 12−1 skip was missing** — `trendscan.py` applies `SKIP=21` in `reductions()`,
  `loss_test.py` never did. IC t 2.78 → 3.09. **NASDAQ** adds 2,501 names: IC 0.0242, t 3.43.

## 4. Settled — do not retry

| | verdict |
|---|---|
| differentiable economic loss, GP rule, horizon vector | never beat an unfitted column — the whole §1–§5 programme of the old file |
| the fitted model (`Structured` 10p, `Dense` 20p) | IC worse than raw `tb_252` in 9/10 splits, every mode. It beat only a bad position map |
| DMN map (`tanh(s)·σ_tgt/σ_i`, floating gross) | alpha 0.317 vs gross-1's 0.695; strong gradient on a quantity worth 0.004 Sharpe |
| **cross-sectional 1/σ** — four independent tests | `tsv` IR −0.362; every quantile −0.09..−0.12 against equal weight's +0.10..+0.33. Its Sharpe gain is de-risking; active return ≈ 0 |
| curvature — as blend, sleeve and gate | standalone alpha −0.246 at 6.2 turnover; blends lower alpha; the gate works only *down*-curving (+0.335 vs +0.018 up) and is dominated by the vol overlay on every axis |
| sector | loses to its random-sector placebo (4/10 `lo`, 1/10 `tsv`). `Unknown` is a death flag — 98.7% of live names have a sector, 33.5% of delisted |
| fundamentals | 100% file coverage, but quarterly financials 98.3% active / 59.5% delisted, 9% for pre-2006 exits. Unusable before ~2015 |
| conviction weighting on **raw** `tb` | IR +0.045 vs equal weight's +0.306 — raw `tb`'s top decile is flat (d7 +2.38% > d10 +1.65%). It does pay after neutralisation (+0.563 at q=0.10) but at 6.2 turnover for the same IR as q=0.50 at 2.0 |
| longer holding periods | IR +0.306 / +0.146 / +0.008 at 21 / 63 / 126 bars |
| dollar bars at daily resolution | kurtosis 16.78 → 16.45, abs-return autocorrelation +0.231 → **worse**. Needs intraday: the daily floor gives only the aggregation half and adds duration heteroskedasticity |
| the magnitude leg `b·L` = `tb·sd_L·L/√Su2` | IR +0.284 alone, +0.389 neutralised — loses to neutralised `tb` at +0.527 |

## 5. Broken and open, ranked

1. **Nothing is validated out of sample** — ~40 decisions on one panel. Both configs are
   unfitted, so walk-forward is cheap. There is no excuse.
2. **`impact_coef = 0`**, and every improvement bought turnover (1.38 → 2.05).
3. **`borrow_fee = 0`** in every run; the neutral sleeve shorts high-vol small caps.
4. **Delisting returns** — a departed name is held flat, not written down: upward bias on a
   long-only book. Flagged in `TRENDSCAN.md` §6, never fixed.
5. **float32 saturation ties** — `tanh(x/0.25·MAD)` rounds to 1.0 above ~2 MADs, so top-of-
   bucket membership is arbitrary (0.036 IR). Rank maps need no saturation: feed raw `tb`.
6. **The vol fix is downstream.** Daily floor fixed ⇒ the principled version is a HAC or
   robust-scale `se(b)` **inside `trendscan.py`**; the residualisation estimates exactly that.
7. **One cell of a ten-rung surface** — whether the ladder carries anything *after*
   neutralisation is unmeasured, and neutralisation moved the top decile +1.65% → +5.66%.
8. **Era instability** — long-only IR +0.84 / −0.46 / +0.77 / −0.23 / +0.83 by block.
9. **End-to-end learning is orphaned**, probably unrecoverably: a rank/bucket map has zero
   gradient almost everywhere, so the winning construction cannot be trained through.

## 6. Reproducing

`prep NYSE,NASDAQ 21` (~3 min) · `proof` (~1 min) · `ls`|`lo`|`tsv` CPCV (~40-80 min) ·
`look` (construction + factor attribution). `configure(venues, skip)` parameterises the
universe; each combination writes its own cache.
