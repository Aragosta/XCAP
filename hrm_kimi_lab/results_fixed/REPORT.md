# HRM x Kimi linear attention — small-scale results

Char-level LM on tiny Shakespeare (1.1 MB real text, val = disjoint final 10%). d_model=96, heads=4, seq_len=192, batch=8, 650 steps (998,400 tokens — identical budget for every variant), lr=0.003. CPU only.

`blocks/fwd` = block forward passes per token per model forward (compute proxy); `unique` = distinct blocks holding parameters; `grad cov` = fraction of those block applications that receive gradients at the end of training. HRM's 1-step gradient buys O(1) memory at the cost of grad cov < 1, so recurrent rows are not gradient-comparable to the plain baselines (grad cov 1.00) -- the `*_fullbp` twins are.

| Variant | Val bits/char | seeds | Params | Active | blocks/fwd | unique | grad cov | tok/s | Train s |
|---|---|---|---|---|---|---|---|---|---|
| base_hybrid_kda_mhamoe_x | **2.3362** ±0.0042 | 3 | 1,730,508 | 845,772 | 4 | 4 | 1.00 | 2,215 | 451 |
| hrm_hybrid_kda_mhamoe_fullbp | **2.3811** ±0.0087 | 3 | 868,324 | 425,956 | 6 | 2 | 1.00 | 1,628 | 613 |
| hrm_loop5_kda_mhamoe_fullbp | **2.4161** ±0.0095 | 3 | 868,324 | 425,956 | 15 | 2 | 1.00 | 569 | 1754 |

## What each variant is

- **base_hybrid_kda_mhamoe_x** — No recurrence: Kimi's 3:1 linear/full-attention stack, MoE throughout. The reference an HRM module has to beat.
- **hrm_hybrid_kda_mhamoe_fullbp** — HRM module, fast(L)=Kimi linear, slow(H)=MHA, MoE in both, full gradients.
- **hrm_loop5_kda_mhamoe_fullbp** — The 5-loop version of the above with gradients through all 15 applications: does looping to 5 help once truncation is removed?

## Samples (seed 0, temperature 0.8)

### base_hybrid_kda_mhamoe_x (bpc 2.336)

```
KING RICHARD:
Thou must there, lead?
How near you do I the father fairy hath were:
Our justice of loves. I have that of your come
Your senate forfend there the executy one.

Shepherd:
And the concling.

First March
```

### hrm_hybrid_kda_mhamoe_fullbp (bpc 2.381)

```
KING RICHARD:
Thou make these nut medicateed, and in I the can you are
In that here hath this being some one.

GLOUCESTER:
With we all this day true: there thou well you for my his hither.

DUCHESS OF YORK:
But for
```

### hrm_loop5_kda_mhamoe_fullbp (bpc 2.416)

```
KING RICHARD:
Thou make the prelenced canneam'd you.
I to the Mecome with than here hath think
And sensent him some so speaking wear.
As your foer not my father we would for me I men,
And the concling. Why she. So 
```
