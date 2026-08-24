# HRM x Kimi linear attention — small-scale architecture test

A self-contained, CPU-sized experiment comparing Sapient's **HRM** recurrent
hierarchy against **Kimi Delta Attention** (Kimi linear attention) and Kimi's
**Stable LatentMoE**, on real text (char-level tiny Shakespeare, 1.1 MB).

Everything trains in minutes on 4 CPU cores; the point is a like-for-like
ranking of architectures at a fixed token budget, not absolute quality.

## Where the code comes from

| Piece | Origin | Changes |
|---|---|---|
| `lab/layers.py` | [sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text) `models/layers.py`, `models/transformer.py` | FlashAttention → `F.scaled_dot_product_attention` (no CUDA here). Gated MHA, RoPE, truncated-LeCun init and SwiGLU shapes are upstream's. |
| `lab/model.py` `HRMCore` | HRM-Text `models/baselines/hrm_nocarry_bp_warmup.py` | Same H/L cycle structure, same 1-step-gradient + backprop-warmup policy; H/L levels may now use different block types. |
| `vendor/src/kda/`, `vendor/src/kimi_primitives/` | [pablo-reyes8/kimi-k3-pytorch](https://github.com/pablo-reyes8/kimi-k3-pytorch) `src/` | Vendored verbatim. |
| `vendor/src/stable_latent_moe/`, `vendor/src/transformer_modules/rms_norm.py` | idem | Vendored verbatim; configured small (8 routed experts, top-2, 1 shared). |

## Vocabulary

- **H level = "slow" state**, **L level = "fast" state** (HRM terminology).
- `mha` = HRM-Text gated softmax attention; `kda` = Kimi Delta Attention (linear).
- `dense` = SwiGLU FFN; `moe` = Kimi Stable LatentMoE.

## Run

```bash
cd hrm_kimi_lab
export PYTHONPATH=.:vendor
python3 run_all.py --steps 700               # all variants, 2 at a time
python3 -m lab.loop_scaling                  # test-time loop scaling of the HRM runs
python3 -m lab.report                        # -> results/REPORT.md
```

Single variant: `python3 -m lab.train hrm_loop5_kda_mhamoe --steps 700`.

Results land in `results/*.json` (loss curve, params, throughput, a sample) and
`results/REPORT.md`.
