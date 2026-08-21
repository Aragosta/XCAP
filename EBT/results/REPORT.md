# Sparse attention bake-off: MoSA x sparsemax vs. softmax attention

* 2 seeds x 1500 steps, N=64, d_model=64, 4 heads, 2 layers, batch 32, lr 0.003
* routed variants get a capacity ratio of 0.25 (k = 16 of 64 tokens per head)
* every number is mean ± population std over seeds

## Variants

| variant             | what it is                                                 |
|---------------------|------------------------------------------------------------|
| baseline-softmax    | dense attention, softmax (control)                         |
| dense-sparsemax     | dense attention, sparsemax rows                            |
| mosa-softmax        | MoSA top-k routing + softmax (MoSA as published)           |
| mosa-sparsemax      | MoSA top-k routing + sparsemax (the two-tier proposal)     |
| smaxroute-softmax   | sparsemax router + softmax (router ablation)               |
| smaxroute-sparsemax | sparsemax router + sparsemax (fully differentiable sparse) |

## Accuracy (final eval)

| variant             | associative_recall | needle        | majority      |
|---------------------|--------------------|---------------|---------------|
| baseline-softmax    | 0.320 ± 0.023      | 0.357 ± 0.018 | 1.000 ± 0.000 |
| dense-sparsemax     | 0.340 ± 0.043      | 0.314 ± 0.002 | 1.000 ± 0.000 |
| mosa-softmax        | 0.123 ± 0.002      | 0.129 ± 0.027 | 0.316 ± 0.055 |
| mosa-sparsemax      | 0.121 ± 0.016      | 0.111 ± 0.010 | 0.223 ± 0.027 |
| smaxroute-softmax   | 0.189 ± 0.053      | 0.129 ± 0.027 | 0.188 ± 0.016 |
| smaxroute-sparsemax | 0.115 ± 0.021      | 0.182 ± 0.025 | 0.191 ± 0.012 |

## Loss (final eval)

| variant             | associative_recall | needle        | majority      |
|---------------------|--------------------|---------------|---------------|
| baseline-softmax    | 1.282 ± 0.064      | 1.313 ± 0.047 | 0.000 ± 0.000 |
| dense-sparsemax     | 1.239 ± 0.029      | 1.395 ± 0.088 | 0.009 ± 0.004 |
| mosa-softmax        | 2.080 ± 0.000      | 2.081 ± 0.002 | 1.812 ± 0.089 |
| mosa-sparsemax      | 2.081 ± 0.001      | 2.078 ± 0.003 | 1.992 ± 0.030 |
| smaxroute-softmax   | 1.993 ± 0.087      | 2.080 ± 0.001 | 2.053 ± 0.023 |
| smaxroute-sparsemax | 2.079 ± 0.001      | 2.026 ± 0.053 | 2.056 ± 0.004 |

## Sample efficiency (steps to 90% accuracy, '-' = never reached)

| variant             | associative_recall | needle | majority         |
|---------------------|--------------------|--------|------------------|
| baseline-softmax    | -                  | -      | 250 (2/2 seeds)  |
| dense-sparsemax     | -                  | -      | 1000 (2/2 seeds) |
| mosa-softmax        | -                  | -      | -                |
| mosa-sparsemax      | -                  | -      | -                |
| smaxroute-softmax   | -                  | -      | -                |
| smaxroute-sparsemax | -                  | -      | -                |

## Mechanism diagnostics (averaged over tasks and seeds)

`attn_zero_frac`: share of *exactly zero* attention weights inside the block. `attn_support`: non-zero weights per query row. `token_coverage`: share of tokens picked by at least one head. `route_support`: tokens per head after routing. `router_grad_frac`: share of router logits receiving non-zero gradient.

| variant             | attn_zero_frac | attn_support    | attn_entropy  | token_coverage | route_support  | route_support_std | router_grad_frac |
|---------------------|----------------|-----------------|---------------|----------------|----------------|-------------------|------------------|
| baseline-softmax    | 0.000 ± 0.000  | 63.982 ± 0.015  | 2.530 ± 1.178 | 1.000 ± 0.000  | 64.000 ± 0.000 | 0.000 ± 0.000     | -                |
| dense-sparsemax     | 0.790 ± 0.228  | 13.445 ± 14.588 | 1.501 ± 1.195 | 1.000 ± 0.000  | 64.000 ± 0.000 | 0.000 ± 0.000     | -                |
| mosa-softmax        | 0.000 ± 0.000  | 15.999 ± 0.001  | 1.868 ± 0.253 | 0.664 ± 0.052  | 16.000 ± 0.000 | 0.000 ± 0.000     | 0.015 ± 0.011    |
| mosa-sparsemax      | 0.597 ± 0.285  | 6.455 ± 4.553   | 1.214 ± 0.779 | 0.624 ± 0.106  | 16.000 ± 0.000 | 0.000 ± 0.000     | 0.013 ± 0.016    |
| smaxroute-softmax   | 0.000 ± 0.000  | 9.788 ± 2.887   | 1.975 ± 0.407 | 0.354 ± 0.100  | 8.389 ± 2.670  | 3.301 ± 1.016     | 0.002 ± 0.003    |
| smaxroute-sparsemax | 0.223 ± 0.138  | 8.211 ± 7.136   | 1.364 ± 0.787 | 0.334 ± 0.181  | 6.791 ± 4.161  | 4.012 ± 3.408     | 0.022 ± 0.045    |

## Cost (as trained: N=64, batch 32, CPU)

| variant             | fwd ms/batch | fwd+bwd ms/batch | MFLOPs/seq | attn matrix MB | rel fwd |
|---------------------|--------------|------------------|------------|----------------|---------|
| baseline-softmax    | 17.7         | 50.3             | 7.4        | 4.2            | 1.01x   |
| dense-sparsemax     | 46.8         | 73.1             | 7.4        | 4.2            | 2.67x   |
| mosa-softmax        | 8.8          | 22.4             | 6.4        | 0.3            | 0.50x   |
| mosa-sparsemax      | 10.1         | 24.8             | 6.4        | 0.3            | 0.57x   |
| smaxroute-softmax   | 10.3         | 24.9             | 6.4        | 0.1            | 0.59x   |
| smaxroute-sparsemax | 12.4         | 27.0             | 6.4        | 0.1            | 0.70x   |

## Scaling with sequence length (forward ms/batch, batch 8)

| variant             | N=128 | N=256 | N=512 |
|---------------------|-------|-------|-------|
| baseline-softmax    | 13.3  | 48.7  | 199.3 |
| dense-sparsemax     | 17.9  | 80.4  | 497.8 |
| mosa-softmax        | 6.5   | 10.0  | 14.9  |
| mosa-sparsemax      | 10.1  | 14.4  | 42.5  |
| smaxroute-softmax   | 13.0  | 10.4  | 15.0  |
| smaxroute-sparsemax | 7.4   | 11.9  | 14.0  |

Attention-matrix memory (MB):

| variant             | N=128 | N=256 | N=512 |
|---------------------|-------|-------|-------|
| baseline-softmax    | 4.2   | 16.8  | 67.1  |
| dense-sparsemax     | 4.2   | 16.8  | 67.1  |
| mosa-softmax        | 0.3   | 1.0   | 4.2   |
| mosa-sparsemax      | 0.3   | 1.0   | 4.2   |
| smaxroute-softmax   | 0.0   | 0.1   | 0.1   |
| smaxroute-sparsemax | 0.0   | 0.1   | 0.1   |

