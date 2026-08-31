# Results

Every number here comes from the real FlyWire v783 connectome or from training runs on real text (Project Gutenberg, with held-out books, Brown and Reuters as increasingly distant OOD sets). Baselines are degree-preserving rewirings or hot-limit S1 samples, never Erdos-Renyi.


## M1 - geometric temperature

Full >=5-synapse graph, giant component: N=132483, <k>=37.88, mean local clustering=0.1628, gamma=3.135 (k_min=195, KS=0.0153).

| subsample | n | <k> | clustering | gamma | beta_fit | status |
|---|---|---|---|---|---|---|
| 8000 (rep 0) | 5386 | 3.24 | 0.1396 | 3.366 | **1.315** | converged |
| 8000 (rep 1) | 5448 | 3.37 | 0.1335 | 3.104 | **1.294** | converged |
| 16000 (rep 0) | 13195 | 5.48 | 0.1293 | 3.291 | **1.213** | converged |
| 16000 (rep 1) | 13454 | 5.79 | 0.1368 | 3.037 | **1.213** | converged |
| 32000 (rep 0) | 29629 | 9.97 | 0.1510 | 3.163 | **1.213** | converged |
| 32000 (rep 1) | 29595 | 10.26 | 0.1575 | 3.128 | **1.220** | converged |

**beta = 1.245 +- 0.043** across scales -> quasi-geometric (beta_c = 1, beta = 2D = 2).


## M2 - Balanced Forman curvature

20000 edges sampled from 2509503, 3 degree-preserving rewirings.

| | mean | sd | median | frac negative | 1st pct | 5th pct |
|---|---|---|---|---|---|---|
| FlyWire | 1.4230 | 1.4283 | 1.1531 | 0.1219 | -0.6284 | -0.3081 |
| degree-preserving null | 2.8066 | 2.6540 | 2.2702 | 0.0862 | -1.7667 | -0.3487 |

Difference in mean curvature -1.3836 (z = -96.6 against the spread of null replicate means); negative-edge fraction +0.0357.


## M3 - asymmetry census and rich club

| min total degree | neurons | broadcasters | integrators | asymmetric fraction of brain |
|---|---|---|---|---|
| 0 | 134181 | 13754 | 7785 | 0.1605 |
| 5 | 120435 | 6548 | 4659 | 0.0835 |
| 10 | 106528 | 4244 | 2348 | 0.0491 |
| 20 | 73518 | 2046 | 1519 | 0.0266 |
| 30 | 52244 | 1304 | 1183 | 0.0185 |
| 50 | 27918 | 680 | 782 | 0.0109 |
| 100 | 9984 | 174 | 394 | 0.0042 |

Normalised rich-club coefficient phi(k)/phi_null(k):

| k | phi | phi normalised |
|---|---|---|
| 21 | 0.00086 | 0.9949 |
| 22 | 0.00091 | 0.9954 |
| 23 | 0.00096 | 0.9983 |
| 24 | 0.00100 | 0.9974 |
| 26 | 0.00110 | 1.0025 |
| 27 | 0.00115 | 1.0055 |
| 29 | 0.00126 | 1.0124 |
| 30 | 0.00131 | 1.0158 |
| 32 | 0.00143 | 1.0244 |
| 33 | 0.00149 | 1.0257 |
| 35 | 0.00162 | 1.0312 |
| 37 | 0.00177 | 1.0351 |
| 40 | 0.00200 | 1.0383 |
| 42 | 0.00216 | 1.0397 |
| 45 | 0.00241 | 1.0429 |
| 48 | 0.00266 | 1.0385 |
| 50 | 0.00283 | 1.0376 |
| 52 | 0.00301 | 1.0386 |
| 57 | 0.00344 | 1.0362 |
| 63 | 0.00401 | 1.0485 |
| 69 | 0.00467 | 1.0675 |
| 75 | 0.00534 | 1.0678 |
| 78 | 0.00566 | 1.0545 |
| 90 | 0.00698 | 1.0092 |
| 100 | 0.00809 | 0.9827 |
| 108 | 0.00891 | 0.9588 |
| 139 | 0.01203 | 0.8857 |
| 150 | 0.01338 | 0.8710 |
| 194 | 0.01920 | 0.7946 |
| 200 | 0.02011 | 0.7871 |
| 300 | 0.03172 | 0.5845 |
| 500 | 0.04476 | 0.3113 |

## T1 - pruning and the shuffling threshold

W_V/W_O parameters: 131072. Dense val loss 1.6131 (hard 2.9568). No retraining.

| density | kept | trained | shuffled_w | shuffled_mask | random_both | trained - shuffled_w |
|---|---|---|---|---|---|---|
| 1 | 131072 | 1.6131 | 4.7752 | 1.6131 | 5.0433 | +3.1621 |
| 0.3 | 39322 | 2.4434 | 4.8758 | 4.9073 | 4.9445 | +2.4324 |
| 0.1 | 13107 | 4.7109 | 4.9791 | 5.2865 | 5.2942 | +0.2682 |
| 0.05 | 6554 | 5.2275 | 5.2544 | 5.3262 | 5.3439 | +0.0269 |
| 0.02 | 2621 | 4.7461 | 5.2871 | 5.3080 | 5.1415 | +0.5410 |
| 0.01 | 1311 | 5.3143 | 5.3691 | 5.3261 | 5.3164 | +0.0548 |
| 0.005 | 655 | 5.4595 | 5.3483 | 5.3172 | 5.2387 | -0.1112 |
| 0.002 | 262 | 5.3528 | 5.2771 | 5.2367 | 5.3066 | -0.0758 |
| 0.001 | 131 | 5.3153 | 5.3320 | 5.1858 | 5.1863 | +0.0167 |
| 0.0005 | 66 | 5.2684 | 5.2049 | 5.2828 | 5.3152 | -0.0635 |
| 0.0002 | 26 | 5.3028 | 5.3028 | 5.2349 | 5.2983 | +0.0000 |
| 0.0001 | 13 | 5.3028 | 5.3028 | 5.3028 | 5.3028 | +0.0000 |

## T2 - key-norm bias

| arm | seeds | val | val_hard | ood_book | ood_brown | ood_reuters |
|---|---|---|---|---|---|---|
| euclid | 2 | 1.6800 ± 0.0053 | 3.0855 ± 0.0078 | 1.6665 ± 0.0061 | 1.9926 ± 0.0095 | 2.4520 ± 0.0225 |
| keybias | 2 | 1.6769 ± 0.0112 | 3.0817 ± 0.0195 | 1.6660 ± 0.0061 | 1.9970 ± 0.0043 | 2.4718 ± 0.0324 |
| qk | 2 | 1.6855 ± 0.0051 | 3.0919 ± 0.0071 | 1.6658 ± 0.0007 | 2.0116 ± 0.0005 | 2.4633 ± 0.0035 |
