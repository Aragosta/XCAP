"""Match the *effective temperature* of different score functions at init.

Why this exists
---------------
Ramsauer et al.'s modern Hopfield energy,

    E(xi) = -lse(beta, X^T xi) + 1/2 xi^T xi + const,

has an attractor landscape that depends sharply on the inverse temperature
beta: below a critical value there is a single global attractor (all patterns
average together), above it each pattern gets its own basin.  Attention is one
CCCP step of that energy, so *beta decides what attention can do at all*.

Now compare score magnitudes at initialisation, for unit-variance projections:

    dot     q.k          ~ O(sqrt(d))
    energy  -||q-k||^2   ~ O(2d)

Divide both by sqrt(d_head) with the same temperature -- which is what the
literature's scaling and my first run both did -- and the energy variant starts
at a systematically *higher* beta than the dot variant.  A benchmark run that
way compares two points on the phase diagram, not two score functions.

So: before training, binary-search each layer's temperature so that the mean
attention-row entropy at initialisation hits a common target.  Every variant
then starts with the same sharpness and the temperature stays learnable from
there.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .attention import Attention


@torch.no_grad()
def _entropy_of_layer(model: nn.Module, atts: list[Attention], i: int,
                      x: Tensor, log_temp: float) -> float:
    """Entropy of layer i's attention after a full forward at this temperature.

    A full forward matters: layer i sees whatever layer i-1 produced, so the
    layers are calibrated in order, each on the input the previous one makes.
    """
    atts[i].log_temp.fill_(log_temp)
    model(x)
    return atts[i].last_stats["attn_entropy"]


@torch.no_grad()
def calibrate_temperature(model: nn.Module, x: Tensor, target_frac: float = 0.9,
                          lo: float = -8.0, hi: float = 8.0, iters: int = 24) -> dict[str, float]:
    """Set every attention layer's temperature so init entropy = target_frac * log(N).

    ``x`` is a batch of token ids.  Entropy decreases monotonically in the
    temperature, so a bisection is exact enough.  Returns the chosen
    log-temperature per layer.
    """
    was_training = model.training
    model.eval()
    target = target_frac * math.log(x.size(1))
    atts = [m for m in model.modules() if isinstance(m, Attention)]
    chosen = {}
    for i in range(len(atts)):
        a, b = lo, hi
        if _entropy_of_layer(model, atts, i, x, a) < target:
            chosen[f"layer{i}"] = a
        elif _entropy_of_layer(model, atts, i, x, b) > target:
            chosen[f"layer{i}"] = b
        else:
            for _ in range(iters):
                mid = 0.5 * (a + b)
                if _entropy_of_layer(model, atts, i, x, mid) > target:
                    a = mid
                else:
                    b = mid
            chosen[f"layer{i}"] = 0.5 * (a + b)
        atts[i].log_temp.fill_(chosen[f"layer{i}"])
    model(x)
    if was_training:
        model.train()
    return chosen
