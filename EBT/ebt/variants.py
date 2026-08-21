"""The 2 x 3 grid under test: {softmax, sparsemax} x {dense, top-k MoSA, sparsemax router}."""

from __future__ import annotations

from dataclasses import replace

from .attention import AttentionConfig

_NAMES = {
    ("none", "softmax"): "baseline-softmax",
    ("none", "sparsemax"): "dense-sparsemax",
    ("topk", "softmax"): "mosa-softmax",
    ("topk", "sparsemax"): "mosa-sparsemax",
    ("sparsemax", "softmax"): "smaxroute-softmax",
    ("sparsemax", "sparsemax"): "smaxroute-sparsemax",
}


def all_variants(d_model: int = 128, n_heads: int = 4, capacity_ratio: float = 0.25,
                 causal: bool = False) -> list[AttentionConfig]:
    base = AttentionConfig(d_model=d_model, n_heads=n_heads,
                           capacity_ratio=capacity_ratio, causal=causal)
    return [replace(base, routing=r, normaliser=nrm, name=name)
            for (r, nrm), name in _NAMES.items()]


def variant(name: str, **kwargs) -> AttentionConfig:
    for cfg in all_variants(**kwargs):
        if cfg.name == name:
            return cfg
    raise ValueError(f"unknown variant {name!r}; expected one of {[c.name for c in all_variants()]}")


VARIANT_NAMES = list(_NAMES.values())
