"""The two configurations under test."""

from __future__ import annotations

from dataclasses import replace

from .attention import AttentionConfig

_NAMES = {"softmax": "baseline-softmax", "sparsemax": "sparsemax"}


def all_variants(d_model: int = 128, n_heads: int = 4, causal: bool = False) -> list[AttentionConfig]:
    base = AttentionConfig(d_model=d_model, n_heads=n_heads, causal=causal)
    return [replace(base, normaliser=nrm, name=name) for nrm, name in _NAMES.items()]


def variant(name: str, **kwargs) -> AttentionConfig:
    for cfg in all_variants(**kwargs):
        if cfg.name == name:
            return cfg
    raise ValueError(f"unknown variant {name!r}; expected one of {VARIANT_NAMES}")


VARIANT_NAMES = list(_NAMES.values())
