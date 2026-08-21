"""Named configurations under test."""

from __future__ import annotations

from dataclasses import replace

from .attention import AttentionConfig

_NAMES = {
    ("dot", "softmax"): "dot-softmax",          # the control: standard attention
    ("dot", "sparsemax"): "dot-sparsemax",      # exact-zero rows, same score
    ("energy", "softmax"): "energy-softmax",    # E = ||q-k||^2, competitive gate
    ("energy", "sigmoid"): "energy-sigmoid",    # E = ||q-k||^2, independent gates
    ("transe", "softmax"): "transe-softmax",    # relational: g_r(z) = z + r
    ("transe", "sigmoid"): "transe-sigmoid",
    ("rotate", "softmax"): "rotate-softmax",    # relational: g_r(z) = z o r
}

# tied-projection energy attention: the formulation the Lipschitz analysis of
# Kim et al. (ICML 2021) actually studies.  Named separately because tying
# removes a projection, so it is not parameter-matched to the rest.
_TIED = {"energy-softmax-tied": ("energy", "softmax"), "dot-softmax-tied": ("dot", "softmax")}

DEFAULT_GRID = ["dot-softmax", "energy-softmax", "energy-sigmoid",
                "transe-softmax", "transe-sigmoid", "rotate-softmax"]


def all_variants(d_model: int = 128, n_heads: int = 4, causal: bool = False,
                 n_relations: int = 4) -> list[AttentionConfig]:
    base = AttentionConfig(d_model=d_model, n_heads=n_heads, causal=causal,
                           n_relations=n_relations)
    out = [replace(base, score=sc, gate=g, name=name) for (sc, g), name in _NAMES.items()]
    out += [replace(base, score=sc, gate=g, tie_qk=True, name=name)
            for name, (sc, g) in _TIED.items()]
    return out


def variant(name: str, **kwargs) -> AttentionConfig:
    for cfg in all_variants(**kwargs):
        if cfg.name == name:
            return cfg
    raise ValueError(f"unknown variant {name!r}; expected one of {VARIANT_NAMES}")


VARIANT_NAMES = list(_NAMES.values()) + list(_TIED)
