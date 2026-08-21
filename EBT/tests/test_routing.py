"""Tests for the routing tier: what gets selected, gated, and gradient."""
import math

import pytest
import torch

from ebt.attention import Attention, AttentionConfig

B, N, D, H = 2, 24, 32, 4


def cfg(**kw):
    kw.setdefault("d_model", D)
    kw.setdefault("n_heads", H)
    return AttentionConfig(**kw)


def _x(seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, N, D)


def _selected(att, x):
    valid = torch.ones(x.size(0), x.size(1), dtype=torch.bool)
    idx, gate, alive, stats = att._route(x, valid)
    return idx, gate, alive, stats


def test_topk_selects_exactly_capacity_tokens_in_order():
    att = Attention(cfg(routing="topk", capacity_ratio=0.25))
    idx, gate, alive, stats = _selected(att, _x())
    assert idx.shape == (B, H, att.capacity(N)) == (B, H, 6)
    assert (idx.diff(dim=-1) > 0).all()          # sorted, no duplicates
    assert alive.all() and stats["route_support"] == 6.0


def test_topk_picks_the_highest_scoring_tokens():
    att = Attention(cfg(routing="topk", capacity_ratio=0.25))
    x = _x()
    scores = att.router(x).transpose(1, 2)
    idx, _, _, _ = _selected(att, x)
    chosen = scores.detach().gather(-1, idx)
    rest = scores.detach().scatter(-1, idx, float("-inf"))
    assert (chosen.min(-1).values >= rest.max(-1).values - 1e-6).all()


def test_heads_specialise_on_different_tokens():
    att = Attention(cfg(routing="topk", capacity_ratio=0.25))
    idx, _, _, _ = _selected(att, _x())
    sets = [set(idx[0, h].tolist()) for h in range(H)]
    assert any(sets[i] != sets[j] for i in range(H) for j in range(i + 1, H))


def test_capacity_ratio_controls_the_block_width():
    for ratio, want in ((0.25, 6), (0.5, 12), (1.0, 24)):
        att = Attention(cfg(routing="topk", capacity_ratio=ratio))
        assert att.capacity(N) == want


def test_min_capacity_is_respected():
    att = Attention(cfg(routing="topk", capacity_ratio=0.01, min_capacity=4))
    assert att.capacity(N) == 4


def test_unselected_tokens_get_no_output_from_that_head():
    """Tokens a head skips must receive exactly zero contribution from it."""
    att = Attention(cfg(routing="topk", capacity_ratio=0.25, n_heads=1)).eval()
    x = _x()
    idx, _, _, _ = _selected(att, x)
    att.out_proj.weight.data = torch.eye(D)
    out = att(x)
    skipped = [i for i in range(N) if i not in idx[0, 0].tolist()]
    assert out[0, skipped].abs().max() == 0.0


def test_sparsemax_router_support_is_data_dependent():
    torch.manual_seed(3)
    att = Attention(cfg(routing="sparsemax"))
    with torch.no_grad():
        att.router.weight.mul_(3.0)              # moderately peaked routing scores
    _, _, alive, stats = _selected(att, _x())
    per_head = alive.sum(-1)
    assert stats["route_support"] < N
    assert per_head.float().std() > 0, "support size should vary across heads/sequences"


def test_sparsemax_router_drops_zero_probability_tokens():
    att = Attention(cfg(routing="sparsemax"))
    with torch.no_grad():
        att.router.weight.mul_(50.0)
    _, gate, alive, _ = _selected(att, _x())
    assert (gate[~alive] == 0).all()
    assert (gate[alive] > 0).all()


def test_sparsemax_router_gate_is_scale_normalised():
    """gate = p * |S| keeps the average gate near 1 whatever the support size."""
    att = Attention(cfg(routing="sparsemax", router_gate="mean"))
    _, gate, alive, _ = _selected(att, _x())
    assert 0.5 < float(gate[alive].detach().mean()) < 2.0
    raw = Attention(cfg(routing="sparsemax", router_gate="raw"))
    raw.router.weight.data = att.router.weight.data.clone()
    _, gate_raw, alive_raw, _ = _selected(raw, _x())
    assert float(gate_raw[alive_raw].detach().mean()) < float(gate[alive].detach().mean())


def test_topk_gives_no_gradient_to_unselected_tokens_but_sparsemax_router_is_smooth():
    """The differentiability claim, measured rather than asserted."""
    fracs = {}
    for routing in ("topk", "sparsemax"):
        att = Attention(cfg(routing=routing, capacity_ratio=0.25))
        x = _x()
        scores = att.router(x).transpose(1, 2)
        scores.retain_grad()
        att.retain_router_grad = True
        att(x).pow(2).sum().backward()
        g = att.last_router_scores.grad
        fracs[routing] = float((g != 0).float().mean())
    assert fracs["topk"] <= 0.25 + 1e-6, "top-k must not touch unselected tokens"
    assert fracs["sparsemax"] > 0.0


def test_router_gradient_flows_to_the_router_weights():
    for routing in ("topk", "sparsemax"):
        att = Attention(cfg(routing=routing))
        att(_x()).pow(2).sum().backward()
        assert att.router.weight.grad.abs().sum() > 0


def test_causal_block_attention_respects_original_positions():
    att = Attention(cfg(routing="topk", capacity_ratio=0.5, causal=True)).eval()
    x = torch.randn(1, N, D, requires_grad=True)
    idx, _, _, _ = _selected(att, x.detach())
    chosen = idx[0, 0].tolist()
    att(x)[0, chosen[1]].sum().backward()
    later = [i for i in chosen if i > chosen[1]]
    # gradient may still reach later tokens through *other* heads' routing, so
    # check the attention mask itself
    pos = idx
    allowed = att._allowed(pos, pos, torch.ones_like(pos, dtype=torch.bool))
    assert not allowed[0, 0, 1, 2:].any()
    assert allowed[0, 0, 1, :2].all()
    assert later  # sanity: there were later tokens to exclude


def test_coverage_is_below_one_for_routed_variants_and_one_for_dense():
    dense = Attention(cfg(routing="none"))
    dense(_x())
    assert dense.last_stats["token_coverage"] == 1.0
    routed = Attention(cfg(routing="topk", capacity_ratio=0.25))
    routed(_x())
    assert routed.last_stats["token_coverage"] < 1.0


def test_padded_tokens_are_never_routed():
    for routing in ("topk", "sparsemax"):
        att = Attention(cfg(routing=routing, capacity_ratio=0.5))
        x = _x()
        valid = torch.ones(B, N, dtype=torch.bool)
        valid[:, N // 2 :] = False
        idx, _, alive, _ = att._route(x, valid)
        assert (idx[alive] < N // 2).all()
