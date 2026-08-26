"""Model- and MoE-level correctness, including cross-arm parameter parity."""

import pytest
import torch

from hmha.model import ModelConfig, MoETransformer
from hmha.moe import MoEFeedForward

torch.manual_seed(0)


def _cfg(**kw):
    base = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16)
    base.update(kw)
    return ModelConfig(**base)


def _batch(vocab=64, b=2, t=16):
    x = torch.randint(0, vocab, (b, t))
    return x, torch.randint(0, vocab, (b, t))


# --------------------------------------------------------------------- MoE


def test_moe_shape_and_aux_loss():
    moe = MoEFeedForward(32, 64, n_routed_experts=4, top_k=2)
    out = moe(torch.randn(2, 8, 32))
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(moe.last_aux_loss)
    assert moe.last_aux_loss > 0


def test_moe_routes_sparsely_but_covers_every_token():
    """Every token must be produced by exactly top_k routed experts + the shared one."""
    moe = MoEFeedForward(32, 64, n_routed_experts=4, n_shared_experts=0, top_k=2)
    x = torch.randn(4, 8, 32)
    with torch.no_grad():
        out = moe(x)
    # with no shared expert and gates summing to 1, no token may come out empty
    assert (out.abs().sum(dim=-1) > 0).all()
    assert moe.last_expert_fractions.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_balanced_routing_has_higher_entropy_than_collapsed():
    moe = MoEFeedForward(32, 64, n_routed_experts=4, top_k=2)
    moe(torch.randn(4, 16, 32))
    balanced = moe.expert_utilisation_entropy()

    with torch.no_grad():  # force the router onto a single expert
        moe.router.weight.zero_()
        moe.router.weight[0] = 10.0
    moe(torch.randn(4, 16, 32))
    assert moe.expert_utilisation_entropy() < balanced


def test_moe_rejects_top_k_larger_than_experts():
    with pytest.raises(ValueError):
        MoEFeedForward(32, 64, n_routed_experts=2, top_k=4)


# ------------------------------------------------------------------- model


@pytest.mark.parametrize("attention", ["euclidean", "hyperbolic"])
def test_forward_shapes_and_loss(attention):
    cfg = _cfg(attention=attention)
    model = MoETransformer(cfg)
    x, y = _batch(cfg.vocab_size)
    out = model(x, y)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
    assert out["loss"] > 0


def test_arms_have_matched_parameter_counts():
    """The controlled-experiment guarantee, asserted at model level."""
    euc = MoETransformer(_cfg(attention="euclidean")).param_counts()
    hyp = MoETransformer(_cfg(attention="hyperbolic")).param_counts()
    assert hyp["trainable"] == euc["trainable"]
    # frozen per-head curvature scalars are the only extra tensors
    assert hyp["total"] - euc["total"] == 2 * 4  # n_layers * n_heads
    assert abs(hyp["total"] - euc["total"]) / euc["total"] < 0.005


def test_active_params_are_fewer_than_total():
    """Sparsity has to actually be sparse, or the MoE is decorative."""
    counts = MoETransformer(_cfg(n_routed_experts=4, top_k=2)).param_counts()
    assert counts["active_per_token"] < counts["total"]


@pytest.mark.parametrize("attention", ["euclidean", "hyperbolic"])
def test_backward_produces_finite_grads_everywhere(attention):
    model = MoETransformer(_cfg(attention=attention))
    x, y = _batch(64)
    model(x, y)["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    # the residual gate of the first layer sees no prev_attn, so it is legitimately unused
    assert missing == ["blocks.0.residual_gate"], missing
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("attention", ["euclidean", "hyperbolic"])
def test_model_is_causal(attention):
    """Changing a later token must not alter logits at earlier positions."""
    model = MoETransformer(_cfg(attention=attention)).eval()
    x, _ = _batch(64, b=1)
    with torch.no_grad():
        a = model(x)["logits"]
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % 64
        b = model(x2)["logits"]
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_attention_residual_starts_as_identity():
    """The gate initialises to 0, so the extra path cannot hurt at init."""
    torch.manual_seed(3)
    with_res = MoETransformer(_cfg(attn_residual=True)).eval()
    torch.manual_seed(3)
    without = MoETransformer(_cfg(attn_residual=False)).eval()
    x, _ = _batch(64)
    with torch.no_grad():
        assert torch.allclose(with_res(x)["logits"], without(x)["logits"], atol=1e-5)


def test_attention_residual_changes_output_once_gate_opens():
    model = MoETransformer(_cfg(attn_residual=True)).eval()
    x, _ = _batch(64)
    with torch.no_grad():
        before = model(x)["logits"].clone()
        for block in model.blocks:
            block.residual_gate.fill_(0.5)
        after = model(x)["logits"]
    assert not torch.allclose(before, after, atol=1e-4)


@pytest.mark.parametrize("attention", ["euclidean", "hyperbolic"])
def test_generation_shape_and_range(attention):
    cfg = _cfg(attention=attention)
    model = MoETransformer(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=6)
    assert out.shape == (1, 10)
    assert (out >= 0).all() and (out < cfg.vocab_size).all()


@pytest.mark.parametrize("attention", ["euclidean", "hyperbolic"])
def test_training_reduces_loss_on_a_memorisable_batch(attention):
    """End-to-end sanity: both arms must be able to learn *something*."""
    torch.manual_seed(0)
    model = MoETransformer(_cfg(attention=attention))
    x, y = _batch(64)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = model(x, y)["ce_loss"].item()
    for _ in range(30):
        opt.zero_grad()
        loss = model(x, y)["loss"]
        loss.backward()
        opt.step()
    last = model(x, y)["ce_loss"].item()
    assert last < first * 0.9, f"{attention}: {first:.3f} -> {last:.3f}"


@pytest.mark.parametrize("attention", ["euclidean", "hyperbolic"])
def test_seed_determinism(attention):
    losses = []
    for _ in range(2):
        torch.manual_seed(11)
        model = MoETransformer(_cfg(attention=attention))
        torch.manual_seed(12)
        x, y = _batch(64)
        losses.append(model(x, y)["loss"].item())
    assert losses[0] == losses[1]


def test_attention_stats_are_populated():
    model = MoETransformer(_cfg(attention="hyperbolic"))
    x, y = _batch(64)
    model(x, y, need_stats=True)
    stats = model.attention_stats()
    assert stats["attn_entropy_mean"] >= 0
    assert 0 <= stats["expert_entropy_mean"] <= stats["expert_entropy_max"] + 1e-6
    assert "curvature_mean" in stats and stats["curvature_mean"] > 0
    assert "manifold_radius_mean" in stats
