import math

import pytest
import torch

from ebt.attention import Attention, AttentionConfig, attention_flops
from ebt.variants import VARIANT_NAMES, all_variants, variant

B, N, D, H = 3, 32, 32, 4


def cfg(**kw):
    kw.setdefault("d_model", D)
    kw.setdefault("n_heads", H)
    return AttentionConfig(**kw)


def _x(requires_grad=False, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, N, D, requires_grad=requires_grad)


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_shape_and_finiteness(name):
    att = Attention(variant(name, d_model=D, n_heads=H))
    out = att(_x())
    assert out.shape == (B, N, D)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_backward_reaches_every_parameter(name):
    att = Attention(variant(name, d_model=D, n_heads=H))
    att(_x()).pow(2).sum().backward()
    for pname, p in att.named_parameters():
        if pname == "log_temp":
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), pname
        assert p.grad.abs().sum() > 0, pname


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_permutation_of_batch_is_independent(name):
    att = Attention(variant(name, d_model=D, n_heads=H)).eval()
    x = _x()
    out = att(x)
    perm = torch.tensor([2, 0, 1])
    assert torch.allclose(att(x[perm]), out[perm], atol=1e-5)


def test_softmax_matches_reference_attention():
    att = Attention(cfg(normaliser="softmax", learn_temp=False)).eval()
    x = _x()
    q, k, v = (att._split(t) for t in att.qkv(x).chunk(3, dim=-1))
    ref = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D // H), dim=-1) @ v
    ref = att.out_proj(ref.transpose(1, 2).reshape(B, N, D))
    assert torch.allclose(att(x), ref, atol=1e-5)


def test_causal_mask_blocks_future_information():
    att = Attention(cfg(causal=True)).eval()
    x = _x(requires_grad=True)
    att(x)[:, N // 2].sum().backward()
    assert x.grad[:, N // 2 + 1 :].abs().max() == 0.0
    assert x.grad[:, : N // 2 + 1].abs().max() > 0.0


def test_noncausal_does_see_the_future():
    att = Attention(cfg(causal=False)).eval()
    x = _x(requires_grad=True)
    att(x)[:, 0].sum().backward()
    assert x.grad[:, 1:].abs().max() > 0.0


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_padding_mask_removes_influence_of_padded_tokens(name):
    att = Attention(variant(name, d_model=D, n_heads=H)).eval()
    x = _x()
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -5:] = False
    out = att(x, key_padding_mask=mask)
    x2 = x.clone()
    x2[:, -5:] = torch.randn(B, 5, D)          # scramble the padded region
    out2 = att(x2, key_padding_mask=mask)
    assert torch.allclose(out[:, :-5], out2[:, :-5], atol=1e-5)


def test_sparsemax_attention_has_exact_zeros_when_logits_are_peaked():
    att = Attention(cfg(normaliser="sparsemax"))
    with torch.no_grad():
        att.log_temp.fill_(math.log(20.0))     # sharpen
    att(_x())
    assert att.last_stats["attn_zero_frac"] > 0.5


def test_softmax_attention_has_no_exact_zeros():
    att = Attention(cfg(normaliser="softmax"))
    with torch.no_grad():
        att.log_temp.fill_(math.log(20.0))
    att(_x())
    assert att.last_stats["attn_zero_frac"] == 0.0


def test_stats_are_recorded_for_every_variant():
    for c in all_variants(d_model=D, n_heads=H):
        att = Attention(c)
        att(_x())
        for key in ("attn_zero_frac", "attn_entropy", "attn_support", "attn_max"):
            assert key in att.last_stats
            assert math.isfinite(att.last_stats[key])


def test_sparsemax_support_shrinks_as_the_temperature_rises():
    """The sparsity is data- and scale-dependent, not a fixed budget."""
    supports = []
    for temp in (0.25, 1.0, 8.0):
        att = Attention(cfg(normaliser="sparsemax"))
        with torch.no_grad():
            att.log_temp.fill_(math.log(temp))
        att(_x())
        supports.append(att.last_stats["attn_support"])
    assert supports[0] > supports[1] > supports[2]


def test_sparsemax_rows_still_sum_to_one():
    att = Attention(cfg(normaliser="sparsemax"))
    x = _x()
    q, k, _ = (att._split(t) for t in att.qkv(x).chunk(3, dim=-1))
    attn = att.normalise(att._scores(q, k), dim=-1, mask=att._allowed(N, torch.ones(B, N, dtype=torch.bool)))
    assert torch.allclose(attn.sum(-1), torch.ones(B, H, N), atol=1e-5)


def test_flops_are_quadratic_in_sequence_length():
    c = cfg()
    quad = lambda n: attention_flops(c, n) - 4 * n * D * D
    assert quad(512) == pytest.approx(quad(256) * 4, rel=1e-6)


def test_flops_do_not_depend_on_the_normaliser():
    """sparsemax changes which weights survive, not how many are computed."""
    assert attention_flops(cfg(normaliser="softmax"), 256) == \
        attention_flops(cfg(normaliser="sparsemax"), 256)


def test_config_validation():
    with pytest.raises(ValueError):
        AttentionConfig(normaliser="nope")
    with pytest.raises(ValueError):
        AttentionConfig(d_model=10, n_heads=4)
    with pytest.raises(ValueError):
        variant("no-such-variant")
