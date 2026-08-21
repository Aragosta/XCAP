"""Tests for the energy formulation itself -- the maths, not the benchmark."""
import math

import pytest
import torch

from ebt.attention import Attention, AttentionConfig
from ebt.variants import VARIANT_NAMES, variant

B, N, D, H = 2, 12, 32, 4


def cfg(**kw):
    kw.setdefault("d_model", D)
    kw.setdefault("n_heads", H)
    return AttentionConfig(**kw)


def _x(seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, N, D)


# ------------------------------------------------------------------ the energy
def test_energy_matches_the_explicit_squared_distance():
    att = Attention(cfg(score="energy"))
    q, k = torch.randn(B, H, N, D // H), torch.randn(B, H, N, D // H)
    ref = (q[:, :, :, None, :] - k[:, :, None, :, :]).pow(2).sum(-1)
    assert torch.allclose(att._energy(q, k), ref, atol=1e-4)


def test_energy_is_zero_on_the_diagonal_when_query_equals_key():
    att = Attention(cfg(score="energy"))
    q = torch.randn(B, H, N, D // H)
    assert torch.allclose(att._energy(q, q).diagonal(dim1=-2, dim2=-1),
                          torch.zeros(B, H, N), atol=1e-4)


def test_energy_is_never_negative():
    att = Attention(cfg(score="energy"))
    e = att._energy(torch.randn(B, H, N, D // H) * 5, torch.randn(B, H, N, D // H) * 5)
    assert (e >= 0).all()


def test_energy_attention_equals_dot_attention_up_to_the_key_norm_penalty():
    """-||q-k||^2 = 2 q.k - ||q||^2 - ||k||^2, and ||q||^2 is constant in a row.

    So with keys normalised to equal norm, energy attention *is* dot attention
    at twice the temperature.  This is the identity that bounds how novel the
    competitive energy gate can be.
    """
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D // H)
    k = torch.nn.functional.normalize(torch.randn(B, H, N, D // H), dim=-1)
    att = Attention(cfg(score="energy"))
    energy_scores = -att._energy(q, k)
    dot_scores = 2.0 * (q @ k.transpose(-1, -2))
    assert torch.allclose(torch.softmax(energy_scores, -1),
                          torch.softmax(dot_scores, -1), atol=1e-4)


def test_energy_attention_penalises_large_norm_keys():
    """The one behavioural difference the identity predicts."""
    torch.manual_seed(0)
    q = torch.zeros(1, 1, 1, 4)
    k = torch.tensor([[1.0, 0, 0, 0], [3.0, 0, 0, 0]]).view(1, 1, 2, 4)
    att = Attention(cfg(score="energy", d_model=4, n_heads=1))
    a = torch.softmax(-att._energy(q, k), -1)[0, 0, 0]
    assert a[0] > a[1], "the closer (smaller-norm) memory must win"


# -------------------------------------------------------------------- the gate
def test_softmax_gate_rows_sum_to_one_and_sigmoid_rows_do_not():
    x = _x()
    soft = Attention(cfg(score="energy", gate="softmax"))
    soft(x)
    assert soft.last_stats["attn_row_mass"] == pytest.approx(1.0, abs=1e-4)
    sig = Attention(cfg(score="energy", gate="sigmoid"))
    sig(x)
    assert sig.last_stats["attn_row_mass"] != pytest.approx(1.0, abs=1e-2)


def test_sigmoid_gate_lets_several_memories_be_active_at_once():
    """The whole point of dropping the competition: facts can co-occur."""
    att = Attention(cfg(score="energy", gate="sigmoid"))
    with torch.no_grad():
        att.tau_bias.fill_(50.0)              # everything below threshold -> active
    att(_x())
    assert att.last_stats["attn_active"] > 1.0


def test_sigmoid_gate_can_activate_nothing_at_all():
    """A query that matches no memory should be able to say so."""
    att = Attention(cfg(score="energy", gate="sigmoid"))
    with torch.no_grad():
        att.tau_bias.fill_(-50.0)             # nothing passes threshold
    out = att(_x())
    assert att.last_stats["attn_active"] == 0.0
    assert att.last_stats["attn_row_mass"] < 1e-3
    assert out.abs().max() < 1e-4, "no active memory -> no contribution"


def test_sigmoid_weights_are_independent_across_memories():
    """Changing one key must not move the weight on another.

    This is the structural property softmax cannot have: its denominator ties
    every weight in a row to every other key.  Under the sigmoid gate the
    weight on key 1 depends only on q_i, k_1 and tau_1, so scrambling key 5
    leaves it *exactly* unchanged.
    """
    torch.manual_seed(0)
    x = _x()
    x2 = x.clone()
    x2[:, 5] = torch.randn(B, D) * 3                  # scramble a different key
    rows = [i for i in range(N) if i != 5]            # query row 5 legitimately moves

    def _delta(gate):
        att = Attention(cfg(score="energy", gate=gate)).eval()
        att(x)
        before = att.last_attn[:, :, rows, 1].clone()
        att(x2)
        return float((att.last_attn[:, :, rows, 1] - before).abs().max())

    assert _delta("sigmoid") < 1e-6, "sigmoid weights must be strictly independent"
    assert _delta("softmax") > 1e-3, "softmax weights are coupled through the denominator"


# -------------------------------------------------------------- the relations
def test_relation_probabilities_are_a_distribution_over_the_codebook():
    att = Attention(cfg(score="transe", n_relations=4))
    att(_x())
    p = att.last_relation_probs
    assert p.shape == (B, H, N, 4)
    assert torch.allclose(p.sum(-1), torch.ones(B, H, N), atol=1e-5)


def test_transe_with_a_frozen_single_relation_reduces_to_energy_attention():
    """With one relation slot, g_r(z) = z + r is a constant shift of every query.

    Combined with equal shifts on all queries this changes the energy by a term
    that is *not* constant per row, so the two are close but not identical --
    what must hold exactly is that a zero relation vector recovers plain energy.
    """
    torch.manual_seed(0)
    rel = Attention(cfg(score="transe", n_relations=2))
    with torch.no_grad():
        rel.relations.zero_()                       # g_r(z) = z
    plain = Attention(cfg(score="energy"))
    plain.load_state_dict({k: v for k, v in rel.state_dict().items()
                           if k in plain.state_dict()}, strict=False)
    x = _x()
    assert torch.allclose(rel(x), plain(x), atol=1e-5)


def test_different_relation_slots_send_a_subject_to_different_places():
    """The Paris/France vs Paris/Seine property, in geometry."""
    att = Attention(cfg(score="transe", n_relations=3))
    with torch.no_grad():
        att.relations.normal_(0, 1.0)
    q = torch.zeros(1, H, 1, D // H)
    shifted = [q + att.relations[:, c][None, :, None, :] for c in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert (shifted[i] - shifted[j]).abs().max() > 1e-3


def test_rotate_preserves_the_query_norm():
    """A rotation cannot change length -- that is what distinguishes it from a shift."""
    att = Attention(cfg(score="rotate", n_relations=3))
    with torch.no_grad():
        att.relations.normal_(0, 1.0)
    x = _x()
    q = att._split(att.qkv(x).chunk(3, dim=-1)[0])
    assert torch.allclose(att._relation(x, q).norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_transe_does_not_preserve_the_query_norm():
    att = Attention(cfg(score="transe", n_relations=3))
    with torch.no_grad():
        att.relations.normal_(0, 1.0)
    x = _x()
    q = att._split(att.qkv(x).chunk(3, dim=-1)[0])
    assert not torch.allclose(att._relation(x, q).norm(dim=-1), q.norm(dim=-1), atol=1e-3)


def test_rotate_needs_an_even_head_dimension():
    with pytest.raises(ValueError):
        AttentionConfig(d_model=12, n_heads=8, score="rotate")


# ------------------------------------------------------------------- plumbing
@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_every_variant_runs_forward_and_backward(name):
    att = Attention(variant(name, d_model=D, n_heads=H))
    out = att(_x())
    out.pow(2).sum().backward()
    assert out.shape == (B, N, D) and torch.isfinite(out).all()
    for pname, p in att.named_parameters():
        if pname == "log_temp":
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), pname


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_every_variant_reports_diagnostics(name):
    att = Attention(variant(name, d_model=D, n_heads=H))
    att(_x())
    for key in ("attn_entropy", "attn_max", "attn_row_mass", "attn_active"):
        assert math.isfinite(att.last_stats[key])
    if "energy" in name or "transe" in name or "rotate" in name:
        assert att.last_stats["energy_mean"] >= 0


def test_config_validation():
    with pytest.raises(ValueError):
        AttentionConfig(score="nope")
    with pytest.raises(ValueError):
        AttentionConfig(gate="nope")
    with pytest.raises(ValueError):
        AttentionConfig(score="transe", n_relations=1)
