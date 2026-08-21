"""Tests for init-temperature calibration and tied projections."""
import math

import pytest
import torch

from ebt.attention import Attention
from ebt.calibrate import calibrate_temperature
from ebt.model import build_model
from ebt.tasks import build_task
from ebt.variants import VARIANT_NAMES, variant

SEQ, D, H = 32, 32, 4


def _setup(name, seed=0):
    torch.manual_seed(seed)
    task = build_task("relational", SEQ)
    model = build_model(task, variant(name, d_model=D, n_heads=H))
    x = task.batch(16, torch.Generator().manual_seed(0))[0]
    return model, x


@pytest.mark.parametrize("name", ["dot-softmax", "energy-softmax", "energy-sigmoid",
                                  "transe-softmax", "energy-softmax-tied"])
def test_calibration_hits_the_target_entropy(name):
    model, x = _setup(name)
    calibrate_temperature(model, x, target_frac=0.8)
    target = 0.8 * math.log(SEQ)
    with torch.no_grad():
        model(x)
    for blk in model.blocks:
        assert blk.attn.last_stats["attn_entropy"] == pytest.approx(target, abs=0.05)


def test_different_scores_need_different_temperatures_for_the_same_sharpness():
    """The confound calibration exists to remove: same knob, different meaning."""
    temps = {}
    for name in ("dot-softmax", "energy-softmax"):
        model, x = _setup(name)
        temps[name] = calibrate_temperature(model, x, target_frac=0.8)["layer0"]
    assert abs(temps["dot-softmax"] - temps["energy-softmax"]) > 0.3


def test_calibration_leaves_the_temperature_trainable():
    model, x = _setup("energy-softmax")
    calibrate_temperature(model, x)
    for blk in model.blocks:
        assert blk.attn.log_temp.requires_grad
    model(x).sum().backward()
    assert all(blk.attn.log_temp.grad is not None for blk in model.blocks)


def test_calibration_restores_training_mode():
    model, x = _setup("energy-softmax")
    model.train()
    calibrate_temperature(model, x)
    assert model.training


def test_tied_projection_makes_query_and_key_identical():
    att = Attention(variant("energy-softmax-tied", d_model=D, n_heads=H))
    x = torch.randn(2, SEQ, D)
    parts = att.qkv(x).chunk(2, dim=-1)
    assert len(parts) == 2, "tied attention projects only qk and v"
    q = att._split(parts[0])
    assert torch.allclose(att._energy(q, q).diagonal(dim1=-2, dim2=-1),
                          torch.zeros(2, H, SEQ), atol=1e-4), \
        "with W_Q = W_K a token is at zero energy with itself"


def test_tied_energy_attention_puts_its_largest_weight_on_the_diagonal():
    """Zero self-energy is the lowest possible, so a token attends to itself."""
    att = Attention(variant("energy-softmax-tied", d_model=D, n_heads=H)).eval()
    torch.manual_seed(0)
    att(torch.randn(2, SEQ, D) * 3)
    a = att.last_attn
    diag = a.diagonal(dim1=-2, dim2=-1)
    assert (diag >= a.max(-1).values - 1e-6).all()


def test_untied_attention_has_no_such_guarantee():
    att = Attention(variant("energy-softmax", d_model=D, n_heads=H)).eval()
    torch.manual_seed(0)
    att(torch.randn(2, SEQ, D) * 3)
    a = att.last_attn
    diag = a.diagonal(dim1=-2, dim2=-1)
    assert not (diag >= a.max(-1).values - 1e-6).all()
