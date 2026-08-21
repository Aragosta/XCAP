import math

import pytest
import torch

from ebt.attention import AttentionConfig
from ebt.metrics import (attention_memory_bytes, benchmark_speed, evaluate,
                         masked_loss_and_acc, router_grad_coverage)
from ebt.model import build_model
from ebt.tasks import build_task
from ebt.train import TrainConfig, run
from ebt.variants import VARIANT_NAMES, variant

SEQ, D, H = 32, 32, 4


def small(name):
    return variant(name, d_model=D, n_heads=H)


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_model_forward_backward(name):
    task = build_task("needle", SEQ)
    model = build_model(task, small(name))
    x, y, m = task.batch(4, torch.Generator().manual_seed(0))
    logits = model(x)
    assert logits.shape == (4, SEQ, task.n_classes)
    loss, acc = masked_loss_and_acc(logits, y, m)
    loss.backward()
    assert torch.isfinite(loss) and 0.0 <= float(acc) <= 1.0
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_parameter_counts_are_comparable_across_variants(name):
    task = build_task("needle", SEQ)
    base = build_model(task, small("baseline-softmax")).n_params()
    other = build_model(task, small(name)).n_params()
    assert abs(other - base) / base < 0.02, "variants must not differ in capacity"


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_model_can_overfit_a_single_batch(name):
    """Sanity floor: every mechanism must be able to memorise 8 sequences."""
    torch.manual_seed(0)
    task = build_task("associative_recall", SEQ)
    model = build_model(task, small(name))
    x, y, m = task.batch(8, torch.Generator().manual_seed(0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(120):
        loss, acc = masked_loss_and_acc(model(x), y, m)
        opt.zero_grad(); loss.backward(); opt.step()
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
    assert last < first * 0.5, f"{name}: loss {first:.3f} -> {last:.3f}"


def test_evaluate_is_deterministic_for_a_seed():
    task = build_task("majority", SEQ)
    model = build_model(task, small("baseline-softmax"))
    a = evaluate(model, task, 2, 8, seed=3)
    b = evaluate(model, task, 2, 8, seed=3)
    assert a["loss"] == pytest.approx(b["loss"])
    assert set(a) >= {"loss", "acc", "attn_entropy", "attn_zero_frac", "token_coverage"}


def test_evaluate_restores_training_mode():
    task = build_task("majority", SEQ)
    model = build_model(task, small("baseline-softmax"))
    model.train()
    evaluate(model, task, 1, 4, seed=0)
    assert model.training


def test_flops_and_memory_shrink_with_routing():
    task = build_task("majority", 256)
    dense = build_model(task, variant("baseline-softmax", d_model=D, n_heads=H))
    mosa = build_model(task, variant("mosa-softmax", d_model=D, n_heads=H, capacity_ratio=0.25))
    assert mosa.flops_per_sequence() < dense.flops_per_sequence()
    assert (attention_memory_bytes(mosa, 256, 8)
            < attention_memory_bytes(dense, 256, 8) / 10)


def test_router_grad_coverage_reports_the_expected_ordering():
    task = build_task("needle", SEQ)
    cov = {}
    for name in ("mosa-softmax", "smaxroute-softmax"):
        torch.manual_seed(0)
        model = build_model(task, small(name))
        cov[name] = router_grad_coverage(model, task, 8, seed=0)["router_grad_frac"]
    assert 0.0 < cov["mosa-softmax"] <= 0.30
    assert cov["smaxroute-softmax"] > 0.0


def test_router_grad_coverage_is_empty_for_dense():
    task = build_task("needle", SEQ)
    model = build_model(task, small("baseline-softmax"))
    assert router_grad_coverage(model, task, 4, seed=0) == {}


def test_benchmark_speed_returns_positive_timings():
    task = build_task("majority", SEQ)
    model = build_model(task, small("baseline-softmax"))
    out = benchmark_speed(model, task, 4, iters=2, warmup=1)
    # no relative timing assertions here: wall clock is far too noisy on a
    # loaded machine to be a unit test.  Scaling behaviour is measured by
    # experiments/run_scaling.py instead.
    assert out["fwd_ms"] > 0 and out["fwd_bwd_ms"] > 0
    assert out["tokens_per_s_fwd"] > 0


def test_run_produces_a_complete_result_record():
    cfg = TrainConfig(steps=20, batch_size=8, eval_every=10, eval_batches=1,
                      eval_batch_size=8, warmup=5, n_layers=1, seed=0)
    res = run("associative_recall", small("mosa-sparsemax"), cfg, seq_len=SEQ)
    for key in ("task", "variant", "final_acc", "final_loss", "history", "params",
                "flops_per_seq", "fwd_ms", "grad_norm_mean", "router_grad_frac"):
        assert key in res
    assert len(res["history"]) == 2
    assert math.isfinite(res["final_loss"])


def test_run_is_reproducible():
    cfg = TrainConfig(steps=10, batch_size=8, eval_every=10, eval_batches=1,
                      eval_batch_size=8, warmup=2, n_layers=1, seed=1)
    a = run("needle", small("baseline-softmax"), cfg, seq_len=SEQ)
    b = run("needle", small("baseline-softmax"), cfg, seq_len=SEQ)
    assert a["final_loss"] == pytest.approx(b["final_loss"], rel=1e-6)
