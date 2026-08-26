"""Cost microbenchmarks: attention scaling in sequence length, and inference latency.

The scaling benchmark answers the quadratic-versus-linear question empirically
rather than by inspection. Both arms should come out with an exponent near 2 in
the regime where the score matmul dominates; the hyperbolic arm should differ by
a constant factor, not by an order. Measuring it also catches the failure mode
where an implementation accidentally introduces an extra factor of ``seq_len``.
"""

from __future__ import annotations

import gc
import time

import torch

from .attention import build_attention
from .metrics import attention_flops_per_token, log_log_slope, peak_rss_mb
from .model import ModelConfig, MoETransformer

DEFAULT_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096]


def _time_calls(fn, warmup: int, repeats: int) -> tuple[float, float]:
    """Return (median seconds, min seconds) over ``repeats`` timed calls.

    Median rather than mean: a single scheduler hiccup on a shared 4-core box
    would otherwise dominate the estimate. The min is kept as the cleanest
    lower bound on the true cost.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2], samples[0]


@torch.no_grad()
def attention_scaling(
    d_model: int = 192,
    n_heads: int = 6,
    batch_size: int = 1,
    seq_lens: list[int] | None = None,
    warmup: int = 2,
    repeats: int = 5,
    verbose: bool = True,
) -> dict:
    """Time attention alone across sequence lengths, then fit the exponent."""
    seq_lens = seq_lens or DEFAULT_SEQ_LENS
    results: dict[str, dict] = {}

    for kind in ("euclidean", "hyperbolic"):
        attn = build_attention(kind, d_model, n_heads).eval()
        rows = []
        for seq_len in seq_lens:
            x = torch.randn(batch_size, seq_len, d_model)
            # need_stats=True forces the explicit score matmul in both arms, so
            # the Euclidean side is not silently timed on a fused kernel while
            # the hyperbolic side runs unfused. Same work, fair comparison.
            median, best = _time_calls(lambda: attn(x, need_stats=True), warmup, repeats)
            rows.append(
                {
                    "seq_len": seq_len,
                    "median_s": median,
                    "min_s": best,
                    "ms_per_token": 1000.0 * median / (batch_size * seq_len),
                    "analytic_flops_per_token": attention_flops_per_token(
                        seq_len, d_model, kind
                    ),
                    "peak_rss_mb": peak_rss_mb(),
                }
            )
            if verbose:
                print(f"    {kind:11s} L={seq_len:5d}  {1000 * median:8.1f} ms", flush=True)
            del x
            gc.collect()

        results[kind] = {
            "points": rows,
            "fit": log_log_slope([r["seq_len"] for r in rows], [r["median_s"] for r in rows]),
            "fit_large_only": log_log_slope(
                [r["seq_len"] for r in rows[-3:]], [r["median_s"] for r in rows[-3:]]
            ),
        }

    euc = {r["seq_len"]: r["median_s"] for r in results["euclidean"]["points"]}
    hyp = {r["seq_len"]: r["median_s"] for r in results["hyperbolic"]["points"]}
    results["overhead_ratio"] = {str(k): hyp[k] / euc[k] for k in euc}
    results["config"] = {
        "d_model": d_model,
        "n_heads": n_heads,
        "batch_size": batch_size,
        "seq_lens": seq_lens,
        "repeats": repeats,
    }
    return results


@torch.no_grad()
def inference_latency(
    model_cfg: ModelConfig,
    prefill_len: int = 256,
    decode_tokens: int = 16,
    repeats: int = 4,
) -> dict:
    """Prefill (one full-sequence forward) and per-token decode latency.

    Decode here recomputes the whole prefix each step -- there is no KV cache in
    this implementation -- so the number is a fair *relative* comparison between
    arms, not an absolute figure for a production decoder.
    """
    model = MoETransformer(model_cfg).eval()
    x = torch.randint(0, model_cfg.vocab_size, (1, prefill_len))

    prefill_median, _ = _time_calls(lambda: model(x), warmup=2, repeats=repeats)
    decode_median, _ = _time_calls(
        lambda: model.generate(x, max_new_tokens=decode_tokens, top_k=None),
        warmup=1,
        repeats=max(1, repeats - 1),
    )
    return {
        "prefill_ms": 1000.0 * prefill_median,
        "prefill_len": prefill_len,
        "decode_ms_per_token": 1000.0 * decode_median / decode_tokens,
        "decode_tokens": decode_tokens,
    }


def run_all(verbose: bool = True, seq_lens: list[int] | None = None) -> dict:
    """Full cost benchmark: attention scaling plus end-to-end inference latency."""
    if verbose:
        print("  attention scaling", flush=True)
    scaling = attention_scaling(seq_lens=seq_lens, verbose=verbose)

    latency = {}
    for kind in ("euclidean", "hyperbolic"):
        cfg = ModelConfig(
            vocab_size=256, d_model=192, n_layers=4, n_heads=6, d_ff=512,
            max_seq_len=256, attention=kind,
        )
        latency[kind] = inference_latency(cfg)
        if verbose:
            print(
                f"    {kind:11s} prefill {latency[kind]['prefill_ms']:.0f} ms  "
                f"decode {latency[kind]['decode_ms_per_token']:.0f} ms/token",
                flush=True,
            )

    return {"scaling": scaling, "latency": latency}
