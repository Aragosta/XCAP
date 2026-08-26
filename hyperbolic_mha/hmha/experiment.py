"""Experiment driver: main arms across seeds, ablations, benchmark, and samples.

Results are written to ``results/metrics.json`` **incrementally**, after every
stage. The container running this is ephemeral, so a crash or timeout two thirds
of the way through still leaves usable, already-committed numbers behind.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import torch

from . import bench
from .data import decode, encode, load_corpus
from .metrics import compare_arms, embedding_geometry
from .model import ModelConfig, MoETransformer
from .train import TrainConfig, evaluate, train_arm

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SAMPLE_PROMPT = " The history of the city"

PRESETS = {
    # Sized from a measured throughput probe on this 4-core CPU box: the
    # Euclidean arm runs ~261 ms/step and the hyperbolic ~351 ms/step at this
    # shape, so 6 main runs + 8 ablations + the benchmark land in ~35 minutes.
    "fast": {
        "model": dict(d_model=192, n_layers=4, n_heads=6, d_ff=512, max_seq_len=256),
        "train": dict(steps=700, batch_size=16, seq_len=256, eval_every=175, eval_batches=12),
        "ablation_steps": 300,
        "seeds": [0, 1, 2],
        "bench_seq_lens": [128, 256, 512, 1024, 2048, 4096],
    },
    "smoke": {
        "model": dict(d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=64),
        "train": dict(steps=10, batch_size=4, seq_len=64, eval_every=5, eval_batches=2),
        "ablation_steps": 5,
        "seeds": [0],
        "bench_seq_lens": [64, 128, 256],
    },
}


def _base_config(preset: dict, attention: str, **overrides) -> ModelConfig:
    return ModelConfig(vocab_size=256, attention=attention, **preset["model"], **overrides)


def _save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _strip_model(result: dict) -> tuple[dict, MoETransformer]:
    """Split the trained module out of the result so the rest is JSON-serialisable."""
    model = result.pop("_model")
    return result, model


def _sample(model: MoETransformer, n_tokens: int = 200) -> str:
    prompt = encode(SAMPLE_PROMPT).long().unsqueeze(0)
    return decode(model.generate(prompt, max_new_tokens=n_tokens, temperature=0.8)[0])


def run(preset_name: str = "fast", out_path: Path | None = None, verbose: bool = True) -> dict:
    preset = PRESETS[preset_name]
    out_path = out_path or RESULTS_DIR / "metrics.json"
    started = time.time()

    torch.set_num_threads(4)
    corpus = load_corpus()

    payload: dict = {
        "preset": preset_name,
        "environment": {
            "torch": torch.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "corpus": corpus.summary(),
        "arms": {},
        "ablations": {},
    }

    # ---------------------------------------------------------------- main arms
    for attention in ("euclidean", "hyperbolic"):
        runs = []
        for seed in preset["seeds"]:
            if verbose:
                print(f"[{attention}] seed {seed}", flush=True)
            model_cfg = _base_config(preset, attention)
            train_cfg = TrainConfig(seed=seed, **preset["train"])
            result, model = _strip_model(
                train_arm(model_cfg, train_cfg, corpus, verbose=verbose)
            )
            result["seed"] = seed
            result["sample"] = _sample(model)
            result["geometry"] = embedding_geometry(
                model, curvature=model_cfg.curvature if attention == "hyperbolic" else None
            )
            # Held-out test set, full pass, evaluated once at the end only.
            result["full_test"] = evaluate(
                model, corpus.test, train_cfg.batch_size, train_cfg.seq_len, max_batches=None
            )
            runs.append(result)
            del model

        payload["arms"][attention] = {"runs": runs}
        _save(payload, out_path)

    payload["comparison"] = _compare(payload["arms"])
    _save(payload, out_path)

    # ---------------------------------------------------------------- ablations
    ablation_train = TrainConfig(
        seed=0, **{**preset["train"], "steps": preset["ablation_steps"]}
    )
    ablation_train = replace(
        ablation_train, eval_every=max(1, preset["ablation_steps"] // 2)
    )

    for name, overrides in _ablation_grid().items():
        if verbose:
            print(f"[ablation] {name}", flush=True)
        model_cfg = _base_config(preset, "hyperbolic", **overrides)
        result, model = _strip_model(
            train_arm(model_cfg, ablation_train, corpus, verbose=False)
        )
        result["overrides"] = overrides
        payload["ablations"][name] = result
        del model
        _save(payload, out_path)

    # Euclidean reference at the ablation budget, so the grid has a baseline
    # trained for the same number of steps rather than being read against the
    # longer main run.
    if verbose:
        print("[ablation] euclidean_reference", flush=True)
    result, _ = _strip_model(
        train_arm(_base_config(preset, "euclidean"), ablation_train, corpus, verbose=False)
    )
    result["overrides"] = {"attention": "euclidean"}
    payload["ablations"]["euclidean_reference"] = result
    _save(payload, out_path)

    # ---------------------------------------------------------------- benchmark
    if verbose:
        print("[benchmark]", flush=True)
    payload["benchmark"] = bench.run_all(verbose=verbose, seq_lens=preset["bench_seq_lens"])

    payload["total_wall_s"] = time.time() - started
    _save(payload, out_path)
    if verbose:
        print(f"\nDone in {payload['total_wall_s'] / 60:.1f} min -> {out_path}", flush=True)
    return payload


def _ablation_grid() -> dict[str, dict]:
    """Single-seed, short-budget variants isolating each design choice."""
    return {
        "curvature_0.25": {"curvature": 0.25},
        "curvature_1.0": {"curvature": 1.0},
        "curvature_4.0": {"curvature": 4.0},
        "curvature_learnable": {"curvature": 1.0, "learnable_curvature": True},
        "score_sign_spec": {"score_sign": "spec"},
        "score_scale_learned": {"score_scale": "learned"},
        "aggregation_klein": {"aggregation": "klein"},
        "aggregation_tangent_mean": {"aggregation": "tangent_mean"},
    }


def _compare(arms: dict) -> dict:
    """Cross-seed statistics on the metrics that decide the question."""
    out = {}
    for metric, path in [
        ("val_bits_per_byte", ("final_val", "bits_per_byte")),
        ("test_bits_per_byte", ("full_test", "bits_per_byte")),
        ("best_val_bits_per_byte", ("best_val_bits_per_byte",)),
    ]:
        values = {}
        for arm, data in arms.items():
            seq = []
            for run_result in data["runs"]:
                node = run_result
                for key in path:
                    node = node[key]
                seq.append(float(node))
            values[arm] = seq
        out[metric] = compare_arms(
            values["hyperbolic"], values["euclidean"], "hyperbolic", "euclidean"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hyperbolic-MHA evaluation lab.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="fast")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(args.preset, out_path=args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
