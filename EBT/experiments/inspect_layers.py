#!/usr/bin/env python3
"""Look inside a trained model: what does each layer/head actually attend to?

For the relational task every token has a known role (subject / relation /
object / noise / query), so we can ask concrete questions instead of staring at
heatmaps:

  * where does the *answer position* put its attention mass, per layer, per
    head, broken down by token role?
  * does it find the object token of the queried fact specifically, or just
    "some object token"?
  * how many memories does each row activate (the sigmoid gate's whole point)?
  * for relational scores: does the relation codebook specialise?  We measure
    the purity of  argmax-slot  against the true relation id -- 1.0 means each
    relation token reliably picks its own slot, 1/n_relations means the
    codebook is ignored.

Trains each variant briefly, then reports.  Usage:

    python experiments/inspect_layers.py --steps 600 --variants dot-softmax energy-sigmoid
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ebt.attention import Attention, RELATIONAL
from ebt.metrics import masked_loss_and_acc
from ebt.model import build_model
from ebt.tasks import Relational
from ebt.train import TrainConfig, run
from ebt.variants import DEFAULT_GRID, variant

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("subject", "relation", "object", "noise", "query")


def role_map(task: Relational, x: torch.Tensor) -> torch.Tensor:
    """[B,N] -> index into ROLES."""
    r = torch.full_like(x, ROLES.index("noise"))
    r[(x >= task.subj0) & (x < task.rel0)] = ROLES.index("subject")
    r[(x >= task.rel0) & (x < task.obj0)] = ROLES.index("relation")
    r[(x >= task.obj0) & (x < task.query_tok)] = ROLES.index("object")
    r[x == task.query_tok] = ROLES.index("query")
    return r


def gold_object_position(task: Relational, x: torch.Tensor) -> torch.Tensor:
    """[B] position of the object token of the queried (s, r) fact."""
    b, n = x.shape
    s, rel = x[:, -2], x[:, -1]
    body = x[:, : n - 3]
    hit = torch.zeros(b, dtype=torch.long)
    for i in range(body.size(1) - 2):
        match = (body[:, i] == s) & (body[:, i + 1] == rel)
        hit = torch.where(match, torch.full_like(hit, i + 2), hit)
    return hit


@torch.no_grad()
def inspect(model, task: Relational, batch_size: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    x, y, m = task.batch(batch_size, g)
    logits = model(x)
    _, acc = masked_loss_and_acc(logits, y, m)
    roles = role_map(task, x)
    gold = gold_object_position(task, x)
    rows = torch.arange(x.size(0))

    report = {"acc": float(acc), "layers": []}
    for li, blk in enumerate(model.blocks):
        att: Attention = blk.attn
        a = att.last_attn[:, :, -1, :]                       # answer row: [B,H,N]
        mass = a / a.sum(-1, keepdim=True).clamp(min=1e-9)
        by_role = {role: float(mass[:, :, :][roles[:, None, :].expand_as(mass) == i].sum()
                               / mass.sum()) for i, role in enumerate(ROLES)}
        on_gold = mass[rows[:, None], torch.arange(mass.size(1))[None, :], gold[:, None]]
        layer = {
            "layer": li,
            "mass_by_role": by_role,
            "mass_on_gold_object": float(on_gold.mean()),
            "mass_on_gold_object_best_head": float(on_gold.mean(0).max()),
            "argmax_is_gold_frac": float((a.argmax(-1) == gold[:, None]).float().mean()),
            "per_head_gold_mass": [round(float(v), 3) for v in on_gold.mean(0)],
            "row_mass": att.last_stats["attn_row_mass"],
            "active_memories": att.last_stats["attn_active"],
            "entropy": att.last_stats["attn_entropy"],
        }
        if att.cfg.score in RELATIONAL:
            probs = att.last_relation_probs[:, :, -1, :]     # [B,H,R] at the answer row
            slot = probs.argmax(-1)
            true_rel = (x[:, -1] - task.rel0)
            purity = []
            for h in range(slot.size(1)):
                counts: dict[int, Counter] = {}
                for r_id, s_id in zip(true_rel.tolist(), slot[:, h].tolist()):
                    counts.setdefault(s_id, Counter())[r_id] += 1
                correct = sum(c.most_common(1)[0][1] for c in counts.values())
                purity.append(correct / slot.size(0))
            # NOTE: the selector reads the relation token directly, so even a
            # random projection separates relations.  The number is only
            # meaningful against the untrained control reported alongside it.
            layer["relation_slot_purity"] = round(max(purity), 3)
            layer["relation_slots_used"] = att.last_stats["relation_slots_used"]
            layer["relation_entropy"] = round(att.last_stats["relation_entropy"], 3)
        report["layers"].append(layer)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "introspection.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    task = Relational(seq_len=a.seq_len)
    chance = 1.0 / task.n_classes
    out = {"config": vars(a), "chance": chance, "variants": {}}
    print(f"relational task: {task.n_facts} facts, {task.n_classes} objects, chance {chance:.3f}\n")

    for name in a.variants:
        cfg = variant(name, d_model=a.d_model, n_heads=a.n_heads)
        tcfg = TrainConfig(steps=a.steps, batch_size=a.batch_size, lr=a.lr,
                           n_layers=a.n_layers, eval_every=max(1, a.steps // 3),
                           eval_batches=4, seed=0)
        res = run("relational", cfg, tcfg, seq_len=a.seq_len)
        model = build_model(task, cfg, n_layers=a.n_layers)
        model.load_state_dict(res.pop("state_dict"))
        rep = inspect(model, task, 256, seed=1234)
        torch.manual_seed(123)
        rep_init = inspect(build_model(task, cfg, n_layers=a.n_layers), task, 256, seed=1234)
        for L, L0 in zip(rep["layers"], rep_init["layers"]):
            L["mass_on_gold_object_init"] = L0["mass_on_gold_object"]
            if "relation_slot_purity" in L:
                L["relation_slot_purity_init"] = L0["relation_slot_purity"]
        rep["train_acc_curve"] = [round(h["acc"], 3) for h in res["history"]]
        rep["final_acc"] = res["final_acc"]
        rep["fwd_ms"] = res["fwd_ms"]
        out["variants"][name] = rep

        print(f"=== {name}   acc {res['final_acc']:.3f}   "
              f"curve {rep['train_acc_curve']}   fwd {res['fwd_ms']:.1f} ms")
        for L in rep["layers"]:
            m = L["mass_by_role"]
            print(f"  layer {L['layer']}: mass  subj {m['subject']:.2f} rel {m['relation']:.2f} "
                  f"obj {m['object']:.2f} noise {m['noise']:.2f} | gold-obj {L['mass_on_gold_object']:.3f} "
                  f"(init {L['mass_on_gold_object_init']:.3f}, best head "
                  f"{L['mass_on_gold_object_best_head']:.3f}, argmax {L['argmax_is_gold_frac']:.2f}) "
                  f"| rowmass {L['row_mass']:.2f} active {L['active_memories']:.1f} H {L['entropy']:.2f}"
              + (f" | slot purity {L['relation_slot_purity']:.2f}"
                 f" (init {L['relation_slot_purity_init']:.2f})"
                 if "relation_slot_purity" in L else ""))
        print(f"  heads (gold-object mass): {rep['layers'][-1]['per_head_gold_mass']}\n", flush=True)

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
