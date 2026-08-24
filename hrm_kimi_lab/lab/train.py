"""Train/eval one variant and dump a JSON result."""
from dataclasses import asdict
import argparse, json, math, time
from pathlib import Path

import torch
from torch import nn

from lab.blocks import BlockConfig
from lab.data import CharData
from lab.model import LM, build
from lab.variants import VARIANTS


def count_params(model: nn.Module, moe_top_k_ratio: float) -> dict:
    total = sum(p.numel() for p in model.parameters())
    routed = sum(
        p.numel() for name, p in model.named_parameters() if ".routed_experts." in name
    )
    active = total - routed + routed * moe_top_k_ratio
    return {"total_params": total, "routed_expert_params": routed, "active_params": int(active)}


@torch.no_grad()
def evaluate(model: LM, data: CharData, batch_size: int, seq_len: int, iters: int, seed: int = 1234) -> float:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(iters):
        x, y = data.batch("val", batch_size, seq_len, g)
        losses.append(float(model.loss(x, y, bp_steps=99)))
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model: LM, data: CharData, prompt: str, n_tokens: int, seq_len: int, temperature: float = 0.8, seed: int = 0) -> str:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    ids = data.encode(prompt)[None]
    for _ in range(n_tokens):
        logits = model(ids[:, -seq_len:], bp_steps=99)[:, -1].float() / temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=g)
        ids = torch.cat([ids, nxt], dim=1)
    model.train()
    return data.decode(ids[0])


def run(variant_name: str, steps: int, batch_size: int, seq_len: int, hidden: int, heads: int,
        lr: float, seed: int, out_dir: Path, eval_iters: int = 20, log_every: int = 50) -> dict:
    torch.manual_seed(seed)
    variant = VARIANTS[variant_name]
    data = CharData()
    base = BlockConfig(hidden_size=hidden, num_heads=heads, max_seq_len=seq_len)
    model = build(variant, data.vocab_size, base)
    top_k_ratio = base.moe_top_k / base.moe_routed_experts
    stats = count_params(model, top_k_ratio)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = max(10, int(0.05 * steps))
    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))  # cosine decay to 10%
    g = torch.Generator().manual_seed(seed)

    history, t0 = [], time.time()
    for step in range(steps):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step)
        bp = model.core.bp_steps_for(step, steps)
        x, y = data.batch("train", batch_size, seq_len, g)
        loss = model.loss(x, y, bp_steps=bp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % log_every == 0 or step == steps - 1:
            history.append({"step": step, "train_loss": float(loss.detach()), "bp_steps": bp,
                            "elapsed_s": round(time.time() - t0, 1)})
            print(f"[{variant_name}] step {step:5d} loss {float(loss.detach()):.4f} bp {bp} "
                  f"{time.time()-t0:.0f}s", flush=True)
    train_time = time.time() - t0

    t1 = time.time()
    val_loss = evaluate(model, data, batch_size, seq_len, eval_iters)
    result = {
        "variant": variant_name,
        "config": asdict(variant),
        "hidden_size": hidden, "num_heads": heads, "seq_len": seq_len,
        "batch_size": batch_size, "steps": steps, "lr": lr, "seed": seed,
        **stats,
        "final_train_loss": history[-1]["train_loss"],
        "val_loss": val_loss,
        "val_bpc": val_loss / math.log(2),
        "val_ppl": math.exp(val_loss),
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(time.time() - t1, 1),
        "tokens_per_s": round(steps * batch_size * seq_len / train_time, 1),
        "history": history,
        "sample": sample(model, data, "KING RICHARD:\n", 200, seq_len),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": variant_name, "hidden": hidden, "heads": heads, "seq_len": seq_len,
                "state_dict": model.state_dict()}, out_dir / f"{variant_name}.pt")
    (out_dir / f"{variant_name}.json").write_text(json.dumps(result, indent=2))
    print(f"[{variant_name}] val_loss {val_loss:.4f} bpc {result['val_bpc']:.4f} "
          f"({train_time:.0f}s, {stats['total_params']:,} params)", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("variant")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    run(a.variant, a.steps, a.batch_size, a.seq_len, a.hidden, a.heads, a.lr, a.seed, a.out)
