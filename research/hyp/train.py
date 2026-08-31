"""Train the MoE MHA stack (and a dense twin as control) on GSM8K CoT text."""
import argparse, json, os, time, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Vocab, load_gsm8k, encode_corpus
from model import Config, MoETransformer

HERE = os.path.dirname(os.path.abspath(__file__))


def batches(stream, seq_len, batch, rng):
    while True:
        ix = rng.integers(0, len(stream) - seq_len - 1, size=batch)
        x = np.stack([stream[i:i + seq_len] for i in ix]).astype(np.int64)
        y = np.stack([stream[i + 1:i + 1 + seq_len] for i in ix]).astype(np.int64)
        yield torch.from_numpy(x), torch.from_numpy(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dense", action="store_true", help="control: no MoE")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bench", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    tag = args.tag or ("dense" if args.dense else "moe")
    out = os.path.join(HERE, "runs", tag)
    os.makedirs(out, exist_ok=True)

    train_txt = load_gsm8k(os.path.join(HERE, "data/gsm8k_train.jsonl"))
    val_txt = load_gsm8k(os.path.join(HERE, "data/gsm8k_test.jsonl"))
    vpath = os.path.join(HERE, "runs", "vocab.json")
    if os.path.exists(vpath):
        vocab = Vocab.load(vpath)
    else:
        vocab = Vocab(train_txt, max_size=4096)
        vocab.save(vpath)
    tr = encode_corpus(train_txt, vocab)
    va = encode_corpus(val_txt, vocab)

    cfg = Config(vocab_size=len(vocab), d_model=192, n_layers=6, n_heads=6,
                 seq_len=224, n_experts=8, top_k=2, d_ff=384, moe=not args.dense)
    model = MoETransformer(cfg)
    print(f"[{tag}] params {model.n_params()/1e6:.2f}M  active/token {model.n_active_params()/1e6:.2f}M "
          f"vocab {len(vocab)}  train tokens {len(tr)/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1, anneal_strategy="cos")
    rng = np.random.default_rng(0)
    gen = batches(tr, cfg.seq_len, args.batch, rng)

    if args.bench:
        t0 = time.time()
        for _ in range(args.bench):
            x, y = next(gen)
            _, loss, _ = model(x, y)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        print("bench %.2f s/step" % ((time.time() - t0) / args.bench), flush=True)
        return

    # Checkpoint early as well as late: whether hyperbolic structure is present
    # at init or emerges with training is itself one of the questions.
    ckpts = {0, 250, 750, 1500, args.steps}
    log, t0, best = [], time.time(), float("inf")
    for step in range(args.steps + 1):
        if step in ckpts:
            torch.save({"model": model.state_dict(), "cfg": vars(cfg)},
                       os.path.join(out, f"ckpt_{step}.pt"))
        if step == args.steps:
            break
        x, y = next(gen)
        _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                vg = batches(va, cfg.seq_len, args.batch, np.random.default_rng(1))
                vl = float(np.mean([model(*next(vg))[1].item() for _ in range(4)]))
            model.train()
            el = time.time() - t0
            print(f"[{tag}] step {step:5d}  train {loss.item():.4f}  val {vl:.4f}  "
                  f"ppl {np.exp(vl):.1f}  {el:.0f}s", flush=True)
            log.append({"step": step, "train": loss.item(), "val": vl, "secs": el})
            if vl < best:                      # the probes should read a model
                best = vl                      # that generalises, not one that
                torch.save({"model": model.state_dict(), "cfg": vars(cfg),   # memorised
                            "step": step, "val": vl}, os.path.join(out, "ckpt_best.pt"))
    json.dump(log, open(os.path.join(out, "log.json"), "w"), indent=1)
    print(f"[{tag}] done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
