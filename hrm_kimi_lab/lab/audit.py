"""Sanity audit: strict causality of every mixer, KDA mode consistency,
and train/val separation. A leak here would invalidate the comparison."""
import torch

from lab.blocks import BlockConfig, Stack
from lab.data import CharData
from lab.model import build
from lab.variants import VARIANTS


def causality(module, T=96, D=64, cut=48):
    """Change every token after `cut`; nothing at or before `cut` may move."""
    torch.manual_seed(0)
    x = torch.randn(2, T, D)
    y1 = module(x)
    x2 = x.clone()
    x2[:, cut + 1 :] = torch.randn_like(x2[:, cut + 1 :])
    y2 = module(x2)
    past = (y1[:, : cut + 1] - y2[:, : cut + 1]).abs().max().item()
    future = (y1[:, cut + 1 :] - y2[:, cut + 1 :]).abs().max().item()
    return past, future


def lm_causality(name, T=64):
    data = CharData()
    base = BlockConfig(hidden_size=64, num_heads=4, max_seq_len=T)
    torch.manual_seed(0)
    model = build(VARIANTS[name], data.vocab_size, base).eval()
    torch.manual_seed(1)
    x = torch.randint(0, data.vocab_size, (2, T))
    with torch.no_grad():
        l1 = model(x, bp_steps=99)
        x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 7) % data.vocab_size
        l2 = model(x2, bp_steps=99)
    return (l1[:, :-1] - l2[:, :-1]).abs().max().item()


if __name__ == "__main__":
    print("== per-block causality (max |delta| on past positions must be ~0) ==")
    configs = [("mha", "dense", {}), ("mha", "moe", {}), ("kda", "dense", {}), ("kda", "moe", {}),
               ("mha", "dense", {"mha_conv_kernel": 4}), ("kda", "dense", {"kda_conv_kernel": 1})]
    for mixer, ffn, extra in configs:
        if True:
            cfg = BlockConfig(hidden_size=64, num_heads=4, max_seq_len=96, mixer=mixer, ffn=ffn, **extra)
            torch.manual_seed(0)
            s = Stack(cfg, 2).eval()
            with torch.no_grad():
                past, future = causality(s)
            label = f"{mixer}+{ffn}" + (f" {extra}" if extra else "")
            print(f"  {label:34s} past {past:.2e}   future {future:.2e} (should be large)")

    print("== full-model causality: flip the LAST token, earlier logits must not move ==")
    for name in ("base_kda_dense", "hrm_kda_dense", "hrm_loop5_kda_mhamoe", "base_mha_dense",
                 "hrm_kda_x2_mhamoe", "base_hybrid_kda_mhamoe", "hrm_hybrid_kda_mha"):
        print(f"  {name:22s} max |delta| {lm_causality(name):.2e}")

    print("== KDA chunkwise vs recurrent (same weights, same input) ==")
    from src.kda import KDAConfig, KimiDeltaAttention
    torch.manual_seed(0)
    kda = KimiDeltaAttention(KDAConfig(d_model=64, num_heads=4, key_head_dim=16,
                                       value_head_dim=16, chunk_size=32,
                                       secondary_tile_size=16)).eval()
    x = torch.randn(2, 96, 64, dtype=torch.float64)
    kda = kda.double()
    with torch.no_grad():
        a = kda(x, mode="chunkwise").hidden_states
        b = kda(x, mode="recurrent").hidden_states
    print(f"  max |chunkwise - recurrent| {(a - b).abs().max().item():.2e}")

    print("== data split ==")
    d = CharData()
    tr, va = d.train.tolist(), d.val.tolist()
    print(f"  train chars {len(tr):,}  val chars {len(va):,}  (val = final 10%, disjoint tail)")
    n = 200
    print(f"  val text appears in train? {''.join(d.decode(d.val[:n])) in d.decode(d.train)}")
