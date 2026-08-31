"""GSM8K chain-of-thought text, tokenized so that reasoning depth is recoverable.

Byte level makes GSM8K traces ~490 tokens, which is more sequence than 4 CPU
cores can afford. A word/digit level vocabulary keeps traces short while
preserving the two things the probes depend on: digits stay separate (so
arithmetic is learnable) and the newline between solution steps is its own
token, which is the ground-truth marker of reasoning depth.
"""
import json
import re
import numpy as np

TOKEN_RE = re.compile(r"\n|[0-9]|[a-zA-Z]+(?:'[a-zA-Z]+)?|[^\sa-zA-Z0-9]|\s+")
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def pieces(text):
    out = []
    for m in TOKEN_RE.finditer(text):
        s = m.group(0)
        if s.isspace() and s != "\n":
            continue          # ordinary spacing carries no reasoning structure
        out.append(s)
    return out


def format_example(rec):
    """Q/A text with the calculator annotations stripped -- real natural text."""
    answer = re.sub(r"<<[^>]*>>", "", rec["answer"]).strip()
    return "Q: " + rec["question"].strip() + "\nA: " + answer


def load_gsm8k(path):
    return [format_example(json.loads(l)) for l in open(path)]


class Vocab:
    def __init__(self, texts, max_size=4096):
        from collections import Counter
        c = Counter()
        for t in texts:
            c.update(pieces(t))
        self.itos = list(SPECIALS) + [w for w, _ in c.most_common(max_size - len(SPECIALS))]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.nl = self.stoi.get("\n", UNK)

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(p, UNK) for p in pieces(text)]

    def save(self, path):
        json.dump(self.itos, open(path, "w"))

    @classmethod
    def load(cls, path):
        v = cls.__new__(cls)
        v.itos = json.load(open(path))
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        v.nl = v.stoi.get("\n", UNK)
        return v


def encode_corpus(texts, vocab):
    """One flat stream with BOS/EOS separators, for the training loader."""
    ids = []
    for t in texts:
        ids.append(BOS)
        ids.extend(vocab.encode(t))
        ids.append(EOS)
    return np.array(ids, dtype=np.int32)


def encode_probe_set(texts, vocab, seq_len, limit=None):
    """Per-example sequences with aligned reasoning-structure metadata.

    Returns ids plus, for every token position:
      depth  -- which solution step the token is in (0 = the question)
      inans  -- whether the token is inside the answer at all
      ex     -- which example the token came from
    Depth is what the radial and energy hypotheses are tested against, so it is
    computed from the newline tokens rather than from position.
    """
    X, D, A, E = [], [], [], []
    for ei, t in enumerate(texts):
        ids = [BOS] + vocab.encode(t)
        if len(ids) > seq_len:
            continue                       # keep only complete traces
        q_end = t.index("\nA:")
        n_q = len(vocab.encode(t[:q_end])) + 1   # +1 for BOS
        depth, inans, d = [], [], 0
        for i, tok in enumerate(ids):
            if i < n_q:
                depth.append(0)
                inans.append(0)
            else:
                if tok == vocab.nl:
                    d += 1
                depth.append(d)
                inans.append(1)
        pad = seq_len - len(ids)
        X.append(ids + [PAD] * pad)
        D.append(depth + [-1] * pad)
        A.append(inans + [0] * pad)
        E.append(ei)
        if limit and len(X) >= limit:
            break
    return (np.array(X, dtype=np.int64), np.array(D, dtype=np.int64),
            np.array(A, dtype=np.int64), np.array(E, dtype=np.int64))
