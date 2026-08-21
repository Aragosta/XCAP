"""Synthetic probe tasks.

Three tasks chosen so that the comparison cannot be won by one bias alone:

  associative_recall  one token in the sequence matters.  Pure retrieval; the
                      ideal attention row is a delta.  Favours sparsity.
  needle              a handful of *marked* tokens matter and the model must
                      first find them, then pick one by index.  Favours a
                      mechanism that can route to a small relevant subset.
  majority            every token matters (count the most frequent symbol).
                      Favours dense, high-entropy attention: the honest
                      counter-example where dropping tokens must hurt.

Every task is a per-position classification problem with a loss mask, so the
same training loop and the same head work for all three.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

Batch = tuple[Tensor, Tensor, Tensor]  # x [B,N] int64, y [B,N] int64, mask [B,N] bool


@dataclass
class Task:
    name: str
    seq_len: int
    vocab_size: int
    n_classes: int

    def batch(self, batch_size: int, generator: torch.Generator) -> Batch:  # pragma: no cover
        raise NotImplementedError


def _randint(high: int, shape, g: torch.Generator) -> Tensor:
    return torch.randint(high, shape, generator=g, dtype=torch.long)


class AssociativeRecall(Task):
    """Key-value pairs hidden among noise; the query names one key.

    Layout: `[... noise ... k_j v_j ... noise ..., QUERY, k_i]`.  Keys are drawn
    without replacement from a pool larger than the number of pairs, so the match
    has to be made on content rather than position, and only `2 * n_pairs` of the
    tokens carry any information at all.
    """

    def __init__(self, seq_len: int = 64, n_pairs: int = 4, n_noise: int = 32,
                 n_keys: int = 16, n_values: int = 8):
        if n_pairs < 2 or seq_len < 4 * n_pairs + 2:
            raise ValueError("need at least 2 pairs and room for them")
        if n_keys < n_pairs:
            raise ValueError("n_keys must be >= n_pairs so keys stay distinct")
        self.n_pairs, self.n_noise = n_pairs, n_noise
        self.n_keys, self.n_values = n_keys, n_values
        self.noise0 = 1
        self.key0 = 1 + n_noise
        self.val0 = self.key0 + n_keys
        self.query_tok = self.val0 + n_values
        super().__init__("associative_recall", seq_len, self.query_tok + 1, n_values)

    def batch(self, batch_size: int, g: torch.Generator) -> Batch:
        b, n, m = batch_size, self.seq_len, self.n_pairs
        body = n - 2
        x = torch.zeros(b, n, dtype=torch.long)
        x[:, :body] = _randint(self.n_noise, (b, body), g) + self.noise0
        slots = torch.argsort(torch.rand(b, body // 2, generator=g), dim=1)[:, :m] * 2
        keys = torch.argsort(torch.rand(b, self.n_keys, generator=g), dim=1)[:, :m]
        vals = _randint(self.n_values, (b, m), g)
        rows = torch.arange(b)[:, None].expand(-1, m)
        x[rows, slots] = keys + self.key0
        x[rows, slots + 1] = vals + self.val0
        which = _randint(m, (b,), g)
        x[:, -2] = self.query_tok
        x[:, -1] = keys[torch.arange(b), which] + self.key0
        y = torch.zeros(b, n, dtype=torch.long)
        y[:, -1] = vals[torch.arange(b), which]
        mask = torch.zeros(b, n, dtype=torch.bool)
        mask[:, -1] = True
        return x, y, mask


class Needle(Task):
    """Noise everywhere except a few tagged needles; the query asks for one tag.

    Layout: `[... noise ... TAG_j value ... noise ..., QUERY, TAG_i]` with a
    distinct tag token per needle.  Unlike `associative_recall`, ~90% of the
    sequence is pure noise, so a head that routes to the tagged positions and
    ignores the rest has a real advantage.
    """

    def __init__(self, seq_len: int = 64, n_noise: int = 32, n_values: int = 8,
                 n_needles: int = 4):
        if seq_len < 4 * n_needles + 2:
            raise ValueError("sequence too short for the requested needles")
        self.n_noise, self.n_values, self.n_needles = n_noise, n_values, n_needles
        self.noise0 = 1
        self.tag0 = 1 + n_noise
        self.val0 = self.tag0 + n_needles
        self.query_tok = self.val0 + n_values
        super().__init__("needle", seq_len, self.query_tok + 1, n_values)

    def batch(self, batch_size: int, g: torch.Generator) -> Batch:
        b, n, m = batch_size, self.seq_len, self.n_needles
        body = n - 2
        x = torch.zeros(b, n, dtype=torch.long)
        x[:, :body] = _randint(self.n_noise, (b, body), g) + self.noise0
        # distinct even slots so each TAG keeps its value in the next position
        slots = torch.argsort(torch.rand(b, body // 2, generator=g), dim=1)[:, :m] * 2
        vals = _randint(self.n_values, (b, m), g)
        rows = torch.arange(b)[:, None].expand(-1, m)
        tags = torch.arange(m).expand(b, m)
        x[rows, slots] = tags + self.tag0
        x[rows, slots + 1] = vals + self.val0
        which = _randint(m, (b,), g)
        x[:, -2] = self.query_tok
        x[:, -1] = which + self.tag0
        y = torch.zeros(b, n, dtype=torch.long)
        y[:, -1] = vals[torch.arange(b), which]
        mask = torch.zeros(b, n, dtype=torch.bool)
        mask[:, -1] = True
        return x, y, mask


class Majority(Task):
    """Predict the most frequent symbol in the sequence: needs every token."""

    def __init__(self, seq_len: int = 64, n_symbols: int = 8, margin: int = 2):
        self.n_symbols, self.margin = n_symbols, margin
        super().__init__("majority", seq_len, n_symbols + 1, n_symbols)

    def batch(self, batch_size: int, g: torch.Generator) -> Batch:
        b, n, s = batch_size, self.seq_len, self.n_symbols
        x = _randint(s, (b, n), g)
        counts = torch.zeros(b, s, dtype=torch.long)
        counts.scatter_add_(1, x, torch.ones_like(x))
        top2 = counts.topk(2, dim=1)
        winner = top2.indices[:, 0]
        # enforce a clear margin: overwrite random positions with the winner symbol
        need = (self.margin - (top2.values[:, 0] - top2.values[:, 1])).clamp(min=0)
        if int(need.max()) > 0:
            # only overwrite positions that do not already hold the winner, so
            # the margin really grows and the label stays unambiguous
            r = torch.rand(b, n, generator=g) + (x == winner[:, None]).float() * 2.0
            order = torch.argsort(r, dim=1).argsort(dim=1)
            fill = order < need[:, None]
            x = torch.where(fill, winner[:, None].expand(-1, n), x)
        y = torch.zeros(b, n, dtype=torch.long)
        y[:, -1] = winner
        mask = torch.zeros(b, n, dtype=torch.bool)
        mask[:, -1] = True
        return x + 1, y, mask  # +1 keeps 0 free as PAD


class Relational(Task):
    """Facts as (subject, relation, object) triples hidden among noise.

    Layout: `[... noise ... s r o ... noise ..., QUERY, s, r]` -> the object.

    Every subject appears with **two different relations and two different
    objects**, so the subject alone is worth at most chance-between-two: the
    model is forced to combine subject *and* relation.  This is the structure
    the relational energy scores are supposed to exploit --
    `Paris --capital-of--> France` vs `Paris --located-on--> Seine`.
    """

    def __init__(self, seq_len: int = 64, n_noise: int = 16, n_subjects: int = 8,
                 n_relations: int = 3, n_objects: int = 8, n_facts_per_subject: int = 2,
                 n_subjects_shown: int = 3):
        if n_relations < n_facts_per_subject or n_objects < 2:
            raise ValueError("need at least as many relations as facts per subject")
        n_facts = n_subjects_shown * n_facts_per_subject
        if seq_len < 3 * n_facts + 3:
            raise ValueError("sequence too short for the requested facts")
        self.n_noise, self.n_subjects = n_noise, n_subjects
        self.n_relations, self.n_objects = n_relations, n_objects
        self.per_subject, self.n_shown, self.n_facts = n_facts_per_subject, n_subjects_shown, n_facts
        self.noise0 = 1
        self.subj0 = 1 + n_noise
        self.rel0 = self.subj0 + n_subjects
        self.obj0 = self.rel0 + n_relations
        self.query_tok = self.obj0 + n_objects
        super().__init__("relational", seq_len, self.query_tok + 1, n_objects)

    def batch(self, batch_size: int, g: torch.Generator) -> Batch:
        b, n, f = batch_size, self.seq_len, self.n_facts
        body = n - 3
        x = torch.zeros(b, n, dtype=torch.long)
        x[:, :body] = _randint(self.n_noise, (b, body), g) + self.noise0

        rows = torch.arange(b)[:, None].expand(-1, f)
        subs = torch.argsort(torch.rand(b, self.n_subjects, generator=g), 1)[:, : self.n_shown]
        subs = subs.repeat_interleave(self.per_subject, dim=1)                 # [B,F]
        rels = torch.stack([torch.argsort(torch.rand(b, self.n_relations, generator=g), 1)
                            [:, : self.per_subject] for _ in range(self.n_shown)], 1).flatten(1)
        objs = torch.stack([torch.argsort(torch.rand(b, self.n_objects, generator=g), 1)
                            [:, : self.per_subject] for _ in range(self.n_shown)], 1).flatten(1)
        slots = torch.argsort(torch.rand(b, body // 3, generator=g), 1)[:, :f] * 3
        x[rows, slots] = subs + self.subj0
        x[rows, slots + 1] = rels + self.rel0
        x[rows, slots + 2] = objs + self.obj0

        which = _randint(f, (b,), g)
        idx = torch.arange(b)
        x[:, -3] = self.query_tok
        x[:, -2] = subs[idx, which] + self.subj0
        x[:, -1] = rels[idx, which] + self.rel0
        y = torch.zeros(b, n, dtype=torch.long)
        y[:, -1] = objs[idx, which]
        mask = torch.zeros(b, n, dtype=torch.bool)
        mask[:, -1] = True
        return x, y, mask


TASKS = {
    "relational": Relational,
    "associative_recall": AssociativeRecall,
    "needle": Needle,
    "majority": Majority,
}


def build_task(name: str, seq_len: int, **kwargs) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; expected one of {sorted(TASKS)}")
    return TASKS[name](seq_len=seq_len, **kwargs)
