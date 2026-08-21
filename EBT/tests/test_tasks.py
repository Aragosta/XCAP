import torch
import pytest

from ebt.tasks import AssociativeRecall, Majority, Needle, TASKS, build_task


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("name", sorted(TASKS))
def test_shapes_dtypes_and_vocab_bounds(name):
    task = build_task(name, 64)
    x, y, m = task.batch(16, gen())
    assert x.shape == y.shape == m.shape == (16, task.seq_len)
    assert x.dtype == y.dtype == torch.long and m.dtype == torch.bool
    assert 0 <= int(x.min()) and int(x.max()) < task.vocab_size
    assert int(y[m].max()) < task.n_classes


@pytest.mark.parametrize("name", sorted(TASKS))
def test_reproducible_for_a_given_seed(name):
    task = build_task(name, 64)
    a = task.batch(8, gen(1))
    b = task.batch(8, gen(1))
    c = task.batch(8, gen(2))
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert not torch.equal(a[0], c[0])


@pytest.mark.parametrize("name", sorted(TASKS))
def test_exactly_one_supervised_position_per_sequence(name):
    task = build_task(name, 64)
    _, _, m = task.batch(32, gen())
    assert (m.sum(1) == 1).all() and m[:, -1].all()


@pytest.mark.parametrize("name", sorted(TASKS))
def test_label_is_not_trivially_predictable(name):
    """A constant predictor must do far worse than the model is expected to."""
    task = build_task(name, 64)
    _, y, m = task.batch(2048, gen())
    labels = y[m]
    freq = torch.bincount(labels, minlength=task.n_classes).float() / labels.numel()
    assert float(freq.max()) < 2.5 / task.n_classes


def test_associative_recall_label_is_the_value_bound_to_the_query_key():
    task = AssociativeRecall(seq_len=64)
    x, y, _ = task.batch(64, gen())
    for b in range(64):
        body = x[b, :-2]
        pairs = {int(v): int(body[i + 1]) - task.val0
                 for i, v in enumerate(body) if task.key0 <= int(v) < task.val0}
        assert len(pairs) == task.n_pairs, "keys must be distinct and all present"
        assert int(x[b, -2]) == task.query_tok
        assert pairs[int(x[b, -1])] == int(y[b, -1])


def test_associative_recall_values_are_in_range_and_keys_are_content_addressed():
    task = AssociativeRecall(seq_len=64)
    x, y, _ = task.batch(128, gen())
    key_positions = ((x[:, :-2] >= task.key0) & (x[:, :-2] < task.val0))
    assert (key_positions.sum(1) == task.n_pairs).all()
    # the queried key sits at a different position every time -> no positional shortcut
    first_key_pos = key_positions.float().argmax(1)
    assert len(set(first_key_pos.tolist())) > 8


def test_associative_recall_answer_depends_on_the_query():
    """Changing which key is queried changes the answer."""
    task = AssociativeRecall(seq_len=64)
    x, y, _ = task.batch(256, gen())
    assert len(set(y[:, -1].tolist())) > 1
    queried = x[:, -1]
    assert len(set(queried.tolist())) > task.n_pairs   # query ranges over the key pool


def test_associative_recall_signal_is_sparse():
    task = AssociativeRecall(seq_len=64)
    x, _, _ = task.batch(32, gen())
    informative = (x >= task.key0).float().sum(1).mean() / task.seq_len
    assert float(informative) < 0.2


def test_needle_places_tagged_pairs_and_answers_the_queried_tag():
    task = Needle(seq_len=64, n_needles=4)
    x, y, _ = task.batch(64, gen())
    for b in range(64):
        body = x[b, :-2]
        tags = [(int(v) - task.tag0, i) for i, v in enumerate(body)
                if task.tag0 <= int(v) < task.val0]
        assert sorted(t for t, _ in tags) == list(range(task.n_needles)), "one slot per tag"
        queried = int(x[b, -1]) - task.tag0
        pos = dict(tags)[queried]
        assert int(x[b, pos + 1]) - task.val0 == int(y[b, -1])


def test_needle_answer_depends_on_the_queried_tag():
    """Different tags must give different answers, else the query is decoration."""
    task = Needle(seq_len=64, n_needles=4)
    x, y, _ = task.batch(256, gen())
    tag_of_query = x[:, -1] - task.tag0
    assert len(set(tag_of_query.tolist())) == task.n_needles
    assert len(set(y[:, -1].tolist())) > 1


def test_needle_signal_is_sparse():
    """Only a few tokens carry information -- that is the point of the task."""
    task = Needle(seq_len=128, n_needles=4)
    x, _, _ = task.batch(32, gen())
    informative = (x >= task.tag0).float().sum(1).mean() / task.seq_len
    assert float(informative) < 0.12


def test_majority_label_is_the_modal_symbol_with_a_margin():
    task = Majority(seq_len=64, n_symbols=8, margin=2)
    x, y, _ = task.batch(256, gen())
    counts = torch.zeros(256, 8, dtype=torch.long)
    counts.scatter_add_(1, x - 1, torch.ones_like(x))
    top2 = counts.topk(2, dim=1)
    assert torch.equal(top2.indices[:, 0], y[:, -1])
    assert int((top2.values[:, 0] - top2.values[:, 1]).min()) >= 2


def test_majority_needs_the_whole_sequence():
    """A prefix-only view is much less informative than the full sequence."""
    task = Majority(seq_len=128, n_symbols=8)
    x, y, _ = task.batch(512, gen())
    for frac, ceiling in ((0.25, 0.75), (0.5, 0.9)):
        cut = int(frac * task.seq_len)
        counts = torch.zeros(512, 8, dtype=torch.long)
        counts.scatter_add_(1, x[:, :cut] - 1, torch.ones_like(x[:, :cut]))
        acc = (counts.argmax(1) == y[:, -1]).float().mean()
        assert float(acc) < ceiling


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        build_task("nope", 32)


def test_sequence_length_too_short_raises():
    with pytest.raises(ValueError):
        Needle(seq_len=8, n_needles=4)
    with pytest.raises(ValueError):
        AssociativeRecall(seq_len=4)
