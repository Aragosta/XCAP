"""Statistics and geometry probes must be right, or every number they produce lies."""

import math

import torch

from hmha.metrics import (
    attention_flops_per_token,
    bits_per_byte,
    bootstrap_diff_ci,
    cohens_d,
    compare_arms,
    gromov_delta,
    log_log_slope,
    mean_std,
    welch_t_test,
)


def test_bits_per_byte_conversion():
    assert bits_per_byte(math.log(2)) == 1.0
    # a uniform byte model costs exactly 8 bits/byte
    assert bits_per_byte(math.log(256)) == 8.0


def test_mean_std():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert m == 2.0
    assert s == 1.0


def test_log_log_slope_recovers_a_known_exponent():
    xs = [128, 256, 512, 1024, 2048]
    ys = [3e-7 * x**2 for x in xs]
    fit = log_log_slope(xs, ys)
    assert abs(fit["exponent"] - 2.0) < 1e-6
    assert abs(fit["coefficient"] - 3e-7) / 3e-7 < 1e-6
    assert fit["r_squared"] > 0.999


def test_log_log_slope_distinguishes_linear_from_quadratic():
    xs = [128, 256, 512, 1024]
    assert abs(log_log_slope(xs, [2.0 * x for x in xs])["exponent"] - 1.0) < 1e-6
    assert abs(log_log_slope(xs, [2.0 * x**2 for x in xs])["exponent"] - 2.0) < 1e-6


def test_bootstrap_ci_brackets_the_difference():
    a = [1.0, 1.1, 0.9, 1.05]
    b = [2.0, 2.1, 1.9, 2.05]
    ci = bootstrap_diff_ci(a, b)
    assert ci["diff"] < 0
    assert ci["ci_low"] <= ci["diff"] <= ci["ci_high"]
    assert ci["excludes_zero"], "a clear separation must be detected"


def test_bootstrap_ci_includes_zero_for_identical_arms():
    a = [1.0, 1.2, 0.8]
    ci = bootstrap_diff_ci(a, list(a))
    assert not ci["excludes_zero"]


def test_welch_and_cohens_d_signs():
    a = [1.0, 1.1, 0.9]
    b = [2.0, 2.1, 1.9]
    assert welch_t_test(a, b)["p_value"] < 0.05
    assert cohens_d(a, b) < 0  # a is smaller
    assert welch_t_test(a, a)["p_value"] > 0.5


def test_compare_arms_picks_the_lower_mean():
    result = compare_arms([1.0, 1.1], [2.0, 2.2], "hyperbolic", "euclidean")
    assert result["winner"] == "hyperbolic"
    assert result["hyperbolic"]["n"] == 2


def test_flops_hyperbolic_costs_more_than_euclidean():
    for seq in [128, 1024]:
        euc = attention_flops_per_token(seq, 256, "euclidean")
        hyp = attention_flops_per_token(seq, 256, "hyperbolic")
        assert hyp > euc
        assert hyp / euc < 2.0, "overhead should be a modest constant factor"


def test_flops_grow_with_sequence_length():
    assert attention_flops_per_token(1024, 256, "euclidean") > attention_flops_per_token(
        128, 256, "euclidean"
    )


def test_gromov_delta_is_near_zero_for_a_tree():
    """A path graph embedded on a line is an exact tree: delta must be ~0."""
    points = torch.arange(64, dtype=torch.float32).unsqueeze(-1)
    result = gromov_delta(points, "euclidean")
    assert result["delta_rel"] < 1e-6, result


def test_gromov_delta_is_larger_for_a_grid_than_a_line():
    """A 2-D grid is markedly less tree-like than a 1-D line."""
    line = torch.arange(64, dtype=torch.float32).unsqueeze(-1)
    g = torch.arange(8, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), dim=-1).reshape(-1, 2)
    assert gromov_delta(grid, "euclidean")["delta_rel"] > gromov_delta(line, "euclidean")["delta_rel"]


def test_gromov_delta_lorentz_distance_runs():
    points = torch.randn(48, 6)
    result = gromov_delta(points, "lorentz", curvature=1.0)
    assert result["delta"] >= 0
    assert result["diameter"] > 0


def test_gromov_delta_hyperbolic_is_more_tree_like_than_euclidean():
    """The motivating property: the same cloud looks more tree-like when curved."""
    torch.manual_seed(0)
    points = torch.randn(64, 8) * 2.0
    euc = gromov_delta(points, "euclidean")["delta_rel"]
    hyp = gromov_delta(points, "lorentz", curvature=1.0)["delta_rel"]
    assert hyp < euc, f"lorentz {hyp:.4f} should be below euclidean {euc:.4f}"


def test_bootstrap_refuses_significance_with_too_few_seeds():
    """A one-value bootstrap collapses to a point interval; it must not read as significant."""
    ci = bootstrap_diff_ci([1.0], [2.0])
    assert ci["ci_low"] == ci["ci_high"], "degenerate by construction"
    assert not ci["excludes_zero"], "must not claim significance from a single observation"
    assert ci["underpowered"]


def test_bootstrap_is_powered_at_three_seeds():
    ci = bootstrap_diff_ci([1.0, 1.05, 0.95], [2.0, 2.05, 1.95])
    assert not ci["underpowered"]
    assert ci["excludes_zero"]
