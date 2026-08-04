import numpy as np

from src.harnesses.margin_aware_splitting_harness import MarginAwareTree


def test_binary_exclusion_zeroes_gap_multiplier():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 0, 1, 1])
    model = MarginAwareTree(
        max_depth=1,
        alpha=0.5,
        score_mode="gain_gated",
        binary_features=[True],
        binary_margin_policy="exclude",
    )
    model.fit(X, y)
    assert model.tree["is_binary"] is True
    assert model.tree["score_margin"] == 0.0


def test_standard_binary_policy_retains_unit_gap():
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 0, 1, 1])
    model = MarginAwareTree(
        max_depth=1,
        alpha=0.5,
        score_mode="gain_gated",
        binary_features=[True],
        binary_margin_policy="standard",
    )
    model.fit(X, y)
    assert model.tree["score_margin"] == 1.0
