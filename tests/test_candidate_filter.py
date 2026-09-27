import numpy as np
import pytest
from src.harnesses.margin_aware_splitting_harness import MarginAwareTree, generate_data


def test_1_simple_transition():
    # x = [0,1,2,3], y = [0,0,1,1]
    # Valid class-aware candidate is only between 1 and 2 (v=1.5)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    
    # Class-aware tree
    model_ca = MarginAwareTree(max_depth=1, alpha=0.5, candidate_filter="class_aware")
    model_ca.fit(X, y)
    assert not model_ca.tree["leaf"]
    assert model_ca.tree["threshold"] == 1.5


def test_2_same_class_interval_exclusion():
    # x = [0,1,2,3], y = [0,1,1,0]
    # Interval between 1 and 2 is same-class (1 vs 1)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1, 0])
    
    model_ca = MarginAwareTree(max_depth=1, alpha=0.5, candidate_filter="class_aware")
    model_ca.fit(X, y)
    assert not model_ca.tree["leaf"]
    # Eligible thresholds are 0.5 and 2.5, threshold 1.5 is excluded
    assert model_ca.tree["threshold"] in [0.5, 2.5]
    assert model_ca.tree["threshold"] != 1.5


def test_3_duplicate_values_tie_safe():
    # Duplicate feature values: x = [0,0,1,1], mixed labels
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    
    model_ca = MarginAwareTree(max_depth=1, alpha=0.5, candidate_filter="class_aware")
    model_ca.fit(X, y)
    # Mixed groups are retained tie-safely
    assert not model_ca.tree["leaf"]
    assert model_ca.tree["threshold"] == 0.5


def test_4_alpha_zero_equivalence():
    # Class-aware MAGS(alpha=0) must be identical to Class-aware matched Gini(alpha=0)
    X, y = generate_data(seed=42, n_samples=200)
    
    mags_0 = MarginAwareTree(max_depth=4, alpha=0.0, candidate_filter="class_aware")
    mags_0.fit(X, y)
    
    gini_0 = MarginAwareTree(max_depth=4, alpha=0.0, score_mode="gain_only", candidate_filter="class_aware")
    gini_0.fit(X, y)
    
    preds_mags = mags_0.predict(X)
    preds_gini = gini_0.predict(X)
    np.testing.assert_array_equal(preds_mags, preds_gini)
    assert mags_0.tree["threshold"] == gini_0.tree["threshold"]
    assert mags_0.tree["feature_idx"] == gini_0.tree["feature_idx"]


def test_5_determinism():
    X, y = generate_data(seed=123, n_samples=300)
    
    m1 = MarginAwareTree(max_depth=3, alpha=0.5, candidate_filter="class_aware")
    m1.fit(X, y)
    
    m2 = MarginAwareTree(max_depth=3, alpha=0.5, candidate_filter="class_aware")
    m2.fit(X, y)
    
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
    assert m1.tree["threshold"] == m2.tree["threshold"]
