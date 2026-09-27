import numpy as np

from exp.run_experiments import exact_continuous_tree_attack
from scripts.compare_robust_trees import groot_classifier


def test_groot_exact_attack_reaches_opposite_leaf_within_budget():
    classifier = groot_classifier()
    X_train = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1, 1])
    model = classifier(max_depth=1, min_samples_leaf=1, attack_model=[0.05], compile=False, random_state=0)
    model.fit(X_train, y_train)
    assert not model.root_.is_leaf()

    threshold = float(model.root_.threshold)
    X_test = np.array([[threshold - 0.01], [threshold + 0.01]])
    y_test = model.predict(X_test)
    result = exact_continuous_tree_attack(model, X_test, y_test, np.array([False]), eps=0.05)

    assert result["attack_success_rate"] == 1.0
    assert result["reachable_fraction"] == 1.0
    assert result["attacked_accuracy"] == 0.0
