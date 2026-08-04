#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import numpy as np
from pathlib import Path

def gini(y):
    if len(y) == 0: return 0
    # Assume binary or small number of classes
    counts = np.bincount(y)
    p = counts / len(y)
    return 1 - np.sum(p**2)

class MarginAwareTree:
    def __init__(
        self,
        max_depth=3,
        alpha=0.5,
        max_thresholds=None,
        min_samples_leaf=1,
        score_mode="convex",
        binary_features=None,
        binary_margin_policy="standard",
    ):
        self.max_depth = max_depth
        self.alpha = alpha
        self.max_thresholds = max_thresholds
        self.min_samples_leaf = min_samples_leaf
        self.score_mode = score_mode
        self.binary_features = None if binary_features is None else np.asarray(binary_features, dtype=bool)
        self.binary_margin_policy = binary_margin_policy
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if len(y) == 0:
            return {'leaf': True, 'class': 0}
        
        counts = np.bincount(y)
        majority_class = np.argmax(counts)

        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {'leaf': True, 'class': majority_class}

        best_score = -1
        best_split = None

        current_gini = gini(y)
        n_classes = max(2, len(np.bincount(y)))
        max_possible_gini = 1.0 - 1.0 / n_classes
        
        for feature_idx in range(n_features):
            vals = X[:, feature_idx]
            sorted_indices = np.argsort(vals)
            sorted_vals = vals[sorted_indices]
            sorted_y = y[sorted_indices]
            
            f_min, f_max = sorted_vals[0], sorted_vals[-1]
            f_range = f_max - f_min if f_max > f_min else 1.0

            # Candidate splits are between unique values. For larger real
            # tabular data, cap candidates by evenly spaced ranks.
            unique_indices = np.where(np.diff(sorted_vals) > 1e-9)[0]
            if self.max_thresholds is not None and len(unique_indices) > self.max_thresholds:
                ranks = np.linspace(0, len(unique_indices) - 1, self.max_thresholds).round().astype(int)
                unique_indices = unique_indices[np.unique(ranks)]
            for i in unique_indices:
                if i + 1 < self.min_samples_leaf or n_samples - (i + 1) < self.min_samples_leaf:
                    continue
                threshold = (sorted_vals[i] + sorted_vals[i+1]) / 2
                
                # Split
                left_y = sorted_y[:i+1]
                right_y = sorted_y[i+1:]
                
                # Gini Gain
                w_left = len(left_y) / n_samples
                w_right = len(right_y) / n_samples
                gini_gain = current_gini - (w_left * gini(left_y) + w_right * gini(right_y))
                norm_gini_gain = gini_gain / max_possible_gini if max_possible_gini > 0 else 0.0
                
                # Margin (distance from threshold to nearest points)
                margin = (sorted_vals[i+1] - sorted_vals[i]) / 2
                norm_margin = (2 * margin) / f_range
                is_binary = bool(
                    self.binary_features is not None
                    and feature_idx < len(self.binary_features)
                    and self.binary_features[feature_idx]
                )
                score_margin = 0.0 if self.binary_margin_policy == "exclude" and is_binary else norm_margin
                
                if self.score_mode == "gain_gated":
                    score = norm_gini_gain * (1.0 + self.alpha * score_margin)
                elif self.score_mode == "gain_only":
                    score = norm_gini_gain
                elif self.score_mode == "margin_only":
                    score = norm_margin
                elif self.score_mode == "unnormalized_margin":
                    score = norm_gini_gain * (1.0 + self.alpha * margin)
                else:
                    score = (1 - self.alpha) * norm_gini_gain + self.alpha * norm_margin
                
                if score > best_score:
                    best_score = score
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_indices': sorted_indices[:i+1],
                        'right_indices': sorted_indices[i+1:],
                        'margin': margin,
                        'norm_margin': norm_margin,
                        'score_margin': score_margin,
                        'is_binary': is_binary,
                    }

        if best_split is None:
            return {'leaf': True, 'class': majority_class}

        left_node = self._build_tree(X[best_split['left_indices']], y[best_split['left_indices']], depth + 1)
        right_node = self._build_tree(X[best_split['right_indices']], y[best_split['right_indices']], depth + 1)
        
        return {
            'leaf': False,
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'margin': best_split['margin'],
            'norm_margin': best_split['norm_margin'],
            'score_margin': best_split['score_margin'],
            'is_binary': best_split['is_binary'],
            'left': left_node,
            'right': right_node
        }

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['class']
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
            
    def get_all_norm_margins(self, node=None):
        if node is None: node = self.tree
        if node['leaf']: return []
        margins = [node['norm_margin']]
        margins.extend(self.get_all_norm_margins(node['left']))
        margins.extend(self.get_all_norm_margins(node['right']))
        return margins

def generate_data(seed, n_samples=300):
    rng = np.random.RandomState(seed)
    # Class 0: two sub-clusters
    X0_a = rng.randn(n_samples // 4, 2) * 0.1 + np.array([-0.5, -0.5])
    X0_b = rng.randn(n_samples // 4, 2) * 0.1 + np.array([0.5, 0.5])
    y0 = np.zeros(n_samples // 2, dtype=int)
    
    # Class 1: two sub-clusters
    X1_a = rng.randn(n_samples // 4, 2) * 0.1 + np.array([-0.5, 0.5])
    X1_b = rng.randn(n_samples // 4, 2) * 0.1 + np.array([0.5, -0.5])
    y1 = np.ones(n_samples // 2, dtype=int)
    
    # This is an XOR-like problem. 
    # Standard Gini will pick a split that might be very close to points.
    
    X = np.vstack([X0_a, X0_b, X1_a, X1_b])
    y = np.concatenate([y0, y1])
    
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["quick", "expanded"], default="quick")
    parser.add_argument("--config", required=False)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)

    # Use seed for data generation
    X_train, y_train = generate_data(args.seed, n_samples=400)
    X_test, y_test = generate_data(args.seed + 1000, n_samples=400)
    
    # Noise for robustness testing - significantly higher
    noise_std = 0.25
    X_test_noisy = X_test + np.random.RandomState(args.seed + 2000).randn(*X_test.shape) * noise_std

    # Baseline: alpha = 0 (Gini only)
    baseline_model = MarginAwareTree(max_depth=3, alpha=0.0)
    baseline_model.fit(X_train, y_train)
    
    b_clean_acc = np.mean(baseline_model.predict(X_test) == y_test)
    b_robust_acc = np.mean(baseline_model.predict(X_test_noisy) == y_test)
    b_margins = baseline_model.get_all_norm_margins()
    b_avg_margin = np.mean(b_margins) if b_margins else 0.0

    # Treatment: alpha = 0.5 (Mixed)
    treatment_model = MarginAwareTree(max_depth=3, alpha=0.5)
    treatment_model.fit(X_train, y_train)
    
    t_clean_acc = np.mean(treatment_model.predict(X_test) == y_test)
    t_robust_acc = np.mean(treatment_model.predict(X_test_noisy) == y_test)
    t_margins = treatment_model.get_all_norm_margins()
    t_avg_margin = np.mean(t_margins) if t_margins else 0.0

    robustness_delta = t_robust_acc - b_robust_acc
    clean_acc_delta = t_clean_acc - b_clean_acc
    margin_improvement = t_avg_margin - b_avg_margin

    row = {
        "seed": args.seed,
        "mode": args.mode,
        "metric_name": "robustness_delta",
        "metric_value": robustness_delta,
        "baseline": b_robust_acc,
        "treatment": t_robust_acc,
        "higher_is_better": True,
        "primary": True,
        "decision_hint": "promising" if robustness_delta > 0.01 and t_clean_acc >= b_clean_acc - 0.05 else "weak",
    }
    
    metrics = [row]
    metrics.append({
        "seed": args.seed,
        "mode": args.mode,
        "metric_name": "clean_acc_delta",
        "metric_value": clean_acc_delta,
        "baseline": b_clean_acc,
        "treatment": t_clean_acc,
        "higher_is_better": True,
    })
    metrics.append({
        "seed": args.seed,
        "mode": args.mode,
        "metric_name": "margin_improvement",
        "metric_value": margin_improvement,
        "baseline": b_avg_margin,
        "treatment": t_avg_margin,
        "higher_is_better": True,
    })

    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m, sort_keys=True) + "\n")

    artifact = {
        "seed": args.seed,
        "baseline": {
            "clean_acc": b_clean_acc,
            "robust_acc": b_robust_acc,
            "avg_margin": b_avg_margin
        },
        "treatment": {
            "clean_acc": t_clean_acc,
            "robust_acc": t_robust_acc,
            "avg_margin": t_avg_margin
        },
        "delta": {
            "robustness": robustness_delta,
            "clean_acc": clean_acc_delta,
            "margin": margin_improvement
        }
    }
    (run_dir / "artifacts" / f"mags_seed_{args.seed}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    
    print(json.dumps(row, sort_keys=True))

if __name__ == "__main__":
    main()
