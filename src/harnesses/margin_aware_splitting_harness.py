#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import numpy as np
from pathlib import Path

def gini(y):
    if len(y) == 0: return 0
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
        score_mode="gain_gated",
        binary_features=None,
        binary_margin_policy="standard",
        candidate_filter="class_aware",
    ):
        self.max_depth = max_depth
        self.alpha = alpha
        self.max_thresholds = max_thresholds
        self.min_samples_leaf = min_samples_leaf
        self.score_mode = score_mode
        self.binary_features = None if binary_features is None else np.asarray(binary_features, dtype=bool)
        self.binary_margin_policy = binary_margin_policy
        self.candidate_filter = candidate_filter
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if len(y) == 0:
            return {'leaf': True, 'class': 0}
        
        counts = np.bincount(y)
        majority_class = int(np.argmax(counts))

        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {'leaf': True, 'class': majority_class}

        best_score = -1.0
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

            unique_vals, split_starts = np.unique(sorted_vals, return_index=True)
            if len(unique_vals) <= 1:
                continue

            # Group classes for each distinct value group
            group_classes = []
            for r in range(len(unique_vals)):
                start = split_starts[r]
                end = split_starts[r + 1] if r + 1 < len(unique_vals) else n_samples
                group_classes.append(np.unique(sorted_y[start:end]))

            candidate_list = []
            for r in range(len(unique_vals) - 1):
                end_idx = split_starts[r + 1] - 1
                n_left = end_idx + 1
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                u_left = group_classes[r]
                u_right = group_classes[r + 1]
                is_same_class = (len(u_left) == 1 and len(u_right) == 1 and u_left[0] == u_right[0])

                if self.candidate_filter == "class_aware" and is_same_class:
                    continue

                threshold = float((unique_vals[r] + unique_vals[r + 1]) / 2.0)
                margin = float((unique_vals[r + 1] - unique_vals[r]) / 2.0)
                candidate_list.append({
                    "r": r,
                    "end_idx": end_idx,
                    "threshold": threshold,
                    "margin": margin,
                    "is_same_class": is_same_class,
                })

            if len(candidate_list) == 0:
                continue

            if self.max_thresholds is not None and len(candidate_list) > self.max_thresholds:
                ranks = np.linspace(0, len(candidate_list) - 1, self.max_thresholds).round().astype(int)
                candidate_list = [candidate_list[k] for k in np.unique(ranks)]

            for cand in candidate_list:
                end_idx = cand["end_idx"]
                threshold = cand["threshold"]
                margin = cand["margin"]
                is_same_class = cand["is_same_class"]

                left_y = sorted_y[:end_idx + 1]
                right_y = sorted_y[end_idx + 1:]

                w_left = len(left_y) / n_samples
                w_right = len(right_y) / n_samples
                gini_gain = current_gini - (w_left * gini(left_y) + w_right * gini(right_y))
                norm_gini_gain = gini_gain / max_possible_gini if max_possible_gini > 0 else 0.0

                norm_margin = (2.0 * margin) / f_range
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
                    score = (1.0 - self.alpha) * norm_gini_gain + self.alpha * norm_margin

                if score > best_score:
                    best_score = score
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_indices': sorted_indices[:end_idx + 1],
                        'right_indices': sorted_indices[end_idx + 1:],
                        'margin': margin,
                        'norm_margin': norm_margin,
                        'score_margin': score_margin,
                        'is_binary': is_binary,
                        'is_same_class_boundary': is_same_class,
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
            'is_same_class_boundary': best_split['is_same_class_boundary'],
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
    X0_a = rng.randn(n_samples // 4, 2) * 0.1 + np.array([-0.5, -0.5])
    X0_b = rng.randn(n_samples // 4, 2) * 0.1 + np.array([0.5, 0.5])
    y0 = np.zeros(n_samples // 2, dtype=int)
    
    X1_a = rng.randn(n_samples // 4, 2) * 0.1 + np.array([-0.5, 0.5])
    X1_b = rng.randn(n_samples // 4, 2) * 0.1 + np.array([0.5, -0.5])
    y1 = np.ones(n_samples // 2, dtype=int)
    
    X = np.vstack([X0_a, X0_b, X1_a, X1_b])
    y = np.concatenate([y0, y1])
    
    idx = rng.permutation(len(y))
    return X[idx], y[idx]
