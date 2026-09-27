#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "src" / "harnesses"))

from experiment_components import DATASETS
from run_experiments import (
    load_raw_dataset,
    preprocess_train_test,
    set_thread_limits,
    spec_for,
)
from margin_aware_splitting_harness import MarginAwareTree, gini


def inspect_node_candidates(X: np.ndarray, y: np.ndarray, spec, binary_mask: np.ndarray) -> dict:
    """Audit candidates across all features for a single node."""
    n_samples, n_features = X.shape
    total_unrestricted = 0
    total_class_aware = 0

    for feature_idx in range(n_features):
        vals = X[:, feature_idx]
        sorted_indices = np.argsort(vals)
        sorted_vals = vals[sorted_indices]
        sorted_y = y[sorted_indices]

        unique_vals, split_starts = np.unique(sorted_vals, return_index=True)
        if len(unique_vals) <= 1:
            continue

        group_classes = []
        for r in range(len(unique_vals)):
            start = split_starts[r]
            end = split_starts[r + 1] if r + 1 < len(unique_vals) else n_samples
            group_classes.append(np.unique(sorted_y[start:end]))

        for r in range(len(unique_vals) - 1):
            end_idx = split_starts[r + 1] - 1
            n_left = end_idx + 1
            n_right = n_samples - n_left
            if n_left < spec.min_samples_leaf or n_right < spec.min_samples_leaf:
                continue

            total_unrestricted += 1
            u_left = group_classes[r]
            u_right = group_classes[r + 1]
            is_same_class = (len(u_left) == 1 and len(u_right) == 1 and u_left[0] == u_right[0])
            if not is_same_class:
                total_class_aware += 1

    removed = total_unrestricted - total_class_aware
    pct_removed = (removed / total_unrestricted * 100.0) if total_unrestricted > 0 else 0.0
    return {
        "n_unrestricted_candidates": total_unrestricted,
        "n_class_aware_candidates": total_class_aware,
        "n_candidates_removed": removed,
        "pct_candidates_removed": pct_removed,
    }


def audit_tree_splits(tree: dict) -> tuple[int, int]:
    """Count internal splits and how many chose same-class boundaries."""
    n_splits = 0
    n_same_class = 0

    def walk(node):
        nonlocal n_splits, n_same_class
        if node.get("leaf", False):
            return
        n_splits += 1
        if node.get("is_same_class_boundary", False):
            n_same_class += 1
        walk(node["left"])
        walk(node["right"])

    if tree is not None:
        walk(tree)
    return n_splits, n_same_class


def count_changed_splits(tree_unres: dict, tree_ca: dict) -> tuple[int, int]:
    """Compare split decisions at identical tree path locations."""
    n_compared = 0
    n_changed = 0

    def walk(n1, n2):
        nonlocal n_compared, n_changed
        if n1 is None or n2 is None:
            return
        l1 = n1.get("leaf", False)
        l2 = n2.get("leaf", False)
        if l1 and l2:
            return
        if l1 != l2:
            n_compared += 1
            n_changed += 1
            return
        n_compared += 1
        if (
            n1.get("feature_idx") != n2.get("feature_idx")
            or not np.isclose(n1.get("threshold", 0.0), n2.get("threshold", 0.0), atol=1e-6)
        ):
            n_changed += 1
        walk(n1.get("left"), n2.get("left"))
        walk(n1.get("right"), n2.get("right"))

    walk(tree_unres, tree_ca)
    return n_compared, n_changed


def main():
    set_thread_limits()
    os.environ["MAGS_DATA_ROOT"] = "D:/data"

    audit_rows = []

    print("Running Reviewer 2 candidate audit across 27 datasets and 30 seeds...")
    for spec in DATASETS:
        dataset_name = spec.name
        for seed in range(30):
            X_df, y_all, data_meta = load_raw_dataset(spec, seed)
            stratify = y_all if np.min(np.bincount(y_all)) >= 2 else None
            X_train_df, X_test_df, y_train, y_test = train_test_split(
                X_df, y_all, test_size=0.35, random_state=seed, stratify=stratify
            )
            X_train, X_test, binary_mask, data_meta = preprocess_train_test(
                X_train_df, X_test_df, data_meta
            )

            # Root node candidate inspection
            root_audit = inspect_node_candidates(X_train, y_train, spec, binary_mask)

            # Fit trees
            # 1. Unrestricted Gini
            tree_gini_unres = MarginAwareTree(
                max_depth=spec.max_depth,
                alpha=0.0,
                max_thresholds=spec.max_thresholds,
                min_samples_leaf=spec.min_samples_leaf,
                score_mode="gain_gated",
                binary_features=binary_mask,
                binary_margin_policy="standard",
                candidate_filter="unrestricted",
            )
            tree_gini_unres.fit(X_train, y_train)

            # 2. Class-aware Gini
            tree_gini_ca = MarginAwareTree(
                max_depth=spec.max_depth,
                alpha=0.0,
                max_thresholds=spec.max_thresholds,
                min_samples_leaf=spec.min_samples_leaf,
                score_mode="gain_gated",
                binary_features=binary_mask,
                binary_margin_policy="standard",
                candidate_filter="class_aware",
            )
            tree_gini_ca.fit(X_train, y_train)

            # 3. Unrestricted MAGS alpha=0.5
            tree_mags_unres = MarginAwareTree(
                max_depth=spec.max_depth,
                alpha=0.5,
                max_thresholds=spec.max_thresholds,
                min_samples_leaf=spec.min_samples_leaf,
                score_mode="gain_gated",
                binary_features=binary_mask,
                binary_margin_policy="standard",
                candidate_filter="unrestricted",
            )
            tree_mags_unres.fit(X_train, y_train)

            # 4. Class-aware MAGS alpha=0.5
            tree_mags_ca = MarginAwareTree(
                max_depth=spec.max_depth,
                alpha=0.5,
                max_thresholds=spec.max_thresholds,
                min_samples_leaf=spec.min_samples_leaf,
                score_mode="gain_gated",
                binary_features=binary_mask,
                binary_margin_policy="standard",
                candidate_filter="class_aware",
            )
            tree_mags_ca.fit(X_train, y_train)

            # Split stats
            g_unres_splits, g_unres_same = audit_tree_splits(tree_gini_unres.tree)
            m_unres_splits, m_unres_same = audit_tree_splits(tree_mags_unres.tree)

            g_compared, g_changed = count_changed_splits(tree_gini_unres.tree, tree_gini_ca.tree)
            m_compared, m_changed = count_changed_splits(tree_mags_unres.tree, tree_mags_ca.tree)

            audit_rows.append({
                "dataset": dataset_name,
                "seed": seed,
                "root_unrestricted_candidates": root_audit["n_unrestricted_candidates"],
                "root_class_aware_candidates": root_audit["n_class_aware_candidates"],
                "root_candidates_removed": root_audit["n_candidates_removed"],
                "root_pct_candidates_removed": root_audit["pct_candidates_removed"],
                "gini_unres_total_splits": g_unres_splits,
                "gini_unres_same_class_splits": g_unres_same,
                "gini_unres_same_class_pct": (g_unres_same / g_unres_splits * 100.0) if g_unres_splits > 0 else 0.0,
                "mags_unres_total_splits": m_unres_splits,
                "mags_unres_same_class_splits": m_unres_same,
                "mags_unres_same_class_pct": (m_unres_same / m_unres_splits * 100.0) if m_unres_splits > 0 else 0.0,
                "gini_split_change_rate": (g_changed / g_compared * 100.0) if g_compared > 0 else 0.0,
                "mags_split_change_rate": (m_changed / m_compared * 100.0) if m_compared > 0 else 0.0,
            })

    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(ROOT / "results" / "raw" / "candidate_space_audit.csv", index=False)

    # Dataset-level and overall summaries
    summary_cols = [
        "root_pct_candidates_removed",
        "gini_unres_same_class_pct",
        "mags_unres_same_class_pct",
        "gini_split_change_rate",
        "mags_split_change_rate",
    ]

    ds_summary = df_audit.groupby("dataset")[summary_cols].mean().reset_index()

    overall_stats = []
    for col in summary_cols:
        vals = ds_summary[col]
        q25, q75 = np.percentile(vals, [25, 75])
        overall_stats.append({
            "metric": col,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "median": float(np.median(vals)),
            "iqr": float(q75 - q25),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })

    df_overall = pd.DataFrame(overall_stats)
    df_overall.to_csv(ROOT / "results" / "summary" / "candidate_space_summary.csv", index=False)

    print("=== Candidate Space Audit Summary Across 27 Datasets ===")
    print(df_overall.to_string(index=False))


if __name__ == "__main__":
    main()
