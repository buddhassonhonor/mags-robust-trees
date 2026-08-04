from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp"))

from run_experiments import (  # noqa: E402
    fit_custom,
    load_raw_dataset,
    preprocess_train_test,
    spec_for,
)
from margin_aware_splitting_harness import gini  # noqa: E402

OUT = ROOT / "exp" / "final_validation"


def collect_divergences(node_a: dict, node_b: dict, X: np.ndarray, y: np.ndarray, path: str, rows: list[dict]) -> None:
    if node_a.get("leaf", False) or node_b.get("leaf", False):
        return
    split_a = (int(node_a["feature_idx"]), float(node_a["threshold"]))
    split_b = (int(node_b["feature_idx"]), float(node_b["threshold"]))
    same = split_a[0] == split_b[0] and abs(split_a[1] - split_b[1]) < 1e-10
    if not same:
        rows.append({
            "path": path or "root",
            "n_node": int(len(y)),
            "gini_feature": split_a[0],
            "gini_threshold": split_a[1],
            "mags_feature": split_b[0],
            "mags_threshold": split_b[1],
            "same_feature": split_a[0] == split_b[0],
        })
        return
    j, v = split_a
    left = X[:, j] <= v
    collect_divergences(node_a["left"], node_b["left"], X[left], y[left], path + "L", rows)
    collect_divergences(node_a["right"], node_b["right"], X[~left], y[~left], path + "R", rows)


def descend(X: np.ndarray, y: np.ndarray, root: dict, path: str) -> tuple[np.ndarray, np.ndarray]:
    node = root
    for direction in path:
        j, v = int(node["feature_idx"]), float(node["threshold"])
        left = X[:, j] <= v
        if direction == "L":
            X, y, node = X[left], y[left], node["left"]
        else:
            X, y, node = X[~left], y[~left], node["right"]
    return X, y


def candidate_rows(X: np.ndarray, y: np.ndarray, max_thresholds: int, min_samples_leaf: int) -> list[dict]:
    rows: list[dict] = []
    current = gini(y)
    n_classes = max(2, len(np.bincount(y)))
    max_gini = 1.0 - 1.0 / n_classes
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j])
        values = X[order, j]
        labels = y[order]
        indices = np.where(np.diff(values) > 1e-9)[0]
        if len(indices) > max_thresholds:
            ranks = np.linspace(0, len(indices) - 1, max_thresholds).round().astype(int)
            indices = indices[np.unique(ranks)]
        value_range = values[-1] - values[0] if values[-1] > values[0] else 1.0
        for k in indices:
            if k + 1 < min_samples_leaf or len(y) - (k + 1) < min_samples_leaf:
                continue
            left_y, right_y = labels[: k + 1], labels[k + 1 :]
            gain = current - (len(left_y) * gini(left_y) + len(right_y) * gini(right_y)) / len(y)
            norm_gain = gain / max_gini if max_gini > 0 else 0.0
            gap = (values[k + 1] - values[k]) / value_range
            threshold = (values[k + 1] + values[k]) / 2.0
            row = {
                "feature": j,
                "threshold": float(threshold),
                "normalized_gini_gain": float(norm_gain),
                "normalized_adjacent_gap": float(gap),
                "score_margin_only": float(gap),
                "left_n": int(k + 1),
                "right_n": int(len(y) - k - 1),
            }
            for alpha, suffix in [(0.0, "0"), (0.1, "01"), (0.3, "03"), (0.5, "05"), (0.7, "07"), (1.0, "10")]:
                row[f"score_alpha_{suffix}"] = float(norm_gain * (1.0 + alpha * gap))
            rows.append(row)
    return rows


def main() -> None:
    cases: list[dict] = []
    payloads: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, object, object, dict, object]] = {}
    # Keep the explanatory case tied to the Sonar example discussed in the paper.
    for dataset in ["sonar"]:
        for seed in range(30):
            spec = spec_for(dataset)
            X, y, meta = load_raw_dataset(spec, seed)
            X_train_raw, X_test_raw, y_train, _ = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
            X_train, _, binary, meta = preprocess_train_test(X_train_raw, X_test_raw, meta)
            gini_tree, _ = fit_custom(X_train, y_train, spec, "matched_gini", 0.0, binary)
            mags_tree, _ = fit_custom(X_train, y_train, spec, "mags_fixed", 0.5, binary)
            found: list[dict] = []
            collect_divergences(gini_tree.tree, mags_tree.tree, X_train, y_train, "", found)
            for row in found:
                row.update(dataset=dataset, seed=seed)
                row["both_continuous"] = not binary[row["gini_feature"]] and not binary[row["mags_feature"]]
                cases.append(row)
            payloads[(dataset, seed)] = (X_train, y_train, gini_tree, mags_tree, meta, spec)
    if not cases:
        raise RuntimeError("No divergent real-data node found")
    ranked = sorted(
        cases,
        key=lambda r: (
            bool(r["same_feature"]),
            bool(r["both_continuous"]),
            15 <= int(r["n_node"]) <= 150,
            -abs(int(r["n_node"]) - 60),
        ),
        reverse=True,
    )
    chosen = ranked[0]
    X_train, y_train, gini_tree, mags_tree, meta, spec = payloads[(chosen["dataset"], chosen["seed"])]
    X_node, y_node = descend(X_train, y_train, gini_tree.tree, "" if chosen["path"] == "root" else chosen["path"])
    rows = candidate_rows(X_node, y_node, spec.max_thresholds, spec.min_samples_leaf)
    frame = pd.DataFrame(rows)
    for col in ["score_alpha_0", "score_alpha_01", "score_alpha_03", "score_alpha_05", "score_alpha_07", "score_alpha_10", "score_margin_only"]:
        frame[f"selected_{col}"] = frame[col] == frame[col].max()
    frame.to_csv(OUT / "threshold_case_candidates.csv", index=False)
    selected_features = sorted({int(chosen["gini_feature"]), int(chosen["mags_feature"]), int(frame.loc[frame.score_margin_only.idxmax(), "feature"])})
    point_rows = []
    for j in selected_features:
        for i, (value, label) in enumerate(zip(X_node[:, j], y_node)):
            point_rows.append({"feature": j, "point": i, "value": float(value), "class": int(label)})
    pd.DataFrame(point_rows).to_csv(OUT / "threshold_case_points.csv", index=False)
    chosen.update(
        n_classes=int(len(np.unique(y_node))),
        candidate_count=int(len(frame)),
        selected_features=selected_features,
        note="Feature indices refer to the encoded matrix; the example is a real training node.",
    )
    (OUT / "threshold_case.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
